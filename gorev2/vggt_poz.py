# -*- coding: utf-8 -*-
"""GOREV 2 — VGGT ile poz dizisi uretimi (VGGT-Long yaklasimi: parca + Sim(3) zincirleme).

VGGT (CVPR 2025) feed-forward olarak bir kare KUMESINDEN kamera pozlarini kestirir.
Tum videoyu tek seferde veremeyiz (VRAM); bu yuzden video ORTUSEN PARCALARA bolunur,
her parca ayri kestirilir ve parcalar **ortusen karelerdeki Sim(3)** ile zincirlenir
(VGGT-Long'un temel fikri).

Olculen davranis (RTX 5060, 8,5 GB):
    N= 4 -> 3,47 s/kare | tepe VRAM  8,31 GB
    N= 8 -> 0,89 s/kare | tepe VRAM  9,79 GB   <-- en verimli
    N=16 -> 5,52 s/kare | tepe VRAM 10,27 GB   (VRAM tasmasi -> yavas)
Model tek basina 5,03 GB tuttugu icin parca buyudukce tasma basliyor.

Cikti: `outputs/poz_vggt.npz` -> `konumlar` (N,3) + `sureler` (N,)
Degerlendirme: `slam_degerlendirme.py --poz outputs/poz_vggt.npz --ad vggt`

Calistir:
    .venv\\Scripts\\python gorev2\\vggt_poz.py --parca 8 --ortusme 2
    .venv\\Scripts\\python gorev2\\vggt_poz.py --n 200        # hizli deneme
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

GOREV2 = Path(__file__).resolve().parent
ROOT = GOREV2.parent
sys.path.insert(0, str(GOREV2))

from nokta_onbellek import _tracker_kur  # noqa: F401,E402  (gorev2_src'yi kaydeder)
from gorev2_src import data_io           # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

VIDEO = ROOT / "THYZ_2026_Ornek_Veri_Seti" / "THYZ_2026_Ornek_Veri_1.MP4"
CIKTI = GOREV2 / "outputs" / "poz_vggt.npz"
GIRDI_BOYUT = 294          # VRAM esigi: 518->9,8 GB (paging, 40x yavas); 294->7,8 GB
OLCEK_BANT = 1.02          # parcalar arasi goreli olcek sapmasi siniri (patlamayi onler)


def _hazirla(gray_list, boyut=GIRDI_BOYUT):
    """Gri kareleri VGGT girdisine cevir: RGB, [0,1], kare boyuta olcekle."""
    import cv2
    import torch
    yig = []
    for g in gray_list:
        rgb = cv2.cvtColor(g, cv2.COLOR_GRAY2RGB) if g.ndim == 2 else g
        rgb = cv2.resize(rgb, (boyut, boyut), interpolation=cv2.INTER_AREA)
        yig.append(torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0)
    return torch.stack(yig)


def _kamera_merkezleri(extri: np.ndarray) -> np.ndarray:
    """world-to-camera [R|t] (M,3,4) -> kamera merkezleri (M,3): C = -R^T t."""
    R = extri[:, :3, :3]
    t = extri[:, :3, 3]
    return -np.einsum("mij,mj->mi", np.transpose(R, (0, 2, 1)), t)


def sim3_fit(A: np.ndarray, B: np.ndarray):
    """Umeyama: B ~= s*R*A + t."""
    if len(A) < 3:
        return None
    mu_a, mu_b = A.mean(0), B.mean(0)
    A0, B0 = A - mu_a, B - mu_b
    H = A0.T @ B0 / len(A)
    U, D, Vt = np.linalg.svd(H)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = Vt.T @ S @ U.T
    var_a = (A0 ** 2).sum() / len(A)
    s = float((D * np.diag(S)).sum() / var_a) if var_a > 1e-12 else 1.0
    return s, R, mu_b - s * (R @ mu_a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parca", type=int, default=8, help="parca basina kare (VRAM/hiz)")
    # DIKKAT: Sim(3) icin en az 3 ortusen kare sart (Umeyama 3 nokta ister).
    ap.add_argument("--ortusme", type=int, default=4, help="parcalar arasi ortusen kare (>=3)")
    ap.add_argument("--n", type=int, default=None, help="ilk N kare (hizli deneme)")
    ap.add_argument("--cikti", default=str(CIKTI))
    args = ap.parse_args()
    if args.ortusme >= args.parca:
        sys.exit("ortusme < parca olmali")
    if args.ortusme < 3:
        sys.exit("ortusme >= 3 olmali (Sim(3) fit'i en az 3 nokta ister)")

    import torch
    from vggt.models.vggt import VGGT
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    print("VGGT yukleniyor...")
    model = VGGT.from_pretrained("facebook/VGGT-1B").to("cuda").eval()

    print("Kareler okunuyor...")
    kareler = []
    for _, g in data_io.iter_competition_frames(VIDEO, progress=True):
        kareler.append(g)
        if args.n is not None and len(kareler) >= args.n:
            break
    n = len(kareler)
    print(f"{n} kare | parca={args.parca}, ortusme={args.ortusme}")

    adim = args.parca - args.ortusme
    konumlar = np.zeros((n, 3))
    sureler = np.zeros(n)
    global_fit = None            # onceki parcadan devralinan Sim(3)
    onceki_global = None         # (ortusen karelerin GLOBAL konumlari)
    t_bas = time.perf_counter()

    for bas in range(0, n, adim):
        son = min(bas + args.parca, n)
        if son - bas < 2:
            break
        imgs = _hazirla(kareler[bas:son]).to("cuda")
        t0 = time.perf_counter()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred = model(imgs[None])
        extri, _ = pose_encoding_to_extri_intri(pred["pose_enc"], imgs.shape[-2:])
        dt = time.perf_counter() - t0
        yerel = _kamera_merkezleri(extri[0].float().cpu().numpy())   # (M,3) parca-yerel

        if onceki_global is None:
            konumlar[bas:son] = yerel
        else:
            # ortusen kareler: yerel <-> onceki global -> Sim(3)
            k = min(args.ortusme, len(yerel), len(onceki_global))
            fit = sim3_fit(yerel[:k], onceki_global[-k:])
            if fit is None:                      # dejenere ortusme: son gecerli fit'e dus
                fit = global_fit or (1.0, np.eye(3), np.zeros(3))
            s, R, t = fit
            # OLCEK PATLAMASI KORUMASI (kritik):
            # Her parca kendi olceginde cozulur; zincirleme carpim s1*s2*... seklinde
            # birikir. 563 parcada parca basi %5'lik sapma bile 1.05^563 ~ 1e12 yapar
            # (ilk denemede hata 1e28 m'ye ulasti). Ardisik parcalar ayni sahneyi
            # gordugu icin GERCEK goreli olcek 1'e cok yakin olmalidir; sapmayi dar
            # bir bantla sinirlayip yeniden olceklenmis fit kullaniyoruz.
            if not np.isfinite(s) or s <= 0:
                s = 1.0
            s = float(np.clip(s, 1.0 / OLCEK_BANT, OLCEK_BANT))
            t = onceki_global[-k:].mean(0) - s * (R @ yerel[:k].mean(0))   # t'yi s ile tutarli kur
            fit = (s, R, t)
            konumlar[bas:son] = (s * (R @ yerel.T).T) + t
            global_fit = fit
        onceki_global = konumlar[bas:son].copy()
        sureler[bas:son] = dt / (son - bas)

        if (bas // adim) % 20 == 0:
            gecen = time.perf_counter() - t_bas
            print(f"  kare {son}/{n} | {dt:.1f} s/parca | toplam {gecen/60:.1f} dk", flush=True)
        del imgs, pred
        torch.cuda.empty_cache()

    Path(args.cikti).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.cikti, konumlar=konumlar, sureler=sureler)
    print(f"\n{n} poz, {(time.perf_counter()-t_bas)/60:.1f} dk -> {Path(args.cikti).name}")
    print(f"kare basi ort: {sureler.mean():.2f} s (butce 2,4 s)")


if __name__ == "__main__":
    main()
