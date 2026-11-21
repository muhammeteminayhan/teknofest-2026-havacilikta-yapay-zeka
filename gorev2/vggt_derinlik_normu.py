# -*- coding: utf-8 -*-
"""GOREV 2 — VGGT'nin son sansi: DERINLIKLE NORMALIZE edilmis yerel olcum

NEDEN VAR
---------
Olculen iki gercek (`tavan_vggt_parca.py`, `vggt_cozunurluk.py`):

  * VGGT parca ici sekil hatasi cozunurlukle duzeliyor (294px %6,2 -> 392px %3,4)
  * ama PARCA OLCEGININ tutarsizligi duzelmiyor (%52 -> %47 degisim katsayisi)

Zincirlemeyi olduren sey ikincisi. Ne kadar iyi bir global Sim(3) graf optimizasyonu
kurulursa kurulsun, girdideki %47'lik olcek gurultusunu geri getiremez.

AMA: parca olcegi keyfi olsa bile VGGT **derinligi de ayni keyfi birimde** tahmin
ediyor. Oyleyse

        e = |kamera merkezi adimi| / (parca icindeki medyan derinlik)

parca olceginden BAGIMSIZ olmali. Ve bu tam olarak boru hattimizin bekledigi
buyukluk: homografi de `delta/d` (duzlem uzakligina bolunmus oteleme) uretiyor ve
metrige cevirmeyi ogrenilmis `height` yapiyor.

Eger bu dogruysa VGGT'yi GLOBAL yorunge uretmek icin degil, mevcut (dogrulanmis)
durum makinesine YEREL OLCUM besleyen bir on uc olarak kullanabiliriz — parca
zincirleme sorunu tamamen ortadan kalkar.

OLCUT
-----
`g / e` orani (GT adimi / birimsiz olcum) fiziksel olarak UCUS YUKSEKLIGIDIR ve
yavas degismelidir. Degisim katsayisi:
  * kucukse (RAFT'in %2,5 olcek hatasi mertebesinde) -> on uc KULLANILABILIR
  * buyukse                                          -> aile kapanir

CALISTIR
--------
    .venv\\Scripts\\python gorev2\\vggt_derinlik_normu.py
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

GOREV2 = Path(__file__).resolve().parent
ROOT = GOREV2.parent
sys.path.insert(0, str(GOREV2))

import kabul                                                       # noqa: E402,F401
from gorev2_src import data_io                                     # noqa: E402
from vggt_poz import sim3_fit, _hazirla, _kamera_merkezleri, VIDEO  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

GT_CSV = ROOT / "THYZ_2026_Ornek_Veri_Seti" / "THYZ_2026_Ornek_Veri_1_translation.csv"


def _medyan_derinlik(pred) -> float:
    """Parca icindeki medyan tahmin derinligi (parca-yerel birimde)."""
    for anahtar in ("depth", "depths", "depth_map"):
        if anahtar in pred:
            d = pred[anahtar]
            a = d.float().cpu().numpy()
            a = a[np.isfinite(a) & (a > 0)]
            return float(np.median(a)) if a.size else float("nan")
    return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-parca", type=int, default=60)
    ap.add_argument("--parca", type=int, default=8)
    ap.add_argument("--adim", type=int, default=16)
    ap.add_argument("--boyut", type=int, default=392)
    args = ap.parse_args()

    import torch
    from vggt.models.vggt import VGGT
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    gt_sub = data_io.subsample_competition(data_io.load_translations(GT_CSV))
    G = gt_sub[["translation_x", "translation_y", "translation_z"]].to_numpy()

    gerek = args.adim * args.n_parca + args.parca
    kareler = []
    for _, g in data_io.iter_competition_frames(VIDEO, progress=False):
        kareler.append(g)
        if len(kareler) >= gerek:
            break
    print(f"{len(kareler)} kare · {args.n_parca} parca · girdi {args.boyut} px\n")

    print("VGGT yukleniyor...")
    model = VGGT.from_pretrained("facebook/VGGT-1B").to("cuda").eval()
    anahtarlar_yazildi = False

    kayit = []
    for i in range(args.n_parca):
        bas = i * args.adim
        son = bas + args.parca
        if son > len(kareler) or son > len(G):
            break
        imgs = _hazirla(kareler[bas:son], args.boyut).to("cuda")
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred = model(imgs[None])
        if not anahtarlar_yazildi:
            print(f"VGGT cikti anahtarlari: {sorted(pred.keys())}\n")
            anahtarlar_yazildi = True
        extri, _ = pose_encoding_to_extri_intri(pred["pose_enc"], imgs.shape[-2:])
        P = _kamera_merkezleri(extri[0].float().cpu().numpy())
        d_yerel = _medyan_derinlik(pred)
        del imgs, pred
        torch.cuda.empty_cache()

        fit = sim3_fit(P, G[bas:son])
        if fit is None or not np.isfinite(d_yerel):
            continue
        s, R, t = fit
        if not np.isfinite(s) or s <= 0:
            continue
        adim_yerel = np.linalg.norm(np.diff(P, axis=0), axis=1)      # parca-yerel
        adim_gt = np.linalg.norm(np.diff(G[bas:son], axis=0), axis=1)  # metre
        e = adim_yerel / max(d_yerel, 1e-12)                          # birimsiz
        gecerli = (e > 1e-9) & (adim_gt > 1e-6)
        if gecerli.sum() < 2:
            continue
        kayit.append(dict(
            s=s, d=d_yerel, s_d=s * d_yerel,
            oran=float(np.median(adim_gt[gecerli] / e[gecerli])),
            e_med=float(np.median(e[gecerli])),
            g_med=float(np.median(adim_gt[gecerli])),
        ))
        if i and i % 20 == 0:
            print(f"  {i}/{args.n_parca} parca", flush=True)

    if not kayit:
        raise SystemExit("olcum uretilemedi")

    s = np.array([k["s"] for k in kayit])
    sd = np.array([k["s_d"] for k in kayit])
    oran = np.array([k["oran"] for k in kayit])
    kv = lambda a: 100 * a.std() / max(abs(a.mean()), 1e-12)

    print("\n" + "=" * 78)
    print(f"DERINLIK NORMALIZASYONU — {len(kayit)} parca, girdi {args.boyut} px")
    print("=" * 78)
    print(f"\n  parca olcegi  s_c            : degisim katsayisi %{kv(s):.0f}")
    print(f"  s_c x medyan derinlik        : degisim katsayisi %{kv(sd):.0f}"
          f"   <- derinlik olcegi tasiyorsa BU KUCUK olmali")
    print(f"\n  g/e orani (= ucus yuksekligi): medyan {np.median(oran):.2f} m"
          f" · degisim katsayisi **%{kv(oran):.0f}**")
    print(f"    p05-p95: {np.percentile(oran,5):.2f} - {np.percentile(oran,95):.2f} m")

    print("\n  KIYAS")
    print("   RAFT + homografi olcek hatasi (video1, RGB) : %2,5")
    print("   VGGT parca olcegi ham degisim katsayisi     : "
          f"%{kv(s):.0f}")
    print("\n  YORUM ANAHTARI")
    print("   g/e degisim katsayisi %5'in altindaysa: derinlik normalizasyonu parca")
    print("     olcegini gercekten kaldiriyor -> VGGT on uc olarak KULLANILABILIR")
    print("   Hala buyukse (>%20): derinlik de ayni tutarsizligi tasiyor -> KAPANIR")


if __name__ == "__main__":
    main()
