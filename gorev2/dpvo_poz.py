# -*- coding: utf-8 -*-
"""GOREV 2 — DPVO (Deep Patch Visual Odometry) ile poz dizisi uretimi.

DPVO: goruntu YAMALARINI (patch) yineleyerek takip eden ag + turevlenebilir bundle
adjustment. DROID-SLAM'den ~3x hizli, ucte bir bellek. Bizim icin onemi: BA, coklu
kare uzerinde ORTAK optimizasyon yapar — mevcut tek-cift homografi yaklasimimizin
olcek yanliligini (Procrustes: %2,11) azaltma potansiyeli tasir.

Not (Windows uyarlamasi): DPVO Linux/GCC icin yazilmis; bu makinede derlenebilmesi icin
kaynakta uc duzeltme yapildi — `tensor.type()` -> `.scalar_type()`, `<long>` sablonlari
-> `<int64_t>` (Windows'ta long 32-bit), GCC compound literal -> standart C++.

Cikti: `outputs/poz_dpvo.npz` -> `konumlar` (N,3) + `sureler` (N,)
Degerlendirme: `slam_degerlendirme.py --poz outputs/poz_dpvo.npz --ad dpvo`

Calistir:
    .venv\\Scripts\\python gorev2\\dpvo_poz.py
    .venv\\Scripts\\python gorev2\\dpvo_poz.py --n 300     # hizli deneme
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
from gorev2_src.calibration import RGB_1080P  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

VIDEO = ROOT / "THYZ_2026_Ornek_Veri_Seti" / "THYZ_2026_Ornek_Veri_1.MP4"
CIKTI = GOREV2 / "outputs" / "poz_dpvo.npz"
DPVO_KAYNAK = Path("C:/Users/merto/AppData/Local/Temp/slam_kaynak/DPVO")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None, help="ilk N kare")
    ap.add_argument("--olcek", type=float, default=0.5,
                    help="goruntu olcegi (1920x1080 -> 0.5 = 960x540)")
    ap.add_argument("--agirlik", default=None, help="dpvo.pth yolu (yoksa indirilir)")
    ap.add_argument("--cikti", default=str(CIKTI))
    args = ap.parse_args()

    import cv2
    import torch
    from dpvo.dpvo import DPVO
    from dpvo.config import cfg

    # --- agirlik ---
    agirlik = Path(args.agirlik) if args.agirlik else (DPVO_KAYNAK / "dpvo.pth")
    if not agirlik.exists():
        import urllib.request
        url = "https://www.dropbox.com/s/nap0u8zslspdwm4/models.zip"
        sys.exit(f"HATA: DPVO agirligi yok: {agirlik}\n"
                 f"  Indir: {url} (models.zip icindeki dpvo.pth) ve --agirlik ile ver.")

    cfg.merge_from_file(str(DPVO_KAYNAK / "config" / "default.yaml"))
    print(f"DPVO agirligi: {agirlik.name}")

    # --- kamera ic parametreleri (olceklenmis) ---
    s = args.olcek
    intr = torch.tensor([RGB_1080P.fx * s, RGB_1080P.fy * s,
                         RGB_1080P.cx * s, RGB_1080P.cy * s], dtype=torch.float32)

    slam = None
    konumlar, sureler = [], []
    t_bas = time.perf_counter()
    print("Kareler isleniyor (DPVO)...")
    for i, (_, gray) in enumerate(data_io.iter_competition_frames(VIDEO, gray=False,
                                                                  progress=True)):
        if args.n is not None and i >= args.n:
            break
        img = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        t = torch.from_numpy(img).permute(2, 0, 1).cuda()
        if slam is None:
            slam = DPVO(cfg, str(agirlik), ht=t.shape[1], wd=t.shape[2], viz=False)
        t0 = time.perf_counter()
        with torch.no_grad():
            slam(i, t, intr)
        sureler.append(time.perf_counter() - t0)
        # anlik (causal) konum: DPVO'nun son poz tahmini
        try:
            p = slam.poses_[max(slam.n - 1, 0)].cpu().numpy()   # [tx,ty,tz,qx,qy,qz,qw]
            konumlar.append(p[:3].copy())
        except Exception:
            konumlar.append(konumlar[-1] if konumlar else np.zeros(3))

    konumlar = np.array(konumlar, dtype=float)
    sureler = np.array(sureler, dtype=float)
    Path(args.cikti).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.cikti, konumlar=konumlar, sureler=sureler)
    print(f"\n{len(konumlar)} poz, {(time.perf_counter()-t_bas)/60:.1f} dk "
          f"-> {Path(args.cikti).name}")
    print(f"kare basi ort: {sureler.mean():.2f} s (butce 2,4 s)")


if __name__ == "__main__":
    main()
