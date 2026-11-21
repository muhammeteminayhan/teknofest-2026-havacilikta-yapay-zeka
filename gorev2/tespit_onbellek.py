# -*- coding: utf-8 -*-
"""GOREV 2 — GT'li yataklarda Gorev 1 tespit onbellegi (olcek capasi deneyi icin).

Final oturumlarinda Gorev 1 kutulari loglarda var (final_yatak.py). GT'li dogrulama
yataklarinda (video1, video2, 2025 ucuslari) yok; olcek capasinin "zarar vermiyor mu"
sinavi icin ayni dedektor bu videolarda kosturulur.

Dedektor: FINAL PAKETINDEKI uretim dedektoru (final-diyarbakir/gorev1, imgsz 1280),
o paketin kendi ortamiyla:
    ..\\final-diyarbakir\\.venv\\Scripts\\python gorev2\\tespit_onbellek.py video1 video2 ...

Kareler `data_io.iter_competition_frames` ile ayni kuralla secilir (her 4. kare, 2250 kare).
4K videolar yarismadaki gibi 1920x1080'e kucultulup dedektore verilir; kutular ORIJINAL
cozunurluge geri olceklenir (boylece yatagin kamera profili dogrudan gecerli).
Cikti: outputs/final/tespit_<yatak>.npz  (kutu_kare, kutu_sinif 1..4 = sunucu sinifi, kutu_xyxy)
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.stdout.reconfigure(encoding="utf-8")

GOREV2 = Path(__file__).resolve().parent
KOK = GOREV2.parent
FINAL = Path.home() / "Desktop" / "final-diyarbakir"
sys.path.insert(0, str(FINAL))
from gorev1.dedektor import Gorev1Detector                      # noqa: E402

VERI = KOK / "THYZ_2026_Ornek_Veri_Seti"
VERI25 = Path.home() / "Desktop" / "datasets" / "THYZ_2025_Ornek_Veri_Seti"
VIDEOLAR = {
    "video1": VERI / "THYZ_2026_Ornek_Veri_1.MP4",
    "video2": VERI / "THYZ_2026_Ornek_Veri_2_Termal.MP4",
    "v25_rgb_a": VERI25 / "Ornek-Veri-1-RGB.MP4",
    "v25_rgb_b": VERI25 / "Ornek-Veri-2-RGB.MP4",
    "v25_termal_a": VERI25 / "Ornek-Veri-1-Termal.MP4",
    "v25_termal_b": VERI25 / "Ornek-Veri-2-Termal.MP4",
}
N_KARE, ADIM = 2250, 4
CIKTI = GOREV2 / "outputs" / "final"


def kareler(video: Path):
    cap = cv2.VideoCapture(str(video))
    idx = ordinal = 0
    while ordinal < N_KARE:
        if idx % ADIM == 0:
            ok, fr = cap.read()
            if not ok:
                break
            yield ordinal, fr
            ordinal += 1
        elif not cap.grab():
            break
        idx += 1


def main():
    for ad in sys.argv[1:]:
        hedef = CIKTI / f"tespit_{ad}.npz"
        if hedef.exists():
            print(f"{ad}: zaten var")
            continue
        det = Gorev1Detector()
        kk, ks, kx = [], [], []
        t0 = time.perf_counter()
        for k, fr in kareler(VIDEOLAR[ad]):
            s = 1.0
            if fr.shape[1] > 1920:
                s = fr.shape[1] / 1920
                fr = cv2.resize(fr, (1920, round(fr.shape[0] / s)), interpolation=cv2.INTER_AREA)
            for o in det.detect(fr):
                kk.append(k)
                ks.append(int(o["cls"]) + 1)          # sunucu numaralandirmasi: 1 tasit ... 4 UAI
                kx.append([o["top_left_x"] * s, o["top_left_y"] * s,
                           o["bottom_right_x"] * s, o["bottom_right_y"] * s])
            if k % 250 == 0:
                print(f"  {ad}: kare {k} ({time.perf_counter() - t0:.0f} s)", flush=True)
        np.savez_compressed(hedef, kutu_kare=np.array(kk, np.int32), kutu_sinif=np.array(ks, np.int8),
                            kutu_xyxy=np.array(kx, np.float32).reshape(-1, 4))
        print(f"{ad}: {len(kk)} kutu ({sum(c == 1 for c in ks)} tasit, "
              f"{sum(c in (3, 4) for c in ks)} ped) -> {hedef.name}", flush=True)


if __name__ == "__main__":
    main()
