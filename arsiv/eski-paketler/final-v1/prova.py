# -*- coding: utf-8 -*-
"""PROVA — temiz paketi GT'li bir video uzerinde ucтan uca kostur ve resmi metrigi olc.

Sartname Denklem 2: TUM karelerde ortalama Oklid hatasi. Saglikli karelerde sunucunun
gonderdigi konum aynen geri gonderildigi icin hata 0'dir; **puani yalniz kesinti
kareleri belirler**.

Kullanilan kesinti takvimi, puanli oturum logundan cikarilan GERCEK yapidir:
    echo 0-450 | KESINTI 451-899 | echo 900-1050 | KESINTI 1051-1549
                | echo 1550-1650 | KESINTI 1651-2249
(703 saglikli + 1547 kesinti kare)

Kullanim:
    python prova.py --video ...MP4 --gt ...translation.csv
    python prova.py --video ...MP4 --gt ...csv --motor sift
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2                                      # noqa: E402
import pandas as pd                             # noqa: E402
from konum import LivePositionEstimator         # noqa: E402

KESINTI = [(451, 900), (1051, 1550), (1651, 2250)]
N_KARE = 2250
KARE_ADIMI = 4                                  # 30 FPS video -> 7,5 FPS yarisma temposu


def takvim(n: int) -> np.ndarray:
    h = np.ones(n, bool)
    for a, b in KESINTI:
        h[a:b] = False
    return h


def taban(gt: np.ndarray, healthy: np.ndarray, tur: str) -> np.ndarray:
    """Gorsel bilgi KULLANMAYAN kiyas tabanlari."""
    n = len(gt)
    p = np.zeros((n, 3))
    p[0] = gt[0]
    vel = np.zeros(3)
    son, gecmis = gt[0].copy(), [gt[0].copy()]
    for k in range(1, n):
        if healthy[k]:
            p[k] = gt[k]
            son = gt[k].copy()
            gecmis.append(son)
            if len(gecmis) > 9:
                gecmis.pop(0)
            if len(gecmis) >= 2:
                vel = (gecmis[-1] - gecmis[0]) / (len(gecmis) - 1)
        elif tur == "tut":
            p[k] = son
        else:
            p[k] = p[k - 1] + vel
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--motor", default="auto")
    ap.add_argument("--n", type=int, default=N_KARE)
    args = ap.parse_args()

    g = pd.read_csv(args.gt)
    if "translation_z" not in g.columns:
        g["translation_z"] = 0.0
    gt_tam = g[["translation_x", "translation_y", "translation_z"]].to_numpy()
    gt = gt_tam[::KARE_ADIMI][:args.n]

    cap = cv2.VideoCapture(args.video)
    est = LivePositionEstimator(engine=args.motor)
    healthy = takvim(len(gt))
    preds = np.zeros((len(gt), 3))

    t0 = time.perf_counter()
    idx = ord_ = 0
    while ord_ < len(gt):
        ok = cap.grab()
        if not ok:
            break
        if idx % KARE_ADIMI == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            s = bool(healthy[ord_])
            preds[ord_] = est.process_frame(frame, gt[ord_] if s else None, s)
            ord_ += 1
            if ord_ % 500 == 0:
                print(f"  {ord_}/{len(gt)}  {time.perf_counter()-t0:.0f} s", flush=True)
        idx += 1
    cap.release()
    n = ord_
    gt, preds, healthy = gt[:n], preds[:n], healthy[:n]

    hata = np.linalg.norm(preds - gt, axis=1)
    kesinti = ~healthy
    t = est.timing_stats

    print(f"\n{'=' * 66}")
    print(f"PROVA — {Path(args.video).name}  ({n} kare · motor={est.engine_used} "
          f"· profil={est.profile.name})")
    print("=" * 66)
    print(f">>> RESMI METRIK (Denklem 2, TUM kareler) : {hata.mean():.3f} m")
    print(f"    yalniz kesinti karelerinde            : {hata[kesinti].mean():.3f} m")
    print(f"    saglikli karelerde                    : {hata[healthy].mean():.4f} m")
    d = np.abs(preds - gt)[kesinti]
    print(f"    eksen RMS (kesinti) x {np.sqrt((d[:,0]**2).mean()):.2f} · "
          f"y {np.sqrt((d[:,1]**2).mean()):.2f} · z {np.sqrt((d[:,2]**2).mean()):.2f} m")

    print(f"\n    kiyas tabanlari (gorsel bilgi YOK):")
    for tur, ad in (("tut", "son saglikli konumu tut"), ("hiz", "son hizla duz devam")):
        e = np.linalg.norm(taban(gt, healthy, tur) - gt, axis=1).mean()
        print(f"      {ad:<26}: {e:7.2f} m   ({e/max(hata.mean(),1e-9):.1f}x)")

    print(f"\n    HIZ: {t['ort_ms']:.0f} ms/kare · p95 {t['p95_ms']:.0f} ms · "
          f"toplam {(time.perf_counter()-t0)/60:.1f} dk")
    return 0


if __name__ == "__main__":
    sys.exit(main())
