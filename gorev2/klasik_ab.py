# -*- coding: utf-8 -*-
"""GOREV 2 — KLASIK ESLESTIRICI A/B: SIFT / ORB / AKAZE vs RAFT.

NEDEN
-----
Olcum katmani turlarinda *ogrenilmis* seyrek eslestiriciler denendi (XFeat,
DISK+LightGlue, gercek SuperPoint+LightGlue) ve hepsi RAFT'a yenildi; ayrica
ORB-SLAM'in TAMAMI bir sistem olarak denendi. Ama **klasik dedektorler kendi
boru hattimizda bir olcum ureticisi olarak hic sinanmadi.**

Ilk bakis (video1, 4 cift) klasiklerin baştan elenemeyecegini gosterdi:
    raft   n_kept  627 · hom inlier 0,696 · 286 ms/cift
    sift   n_kept 1784 · hom inlier 0,742 · 214 ms/cift
    akaze  n_kept 1298 · hom inlier 0,658 · 115 ms/cift
    orb    n_kept 1687 · hom inlier 0,292 · 122 ms/cift

ONEMLI — HER ESLESTIRICI KENDI ESIGIYLE
---------------------------------------
Yeniden-izdusum esigi eslestiriciye ozgudur: her yontemin piksel gurultusu farkli.
Gecmiste SuperPoint'in optimumu RAFT'inkinin **8 kati** cikmisti. Bu yuzden A/B
her eslestirici icin esik taramasiyla yapilir; tek esikle karsilastirmak yaniltir.

KABUL
-----
Kazanmasi icin RAFT tabanini hem yarisma takviminde hem 15 uzun-kesinti
senaryosunda gecmeli (eslestirici_ab.py ile ayni olcut).

CALISTIR
--------
    .venv\\Scripts\\python gorev2\\klasik_ab.py --tracker sift akaze orb
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

GOREV2 = Path(__file__).resolve().parent
sys.path.insert(0, str(GOREV2))

from nokta_onbellek import OUTPUTS, onbellek_yukle, olcum_uret, yatak_npz  # noqa: E402
from kabul import _kos, procrustes_tani, yarisma_takvimi, uzun_takvim      # noqa: E402
from gorev2_src import data_io, evaluation                                  # noqa: E402
from gorev2_src.calibration import RGB_1080P                                # noqa: E402
from gorev2_src.vo_pipeline import EstimatorConfig                          # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

ROOT = GOREV2.parent
GT_CSV = ROOT / "THYZ_2026_Ornek_Veri_Seti" / "THYZ_2026_Ornek_Veri_1_translation.csv"
N_UZUN = 15
CFG = EstimatorConfig(method="homografi")          # sevk edilen recete


def npz_yolu(tracker: str, spacing) -> Path:
    """`tracker` "sift" ya da "sift:_ds1" gibi bir sonek tasiyabilir (varyant onbellek)."""
    if tracker == "raft":
        return yatak_npz("video1", spacing)
    ad, _, sonek = tracker.partition(":")
    ek = "cons" if spacing is None else f"sp{spacing}"
    return OUTPUTS / f"nokta_{ad}_video1_{ek}{sonek}.npz"


def degerlendir(tracker: str, esik: float, gt, n, uzunlar, karistir=None):
    ob_c = onbellek_yukle(npz_yolu(tracker, None))
    ob_l = onbellek_yukle(npz_yolu(tracker, 8))
    gk = dict(ransac_thresh=esik)
    cons = olcum_uret(ob_c, RGB_1080P, karistir=karistir, **gk).iloc[:n - 1]
    lad = olcum_uret(ob_l, RGB_1080P, karistir=karistir, **gk).iloc[:n - 1]
    yar = yarisma_takvimi(n)
    p = _kos(cons, lad, gt, yar, RGB_1080P, CFG, 8)
    y = evaluation.competition_error(p, gt)
    uz = [evaluation.competition_error(_kos(cons, lad, gt, h, RGB_1080P, CFG, 8), gt)
          for h in uzunlar]
    yon, olc = procrustes_tani(p, gt, yar)
    inl = float(np.nanmean(cons["hom_inlier_ratio"].replace(0, np.nan)))
    return y, float(np.mean(uz)), np.array(uz), yon, 100 * olc, inl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracker", nargs="*", default=["sift", "akaze", "orb"])
    ap.add_argument("--esikler", type=float, nargs="*",
                    default=[0.5, 1.0, 2.0, 3.0, 5.0, 8.0])
    args = ap.parse_args()

    gt_sub = data_io.subsample_competition(data_io.load_translations(GT_CSV))
    n = min(2250, len(gt_sub))
    gt = gt_sub.iloc[:n][evaluation.GT_COLS].to_numpy()
    uzunlar = [uzun_takvim(n, rng=s) for s in range(N_UZUN)]

    print(f"=== KLASIK ESLESTIRICI A/B (video1, {n} kare) ===\n")
    t0 = time.perf_counter()
    ty, tu, tu_liste, ta, ts, ti = degerlendir("raft", 1.0, gt, n, uzunlar)
    print(f"TEMEL (raft, esik 1.0): yarisma {ty:.3f} m | {N_UZUN} uzun {tu:.3f} m | "
          f"yon {ta:.2f}° | olcek %{ts:.2f} | inlier {ti:.3f}  ({time.perf_counter()-t0:.0f}s)\n")

    for tr in args.tracker:
        if not npz_yolu(tr, None).exists() or not npz_yolu(tr, 8).exists():
            print(f"--- {tr}: onbellek yok, atlandi ---\n")
            continue
        print(f"--- {tr} ---")
        print(f"{'esik':>6}{'yarisma':>10}{'uzun':>10}{'yon°':>8}{'olcek%':>9}"
              f"{'inlier':>8}{'kazanan':>9}  karar")
        for esik in args.esikler:
            y, u, u_liste, yon, olc, inl = degerlendir(tr, esik, gt, n, uzunlar)
            kaz = int((u_liste < tu_liste).sum())
            kabul = (y < ty) and (u < tu)
            print(f"{esik:>6}{y:>10.3f}{u:>10.3f}{yon:>8.2f}{olc:>9.2f}{inl:>8.3f}"
                  f"{kaz:>6}/{N_UZUN}  {'KABUL' if kabul else '-'}", flush=True)
        print()


if __name__ == "__main__":
    main()
