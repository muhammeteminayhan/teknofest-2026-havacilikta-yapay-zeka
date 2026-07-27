# -*- coding: utf-8 -*-
"""GOREV 2 — GEOMETRI TARAMASI (yontem 6: MAGSAC++ ve robust kestirici varyantlari).

TESHIS: kesinti hatasi rastgele degil SISTEMATIK (yon ~2 derece, olcek %3-4,5).
Kaynak durum makinesi degil OLCUM katmani. Bu tarama, olcum katmaninin ikinci
yarisini — yani **robust hareket kestirimini** — hedefler: ayni nokta eslesmeleri
uzerinde RANSAC / MAGSAC++ / USAC_ACCURATE ve farkli yeniden-izdusum esikleri.

HIZ: nokta onbellegi (`nokta_onbellek.py`) sayesinde video YENIDEN TARANMAZ;
her varyant icin yalnizca geometri yeniden cozulur (saniyeler).

KABUL OLCUTU (onceki turda kendini kanitladi):
  * yarismanin GERCEK kesinti takviminde >= %1 kazanc, VE
  * farkli rastgele takvimlerin ortalamasinda da kazanc (asiri uyum koruması).

Calistir:
    .venv\\Scripts\\python gorev2\\geometri_tarama.py
    .venv\\Scripts\\python gorev2\\geometri_tarama.py --tracker xfeat --senaryo 6
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

GOREV2 = Path(__file__).resolve().parent
ROOT = GOREV2.parent
OUT = GOREV2 / "degerlendirme"
sys.path.insert(0, str(GOREV2))

from nokta_onbellek import onbellek_yukle, olcum_uret               # noqa: E402
from gorev2_src import data_io, evaluation                          # noqa: E402
from gorev2_src.calibration import RGB_1080P                        # noqa: E402
from gorev2_src.vo_pipeline import EstimatorConfig, VoLadderEstimator  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

GT_CSV = ROOT / "THYZ_2026_Ornek_Veri_Seti" / "THYZ_2026_Ornek_Veri_1_translation.csv"
KESINTI = [(451, 900), (1051, 1550), (1651, 2250)]
N_KARE, SPACING, SMOOTH = 2250, 8, 1.0

YONTEMLER = {
    "RANSAC": cv2.RANSAC,                 # mevcut
    "MAGSAC++": cv2.USAC_MAGSAC,
    "USAC_ACCURATE": cv2.USAC_ACCURATE,
}


def takvim(n=N_KARE):
    h = np.ones(n, dtype=bool)
    for b, s in KESINTI:
        h[b:s] = False
    return h


def kos(cons_rows, ladder_rows, gt, healthy, cfg):
    n = len(gt)
    est = VoLadderEstimator(RGB_1080P, cfg, spacing=SPACING)
    preds = np.zeros((n, 3)); preds[0] = gt[0]
    est.update_ladder(0, None, None, gt[0], True)
    for k in range(1, n):
        mc = cons_rows[k - 1] if k - 1 < len(cons_rows) else None
        ml = ladder_rows[k - 1] if k - 1 < len(ladder_rows) else None
        preds[k] = est.update_ladder(k, mc, ml, gt[k] if healthy[k] else None, bool(healthy[k]))
    return preds


def procrustes_tani(preds, gt, healthy):
    """Kesinti bloklarinda ORTALAMA yon (derece) ve olcek hatasi — asil hedef bunlar."""
    aci, olcek = [], []
    bas = None
    for i in range(len(healthy) + 1):
        icinde = i < len(healthy) and not healthy[i]
        if icinde and bas is None:
            bas = i
        elif not icinde and bas is not None:
            p = preds[bas:i, :2] - preds[bas, :2]
            g = gt[bas:i, :2] - gt[bas, :2]
            if len(p) > 10 and (p ** 2).sum() > 1e-9:
                H = g.T @ p
                U, S, Vt = np.linalg.svd(H)
                R = U @ Vt
                if np.linalg.det(R) < 0:
                    U[:, -1] *= -1; R = U @ Vt
                aci.append(abs(np.degrees(np.arctan2(R[1, 0], R[0, 0]))))
                olcek.append(abs(float(S.sum() / max((p ** 2).sum(), 1e-9)) - 1.0))
            bas = None
    return (float(np.mean(aci)) if aci else 0.0), (float(np.mean(olcek)) if olcek else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracker", default="raft")
    ap.add_argument("--senaryo", type=int, default=6)
    ap.add_argument("--esikler", default="1.0,2.0,3.0")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    cons_npz = GOREV2 / "outputs" / f"nokta_{args.tracker}_cons.npz"
    lad_npz = GOREV2 / "outputs" / f"nokta_{args.tracker}_sp{SPACING}.npz"
    for p in (cons_npz, lad_npz):
        if not p.exists():
            sys.exit(f"HATA: nokta onbellegi yok: {p.name}\n"
                     f"  once: python gorev2/nokta_onbellek.py --tracker {args.tracker}"
                     f" [--spacing {SPACING}]")
    ob_cons, ob_lad = onbellek_yukle(cons_npz), onbellek_yukle(lad_npz)

    gt_sub = data_io.subsample_competition(data_io.load_translations(GT_CSV))
    n = min(N_KARE, len(gt_sub), len(ob_cons["offsets"]), len(ob_lad["offsets"]))
    gt = gt_sub.iloc[:n][evaluation.GT_COLS].to_numpy()
    cfg = replace(EstimatorConfig(method="homografi"), smooth_alpha=SMOOTH)
    yar = takvim(n)
    ras = [evaluation.make_outage_schedule(n_frames=n, rng=s) for s in range(args.senaryo)]

    print(f"=== GEOMETRI TARAMASI (tracker={args.tracker}, {n} kare) ===")
    print(f"{'yontem':<15}{'esik':>6}{'yarisma':>11}{'rastgele':>11}{'yon°':>8}{'olcek%':>8}{'karar':>9}")
    print("-" * 70)

    temel = None
    satirlar = []
    for ad, yontem in YONTEMLER.items():
        for esik in [float(x) for x in args.esikler.split(",")]:
            t0 = time.perf_counter()
            gk = dict(robust_method=yontem, ransac_thresh=esik)
            dc = olcum_uret(ob_cons, RGB_1080P, **gk).iloc[:n - 1]
            dl = olcum_uret(ob_lad, RGB_1080P, **gk).iloc[:n - 1]
            cr = [evaluation._meas_from_row(r) for r in dc.to_dict("records")]
            lr = [evaluation._meas_from_row(r) for r in dl.to_dict("records")]
            py = kos(cr, lr, gt, yar, cfg)
            y = evaluation.competition_error(py, gt)
            r = float(np.mean([evaluation.competition_error(kos(cr, lr, gt, h, cfg), gt)
                               for h in ras]))
            a_err, s_err = procrustes_tani(py, gt, yar)
            mevcut = (ad == "RANSAC" and abs(esik - 2.0) < 1e-9)
            if mevcut:
                temel = (y, r)
            karar = "MEVCUT" if mevcut else ""
            print(f"{ad:<15}{esik:>6.1f}{y:>10.3f}m{r:>10.3f}m{a_err:>8.2f}{100*s_err:>8.2f}"
                  f"{karar:>9}   ({time.perf_counter()-t0:.0f}s)")
            satirlar.append([ad, esik, y, r, a_err, 100 * s_err])

    # karar
    L = ["=== GOREV 2 — GEOMETRI TARAMASI (yontem 6: MAGSAC++) ===",
         f"tracker={args.tracker}, {n} kare, homografi, spacing={SPACING}, smooth_alpha={SMOOTH}",
         f"kabul: yarisma VE rastgele ortalamasinda >=%1 kazanc", ""]
    if temel:
        L.append(f"TEMEL (RANSAC, esik 2.0): yarisma {temel[0]:.3f} m | rastgele {temel[1]:.3f} m")
    L.append("")
    L.append(f"{'yontem':<15}{'esik':>6}{'yarisma':>11}{'rastgele':>11}{'yon°':>8}{'olcek%':>8}{'karar':>9}")
    en_iyi = None
    for ad, esik, y, r, a_err, s_err in satirlar:
        kabul = temel and y < temel[0] * 0.99 and r < temel[1] * 0.99
        karar = "KABUL" if kabul else ("MEVCUT" if ad == "RANSAC" and abs(esik - 2.0) < 1e-9 else "")
        L.append(f"{ad:<15}{esik:>6.1f}{y:>10.3f}m{r:>10.3f}m{a_err:>8.2f}{s_err:>8.2f}{karar:>9}")
        if kabul and (en_iyi is None or (y + r) < (en_iyi[2] + en_iyi[3])):
            en_iyi = [ad, esik, y, r, a_err, s_err]
    L.append("")
    if en_iyi:
        L += ["--- KAZANAN ---",
              f"  {en_iyi[0]}, ransac_thresh={en_iyi[1]}",
              f"  yarisma  {en_iyi[2]:.3f} m ({100*(en_iyi[2]-temel[0])/temel[0]:+.1f}%)",
              f"  rastgele {en_iyi[3]:.3f} m ({100*(en_iyi[3]-temel[1])/temel[1]:+.1f}%)",
              f"  yon hatasi {en_iyi[4]:.2f}° | olcek hatasi %{en_iyi[5]:.2f}",
              "", ">>> TAVSIYE: bu geometri ayarini UYGULA"]
    else:
        L.append(">>> TAVSIYE: mevcut (RANSAC, 2.0) kalsin — daha iyisi bulunamadi.")
    metin = "\n".join(L)
    (OUT / f"geometri_tarama_{args.tracker}.txt").write_text(metin, encoding="utf-8")
    with open(OUT / f"geometri_tarama_{args.tracker}.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["yontem", "esik", "yarisma_m", "rastgele_m", "yon_derece", "olcek_yuzde"])
        w.writerows(satirlar)
    print()
    print(metin)


if __name__ == "__main__":
    main()
