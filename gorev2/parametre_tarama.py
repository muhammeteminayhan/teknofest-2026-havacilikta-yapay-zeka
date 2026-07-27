# -*- coding: utf-8 -*-
"""GOREV 2 PARAMETRE TARAMASI — riskisiz ayarlarin resmi metrikle A/B'si.

`test_yatagi.py` ile ayni veri ve GERCEK yarisma kesinti takvimi kullanilir; fark:
burada `EstimatorConfig` parametreleri tek tek degistirilip resmi metrik
(`competition_error`) olculur.

ASIRI UYUM (overfitting) KORUMASI — bir ayar YALNIZCA su iki kosulu birden saglarsa
kabul edilir:
  1) yarismanin gercek takviminde hatayi dusurur,
  2) FARKLI rastgele kesinti takvimlerinin ortalamasinda da dusurur (genelleme).
Tek senaryoda iyi gorunen ama digerlerinde bozan ayar REDDEDILIR.

Cikti: gorev2/degerlendirme/parametre_tarama.txt|csv

Calistir (proje kokunden):
    .venv\\Scripts\\python gorev2\\parametre_tarama.py
    .venv\\Scripts\\python gorev2\\parametre_tarama.py --senaryo 8
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GOREV2 = Path(__file__).resolve().parent
OUT = GOREV2 / "degerlendirme"

_SRC = GOREV2 / "src"
if "gorev2_src" not in sys.modules:
    _spec = importlib.util.spec_from_file_location(
        "gorev2_src", _SRC / "__init__.py", submodule_search_locations=[str(_SRC)])
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules["gorev2_src"] = _mod
    _spec.loader.exec_module(_mod)

from gorev2_src import data_io, evaluation                       # noqa: E402
from gorev2_src.calibration import RGB_1080P                     # noqa: E402
from gorev2_src.vo_pipeline import EstimatorConfig, VoLadderEstimator  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

GT_CSV = ROOT / "THYZ_2026_Ornek_Veri_Seti" / "THYZ_2026_Ornek_Veri_1_translation.csv"
CACHE_CONS = GOREV2 / "outputs" / "olcumler_video1_raft.csv"
CACHE_LADDER = GOREV2 / "outputs" / "merdiven_video1_raft.csv"

KESINTI_BLOKLARI = [(451, 900), (1051, 1550), (1651, 2250)]
N_KARE = 2250
SPACING = 8


def yarisma_takvimi(n=N_KARE):
    h = np.ones(n, dtype=bool)
    for bas, son in KESINTI_BLOKLARI:
        h[bas:son] = False
    return h


def kos(cons_rows, ladder_rows, gt, healthy, cfg, spacing=SPACING):
    """Onceden parse edilmis olcumlerle hizli kosum (parse maliyeti bir kez odenir)."""
    n = len(gt)
    est = VoLadderEstimator(RGB_1080P, cfg, spacing=spacing)
    preds = np.zeros((n, 3))
    preds[0] = gt[0]
    est.update_ladder(0, None, None, gt[0], True)
    for k in range(1, n):
        mc = cons_rows[k - 1] if k - 1 < len(cons_rows) else None
        ml = ladder_rows[k - 1] if k - 1 < len(ladder_rows) else None
        preds[k] = est.update_ladder(k, mc, ml, gt[k] if healthy[k] else None, bool(healthy[k]))
    return float(np.linalg.norm(preds - gt, axis=1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--senaryo", type=int, default=6,
                    help="genelleme icin rastgele kesinti takvimi sayisi")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    gt_sub = data_io.subsample_competition(data_io.load_translations(GT_CSV))
    cons = pd.read_csv(CACHE_CONS)
    ladder = pd.read_csv(CACHE_LADDER)
    n = min(N_KARE, len(gt_sub), len(cons) + 1, len(ladder) + 1)
    gt_sub = gt_sub.iloc[:n].reset_index(drop=True)
    gt = gt_sub[evaluation.GT_COLS].to_numpy()

    # olcumleri BIR KEZ parse et
    cons_rows = [evaluation._meas_from_row(r) for r in cons.iloc[:n - 1].to_dict("records")]
    ladder_rows = [evaluation._meas_from_row(r) for r in ladder.iloc[:n - 1].to_dict("records")]

    yarisma = yarisma_takvimi(n)
    rastgele = [evaluation.make_outage_schedule(n_frames=n, rng=s) for s in range(args.senaryo)]

    def olc(cfg):
        y = kos(cons_rows, ladder_rows, gt, yarisma, cfg)
        r = [kos(cons_rows, ladder_rows, gt, h, cfg) for h in rastgele]
        return y, float(np.mean(r))

    # mevcut canli ayar: live_estimator RAFT'ta method="homografi" seciyor
    temel_cfg = EstimatorConfig(method="homografi")
    t_yar, t_ras = olc(temel_cfg)
    print(f"=== TEMEL (canli ayar: homografi) ===")
    print(f"  yarisma takvimi : {t_yar:.3f} m")
    print(f"  {args.senaryo} rastgele ort: {t_ras:.3f} m\n")
    print(f"{'parametre':<22} {'deger':>10} {'yarisma':>10} {'rastgele':>10} {'karar':>18}")
    print("-" * 76)

    ADAYLAR = {
        "method": ["homografi", "fuzyon", "benzerlik"],
        "calib_window": [30, 45, 60, 90, 120],
        "min_calib": [8, 12, 20],
        "min_move_m": [0.05, 0.10, 0.20],
        "min_inlier_ratio": [0.30, 0.40, 0.50],
        "min_kept": [20, 30, 50],
        "max_dz_unit": [0.10, 0.15, 0.25],
        "cv_alpha": [0.15, 0.30, 0.50],
        "residual_gate": [0.30, 0.50, 0.70],
        "smooth_alpha": [0.4, 0.6, 0.8, 1.0],
        "max_step_xy": [2.5, 3.5, 5.0],
    }

    satirlar = []
    kazananlar = {}
    for ad, degerler in ADAYLAR.items():
        en_iyi = None
        for d in degerler:
            cfg = replace(temel_cfg, **{ad: d})
            y, r = olc(cfg)
            mevcut = (d == getattr(temel_cfg, ad))
            # KABUL: her iki senaryoda da temelden iyi (>= %1 kazanc)
            iyi = (y < t_yar * 0.99) and (r < t_ras * 0.99)
            karar = "MEVCUT" if mevcut else ("KABUL" if iyi else "-")
            print(f"{ad:<22} {str(d):>10} {y:>9.3f}m {r:>9.3f}m {karar:>18}")
            satirlar.append([ad, d, y, r, karar])
            if iyi and (en_iyi is None or (y + r) < (en_iyi[1] + en_iyi[2])):
                en_iyi = (d, y, r)
        if en_iyi:
            kazananlar[ad] = en_iyi
        print()

    # --- kazananlari BIRLESTIR ve dogrula ---
    L = ["=== GOREV 2 PARAMETRE TARAMASI ===",
         f"temel (homografi): yarisma {t_yar:.3f} m | {args.senaryo} rastgele ort {t_ras:.3f} m",
         "",
         "Kabul olcutu: HEM yarisma takviminde HEM rastgele takvimlerin ortalamasinda",
         "temelden en az %1 daha iyi (tek senaryoya asiri uyumu onler).",
         ""]
    if kazananlar:
        L.append("--- tek tek kazanan ayarlar ---")
        for ad, (d, y, r) in kazananlar.items():
            L.append(f"  {ad:<20} = {d!r:>10}   (yarisma {y:.3f} m, rastgele {r:.3f} m)")
        birlesik = replace(temel_cfg, **{ad: v[0] for ad, v in kazananlar.items()})
        by, br = olc(birlesik)
        L += ["", "--- BIRLESIK ayar ---",
              f"  yarisma  : {by:.3f} m  (temel {t_yar:.3f} m, fark {by-t_yar:+.3f})",
              f"  rastgele : {br:.3f} m  (temel {t_ras:.3f} m, fark {br-t_ras:+.3f})"]
        birlesik_iyi = (by < t_yar) and (br < t_ras)
        L.append("")
        if birlesik_iyi:
            L.append(">>> TAVSIYE: birlesik ayari UYGULA")
            L.append(f"    {', '.join(f'{a}={v[0]!r}' for a,v in kazananlar.items())}")
        else:
            L.append(">>> TAVSIYE: birlesik ayar TEMELDEN IYI DEGIL — tek tek en iyisini kullan")
            L.append("    (parametreler etkilesiyor; birlesim her zaman toplanmaz)")
    else:
        L += [">>> SONUC: hicbir parametre degisikligi anlamli kazanc saglamadi.",
              "    Sistem parametrelere karsi gurbuz; kazanc icin YAPISAL degisiklik",
              "    gerekiyor (loop closure / yer tanima)."]

    metin = "\n".join(L)
    (OUT / "parametre_tarama.txt").write_text(metin, encoding="utf-8")
    with open(OUT / "parametre_tarama.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["parametre", "deger", "yarisma_hata_m", "rastgele_ort_m", "karar"])
        w.writerows(satirlar)
    print(metin)
    print(f"\nCikti: {OUT}")


if __name__ == "__main__":
    main()
