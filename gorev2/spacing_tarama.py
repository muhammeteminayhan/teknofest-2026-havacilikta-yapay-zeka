# -*- coding: utf-8 -*-
"""GOREV 2 — KEYFRAME ARALIGI (spacing) taramasi.

Merdiven (ladder) kestirimi, konumu son keyframe'e gore hesaplar; boylece hata
her karede degil, yalnizca keyframe gecislerinde birikir. `spacing` bu gecislerin
sikligidir:
  * kucuk spacing -> sik gecis -> daha cok birikim, ama kisa taban cizgisi (kolay eslesme)
  * buyuk spacing -> az gecis -> az birikim, ama uzun taban cizgisi (eslesme zorlasir)

Bu tarama, her `spacing` icin merdiven olcumlerini (RAFT) bir kez uretip onbellekler
(`outputs/merdiven_video1_raft_sp<N>.csv`) ve resmi metrikle karsilastirir.

DIKKAT: her yeni spacing degeri icin video bir kez taranir (birkac dakika).

Calistir:
    .venv\\Scripts\\python gorev2\\spacing_tarama.py --degerler 4,8,12,16,24
"""
from __future__ import annotations

import argparse
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

from gorev2_src import data_io, evaluation                        # noqa: E402
from gorev2_src.calibration import RGB_1080P                      # noqa: E402
from gorev2_src.vo_pipeline import EstimatorConfig, VoLadderEstimator  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

VIDEO = ROOT / "THYZ_2026_Ornek_Veri_Seti" / "THYZ_2026_Ornek_Veri_1.MP4"
GT_CSV = ROOT / "THYZ_2026_Ornek_Veri_Seti" / "THYZ_2026_Ornek_Veri_1_translation.csv"
CACHE_CONS = GOREV2 / "outputs" / "olcumler_video1_raft.csv"
KESINTI = [(451, 900), (1051, 1550), (1651, 2250)]
N_KARE = 2250


def takvim(n=N_KARE):
    h = np.ones(n, dtype=bool)
    for b, s in KESINTI:
        h[b:s] = False
    return h


def merdiven_onbellek(spacing: int) -> pd.DataFrame:
    yol = GOREV2 / "outputs" / (f"merdiven_video1_raft.csv" if spacing == 8
                                else f"merdiven_video1_raft_sp{spacing}.csv")
    if yol.exists():
        return pd.read_csv(yol)
    print(f"  spacing={spacing}: merdiven olcumleri uretiliyor (video taranacak)...")
    from gorev2_src.raft_matching import RaftTracker
    tracker = RaftTracker(RGB_1080P, model_size="large")
    return evaluation.build_ladder_measurements(VIDEO, RGB_1080P, yol,
                                                spacing=spacing, progress=True,
                                                tracker=tracker)


def kos(cons_rows, ladder_rows, gt, healthy, cfg, spacing):
    n = len(gt)
    est = VoLadderEstimator(RGB_1080P, cfg, spacing=spacing)
    preds = np.zeros((n, 3)); preds[0] = gt[0]
    est.update_ladder(0, None, None, gt[0], True)
    for k in range(1, n):
        mc = cons_rows[k - 1] if k - 1 < len(cons_rows) else None
        ml = ladder_rows[k - 1] if k - 1 < len(ladder_rows) else None
        preds[k] = est.update_ladder(k, mc, ml, gt[k] if healthy[k] else None, bool(healthy[k]))
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--degerler", default="4,8,12,16,24")
    ap.add_argument("--senaryo", type=int, default=6)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    gt_sub = data_io.subsample_competition(data_io.load_translations(GT_CSV))
    cons = pd.read_csv(CACHE_CONS)
    cfg = replace(EstimatorConfig(method="homografi"), smooth_alpha=1.0)

    sonuc = []
    for sp in [int(x) for x in args.degerler.split(",")]:
        ladder = merdiven_onbellek(sp)
        n = min(N_KARE, len(gt_sub), len(cons) + 1, len(ladder) + 1)
        gt = gt_sub.iloc[:n][evaluation.GT_COLS].to_numpy()
        cons_rows = [evaluation._meas_from_row(r) for r in cons.iloc[:n - 1].to_dict("records")]
        ladder_rows = [evaluation._meas_from_row(r) for r in ladder.iloc[:n - 1].to_dict("records")]
        yar = takvim(n)
        ras = [evaluation.make_outage_schedule(n_frames=n, rng=s) for s in range(args.senaryo)]
        y = evaluation.competition_error(kos(cons_rows, ladder_rows, gt, yar, cfg, sp), gt)
        r = float(np.mean([evaluation.competition_error(
            kos(cons_rows, ladder_rows, gt, h, cfg, sp), gt) for h in ras]))
        print(f"spacing={sp:>3}: yarisma {y:.3f} m | {args.senaryo} rastgele ort {r:.3f} m")
        sonuc.append((sp, y, r))

    temel = next((s for s in sonuc if s[0] == 8), None)
    L = ["=== GOREV 2 — KEYFRAME ARALIGI (spacing) TARAMASI ===",
         "config: homografi, smooth_alpha=1.0", "",
         f"{'spacing':>8} {'yarisma':>10} {'rastgele':>10} {'karar':>10}"]
    en_iyi = min(sonuc, key=lambda s: s[1] + s[2])
    for sp, y, r in sonuc:
        karar = "MEVCUT" if sp == 8 else ""
        if temel and sp != 8 and y < temel[1] * 0.99 and r < temel[2] * 0.99:
            karar = "KABUL"
        L.append(f"{sp:>8} {y:>9.3f}m {r:>9.3f}m {karar:>10}")
    L.append("")
    if temel and en_iyi[0] != 8 and en_iyi[1] < temel[1] * 0.99 and en_iyi[2] < temel[2] * 0.99:
        L.append(f">>> TAVSIYE: spacing={en_iyi[0]} kullan "
                 f"(yarisma {en_iyi[1]:.3f} vs {temel[1]:.3f}, "
                 f"rastgele {en_iyi[2]:.3f} vs {temel[2]:.3f})")
    else:
        L.append(">>> TAVSIYE: spacing=8 (mevcut) kalsin — daha iyisi bulunamadi.")
    metin = "\n".join(L)
    (OUT / "spacing_tarama.txt").write_text(metin, encoding="utf-8")
    print(); print(metin); print(f"\nCikti: {OUT}")


if __name__ == "__main__":
    main()
