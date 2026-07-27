# -*- coding: utf-8 -*-
"""GOREV 2 — SISTEMATIK SURUKLENME SONUMLEME taramasi (asil kazanc adayi).

TESHIS (bkz. degerlendirme/): kesinti bloklarinda hata RASTGELE DEGIL, SISTEMATIK.
Procrustes cozumlemesi: tahmin yorungesi, GT'nin ~2 derece DONDURULMUS ve %3-4,5
OLCEKLENMIS halidir; bu iki bilesen duzeltilse blok hatasi 6,28 m yerine 1,45 m olurdu.

Kaynak (vo_pipeline):
  * olcek : her adimda `height *= (1 - dz_unit)` -> 449 adimda kucuk gurultu birikir
  * yon   : her adimda `psi += yaw`             -> kucuk sistematik yaw birikir

Bu arac, yeni sonumleme parametrelerini resmi metrikle tarar:
  height_dz_katsayi (0 = yukseklik sabit), yaw_katsayi (0 = yon sabit), yaw_olu_bant.

KABUL OLCUTU (asiri uyuma karsi): hem yarismanin GERCEK kesinti takviminde hem de
farkli rastgele takvimlerin ortalamasinda >= %1 kazanc.

Calistir: .venv\\Scripts\\python gorev2\\surukleme_tarama.py
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

from gorev2_src import data_io, evaluation                        # noqa: E402
from gorev2_src.calibration import RGB_1080P                      # noqa: E402
from gorev2_src.vo_pipeline import EstimatorConfig, VoLadderEstimator  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

GT_CSV = ROOT / "THYZ_2026_Ornek_Veri_Seti" / "THYZ_2026_Ornek_Veri_1_translation.csv"
CACHE_CONS = GOREV2 / "outputs" / "olcumler_video1_raft.csv"
CACHE_LADDER = GOREV2 / "outputs" / "merdiven_video1_raft.csv"
KESINTI = [(451, 900), (1051, 1550), (1651, 2250)]
N_KARE, SPACING = 2250, 8


def takvim(n=N_KARE):
    h = np.ones(n, dtype=bool)
    for b, s in KESINTI:
        h[b:s] = False
    return h


def kos(cons_rows, ladder_rows, gt, healthy, cfg):
    n = len(gt)
    est = VoLadderEstimator(RGB_1080P, cfg, spacing=SPACING)
    preds = np.zeros((n, 3))
    preds[0] = gt[0]
    est.update_ladder(0, None, None, gt[0], True)
    for k in range(1, n):
        mc = cons_rows[k - 1] if k - 1 < len(cons_rows) else None
        ml = ladder_rows[k - 1] if k - 1 < len(ladder_rows) else None
        preds[k] = est.update_ladder(k, mc, ml, gt[k] if healthy[k] else None, bool(healthy[k]))
    return preds


def blok_ort(preds, gt, healthy):
    h = np.linalg.norm(preds - gt, axis=1)
    out, bas = [], None
    for i in range(len(healthy) + 1):
        icinde = i < len(healthy) and not healthy[i]
        if icinde and bas is None:
            bas = i
        elif not icinde and bas is not None:
            out.append(float(h[bas:i].mean())); bas = None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--senaryo", type=int, default=6)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    gt_sub = data_io.subsample_competition(data_io.load_translations(GT_CSV))
    cons = pd.read_csv(CACHE_CONS); ladder = pd.read_csv(CACHE_LADDER)
    n = min(N_KARE, len(gt_sub), len(cons) + 1, len(ladder) + 1)
    gt_sub = gt_sub.iloc[:n].reset_index(drop=True)
    gt = gt_sub[evaluation.GT_COLS].to_numpy()
    cons_rows = [evaluation._meas_from_row(r) for r in cons.iloc[:n - 1].to_dict("records")]
    ladder_rows = [evaluation._meas_from_row(r) for r in ladder.iloc[:n - 1].to_dict("records")]

    yar = takvim(n)
    ras = [evaluation.make_outage_schedule(n_frames=n, rng=s) for s in range(args.senaryo)]

    def olc(cfg):
        py = kos(cons_rows, ladder_rows, gt, yar, cfg)
        y = evaluation.competition_error(py, gt)
        r = float(np.mean([evaluation.competition_error(
            kos(cons_rows, ladder_rows, gt, h, cfg), gt) for h in ras]))
        return y, r, blok_ort(py, gt, yar)

    # temel = parametre taramasinda kazanan ayar (smooth_alpha=1.0)
    temel = replace(EstimatorConfig(method="homografi"), smooth_alpha=1.0)
    t_y, t_r, t_blok = olc(temel)
    print(f"=== TEMEL (homografi, smooth_alpha=1.0) ===")
    print(f"  yarisma {t_y:.3f} m | {args.senaryo} rastgele ort {t_r:.3f} m")
    print(f"  blok ortalamalari: {[f'{b:.2f}' for b in t_blok]}\n")
    print(f"{'ayar':<38} {'yarisma':>10} {'rastgele':>10} {'karar':>8}")
    print("-" * 70)

    ADAYLAR = []
    for k in (0.0, 0.25, 0.5, 0.75, 1.0):
        ADAYLAR.append((f"height_dz_katsayi={k}", dict(height_dz_katsayi=k)))
    for k in (0.0, 0.5, 0.8, 1.0):
        ADAYLAR.append((f"yaw_katsayi={k}", dict(yaw_katsayi=k)))
    for b in (0.0, 0.0005, 0.001, 0.002, 0.005):
        ADAYLAR.append((f"yaw_olu_bant={b}", dict(yaw_olu_bant=b)))

    satirlar, kazanan = [], {}
    for ad, kw in ADAYLAR:
        cfg = replace(temel, **kw)
        y, r, _ = olc(cfg)
        mevcut = all(getattr(temel, a) == v for a, v in kw.items())
        iyi = (y < t_y * 0.99) and (r < t_r * 0.99)
        karar = "MEVCUT" if mevcut else ("KABUL" if iyi else "-")
        print(f"{ad:<38} {y:>9.3f}m {r:>9.3f}m {karar:>8}")
        satirlar.append([ad, y, r, karar])
        if iyi:
            a = list(kw)[0]
            if a not in kazanan or (y + r) < (kazanan[a][1] + kazanan[a][2]):
                kazanan[a] = (list(kw.values())[0], y, r)

    L = ["=== GOREV 2 — SISTEMATIK SURUKLENME SONUMLEME ===",
         f"temel (homografi, smooth_alpha=1.0): yarisma {t_y:.3f} m | rastgele {t_r:.3f} m",
         "",
         "Teshis: kesinti hatasi sistematik (yon ~2 derece + olcek %3-4,5).",
         "Kabul: yarisma VE rastgele ortalamasinda >=%1 kazanc.",
         "",
         f"{'ayar':<38} {'yarisma':>10} {'rastgele':>10} {'karar':>8}"]
    for s in satirlar:
        L.append(f"{s[0]:<38} {s[1]:>9.3f}m {s[2]:>9.3f}m {s[3]:>8}")
    L.append("")

    if kazanan:
        L.append("--- tek tek kazananlar ---")
        for a, (v, y, r) in kazanan.items():
            L.append(f"  {a} = {v}   (yarisma {y:.3f} m, rastgele {r:.3f} m)")
        birlesik_kw = {a: v[0] for a, v in kazanan.items()}
        bcfg = replace(temel, **birlesik_kw)
        by, br, bblok = olc(bcfg)
        blok_bozdu = any(b > t + 1e-6 for b, t in zip(bblok, t_blok))
        L += ["", "--- BIRLESIK ---",
              f"  {birlesik_kw}",
              f"  yarisma  : {by:.3f} m  (temel {t_y:.3f}, {100*(by-t_y)/t_y:+.1f}%)",
              f"  rastgele : {br:.3f} m  (temel {t_r:.3f}, {100*(br-t_r)/t_r:+.1f}%)",
              f"  blok ort : {[f'{b:.2f}' for b in bblok]}  (temel {[f'{b:.2f}' for b in t_blok]})"]
        if by < t_y and br < t_r and not blok_bozdu:
            L += ["", ">>> TAVSIYE: BIRLESIK ayari UYGULA"]
        else:
            en = min(kazanan.items(), key=lambda kv: kv[1][1] + kv[1][2])
            L += ["", f">>> TAVSIYE: birlesik degil, TEK EN IYI ayari uygula: {en[0]}={en[1][0]}"]
    else:
        L.append(">>> SONUC: sonumleme kazanc saglamadi.")

    metin = "\n".join(L)
    (OUT / "surukleme_tarama.txt").write_text(metin, encoding="utf-8")
    with open(OUT / "surukleme_tarama.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh); w.writerow(["ayar", "yarisma_m", "rastgele_m", "karar"]); w.writerows(satirlar)
    print(); print(metin); print(f"\nCikti: {OUT}")


if __name__ == "__main__":
    main()
