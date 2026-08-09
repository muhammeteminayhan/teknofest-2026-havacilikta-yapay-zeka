# -*- coding: utf-8 -*-
"""GOREV 2 — ESLESTIRICI A/B: bir tracker'i mevcut RAFT hattina karsi olcer.

NEDEN AYRI BIR ARAC: `geometri_tarama.py` robust kestirici/esik varyantlarini RASTGELE
(karisik uzunlukta) kesinti takvimlerinde karsilastirir. Ama yarismanin YAPISAL kosulu
UZUN kesintidir (gercek oturum logu: 449/499/599 kare). Bu arac, kabul olcutunu
`gorev2/degerlendirme/OZET.md`'deki kurala gore uygular:

    bir degisiklik ancak YARISMA TAKVIMINDE **ve** 15 UZUN-KESINTI senaryosunda
    mevcut hatti gecerse kabul edilir (asiri uyum korumasi).

Farkli eslestiriciler farkli piksel gurultusu urettigi icin RANSAC esigi her tracker
icin ayri taranir — SuperPoint turunda optimum esik RAFT'inkinin 8 kati cikmisti.

Calistir:
    .venv\\Scripts\\python gorev2\\eslestirici_ab.py --tracker vggt
    .venv\\Scripts\\python gorev2\\eslestirici_ab.py --tracker vggt --esikler 1,2,3,5,8
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

GOREV2 = Path(__file__).resolve().parent
ROOT = GOREV2.parent
OUT = GOREV2 / "degerlendirme"
sys.path.insert(0, str(GOREV2))

from nokta_onbellek import onbellek_yukle, olcum_uret                  # noqa: E402
from geometri_tarama import procrustes_tani, takvim                    # noqa: E402
from slam_degerlendirme import uzun_takvim                             # noqa: E402
from gorev2_src import data_io, evaluation                             # noqa: E402
from gorev2_src.calibration import RGB_1080P                           # noqa: E402
from gorev2_src.vo_pipeline import EstimatorConfig, VoLadderEstimator  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

GT_CSV = ROOT / "THYZ_2026_Ornek_Veri_Seti" / "THYZ_2026_Ornek_Veri_1_translation.csv"
N_KARE, SPACING, SMOOTH = 2250, 8, 1.0
N_UZUN = 15                       # uzun-kesinti senaryo sayisi (OZET.md ile ayni)
TEMEL_ESIK = 1.0                  # mevcut sistemin ayari (geometry.py varsayilani)


def _olcumler(ob, n, **gk):
    df = olcum_uret(ob, RGB_1080P, **gk).iloc[:n - 1]
    return [evaluation._meas_from_row(r) for r in df.to_dict("records")], df


def kos(cons_rows, ladder_rows, gt, healthy, cfg, spacing=SPACING):
    """Merdiven kestiricisini bir sağlık çizelgesinde koştur (spacing parametreli)."""
    n = len(gt)
    est = VoLadderEstimator(RGB_1080P, cfg, spacing=spacing)
    preds = np.zeros((n, 3)); preds[0] = gt[0]
    est.update_ladder(0, None, None, gt[0], True)
    for k in range(1, n):
        mc = cons_rows[k - 1] if k - 1 < len(cons_rows) else None
        ml = ladder_rows[k - 1] if k - 1 < len(ladder_rows) else None
        preds[k] = est.update_ladder(k, mc, ml, gt[k] if healthy[k] else None, bool(healthy[k]))
    return preds


def degerlendir(cr, lr, gt, cfg, yar, uzunlar, spacing=SPACING):
    """(yarisma_hatasi, uzun_kesinti_ortalamasi, uzun_kesinti_listesi, yon, olcek)"""
    py = kos(cr, lr, gt, yar, cfg, spacing)
    y = evaluation.competition_error(py, gt)
    uz = [evaluation.competition_error(kos(cr, lr, gt, h, cfg, spacing), gt) for h in uzunlar]
    a_err, s_err = procrustes_tani(py, gt, yar)
    return y, float(np.mean(uz)), np.array(uz), a_err, 100 * s_err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracker", required=True, help="degerlendirilecek tracker (orn. vggt)")
    ap.add_argument("--temel", default="raft", help="karsilastirma hatti (varsayilan raft)")
    ap.add_argument("--esikler", default="1.0,2.0,3.0,5.0,8.0")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    def _yukle(tr):
        c = GOREV2 / "outputs" / f"nokta_{tr}_cons.npz"
        l = GOREV2 / "outputs" / f"nokta_{tr}_sp{SPACING}.npz"
        for p in (c, l):
            if not p.exists():
                sys.exit(f"HATA: nokta onbellegi yok: {p.name}\n"
                         f"  once: python gorev2/nokta_onbellek.py --tracker {tr} [--spacing {SPACING}]")
        return onbellek_yukle(c), onbellek_yukle(l)

    ob_c, ob_l = _yukle(args.tracker)
    tb_c, tb_l = _yukle(args.temel)

    gt_sub = data_io.subsample_competition(data_io.load_translations(GT_CSV))
    n = min(N_KARE, len(gt_sub), len(ob_c["offsets"]), len(ob_l["offsets"]),
            len(tb_c["offsets"]), len(tb_l["offsets"]))
    gt = gt_sub.iloc[:n][evaluation.GT_COLS].to_numpy()
    cfg = replace(EstimatorConfig(method="homografi"), smooth_alpha=SMOOTH)
    yar = takvim(n)
    uzunlar = [uzun_takvim(n, rng=s) for s in range(N_UZUN)]

    print(f"=== ESLESTIRICI A/B: {args.tracker} vs {args.temel} ({n} kare) ===\n")

    # --- temel hat (mevcut sistem ayari) ---
    t0 = time.perf_counter()
    cr, lr = _olcumler(tb_c, n, ransac_thresh=TEMEL_ESIK)[0], _olcumler(tb_l, n, ransac_thresh=TEMEL_ESIK)[0]
    ty, tu, tu_liste, ta, ts = degerlendir(cr, lr, gt, cfg, yar, uzunlar)
    print(f"TEMEL ({args.temel}, esik {TEMEL_ESIK}): yarisma {ty:.3f} m | "
          f"{N_UZUN} uzun-kesinti {tu:.3f} m | yon {ta:.2f}° | olcek %{ts:.2f}  ({time.perf_counter()-t0:.0f}s)\n")

    print(f"{'esik':>6}{'yarisma':>11}{'uzun':>11}{'yon°':>8}{'olcek%':>8}{'inlier':>8}{'uzun W/L':>10}{'karar':>8}")
    print("-" * 72)

    satirlar, en_iyi = [], None
    for esik in [float(x) for x in args.esikler.split(",")]:
        t0 = time.perf_counter()
        gk = dict(ransac_thresh=esik)
        cr, dfc = _olcumler(ob_c, n, **gk)
        lr, _ = _olcumler(ob_l, n, **gk)
        y, u, u_liste, a_err, s_err = degerlendir(cr, lr, gt, cfg, yar, uzunlar)
        inl = float(np.nanmean(dfc["hom_inlier_ratio"].replace(0, np.nan)))
        kazanc = int((u_liste < tu_liste).sum())
        kabul = (y < ty) and (u < tu)
        karar = "KABUL" if kabul else ""
        print(f"{esik:>6.1f}{y:>10.3f}m{u:>10.3f}m{a_err:>8.2f}{s_err:>8.2f}{inl:>8.2f}"
              f"{kazanc:>7}/{N_UZUN}{karar:>8}   ({time.perf_counter()-t0:.0f}s)")
        satirlar.append([esik, y, u, a_err, s_err, inl, kazanc])
        if kabul and (en_iyi is None or (y + u) < (en_iyi[1] + en_iyi[2])):
            en_iyi = satirlar[-1]

    # --- rapor ---
    L = [f"=== GOREV 2 — ESLESTIRICI A/B: {args.tracker} vs {args.temel} ===",
         f"{n} kare, homografi, spacing={SPACING}, smooth_alpha={SMOOTH}",
         f"kabul olcutu: yarisma takviminde VE {N_UZUN} uzun-kesinti ortalamasinda temeli gecmek",
         "",
         f"TEMEL ({args.temel}, esik {TEMEL_ESIK}): yarisma {ty:.3f} m | uzun {tu:.3f} m "
         f"| yon {ta:.2f}° | olcek %{ts:.2f}",
         "",
         f"{'esik':>6}{'yarisma':>11}{'uzun':>11}{'yon°':>8}{'olcek%':>8}{'inlier':>8}{'uzun W/L':>10}{'karar':>8}"]
    for esik, y, u, a_err, s_err, inl, kz in satirlar:
        karar = "KABUL" if (y < ty and u < tu) else ""
        L.append(f"{esik:>6.1f}{y:>10.3f}m{u:>10.3f}m{a_err:>8.2f}{s_err:>8.2f}{inl:>8.2f}"
                 f"{kz:>7}/{N_UZUN}{karar:>8}")
    L.append("")
    if en_iyi:
        L += ["--- KAZANAN ---",
              f"  {args.tracker}, ransac_thresh={en_iyi[0]}",
              f"  yarisma {en_iyi[1]:.3f} m ({100*(en_iyi[1]-ty)/ty:+.1f}%)",
              f"  uzun    {en_iyi[2]:.3f} m ({100*(en_iyi[2]-tu)/tu:+.1f}%)",
              f"  yon {en_iyi[3]:.2f}° (temel {ta:.2f}°) | olcek %{en_iyi[4]:.2f} (temel %{ts:.2f})",
              "", f">>> TAVSIYE: {args.tracker} hattini UYGULA"]
    else:
        L.append(f">>> TAVSIYE: mevcut {args.temel} hatti kalsin — {args.tracker} gecemedi.")
    metin = "\n".join(L)
    (OUT / f"eslestirici_ab_{args.tracker}.txt").write_text(metin, encoding="utf-8")
    with open(OUT / f"eslestirici_ab_{args.tracker}.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["esik", "yarisma_m", "uzun_m", "yon_derece", "olcek_yuzde", "inlier", "uzun_kazanc"])
        w.writerows(satirlar)
    print()
    print(metin)


if __name__ == "__main__":
    main()
