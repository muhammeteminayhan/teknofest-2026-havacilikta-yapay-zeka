# -*- coding: utf-8 -*-
"""GOREV 2 TEST YATAGI — yarismanin GERCEK kesinti takvimiyle resmi metrik olcumu.

Neden: puanli oturumda PK 55,29 (8/8) aldik. Resmi metrik (sartname Denklem 2,
`evaluation.competition_error`) TUM karelerde ortalama Oklid hatasidir; saglikli
karelerde herkes GT'yi aynalar (hata 0), yani **puani yalnizca kesinti kareleri belirler**.

Bu arac, iyilestirmeleri sahadaki kosulun BIREBIR kopyasinda olcer:
  * veri  : ornek veri seti video 1 (tam GT) — yarisma temposuna altornekli
  * takvim: puanli oturumdan cikarilan GERCEK kesinti yapisi
              echo 0-450 (451) | KESINTI 451-899 (449) | echo 900-1050 (151)
            | KESINTI 1051-1549 (499) | echo 1550-1650 (101) | KESINTI 1651-2249 (599)
            (toplam 703 saglikli + 1547 kesinti — yarisma logu ile birebir ayni)
  * olcum : onbellekli RAFT olcumleri (outputs/*_raft.csv) -> senaryo saniyeler icinde kosar

Cikti: gorev2/degerlendirme/
    temel_olcum.txt   — ozet rapor (resmi metrik + blok blok tani)
    temel_olcum.csv   — kare basi hata (grafik/analiz icin)
    hata_egrisi.png   — kesinti bloklarinda hatanin buyume egrisi

Calistir (proje kokunden):
    .venv\\Scripts\\python gorev2\\test_yatagi.py
    .venv\\Scripts\\python gorev2\\test_yatagi.py --etiket spacing16 --spacing 16
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GOREV2 = Path(__file__).resolve().parent
OUT = GOREV2 / "degerlendirme"

# gorev2/src'i 'gorev2_src' adiyla yukle (sistem/src ile isim cakismasini onler)
_SRC = GOREV2 / "src"
if "gorev2_src" not in sys.modules:
    _spec = importlib.util.spec_from_file_location(
        "gorev2_src", _SRC / "__init__.py", submodule_search_locations=[str(_SRC)])
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules["gorev2_src"] = _mod
    _spec.loader.exec_module(_mod)

from gorev2_src import data_io, evaluation                      # noqa: E402
from gorev2_src.calibration import RGB_1080P                    # noqa: E402
from gorev2_src.vo_pipeline import EstimatorConfig              # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

VIDEO = ROOT / "THYZ_2026_Ornek_Veri_Seti" / "THYZ_2026_Ornek_Veri_1.MP4"
GT_CSV = ROOT / "THYZ_2026_Ornek_Veri_Seti" / "THYZ_2026_Ornek_Veri_1_translation.csv"
CACHE_CONS = GOREV2 / "outputs" / "olcumler_video1_raft.csv"
CACHE_LADDER = GOREV2 / "outputs" / "merdiven_video1_raft.csv"

# --- Puanli oturumdan cikarilan GERCEK kesinti takvimi ---
# (yarisma logu: 703 saglikli + 1547 kesinti kare)
KESINTI_BLOKLARI = [(451, 900), (1051, 1550), (1651, 2250)]
N_KARE = 2250


def yarisma_takvimi(n: int = N_KARE) -> np.ndarray:
    """True = saglikli (GT var), False = kesinti (kestirim yapilacak)."""
    healthy = np.ones(n, dtype=bool)
    for bas, son in KESINTI_BLOKLARI:
        healthy[bas:son] = False
    return healthy


def _veri_yukle(spacing: int):
    if not CACHE_CONS.exists() or not CACHE_LADDER.exists():
        sys.exit(f"HATA: olcum onbellegi yok.\n  {CACHE_CONS}\n  {CACHE_LADDER}\n"
                 f"(Bunlar RAFT ile bir kez uretilir; bkz. evaluation.build_*_measurements)")
    gt_sub = data_io.subsample_competition(data_io.load_translations(GT_CSV))
    cons = pd.read_csv(CACHE_CONS)
    ladder = pd.read_csv(CACHE_LADDER)
    n = min(N_KARE, len(gt_sub), len(cons) + 1, len(ladder) + 1)
    return gt_sub.iloc[:n].reset_index(drop=True), cons.iloc[:n - 1], ladder.iloc[:n - 1], n


def blok_tanisi(hata: np.ndarray, healthy: np.ndarray) -> list[dict]:
    """Her kesinti blogu icin: uzunluk, ortalama/son/maks hata, buyume hizi."""
    sonuc = []
    bas = None
    for i in range(len(healthy) + 1):
        icinde = i < len(healthy) and not healthy[i]
        if icinde and bas is None:
            bas = i
        elif not icinde and bas is not None:
            h = hata[bas:i]
            sonuc.append(dict(bas=bas, son=i - 1, uzunluk=i - bas,
                              ort=float(h.mean()), bitis=float(h[-1]),
                              maks=float(h.max()),
                              buyume=float(h[-1]) / max(i - bas, 1)))
            bas = None
    return sonuc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--etiket", default="temel", help="cikti dosya adi eki (A/B icin)")
    ap.add_argument("--spacing", type=int, default=8, help="keyframe merdiven araligi")
    ap.add_argument("--yontem", default="homografi", help="EstimatorConfig.method")
    ap.add_argument("--smooth-alpha", type=float, default=None,
                    help="olcum EMA katsayisi (1.0 = yumusatma kapali)")
    ap.add_argument("--grafik", action="store_true", default=True)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    gt_sub, cons, ladder, n = _veri_yukle(args.spacing)
    healthy = yarisma_takvimi(n)
    gt = gt_sub[evaluation.GT_COLS].to_numpy()

    print(f"=== GOREV 2 TEST YATAGI ({args.etiket}) ===")
    print(f"kare: {n} | saglikli: {int(healthy.sum())} | kesinti: {int((~healthy).sum())}")
    print(f"takvim: yarismanin GERCEK kesinti yapisi {KESINTI_BLOKLARI}")
    print(f"config: yontem={args.yontem} spacing={args.spacing}\n")

    cfg = EstimatorConfig(method=args.yontem)
    if args.smooth_alpha is not None:
        from dataclasses import replace as _replace
        cfg = _replace(cfg, smooth_alpha=args.smooth_alpha)
    print(f"        smooth_alpha={cfg.smooth_alpha}")
    preds = evaluation.run_estimator_ladder(
        cons, ladder, gt_sub, healthy, RGB_1080P, cfg, spacing=args.spacing)

    hata = np.linalg.norm(preds - gt, axis=1)
    resmi = evaluation.competition_error(preds, gt)
    kesinti = evaluation.outage_error(preds, gt, healthy)

    # kiyas tabanlari (gorsel bilgi kullanmayan)
    tut = evaluation.run_baseline(gt_sub, healthy, kind="tut")
    hiz = evaluation.run_baseline(gt_sub, healthy, kind="sabit_hiz")

    L = [f"=== GOREV 2 OLCUM — {args.etiket} ===",
         f"veri   : ornek veri seti video 1 (tam GT), {n} kare",
         f"takvim : yarismanin GERCEK kesinti yapisi ({int(healthy.sum())} saglikli / "
         f"{int((~healthy).sum())} kesinti)",
         f"config : yontem={args.yontem}, spacing={args.spacing}",
         "",
         ">>> RESMI METRIK (sartname Denklem 2 = PK puanini belirleyen deger) <<<",
         f"  ortalama Oklid hatasi (TUM kareler) : {resmi:.3f} m",
         f"  yalniz kesinti karelerinde          : {kesinti:.3f} m",
         "",
         "kiyas tabanlari (gorsel bilgi YOK):",
         f"  'tut'       (son saglikli konumu tut) : {evaluation.competition_error(tut, gt):.3f} m",
         f"  'sabit_hiz' (son hizla duz devam)     : {evaluation.competition_error(hiz, gt):.3f} m",
         "",
         "--- kesinti bloklari (tani) ---",
         f"{'blok':>16} {'uzunluk':>8} {'ort hata':>10} {'bitis':>9} {'maks':>9} {'m/kare':>8}"]
    for b in blok_tanisi(hata, healthy):
        L.append(f"{b['bas']:>6}-{b['son']:<9} {b['uzunluk']:>8} {b['ort']:>9.2f}m "
                 f"{b['bitis']:>8.2f}m {b['maks']:>8.2f}m {b['buyume']:>8.4f}")
    L += ["",
          f"hata persentilleri (tum kareler): "
          f"medyan {np.median(hata):.2f} | %90 {np.percentile(hata,90):.2f} | maks {hata.max():.2f} m"]

    metin = "\n".join(L)
    (OUT / f"{args.etiket}_olcum.txt").write_text(metin, encoding="utf-8")
    pd.DataFrame({"kare": np.arange(n), "saglikli": healthy, "hata_m": hata,
                  "pred_x": preds[:, 0], "pred_y": preds[:, 1], "pred_z": preds[:, 2],
                  "gt_x": gt[:, 0], "gt_y": gt[:, 1], "gt_z": gt[:, 2]}
                 ).to_csv(OUT / f"{args.etiket}_olcum.csv", index=False)
    print(metin)

    if args.grafik:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(14, 4.5))
        ax.plot(hata, lw=1.0, label=f"{args.etiket} (ort {resmi:.2f} m)")
        for bas, son in KESINTI_BLOKLARI:
            if bas < n:
                ax.axvspan(bas, min(son, n), color="red", alpha=0.10)
        ax.set_xlabel("kare"); ax.set_ylabel("konum hatasi [m]")
        ax.set_title(f"Gorev 2 — hata egrisi ({args.etiket}); kirmizi = kesinti blogu")
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT / f"{args.etiket}_hata_egrisi.png", dpi=90)
        plt.close()
        print(f"\ngrafik: {args.etiket}_hata_egrisi.png")

    print(f"\nCikti: {OUT}")


if __name__ == "__main__":
    main()
