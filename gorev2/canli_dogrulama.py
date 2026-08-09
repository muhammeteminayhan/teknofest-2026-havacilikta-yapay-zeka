# -*- coding: utf-8 -*-
"""GOREV 2 — CANLI SISTEM DOGRULAMASI (gercek video, onbellek YOK).

NEDEN: A/B taramalari nokta onbellegi uzerinden kosar (hizli). Ama yarismada
`LivePositionEstimator` gercek kareleri isler: eslestirme, donmus-kare tespiti,
keyframe yonetimi ve zamanlama dahil. Bir kazancin gercek olmasi icin BU yolda
da gorunmesi gerekir.

Ayrica yarismanin sert kisiti HIZ: gecen oturumda 2250 kare 46,5 dakikada islendi
(1,24 s/kare, uc gorev birlikte) ve pencere 90 dakikaydi. Bu arac kare basi sureyi
de olcer.

Calistir:
    .venv\\Scripts\\python gorev2\\canli_dogrulama.py
    .venv\\Scripts\\python gorev2\\canli_dogrulama.py --n 400     # hizli deneme
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

GOREV2 = Path(__file__).resolve().parent
ROOT = GOREV2.parent
OUT = GOREV2 / "degerlendirme"
sys.path.insert(0, str(GOREV2))

from nokta_onbellek import _tracker_kur          # noqa: F401,E402  (gorev2_src'yi kaydeder)
from geometri_tarama import takvim, KESINTI      # noqa: E402
from gorev2_src import data_io, evaluation       # noqa: E402
from gorev2_src.calibration import RGB_1080P     # noqa: E402
from gorev2_src.live_estimator import LivePositionEstimator   # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

VIDEO = ROOT / "THYZ_2026_Ornek_Veri_Seti" / "THYZ_2026_Ornek_Veri_1.MP4"
GT_CSV = ROOT / "THYZ_2026_Ornek_Veri_Seti" / "THYZ_2026_Ornek_Veri_1_translation.csv"
PENCERE_DK = 90.0          # yarisma penceresi
GECEN_S_KARE = 1.24        # gecen oturum: 46,5 dk / 2250 kare (uc gorev birlikte)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None, help="ilk N kare")
    ap.add_argument("--etiket", default="teslim")
    ap.add_argument("--motor", default="raft", help="raft | lk | auto")
    ap.add_argument("--alpha", type=float, default=None,
                    help="smooth_alpha'yi elle ayarla (verilmezse motor recetesi gecerli)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    gt_sub = data_io.subsample_competition(data_io.load_translations(GT_CSV))
    gt_all = gt_sub[evaluation.GT_COLS].to_numpy()

    # DIKKAT: ilk parametre `engine`, profil DEGIL — profil ilk kareden belirlenir.
    cfg = None
    if args.alpha is not None:      # recete yerine elle konfigurasyon (once/sonra olcumu)
        from gorev2_src.vo_pipeline import EstimatorConfig
        cfg = EstimatorConfig(method="homografi", smooth_alpha=args.alpha)
    est = LivePositionEstimator(engine=args.motor, config=cfg)
    preds, k = [], 0
    t0 = time.perf_counter()
    hedef = args.n
    saglik = takvim(len(gt_all))
    for _, gray in data_io.iter_competition_frames(VIDEO, progress=True):
        if k >= len(gt_all) or (hedef is not None and k >= hedef):
            break
        h = bool(saglik[k])
        preds.append(est.process_frame(gray, gt_all[k] if h else None, h))
        k += 1
    gecen = time.perf_counter() - t0

    # Sessiz yedege dusmeyi YAKALA: RAFT kurulamazsa sistem LK'ye duser ve
    # RAFT receteleri hic devreye girmez.
    if args.motor == "raft" and est.engine_used != "raft":
        print(f"\n!!! UYARI: motor '{est.engine_used}' — RAFT kurulamadi, "
              f"bu kosu RAFT recetesini DOGRULAMAZ !!!\n")

    preds = np.asarray(preds)
    gt = gt_all[:len(preds)]
    yar = saglik[:len(preds)]
    kes = ~yar
    hata = evaluation.competition_error(preds, gt)
    d = np.linalg.norm(preds - gt, axis=1)
    e = (preds - gt)[kes]
    rms = [float(np.sqrt((e[:, i] ** 2).mean())) for i in range(3)]
    ts = est.timing_stats

    # yarisma suresi tahmini: bu gorevin kare basi maliyeti + diger iki gorev
    s_kare_g2 = ts.get("ort_ms", 0) / 1000.0
    L = [f"=== GOREV 2 — CANLI SISTEM DOGRULAMASI ({args.etiket}) ===",
         f"video: {VIDEO.name} | {len(preds)} kare | motor={est.engine_used} "
         f"| yontem={est.config.method} | smooth_alpha={est.config.smooth_alpha}",
         "",
         f"ORTALAMA OKLID HATASI (resmi metrik, tum kareler): {hata:.3f} m",
         f"  kesinti karelerinde: {d[kes].mean():.3f} m | saglikli: {d[yar].mean():.4f} m",
         f"  eksen RMS (kesinti): x {rms[0]:.3f} | y {rms[1]:.3f} | z {rms[2]:.3f} m",
         ""]
    for i, (b, s) in enumerate(KESINTI, 1):
        b2, s2 = min(b, len(preds)), min(s, len(preds))
        if b2 < s2:
            L.append(f"  blok{i} ({s2-b2:3d} kare): ort {d[b2:s2].mean():6.3f} m | "
                     f"bitis {d[s2-1]:6.3f} m")
    L += ["",
          f"HIZ: kare basi ort {ts.get('ort_ms', 0):.0f} ms | p95 {ts.get('p95_ms', 0):.0f} ms",
          f"  toplam koşu {gecen/60:.1f} dk ({len(preds)} kare)",
          f"  yarismada beklenen: ~{s_kare_g2 * 2250 / 60:.0f} dk (yalniz Gorev 2) — "
          f"gecen oturum uc gorevle {GECEN_S_KARE * 2250 / 60:.0f} dk, pencere {PENCERE_DK:.0f} dk"]
    metin = "\n".join(L)
    (OUT / f"canli_dogrulama_{args.etiket}.txt").write_text(metin, encoding="utf-8")
    print()
    print(metin)


if __name__ == "__main__":
    main()
