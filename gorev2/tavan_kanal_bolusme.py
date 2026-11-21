# -*- coding: utf-8 -*-
"""GOREV 2 — TAVAN OLCUMU: yonu bir motordan, olcegi digerinden almak kazandirir mi?

NEDEN VAR
---------
`tani_kanal_farki.py` net bir gerilim buldu (video1, 3 permutasyon):

    motor        hata      YON      OLCEK
    raft         3,904     1,707°   %2,99
    searaft_g    4,271     1,341°   %3,93

SEA-RAFT **yonu %21 daha iyi**, **olcegi %31 daha kotu** veriyor. Net sonuc kaybediyor
cunku kesintide hata olcek-baskin.

Bu, depoda ZATEN kayitli bir gerilimin baska bir noktasidir. `PairMeasurement`
dokumani: *"yon ve olcek hatalari zit esik istiyor... cozum: oteleme/olcek siki
fitten, yon gevsek fitten"* — sistem bunu iki ESIKLE yapiyor (`hom_yaw_gevsek`).
Ayni deseni iki MOTORLA kurmak dogrudan test edilebilir bir hipotezdir:

    yon      <- SEA-RAFT   (daha temiz donme ayristirmasi)
    oteleme  <- RAFT       (daha iyi duzlem uzakligi / olcek)

BU BIR TAVAN OLCUMUDUR: sevk edilirse her karede IKI motor kosmasi gerekir
(~2x akis maliyeti). Kazanc kucukse aile kapanir — ensemble turunda ayni takas
zaten reddedilmisti (degerlendirme/ENSEMBLE.md).

CALISTIR
--------
    .venv\\Scripts\\python gorev2\\tavan_kanal_bolusme.py
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

GOREV2 = Path(__file__).resolve().parent
sys.path.insert(0, str(GOREV2))

import kabul                                                        # noqa: E402
from kabul import _kos, procrustes_tani, yarisma_takvimi, PENCERELER  # noqa: E402
from nokta_onbellek import YATAKLAR, yatak_yukle, yatak_npz         # noqa: E402
from gorev2_src.evaluation import competition_error                 # noqa: E402
from gorev2_src.vo_pipeline import EstimatorConfig                  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

# Yon tasiyan sutunlar — geri kalan her sey oteleme/olcek motorundan gelir.
YON_SUTUNLARI = ["hom_yaw", "hom_yaw_gevsek", "hom_inlier_gevsek", "sim_yaw"]


def _birlestir(df_oteleme, df_yon):
    """Oteleme motorunun tablosunu al, YON sutunlarini yon motorundan yaz."""
    d = df_oteleme.copy()
    n = min(len(d), len(df_yon))
    for s in YON_SUTUNLARI:
        if s in d.columns and s in df_yon.columns:
            d.iloc[:n, d.columns.get_loc(s)] = df_yon[s].to_numpy()[:n]
    return d


def kos_yatak(yatak, m_ote, m_yon, cfg, spacing, karistir):
    gt, c_o, l_o, prof, n_o = yatak_yukle(yatak, spacing=spacing, karistir=karistir,
                                          tracker=m_ote)
    _, c_y, l_y, _, n_y = yatak_yukle(yatak, spacing=spacing, karistir=karistir,
                                      tracker=m_yon)
    n = min(n_o, n_y)
    gt = gt[:n]
    cons = _birlestir(c_o.iloc[:n - 1], c_y.iloc[:n - 1])
    lad = _birlestir(l_o.iloc[:n - 1], l_y.iloc[:n - 1])

    if yatak == "yarisma":
        gercek = np.load(YATAKLAR["yarisma"]["gt_npz"])["saglikli"]
        out = []
        for w, p in PENCERELER.items():
            son = p["son"]
            h = gercek[:son].copy()
            a, b = p["puan"]
            h[a:b + 1] = False
            pr = _kos(cons.iloc[:son - 1], lad.iloc[:son - 1], gt[:son], h,
                      prof, cfg, spacing)
            out.append(float(np.linalg.norm(pr[a:b + 1] - gt[a:b + 1], axis=1).mean()))
        return float(np.mean(out)), np.nan, np.nan

    h = yarisma_takvimi(n)
    pr = _kos(cons, lad, gt, h, prof, cfg, spacing)
    yon, olcek = procrustes_tani(pr, gt, h)
    return competition_error(pr, gt), yon, 100 * olcek


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yatak", nargs="*", default=["video1", "yarisma"])
    ap.add_argument("--oteleme", default="raft")
    ap.add_argument("--yon", default="searaft_g")
    ap.add_argument("--n-perm", type=int, default=3)
    args = ap.parse_args()

    cfg = EstimatorConfig(method="homografi")
    print("=" * 82)
    print(f"KANAL BOLUSMESI TAVANI — oteleme/olcek: {args.oteleme} · "
          f"yon: {args.yon}")
    print(f"  {args.n_perm} permutasyon · SEVK EDILEBILIR DEGIL (iki motor kosar)")
    print("=" * 82)
    print(f"\n  {'yatak':<12}{'MEVCUT':>10}{'BOLUSME':>10}{'fark':>10}"
          f"{'YON [deg]':>22}{'OLCEK [%]':>20}")
    print(f"  {'':<12}{'':>30}{'mevcut -> bolusme':>22}{'mevcut -> bolusme':>20}")
    print("  " + "-" * 74)

    for y in args.yatak:
        for m in (args.oteleme, args.yon):
            if not (yatak_npz(y, None, m).exists() and yatak_npz(y, 8, m).exists()):
                print(f"  {y:<12} {m} onbellegi yok — atlandi")
                break
        else:
            A, B, YA, YB, OA, OB = [], [], [], [], [], []
            for k in range(args.n_perm):
                kabul.MOTOR = args.oteleme
                from kabul import yatak_olc
                r = yatak_olc(y, cfg, 8, k, {})
                A.append(r["ana"]); YA.append(r.get("yon", np.nan))
                OA.append(r.get("olcek", np.nan))
                e, yn, ol = kos_yatak(y, args.oteleme, args.yon, cfg, 8, k)
                B.append(e); YB.append(yn); OB.append(ol)
            kabul.MOTOR = "raft"
            a, b = np.mean(A), np.mean(B)
            print(f"  {y:<12}{a:>10.3f}{b:>10.3f}{b - a:>+10.3f}"
                  f"{np.nanmean(YA):>11.3f} ->{np.nanmean(YB):>8.3f}"
                  f"{np.nanmean(OA):>11.3f} ->{np.nanmean(OB):>8.3f}", flush=True)

    print("\n  YORUM ANAHTARI")
    print("   Kazanc anlamliysa (>0,2 m) ve iki yatakta da varsa: mekanizma dogrulanir,")
    print("     ama sevkiyat icin yine de iki motor maliyeti tartilmali.")
    print("   Kazanc yoksa: yon iyilesmesi olcek kaybini karsilamiyor -> aile kapanir.")


if __name__ == "__main__":
    main()
