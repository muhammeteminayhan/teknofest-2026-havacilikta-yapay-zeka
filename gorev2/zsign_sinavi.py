# -*- coding: utf-8 -*-
"""GOREV 2 — z_sign KANIT KAPISI: eslestirilmis A/B, butun yataklarda.

DEGISIKLIK
----------
`scale_alignment.fit_alignment` icindeki z isareti karari.

  ESKI : guc = sum|dz_unit| * c > 1,0 m  ise  sign(sum dz_unit*gt_dz)
  YENI : GT irtifa gezinmesi >= z_min_hareket VE tutarlilik >= z_min_kanit ise
         sign(...); degilse onceki isaret korunur

GEREKCE (tani, gorev2/tani_pencere1.py):
  Sinav penceresi 1'in kalibrasyon penceresinde ucak DUZ ucuyor (GT z gezinmesi
  0,188 m). Buna ragmen eski kapi GECIYOR (guc 1,46 m) cunku `guc` OLCUMUN
  gurultusunu sinyal saniyor. Isaret t = -1,04e-4 gibi kendi gurultu tabaninin
  4 kat altindaki bir sayidan seciliyor ve -1 cikiyor. Puanlanan blokta ucak
  ~10 m irtifa degistirince dusey hata 11 m oluyor.

  Yataklarda olculen kararsizlik (saglikli karelerin yuzdesi):
    video1 %27 · video2 %38 (son karar YANLIS) · v25_rgb_b %26 · v25_termal_b %40

KABUL OLCUTU (olcumden once yazildi)
------------------------------------
  1. genelleme sinavi: HER IKI pencerede de kotulesmeyecek, en az birinde kazanacak
  2. 8 yatak ortalamasi kotulesmeyecek
  3. hicbir yatakta felaket (>+3 m) olmayacak

CALISTIR
--------
    .venv\\Scripts\\python gorev2\\zsign_sinavi.py --n-perm 3
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np

GOREV2 = Path(__file__).resolve().parent
sys.path.insert(0, str(GOREV2))

import kabul                                                        # noqa: E402
from kabul import PENCERELER, yatak_olc                             # noqa: E402
from nokta_onbellek import YATAKLAR, yatak_npz, yataklari_dogrula   # noqa: E402
from gorev2_src import scale_alignment as SA                        # noqa: E402
from gorev2_src.vo_pipeline import EstimatorConfig                  # noqa: E402
from yontem_sinavi import olc_2b                                    # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

YATAK_LISTESI = ["video1", "video2", "v25_rgb_a", "v25_rgb_b",
                 "v25_termal_a", "v25_termal_b", "v24"]


# --------------------------------------------------------------------------- eski kural
_YENI = SA.fit_alignment


def _eski_fit(E, G, dz_unit, gt_dz, gt_z, prev_z_sign=1.0, **_yoksay):
    """Degisiklikten ONCEKI davranisin birebir kopyasi (temel hat kolu)."""
    E = np.asarray(E, dtype=float)
    G = np.asarray(G, dtype=float)
    if len(E) < 3 or (E ** 2).sum() < 1e-12:
        return None
    H = G.T @ E
    U, S, Vt = np.linalg.svd(H)
    M = U @ Vt
    c = float(S.sum() / (E ** 2).sum())
    if not np.isfinite(c) or c <= 0:
        return None
    pred = (M @ E.T).T * c
    g_norm = float(np.linalg.norm(G))
    residual = float(np.linalg.norm(G - pred) / g_norm) if g_norm > 0 else 1.0
    t = float(np.sum(np.asarray(dz_unit) * np.asarray(gt_dz)))
    guc = float(np.sum(np.abs(dz_unit)) * c)
    z_sign = float(np.sign(t)) if (guc > 1.0 and t != 0.0) else prev_z_sign
    return SA.Alignment(M=M, height=c, z_sign=z_sign,
                        mean_gt_z=float(np.mean(gt_z)), n_used=len(E),
                        residual_ratio=residual)


@contextmanager
def kural(ad: str):
    """`eski` / `yeni` kuralini butun cagri yerlerinde etkinlestir."""
    import gorev2_src.vo_pipeline as VP
    hedef = _eski_fit if ad == "eski" else _YENI
    onceki_sa, onceki_vp = SA.fit_alignment, VP.fit_alignment
    SA.fit_alignment = hedef
    VP.fit_alignment = hedef
    try:
        yield
    finally:
        SA.fit_alignment = onceki_sa
        VP.fit_alignment = onceki_vp


# --------------------------------------------------------------------------- olcum
def sinav_pencereleri(cfg, n_perm):
    """genelleme sinavinin iki penceresi, permutasyon basina."""
    out = {1: [], 2: []}
    for k in range(n_perm):
        r = yatak_olc("yarisma", cfg, 8, k, {})
        out[1].append(r["pencere1"])
        out[2].append(r["pencere2"])
    return {w: np.array(v) for w, v in out.items()}


def yatak_dizisi(y, cfg, n_perm):
    if YATAKLAR[y].get("iki_boyut", False):
        return np.array([olc_2b(y, cfg, k, {}, kabul.MOTOR) for k in range(n_perm)])
    return np.array([yatak_olc(y, cfg, 8, k, {})["ana"] for k in range(n_perm)])


def _karar(d):
    se = 2 * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
    return ("KAZANDI" if d.mean() + se < 0 else
            "KAYBETTI" if d.mean() - se > 0 else "esit"), se


def tara(args):
    """Esik duyarliligi: sonuc tek bir esik degerine mi bagli?

    Mekanizma geregi beklenti INSENSITIF olmasi: kapi kapaliysa onceki isaret
    korunuyor, acikken de elimizdeki butun veride karar +1 cikiyor. Bunu
    varsaymak yerine olcuyoruz.
    """
    yataklar = args.yatak or ["video2", "v25_termal_b"]
    kombinasyon = [(h, k) for h in (0.5, 1.0, 2.0, 5.0) for k in (0.30,)]
    kombinasyon += [(2.0, k) for k in (0.0, 0.15, 0.50, 0.80)]
    print("=" * 88)
    print(f"ESIK DUYARLILIGI — sinav (2 pencere) + {', '.join(yataklar)}")
    print(f"  {args.n_perm} permutasyon · temel hat (ESKI kural) referans")
    print("=" * 88)
    with kural("eski"):
        t_cfg = EstimatorConfig(method="homografi")
        A = sinav_pencereleri(t_cfg, args.n_perm)
        AY = {y: yatak_dizisi(y, t_cfg, args.n_perm) for y in yataklar}
    basliklar = "".join(f"{y:>14}" for y in yataklar)
    print(f"\n  {'hareket':>8}{'kanit':>7}{'pencere1':>11}{'pencere2':>11}{basliklar}")
    print("  " + "-" * (26 + 22 + 14 * len(yataklar)))
    print(f"  {'ESKI':>8}{'-':>7}{A[1].mean():>11.2f}{A[2].mean():>11.2f}"
          + "".join(f"{AY[y].mean():>14.2f}" for y in yataklar))
    for h, k in kombinasyon:
        cfg = EstimatorConfig(method="homografi", z_min_hareket=h, z_min_kanit=k)
        with kural("yeni"):
            B = sinav_pencereleri(cfg, args.n_perm)
            BY = {y: yatak_dizisi(y, cfg, args.n_perm) for y in yataklar}
        print(f"  {h:>8.2f}{k:>7.2f}{B[1].mean():>11.2f}{B[2].mean():>11.2f}"
              + "".join(f"{BY[y].mean():>14.2f}" for y in yataklar), flush=True)
    print("\n  Sonuc genis bir esik araliginda ayni ise deger tek yataga uydurulmamistir.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=3)
    ap.add_argument("--hareket", type=float, default=2.0)
    ap.add_argument("--kanit", type=float, default=0.30)
    ap.add_argument("--yatak", nargs="*", default=None)
    ap.add_argument("--tara", action="store_true",
                    help="esik duyarliligi: sinav + secili yataklarda esik taramasi")
    args = ap.parse_args()

    cfg = EstimatorConfig(method="homografi", z_min_hareket=args.hareket,
                          z_min_kanit=args.kanit)
    if args.tara:
        tara(args)
        return
    print("=" * 88)
    print("z_sign KANIT KAPISI — eslestirilmis A/B")
    print(f"  yeni esikler: z_min_hareket={cfg.z_min_hareket} m · "
          f"z_min_kanit={cfg.z_min_kanit}")
    print(f"  {args.n_perm} permutasyon")
    print("=" * 88)

    # ---------------------------------------------------------------- 1) sinav
    t0 = time.perf_counter()
    with kural("eski"):
        A = sinav_pencereleri(cfg, args.n_perm)
    with kural("yeni"):
        B = sinav_pencereleri(cfg, args.n_perm)
    print(f"\nGENELLEME SINAVI (puanli oturumun kendi kareleri)  "
          f"[{time.perf_counter()-t0:.0f}s]")
    print(f"  {'':<14}{'ESKI':>14}{'YENI':>14}{'fark':>16}")
    for w in (1, 2):
        d = B[w] - A[w]
        k, se = _karar(d)
        print(f"  pencere {w:<6}{A[w].mean():>9.2f}±{A[w].std(ddof=1):<4.2f}"
              f"{B[w].mean():>9.2f}±{B[w].std(ddof=1):<4.2f}"
              f"{d.mean():>+10.2f} ±{se:4.2f} {k:>9}")
    oa = np.mean([A[w].mean() for w in (1, 2)])
    ob = np.mean([B[w].mean() for w in (1, 2)])
    print(f"  {'ORTALAMA':<14}{oa:>14.2f}{ob:>14.2f}{ob-oa:>+16.2f}")

    # ---------------------------------------------------------------- 2) yataklar
    yataklar = [y for y in YATAK_LISTESI
                if yatak_npz(y, None, kabul.MOTOR).exists()
                and yatak_npz(y, 8, kabul.MOTOR).exists()]
    supheli = yataklari_dogrula(yataklar, kabul.MOTOR, sessiz=True)
    if supheli:
        print(f"\n!!! ESLESMESI SUPHELI YATAK(LAR) ATLANIYOR: {supheli}")
        yataklar = [y for y in yataklar if y not in supheli]

    print(f"\nYONTEM SINAVI — {len(yataklar)} bagimsiz yatak")
    print(f"  {'yatak':<15}{'ESKI':>14}{'YENI':>14}{'fark':>16}")
    print("  " + "-" * 60)
    ozet = {}
    for y in yataklar:
        t0 = time.perf_counter()
        with kural("eski"):
            a = yatak_dizisi(y, cfg, args.n_perm)
        with kural("yeni"):
            b = yatak_dizisi(y, cfg, args.n_perm)
        ozet[y] = (a, b)
        d = b - a
        k, se = _karar(d)
        print(f"  {y:<15}{a.mean():>9.2f}±{a.std(ddof=1):<4.2f}"
              f"{b.mean():>9.2f}±{b.std(ddof=1):<4.2f}"
              f"{d.mean():>+10.2f} ±{se:4.2f} {k:>9}  ({time.perf_counter()-t0:.0f}s)",
              flush=True)
    if ozet:
        ea = np.mean([ozet[y][0].mean() for y in ozet])
        eb = np.mean([ozet[y][1].mean() for y in ozet])
        kot = [y for y in ozet if ozet[y][1].mean() - ozet[y][0].mean() > 3.0]
        print("  " + "-" * 60)
        print(f"  {'ORTALAMA':<15}{ea:>14.2f}{eb:>14.2f}{eb-ea:>+16.2f}")
        print(f"\n  KABUL: sinav iki pencerede de kotulesmemeli · yatak ortalamasi "
              f"kotulesmemeli · felaket (>+3 m) yok")
        print(f"  felaket yatak: {kot if kot else 'YOK'}")


if __name__ == "__main__":
    main()
