# -*- coding: utf-8 -*-
"""GOREV 2 — TAVAN OLCUMU: yataklarda GERCEKTEN dusuk kaliteli kare var mi?

NEDEN VAR
---------
Sartname bulaniklik, olu piksel, goruntu tekrari ve tamamen kayip kareye izin
verebilecegini belirtiyor; Manus raporu (§3.2) bu yuzden her karede bir kalite skoru
hesaplanmasini oneriyor. Depoda su an yalnizca **donmus kare** tespiti var
(`vo_pipeline.PairMeasurement.compute` -> `frozen`).

Kalite kapisi yazmadan once sunu bilmek gerekir: elimizdeki 8 yatakta dusuk kaliteli
kare **var mi**? Yoksa kapi bizim verimizde OLU olur; o zaman karar "bilinmeyen final
ucusu icin sigorta mi?" sorusuna iner ve kazanc beklenmez.

OLCULENLER (kare basina, gri tonda)
  laplas_var   : Laplacian varyansi — dusuk = BULANIK (klasik olcut)
  parlaklik    : ortalama yogunluk (0-255)
  kontrast     : yogunluk standart sapmasi — cok dusuk = duz/sisli/asiri pozlanmis
  kirpilmis    : 0 ya da 255'e yapisan piksel orani — olu piksel / asiri pozlama
  tekrar       : bir onceki kareyle ortalama mutlak fark (dusuk = DONMUS)

CALISTIR
--------
    .venv\\Scripts\\python gorev2\\tavan_kare_kalitesi.py
    .venv\\Scripts\\python gorev2\\tavan_kare_kalitesi.py --adim 1     # her kare
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np

GOREV2 = Path(__file__).resolve().parent
sys.path.insert(0, str(GOREV2))

from nokta_onbellek import YATAKLAR, yatak_kareler                # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

TUM = ["yarisma", "video1", "video2", "v25_rgb_a", "v25_rgb_b",
       "v25_termal_a", "v25_termal_b", "v24"]


def kare_kalitesi(gray: np.ndarray, onceki: np.ndarray | None) -> dict:
    g = gray if gray.ndim == 2 else cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    # buyuk karelerde hiz icin altornekle — olcutlerin sirasi korunur
    kucuk = g[::2, ::2] if g.shape[0] > 1200 else g
    lap = float(cv2.Laplacian(kucuk, cv2.CV_64F).var())
    kirpilmis = float(((g == 0) | (g == 255)).mean())
    tekrar = (float(np.abs(g.astype(np.int16) - onceki.astype(np.int16)).mean())
              if onceki is not None and onceki.shape == g.shape else np.nan)
    return dict(laplas_var=lap, parlaklik=float(g.mean()),
                kontrast=float(g.std()), kirpilmis=kirpilmis, tekrar=tekrar)


def yatak_tara(ad: str, adim: int, n_maks: int) -> list[dict]:
    kayit, onceki, i = [], None, 0
    for _, gray in yatak_kareler(YATAKLAR[ad], progress=False):
        if i % adim == 0:
            kayit.append(kare_kalitesi(gray, onceki))
            onceki = gray if gray.ndim == 2 else cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        i += 1
        if n_maks and len(kayit) >= n_maks:
            break
    return kayit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yatak", nargs="*", default=TUM)
    ap.add_argument("--adim", type=int, default=5, help="her N. kare")
    ap.add_argument("--n-maks", type=int, default=450)
    args = ap.parse_args()

    print("=" * 96)
    print(f"KARE KALITESI DAGILIMI — her {args.adim}. kare, yatak basina en fazla "
          f"{args.n_maks} ornek")
    print("=" * 96)
    print(f"\n  {'yatak':<15}{'n':>5}" + "".join(
        f"{b:>17}" for b in ("laplas_var", "kontrast", "kirpilmis%", "tekrar")))
    print(f"  {'':<15}{'':>5}" + "".join(f"{'medyan   p05':>17}" for _ in range(4)))
    print("  " + "-" * 88)

    ozet = {}
    for ad in args.yatak:
        try:
            r = yatak_tara(ad, args.adim, args.n_maks)
        except Exception as e:
            print(f"  {ad:<15} ATLANDI ({type(e).__name__}: {e})")
            continue
        if not r:
            continue
        ozet[ad] = r
        A = {k: np.array([x[k] for x in r], float) for k in r[0]}
        s = f"  {ad:<15}{len(r):>5}"
        for k in ("laplas_var", "kontrast", "kirpilmis", "tekrar"):
            v = A[k][np.isfinite(A[k])]
            if len(v) == 0:
                s += f"{'-':>17}"
                continue
            olcek = 100 if k == "kirpilmis" else 1
            s += f"{np.median(v)*olcek:>9.1f}{np.percentile(v, 5)*olcek:>8.1f}"
        print(s, flush=True)

    # --- kac kare "dusuk kaliteli" sayilir? Esikler yataklarin KENDI dagilimindan
    #     turetilir: medyanin dortte birinden dusuk laplas = belirgin bulaniklik.
    print(f"\n{'=' * 96}\nDUSUK KALITELI KARE SAYIMI\n{'=' * 96}")
    print(f"\n  {'yatak':<15}{'bulanik':>12}{'dusuk kontrast':>17}"
          f"{'kirpilmis>%1':>15}{'donmus':>10}")
    print("  " + "-" * 69)
    for ad, r in ozet.items():
        A = {k: np.array([x[k] for x in r], float) for k in r[0]}
        med_lap = np.median(A["laplas_var"])
        med_kon = np.median(A["kontrast"])
        tek = A["tekrar"][np.isfinite(A["tekrar"])]
        bulanik = int((A["laplas_var"] < 0.25 * med_lap).sum())
        dusuk_kon = int((A["kontrast"] < 0.5 * med_kon).sum())
        kirp = int((A["kirpilmis"] > 0.01).sum())
        donmus = int((tek < 0.5).sum()) if len(tek) else 0
        n = len(r)
        f = lambda x: f"{x} (%{100*x/n:.1f})"
        print(f"  {ad:<15}{f(bulanik):>12}{f(dusuk_kon):>17}"
              f"{f(kirp):>15}{f(donmus):>10}")

    print("\n  YORUM ANAHTARI")
    print("   Butun sutunlar ~0 ise: yataklarda dusuk kaliteli kare YOK ->")
    print("   kalite kapisi bu veride OLU olur, kazanc olculemez (yalniz sigorta degeri).")
    print("   Anlamli sayida varsa: kapi ACIK, A/B ile olculur.")


if __name__ == "__main__":
    main()
