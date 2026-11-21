# -*- coding: utf-8 -*-
"""GOREV 2 — KURTARMA SECICISI: esik duyarliligi + tetiklenme orani + kapi

NEDEN VAR
---------
`tavan_ensemble.py` sunu buldu: iki on uc (RAFT yogun akis, SIFT seyrek ozellik)
arasinda HER KAREDE secim yapmak sinav penceresi 1'i +3,88 m bozuyor (inlier
karsilastiricisi). Kahin bile orada kazandirmiyor (11,27 vs 11,29) — yani o pencerede
secilecek bir sey yok.

Ama "KURTARMA" varyanti farkli bir mekanizma: her karede secim yapmaz, SIFT'e YALNIZ
RAFT acikca coktugunde gecer (RAFT merdiven inlier'i tabanin altinda VE SIFT belirgin
ustun). Ilk olcumde sinavin IKI PENCERESI DE iyilesti ve felaket cikmadi.

Bu betik o sonucun gercek mi yoksa esik/permutasyon gurultusu mu oldugunu sinar:
  1. esik taramasi  — sonuc genis bir aralikta ayakta mi?
  2. tetiklenme     — kurtarma kac karede devreye giriyor? (calisma maliyeti bundan)
  3. kapi           — genelleme sinavi iki pencere + 7 yatak, eslestirilmis

CALISTIR
--------
    .venv\\Scripts\\python gorev2\\kurtarma_sinavi.py --n-perm 4
    .venv\\Scripts\\python gorev2\\kurtarma_sinavi.py --tara
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

GOREV2 = Path(__file__).resolve().parent
sys.path.insert(0, str(GOREV2))

from kabul import PENCERELER, yarisma_takvimi                       # noqa: E402
from nokta_onbellek import YATAKLAR                                 # noqa: E402
from gorev2_src.evaluation import competition_error                 # noqa: E402
from gorev2_src.vo_pipeline import EstimatorConfig                  # noqa: E402
from tavan_ensemble import _satirlar, kos, TUM                      # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")


def yatak_olc(ad, cfg, spacing, karistir, esik):
    motorlar, gt, prof, n = {}, None, None, None
    for m in ("raft", "sift"):
        g, cr, lr, p, nn = _satirlar(ad, m, spacing, karistir)
        motorlar[m] = (cr, lr)
        if gt is None or nn < n:
            gt, prof, n = g, p, nn
    gt = gt[:n]

    out = {}
    if ad == "yarisma":
        gercek = np.load(YATAKLAR["yarisma"]["gt_npz"])["saglikli"]
        for w, pw in PENCERELER.items():
            son = pw["son"]
            h = gercek[:son].copy()
            a, b = pw["puan"]
            h[a:b + 1] = False
            mot = {k: (c[:son - 1], l[:son - 1]) for k, (c, l) in motorlar.items()}
            for mod in ("raft", "kurtarma"):
                pr, sec = kos(gt[:son], mot, h, prof, cfg, spacing, mod, esik)
                out[(f"pencere{w}", mod)] = float(
                    np.linalg.norm(pr[a:b + 1] - gt[a:b + 1], axis=1).mean())
                if mod == "kurtarma":
                    t = sum(sec.values())
                    out[(f"pencere{w}", "tetik")] = sec["sift"] / max(t, 1)
        return out

    h = yarisma_takvimi(n)
    for mod in ("raft", "kurtarma"):
        pr, sec = kos(gt, motorlar, h, prof, cfg, spacing, mod, esik)
        out[(ad, mod)] = competition_error(pr, gt)
        if mod == "kurtarma":
            out[(ad, "tetik")] = sec["sift"] / max(sum(sec.values()), 1)
    return out


def topla(yataklar, cfg, spacing, n_perm, esik):
    ham: dict = {}
    for i in range(n_perm):
        for ad in yataklar:
            try:
                for k, v in yatak_olc(ad, cfg, spacing, i, esik).items():
                    ham.setdefault(k, []).append(v)
            except Exception as e:
                print(f"    {ad}: ATLANDI ({type(e).__name__}: {e})")
    return ham


def _etiketler(ham):
    out = []
    for k in ham:
        if k[0] not in out:
            out.append(k[0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yatak", nargs="*", default=TUM)
    ap.add_argument("--n-perm", type=int, default=4)
    ap.add_argument("--spacing", type=int, default=8)
    ap.add_argument("--taban", type=float, default=0.20)
    ap.add_argument("--ustunluk", type=float, default=1.50)
    ap.add_argument("--tara", action="store_true")
    args = ap.parse_args()

    cfg = EstimatorConfig(method="homografi")

    if args.tara:
        print("=" * 74)
        print(f"KURTARMA ESIK TARAMASI — {args.n_perm} permutasyon")
        print("=" * 74)
        print(f"\n  {'taban':>7}{'ustunluk':>10}{'pencere1':>11}{'pencere2':>11}"
              f"{'7 yatak':>10}{'tetik %':>10}")
        print("  " + "-" * 59)
        kombin = [(t, 1.5) for t in (0.10, 0.15, 0.20, 0.30, 0.40)]
        kombin += [(0.20, u) for u in (1.0, 2.0, 3.0)]
        temel = None
        for t, u in kombin:
            ham = topla(args.yatak, cfg, args.spacing, args.n_perm, (t, u))
            et = _etiketler(ham)
            g = lambda e, m: float(np.mean(ham[(e, m)])) if (e, m) in ham else np.nan
            yat = [e for e in et if not e.startswith("pencere")]
            if temel is None:
                temel = (g("pencere1", "raft"), g("pencere2", "raft"),
                         float(np.mean([g(e, "raft") for e in yat])))
                print(f"  {'MEVCUT':>7}{'-':>10}{temel[0]:>11.2f}{temel[1]:>11.2f}"
                      f"{temel[2]:>10.2f}{'-':>10}")
            print(f"  {t:>7.2f}{u:>10.1f}{g('pencere1','kurtarma'):>11.2f}"
                  f"{g('pencere2','kurtarma'):>11.2f}"
                  f"{np.mean([g(e,'kurtarma') for e in yat]):>10.2f}"
                  f"{100*np.mean([g(e,'tetik') for e in et]):>9.1f}%", flush=True)
        print("\n  Sonuc genis bir esik araliginda ayakta kaliyorsa deger uydurulmamistir.")
        return

    esik = (args.taban, args.ustunluk)
    print("=" * 78)
    print(f"KURTARMA SECICISI — KAPI  (taban {esik[0]}, ustunluk {esik[1]}x, "
          f"{args.n_perm} permutasyon)")
    print("=" * 78)
    ham = topla(args.yatak, cfg, args.spacing, args.n_perm, esik)
    et = _etiketler(ham)
    print(f"\n  {'yatak':<15}{'MEVCUT':>14}{'KURTARMA':>14}{'fark':>14}"
          f"{'tetik %':>9}")
    print("  " + "-" * 66)
    felaket, yat_m, yat_k = [], [], []
    for e in et:
        a = np.array(ham[(e, "raft")]); b = np.array(ham[(e, "kurtarma")])
        d = b - a
        std = d.std(ddof=1) if len(d) > 1 else 0.0
        etiket = ("KAZANDI" if d.mean() < -0.05 else
                  "kaybetti" if d.mean() > 0.05 else "esit")
        if d.mean() > 3.0:
            felaket.append(e)
        if not e.startswith("pencere"):
            yat_m.append(a.mean()); yat_k.append(b.mean())
        print(f"  {e:<15}{a.mean():>8.2f}±{a.std(ddof=1) if len(a)>1 else 0:<5.2f}"
              f"{b.mean():>8.2f}±{b.std(ddof=1) if len(b)>1 else 0:<5.2f}"
              f"{d.mean():>+8.2f}±{std:<5.2f}{100*np.mean(ham[(e,'tetik')]):>8.1f}%"
              f"  {etiket}")
    print("  " + "-" * 66)
    print(f"  {'7 yatak ort.':<15}{np.mean(yat_m):>14.2f}{np.mean(yat_k):>14.2f}"
          f"{np.mean(yat_k)-np.mean(yat_m):>+14.2f}")
    print(f"\n  KABUL: sinav IKI pencerede de kotulesmeyecek · yatak ortalamasi "
          f"kotulesmeyecek · felaket (>+3 m) yok")
    print(f"  felaket yatak: {', '.join(felaket) if felaket else 'YOK'}")


if __name__ == "__main__":
    main()
