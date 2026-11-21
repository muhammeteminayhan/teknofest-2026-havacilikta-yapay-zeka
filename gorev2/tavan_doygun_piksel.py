# -*- coding: utf-8 -*-
"""GOREV 2 — TAVAN OLCUMU: DOYGUN (kirpilmis) piksellerdeki noktalar zararli mi?

NEDEN VAR
---------
`tavan_kare_kalitesi.py` sunu buldu: bulanik / dusuk kontrastli / donmus kare
yataklarda pratikte YOK (%0), ama **termal yataklarin karelerinin %66-78'inde**
piksellerin %1'inden fazlasi 0 ya da 255'e yapismis. Bu bir ariza degil, 8-bit AGC'li
termal goruntunun dogasidir (sicak noktalar doyar).

Bu yuzden raporun onerdigi KARE DUZEYINDE kalite kapisi burada yanlis aractir: termal
karelerin ucte ikisini atmak sacma olur. Anlamli olabilecek sey PIKSEL DUZEYINDE bir
maskedir — ornekleme izgarasindaki noktalardan doygun bolgeye dusenleri atmak.

Bu betigin cevapladigi soru: doygun bolgeye dusen noktalar, RANSAC tarafindan
DIGERLERINDEN DAHA SIK aykiri sayiliyor mu?
  * Belirgin yuksekse -> piksel maskesi ACIK, A/B'ye deger
  * Farksizsa         -> akis o bolgelerde de saglikli, is KAPANIR

Not: termal, kuyruk riskimizin yasadigi yer (video2'de hatanin %25'i en kotu %5
kareden, en kotu kare 85 m) — bu yuzden ucuz bir olcum hak ediyor.

CALISTIR
--------
    .venv\\Scripts\\python gorev2\\tavan_doygun_piksel.py
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

from nokta_onbellek import YATAKLAR, onbellek_yukle, yatak_npz, yatak_kareler  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yatak", nargs="*",
                    default=["video2", "v25_termal_a", "v25_termal_b"])
    ap.add_argument("--n-kare", type=int, default=150)
    ap.add_argument("--esik", type=float, default=1.0)
    ap.add_argument("--genislet", type=int, default=3,
                    help="doygun bolgeyi bu yaricapla genislet [px]")
    args = ap.parse_args()

    print("=" * 92)
    print(f"DOYGUN PIKSEL TAVANI — {args.n_kare} kare/yatak, "
          f"doygun bolge {args.genislet} px genisletildi")
    print("=" * 92)
    print(f"\n  {'yatak':<15}{'katman':<11}{'doygun alan %':>15}"
          f"{'aykiri: DOYGUN':>17}{'aykiri: temiz':>16}{'oran':>8}")
    print("  " + "-" * 82)

    cek = np.ones((2 * args.genislet + 1,) * 2, np.uint8)
    for ad in args.yatak:
        obs = {"ardisik": onbellek_yukle(yatak_npz(ad, None, "raft")),
               "merdiven": onbellek_yukle(yatak_npz(ad, 8, "raft"))}
        n_cift = len(obs["ardisik"]["offsets"]) - 1
        hedef = set(np.linspace(1, n_cift - 1, args.n_kare).astype(int).tolist())

        top = {k: dict(d_ayk=0, d_top=0, t_ayk=0, t_top=0, alan=[])
               for k in obs}
        i = 0
        for _, gray in yatak_kareler(YATAKLAR[ad], progress=False):
            if i in hedef:
                g = gray if gray.ndim == 2 else cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
                doygun = ((g == 0) | (g == 255)).astype(np.uint8)
                if args.genislet:
                    doygun = cv2.dilate(doygun, cek)
                doygun = doygun.astype(bool)
                for katman, ob in obs.items():
                    off = ob["offsets"]
                    if i >= len(off) - 1:
                        continue
                    a, b = int(off[i - 1]), int(off[i])
                    if b - a < 20:
                        continue
                    p0 = ob["pts0"][a:b].astype(np.float64)
                    p1 = ob["pts1"][a:b].astype(np.float64)
                    H, mask = cv2.findHomography(p0, p1, cv2.RANSAC, args.esik)
                    if H is None or mask is None:
                        continue
                    inl = mask.reshape(-1).astype(bool)
                    xi = np.clip(p0[:, 0].astype(int), 0, g.shape[1] - 1)
                    yi = np.clip(p0[:, 1].astype(int), 0, g.shape[0] - 1)
                    icinde = doygun[yi, xi]
                    t = top[katman]
                    t["alan"].append(float(doygun.mean()))
                    t["d_top"] += int(icinde.sum())
                    t["d_ayk"] += int((~inl & icinde).sum())
                    t["t_top"] += int((~icinde).sum())
                    t["t_ayk"] += int((~inl & ~icinde).sum())
            i += 1
            if i > max(hedef):
                break

        for katman in ("ardisik", "merdiven"):
            t = top[katman]
            if t["d_top"] < 50 or t["t_top"] < 50:
                print(f"  {ad:<15}{katman:<11} yetersiz ornek "
                      f"(doygun nokta {t['d_top']})")
                continue
            pd = 100 * t["d_ayk"] / t["d_top"]
            pt = 100 * t["t_ayk"] / t["t_top"]
            print(f"  {ad:<15}{katman:<11}{100*np.mean(t['alan']):>15.2f}"
                  f"{pd:>17.1f}{pt:>16.1f}{pd/max(pt, 1e-9):>8.2f}", flush=True)

    print("\n  YORUM ANAHTARI")
    print("   oran >> 1 (ornegin >1,5) ise doygun bolgedeki akis gercekten guvenilmez")
    print("     -> piksel maskesi ACIK")
    print("   oran ~1 ise RANSAC zaten ayirt etmiyor demektir; ayri maske bir sey")
    print("     kazandirmaz -> is KAPANIR")


if __name__ == "__main__":
    main()
