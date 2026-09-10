# -*- coding: utf-8 -*-
"""GOREV 2 — TAVAN OLCUMU: RANSAC aykirilari HAREKETLI ARAC uzerinde mi?

NEDEN VAR
---------
Manus raporu (§6.3, kaynak [13]) baskin homografiye uymayan akis kumelerinin dinamik
nesne sayilip ayrica agirliklandirilmasini oneriyor. Bizde bunu yalniz RANSAC yapiyor
(ayrik kabul/ret); acik bir dinamik-nesne modeli yok.

Ama bizim elimizde literaturdeki calismalarin cogunda olmayan bir sey var: **Gorev 1
dedektoru hareketli/duran ayrimini zaten uretiyor** (`motion_status`, ego-hareket
telafili). Yani "hangi aykirilar gercekten hareketli arac uzerinde" sorusunu tahmin
etmek yerine OLCEBILIRIZ.

Bu betigin cevapladigi soru: RANSAC'in attigi noktalarin kacta kaci hareketli arac
kutularinin icine dusuyor?
  * Oran YUKSEKSE  -> dinamik nesneler gercek bir gurultu kaynagi, is ACIK
  * Oran DUSUKSE   -> aykirilar bina/agac paralaksindan geliyor (3B yapi), dinamik
                      nesne modeli bize bir sey kazandirmaz, is KAPANIR

Merdiven katmaninda inlier oraninin 0,369 olmasi ikinci ihtimali dusundururuyor:
hareketli araclar kadrajin kucuk bir bolumunu kaplar, %63 aykiri uretemez.

CALISTIR
--------
    .venv\\Scripts\\python gorev2\\tavan_dinamik_nokta.py
    .venv\\Scripts\\python gorev2\\tavan_dinamik_nokta.py --n-kare 300
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np

GOREV2 = Path(__file__).resolve().parent
KOK = GOREV2.parent
sys.path.insert(0, str(GOREV2))
sys.path.insert(0, str(KOK))

from nokta_onbellek import YATAKLAR, onbellek_yukle, yatak_npz    # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")


def kutu_maskesi(nesneler, sekil, yalniz_hareketli=True, pay=0.0):
    """Tespit kutularindan bool maske uret (True = kutu ici)."""
    m = np.zeros(sekil[:2], bool)
    for o in nesneler:
        if yalniz_hareketli and str(o.get("motion_status", "0")) != "1":
            continue
        x0, y0 = float(o["top_left_x"]), float(o["top_left_y"])
        x1, y1 = float(o["bottom_right_x"]), float(o["bottom_right_y"])
        dw, dh = (x1 - x0) * pay, (y1 - y0) * pay
        a = max(int(y0 - dh), 0); b = min(int(y1 + dh), sekil[0])
        c = max(int(x0 - dw), 0); d = min(int(x1 + dw), sekil[1])
        if b > a and d > c:
            m[a:b, c:d] = True
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yatak", default="yarisma")
    ap.add_argument("--n-kare", type=int, default=200)
    ap.add_argument("--esik", type=float, default=1.0)
    ap.add_argument("--pay", type=float, default=0.15, help="kutuyu bu oranda genislet")
    args = ap.parse_args()

    from gorev1 import Gorev1Detector

    y = YATAKLAR[args.yatak]
    kareler_dizin = y.get("kareler")
    if kareler_dizin is None:
        raise SystemExit("Bu olcum kare dizini olan yatakla yapilir (yarisma).")
    yollar = sorted(Path(kareler_dizin).glob("frame_*.webp"))

    ob_c = onbellek_yukle(yatak_npz(args.yatak, None, "raft"))
    ob_l = onbellek_yukle(yatak_npz(args.yatak, 8, "raft"))

    idx = np.linspace(1, min(len(yollar), len(ob_c["offsets"]) - 1) - 1,
                      args.n_kare).astype(int)
    det = Gorev1Detector()

    print("=" * 78)
    print(f"DINAMIK NOKTA TAVANI — yatak {args.yatak}, {len(idx)} kare, "
          f"kutu payi %{100*args.pay:.0f}")
    print("=" * 78)

    sonuc = {"ardisik": [], "merdiven": []}
    n_hareketli, n_tasit = [], []
    for j, k in enumerate(idx):
        img = cv2.imread(str(yollar[k]))
        if img is None:
            continue
        nesneler = det.detect(img)
        hareketli = [o for o in nesneler if str(o.get("motion_status", "0")) == "1"]
        n_hareketli.append(len(hareketli))
        n_tasit.append(len(nesneler))
        mask_h = kutu_maskesi(nesneler, img.shape, True, args.pay)
        kutu_alan_orani = float(mask_h.mean())

        for ad, ob in (("ardisik", ob_c), ("merdiven", ob_l)):
            off = ob["offsets"]
            if k >= len(off) - 1:
                continue
            a, b = int(off[k - 1]), int(off[k])
            if b - a < 20:
                continue
            p0 = ob["pts0"][a:b].astype(np.float64)
            p1 = ob["pts1"][a:b].astype(np.float64)
            H, mask = cv2.findHomography(p0, p1, cv2.RANSAC, args.esik)
            if H is None or mask is None:
                continue
            inl = mask.reshape(-1).astype(bool)
            xi = np.clip(p0[:, 0].astype(int), 0, img.shape[1] - 1)
            yi = np.clip(p0[:, 1].astype(int), 0, img.shape[0] - 1)
            icinde = mask_h[yi, xi]
            n_ayk = int((~inl).sum())
            if n_ayk == 0:
                continue
            sonuc[ad].append(dict(
                aykiri_orani=float((~inl).mean()),
                ayk_kutuda=float(icinde[~inl].mean()),      # aykirilarin kaci kutuda
                inl_kutuda=float(icinde[inl].mean()) if inl.sum() else 0.0,
                kutu_alan=kutu_alan_orani,
            ))
        if j and j % 50 == 0:
            print(f"  {j}/{len(idx)} kare", flush=True)

    print(f"\n  Kare basina tespit: toplam medyan {np.median(n_tasit):.0f}, "
          f"HAREKETLI medyan {np.median(n_hareketli):.0f} "
          f"(hareketli tespit iceren kare orani %{100*np.mean(np.array(n_hareketli)>0):.0f})")

    print(f"\n  {'katman':<12}{'aykiri %':>10}{'kutu alani %':>14}"
          f"{'AYKIRI kutuda %':>18}{'inlier kutuda %':>18}")
    print("  " + "-" * 72)
    for ad in ("ardisik", "merdiven"):
        r = sonuc[ad]
        if not r:
            print(f"  {ad:<12} olcum yok")
            continue
        g = lambda k: 100 * float(np.mean([x[k] for x in r]))
        print(f"  {ad:<12}{g('aykiri_orani'):>10.1f}{g('kutu_alan'):>14.2f}"
              f"{g('ayk_kutuda'):>18.2f}{g('inl_kutuda'):>18.2f}")

    print("\n  YORUM ANAHTARI")
    print("   'AYKIRI kutuda' >> 'inlier kutuda' ve mutlak deger anlamli (>%10) ise:")
    print("     hareketli araclar gercek bir aykiri kaynagi -> dinamik nesne modeli ACIK")
    print("   Ikisi de kucuk ve birbirine yakinsa: aykirilar 3B YAPIDAN (bina/agac)")
    print("     geliyor, hareketli nesneden degil -> is KAPANIR")


if __name__ == "__main__":
    main()
