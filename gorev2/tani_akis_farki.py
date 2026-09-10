# -*- coding: utf-8 -*-
"""GOREV 2 — TANI: iki akis motoru arasindaki fark SISTEMATIK mi?

NEDEN VAR
---------
SEA-RAFT uc varyantta da RAFT'i geciyor degil (video1: 3,90 -> 4,27...4,60). Neden?

Depoda ayni soru XFeat icin sorulmus ve kok neden bulunmustu: **hareketi sistematik
olarak az olcuyordu** (medyan akis RAFT +25,21 iken XFeat +16,71) — yani sorun
gurultu degil YANLILIKTI, ve yanlilik dogrudan olcek hatasina donusur.

Bu betik ayni teshisi SEA-RAFT icin yapar. Iki onbellek AYNI kare ciftleri ve AYNI
izgara uzerinde uretildigi icin noktalar birebir eslesir; farki dogrudan olcebiliriz.

OLCUTLER
  medyan akis        : her motorun kare basi medyan |akis| — sistematik olcek farki
  olcek orani        : medyan(|akis_B|) / medyan(|akis_A|), kare basina
  aci farki          : akis yonlerinin medyan acisal farki [derece]
  nokta sayisi       : ileri-geri elemesinden kac nokta gecti

CALISTIR
--------
    .venv\\Scripts\\python gorev2\\tani_akis_farki.py --a raft --b searaft_g
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

GOREV2 = Path(__file__).resolve().parent
sys.path.insert(0, str(GOREV2))

from nokta_onbellek import onbellek_yukle, yatak_npz     # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")


def _akislar(ob, i):
    a, b = int(ob["offsets"][i]), int(ob["offsets"][i + 1])
    if b - a < 10:
        return None
    p0 = ob["pts0"][a:b].astype(np.float64)
    p1 = ob["pts1"][a:b].astype(np.float64)
    return p0, p1 - p0


def _homografi_kalitesi(ob, n, esik=1.0, n_ornek=400):
    """Uretimdekiyle AYNI cagri: inlier orani ve inlier'larin medyan artigi [px].

    Artik, akisin duzlem modeline ne kadar temiz oturdugunu olcer — yani
    YANLILIK degil GURULTU gostergesidir.
    """
    import cv2
    idx = np.linspace(0, n - 1, min(n_ornek, n)).astype(int)
    inl, art = [], []
    for i in idx:
        r = _akislar(ob, i)
        if not r:
            continue
        p0, d = r
        p1 = p0 + d
        H, mask = cv2.findHomography(p0, p1, cv2.RANSAC, esik)
        if H is None or mask is None:
            continue
        m = mask.reshape(-1).astype(bool)
        if m.sum() < 10:
            continue
        inl.append(float(m.mean()))
        ph = np.column_stack([p0[m], np.ones(int(m.sum()))]) @ H.T
        ph = ph[:, :2] / np.maximum(ph[:, 2:3], 1e-12)
        art.append(float(np.median(np.linalg.norm(ph - p1[m], axis=1))))
    return (float(np.median(inl)) if inl else np.nan,
            float(np.median(art)) if art else np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yatak", default="video1")
    ap.add_argument("--a", default="raft")
    ap.add_argument("--b", nargs="*", default=["searaft", "searaft_l", "searaft_g"])
    ap.add_argument("--katman", default="cons", choices=["cons", "sp8"])
    args = ap.parse_args()

    sp = None if args.katman == "cons" else 8
    obA = onbellek_yukle(yatak_npz(args.yatak, sp, args.a))

    print("=" * 84)
    print(f"AKIS FARKI TANISI — yatak {args.yatak}, katman {args.katman}, "
          f"referans {args.a}")
    print("=" * 84)
    print(f"\n  {'motor':<14}{'medyan |akis| px':>18}{'olcek orani':>14}"
          f"{'aci farki':>12}{'nokta/cift':>13}{'inlier':>10}{'artik px':>11}")
    print("  " + "-" * 92)

    nA = len(obA["offsets"]) - 1
    magA_all = []
    for i in range(nA):
        r = _akislar(obA, i)
        if r:
            magA_all.append(float(np.median(np.linalg.norm(r[1], axis=1))))
    nptA = np.mean(np.diff(obA["offsets"]))
    iA, rA = _homografi_kalitesi(obA, nA)
    print(f"  {args.a:<14}{np.median(magA_all):>18.3f}{'1.000':>14}"
          f"{'0.00':>12}{nptA:>13.0f}{iA:>10.3f}{rA:>11.3f}")

    for m in args.b:
        yol = yatak_npz(args.yatak, sp, m)
        if not yol.exists():
            print(f"  {m:<14}{'onbellek yok':>18}")
            continue
        obB = onbellek_yukle(yol)
        n = min(nA, len(obB["offsets"]) - 1)
        magB, oran, aci = [], [], []
        for i in range(n):
            ra, rb = _akislar(obA, i), _akislar(obB, i)
            if not ra or not rb:
                continue
            ma = np.median(np.linalg.norm(ra[1], axis=1))
            mb = np.median(np.linalg.norm(rb[1], axis=1))
            magB.append(float(mb))
            if ma > 1e-6:
                oran.append(float(mb / ma))
            # baskin yon: medyan akis vektorunun acisi
            va, vb = np.median(ra[1], axis=0), np.median(rb[1], axis=0)
            if np.linalg.norm(va) > 1e-6 and np.linalg.norm(vb) > 1e-6:
                c = np.clip(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)),
                            -1, 1)
                aci.append(float(np.degrees(np.arccos(c))))
        nptB = np.mean(np.diff(obB["offsets"]))
        iB, rB = _homografi_kalitesi(obB, n)
        print(f"  {m:<14}{np.median(magB):>18.3f}{np.median(oran):>14.4f}"
              f"{np.median(aci):>12.2f}{nptB:>13.0f}{iB:>10.3f}{rB:>11.3f}", flush=True)

    print("\n  YORUM ANAHTARI")
    print("   olcek orani 1,00'den belirgin sapiyorsa: motor hareketi SISTEMATIK olarak")
    print("     az/cok olcuyor -> dogrudan olcek hatasi (XFeat'in kok nedeni buydu)")
    print("   oran ~1,00 ama sonuc yine kotuyse: fark yanlilikta degil GURULTUDE;")
    print("     nokta bazli sacilim daha yuksek demektir")


if __name__ == "__main__":
    main()
