# -*- coding: utf-8 -*-
"""GOREV 3 — min_confidence A/B: 0,25 vs 0,10, GORSEL kanitla.

NEDEN: test yatagi sayilari 0,10'un daha cok tespit urettigini soyluyor
(recall %81,9 -> %85,7) ama ayni zamanda nesnenin OLMADIGI karelerde de daha cok
tespit uretiyor (6/14 -> 11/14). Sayi tek basina karar veremez: ek tespitler
DOGRU nesnenin uzerinde mi, yoksa sistem her kareye bir kutu mu atiyor?

Bu betik iki ayari AYNI karelerde kosturur, YALNIZ AYRISTIKLARI kareler icin
yan yana panel uretir ve gozle bakilir. (Gorev 1'de trial-8 karari da boyle
verilmisti; iki otomatik kirmizi bayrak orada yanlis alarm cikmisti.)

Calistir:
    .venv\\Scripts\\python gorev3\\mc_ab.py
Cikti: gorev3/degerlendirme/mc_ab_*.jpg + ekrana ozet
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

GOREV3 = Path(__file__).resolve().parent
ROOT = GOREV3.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

import test_yatagi as ty  # noqa: E402  (ayni yatak, ayni ornekleme)

OUT = GOREV3 / "degerlendirme"
ORNEK = 15
DISARI = 2


def kos(m, refs):
    """order -> {kare_no: det|None}  (aktif aralik + aralik disi kontrol kareleri)."""
    sonuc = {}
    for r in refs:
        aralik = list(range(r["bas"], r["son"] + 1))
        if len(aralik) > ORNEK:
            idx = np.linspace(0, len(aralik) - 1, ORNEK).astype(int)
            aralik = [aralik[i] for i in idx]
        dis = [k for k in (r["bas"] - 200, r["bas"] - 120,
                           r["son"] + 120, r["son"] + 200)][:DISARI]
        d = {}
        for no in aralik + dis:
            yol = ty.kare_yolu(no)
            if not yol.exists():
                continue
            img = cv2.imread(str(yol))
            if img is None:
                continue
            dets = m.match(img, frame_id=no + 1, active_ids={r["order"]})
            d[no] = (dets[0] if dets else None, no in dis)
        sonuc[r["order"]] = d
    return sonuc


def _kutu(t):
    if t is None:
        return None
    g = t if isinstance(t, dict) else t.__dict__
    return (int(g["top_left_x"]), int(g["top_left_y"]),
            int(g["bottom_right_x"]), int(g["bottom_right_y"]))


def panel(no, ref_img, kare, a, b, ad):
    """Sol: referans · Orta: 0,25 · Sag: 0,10."""
    h = 420
    def ol(im):
        s = h / im.shape[0]
        return cv2.resize(im, (int(im.shape[1] * s), h))
    ref = ol(ref_img)
    parcalar = [ref]
    for kutu, etiket, renk in ((a, "min_conf 0.25", (60, 180, 255)),
                               (b, "min_conf 0.10", (60, 220, 60))):
        g = kare.copy()
        if kutu:
            cv2.rectangle(g, kutu[:2], kutu[2:], renk, 6)
        else:
            cv2.putText(g, "TESPIT YOK", (40, 90), cv2.FONT_HERSHEY_SIMPLEX,
                        2.4, (0, 0, 255), 6)
        g = ol(g)
        cv2.putText(g, etiket, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, renk, 2)
        parcalar.append(g)
    birlesik = np.hstack(parcalar)
    cv2.putText(birlesik, ad, (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)
    p = OUT / f"mc_ab_{ad.replace(' ', '_').replace('·', '-')}.jpg"
    cv2.imwrite(str(p), birlesik, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return p


def main() -> int:
    OUT.mkdir(exist_ok=True)
    print("A) min_confidence = 0.25")
    ma, refs, _ = ty.matcher_kur(ayar="matching.min_confidence=0.25")
    ra = kos(ma, refs)
    del ma
    print("B) min_confidence = 0.10")
    mb, refs, _ = ty.matcher_kur(ayar="matching.min_confidence=0.10")
    rb = kos(mb, refs)

    ref_gorsel = {r["order"]: cv2.imread(str(r["img"])) for r in refs}
    fark = []
    say = {"a_var_b_yok": 0, "b_var_a_yok": 0, "ikisi_de": 0, "hicbiri": 0}
    for r in refs:
        o = r["order"]
        for no in sorted(ra[o]):
            ka, dis = ra[o][no]
            kb, _ = rb[o].get(no, (None, dis))
            ba, bb = _kutu(ka), _kutu(kb)
            if ba and bb:
                say["ikisi_de"] += 1
            elif ba and not bb:
                say["a_var_b_yok"] += 1
                fark.append((o, no, dis, ba, bb))
            elif bb and not ba:
                say["b_var_a_yok"] += 1
                fark.append((o, no, dis, ba, bb))
            else:
                say["hicbiri"] += 1

    print("\n" + "=" * 66)
    print("min_confidence A/B — 0,25 (A) vs 0,10 (B)")
    print("=" * 66)
    for k, v in say.items():
        print(f"  {k:<14} {v}")
    print(f"\nAYRISAN KARE: {len(fark)}  (panel uretiliyor)")
    for o, no, dis, ba, bb in fark:
        img = cv2.imread(str(ty.kare_yolu(no)))
        ad = (f"ref{o} kare{no} "
              + ("ARALIK-DISI" if dis else "aktif")
              + (" · sadece-B" if bb and not ba else " · sadece-A"))
        p = panel(no, ref_gorsel[o], img, ba, bb, ad)
        print(f"  {ad:<44} -> {p.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
