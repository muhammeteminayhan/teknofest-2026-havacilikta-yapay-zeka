# -*- coding: utf-8 -*-
"""FTR sekli: gorsel denetimin bir degisikligi elemesi (Gorev 3 min_confidence).

Kutu koordinatlari gorev3/mc_ab.py kosusundan alindi (min_confidence=0,10 ile
uretilen, 0,25'te olmayan tespitler). Burada matcher yeniden kosturulmaz;
yalniz kanit paneli cizilir.
"""
from pathlib import Path

import cv2
import numpy as np

KOK = Path(__file__).resolve().parent.parent
OTURUM = KOK / "sistem" / "_images" / "THYZ_2026_Online_Yarisma"
CIKTI = Path(__file__).resolve().parent / "sekiller"
CIKTI.mkdir(exist_ok=True)

# (referans no, kare no, kutu, referansin ne oldugu, bulunan seyin ne oldugu)
# Iki ornek de uretilmisti; sutuna sigmasi icin TEK ornege indirildi (2 panel,
# tek satir). Ikinci ornek altyazida sozle veriliyor.
ORNEK = [
    (2, 200, (745, 696, 941, 1009), "basketbol sahasi", "yol kenari"),
]

SATIR_Y = 420


def _ol(im, h=SATIR_Y):
    s = h / im.shape[0]
    return cv2.resize(im, (max(1, int(im.shape[1] * s)), h))


satirlar = []
for no, kare, (x1, y1, x2, y2), _ref_ad, _bul_ad in ORNEK:
    ref = cv2.imread(str(OTURUM / "references" / f"reference_{no}.webp"))
    img = cv2.imread(str(OTURUM / f"frame_{kare:06d}.webp"))
    pay = 110
    H, W = img.shape[:2]
    cx1, cy1 = max(0, x1 - pay), max(0, y1 - pay)
    cx2, cy2 = min(W, x2 + pay), min(H, y2 + pay)
    kirp = img[cy1:cy2, cx1:cx2].copy()
    cv2.rectangle(kirp, (x1 - cx1, y1 - cy1), (x2 - cx1, y2 - cy1), (40, 40, 220), 5)
    sol, sag = _ol(ref), _ol(kirp)
    bosluk = np.full((SATIR_Y, 26, 3), 255, np.uint8)
    satirlar.append(np.hstack([sol, bosluk, sag]))

gen = max(s.shape[1] for s in satirlar)
satirlar = [cv2.copyMakeBorder(s, 0, 0, 0, gen - s.shape[1],
                               cv2.BORDER_CONSTANT, value=(255, 255, 255))
            for s in satirlar]
panel = satirlar[0] if len(satirlar) == 1 else np.vstack(
    [satirlar[0], np.full((22, gen, 3), 255, np.uint8), satirlar[1]])

p = CIKTI / "sekil_gorsel_denetim.png"
cv2.imwrite(str(p), panel, [cv2.IMWRITE_PNG_COMPRESSION, 4])
print(f"  {p.name}  {panel.shape[1]}x{panel.shape[0]}")
