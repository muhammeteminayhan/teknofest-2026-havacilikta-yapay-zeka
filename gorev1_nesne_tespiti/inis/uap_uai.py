"""
UAP/UAİ iniş alanı — daire geometrisi + iniş-uygunluk (doluluk) kontrolü.

Şartname (s.13-15):
- UAP (Uçan Araba Park) ve UAİ (Uçan Ambulans İniş) 4.5 m çapında dairelerle gösterilir.
- İniş durumu "uygun" olması için: alanın TAMAMI karede olmalı VE üzerinde/yanında
  hiçbir nesne (taşıt/insan veya tespit edilmeyen herhangi bir cisim) olmamalı.
- Daire kısmen karede ise → tespit et ama iniş "uygun değil".
- Yakındaki cisimler de (çekim açısı yanıltması) "uygun değil" saydırır.

Yaklaşım (baseline):
- Alanı YOLO tespiti kabaca verir (UAP/UAİ sınıfı). Daire geometrisini Hough ile
  hassaslaştır (yarıçap/merkez). Doluluk: tespit edilen taşıt/insan kutularının
  merkezleri daire (marjlı yarıçap) içine düşüyor mu?
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


def daire_tespit(roi_gray: np.ndarray, min_r: int = 8,
                 max_r: int = 0) -> Optional[Tuple[float, float, float]]:
    """
    ROI içinde Hough ile en belirgin daireyi bul. Döndürür (cx, cy, r) ROI-yerel
    koordinatlarda; bulunamazsa None.
    """
    import cv2
    g = roi_gray if roi_gray.ndim == 2 else cv2.cvtColor(roi_gray, cv2.COLOR_BGR2GRAY)
    g = cv2.medianBlur(g, 3)
    h, w = g.shape[:2]
    if max_r <= 0:
        max_r = max(h, w) // 2 + 1
    circles = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1.2, minDist=max(h, w),
                               param1=100, param2=30, minRadius=min_r, maxRadius=max_r)
    if circles is None:
        return None
    c = circles[0, 0]
    return float(c[0]), float(c[1]), float(c[2])


def _kutu_merkezi(box) -> np.ndarray:
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)


def inis_durumu_belirle(alan_box: Tuple[float, float, float, float],
                        engel_kutulari: List[Tuple[float, float, float, float]],
                        kare_w: int, kare_h: int,
                        doluluk_marji: float = 1.15) -> int:
    """
    Bir UAP/UAİ alanı için iniş durumunu belirler.

    alan_box       : alanın bbox'u (x1,y1,x2,y2) tam karede.
    engel_kutulari : bu karedeki diğer tespitlerin (taşıt/insan) bbox listesi.
    kare_w, kare_h : kare boyutu (alanın tamamı karede mi kontrolü için).
    Döndürür: 1 (Uygun), 0 (Uygun Değil). (İniş alanı olmayan sınıflar için -1
              çağıran tarafça atanır.)
    """
    x1, y1, x2, y2 = alan_box

    # 1) Alanın tamamı karede mi? Kenara değiyorsa kısmi → uygun değil.
    kenar = 2.0
    if x1 <= kenar or y1 <= kenar or x2 >= kare_w - kenar or y2 >= kare_h - kenar:
        return 0

    # 2) Daire geometrisi (bbox'tan) — merkez ve yarıçap.
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    r = 0.5 * min(x2 - x1, y2 - y1) * doluluk_marji

    # 3) Doluluk: herhangi bir engel merkezi marjlı daire içine düşüyor mu?
    for box in engel_kutulari:
        if box is alan_box:
            continue
        m = _kutu_merkezi(box)
        if np.hypot(m[0] - cx, m[1] - cy) <= r:
            return 0   # üstünde/yakınında cisim var → uygun değil

    return 1
