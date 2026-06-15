"""
Modalite tespiti — gelen kare RGB mi termal mi?

Şartname: kareler RGB *veya* termal kameradan gelebilir. Örnek termal video 640x512;
RGB ise Full HD (1920x1080) / 4K (3840x2160). Çözünürlük güçlü bir ipucudur. Ek olarak
termal kareler genelde tek-kanal benzeri (düşük renk doygunluğu) olur.
"""
from __future__ import annotations

from enum import Enum

import numpy as np


class Modalite(str, Enum):
    RGB = "rgb"
    TERMAL = "termal"


def modalite_tespit(img: np.ndarray) -> Modalite:
    """
    Kareden modaliteyi kestir. Önce çözünürlük (640x512 → termal), sonra renk doygunluğu.
    """
    h, w = img.shape[:2]
    # Örnek termal sensör çözünürlüğü 640x512.
    if (w, h) == (640, 512):
        return Modalite.TERMAL
    # Tek kanallı görüntü → termal varsay.
    if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
        return Modalite.TERMAL
    # Renkli ama doygunluk çok düşükse (gri-benzeri) termal olabilir.
    try:
        import cv2
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        if float(hsv[:, :, 1].mean()) < 12.0:
            return Modalite.TERMAL
    except Exception:
        pass
    return Modalite.RGB
