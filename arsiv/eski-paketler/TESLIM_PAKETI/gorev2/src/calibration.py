"""Kamera kalibrasyon profilleri ve distorsiyon düzeltme yardımcıları.

Değerler ``THYZ_2026_Ornek_Veri_Seti/Kamera_Kalibrasyon_Parametreleri_2026.txt``
dosyasından alınmıştır (MATLAB cameraIntrinsics formatı; MATLAB'da
PrincipalPoint = [cx, cy], ImageSize = [yükseklik, genişlik] sırasıyla verilir).

Kavramlar
---------
* **İç parametreler (intrinsics):** Kameranın 3B dünyayı 2B piksele nasıl
  izdüşürdüğünü tanımlar. ``fx, fy`` piksel cinsinden odak uzaklığı,
  ``cx, cy`` görüntü merkezinin (asal nokta) piksel koordinatıdır.
  Hepsi 3x3'lük **K matrisinde** toplanır.
* **Radyal distorsiyon (k1, k2):** Lensin görüntüyü kenarlara doğru bükmesi.
  Homografi gibi geometrik hesaplardan önce düzeltilmezse metrik kestirim
  sistematik hata alır. Termal kamerada k1=-0.35 ile bu etki oldukça büyüktür.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraProfile:
    """Tek bir kameranın iç parametreleri."""

    name: str
    width: int   # görüntü genişliği (piksel)
    height: int  # görüntü yüksekliği (piksel)
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float
    k2: float

    @property
    def K(self) -> np.ndarray:
        """3x3 kamera matrisi."""
        return np.array(
            [[self.fx, 0.0, self.cx],
             [0.0, self.fy, self.cy],
             [0.0, 0.0, 1.0]]
        )

    @property
    def dist_coeffs(self) -> np.ndarray:
        """OpenCV distorsiyon katsayı vektörü [k1, k2, p1, p2].

        Kalibrasyon dosyasında teğetsel distorsiyon [0 0] verildiği için
        p1 = p2 = 0'dır.
        """
        return np.array([self.k1, self.k2, 0.0, 0.0])

    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        """Tüm görüntünün distorsiyonunu düzelt (görselleştirme için)."""
        return cv2.undistort(image, self.K, self.dist_coeffs)

    def undistort_points(self, points: np.ndarray) -> np.ndarray:
        """Nx2 piksel koordinatlarının distorsiyonunu düzelt.

        Geometri hesaplarında tüm görüntüyü düzeltmek yerine yalnızca takip
        edilen noktaları düzeltmek çok daha ucuzdur. ``P=K`` verildiği için
        sonuç yine piksel koordinatında döner.
        """
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
        out = cv2.undistortPoints(pts, self.K, self.dist_coeffs, P=self.K)
        return out.reshape(-1, 2)


# Kalibrasyon dosyasındaki üç kamera profili
THERMAL = CameraProfile(
    name="termal", width=640, height=512,
    fx=731.7965, fy=732.0172, cx=319.2367, cy=251.2424,
    k1=-0.3507, k2=0.1137,
)
RGB_4K = CameraProfile(
    name="rgb_4k", width=4000, height=3000,
    fx=2792.2, fy=2795.2, cx=1988.0, cy=1562.2,
    k1=0.0798, k2=-0.1867,
)
RGB_1080P = CameraProfile(
    name="rgb_1080p", width=1920, height=1080,
    fx=1389.7, fy=1387.1, cx=954.007, cy=558.896,
    k1=0.1378, k2=-0.2564,
)

PROFILES = (THERMAL, RGB_4K, RGB_1080P)


def profile_for_size(width: int, height: int) -> CameraProfile:
    """Görüntü boyutuna göre doğru kamera profilini seç."""
    for profile in PROFILES:
        if profile.width == width and profile.height == height:
            return profile
    raise ValueError(f"Bilinmeyen goruntu boyutu: {width}x{height}")


def profile_for_frame(frame: np.ndarray) -> CameraProfile:
    """Bir video karesine (numpy dizisi) uyan kamera profilini seç."""
    h, w = frame.shape[:2]
    return profile_for_size(w, h)
