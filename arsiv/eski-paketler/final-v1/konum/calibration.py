"""Kamera kalibrasyon profilleri ve distorsiyon düzeltme.

Değerler yarışma kalibrasyon dosyalarından alınmıştır (MATLAB `cameraIntrinsics`
formatı; PrincipalPoint = [cx, cy], ImageSize = [yükseklik, genişlik]).

Profil, kare boyutundan **otomatik** seçilir — oturumda hangi kameranın geleceği
önceden bilinmediği için bu seçim çalışma anında yapılır.

* **İç parametreler:** `fx, fy` piksel cinsinden odak, `cx, cy` asal nokta; `K` matrisi.
* **Radyal distorsiyon (k1, k2):** homografi gibi geometrik hesaplardan önce
  düzeltilmezse metrik kestirim sistematik hata alır. Termal kamerada k1 = −0,35
  ile bu etki oldukça büyüktür.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraProfile:
    """Tek bir kameranın iç parametreleri."""

    name: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float
    k2: float

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx],
                         [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]])

    @property
    def dist_coeffs(self) -> np.ndarray:
        """OpenCV distorsiyon vektörü [k1, k2, p1, p2] (teğetsel = 0)."""
        return np.array([self.k1, self.k2, 0.0, 0.0])

    def undistort_points(self, points: np.ndarray) -> np.ndarray:
        """Nx2 piksel koordinatını düzelt (P=K verildiği için sonuç yine pikseldir).

        Tüm görüntüyü düzeltmek yerine yalnız takip edilen noktaları düzeltmek
        çok daha ucuzdur; geometri katmanı doğrudan bu koordinatlarla çalışır.
        """
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
        out = cv2.undistortPoints(pts, self.K, self.dist_coeffs, P=self.K)
        return out.reshape(-1, 2)


THERMAL = CameraProfile(
    name="termal", width=640, height=512,
    fx=731.7965, fy=732.0172, cx=319.2367, cy=251.2424,
    k1=-0.3507, k2=0.1137,
)
RGB_1080P = CameraProfile(
    name="rgb_1080p", width=1920, height=1080,
    fx=1389.7, fy=1387.1, cx=954.007, cy=558.896,
    k1=0.1378, k2=-0.2564,
)
RGB_4K = CameraProfile(
    name="rgb_4k", width=4000, height=3000,
    fx=2792.2, fy=2795.2, cx=1988.0, cy=1562.2,
    k1=0.0798, k2=-0.1867,
)

# 4K video 3840x2160 gelirse: kalibrasyon 4000x3000 sensör için verilmiştir.
# Tutarlı tek açıklama 4:3 sensörden 16:9 video — dikeyde 3000→2250 kırpma, sonra
# 4000→3840 ölçekleme (s = 0,96):
#     fx' = fx·s,  cx' = cx·s,  cy' = (cy − (3000−2250)/2)·s
# Tutarlılık kontrolü: özgün cy sensör merkezinden 62,2 px aşağıda; türetilen cy'
# de video merkezinden (1080) 59,7 px aşağıda — aynı fiziksel kayma.
# Distorsiyon katsayıları normalize koordinatta tanımlı olduğu için değişmez.
# 2025 örnek veri setinin 4K videolarında doğrulandı.
_S = 3840.0 / 4000.0
RGB_4K_VIDEO = CameraProfile(
    name="rgb_4k_video", width=3840, height=2160,
    fx=2792.2 * _S, fy=2795.2 * _S,
    cx=1988.0 * _S, cy=(1562.2 - 375.0) * _S,
    k1=0.0798, k2=-0.1867,
)

PROFILES = (THERMAL, RGB_1080P, RGB_4K, RGB_4K_VIDEO)


def profile_for_size(width: int, height: int) -> CameraProfile:
    for p in PROFILES:
        if p.width == width and p.height == height:
            return p
    raise ValueError(
        f"Bilinmeyen goruntu boyutu: {width}x{height}. "
        f"Taninan: {', '.join(f'{q.width}x{q.height}' for q in PROFILES)}"
    )


def profile_for_frame(frame: np.ndarray) -> CameraProfile:
    h, w = frame.shape[:2]
    return profile_for_size(w, h)
