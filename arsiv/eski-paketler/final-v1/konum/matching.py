"""Eşleme motoru: özellik tespiti + Lucas-Kanade takip + ileri-geri eleme.

Kavramlar
---------
* **Shi-Tomasi köşeleri:** Takip etmeye değer, iki yönde de dokulu noktalar
  (`cv2.goodFeaturesToTrack`). Düz zemin/deniz gibi dokusuz bölgelerde az
  köşe çıkar — bu sayı, akış kalitesinin ilk göstergesidir.
* **Piramidal Lucas-Kanade (LK) optik akışı:** Bir noktanın sonraki karede
  nereye gittiğini, çevresindeki küçük pencerenin en iyi örtüştüğü yeri
  arayarak bulur. "Piramidal" = önce küçültülmüş görüntüde kaba, sonra tam
  çözünürlükte ince arama; büyük hareketleri bu sayede yakalar.
* **İleri-geri (forward-backward) kontrolü:** 0→1 takip edilen nokta, 1→0
  geri takip edildiğinde başladığı yere dönmüyorsa güvenilmezdir (örtülme,
  tekrarlı doku, hareketli nesne). Bu eleme RANSAC'a giden gürültüyü azaltır.

Modül, noktaları kamera profiline göre **undistort edilmiş** piksel
koordinatlarında döndürür; geometri katmanı doğrudan bu koordinatlarla çalışır.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .calibration import CameraProfile


@dataclass
class MatchResult:
    """İki kare arasındaki güvenilir nokta eşleşmeleri (undistorted piksel)."""

    pts0: np.ndarray        # Nx2, ilk karedeki konumlar
    pts1: np.ndarray        # Nx2, ikinci karedeki konumlar
    n_detected: int         # bulunan köşe sayısı
    n_tracked: int          # LK'nin takip edebildiği nokta sayısı
    n_kept: int             # ileri-geri elemeden geçen nokta sayısı
    fb_err_median: float    # ileri-geri hatasının medyanı (px)

    @property
    def flow(self) -> np.ndarray:
        """Nx2 akış vektörleri (pts1 - pts0)."""
        return self.pts1 - self.pts0


class FeatureTracker:
    """Kare çiftinde özellik tespiti + takip + eleme yapan durumsuz motor.

    ``profile=None`` verilirse undistort adımı atlanır (sentetik testler için).
    """

    def __init__(
        self,
        profile: Optional[CameraProfile] = None,
        max_corners: int = 600,
        quality_level: float = 0.01,
        min_distance: int = 20,
        block_size: int = 7,
        win_size: int = 21,
        max_level: int = 4,
        fb_threshold: float = 1.5,
        min_points: int = 12,
        work_width: Optional[int] = 2000,
    ):
        self.profile = profile
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.block_size = block_size
        self.lk_params = dict(
            winSize=(win_size, win_size),
            maxLevel=max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        self.fb_threshold = fb_threshold
        self.min_points = min_points
        # Çalışma çözünürlüğü politikası: girdi bundan genişse takip küçültülmüş
        # kopyada yapılır, noktalar yerel (native) çözünürlüğe geri ölçeklenir.
        # 1080p ve termal değişmez; 4K (4000 px) otomatik yarıya iner. Undistort
        # her zaman native K ile, geri ölçeklenmiş noktalara uygulanır (doğru sıra).
        self.work_width = work_width

    def track_pair(self, gray0: np.ndarray, gray1: np.ndarray) -> Optional[MatchResult]:
        """İki gri kare arasında güvenilir eşleşmeleri bul; başarısızsa None."""
        olcek = 1.0
        if self.work_width is not None and gray0.shape[1] > self.work_width:
            olcek = self.work_width / gray0.shape[1]
            gray0 = cv2.resize(gray0, None, fx=olcek, fy=olcek, interpolation=cv2.INTER_AREA)
            gray1 = cv2.resize(gray1, None, fx=olcek, fy=olcek, interpolation=cv2.INTER_AREA)
        p0 = cv2.goodFeaturesToTrack(
            gray0,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            blockSize=self.block_size,
        )
        if p0 is None or len(p0) < self.min_points:
            return None
        n_detected = len(p0)

        # ileri takip: 0 -> 1
        p1, st_f, _ = cv2.calcOpticalFlowPyrLK(gray0, gray1, p0, None, **self.lk_params)
        st_f = st_f.reshape(-1).astype(bool)
        if st_f.sum() < self.min_points:
            return None

        # geri takip: 1 -> 0 (ileri takibin bulduğu noktalardan geriye)
        p0_back, st_b, _ = cv2.calcOpticalFlowPyrLK(gray1, gray0, p1, None, **self.lk_params)
        st_b = st_b.reshape(-1).astype(bool)

        fb_err = np.linalg.norm(p0_back.reshape(-1, 2) - p0.reshape(-1, 2), axis=1)
        keep = st_f & st_b & (fb_err < self.fb_threshold)
        n_tracked = int(st_f.sum())
        if keep.sum() < self.min_points:
            return None

        pts0 = p0.reshape(-1, 2)[keep].astype(np.float64)
        pts1 = p1.reshape(-1, 2)[keep].astype(np.float64)
        if olcek != 1.0:
            pts0 /= olcek       # native piksel koordinatlarına geri dön
            pts1 /= olcek
        if self.profile is not None:
            pts0 = self.profile.undistort_points(pts0)
            pts1 = self.profile.undistort_points(pts1)

        return MatchResult(
            pts0=pts0,
            pts1=pts1,
            n_detected=n_detected,
            n_tracked=n_tracked,
            n_kept=int(keep.sum()),
            fb_err_median=float(np.median(fb_err[keep])),
        )
