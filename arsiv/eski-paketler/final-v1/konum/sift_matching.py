"""SIFT tabanlı eşleme — **CPU yedeği** (GPU/RAFT kullanılamazsa).

Neden SIFT ve neden yalnız yedek:

* **Birincil motor değil.** 8 bağımsız yatakta RAFT'a karşı ölçüldü: SIFT yalnız
  2/8'inde kazandı, 4'ünde felaket (>+3 m) üretti, ortalamada +3,00 m kötü.
* **Ama en iyi CPU yedeği.** RAFT GPU ister. Eski yedek (LK + füzyon + merdiven
  kapalı) üç yatakta ortalama 15,86 m veriyordu; aynı motor merdivenle 12,36 m;
  **SIFT homografi + merdivenle 9,21 m** (GPU'lu RAFT 9,72 m). Yani GPU sorunu
  çıkarsa kayıp neredeyse sıfır. Maliyet ~260 ms/kare (CPU), bütçenin altında.

Eşleştirme politikası (RANSAC'a girmeden önce iki eleme):
1. k=2 en yakın komşu + **Lowe oran testi** — belirsiz eşleşmeyi at,
2. **karşılıklı en yakın komşu** — 0→1 ve 1→0 aynı çifti seçmeli.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .calibration import CameraProfile
from .matching import MatchResult


class SiftTracker:
    """SIFT + Lowe oranı + karşılıklı komşu ile kare çifti eşleme."""

    def __init__(
        self,
        profile: Optional[CameraProfile] = None,
        n_ozellik: int = 3000,
        lowe: float = 0.75,
        downscale: Optional[float] = None,
        min_points: int = 12,
    ):
        self.profile = profile
        self.downscale = downscale
        self.min_points = min_points
        self.lowe = lowe
        self.det = cv2.SIFT_create(nfeatures=n_ozellik)
        self.bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    @staticmethod
    def auto_scale(width: int) -> float:
        """RAFT ile aynı politika — maliyet karşılaştırılabilir kalsın.

        Tam çözünürlük ölçülüp REDDEDİLDİ: 4,48 → 6,17 m (RAFT'ın ds0,75
        bulgusuyla aynı örüntü).
        """
        if width <= 1024:
            return 1.0
        if width <= 2048:
            return 0.5
        return 0.25

    def _olcek(self, width: int) -> float:
        return self.downscale if self.downscale is not None else self.auto_scale(width)

    def _lowe_suz(self, d0, d1):
        if d0 is None or d1 is None or len(d0) < 2 or len(d1) < 2:
            return []
        ciftler = self.bf.knnMatch(d0, d1, k=2)
        return [(m.queryIdx, m.trainIdx)
                for m, n in (c for c in ciftler if len(c) == 2)
                if m.distance < self.lowe * n.distance]

    def track_pair(self, gray0: np.ndarray, gray1: np.ndarray) -> Optional[MatchResult]:
        scale = self._olcek(gray0.shape[1])
        i0, i1 = gray0, gray1
        if scale != 1.0:
            i0 = cv2.resize(gray0, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            i1 = cv2.resize(gray1, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        k0, d0 = self.det.detectAndCompute(i0, None)
        k1, d1 = self.det.detectAndCompute(i1, None)
        if not k0 or not k1:
            return None
        n_detected = min(len(k0), len(k1))

        ileri = self._lowe_suz(d0, d1)
        if not ileri:
            return None
        geri = {(t, q) for q, t in self._lowe_suz(d1, d0)}
        ileri = [(q, t) for q, t in ileri if (q, t) in geri]
        if len(ileri) < self.min_points:
            return None

        pts0 = np.array([k0[q].pt for q, _ in ileri], dtype=np.float64)
        pts1 = np.array([k1[t].pt for _, t in ileri], dtype=np.float64)
        if scale != 1.0:
            pts0, pts1 = pts0 / scale, pts1 / scale
        if self.profile is not None:
            pts0 = self.profile.undistort_points(pts0)
            pts1 = self.profile.undistort_points(pts1)

        return MatchResult(pts0=pts0, pts1=pts1, n_detected=int(n_detected),
                           n_tracked=len(pts0), n_kept=len(pts0), fb_err_median=0.0)
