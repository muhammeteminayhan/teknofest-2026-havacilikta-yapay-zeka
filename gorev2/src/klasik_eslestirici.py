"""Klasik özellik eşleştiriciler (SIFT / ORB / AKAZE / BRISK) — `FeatureTracker` arayüzü.

**Neden bu modül var.** Ölçüm katmanı turlarında *öğrenilmiş* seyrek eşleştiriciler
denendi (XFeat, DISK+LightGlue, gerçek SuperPoint+LightGlue) ve hepsi RAFT'a yenildi;
ayrıca ORB-SLAM'in **tamamı** bir sistem olarak denendi (monoküler ölçek üçgenlemesi
nadir sahnede dejenere oldu). Ama **klasik dedektörler kendi boru hattımızda bir
`MatchResult` üreticisi olarak hiç sınanmadı**. Bu boşluğu kapatır.

Klasiklerin öğrenilmişlerden farklı olabileceği yer:
* **SIFT** ölçek-uzayı ekstremumları + gradyan histogramı kullanır; havadan çekimdeki
  tekrarlayan dokuda (tarla, orman) öğrenilmiş betimleyicilerden farklı davranabilir,
  ve Lowe oran testi belirsiz eşleşmeleri baştan eler.
* **ORB/BRISK/AKAZE** ikili betimleyici — çok hızlı; RAFT'ın GPU gerektirdiği yerde
  CPU yedeği olma potansiyeli taşır (yarışma makinesinde GPU sorunu çıkarsa).

Hepsi ``track_pair(gray0, gray1) -> MatchResult | None`` döndürür; `PairMeasurement`,
`evaluation.build_*` ve `nokta_onbellek.py` değişmeden çalışır.

**Eşleştirme politikası** (üçü birden, aykırıyı RANSAC'a bırakmadan önce eler):
1. k=2 en yakın komşu + **Lowe oran testi** — belirsiz eşleşmeyi at,
2. **karşılıklı en yakın komşu** (0→1 ve 1→0 aynı çifti seçmeli),
3. RANSAC'a yalnız bu ikisinden geçenler girer.
"""
from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from .calibration import CameraProfile
from .matching import MatchResult

# Betimleyici tipine göre doğru mesafe ölçüsü: SIFT float (L2), diğerleri ikili (Hamming)
_NORM = {"sift": cv2.NORM_L2, "orb": cv2.NORM_HAMMING,
         "akaze": cv2.NORM_HAMMING, "brisk": cv2.NORM_HAMMING}


class KlasikTracker:
    """SIFT/ORB/AKAZE/BRISK ile kare çifti eşleme."""

    def __init__(
        self,
        profile: Optional[CameraProfile] = None,
        yontem: str = "sift",
        n_ozellik: int = 3000,
        lowe: float = 0.75,
        downscale: Optional[float] = None,
        min_points: int = 12,
        karsilikli: bool = True,
    ):
        yontem = yontem.lower()
        if yontem not in _NORM:
            raise ValueError(f"bilinmeyen yontem: {yontem} (secenekler: {sorted(_NORM)})")
        self.yontem = yontem
        self.profile = profile
        self.downscale = downscale
        self.min_points = min_points
        self.lowe = lowe
        self.karsilikli = karsilikli

        if yontem == "sift":
            self.det = cv2.SIFT_create(nfeatures=n_ozellik)
        elif yontem == "orb":
            # fastThreshold düşürülür: havadan görüntüde kontrast düşük olabiliyor
            self.det = cv2.ORB_create(nfeatures=n_ozellik, fastThreshold=12)
        elif yontem == "akaze":
            self.det = cv2.AKAZE_create()
        else:
            self.det = cv2.BRISK_create()
        self.bf = cv2.BFMatcher(_NORM[yontem], crossCheck=False)
        logging.info(f"KlasikTracker hazir ({yontem}, n={n_ozellik}, lowe={lowe})")

    # ------------------------------------------------------------------
    @staticmethod
    def auto_scale(width: int) -> float:
        """RaftTracker ile aynı politika (hedef ~1000 px) — maliyet karşılaştırılabilir kalsın."""
        if width <= 1024:
            return 1.0
        if width <= 2048:
            return 0.5
        return 0.25

    def _olcek(self, width: int) -> float:
        return self.downscale if self.downscale is not None else self.auto_scale(width)

    def _lowe_suz(self, d0, d1):
        """k=2 + Lowe oran testi -> (sorgu_idx, hedef_idx) listesi."""
        if d0 is None or d1 is None or len(d0) < 2 or len(d1) < 2:
            return []
        ciftler = self.bf.knnMatch(d0, d1, k=2)
        return [(m.queryIdx, m.trainIdx) for m, n in
                (c for c in ciftler if len(c) == 2)
                if m.distance < self.lowe * n.distance]

    # ------------------------------------------------------------------
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
        if self.karsilikli:
            geri = set((t, q) for q, t in self._lowe_suz(d1, d0))
            ileri = [(q, t) for q, t in ileri if (q, t) in geri]
        if len(ileri) < self.min_points:
            return None

        p0 = np.array([k0[q].pt for q, _ in ileri], dtype=np.float64)
        p1 = np.array([k1[t].pt for _, t in ileri], dtype=np.float64)
        if scale != 1.0:
            p0 = p0 / scale            # doğal piksel koordinatlarına dön
            p1 = p1 / scale
        if self.profile is not None:
            p0 = self.profile.undistort_points(p0)
            p1 = self.profile.undistort_points(p1)
        return MatchResult(pts0=p0, pts1=p1, n_detected=int(n_detected),
                           n_tracked=len(p0), n_kept=len(p0), fb_err_median=0.0)
