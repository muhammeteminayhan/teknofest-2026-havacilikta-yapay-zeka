"""
Özellik ön-yüzü — ardışık kareler arasında eşleşen nokta çiftleri üretir.

İki strateji:
  - KLT  : goodFeaturesToTrack + piramidal Lucas-Kanade optik akış (küçük hareket,
           yüksek fps için ideal; ileri-geri tutarlılık kontrolü ile gürbüzleştirilir).
  - ORB  : tespit + ikili tanımlayıcı eşleştirme (büyük hareket / 7.5 fps için daha
           dayanıklı).

Hepsi ortak arayüz sunar:  process(gray) -> (pts_prev, pts_cur)  (Nx2 piksel).
İlk karede (referans yok) boş döner. Ön-işleme (gri + opsiyonel CLAHE) burada yapılır.
"""
from __future__ import annotations

import cv2
import numpy as np


def preprocess(img: np.ndarray, use_clahe: bool = False, clip: float = 2.0,
               grid: int = 8) -> np.ndarray:
    """BGR/gri kareyi gri tonlamaya çevir; isteğe bağlı CLAHE (kontrast eşitleme)."""
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
        gray = clahe.apply(gray)
    return gray


class KLTFrontEnd:
    """Durum tutan KLT takipçisi: kareden kareye eşleşen nokta çiftleri."""

    def __init__(self, max_features: int = 1500, quality: float = 0.01,
                 min_distance: int = 12, min_tracks: int = 200, fb_thresh: float = 1.0):
        self.max_features = max_features
        self.quality = quality
        self.min_distance = min_distance
        self.min_tracks = min_tracks
        self.fb_thresh = fb_thresh
        self.prev_gray: np.ndarray | None = None
        self.prev_pts: np.ndarray | None = None
        self._lk = dict(winSize=(21, 21), maxLevel=4,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    def _detect(self, gray: np.ndarray) -> np.ndarray:
        pts = cv2.goodFeaturesToTrack(
            gray, maxCorners=self.max_features, qualityLevel=self.quality,
            minDistance=self.min_distance, blockSize=7,
        )
        if pts is None:
            return np.empty((0, 1, 2), dtype=np.float32)
        return pts.astype(np.float32)

    def process(self, gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_pts = self._detect(gray)
            return np.empty((0, 2)), np.empty((0, 2))

        if self.prev_pts is None or len(self.prev_pts) < self.min_tracks:
            self.prev_pts = self._detect(self.prev_gray)

        if len(self.prev_pts) == 0:
            self.prev_gray = gray
            self.prev_pts = self._detect(gray)
            return np.empty((0, 2)), np.empty((0, 2))

        # İleri takip
        cur_pts, st, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.prev_pts, None, **self._lk)
        # Geri takip (forward-backward tutarlılık)
        back_pts, st2, _ = cv2.calcOpticalFlowPyrLK(gray, self.prev_gray, cur_pts, None, **self._lk)

        st = st.reshape(-1).astype(bool)
        st2 = st2.reshape(-1).astype(bool)
        fb_err = np.linalg.norm((self.prev_pts - back_pts).reshape(-1, 2), axis=1)
        good = st & st2 & (fb_err < self.fb_thresh)

        p_prev = self.prev_pts.reshape(-1, 2)[good]
        p_cur = cur_pts.reshape(-1, 2)[good]

        # Takip edilen noktaları sürdür (bir sonraki kare için)
        self.prev_gray = gray
        self.prev_pts = p_cur.reshape(-1, 1, 2).astype(np.float32)
        return p_prev, p_cur


class ORBFrontEnd:
    """Durum tutan ORB tespit + eşleştirme (Hamming + çapraz kontrol)."""

    def __init__(self, max_features: int = 1500):
        self.orb = cv2.ORB_create(nfeatures=max_features, fastThreshold=12)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.prev_gray: np.ndarray | None = None
        self.prev_kp = None
        self.prev_des = None

    def process(self, gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        kp, des = self.orb.detectAndCompute(gray, None)
        if self.prev_des is None or des is None or self.prev_des.shape[0] < 2 or des.shape[0] < 2:
            self.prev_gray, self.prev_kp, self.prev_des = gray, kp, des
            return np.empty((0, 2)), np.empty((0, 2))

        matches = self.bf.match(self.prev_des, des)
        matches = sorted(matches, key=lambda m: m.distance)
        p_prev = np.float32([self.prev_kp[m.queryIdx].pt for m in matches])
        p_cur = np.float32([kp[m.trainIdx].pt for m in matches])

        self.prev_gray, self.prev_kp, self.prev_des = gray, kp, des
        return p_prev, p_cur


def make_frontend(backend: str, max_features: int = 1500, **kw):
    backend = backend.lower()
    if backend == "klt":
        return KLTFrontEnd(max_features=max_features,
                           quality=kw.get("quality", 0.01),
                           min_distance=kw.get("min_distance", 12),
                           min_tracks=kw.get("min_tracks", 200))
    if backend == "orb":
        return ORBFrontEnd(max_features=max_features)
    if backend == "sift":
        return SIFTFrontEnd(max_features=max_features)
    raise ValueError(f"Bilinmeyen frontend: {backend}")


class SIFTFrontEnd:
    """SIFT tespit + oran (ratio) testli eşleştirme — daha güçlü ama yavaş."""

    def __init__(self, max_features: int = 1500):
        self.sift = cv2.SIFT_create(nfeatures=max_features)
        self.bf = cv2.BFMatcher(cv2.NORM_L2)
        self.prev_gray = None
        self.prev_kp = None
        self.prev_des = None

    def process(self, gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        kp, des = self.sift.detectAndCompute(gray, None)
        if self.prev_des is None or des is None or len(self.prev_des) < 2 or len(des) < 2:
            self.prev_gray, self.prev_kp, self.prev_des = gray, kp, des
            return np.empty((0, 2)), np.empty((0, 2))
        knn = self.bf.knnMatch(self.prev_des, des, k=2)
        good = [m for m, n in knn if m.distance < 0.75 * n.distance]
        p_prev = np.float32([self.prev_kp[m.queryIdx].pt for m in good]) if good else np.empty((0, 2))
        p_cur = np.float32([kp[m.trainIdx].pt for m in good]) if good else np.empty((0, 2))
        self.prev_gray, self.prev_kp, self.prev_des = gray, kp, des
        return p_prev, p_cur
