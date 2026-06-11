"""
Kareden kareye göreli hareket kestirimi.

Faz 1 — 2B benzerlik modeli (estimateAffinePartial2D):
  Nadir'e yakın (yere bakan) kamera için ardışık kareler ölçek+dönme+öteleme
  (benzerlik) ile çok iyi modellenir. RANSAC dahili, 4-katlı homografi
  belirsizliği yok.

  Optik eksendeki (principal point) öteleme  t_c (piksel) çıkarılır:
      t_c = s·R·c + t − c
  ve normalize edilir:
      raw_dx = −t_c_x / fx   (≈ T_x / irtifa)
      raw_dy = −t_c_y / fy   (≈ T_y / irtifa)
      raw_dz = −ln(s)        (≈ Δirtifa / irtifa)
      dyaw   = atan2(b, a)   (kare-arası yaw, gövde çerçevesinde)

  Üç eksen de aynı global ölçek (≈irtifa) ile tutarlıdır; metrik dönüşüm
  Umeyama (aligner) tarafından sağlıklı GT segmentinden öğrenilir.

Faz 2 — homografi/essential modelleri aynı arayüzle eklenecek (aşağıda iskelet).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class FrameMotion:
    """Kare t-1 → t için ham göreli hareket (gövde/kamera çerçevesinde)."""
    raw_dx: float          # ≈ T_x / irtifa
    raw_dy: float          # ≈ T_y / irtifa
    raw_dz: float          # ≈ Δirtifa / irtifa  (−ln ölçek)
    dyaw: float            # kare-arası yaw (rad)
    scale: float           # benzerlik ölçeği s
    n_inliers: int
    inlier_ratio: float
    ok: bool               # güvenilir mi?

    @staticmethod
    def zero() -> "FrameMotion":
        return FrameMotion(0.0, 0.0, 0.0, 0.0, 1.0, 0, 0.0, False)


def estimate_similarity_motion(pts_prev: np.ndarray, pts_cur: np.ndarray,
                               K: np.ndarray, ransac_thresh: float = 2.0,
                               min_inliers: int = 30) -> FrameMotion:
    """2B benzerlik dönüşümünden ham göreli hareket çıkar."""
    if len(pts_prev) < min_inliers or len(pts_cur) < min_inliers:
        return FrameMotion.zero()

    M, inliers = cv2.estimateAffinePartial2D(
        pts_prev, pts_cur, method=cv2.RANSAC,
        ransacReprojThreshold=ransac_thresh, maxIters=2000, confidence=0.999,
    )
    if M is None or inliers is None:
        return FrameMotion.zero()

    n_in = int(inliers.sum())
    ratio = n_in / float(len(pts_prev))
    if n_in < min_inliers:
        return FrameMotion.zero()

    a, b = M[0, 0], M[1, 0]            # a = s·cosθ, b = s·sinθ
    tx, ty = M[0, 2], M[1, 2]
    s = float(np.hypot(a, b))
    theta = float(np.arctan2(b, a))
    if s < 1e-6:
        return FrameMotion.zero()

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    # Optik eksendeki (principal point) görüntü hareketi:  t_c = M·c − c
    tc_x = a * cx - b * cy + tx - cx
    tc_y = b * cx + a * cy + ty - cy

    raw_dx = -tc_x / fx               # ≈ T_x / irtifa  (görüntü hareketi kameranın tersi)
    raw_dy = -tc_y / fy
    raw_dz = -float(np.log(s))        # ≈ Δirtifa / irtifa

    return FrameMotion(raw_dx, raw_dy, raw_dz, theta, s, n_in, ratio, True)


# ---------------------------------------------------------------------------
# Faz 2 iskeleti — homografi tabanlı (perspektif/eğim etkilerini de yakalar)
# ---------------------------------------------------------------------------
def estimate_homography_motion(pts_prev: np.ndarray, pts_cur: np.ndarray,
                               K: np.ndarray, ransac_thresh: float = 2.0,
                               min_inliers: int = 30) -> FrameMotion:
    """
    Homografi (yer düzlemi) → decomposeHomographyMat ile düzgün (R, t/d, n) ayrıştırması.

    Nadir'e yakın eğimli kamerada (70-90°) perspektif etkilerini yakalar; benzerlik
    modelinden daha doğru yaw verir. t = T/d (d=düzleme uzaklık ≈ irtifa) olduğundan
    t bileşenleri doğrudan irtifa-normalize öteleme olur:
        raw_dx = t[0],  raw_dy = t[1],  raw_dz = t[2]   (≈ T / irtifa)
        dyaw   = optik eksen (kamera z) etrafındaki dönme  atan2(R[1,0], R[0,0])

    Belirsizlik (4 çözüm) giderme: aşağı bakan kamera için düzlem normali optik
    eksene (±z) yakındır → |n_z| en büyük çözüm seçilir; ayrıca görünür referans
    noktalarıyla filtrelenir.
    """
    if len(pts_prev) < min_inliers:
        return FrameMotion.zero()
    H, mask = cv2.findHomography(pts_prev, pts_cur, cv2.RANSAC, ransac_thresh)
    if H is None or mask is None:
        return FrameMotion.zero()
    inl = mask.ravel().astype(bool)
    n_in = int(inl.sum())
    if n_in < min_inliers:
        return FrameMotion.zero()
    ratio = n_in / float(len(pts_prev))

    try:
        num, Rs, Ts, Ns = cv2.decomposeHomographyMat(H, K)
    except cv2.error:
        return FrameMotion.zero()
    if num == 0:
        return FrameMotion.zero()

    Ns_v = [np.asarray(x).ravel() for x in Ns]   # her biri (3,)
    Ts_v = [np.asarray(x).ravel() for x in Ts]
    cand = list(range(num))
    # Görünür referans noktalarıyla fiziksel olmayan çözümleri ele
    try:
        pp = pts_prev[inl].reshape(-1, 1, 2).astype(np.float32)
        pc = pts_cur[inl].reshape(-1, 1, 2).astype(np.float32)
        keep = cv2.filterHomographyDecompByVisibleRefpoints(
            np.array(Rs), np.array(Ns), pp, pc)
        if keep is not None and len(keep):
            cand = [int(i) for i in keep.ravel()]
    except cv2.error:
        pass

    # Aşağı bakan kamera: normal optik eksene yakın → |n_z| en büyük
    best = max(cand, key=lambda i: abs(Ns_v[i][2]))
    R = np.asarray(Rs[best])
    t = Ts_v[best].copy()
    n = Ns_v[best]
    # Normal kamera tarafına baksın (n_z>0); çift (R,t,n)~(R,-t,-n) tutarlılığı
    if n[2] < 0:
        t = -t
    theta = float(np.arctan2(R[1, 0], R[0, 0]))
    return FrameMotion(float(t[0]), float(t[1]), float(t[2]), theta,
                       1.0, n_in, ratio, True)


def estimate_hybrid_motion(pts_prev: np.ndarray, pts_cur: np.ndarray, K: np.ndarray,
                           ransac_thresh: float = 2.0, min_inliers: int = 30) -> FrameMotion:
    """
    Hibrit: ÖTELEME + Z benzerlikten (optik-akış; düşük gürültü), YAW homografiden
    (perspektifi düzgün ayrıştırır; daha az açısal drift).

    Gözlem: düzgün homografi ayrıştırması daha iyi dönme (X drift ↓) verir ama
    düzlem-dışı öteleme t[2] zayıf gözlemli olduğundan Z/Y gürültülüdür. Benzerliğin
    optik-eksen ötelemesi ve −ln(ölçek) tabanlı Z'si ise daha kararlıdır. İkisinin
    en iyi yanları birleştirilir.
    """
    sim = estimate_similarity_motion(pts_prev, pts_cur, K, ransac_thresh, min_inliers)
    hom = estimate_homography_motion(pts_prev, pts_cur, K, ransac_thresh, min_inliers)
    if not sim.ok:
        return hom
    if not hom.ok:
        return sim
    return FrameMotion(sim.raw_dx, sim.raw_dy, sim.raw_dz, hom.dyaw,
                       sim.scale, min(sim.n_inliers, hom.n_inliers),
                       min(sim.inlier_ratio, hom.inlier_ratio), True)


def estimate_motion(pts_prev: np.ndarray, pts_cur: np.ndarray, K: np.ndarray,
                    model: str = "hybrid", ransac_thresh: float = 2.0,
                    min_inliers: int = 30) -> FrameMotion:
    if model == "similarity":
        return estimate_similarity_motion(pts_prev, pts_cur, K, ransac_thresh, min_inliers)
    if model == "homography":
        return estimate_homography_motion(pts_prev, pts_cur, K, ransac_thresh, min_inliers)
    if model == "hybrid":
        return estimate_hybrid_motion(pts_prev, pts_cur, K, ransac_thresh, min_inliers)
    raise ValueError(f"Bilinmeyen hareket modeli: {model}")
