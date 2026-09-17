"""İki kare arasındaki kamera hareketini kestiren geometri araçları.

Kamera çerçevesi uzlaşımı (nadir bakışta): **X sağa, Y görüntüde aşağı,
Z ileri = yere doğru** (NED benzeri; +z aşağı).

İki kestirim yolu
-----------------
1. **Benzerlik (4 serbestlik):** ``p1 ≈ s·R(θ)·p0 + t``. Tam dik bakış varsayımı;
   çok gürbüz. Yalnız LK son-çare yolunda (``method="fuzyon"``) kullanılır.
2. **Homografi (8 serbestlik) + ayrıştırma:** düzlemsel sahnede
   ``H = K(R − t·nᵀ/d)K⁻¹``. ``cv2.decomposeHomographyMat`` 4 aday verir; fiziksel
   çözüm derinlik pozitifliği ve normalin beklenen yöne yakınlığı ile seçilir.
   **Sevk edilen yol budur.**

`ransac_thresh` VARSAYILANI 1,0 px — kanıta dayalı (2,0'dan indirildi):
yön hatası 1,45° → 1,27°, ölçek %3,85 → %2,11. Kameradan bağımsız 1,0 px
kullanmak, "çalışma pikseline göre normalize etme" denemesi yıllar arası
taşınmadığı için korunmuştur.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


# --------------------------------------------------------------------------
# 1) Benzerlik yolu
# --------------------------------------------------------------------------
@dataclass
class SimilarityMotion:
    du: float            # medyan akış x (undistorted px)
    dv: float            # medyan akış y
    scale: float         # s (>1: zemin büyüyor = alçalma)
    angle_rad: float     # görüntü düzleminde dönme
    n_inliers: int
    inlier_ratio: float

    def camera_delta(self, height: float, f: float) -> np.ndarray:
        """Kamera çerçevesinde metrik yer değiştirme (dx, dy, dz); dz+ = aşağı.

        **Çerçeve uzlaşımı:** dönen vektör **kamera-2 (adım sonrası)** eksenindedir.
        """
        gsd = height / f
        return np.array([-self.du * gsd, -self.dv * gsd,
                         height * (1.0 - 1.0 / self.scale)])

    @property
    def yaw_delta_rad(self) -> float:
        """Kameranın yön değişimi: Δψ = −θ."""
        return -self.angle_rad


def estimate_similarity(
    pts0: np.ndarray,
    pts1: np.ndarray,
    ref_point: Optional[tuple[float, float]] = None,
    ransac_thresh: float = 1.0,
    min_inliers: int = 15,
) -> Optional[SimilarityMotion]:
    """RANSAC ile 4 serbestlikli benzerlik hareketi; başarısızsa None.

    ``ref_point`` **asal nokta (cx, cy)** verilmelidir: optik eksenin zemini
    deldiği nokta olduğu için dönme ve ölçeğin akışa katkısı orada sıfırdır,
    böylece (du, dv) saf öteleme akışı olur.
    """
    if len(pts0) < min_inliers:
        return None
    M, inl = cv2.estimateAffinePartial2D(
        pts0, pts1, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh)
    if M is None or inl is None:
        return None
    mask = inl.reshape(-1).astype(bool)
    if mask.sum() < min_inliers:
        return None
    if ref_point is not None:
        cx, cy = ref_point
        du = M[0, 0] * cx + M[0, 1] * cy + M[0, 2] - cx
        dv = M[1, 0] * cx + M[1, 1] * cy + M[1, 2] - cy
    else:
        du, dv = np.median(pts1[mask] - pts0[mask], axis=0)
    a, b = M[0, 0], M[1, 0]
    return SimilarityMotion(
        du=float(du), dv=float(dv),
        scale=float(np.hypot(a, b)), angle_rad=float(np.arctan2(b, a)),
        n_inliers=int(mask.sum()), inlier_ratio=float(mask.mean()),
    )


# --------------------------------------------------------------------------
# 2) Homografi yolu (sevk edilen)
# --------------------------------------------------------------------------
@dataclass
class HomographyMotion:
    """Homografi ayrıştırmasından seçilen fiziksel çözüm.

    * ``R``: kamera-1'deki noktayı kamera-2'ye taşıyan dönme (``p2 = R·p1 + t``)
    * ``cam_delta_over_d``: kamera-2'nin kamera-1'e göre konumu, kamera-1
      çerçevesinde, düzlem uzaklığına bölünmüş (``c/d``)
    * ``n``: zemin düzleminin normali, kamera-1 çerçevesinde (nadir'de ≈ [0,0,1])
    """

    R: np.ndarray
    cam_delta_over_d: np.ndarray
    n: np.ndarray
    n_inliers: int
    inlier_ratio: float

    @property
    def yaw_delta_rad(self) -> float:
        return -float(np.arctan2(self.R[1, 0], self.R[0, 0]))


def estimate_homography_motion(
    pts0: np.ndarray,
    pts1: np.ndarray,
    K: np.ndarray,
    ransac_thresh: float = 1.0,
    min_inliers: int = 15,
    prev_normal: Optional[np.ndarray] = None,
) -> Optional[HomographyMotion]:
    """Homografi kestir, ayrıştır, fiziksel çözümü seç; başarısızsa None.

    Çözüm seçimi iki kuralla:
    1. **Derinlik pozitifliği** — görünen noktaların ışınları düzlemi kameranın
       önünde kesmeli (``nᵀ·K⁻¹x > 0``, inlier'ların ≥%95'i). 4 adayın ikisini eler.
    2. **Normal tutarlılığı** — kalan iki adaydan normali referans yöne (önceki
       karenin normali; yoksa nadir beklentisi [0,0,1]) en yakın olan seçilir.
    """
    if len(pts0) < min_inliers:
        return None
    H, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, ransac_thresh)
    if H is None or mask is None:
        return None
    mask = mask.reshape(-1).astype(bool)
    if mask.sum() < min_inliers:
        return None

    try:
        _, Rs, ts, ns = cv2.decomposeHomographyMat(H, K)
    except cv2.error:
        return None

    pts_h = np.column_stack([pts0[mask], np.ones(int(mask.sum()))])
    rays = (np.linalg.inv(K) @ pts_h.T).T

    ref = prev_normal if prev_normal is not None else np.array([0.0, 0.0, 1.0])
    best, best_score = None, -np.inf
    for R, t, n in zip(Rs, ts, ns):
        n = n.reshape(3)
        if (rays @ n > 0).mean() < 0.95:          # kural 1
            continue
        score = float(n @ ref)                    # kural 2
        if score > best_score:
            best_score, best = score, (R, t.reshape(3), n)

    if best is None:
        # Saf dönme: öteleme ~0 iken H ≈ K·R·K⁻¹ olur ve düzlem ayrıştırması
        # tanımsızdır. Dönmeyi doğrudan H'den çıkarıp ötelemeyi 0 kabul ederiz;
        # böylece hover/dönüş anlarında yön takibi kopmaz.
        A = np.linalg.inv(K) @ H @ K
        det = np.linalg.det(A)
        if det <= 0:
            return None
        A = A / np.cbrt(det)
        U, _, Vt = np.linalg.svd(A)
        R = U @ np.diag([1.0, 1.0, np.linalg.det(U @ Vt)]) @ Vt
        if np.linalg.norm(A - R) > 0.05:          # gerçekten dönmeye yakın mı?
            return None
        return HomographyMotion(R=R, cam_delta_over_d=np.zeros(3), n=ref.copy(),
                                n_inliers=int(mask.sum()),
                                inlier_ratio=float(mask.mean()))

    R, t, n = best
    # p2 = R·p1 + t  (t = gerçek_t/d)  =>  kamera-2'nin kamera-1'deki konumu: −Rᵀt
    return HomographyMotion(R=R, cam_delta_over_d=-(R.T @ t), n=n,
                            n_inliers=int(mask.sum()),
                            inlier_ratio=float(mask.mean()))
