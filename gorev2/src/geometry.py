"""İki kare arasındaki kamera hareketini kestiren geometri araçları.

Kamera çerçevesi uzlaşımı (nadir bakışta): **X sağa, Y görüntüde aşağı,
Z ileri = yere doğru.** Böylece Z, dünya "aşağı" yönüne yaklaşık hizalıdır
(veri keşfinde +z'nin aşağı baktığını bulmuştuk — NED benzeri).

İki kestirim yolu
-----------------
1. **Benzerlik (4 serbestlik):** ``p1 ≈ s·R(θ)·p0 + t``. Tam dik bakış
   varsayımı altında geçerli; çok gürbüz, yorumu kolay:

   * medyan akış ``(du, dv)`` → yatay hareket:  ``Δxy_cam = -(du, dv)·h/f``
   * ölçek ``s``  → düşey hareket:  ``h₂ = h₁/s``,  ``Δz_cam = h₁(1 - 1/s)``
     (alçalınca zemin büyür: s>1, Δz_cam>0 = aşağı)
   * görüntü dönmesi ``θ`` → yön değişimi: ``Δψ = -θ``
     (kamera +ψ dönerse zemin görüntüde -ψ döner)

2. **Homografi (8 serbestlik) + ayrıştırma:** Düzlemsel sahnede iki görüntü
   ``H = K(R - t·nᵀ/d)K⁻¹`` ile bağlıdır. ``cv2.decomposeHomographyMat``
   4 aday {R, t/d, n} verir; fiziksel çözüm, derinlik pozitifliği ve düzlem
   normalinin beklenen yöne (nadir'de ≈ [0,0,1]) yakınlığı ile seçilir.
   Eğik bakışta (70°) benzerlikten daha doğrudur; karşılığında çözüm seçimi
   ve gürültü hassasiyeti getirir.

Her iki yol da RANSAC ile aykırı eşleşmeleri (hareketli taşıtlar!) eler.
İşaret uzlaşımları ``notebooks/02_vo_cekirdegi.ipynb`` içindeki sentetik
senaryolarla sayısal olarak doğrulanmıştır.
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
    """İki kare arasındaki 4 serbestlikli görüntü hareketi."""

    du: float            # medyan akış x (undistorted px)
    dv: float            # medyan akış y (undistorted px)
    scale: float         # s (>1: zemin büyüyor = alçalma)
    angle_rad: float     # θ, görüntü düzleminde dönme
    n_inliers: int
    inlier_ratio: float

    def camera_delta(self, height: float, f: float) -> np.ndarray:
        """Kamera çerçevesinde metrik yer değiştirme (dx, dy, dz).

        ``height``: kameranın zemine dik uzaklığı (m), ``f``: odak (px).
        dz pozitif = aşağı (yere yaklaşma).

        **Çerçeve uzlaşımı:** dönen vektör **kamera-2 (adım sonrası)**
        ekseninde ifade edilidir — sentetik F senaryosuyla doğrulandı.
        Dünyaya çevirirken güncellenmiş yön kullanılmalı:
        ``Δp_dünya = R_dünya←kam(ψ_yeni) · Δp``.
        """
        gsd = height / f
        dx = -self.du * gsd
        dy = -self.dv * gsd
        dz = height * (1.0 - 1.0 / self.scale)
        return np.array([dx, dy, dz])

    def height_after(self, height: float) -> float:
        """Bu adımdan sonra zemine dik uzaklık."""
        return height / self.scale

    @property
    def yaw_delta_rad(self) -> float:
        """Kameranın yön (yaw) değişimi: Δψ = -θ."""
        return -self.angle_rad


def estimate_similarity(
    pts0: np.ndarray,
    pts1: np.ndarray,
    ref_point: Optional[tuple[float, float]] = None,
    ransac_thresh: float = 2.0,
    min_inliers: int = 15,
) -> Optional[SimilarityMotion]:
    """RANSAC ile 4 serbestlikli benzerlik hareketi kestir; başarısızsa None.

    ``ref_point``: akışın değerlendirileceği piksel — **asal nokta (cx, cy)
    verilmelidir**. Optik eksenin zemini deldiği nokta olduğu için dönme ve
    ölçeğin akışa katkısı orada sıfırdır; böylece ``(du, dv)`` saf öteleme
    akışı olur (medyan akıştaki dönme/ölçek sapması ortadan kalkar).
    ``None`` verilirse inlier akışının medyanı kullanılır (daha az doğru).
    """
    if len(pts0) < min_inliers:
        return None
    M, inl = cv2.estimateAffinePartial2D(
        pts0, pts1, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh
    )
    if M is None or inl is None:
        return None
    mask = inl.reshape(-1).astype(bool)
    if mask.sum() < min_inliers:
        return None
    if ref_point is not None:
        # akisi asal noktada degerlendir: M @ [cx, cy, 1] - [cx, cy]
        cx, cy = ref_point
        du = M[0, 0] * cx + M[0, 1] * cy + M[0, 2] - cx
        dv = M[1, 0] * cx + M[1, 1] * cy + M[1, 2] - cy
    else:
        flow = pts1[mask] - pts0[mask]
        du, dv = np.median(flow, axis=0)
    a, b = M[0, 0], M[1, 0]
    return SimilarityMotion(
        du=float(du),
        dv=float(dv),
        scale=float(np.hypot(a, b)),
        angle_rad=float(np.arctan2(b, a)),
        n_inliers=int(mask.sum()),
        inlier_ratio=float(mask.mean()),
    )


# --------------------------------------------------------------------------
# 2) Homografi yolu
# --------------------------------------------------------------------------
@dataclass
class HomographyMotion:
    """Homografi ayrıştırmasından seçilen fiziksel çözüm.

    * ``R``: kamera-1 koordinatındaki bir noktayı kamera-2 koordinatına
      taşıyan dönme (``p2 = R·p1 + t``).
    * ``cam_delta_over_d``: kamera-2'nin kamera-1'e göre konumu,
      kamera-1 çerçevesinde, düzlem uzaklığına bölünmüş (``c/d``).
    * ``n``: zemin düzleminin normali, kamera-1 çerçevesinde (birim,
      nadir bakışta ≈ [0, 0, 1]).
    """

    R: np.ndarray
    cam_delta_over_d: np.ndarray
    n: np.ndarray
    n_inliers: int
    inlier_ratio: float

    def camera_delta(self, plane_dist: float) -> np.ndarray:
        """Metrik yer değiştirme (d = düzleme dik uzaklık, m).

        **Çerçeve uzlaşımı:** dönen vektör **kamera-1 (adım öncesi)**
        ekseninde ifade edilir (benzerlik yolundan farklı!). Dünyaya
        çevirirken adım öncesi yön kullanılmalı:
        ``Δp_dünya = R_dünya←kam(ψ_eski) · Δp``.
        """
        return self.cam_delta_over_d * plane_dist

    def plane_dist_after(self, plane_dist: float) -> float:
        """Bu adımdan sonra düzleme dik uzaklık: d₂ = d₁(1 - nᵀ·c/d₁)."""
        return plane_dist * (1.0 - float(self.n @ self.cam_delta_over_d))

    @property
    def yaw_delta_rad(self) -> float:
        """Kameranın yön (yaw) değişimi (Z ekseni etrafında)."""
        return -float(np.arctan2(self.R[1, 0], self.R[0, 0]))

    @property
    def tilt_deg(self) -> float:
        """Bakışın nadir'den sapması: normal ile optik eksen arasındaki açı."""
        return float(np.degrees(np.arccos(np.clip(self.n[2], -1.0, 1.0))))


def estimate_homography_motion(
    pts0: np.ndarray,
    pts1: np.ndarray,
    K: np.ndarray,
    ransac_thresh: float = 2.0,
    min_inliers: int = 15,
    prev_normal: Optional[np.ndarray] = None,
) -> Optional[HomographyMotion]:
    """Homografi kestir, ayrıştır ve fiziksel çözümü seç; başarısızsa None.

    Çözüm seçimi iki kuralla yapılır:
    1. **Derinlik pozitifliği:** Görünen noktaların ışınları düzlemi kameranın
       önünde kesmelidir (``nᵀ·K⁻¹x > 0``). Bu, 4 adayın ikisini eler.
    2. **Normal tutarlılığı:** Kalan iki adaydan, normali referans yöne
       (önceki karenin normali; yoksa nadir beklentisi [0,0,1]) en yakın
       olan seçilir.
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

    # Derinlik pozitifliği icin inlier isinlari (kamera-1 cerecevesi)
    pts_h = np.column_stack([pts0[mask], np.ones(int(mask.sum()))])
    rays = (np.linalg.inv(K) @ pts_h.T).T  # Nx3

    ref = prev_normal if prev_normal is not None else np.array([0.0, 0.0, 1.0])
    best, best_score = None, -np.inf
    for R, t, n in zip(Rs, ts, ns):
        n = n.reshape(3)
        # kural 1: noktalarin en az %95'i duzlemin gorunen tarafinda olmali
        if (rays @ n > 0).mean() < 0.95:
            continue
        # kural 2: referans normale yakinlik
        score = float(n @ ref)
        if score > best_score:
            best_score = score
            best = (R, t.reshape(3), n)

    if best is None:
        # Saf donme durumu: oteleme ~0 iken H ≈ K·R·K⁻¹ olur ve duzlem
        # ayristirmasi tanimsizdir (normal belirsiz, adaylar gecersiz cikar).
        # Bu durumda donmeyi dogrudan H'den cikarip otelemeyi 0 kabul ederiz;
        # boylece hover/donus anlarinda yon takibi kopmaz.
        A = np.linalg.inv(K) @ H @ K
        det = np.linalg.det(A)
        if det <= 0:
            return None
        A = A / np.cbrt(det)                      # olcegi normalize et (det=1)
        U, _, Vt = np.linalg.svd(A)
        R = U @ np.diag([1.0, 1.0, np.linalg.det(U @ Vt)]) @ Vt
        if np.linalg.norm(A - R) > 0.05:          # gercekten donmeye yakin mi?
            return None
        return HomographyMotion(
            R=R,
            cam_delta_over_d=np.zeros(3),
            n=ref.copy(),
            n_inliers=int(mask.sum()),
            inlier_ratio=float(mask.mean()),
        )

    R, t, n = best
    # p2 = R p1 + t (t = gercek_t/d)  =>  kamera-2'nin kamera-1'deki konumu: c/d = -Rᵀ t
    cam_delta_over_d = -(R.T @ t)
    return HomographyMotion(
        R=R,
        cam_delta_over_d=cam_delta_over_d,
        n=n,
        n_inliers=int(mask.sum()),
        inlier_ratio=float(mask.mean()),
    )
