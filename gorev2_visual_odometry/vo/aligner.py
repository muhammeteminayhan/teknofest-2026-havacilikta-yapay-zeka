"""
Umeyama benzerlik (Sim3) hizalaması.

Monoküler VO ölçek belirsizdir ve ham yörünge keyfi bir koordinat çerçevesindedir.
Sağlıklı (health=1) karelerde elimizde GT pozisyon olduğundan, ham VO yörüngesini
GT'ye oturtan benzerlik dönüşümünü (ölçek s, dönme R, öteleme t) en küçük kareler
ile buluruz. Bu hem metrik ölçeği hem de referans-koordinat eksen yönelimini verir.

Referans: S. Umeyama, "Least-squares estimation of transformation parameters
between two point patterns", IEEE PAMI 1991.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Similarity:
    """Y ≈ s · R · X + t  dönüşümü."""
    s: float
    R: np.ndarray   # 3x3 dönme
    t: np.ndarray   # (3,) öteleme

    def apply(self, X: np.ndarray) -> np.ndarray:
        """X: (N,3) ham noktalar → (N,3) hizalanmış noktalar."""
        X = np.asarray(X, dtype=np.float64).reshape(-1, 3)
        return (self.s * (self.R @ X.T)).T + self.t

    def apply_vector(self, v: np.ndarray) -> np.ndarray:
        """Öteleme uygulamadan yalnızca ölçek+dönme (göreli hareket vektörleri için)."""
        v = np.asarray(v, dtype=np.float64).reshape(-1, 3)
        return (self.s * (self.R @ v.T)).T

    @staticmethod
    def identity() -> "Similarity":
        return Similarity(1.0, np.eye(3), np.zeros(3))


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True) -> Similarity:
    """
    src (N,3) → dst (N,3) için en iyi benzerlik dönüşümünü hesapla.

    dst ≈ s · R · src + t  hatasını (kare toplamı) minimize eder.
    """
    src = np.asarray(src, dtype=np.float64).reshape(-1, 3)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 3)
    assert src.shape == dst.shape and src.shape[0] >= 3, "En az 3 eşleşen nokta gerekli"
    n = src.shape[0]

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    # Kovaryans matrisi (3x3)
    Sigma = (dst_c.T @ src_c) / n
    U, D, Vt = np.linalg.svd(Sigma)

    # Uygun (proper) dönme garantisi — yansımayı engelle
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt

    if with_scale:
        var_src = (src_c ** 2).sum() / n
        s = float(np.trace(np.diag(D) @ S) / var_src) if var_src > 1e-12 else 1.0
    else:
        s = 1.0

    t = mu_dst - s * (R @ mu_src)
    return Similarity(s, R, t)


def umeyama_ransac(src: np.ndarray, dst: np.ndarray, with_scale: bool = True,
                   iters: int = 200, thresh: float = 2.0, min_inliers: int = 10,
                   seed: int = 0) -> tuple[Similarity, np.ndarray]:
    """
    RANSAC ile gürbüz Umeyama — sağlıklı segmentteki aykırı (outlier) GT/VO
    eşleşmelerine karşı dayanıklı.

    Döndürür: (Similarity, inlier_mask)
    thresh metre cinsinden artık (residual) eşiğidir.
    """
    src = np.asarray(src, dtype=np.float64).reshape(-1, 3)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 3)
    n = src.shape[0]
    if n < 3:
        return Similarity.identity(), np.zeros(n, dtype=bool)
    if n <= 4:
        sim = umeyama(src, dst, with_scale)
        return sim, np.ones(n, dtype=bool)

    rng = np.random.default_rng(seed)
    best_inliers = None
    best_count = -1
    for _ in range(iters):
        idx = rng.choice(n, size=3, replace=False)
        try:
            sim = umeyama(src[idx], dst[idx], with_scale)
        except Exception:
            continue
        res = np.linalg.norm(sim.apply(src) - dst, axis=1)
        inliers = res < thresh
        c = int(inliers.sum())
        if c > best_count:
            best_count = c
            best_inliers = inliers

    if best_inliers is None or best_count < min_inliers:
        # RANSAC başarısız — tüm noktalarla düz fit'e düş
        sim = umeyama(src, dst, with_scale)
        res = np.linalg.norm(sim.apply(src) - dst, axis=1)
        return sim, res < (thresh * 3)

    # İnlier'larla yeniden fit (refine)
    sim = umeyama(src[best_inliers], dst[best_inliers], with_scale)
    res = np.linalg.norm(sim.apply(src) - dst, axis=1)
    inliers = res < thresh
    sim = umeyama(src[inliers], dst[inliers], with_scale)
    return sim, inliers


@dataclass
class Similarity2D:
    """Yer düzlemi (XY) benzerliği:  y ≈ s·R(θ)·x + t  (2B)."""
    s: float
    theta: float       # dönme açısı (rad)
    t: np.ndarray      # (2,) öteleme

    @property
    def R(self) -> np.ndarray:
        c, s_ = np.cos(self.theta), np.sin(self.theta)
        return np.array([[c, -s_], [s_, c]])

    def apply(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64).reshape(-1, 2)
        return (self.s * (self.R @ X.T)).T + self.t

    @staticmethod
    def identity() -> "Similarity2D":
        return Similarity2D(1.0, 0.0, np.zeros(2))


def umeyama_2d(src: np.ndarray, dst: np.ndarray, with_scale: bool = True) -> Similarity2D:
    """2B src→dst için en iyi benzerlik (ölçek, dönme, öteleme)."""
    src = np.asarray(src, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 2)
    n = src.shape[0]
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    Sigma = (dc.T @ sc) / n
    U, D, Vt = np.linalg.svd(Sigma)
    S = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[1, 1] = -1.0
    R = U @ S @ Vt
    theta = float(np.arctan2(R[1, 0], R[0, 0]))
    if with_scale:
        var = (sc ** 2).sum() / n
        s = float(np.trace(np.diag(D) @ S) / var) if var > 1e-12 else 1.0
    else:
        s = 1.0
    t = mu_d - s * (R @ mu_s)
    return Similarity2D(s, theta, t)


def umeyama_2d_ransac(src: np.ndarray, dst: np.ndarray, iters: int = 200,
                      thresh: float = 3.0, min_inliers: int = 10, seed: int = 0
                      ) -> tuple[Similarity2D, np.ndarray]:
    """RANSAC ile gürbüz 2B Umeyama (XY hizalama; sağlıklı segment outlier'larına dayanıklı)."""
    src = np.asarray(src, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 2)
    n = src.shape[0]
    if n < 2:
        return Similarity2D.identity(), np.zeros(n, dtype=bool)
    if n <= 3:
        return umeyama_2d(src, dst), np.ones(n, dtype=bool)
    rng = np.random.default_rng(seed)
    best_inl, best_c = None, -1
    for _ in range(iters):
        idx = rng.choice(n, size=2, replace=False)
        sim = umeyama_2d(src[idx], dst[idx])
        res = np.linalg.norm(sim.apply(src) - dst, axis=1)
        inl = res < thresh
        c = int(inl.sum())
        if c > best_c:
            best_c, best_inl = c, inl
    if best_inl is None or best_c < min_inliers:
        sim = umeyama_2d(src, dst)
        return sim, np.linalg.norm(sim.apply(src) - dst, axis=1) < thresh * 3
    sim = umeyama_2d(src[best_inl], dst[best_inl])
    res = np.linalg.norm(sim.apply(src) - dst, axis=1)
    inl = res < thresh
    return umeyama_2d(src[inl], dst[inl]), inl


if __name__ == "__main__":
    # Sentetik test: bilinen s,R,t ile geri-kurtarım
    rng = np.random.default_rng(42)
    X = rng.normal(size=(200, 3)) * 5
    s_true = 3.7
    theta = 0.8
    R_true = np.array([[np.cos(theta), -np.sin(theta), 0],
                       [np.sin(theta), np.cos(theta), 0],
                       [0, 0, 1]])
    t_true = np.array([10.0, -4.0, 2.0])
    Y = (s_true * (R_true @ X.T)).T + t_true
    Y += rng.normal(size=Y.shape) * 0.01  # hafif gürültü

    sim = umeyama(X, Y)
    print(f"s: gerçek={s_true:.4f} kestirim={sim.s:.4f}")
    print(f"t: gerçek={t_true} kestirim={np.round(sim.t,4)}")
    print(f"R hata (Frobenius): {np.linalg.norm(sim.R - R_true):.6f}")
    resid = np.linalg.norm(sim.apply(X) - Y, axis=1).mean()
    print(f"Ortalama artık: {resid:.6f} m  -> {'TAMAM' if resid < 0.05 else 'HATA'}")

    # RANSAC testi: %20 outlier ekle
    Y2 = Y.copy()
    out_idx = rng.choice(200, 40, replace=False)
    Y2[out_idx] += rng.normal(size=(40, 3)) * 50
    sim_r, inl = umeyama_ransac(X, Y2, thresh=1.0)
    print(f"\nRANSAC: s={sim_r.s:.4f} (gerçek {s_true}), inlier={inl.sum()}/200 "
          f"(beklenen ~160)")
