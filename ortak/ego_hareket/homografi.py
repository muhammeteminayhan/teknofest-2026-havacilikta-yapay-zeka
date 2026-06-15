"""
Ego-hareket (kamera küresel hareketi) — ardışık iki kare arasında homografi kestirimi.

Neden ortak? Görev 1'de "taşıt gerçekten mi hareketli yoksa kamera mı hareket etti"
ayrımı (şartname s.11) ile Görev 2'nin görsel odometrisi aynı temel hesaba dayanır:
sahnenin (yer düzleminin) kareler arası homografisi. Bu modül o homografiyi tek yerde
üretir; Görev 1 hareket-durumu ve (ileride) Görev 2 buradan beslenir.

Yaklaşım: ORB özellikleri (hızlı, lisans-serbest) + BFMatcher + RANSAC homografi.
Yer çoğunlukla düzlemsel ve kamera yüksekte (nadir 70-90°) olduğundan homografi modeli
küresel hareketi iyi temsil eder. Az dokulu sahnede özellik yetersizse None döner.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class EgoHareket:
    """Bir kare çiftinin ego-hareket sonucu."""
    H: Optional[np.ndarray]      # 3x3 homografi (önceki -> şimdiki), None ise kestirilemedi
    n_inlier: int                # RANSAC inlier sayısı
    ok: bool                     # geçerli kestirim mi

    @staticmethod
    def gecersiz() -> "EgoHareket":
        return EgoHareket(H=None, n_inlier=0, ok=False)


def _griye(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def kestir_homografi(prev_img: np.ndarray, cur_img: np.ndarray,
                     max_features: int = 2000, min_inliers: int = 25,
                     ransac_thresh: float = 3.0) -> EgoHareket:
    """
    İki kare arasında küresel kamera hareketini temsil eden homografiyi kestirir.

    prev_img, cur_img : BGR veya gri kareler (aynı çözünürlükte olmalı).
    Döndürür: EgoHareket (H: önceki pikselleri şimdiki kareye taşıyan homografi).
    """
    g_prev, g_cur = _griye(prev_img), _griye(cur_img)

    orb = cv2.ORB_create(nfeatures=max_features)
    k1, d1 = orb.detectAndCompute(g_prev, None)
    k2, d2 = orb.detectAndCompute(g_cur, None)
    if d1 is None or d2 is None or len(k1) < min_inliers or len(k2) < min_inliers:
        return EgoHareket.gecersiz()

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(d1, d2)
    if len(matches) < min_inliers:
        return EgoHareket.gecersiz()

    matches = sorted(matches, key=lambda m: m.distance)
    src = np.float32([k1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([k2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_thresh)
    if H is None or mask is None:
        return EgoHareket.gecersiz()
    n_in = int(mask.sum())
    if n_in < min_inliers:
        return EgoHareket(H=H, n_inlier=n_in, ok=False)
    return EgoHareket(H=H, n_inlier=n_in, ok=True)


def sabit_konum_tahmini(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Önceki karedeki noktaların, nesne SABİT olsaydı şimdiki karede olacağı konumu.

    pts : (N,2) önceki kare piksel koordinatları.
    Döndürür: (N,2) homografi ile taşınmış (beklenen sabit) konumlar.
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(pts, H)
    return warped.reshape(-1, 2)


def artik_hareket_px(H: np.ndarray, prev_pt: np.ndarray, cur_pt: np.ndarray) -> float:
    """
    Bir nesnenin artık (ego-hareketten arındırılmış) piksel hareketi.

    Nesne sabitse cur_pt ≈ H·prev_pt olmalı; aradaki fark gerçek (dünya) harekettir.
    prev_pt, cur_pt : (2,) nesne merkezi (önceki ve şimdiki karede).
    Döndürür: artık hareketin piksel büyüklüğü (skaler).
    """
    beklenen = sabit_konum_tahmini(H, np.asarray(prev_pt).reshape(1, 2))[0]
    return float(np.linalg.norm(np.asarray(cur_pt, dtype=np.float32) - beklenen))
