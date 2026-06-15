"""
Yoğun harita-tabanlı MUTLAK konumlandırma (AVL / DSMAC-tarzı) — Görev 2 SOTA.

Sektör (Baykar vb.) GPS'siz mutlak konum için kamera görüntüsünü geo-referanslı bir
haritaya eşler (Absolute Visual Localization / Digital Scene Matching). Burada referans
harita = **GT'li sağlıklı kareler** (uydu/IMU yok; konum ilk kareye göreli). HER sağlıksız
kareyi bu haritaya konumlandırırız → seyrek loop closure'ın yerine geçen, çok daha sağlam
ve drift'siz mutlak düzeltme.

Geometri (nadir-bakış, düzlemsel yer): anahtar-kare k'nin dünya konumu G_k (GT) ve dünya
yönü ψ_k = θ_align + φ_k bilinir. Query q ile k eşleşince homografi H (q→k) elde edilir;
q'nun merkezi (nadir) k'de H·c noktasına düşer; k merkezine göre piksel ofseti d_px,
metre/piksel (mpp) ve ψ_k ile dünya ofsetine çevrilir:  G_q = G_k + mpp·R(ψ_k)·(flip·d_px).

İşaret/eksen konvansiyonu (flip) ve mpp, **sağlıklı kare çiftlerinden otomatik kalibre
edilir** (bilinen GT ofsetlerini en iyi veren konvansiyon seçilir) → analitik işaret hatası
riski elenir ve kalibrasyon artığı bu veride yöntemin güvenilir olup olmadığını gösterir.

Çok-anahtar oylama (medyan) + inlier eşiği tekrarlayan dokuda (tarla/su) sahte eşleşmeyi eler.
Ağır bağımlılıklar tembel (LoopCloser üzerinden).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .loop_closure import LoopCloser, LoopAnchor, _KF


@dataclass
class MapKF:
    """Geo-referanslı anahtar-kare: özellik + dünya konumu + dünya yönü."""
    kf: _KF                  # LoopCloser._KF (idx, kps, descs, hw, gt)
    heading: float           # dünya yönü ψ_k = θ_align + φ_k (rad)

    @property
    def G(self) -> np.ndarray:
        return self.kf.gt


def _rot(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def _center(hw) -> np.ndarray:
    return np.array([hw[1] / 2.0, hw[0] / 2.0], dtype=np.float64)   # (cx, cy)


class MapLocalizer:
    """GT'li sağlıklı karelerden harita kurar; sağlıksız kareleri mutlak konumlandırır."""

    def __init__(self, closer: LoopCloser, mpp0: float,
                 min_inliers: int = 40, vote_min: int = 1):
        self.closer = closer
        self.mpp = float(mpp0)                  # metre/piksel (s/fx ön-tahmini; kalibre edilir)
        self.min_inliers = min_inliers
        self.vote_min = vote_min
        self.flip = np.array([1.0, -1.0])       # eksen konvansiyonu (kalibre edilir)
        self.calib_resid = None                 # kalibrasyon artığı (m); None=kalibre edilmedi

    # --- bir query → bir anahtar-kareden mutlak konum tahmini --------------
    def _predict_from(self, q_kf: _KF, m: MapKF, H: np.ndarray) -> Optional[np.ndarray]:
        c_q = _center(q_kf.hw).reshape(1, 1, 2).astype(np.float32)
        c_in_k = cv2.perspectiveTransform(c_q, H).reshape(2)
        d_px = c_in_k - _center(m.kf.hw)
        off = self.mpp * (_rot(m.heading) @ (self.flip * d_px))
        return m.G[:2] + off

    # --- konvansiyon + mpp otomatik kalibrasyonu (sağlıklı çiftlerden) ------
    def calibrate(self, map_kfs: List[MapKF], max_pairs: int = 120) -> Optional[float]:
        """
        Sağlıklı anahtar-kare çiftlerini eşleyip bilinen GT ofsetlerinden flip + mpp seç.
        Döndürür: ortalama kalibrasyon artığı (m) veya None (yetersiz eşleşme).
        """
        samples = []   # (d_px, true_off(2,), heading)
        n = len(map_kfs)
        for a in range(n):
            for b in range(a + 1, min(n, a + 6)):    # yakın (komşu) çiftler
                ka, kb = map_kfs[a], map_kfs[b]
                if np.linalg.norm(ka.G[:2] - kb.G[:2]) > 80:
                    continue
                n_in, H = self.closer._match_inliers(ka.kf, kb.kf)
                if n_in < self.min_inliers or H is None:
                    continue
                c_a = _center(ka.kf.hw).reshape(1, 1, 2).astype(np.float32)
                c_in_b = cv2.perspectiveTransform(c_a, H).reshape(2)
                d_px = c_in_b - _center(kb.kf.hw)
                samples.append((d_px, (ka.G[:2] - kb.G[:2]), kb.heading))
                if len(samples) >= max_pairs:
                    break
            if len(samples) >= max_pairs:
                break
        if len(samples) < 5:
            return None

        best = None
        for fx in (1.0, -1.0):
            for fy in (1.0, -1.0):
                flip = np.array([fx, fy])
                # mpp'yi bu flip için en iyi ölçek olarak çöz (en küçük kareler oran)
                pred_dirs, trues = [], []
                for d_px, true_off, hd in samples:
                    pred_dirs.append(_rot(hd) @ (flip * d_px))
                    trues.append(true_off)
                P = np.array(pred_dirs); T = np.array(trues)
                denom = float((P * P).sum())
                if denom < 1e-9:
                    continue
                mpp = float((P * T).sum() / denom)        # skaler en-küçük-kareler ölçek
                resid = float(np.sqrt(((mpp * P - T) ** 2).sum(axis=1)).mean())
                if mpp > 0 and (best is None or resid < best[0]):
                    best = (resid, flip, mpp)
        if best is None:
            return None
        self.calib_resid, self.flip, self.mpp = best
        return best[0]

    # --- bir query'yi adaylara karşı konumlandır (oylama) -------------------
    def localize(self, q_kf: _KF, candidates: List[MapKF]) -> Optional[LoopAnchor]:
        ests, weights, zs, ninl_tot = [], [], [], 0
        for m in candidates:
            n_in, H = self.closer._match_inliers(q_kf, m.kf)
            if n_in < self.min_inliers or H is None:
                continue
            g = self._predict_from(q_kf, m, H)
            if g is None or not np.all(np.isfinite(g)):
                continue
            ests.append(g); weights.append(n_in); zs.append(m.G[2]); ninl_tot += n_in
        if len(ests) < self.vote_min:
            return None
        E = np.array(ests)
        # Medyan oylama (sahte eşleşmeye dayanıklı) + aykırı ele (medyandan uzakları at)
        med = np.median(E, axis=0)
        d = np.linalg.norm(E - med, axis=1)
        keep = d <= max(15.0, np.median(d) * 2.5)
        if keep.sum() >= 1:
            E, zs2 = E[keep], np.array(zs)[keep]
        else:
            zs2 = np.array(zs)
        xy = np.median(E, axis=0)
        z = float(np.median(zs2))
        return LoopAnchor(idx=q_kf.idx, xyz=np.array([xy[0], xy[1], z]),
                          weight=0.0, n_inlier=int(ninl_tot))   # ağırlık çağıran tarafça


def build_map(closer: LoopCloser, frames_iter, healths: np.ndarray, gts: np.ndarray,
              phi: np.ndarray, theta_align: float, db_stride: int = 6,
              query_stride: int = 3, progress=None) -> Tuple[List[MapKF], List[_KF]]:
    """
    Tek video geçişinde harita (sağlıklı, GT'li) ve sorgu (sağlıksız) anahtar-karelerini kur.
    heading = θ_align + φ[idx] (dünya yönü). Döndürür (map_kfs, queries).
    """
    map_kfs: List[MapKF] = []
    queries: List[_KF] = []
    for j, frame in frames_iter:
        if j >= len(healths):
            break
        if bool(healths[j]) and (j % db_stride == 0):
            kps, descs, hw = closer.extract(frame)
            kf = _KF(j, kps, descs, hw, gts[j].copy())
            map_kfs.append(MapKF(kf, float(theta_align + phi[j])))
        elif (not bool(healths[j])) and (j % query_stride == 0):
            kps, descs, hw = closer.extract(frame)
            queries.append(_KF(j, kps, descs, hw, None))
        if progress:
            progress(j)
    return map_kfs, queries


def localize_all(localizer: MapLocalizer, map_kfs: List[MapKF], queries: List[_KF],
                 pred_prior: Optional[np.ndarray] = None, radius: float = 0.0,
                 max_cand: int = 8) -> List[LoopAnchor]:
    """
    Tüm sorguları konumlandır. retrieval: pred_prior+radius verilirse uzamsal kapılama
    (yakın ≤max_cand anahtar-kare); yoksa/az aday ise tüm haritaya düş (robust).
    """
    map_xy = np.array([m.G[:2] for m in map_kfs])
    anchors: List[LoopAnchor] = []
    for q in queries:
        if pred_prior is not None and radius > 0:
            d = np.linalg.norm(map_xy - pred_prior[q.idx, :2], axis=1)
            order = np.argsort(d)
            cand_idx = [i for i in order[:max_cand] if d[i] <= radius]
            if len(cand_idx) < 2:                      # az aday → tüm haritaya düş
                cand_idx = list(order[:max_cand])
            cands = [map_kfs[i] for i in cand_idx]
        else:
            cands = map_kfs
        a = localizer.localize(q, cands)
        if a is not None:
            anchors.append(a)
    return anchors
