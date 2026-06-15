"""
Loop closure (döngü kapama) — çapasız kuyruk için mutlak düzeltme.

Gözlem (örnek veride doğrulandı): hava aracının yörüngesi, GT-çapalı sağlıklı prefix
bölgesini sıkça yeniden ziyaret eder (kuyruğun ~%20'si prefix'e <50m). Saf ölü-hesap
kuyrukta ~180m drift biriktirir; ama bir kuyruk karesini, AYNI yeri gören bir prefix
anahtar-karesine görsel olarak eşleyebilirsek, o prefix karesinin GT'si kuyruk karesi
için **sanal mutlak çapa** olur → drift kırılır.

Sıralı VO drift'ini öğrenilen ön-yüz düşürmedi (ampirik); ama loop closure'ın çekirdek
adımı olan **geniş-bazlı revisit eşleme** tam da öğrenilen eşleştirmenin (DISK+LightGlue)
güçlü olduğu yerdir. Burada kullanılır.

Akış:
1. Prefix (sağlıklı) karelerden anahtar-kare veritabanı (DISK kps/descs + GT).
2. Kuyruk (sağlıksız) sorgu karelerini DB ile eşle (LightGlue) + homografi RANSAC doğrula.
3. Yeterli inlier'lı en iyi eşleşme → sanal çapa (eşleşen DB karesinin GT'si).
4. Sanal çapalar smoother'a düşük-orta ağırlıkla verilir (optimize.smooth_trajectory).

Ağır bağımlılıklar (torch/kornia) tembel yüklenir.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class _KF:
    """Anahtar-kare: işlenmiş-kare indeksi, kps(N,2), descs(N,D), (h,w), gt(3,)."""
    idx: int
    kps: "object"      # torch tensor (cpu)
    descs: "object"    # torch tensor (cpu)
    hw: Tuple[int, int]
    gt: Optional[np.ndarray]


@dataclass
class LoopAnchor:
    """Sanal çapa: işlenmiş-kare indeksi, mutlak konum (3,), güven ağırlığı."""
    idx: int
    xyz: np.ndarray
    weight: float
    n_inlier: int


class LoopCloser:
    """DISK + LightGlue ile geniş-bazlı revisit eşleme ve sanal çapa üretimi."""

    def __init__(self, extractor: str = "disk", max_features: int = 2048,
                 device: Optional[str] = None, proc_max_width: int = 1280):
        import torch
        import kornia.feature as KF
        self._torch = torch
        self._KF = KF
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.max_features = max_features
        self.proc_max_width = proc_max_width
        if extractor == "aliked":
            self.extractor = KF.ALIKED(model_name="aliked-n16",
                                       max_num_keypoints=max_features).to(device).eval()
        else:
            self.extractor = KF.DISK.from_pretrained("depth").to(device).eval()
        self.extractor_name = extractor
        self.matcher = KF.LightGlueMatcher(extractor).to(device).eval()

    # --- özellik çıkarımı ---------------------------------------------------
    def _prep(self, frame_bgr: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        if self.proc_max_width and w > self.proc_max_width:
            s = self.proc_max_width / w
            frame_bgr = cv2.resize(frame_bgr, (self.proc_max_width, int(round(h * s))),
                                   interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if frame_bgr.ndim == 3 else frame_bgr
        return gray

    def extract(self, frame_bgr: np.ndarray):
        t = self._torch
        gray = self._prep(frame_bgr)
        img = np.repeat(gray[:, :, None], 3, axis=2)
        ten = t.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)[None].float().to(self.device) / 255.0
        hw = (ten.shape[-2], ten.shape[-1])
        with t.inference_mode():
            if self.extractor_name == "aliked":
                feats = self.extractor(ten)[0]
            else:
                feats = self.extractor(ten, n=self.max_features, pad_if_not_divisible=True)[0]
            kps, descs = feats.keypoints, feats.descriptors
        return kps.cpu(), descs.cpu(), hw

    # --- eşleştirme ---------------------------------------------------------
    def _match_inliers(self, a: _KF, b: _KF, ransac_thresh: float = 5.0) -> Tuple[int, Optional[np.ndarray]]:
        t, KF = self._torch, self._KF
        if a.kps.shape[0] < 8 or b.kps.shape[0] < 8:
            return 0, None
        dev = self.device
        with t.inference_mode():
            laf_a = KF.laf_from_center_scale_ori(a.kps[None].to(dev))
            laf_b = KF.laf_from_center_scale_ori(b.kps[None].to(dev))
            _, idxs = self.matcher(a.descs.to(dev), b.descs.to(dev), laf_a, laf_b, a.hw, b.hw)
        if idxs.shape[0] < 8:
            return int(idxs.shape[0]), None
        idxs = idxs.cpu().numpy()
        pa = a.kps.numpy()[idxs[:, 0]]
        pb = b.kps.numpy()[idxs[:, 1]]
        H, mask = cv2.findHomography(pa, pb, cv2.RANSAC, ransac_thresh)
        if H is None or mask is None:
            return 0, None
        return int(mask.sum()), H

    def detect(self, db: List[_KF], queries: List[_KF],
               min_inliers: int = 60, weight: float = 0.4) -> List[LoopAnchor]:
        """
        Her sorgu karesini DB ile eşle; en çok inlier'lı (≥min_inliers) DB karesinin
        GT'sini sanal çapa yap. Döndürür: LoopAnchor listesi.
        """
        anchors: List[LoopAnchor] = []
        for q in queries:
            best_n, best_kf = 0, None
            for d in db:
                n, _ = self._match_inliers(q, d)
                if n > best_n:
                    best_n, best_kf = n, d
            if best_kf is not None and best_n >= min_inliers and best_kf.gt is not None:
                anchors.append(LoopAnchor(idx=q.idx, xyz=best_kf.gt.copy(),
                                          weight=weight, n_inlier=best_n))
        return anchors


def build_keyframes(closer: LoopCloser, frames_iter, healths: np.ndarray,
                    gts: np.ndarray, db_stride: int = 5, query_stride: int = 12,
                    progress=None) -> Tuple[List[_KF], List[_KF]]:
    """
    Tek video geçişinde DB (sağlıklı prefix) ve sorgu (sağlıksız) anahtar-karelerini kur.

    frames_iter : (islenmis_idx j, frame_bgr) üreten iterator (evaluate ile aynı sıra).
    healths, gts: işlenmiş-kare hizasında diziler.
    """
    db: List[_KF] = []
    queries: List[_KF] = []
    for j, frame in frames_iter:
        if j >= len(healths):
            break
        is_h = bool(healths[j])
        if is_h and (j % db_stride == 0):
            kps, descs, hw = closer.extract(frame)
            db.append(_KF(j, kps, descs, hw, gts[j].copy()))
        elif (not is_h) and (j % query_stride == 0):
            kps, descs, hw = closer.extract(frame)
            queries.append(_KF(j, kps, descs, hw, None))
        if progress:
            progress(j)
    return db, queries
