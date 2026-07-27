"""Derin öğrenmeli nokta eşleştiriciler — `FeatureTracker` arayüzüyle uyumlu.

**Neden:** Görev 2'de kesinti hatası rastgele değil *sistematik* (yön ~2°, ölçek %3–4,5;
bkz. `gorev2/degerlendirme/OZET.md`). Kaynak, durum makinesi değil **ölçüm katmanı** —
yani kare çiftleri arasındaki nokta eşleşmelerinin kalitesi. Bu modül, mevcut RAFT
(yoğun optik akış) ve LK (klasik) motorlarına alternatif olarak modern öğrenmeli
eşleştiricileri aynı arayüzle sunar:

* **XFeat** (CVPR 2024, Apple/VeRLab) — çok hafif, hızlı; seyrek ama isabetli eşleşme.
* **DISK + LightGlue** (kornia) — LightGlue'nun resmi partner çiftlerinden; güçlü ama
  daha ağır. *(SuperPoint çıkarıcısı kornia'da bulunmadığı ve `lightglue` paketi
  PyPI'da yayınlanmadığı için, aynı ailenin DISK varyantı kullanılır.)*

Hepsi `track_pair(gray0, gray1) -> MatchResult | None` döndürür; böylece
`PairMeasurement.compute`, `evaluation.build_*_measurements` ve `nokta_onbellek.py`
hiç değişmeden çalışır.

**Çevrimdışı:** ağırlıklar ilk kullanımda indirilip `~/.cache` (torch.hub / kornia)
içine alınır; sonrasında internet gerekmez — DINOv3'te kanıtlanan yol.
"""
from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from .calibration import CameraProfile
from .matching import MatchResult


def _gri_to_rgb(gray: np.ndarray) -> np.ndarray:
    if gray.ndim == 2:
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return gray


class _DerinTemel:
    """Ortak: ölçekleme, undistort, MatchResult paketleme."""

    def __init__(self, profile: Optional[CameraProfile], downscale: Optional[float],
                 min_points: int):
        self.profile = profile
        self.downscale = downscale
        self.min_points = min_points

    @staticmethod
    def auto_scale(width: int) -> float:
        """RaftTracker ile aynı politika: hedef ~1000 px çalışma genişliği."""
        if width <= 1024:
            return 1.0
        if width <= 2048:
            return 0.5
        return 0.25

    def _olcek(self, width: int) -> float:
        return self.downscale if self.downscale is not None else self.auto_scale(width)

    def _hazirla(self, gray: np.ndarray, scale: float) -> np.ndarray:
        img = _gri_to_rgb(gray)
        if scale != 1.0:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return img

    def _paketle(self, p0: np.ndarray, p1: np.ndarray, scale: float,
                 n_detected: int) -> Optional[MatchResult]:
        if p0 is None or len(p0) < self.min_points:
            return None
        p0 = np.asarray(p0, dtype=np.float64)
        p1 = np.asarray(p1, dtype=np.float64)
        if scale != 1.0:
            p0 = p0 / scale       # native piksel koordinatlarına dön
            p1 = p1 / scale
        if self.profile is not None:
            p0 = self.profile.undistort_points(p0)
            p1 = self.profile.undistort_points(p1)
        return MatchResult(pts0=p0, pts1=p1, n_detected=int(n_detected),
                           n_tracked=len(p0), n_kept=len(p0), fb_err_median=0.0)


class XFeatTracker(_DerinTemel):
    """XFeat (hızlı öğrenmeli özellik) + kendi eşleştiricisi."""

    def __init__(self, profile: Optional[CameraProfile] = None, top_k: int = 4096,
                 downscale: Optional[float] = None, min_points: int = 12,
                 min_cossim: float = 0.82, yildiz: bool = False):
        super().__init__(profile, downscale, min_points)
        import torch

        self.torch = torch
        self.top_k = top_k
        self.min_cossim = min_cossim
        self.yildiz = yildiz          # True -> match_xfeat_star (yarı-yoğun)
        self.model = torch.hub.load("verlab/accelerated_features", "XFeat",
                                    pretrained=True, top_k=top_k, trust_repo=True)
        logging.info(f"XFeatTracker hazir (top_k={top_k}, yildiz={yildiz})")

    def track_pair(self, gray0: np.ndarray, gray1: np.ndarray) -> Optional[MatchResult]:
        scale = self._olcek(gray0.shape[1])
        i0 = self._hazirla(gray0, scale)
        i1 = self._hazirla(gray1, scale)
        try:
            if self.yildiz:
                p0, p1 = self.model.match_xfeat_star(i0, i1, top_k=self.top_k)
            else:
                p0, p1 = self.model.match_xfeat(i0, i1, top_k=self.top_k,
                                                min_cossim=self.min_cossim)
        except Exception as e:                       # tek karede hata tüm koşuyu bozmasın
            logging.warning(f"XFeat eslestirme hatasi: {e}")
            return None
        return self._paketle(p0, p1, scale, n_detected=self.top_k)


class DiskLightGlueTracker(_DerinTemel):
    """DISK özellikleri + LightGlue eşleştirme (kornia)."""

    def __init__(self, profile: Optional[CameraProfile] = None, n_keypoints: int = 2048,
                 downscale: Optional[float] = None, min_points: int = 12,
                 device: Optional[str] = None):
        super().__init__(profile, downscale, min_points)
        import torch
        import kornia.feature as KF

        self.torch = torch
        self.KF = KF
        self.n_keypoints = n_keypoints
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.disk = KF.DISK.from_pretrained("depth").to(self.device).eval()
        self.matcher = KF.LightGlueMatcher("disk").to(self.device).eval()
        logging.info(f"DiskLightGlueTracker hazir (n={n_keypoints}, {self.device})")

    def _tensor(self, img: np.ndarray):
        t = self.torch.from_numpy(img).permute(2, 0, 1).float()[None] / 255.0
        return t.to(self.device)

    def track_pair(self, gray0: np.ndarray, gray1: np.ndarray) -> Optional[MatchResult]:
        torch, KF = self.torch, self.KF
        scale = self._olcek(gray0.shape[1])
        t0 = self._tensor(self._hazirla(gray0, scale))
        t1 = self._tensor(self._hazirla(gray1, scale))
        try:
            with torch.inference_mode():
                f0, f1 = self.disk(torch.cat([t0, t1]), n=self.n_keypoints,
                                   window_size=5, score_threshold=0.0,
                                   pad_if_not_divisible=True)
                kp0, d0 = f0.keypoints, f0.descriptors
                kp1, d1 = f1.keypoints, f1.descriptors
                if len(kp0) < self.min_points or len(kp1) < self.min_points:
                    return None
                # LightGlueMatcher LAF (yerel afin çerçeve) bekler: merkez + ölçek
                lafs0 = KF.laf_from_center_scale_ori(
                    kp0[None], torch.ones(1, len(kp0), 1, 1, device=self.device))
                lafs1 = KF.laf_from_center_scale_ori(
                    kp1[None], torch.ones(1, len(kp1), 1, 1, device=self.device))
                hw = torch.tensor(t0.shape[2:], device=self.device)
                _, idxs = self.matcher(d0, d1, lafs0, lafs1, hw1=hw, hw2=hw)
        except Exception as e:
            logging.warning(f"DISK+LightGlue eslestirme hatasi: {e}")
            return None
        if idxs is None or len(idxs) < self.min_points:
            return None
        i0 = idxs[:, 0].cpu().numpy()
        i1 = idxs[:, 1].cpu().numpy()
        p0 = kp0.cpu().numpy()[i0]
        p1 = kp1.cpu().numpy()[i1]
        return self._paketle(p0, p1, scale, n_detected=int(len(kp0)))


class SuperPointLightGlueTracker(_DerinTemel):
    """**Resmî** SuperPoint + LightGlue (cvg/LightGlue reposu).

    Kullanıcının yöntem listesindeki "homografi + SuperPoint/LightGlue" hattının asıl
    karşılığı. (Daha önce kornia'da SuperPoint bulunmadığı için DISK ile ikame edilmişti;
    bu sınıf gerçek SuperPoint çıkarıcısını kullanır.)

    SuperPoint **gri** girdi ister (1 kanal); `_DerinTemel` RGB verdiği için burada
    ayrı bir gri hazırlama yolu var. Çalışma ölçeği RAFT ile aynı tutulur (0,5) ki
    karşılaştırma adil olsun.
    """

    def __init__(self, profile: Optional[CameraProfile] = None, n_keypoints: int = 2048,
                 downscale: Optional[float] = None, min_points: int = 12,
                 min_skor: float = 0.0, device: Optional[str] = None):
        super().__init__(profile, downscale, min_points)
        import torch
        from lightglue import LightGlue, SuperPoint

        self.torch = torch
        self.min_skor = min_skor      # LightGlue eşleşme güven eşiği (0 = filtre yok)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sp = SuperPoint(max_num_keypoints=n_keypoints).eval().to(self.device)
        self.lg = LightGlue(features="superpoint").eval().to(self.device)
        logging.info(f"SuperPointLightGlueTracker hazir (n={n_keypoints}, "
                     f"min_skor={min_skor}, {self.device})")

    def _gri_tensor(self, gray: np.ndarray, scale: float):
        img = gray if gray.ndim == 2 else cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
        if scale != 1.0:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        t = self.torch.from_numpy(img).float()[None, None] / 255.0
        return t.to(self.device)

    def track_pair(self, gray0: np.ndarray, gray1: np.ndarray) -> Optional[MatchResult]:
        torch = self.torch
        from lightglue.utils import rbd
        scale = self._olcek(gray0.shape[1])
        t0 = self._gri_tensor(gray0, scale)
        t1 = self._gri_tensor(gray1, scale)
        try:
            with torch.no_grad():
                f0 = self.sp.extract(t0)
                f1 = self.sp.extract(t1)
                out = self.lg({"image0": f0, "image1": f1})
        except Exception as e:
            logging.warning(f"SuperPoint+LightGlue eslestirme hatasi: {e}")
            return None
        f0_, f1_, out_ = (rbd(x) for x in (f0, f1, out))
        m = out_["matches"]
        if m is None or len(m) < self.min_points:
            return None
        if self.min_skor > 0 and "scores" in out_:
            tut = out_["scores"] >= self.min_skor
            m = m[tut]
            if len(m) < self.min_points:
                return None
        p0 = f0_["keypoints"][m[:, 0]].cpu().numpy()
        p1 = f1_["keypoints"][m[:, 1]].cpu().numpy()
        return self._paketle(p0, p1, scale, n_detected=int(len(f0_["keypoints"])))


def tracker_kur(ad: str, profile: Optional[CameraProfile] = None, **kw):
    """Fabrika: `nokta_onbellek.py` ve ölçüm üreticileri bunu kullanır."""
    ad = ad.lower()
    if ad == "xfeat":
        return XFeatTracker(profile, **kw)
    if ad in ("xfeat_star", "xfeats"):
        return XFeatTracker(profile, yildiz=True, **kw)
    if ad in ("splg", "disk", "disklightglue"):
        return DiskLightGlueTracker(profile, **kw)
    if ad in ("superpoint", "sp_lg", "spliteglue", "splg_gercek"):
        return SuperPointLightGlueTracker(profile, **kw)
    raise ValueError(f"bilinmeyen derin eslestirici: {ad}")
