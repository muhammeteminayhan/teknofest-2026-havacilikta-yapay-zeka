"""RAFT yoğun optik akış tabanlı eşleme motoru (GPU) — **birincil motor**.

Neden RAFT: yoğun akış + ileri-geri (cycle) tutarlılık elemesi, havadan çekimdeki
tekrarlayan dokuda yanlış eşleşmeleri baştan eler. Denenen bütün seyrek
alternatifler (XFeat, DISK+LightGlue, SuperPoint+LightGlue, SIFT, ORB, AKAZE)
ölçülerek elendi; SIFT yalnız **CPU yedeği** olarak kaldı (bkz. `sift_matching`).

``FeatureTracker`` ile aynı arayüz: ``track_pair(gray0, gray1) -> MatchResult``.

**ÇEVRİMDIŞI UYARISI:** ağırlıklar ilk kullanımda internetten iner ve TORCH_HOME
önbelleğine yazılır. Yarışma bilgisayarı çevrimdışı olacağı için ağırlıkların
önceden indirilmiş olması ZORUNLUDUR — `on_ucus.py` bunu denetler.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .calibration import CameraProfile
from .matching import MatchResult


class RaftTracker:
    """RAFT ile kare çifti eşleme."""

    def __init__(
        self,
        profile: Optional[CameraProfile] = None,
        model_size: str = "large",
        device: Optional[str] = None,
        downscale: Optional[float] = None,   # None -> genişliğe göre otomatik
        grid_step: int = 28,                 # örnekleme aralığı (işlenen çözünürlükte)
        margin: int = 16,                    # kenar payı (padding bölgesinden uzak dur)
        fb_check: bool = True,               # ileri-geri tutarlılık elemesi
        fb_threshold: float = 1.5,           # [px], işlenen çözünürlükte
        num_flow_updates: int = 12,          # RAFT yineleme sayısı
        min_points: int = 12,
    ):
        import torch
        from torchvision.models.optical_flow import (
            Raft_Large_Weights, Raft_Small_Weights, raft_large, raft_small,
        )

        self.torch = torch
        self.profile = profile
        self.downscale = None if downscale is None else float(downscale)
        self.grid_step = grid_step
        self.margin = margin
        self.fb_check = fb_check
        self.fb_threshold = fb_threshold
        self.num_flow_updates = num_flow_updates
        self.min_points = min_points

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = (raft_large(weights=Raft_Large_Weights.DEFAULT)
                      if model_size == "large"
                      else raft_small(weights=Raft_Small_Weights.DEFAULT))
        self.model = self.model.to(self.device).eval()

    # ------------------------------------------------------------------
    @staticmethod
    def auto_scale(width: int) -> float:
        """Hedef ~1000 px çalışma genişliği: 640 (termal) → 1,0 · 1920 → 0,5 · 4K → 0,25.

        Böylece girdi çözünürlüğünden bağımsız sabit bir işlem maliyeti ve RAFT'ın
        eğitim dağılımına yakın bir görüntü boyutu elde edilir. Daha yüksek çalışma
        çözünürlüğü (ds 0,75) ölçülüp REDDEDİLDİ: yön hatasını iyileştiriyor ama
        ölçek hatasını kötüleştiriyor, maliyeti 2,5×.
        """
        if width <= 1024:
            return 1.0
        if width <= 2048:
            return 0.5
        return 0.25

    def _olcek(self, width: int) -> float:
        return self.downscale if self.downscale is not None else self.auto_scale(width)

    def _hazirla(self, gray: np.ndarray, scale: float):
        """Gri kareyi RAFT girdisine çevir: ölçekle, 8'in katına doldur, normalize et."""
        img = gray
        if scale != 1.0:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
        pad_h, pad_w = (-h) % 8, (-w) % 8
        if pad_h or pad_w:
            img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
        t = self.torch.from_numpy(img).float().div_(255.0).mul_(2.0).sub_(1.0)
        return t.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0), (h, w)   # 1x3xHxW

    def _akis(self, t0, t1):
        with self.torch.no_grad():
            flows = self.model(t0.to(self.device), t1.to(self.device),
                               num_flow_updates=self.num_flow_updates)
        return flows[-1][0].cpu().numpy()                            # 2xHxW

    # ------------------------------------------------------------------
    def track_pair(self, gray0: np.ndarray, gray1: np.ndarray) -> Optional[MatchResult]:
        scale = self._olcek(gray0.shape[1])
        t0, (h, w) = self._hazirla(gray0, scale)
        t1, _ = self._hazirla(gray1, scale)

        flow_f = self._akis(t0, t1)
        xs = np.arange(self.margin, w - self.margin, self.grid_step)
        ys = np.arange(self.margin, h - self.margin, self.grid_step)
        gx, gy = (a.ravel() for a in np.meshgrid(xs, ys))
        n_detected = len(gx)
        if n_detected < self.min_points:
            return None

        du, dv = flow_f[0, gy, gx], flow_f[1, gy, gx]
        px1, py1 = gx + du, gy + dv
        gecerli = (px1 >= 0) & (px1 < w - 1) & (py1 >= 0) & (py1 < h - 1)

        fb_err = np.zeros(n_detected)
        if self.fb_check:
            flow_b = self._akis(t1, t0)
            ix = np.clip(np.rint(px1).astype(int), 0, w - 1)
            iy = np.clip(np.rint(py1).astype(int), 0, h - 1)
            fb_err = np.hypot(du + flow_b[0, iy, ix], dv + flow_b[1, iy, ix])
            gecerli &= fb_err < self.fb_threshold

        if gecerli.sum() < self.min_points:
            return None

        pts0 = np.column_stack([gx, gy]).astype(np.float64)[gecerli]
        pts1 = np.column_stack([px1, py1]).astype(np.float64)[gecerli]
        if scale != 1.0:
            pts0, pts1 = pts0 / scale, pts1 / scale      # doğal piksele dön
        if self.profile is not None:
            pts0 = self.profile.undistort_points(pts0)
            pts1 = self.profile.undistort_points(pts1)

        return MatchResult(
            pts0=pts0, pts1=pts1, n_detected=n_detected, n_tracked=n_detected,
            n_kept=int(gecerli.sum()),
            fb_err_median=float(np.median(fb_err[gecerli])) if self.fb_check else 0.0,
        )
