"""RAFT derin optik akış tabanlı eşleme motoru (GPU).

Kavram — **RAFT (Recurrent All-Pairs Field Transforms):** iki kare arasındaki
akışı *her piksel için* kestiren, öğrenilmiş bir model. Klasik LK'ye göre
avantajları:

* **Öğrenilmiş önsel:** dokusuz bölgelerde (deniz, kar, sis) komşuluktan ve
  sahne bilgisinden yararlanarak yine de makul akış üretir — LK köşe bulamaz.
* **Büyük hareketlere dayanıklılık:** tüm-çiftler korelasyon hacmi sayesinde
  piramit kaçırmalarına uğramaz.
* **Yoğun çıktı:** grid üzerinden istediğimiz kadar eşleşme örnekleriz.

Tasarım: ``RaftTracker`` sınıfı, ``FeatureTracker`` ile aynı arayüzü sunar
(``track_pair(gray0, gray1) -> MatchResult``). Böylece geometri katmanı ve
durum makinesi hiç değişmeden yeni motoru kullanır — Aşama 2'deki modüler
tasarımın karşılığı.

Notlar:
* Model ağırlıkları ilk kullanımda internetten iner ve ``TORCH_HOME``
  önbelleğine yazılır. **Yarışma bilgisayarı çevrimdışı olacağı için**
  ağırlıkların önceden indirilip makinede bulunması zorunludur (Aşama 5
  kontrol listesinde).
* İleri-geri (cycle) kontrolü LK'deki FB elemesinin karşılığıdır: ileri akışla
  gidip geri akışla dönünce başlangıçtan sapma küçük olmalıdır.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .calibration import CameraProfile
from .matching import MatchResult


class RaftTracker:
    """RAFT ile kare çifti eşleme; ``FeatureTracker`` arayüzüyle uyumlu."""

    def __init__(
        self,
        profile: Optional[CameraProfile] = None,
        model_size: str = "large",       # "large" | "small"
        device: Optional[str] = None,
        downscale: Optional[float] = None,  # None -> genişliğe göre otomatik (hedef ~1000 px)
        grid_step: Optional[int] = 28,   # örnekleme grid aralığı; None -> hedef nokta sayısına göre
        hedef_nokta: int = 640,          # grid_step=None iken hedeflenen örnek sayısı
        on_isle: Optional[str] = None,   # None | "clahe" (termal için)
        clahe_clip: float = 2.0,
        margin: int = 16,                # kenar payı (padding bölgesinden uzak dur)
        fb_check: bool = True,           # ileri-geri (cycle) tutarlılık elemesi
        fb_threshold: float = 1.5,       # [px], işlenen çözünürlükte
        num_flow_updates: int = 12,      # RAFT yineleme sayısı (az = hızlı)
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
        self.hedef_nokta = hedef_nokta
        self.on_isle = on_isle
        self._clahe = (cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
                       if on_isle == "clahe" else None)
        self.margin = margin
        self.fb_check = fb_check
        self.fb_threshold = fb_threshold
        self.num_flow_updates = num_flow_updates
        self.min_points = min_points

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if model_size == "large":
            self.model = raft_large(weights=Raft_Large_Weights.DEFAULT)
        else:
            self.model = raft_small(weights=Raft_Small_Weights.DEFAULT)
        self.model = self.model.to(self.device).eval()

    # ------------------------------------------------------------------
    @staticmethod
    def auto_scale(width: int) -> float:
        """Genişliğe göre işlem ölçeği: hedef ~1000 px çalışma genişliği.

        640 (termal) → 1.0, 1920 (Full HD) → 0.5, 4000 (4K) → 0.25.
        Böylece girdi çözünürlüğünden bağımsız, sabit bir işlem maliyeti ve
        RAFT'ın eğitim dağılımına yakın bir görüntü boyutu elde edilir.
        """
        if width <= 1024:
            return 1.0
        if width <= 2048:
            return 0.5
        return 0.25

    def _effective_scale(self, width: int) -> float:
        return self.downscale if self.downscale is not None else self.auto_scale(width)

    def _izgara_adimi(self, h: int, w: int) -> int:
        """Örnekleme aralığı: sabit ``grid_step`` ya da hedef nokta sayısına göre.

        Sabit 28 px, işlenen çözünürlük değiştiğinde nokta sayısını da değiştirir:
        1080p (960×540 işlenir) → ~646 nokta, ama termal (640×512, ölçek 1,0)
        → ~357 nokta. Homografi gürültüsü nokta sayısıyla düştüğü için termal
        ölçüm bu yüzden yapısal olarak daha gürültülü. Hedef sayıya kilitlemek
        modaliteler arası davranışı eşitler.
        """
        if self.grid_step is not None:
            return self.grid_step
        alan = max((h - 2 * self.margin) * (w - 2 * self.margin), 1)
        return int(np.clip(round(np.sqrt(alan / max(self.hedef_nokta, 1))), 8, 40))

    def _onisle(self, gray: np.ndarray) -> np.ndarray:
        """Termal için ön-işleme: FPN bastırma + yerel kontrast sabitleme.

        Termal (LWIR) kareler mikrobolometre sabit-desen gürültüsü (FPN) taşır ve
        kamera AGC'si sahne sıcaklık aralığına göre parlaklık/kontrastı kare kare
        yeniden ölçekler; bu, optik akış için görünürlük kaymasıdır. Literatürdeki
        standart karşı önlem 3×3 Gauss (σ=1) ile FPN'i bastırıp CLAHE ile yerel
        kontrastı sabitlemektir (bkz. arXiv 2603.02114 termal-ataletsel odometri).
        """
        if self._clahe is None:
            return gray
        return self._clahe.apply(cv2.GaussianBlur(gray, (3, 3), 1.0))

    def _prepare(self, gray: np.ndarray, scale: float = None):
        """Gri kareyi RAFT girdisine çevir: ölçekle, 8'in katına doldur, normalize et."""
        img = gray
        scale = self._effective_scale(gray.shape[1]) if scale is None else scale
        if scale != 1.0:
            img = cv2.resize(
                img, None, fx=scale, fy=scale,
                interpolation=cv2.INTER_AREA,
            )
        h, w = img.shape[:2]
        pad_h = (-h) % 8
        pad_w = (-w) % 8
        if pad_h or pad_w:
            img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
        t = self.torch.from_numpy(img).float().div_(255.0).mul_(2.0).sub_(1.0)
        t = t.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)   # 1x3xHxW
        return t, (h, w)

    def _flow(self, t0, t1):
        with self.torch.no_grad():
            flows = self.model(
                t0.to(self.device), t1.to(self.device),
                num_flow_updates=self.num_flow_updates,
            )
        return flows[-1][0].cpu().numpy()                 # 2xHxW

    # ------------------------------------------------------------------
    def track_pair(self, gray0: np.ndarray, gray1: np.ndarray) -> Optional[MatchResult]:
        scale = self._effective_scale(gray0.shape[1])
        t0, (h, w) = self._prepare(self._onisle(gray0), scale)
        t1, _ = self._prepare(self._onisle(gray1), scale)

        flow_f = self._flow(t0, t1)
        # örnekleme grid'i (geçerli bölge içinde, kenarlardan uzak)
        adim = self._izgara_adimi(h, w)
        xs = np.arange(self.margin, w - self.margin, adim)
        ys = np.arange(self.margin, h - self.margin, adim)
        gx, gy = np.meshgrid(xs, ys)
        gx, gy = gx.ravel(), gy.ravel()
        n_detected = len(gx)
        if n_detected < self.min_points:
            return None

        du = flow_f[0, gy, gx]
        dv = flow_f[1, gy, gx]
        px1 = gx + du
        py1 = gy + dv

        # hedef noktalar geçerli bölgede kalsın
        gecerli = (px1 >= 0) & (px1 < w - 1) & (py1 >= 0) & (py1 < h - 1)

        fb_err = np.zeros(n_detected)
        if self.fb_check:
            flow_b = self._flow(t1, t0)
            ix = np.clip(np.rint(px1).astype(int), 0, w - 1)
            iy = np.clip(np.rint(py1).astype(int), 0, h - 1)
            bx = flow_b[0, iy, ix]
            by = flow_b[1, iy, ix]
            fb_err = np.hypot(du + bx, dv + by)
            gecerli &= fb_err < self.fb_threshold

        if gecerli.sum() < self.min_points:
            return None

        pts0 = np.column_stack([gx, gy]).astype(np.float64)[gecerli]
        pts1 = np.column_stack([px1, py1]).astype(np.float64)[gecerli]
        if scale != 1.0:
            pts0 /= scale       # native piksel koordinatlarına geri dön
            pts1 /= scale
        if self.profile is not None:
            pts0 = self.profile.undistort_points(pts0)
            pts1 = self.profile.undistort_points(pts1)

        return MatchResult(
            pts0=pts0,
            pts1=pts1,
            n_detected=n_detected,
            n_tracked=n_detected,
            n_kept=int(gecerli.sum()),
            fb_err_median=float(np.median(fb_err[gecerli])) if self.fb_check else 0.0,
        )
