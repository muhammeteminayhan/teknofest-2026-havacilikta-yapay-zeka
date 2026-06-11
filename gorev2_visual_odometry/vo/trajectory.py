"""
Yörünge birikimi + sağlık durum makinesi + yeniden çapalama.
(Faz 2 — XY ve Z ayrıştırılmış + kare-başı irtifa takibi.)

Gözlemlenebilirlik bulgusu: sağlıklı segmentte (ilk 450 kare) GT-z neredeyse
sabittir; bu yüzden 3B Umeyama z-eksenini öğrenemez ve z'yi yanlış yöne eşler.
Çözüm: hareketi YER DÜZLEMİ (XY) ve İRTİFA (Z) olarak ayır.

  XY: sağlıklı (raw_xy, gt_xy) çiftlerine 2B Umeyama (ölçek s_xy≈irtifa, dönme θ_xy).
      XY iyi gözlemli (bol yatay hareket).
  Z : irtifayı sahne ölçek-değişimi proxy'si ile takip et; dünya-z fiziksel
      işaretle irtifadan türetilir (z = z_çapa + Z_SIGN·(a − a_çapa)).

Kare-başı metrik ölçek: tek donmuş ölçek yerine irtifa a_i her kare güncellenir.
Bu, uzun sağlıksız segmentte irtifa değişiminden kaynaklı XY ölçek sıkışmasını
giderir.

Çapalama: health=1 → çapa = son sağlıklı kare (GT'ye sabitlenir). Sağlıksızda:
    XY = gt_çapa_xy + R(θ_xy)·(S_xy − S_xy_çapa)
    Z  = gt_çapa_z  + Z_SIGN·(a − a_çapa)
"""
from __future__ import annotations

import numpy as np

from .aligner import Similarity2D, umeyama_2d_ransac
from .motion import FrameMotion


class TrajectoryEstimator:
    def __init__(self, align_window: int = 450, align_min_points: int = 30,
                 align_thresh: float = 3.0, alt_prop_gain: float = 1.0,
                 alt_smooth: float = 0.8, alt_clamp=(0.3, 3.0),
                 z_sign: float = 1.0, z_gain: float = 1.0):
        self.align_window = align_window
        self.align_min_points = align_min_points
        self.align_thresh = align_thresh
        self.alt_prop_gain = alt_prop_gain
        self.alt_smooth = alt_smooth
        self.alt_clamp = alt_clamp
        self.z_sign = z_sign
        self.z_gain = z_gain

        self.phi = 0.0                    # birikmiş yaw
        self.idx = -1

        self.raw_xy = np.zeros(2)         # ölçeksiz, yaw-telafili XY (Umeyama fit için)
        self.S_xy = np.zeros(2)           # irtifa-ölçekli, yaw-telafili XY (raw VO çerçevesi)
        self.z_acc = 0.0                  # irtifa proxy'sinden biriken metrik z

        self.a = 1.0                      # güncel irtifa kestirimi
        self.a_ref = 1.0                  # son sağlıklı irtifa (clamp referansı)
        self._dz_ema = 0.0
        self._have_scale = False

        self._last_inc = np.zeros(3)
        self._last_dyaw = 0.0

        # Sağlıklı tamponlar (kayan pencere)
        self._raw_buf: list[np.ndarray] = []
        self._gtxy_buf: list[np.ndarray] = []

        self.sim_xy: Similarity2D = Similarity2D.identity()

        # Çapa
        self._anchor_Sxy = np.zeros(2)
        self._anchor_z_acc = 0.0
        self._anchor_a = 1.0
        self._anchor_gt = np.zeros(3)
        self._have_anchor = False

        self.raw_history: list[np.ndarray] = []

    # ----------------------------------------------------------------- entegrasyon
    def add_motion(self, fm: FrameMotion) -> None:
        self.idx += 1
        if self.idx == 0:
            self.raw_history.append(np.zeros(3))
            return

        if fm.ok:
            inc = np.array([fm.raw_dx, fm.raw_dy, fm.raw_dz])
            dyaw = fm.dyaw
            self._last_inc, self._last_dyaw = inc, dyaw
        else:
            inc, dyaw = self._last_inc, self._last_dyaw   # sabit-hız yedeği

        # İrtifayı sahne ölçek-değişimi proxy'si ile yay
        self._dz_ema = self.alt_smooth * self._dz_ema + (1 - self.alt_smooth) * inc[2]
        if self._have_scale:
            self.a *= float(np.exp(self.alt_prop_gain * self._dz_ema))
            lo, hi = self.alt_clamp[0] * self.a_ref, self.alt_clamp[1] * self.a_ref
            self.a = float(np.clip(self.a, lo, hi))

        # Gövde→raw VO çerçevesi (yaw telafisi)
        c, s_ = np.cos(self.phi), np.sin(self.phi)
        d_world = np.array([c * inc[0] - s_ * inc[1], s_ * inc[0] + c * inc[1]])

        self.raw_xy = self.raw_xy + d_world                 # ölçeksiz (fit için)
        self.S_xy = self.S_xy + self.a * d_world            # irtifa-ölçekli
        self.z_acc = self.z_acc + self.z_gain * self.a * inc[2]
        self.phi += dyaw

        self.raw_history.append(np.array([self.raw_xy[0], self.raw_xy[1], self.z_acc]))

    def update_healthy(self, gt_xyz: np.ndarray) -> None:
        gt_xyz = np.asarray(gt_xyz, dtype=np.float64).reshape(3)
        self._raw_buf.append(self.raw_xy.copy())
        self._gtxy_buf.append(gt_xyz[:2].copy())
        if len(self._raw_buf) > self.align_window:
            self._raw_buf.pop(0)
            self._gtxy_buf.pop(0)

        if len(self._raw_buf) >= self.align_min_points:
            src = np.array(self._raw_buf)
            dst = np.array(self._gtxy_buf)
            sim, _ = umeyama_2d_ransac(src, dst, thresh=self.align_thresh,
                                       min_inliers=self.align_min_points // 2)
            self.sim_xy = sim
            # Sağlıklıyken irtifayı robust Umeyama ölçeğine sabitle
            self.a = sim.s
            self.a_ref = sim.s
            self._have_scale = True

        # Çapa = bu (son sağlıklı) kare
        self._anchor_Sxy = self.S_xy.copy()
        self._anchor_z_acc = self.z_acc
        self._anchor_a = self.a
        self._anchor_gt = gt_xyz.copy()
        self._have_anchor = True

    def predict(self) -> np.ndarray:
        if not (self._have_anchor and self._have_scale):
            return np.array([self.raw_xy[0], self.raw_xy[1], self.z_acc])
        rel_xy = self.S_xy - self._anchor_Sxy
        xy = self._anchor_gt[:2] + self.sim_xy.R @ rel_xy
        z = self._anchor_gt[2] + self.z_sign * (self.z_acc - self._anchor_z_acc)
        return np.array([xy[0], xy[1], z])

    @property
    def scale(self) -> float:
        return self.a
