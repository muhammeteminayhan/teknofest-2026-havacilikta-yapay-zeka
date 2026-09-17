"""Yarışma günü kullanılacak canlı pozisyon kestirici — **giriş noktası**.

Kareler tek tek gelir, ölçümler anında üretilir ve durum makinesi
(``VoLadderEstimator``) beslenir.

Motor zinciri: **RAFT → SIFT → LK**
    * ``engine="auto"`` → önce RAFT denenir (GPU + önceden inmiş ağırlık gerekir);
      kurulamazsa SIFT'e, o da olmazsa LK'ye düşer ve her düşüş loglanır.
    * ``engine="raft"|"sift"|"lk"`` → doğrudan o motor (zincire düşülmez).

Reçete: RAFT ve SIFT'te **homografi + keyframe merdiveni (spacing=8)**.
Yalnız son-çare LK yolunda füzyon/merdivensiz reçete korunur — LK uzun taban
çizgisinde zayıftır.

CPU yedeği neden SIFT (eskiden LK) — üç yatakta ortalama hata ölçüldü:
LK füzyon+merdivensiz **15,86 m** · LK homografi+merdiven 12,36 m ·
**SIFT homografi+merdiven 9,21 m** · (kıyas) RAFT/GPU 9,72 m.

Kullanım::

    est = LivePositionEstimator(engine="auto")
    for image, gt, healthy in akis:
        x, y, z = est.process_frame(image, gt, healthy)
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .calibration import profile_for_frame
from .matching import FeatureTracker
from .vo_pipeline import EstimatorConfig, PairMeasurement, VoLadderEstimator


def raft_agirliklari_yerel() -> bool:
    """RAFT ağırlıkları makinede önceden inmiş mi? (yarışma çevrimdışı!)"""
    torch_home = Path(os.environ.get("TORCH_HOME", Path.home() / ".cache" / "torch"))
    return len(list((torch_home / "hub" / "checkpoints").glob("raft_large_*.pth"))) > 0


def on_ucus_kontrolu() -> dict:
    """Yarışma günü kontrol listesi: her koşudan önce çağrılmalı."""
    sonuc = {"opencv": cv2.__version__}
    try:
        import torch

        sonuc["torch"] = torch.__version__
        sonuc["cuda"] = bool(torch.cuda.is_available())
        sonuc["gpu"] = torch.cuda.get_device_name(0) if sonuc["cuda"] else "-"
    except Exception as e:                      # torch yoksa SIFT yedeği çalışır
        sonuc["torch"] = f"YOK ({type(e).__name__})"
        sonuc["cuda"] = False
        sonuc["gpu"] = "-"
    sonuc["raft_agirligi_yerel"] = raft_agirliklari_yerel()
    sonuc["sift_var"] = hasattr(cv2, "SIFT_create")
    sonuc["onerilen_motor"] = (
        "raft" if (sonuc["cuda"] and sonuc["raft_agirligi_yerel"])
        else ("sift" if sonuc["sift_var"] else "lk"))
    return sonuc


class LivePositionEstimator:
    """Kareleri sırayla alır, yarışma formatında (x, y, z) kestirimi döndürür."""

    def __init__(self, engine: str = "auto",
                 config: Optional[EstimatorConfig] = None, spacing: int = 8):
        self.engine_istek = engine
        self.config = config
        self.spacing = spacing

        self.engine_used: Optional[str] = None   # ilk karede (çözünürlük belli olunca)
        self.profile = None
        self._tracker = None
        self._est: Optional[VoLadderEstimator] = None

        self._k = 0                              # kare sıra numarası (0 tabanlı)
        self._prev_gray = None
        self._kf = None                          # (ordinal, gray) — keyframe
        self._normal_cons = None                 # homografi normal zincirleri
        self._normal_ladder = None
        self.frame_times: list[float] = []

    # ------------------------------------------------------------------
    def _ilk_karede_kur(self, gray: np.ndarray) -> None:
        self.profile = profile_for_frame(gray)
        motor = self.engine_istek

        if motor in ("auto", "raft"):
            try:
                from .raft_matching import RaftTracker

                self._tracker = RaftTracker(self.profile, model_size="large")
                # kurulumun gerçekten çalıştığını küçük bir koşuyla doğrula
                self._tracker.track_pair(gray[::4, ::4], gray[::4, ::4])
                self.engine_used = "raft"
            except Exception as e:
                logging.warning(f"RAFT kurulamadi ({e}); SIFT yedegine geciliyor")
                self._tracker = None

        if self._tracker is None and motor == "lk":
            self._tracker = FeatureTracker(self.profile)      # açık istek
            self.engine_used = "lk"

        if self._tracker is None:
            try:
                from .sift_matching import SiftTracker

                self._tracker = SiftTracker(self.profile)
                self._tracker.track_pair(gray[::4, ::4], gray[::4, ::4])
                self.engine_used = "sift"
            except Exception as e:
                logging.warning(f"SIFT kurulamadi ({e}); LK son caresine geciliyor")
                self._tracker = FeatureTracker(self.profile)
                self.engine_used = "lk"

        if self.config is None:
            self.config = EstimatorConfig(
                method="fuzyon" if self.engine_used == "lk" else "homografi")
        self._est = VoLadderEstimator(self.profile, self.config, spacing=self.spacing)
        logging.info(f"LivePositionEstimator: motor={self.engine_used}, "
                     f"yontem={self.config.method}, profil={self.profile.name}")

    # ------------------------------------------------------------------
    def process_frame(self, image: np.ndarray,
                      gt_xyz: Optional[np.ndarray], healthy: bool) -> np.ndarray:
        """Bir kareyi işle; (x, y, z) kestirimini döndür.

        ``gt_xyz``: sağlıklı karede sunucudan gelen konum; sağlıksızda None olabilir.
        """
        t0 = time.perf_counter()
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if self._k == 0:
            self._ilk_karede_kur(gray)
            pred = self._est.update_ladder(
                0, None, None, gt_xyz if healthy else np.zeros(3), healthy)
            self._prev_gray = gray
            self._kf = (0, gray)
            self._k = 1
            self.frame_times.append(time.perf_counter() - t0)
            return pred

        k = self._k
        meas_cons = PairMeasurement.compute(
            self._prev_gray, gray, self.profile, self._tracker, self._normal_cons)
        if meas_cons.hom_ok:
            self._normal_cons = meas_cons.hom_normal

        meas_ladder = None
        if self.engine_used in ("raft", "sift"):   # merdiven: LK'de kapalı (kanıtlı)
            meas_ladder = PairMeasurement.compute(
                self._kf[1], gray, self.profile, self._tracker, self._normal_ladder)
            if meas_ladder.hom_ok:
                self._normal_ladder = meas_ladder.hom_normal

        pred = self._est.update_ladder(k, meas_cons, meas_ladder,
                                       gt_xyz if healthy else None, healthy)

        if k % self.spacing == 0:                  # sabit ızgara: keyframe'i taşı
            self._kf = (k, gray)
        self._prev_gray = gray
        self._k += 1
        self.frame_times.append(time.perf_counter() - t0)
        return pred

    # ------------------------------------------------------------------
    @property
    def timing_stats(self) -> dict:
        t = np.array(self.frame_times)
        if len(t) == 0:
            return {}
        return dict(n=len(t), ort_ms=float(t.mean() * 1000),
                    p95_ms=float(np.percentile(t, 95) * 1000),
                    maks_ms=float(t.max() * 1000))
