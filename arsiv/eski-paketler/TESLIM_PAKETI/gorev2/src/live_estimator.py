"""Yarışma günü kullanılacak canlı pozisyon kestirici.

Prova altyapısındaki (``evaluation``) önbellekli akışın birebir canlı
karşılığı: kareler tek tek gelir, ölçümler anında üretilir ve aynı durum
makinesi (``VoLadderEstimator``) beslenir. Böylece provada ölçtüğümüz
davranış ile yarışma günü davranışı arasında kod farkı kalmaz.

Motor seçimi:
* ``engine="auto"`` → önce RAFT denenir (GPU + önceden inmiş ağırlık gerekir);
  başarısız olursa **kendiliğinden LK yedeğine düşer** ve bunu loglar.
* RAFT'ta birincil reçete: homografi + keyframe merdiveni (8 adım).
* LK'de merdiven kapatılır (kanıtlandı: LK uzun taban çizgisinde zayıf) ve
  füzyon yöntemi kullanılır.
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


def raft_weights_cached() -> bool:
    """RAFT ağırlıkları makinede önceden inmiş mi? (yarışma çevrimdışı!)"""
    torch_home = Path(os.environ.get("TORCH_HOME", Path.home() / ".cache" / "torch"))
    adaylar = list((torch_home / "hub" / "checkpoints").glob("raft_large_*.pth"))
    return len(adaylar) > 0


def pre_flight_check() -> dict:
    """Yarışma günü kontrol listesi: her koşuldan önce çağrılmalı."""
    sonuc = {}
    sonuc["opencv"] = cv2.__version__
    try:
        import torch

        sonuc["torch"] = torch.__version__
        sonuc["cuda"] = bool(torch.cuda.is_available())
        sonuc["gpu"] = torch.cuda.get_device_name(0) if sonuc["cuda"] else "-"
    except Exception as e:  # torch yoksa LK yedeği çalışır
        sonuc["torch"] = f"YOK ({type(e).__name__})"
        sonuc["cuda"] = False
        sonuc["gpu"] = "-"
    sonuc["raft_agirliklari_yerel"] = raft_weights_cached()
    sonuc["onerilen_motor"] = "raft" if (sonuc["cuda"] and sonuc["raft_agirliklari_yerel"]) else "lk"
    return sonuc


class LivePositionEstimator:
    """Kareleri sırayla alır, yarışma formatında (x, y, z) kestirimi döndürür."""

    def __init__(
        self,
        engine: str = "auto",          # "auto" | "raft" | "lk"
        config: Optional[EstimatorConfig] = None,
        spacing: int = 8,
    ):
        self.engine_istek = engine
        self.config = config
        self.spacing = spacing

        # ilk karede (çözünürlük belli olunca) kurulacak durumlar
        self.engine_used: Optional[str] = None
        self.profile = None
        self._tracker = None
        self._est: Optional[VoLadderEstimator] = None

        self._k = 0                       # kare sıra numarası (0 tabanlı)
        self._prev_gray = None
        self._kf = None                   # (ordinal, gray) — keyframe
        self._normal_cons = None          # homografi normal zincirleri
        self._normal_ladder = None
        self.frame_times: list[float] = []

    # ------------------------------------------------------------------
    def _init_on_first_frame(self, gray: np.ndarray) -> None:
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
                logging.warning(f"RAFT kurulamadi ({e}); LK yedegine geciliyor")
                self._tracker = None
        if self._tracker is None:
            self._tracker = FeatureTracker(self.profile)
            self.engine_used = "lk"

        if self.config is None:
            # TESLIM EDILEN REÇETE = puanlı oturumda (16 Tem 2026) koşan reçetenin
            # kendisi: RAFT + düzlemsel homografi + keyframe merdiveni (spacing=8),
            # EstimatorConfig varsayılanlarıyla (smooth_alpha=0.6).
            #
            # Puanlı oturum sonrası iki değişiklik denendi; ikisi de yarışmanın KENDİ
            # uçuşunda ölçülünce geri alındı (gorev2/degerlendirme/DUSEY_KANAL.md):
            #
            #  1) `z_sim_agirlik=1.0` — düşey hareketi benzerlik ölçeğinden almak.
            #     Örnek veri setinde 3,767 → 2,803 m (−%25,6); puanlı oturumun
            #     kareleriyle yapılan genelleme sınavında iki pencerede de KAYBETTİ
            #     (16,04 → 20,07 m ve 18,46 → 18,62 m; zarar tamamen z ekseninde).
            #
            #  2) `smooth_alpha=1.0` — ölçüm EMA'sını kapatmak. Örnek veri setinde
            #     kazançlıydı (4,535 → 4,343 m; rastgele takvimlerde −%21); ancak
            #     genelleme sınavında iki pencere ters yönde ve ortalamada 0.6 önde
            #     (16,71 vs 17,25 m). Ayar verisi = test oturumu videosu olduğundan
            #     o kanıt bu uçuş için bağlayıcı değil; oturumda koşan değere dönüldü.
            if self.engine_used == "raft":
                self.config = EstimatorConfig(method="homografi")
            else:
                self.config = EstimatorConfig(method="fuzyon")
        self._est = VoLadderEstimator(self.profile, self.config, spacing=self.spacing)
        logging.info(f"LivePositionEstimator: motor={self.engine_used}, "
                     f"yontem={self.config.method}, profil={self.profile.name}")

    # ------------------------------------------------------------------
    def process_frame(
        self,
        image: np.ndarray,
        gt_xyz: Optional[np.ndarray],
        healthy: bool,
    ) -> np.ndarray:
        """Bir kareyi işle; (x, y, z) kestirimini döndür.

        ``gt_xyz``: sağlıklı karede sunucudan gelen konum (float dizi);
        sağlıksızda ``None`` verilebilir. ``healthy`` sağlık bayrağıdır.
        """
        t0 = time.perf_counter()
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if self._k == 0:
            self._init_on_first_frame(gray)
            pred = self._est.update_ladder(0, None, None,
                                           gt_xyz if healthy else np.zeros(3), healthy)
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
        if self.engine_used == "raft":   # merdiven yalnız RAFT'ta (kanıtlı karar)
            meas_ladder = PairMeasurement.compute(
                self._kf[1], gray, self.profile, self._tracker, self._normal_ladder)
            if meas_ladder.hom_ok:
                self._normal_ladder = meas_ladder.hom_normal

        pred = self._est.update_ladder(k, meas_cons, meas_ladder,
                                       gt_xyz if healthy else None, healthy)

        if k % self.spacing == 0:        # sabit ızgara: keyframe'i taşı
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
