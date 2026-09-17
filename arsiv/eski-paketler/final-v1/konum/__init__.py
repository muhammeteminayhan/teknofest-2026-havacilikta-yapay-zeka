"""TEKNOFEST 2026 Havacılıkta YZ — Görev 2 (Pozisyon Kestirimi) · final paketi.

Kullanım::

    from konum import LivePositionEstimator, on_ucus_kontrolu

    print(on_ucus_kontrolu())
    est = LivePositionEstimator(engine="auto")
    for image, gt, healthy in akis:
        x, y, z = est.process_frame(image, gt, healthy)

Sağlıklı karede sunucunun gönderdiği konum aynen geri gönderilir (hata 0);
kesintide görsel ölçümle dead reckoning yapılır.
"""
from .calibration import (CameraProfile, PROFILES, RGB_1080P, RGB_4K,
                          RGB_4K_VIDEO, THERMAL, profile_for_frame, profile_for_size)
from .live_estimator import (LivePositionEstimator, on_ucus_kontrolu,
                             raft_agirliklari_yerel)
from .vo_pipeline import (EstimatorConfig, PairMeasurement, VoLadderEstimator,
                          VoPositionEstimator)

__all__ = [
    "LivePositionEstimator", "on_ucus_kontrolu", "raft_agirliklari_yerel",
    "EstimatorConfig", "PairMeasurement", "VoLadderEstimator", "VoPositionEstimator",
    "CameraProfile", "PROFILES", "THERMAL", "RGB_1080P", "RGB_4K", "RGB_4K_VIDEO",
    "profile_for_frame", "profile_for_size",
]
__version__ = "final-v1"
