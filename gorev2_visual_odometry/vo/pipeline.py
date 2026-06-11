"""
VOEstimator — tüm VO hattını birleştiren tek giriş noktası.

process(image, health_status, gt_xyz) -> np.array([x, y, z])

Bu sınıf hem çevrimdışı değerlendirmede (scripts/evaluate.py) hem de yarışma
sunucusu entegrasyonunda (integration/object_detection_model.py) kullanılır.
Durum (state) tutar; kareler SIRA ile verilmelidir.

Sağlık (health) kuralı (şartname / arayüz koduyla uyumlu):
  '1' (sağlıklı) → GT pozisyon mevcut; çıktı olarak GT echo edilir ve hizalama
                   için estimator'a beslenir.
  '0' (sağlıksız) → kendi kestirimimiz döndürülür.
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs"))
import config  # noqa: E402

from .calibration import CameraModel, parse_calibration_file, intrinsics_for_resolution
from .frontend import make_frontend, preprocess
from .motion import estimate_motion
from .trajectory import TrajectoryEstimator


def _is_healthy(health_status) -> bool:
    """health=1 → sağlıklı (GT var). '0'/0 → sağlıksız."""
    return str(health_status).strip() == "1"


class VOEstimator:
    def __init__(self, camera: CameraModel, proc_size: tuple[int, int],
                 *, modality: str = "rgb", undistort: bool = True,
                 feature_backend: str | None = None, motion_model: str | None = None,
                 use_clahe: bool | None = None):
        self.camera = camera
        self.proc_w, self.proc_h = proc_size
        self.undistort = undistort
        self.modality = modality
        self.motion_model = motion_model or config.MOTION_MODEL
        self.use_clahe = config.USE_CLAHE if use_clahe is None else use_clahe

        backend = feature_backend or config.FEATURE_BACKEND
        self.frontend = make_frontend(
            backend, max_features=config.MAX_FEATURES,
            quality=config.GFTT_QUALITY, min_distance=config.GFTT_MIN_DISTANCE,
            min_tracks=config.KLT_MIN_TRACKS,
        )
        self.traj = TrajectoryEstimator(
            align_window=config.ALIGN_WINDOW,
            align_min_points=config.ALIGN_MIN_POINTS,
            alt_prop_gain=config.ALT_PROP_GAIN,
            alt_smooth=config.ALT_SMOOTH,
            alt_clamp=(config.ALT_CLAMP_LO, config.ALT_CLAMP_HI),
            z_sign=config.Z_SIGN,
            z_gain=config.Z_GAIN,
        )
        # Hareket kestiriminde kullanılacak K (undistort sonrası new_K, değilse K)
        self.K = camera.new_K if undistort else camera.K

        # Kayıtlar
        self.predictions: list[np.ndarray] = []
        self.healths: list[bool] = []
        self.inlier_counts: list[int] = []

    # --------------------------------------------------- kare-hareketi (paylaşılan)
    def compute_frame_motion(self, image: np.ndarray):
        """
        Bir kareden göreli hareketi (FrameMotion) hesapla. Hem çevrimiçi `process`
        hem de batch (vo/batch.py) bunu kullanır — ön-işleme/ön-yüz/hareket aynıdır.
        """
        # 1) İşleme çözünürlüğüne getir
        if image.shape[1] != self.proc_w or image.shape[0] != self.proc_h:
            image = cv2.resize(image, (self.proc_w, self.proc_h), interpolation=cv2.INTER_AREA)
        # 2) Undistort (opsiyonel)
        if self.undistort:
            image = self.camera.undistort_image(image)
        # 3) Gri + (ops.) CLAHE
        gray = preprocess(image, use_clahe=self.use_clahe,
                          clip=config.CLAHE_CLIP, grid=config.CLAHE_GRID)
        # 4) Özellik eşleştirme
        p_prev, p_cur = self.frontend.process(gray)
        # 5) Göreli hareket
        return estimate_motion(p_prev, p_cur, self.K, model=self.motion_model,
                               ransac_thresh=config.RANSAC_REPROJ_THRESH,
                               min_inliers=config.MIN_INLIERS)

    # ----------------------------------------------------------------- ana API
    def process(self, image: np.ndarray, health_status, gt_xyz=None) -> np.ndarray:
        fm = self.compute_frame_motion(image)
        # Entegre et
        self.traj.add_motion(fm)

        healthy = _is_healthy(health_status)
        if healthy and gt_xyz is not None:
            gt = np.array([float(gt_xyz[0]), float(gt_xyz[1]), float(gt_xyz[2])])
            self.traj.update_healthy(gt)
            out = gt                      # echo: sağlıklıyken en iyi skor GT'nin kendisi
        else:
            out = self.traj.predict()

        self.predictions.append(out.copy())
        self.healths.append(healthy)
        self.inlier_counts.append(fm.n_inliers)
        return out


def build_estimator(modality: str, full_w: int, full_h: int,
                    proc_max_width: int | None = None, **kw) -> VOEstimator:
    """
    Verilen tam çözünürlük için uygun kalibrasyonu seçip (gerekiyorsa işleme
    çözünürlüğüne ölçekleyerek) bir VOEstimator kur.
    """
    if proc_max_width is None:
        proc_max_width = config.PROC_MAX_WIDTH
    cams = parse_calibration_file()

    if proc_max_width and full_w > proc_max_width:
        proc_w = proc_max_width
        proc_h = int(round(full_h * proc_w / full_w))
    else:
        proc_w, proc_h = full_w, full_h

    intr_proc = intrinsics_for_resolution(cams, proc_w, proc_h, modality)
    camera = CameraModel(intr_proc)
    return VOEstimator(camera, (proc_w, proc_h), modality=modality, **kw)
