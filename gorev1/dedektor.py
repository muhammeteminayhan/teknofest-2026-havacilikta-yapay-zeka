"""Görev 1 — Nesne Tespiti: durumlu (stateful) per-frame dedektör.

Kendi eğittiğimiz **DINOv3 gövdeli YOLO** modeli (`agirliklar/best.pt`) ile taşıt,
insan ve iniş alanı (UAP / UAİ) tespiti. `model.track` (BoT-SORT) ile takip,
ego-hareket telafili taşıt hareket durumu ve iniş uygunluğu üretilir.

Arkadaşın `teknofest.py` script'i motion_status'u sabit 0 bırakmış (TODO) ve
sınıf takasını atlamış; burada ikisi de tamamlanır.

Sınıf uzlaşımı (model ↔ şartname):
    model.names = {0:Taşıt, 1:İnsan, 2:UAİ, 3:UAP}
    şartname     = {0:Taşıt, 1:İnsan, 2:UAP, 3:UAİ}   → 2↔3 takas edilir.

Çıktı: her nesne için şartname şemasında sözlük (str durumlar, int koordinatlar);
`sistem` tarafında `DetectedObject`e çevrilir.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from . import BEST_PT


def _dinov3_enjekte() -> None:
    """DINOv3 gövdeli ağırlığı **kurulu** ultralytics ile yüklenebilir kıl.

    Ağırlık `ultralytics.nn.modules.dinov3.DINOv3` sınıfına pickle bağımlıdır; bu modül
    yalnız DINOV3-YOLO fork'unda (8.4.80) bulunur. Fork'u kurmak kurulu sürümü
    (8.4.92) geriye alır ve Görev 3'ü de riske atardı. Bunun yerine modülün proje
    içindeki kopyası `sys.modules`'e enjekte edilir — pickle sınıfı bulur, kurulu
    ultralytics hiç değişmez. Ölçüldü: bu yolla model fork olmadan yükleniyor.
    """
    ad = "ultralytics.nn.modules.dinov3"
    if ad in sys.modules:
        return
    from . import dinov3_modul
    sys.modules[ad] = dinov3_modul
    import ultralytics.nn.modules as _M
    import ultralytics.nn.tasks as _T
    _M.DINOv3 = _T.DINOv3 = dinov3_modul.DINOv3


def _calculate_homography(prev_frame: np.ndarray, curr_frame: np.ndarray) -> np.ndarray:
    """Kamera ego-hareketini telafi eden homografi (optik akış + RANSAC)."""
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01,
                                       minDistance=30)
    if prev_pts is None or len(prev_pts) < 4:
        return np.eye(3)
    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
    good_prev = prev_pts[status == 1]
    good_curr = curr_pts[status == 1]
    if len(good_prev) < 4:
        return np.eye(3)
    H, _ = cv2.findHomography(good_prev, good_curr, cv2.RANSAC, 3.0)
    return H if H is not None else np.eye(3)


def _check_landing_status(landing_box, obstacle_boxes, frame_w, frame_h) -> int:
    """İniş alanı uygun mu? Kare kenarına taşıyor ya da bir engel (taşıt/insan)
    ile örtüşüyorsa → 0 (uygun değil), aksi halde 1 (uygun)."""
    lx1, ly1, lx2, ly2 = landing_box
    if lx1 <= 0 or ly1 <= 0 or lx2 >= frame_w or ly2 >= frame_h:
        return 0
    for ox1, oy1, ox2, oy2 in obstacle_boxes:
        if not (ox2 < lx1 or ox1 > lx2 or oy2 < ly1 or oy1 > ly2):
            return 0
    return 1


class Gorev1Detector:
    """DINOv3 gövdeli YOLO + BoT-SORT ile per-frame nesne tespiti (fork gerektirmez).

    Tespit katmanı dışındaki her şey önceki vanilla YOLOv8s kurulumundan aynen devralındı:
    ego-hareket telafili hareket durumu, iniş uygunluğu, takip, UAP/UAİ sınıf takası.
    """

    # imgsz=640: DINOv3 gövdeli model 640'ta EGITILDI (checkpoint train_args) ve ViT
    # tabanli govdeler egitim cozunurlugunden uzaklasinca bozulur. A/B ile dogrulandi
    # (ab_model.py, 74 gercek yarisma karesi, ayni uretim yolu):
    #   640  : net +86 yalniz-aday / -36 gercek kayip, bos kare 15 -> 12
    #   1280 : net +40 / -56 — belirgin daha kotu
    # (Onceki vanilla YOLOv8s icin 1280 en iyiydi: +97/-10. Model degisince optimum da
    #  degisti; eski gerekce artik gecerli degil.)
    def __init__(self, weights_path: Optional[str] = None, motion_thresh: float = 5.0,
                 conf: float = 0.25, iou: float = 0.7, imgsz: int = 640):
        from ultralytics import YOLO  # standart pip surumu (fork DEGIL)

        self.weights = str(weights_path or BEST_PT)
        if "dinov3" in Path(self.weights).name.lower():
            _dinov3_enjekte()
        self.model = YOLO(self.weights)
        self.motion_thresh = motion_thresh
        self.conf, self.iou, self.imgsz = conf, iou, imgsz
        self._prev_frame: Optional[np.ndarray] = None
        self._prev_centroids: dict = {}   # track_id -> (cx, cy)
        logging.info(f"Gorev1Detector hazir: {self.weights} | siniflar={self.model.names}")

    @staticmethod
    def _spec_cls(model_cls: int) -> int:
        """Model sınıfını şartname sınıfına çevir (UAP/UAİ takası)."""
        if model_cls == 2:
            return 3      # model UAİ -> spec 3 (UAİ)
        if model_cls == 3:
            return 2      # model UAP -> spec 2 (UAP)
        return model_cls  # 0 Taşıt, 1 İnsan

    def detect(self, frame_bgr: np.ndarray) -> list[dict]:
        """Bir kareyi işle; şartname şemasında nesne sözlükleri listesi döndür."""
        h, w = frame_bgr.shape[:2]
        preds = self.model.track(frame_bgr, persist=True, tracker="botsort.yaml",
                                 conf=self.conf, iou=self.iou, imgsz=self.imgsz,
                                 verbose=False)

        # ego-hareket telafisi (taşıt hareket durumu için)
        H = np.eye(3)
        if self._prev_frame is not None:
            H = _calculate_homography(self._prev_frame, frame_bgr)

        detected_objects, obstacle_boxes, landing_candidates = [], [], []
        boxes = preds[0].boxes
        yeni_centroids: dict = {}
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            clss = boxes.cls.cpu().numpy()
            ids = (boxes.id.cpu().numpy() if boxes.id is not None
                   else np.full(len(xyxy), -1.0))
            for box, track_id, cls in zip(xyxy, ids, clss):
                x1, y1, x2, y2 = map(int, box)
                spec_cls = self._spec_cls(int(cls))
                obj = {"cls": str(spec_cls), "landing_status": "-1", "motion_status": "-1",
                       "top_left_x": x1, "top_left_y": y1,
                       "bottom_right_x": x2, "bottom_right_y": y2}

                if spec_cls in (0, 1):            # taşıt/insan → iniş engeli
                    obstacle_boxes.append((x1, y1, x2, y2))

                if spec_cls == 0:                 # taşıt → hareket durumu
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    motion = 0
                    tid = float(track_id)
                    if tid in self._prev_centroids:
                        pcx, pcy = self._prev_centroids[tid]
                        beklenen = cv2.perspectiveTransform(
                            np.array([[[pcx, pcy]]], dtype=np.float32), H)[0][0]
                        if np.hypot(cx - beklenen[0], cy - beklenen[1]) > self.motion_thresh:
                            motion = 1
                    yeni_centroids[tid] = (cx, cy)
                    obj["motion_status"] = str(motion)
                elif spec_cls in (2, 3):          # UAP/UAİ → iniş durumu (2. geçişte)
                    landing_candidates.append(obj)

                detected_objects.append(obj)

        # iniş uygunluğu (engeller toplandıktan sonra)
        for obj in landing_candidates:
            kutu = (obj["top_left_x"], obj["top_left_y"],
                    obj["bottom_right_x"], obj["bottom_right_y"])
            obj["landing_status"] = str(_check_landing_status(kutu, obstacle_boxes, w, h))

        self._prev_frame = frame_bgr
        self._prev_centroids = yeni_centroids
        return detected_objects
