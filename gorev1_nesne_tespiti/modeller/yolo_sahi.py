"""
YOLOv11 + SAHI tespit sarıcısı.

İki çıkarım modu:
- Doğrudan YOLO: tek geçiş, hızlı; düşük çözünürlük / büyük nesne.
- SAHI (dilimli): kareyi örtüşen dilimlere böler, her dilimde YOLO çalıştırır,
  sonuçları birleştirir. FHD/4K'da küçük nesne tespitini belirgin artırır.

Ağır bağımlılıklar (torch/ultralytics/sahi) tembel yüklenir; böylece bu modül
testlerde/CI'da bağımlılık olmadan da import edilebilir (Detector init'te yüklenir).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class Detection:
    """Tek bir tespit (yarışma-bağımsız ara biçim)."""
    cls_id: int              # COCO veya fine-tuned model sınıf id'si
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def merkez(self) -> np.ndarray:
        return np.array([(self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0], dtype=np.float32)

    @property
    def xyxy(self) -> tuple:
        return (self.x1, self.y1, self.x2, self.y2)


class Detector:
    """
    YOLOv11 tespit modeli; isteğe bağlı SAHI dilimleme.

    model_path : .pt ağırlık yolu (COCO pretrained 'yolo11n.pt' veya fine-tuned).
    use_sahi   : True ise dilimli çıkarım.
    device     : 'cuda:0' | 'cpu' | None (otomatik).
    """

    def __init__(self, model_path: str, use_sahi: bool = True,
                 device: Optional[str] = None, conf: float = 0.25, iou: float = 0.7,
                 half: bool = True, slice_size: int = 640, overlap: float = 0.25,
                 arch: str = "yolo"):
        self.model_path = model_path
        self.arch = arch                       # "yolo" | "rtdetr" (SOTA DETR; ultralytics)
        # RT-DETR global-bağlamlı; SAHI (YOLO sarıcı) ile uyumsuz → doğrudan çıkarım.
        if arch == "rtdetr":
            use_sahi = False
        self.use_sahi = use_sahi
        self.conf = conf
        self.iou = iou
        self.half = half
        self.slice_size = slice_size
        self.overlap = overlap

        # Cihaz otomatik seçimi
        if device is None:
            try:
                import torch
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        self.device = device
        self.half = half and ("cuda" in str(device))

        self._yolo = None          # doğrudan mod
        self._sahi_model = None    # SAHI mod
        self._yukle()

    def _yukle(self):
        if self.use_sahi:
            from sahi import AutoDetectionModel
            # SAHI 0.11+ ultralytics tipini destekler; eski sürüm için 'yolov8'e düş.
            try:
                self._sahi_model = AutoDetectionModel.from_pretrained(
                    model_type="ultralytics", model_path=self.model_path,
                    confidence_threshold=self.conf, device=self.device,
                )
            except Exception:
                self._sahi_model = AutoDetectionModel.from_pretrained(
                    model_type="yolov8", model_path=self.model_path,
                    confidence_threshold=self.conf, device=self.device,
                )
        elif self.arch == "rtdetr":
            from ultralytics import RTDETR
            self._yolo = RTDETR(self.model_path)   # .predict arayüzü YOLO ile aynı
        else:
            from ultralytics import YOLO
            self._yolo = YOLO(self.model_path)

    def detect(self, image: np.ndarray) -> List[Detection]:
        """BGR/RGB kare → Detection listesi."""
        if self.use_sahi:
            return self._detect_sahi(image)
        return self._detect_yolo(image)

    def _detect_yolo(self, image: np.ndarray) -> List[Detection]:
        res = self._yolo.predict(image, conf=self.conf, iou=self.iou,
                                 device=self.device, half=self.half, verbose=False)[0]
        dets: List[Detection] = []
        if res.boxes is None:
            return dets
        for b in res.boxes:
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
            dets.append(Detection(int(b.cls[0]), float(b.conf[0]), x1, y1, x2, y2))
        return dets

    def _detect_sahi(self, image: np.ndarray) -> List[Detection]:
        from sahi.predict import get_sliced_prediction
        result = get_sliced_prediction(
            image, self._sahi_model,
            slice_height=self.slice_size, slice_width=self.slice_size,
            overlap_height_ratio=self.overlap, overlap_width_ratio=self.overlap,
            verbose=0,
        )
        dets: List[Detection] = []
        for op in result.object_prediction_list:
            x1, y1, x2, y2 = op.bbox.to_xyxy()
            dets.append(Detection(int(op.category.id), float(op.score.value),
                                  float(x1), float(y1), float(x2), float(y2)))
        return dets
