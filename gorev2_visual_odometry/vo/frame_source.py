"""
Kare kaynağı — örnek MP4'lerden kareleri sıralı okur ve GT translation CSV ile
hizalar.

4K video (9022 kare) tüm kareleri diske yazmak ~18 GB tutacağından, kareler
VideoCapture ile sıralı (on-the-fly) okunur. İsteğe bağlı küçültme (downscale)
ve fps alt-örnekleme desteklenir.

Yarışmada kareler sunucudan JPG olarak inecek; o akış connection arayüzü
üzerinden ayrı ele alınır. Bu modül yalnızca çevrimdışı geliştirme/değerlendirme
içindir.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class VideoMeta:
    width: int
    height: int
    fps: float
    frame_count: int          # kapsayıcının raporladığı kare sayısı (yaklaşık olabilir)


def read_video_meta(video_path: str) -> VideoMeta:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Video açılamadı: {video_path}")
    meta = VideoMeta(
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps=float(cap.get(cv2.CAP_PROP_FPS)),
        frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    cap.release()
    return meta


def load_translation_csv(csv_path: str) -> np.ndarray:
    """
    GT translation CSV'sini oku → (N, 3) float dizi [x, y, z].

    Sütunlar: translation_x, translation_y, translation_z, frame_numbers.
    Değerler ilk kareye göre KÜMÜLATİF pozisyondur (delta değil).
    """
    xs, ys, zs = [], [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            xs.append(float(row["translation_x"]))
            ys.append(float(row["translation_y"]))
            zs.append(float(row["translation_z"]))
    return np.array([xs, ys, zs], dtype=np.float64).T


def count_video_frames(video_path: str) -> int:
    """Kareleri gerçekten okuyarak say (CAP_PROP_FRAME_COUNT bazen yanlıştır)."""
    cap = cv2.VideoCapture(video_path)
    n = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        n += 1
    cap.release()
    return n


class VideoFrameSource:
    """
    MP4'ten kareleri sıralı üreten yineleyici.

    Parametreler
    ----------
    video_path : str
    step : int
        fps alt-örnekleme. step=1 her kare; step=4 her 4. kare (örn. 30->7.5 fps).
    max_width : int | None
        Bu genişliğin üzerindeki kareler en-boy korunarak küçültülür.
        Küçültme oranı `scale` olarak raporlanır (intrinsics ölçeklemek için).
    to_gray : bool
        True ise kareler gri tonlamaya çevrilir.
    """

    def __init__(self, video_path: str, step: int = 1, max_width: int | None = None,
                 to_gray: bool = False):
        self.video_path = video_path
        self.step = max(1, int(step))
        self.max_width = max_width
        self.to_gray = to_gray
        self.meta = read_video_meta(video_path)
        # Küçültme oranı (orijinal -> işleme çözünürlüğü)
        if max_width and self.meta.width > max_width:
            self.scale = max_width / float(self.meta.width)
        else:
            self.scale = 1.0
        self.proc_width = int(round(self.meta.width * self.scale))
        self.proc_height = int(round(self.meta.height * self.scale))

    def __iter__(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Video açılamadı: {self.video_path}")
        idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx % self.step == 0:
                    out = frame
                    if self.scale != 1.0:
                        out = cv2.resize(out, (self.proc_width, self.proc_height),
                                         interpolation=cv2.INTER_AREA)
                    if self.to_gray:
                        out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
                    yield idx, out
                idx += 1
        finally:
            cap.release()
