"""Örnek veri setini okuma yardımcıları: GT konum CSV'leri ve video kareleri.

Kavramlar
---------
* **GT (ground truth):** Her kare çekildiği andaki gerçek yer değiştirme
  (metre, video başlangıcına göre). Yarışmada bu bilgi yalnızca
  ``health_status = 1`` iken verilir; kesinti anında biz kestireceğiz.
* **7,5 FPS altörnekleme:** Örnek videolar ~30 FPS kaydedilmiş, yarışmada
  ise kareler 7,5 FPS ile dağıtılacak. Yarışma koşullarını birebir taklit
  etmek için videodan her 4. kareyi alıyoruz.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np
import pandas as pd

# Yarışma sabitleri
COMPETITION_FPS = 7.5
NATIVE_FPS = 30.0  # örnek videolar ~29.97 FPS
FRAME_STEP = 4     # 30 FPS -> 7.5 FPS: her 4. kare


def load_translations(csv_path: str | Path) -> pd.DataFrame:
    """Translation CSV'sini oku ve kare numarasını tamsayı sütuna çevir.

    Dönen tablo: ``frame_idx, translation_x, translation_y, translation_z``
    """
    df = pd.read_csv(csv_path)
    df["frame_idx"] = (
        df["frame_numbers"].str.replace("frame_", "", regex=False).astype(int)
    )
    return df[["frame_idx", "translation_x", "translation_y", "translation_z"]]


def subsample_competition(df: pd.DataFrame, step: int = FRAME_STEP) -> pd.DataFrame:
    """30 FPS GT tablosundan yarışma temposunu (7,5 FPS) simüle et."""
    sub = df[df["frame_idx"] % step == 0].reset_index(drop=True)
    return sub


def video_info(video_path: str | Path) -> dict:
    """Videonun temel özelliklerini döndür (genişlik, yükseklik, fps, kare sayısı)."""
    cap = cv2.VideoCapture(str(video_path))
    try:
        info = {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "n_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
    finally:
        cap.release()
    return info


def read_frame(video_path: str | Path, frame_idx: int, gray: bool = False) -> np.ndarray:
    """Videodan tek kare oku (keşif/görselleştirme için).

    Rastgele erişim (seek) yavaştır; toplu işlem için
    :func:`iter_competition_frames` kullanın. BGR (OpenCV varsayılanı) döner.
    """
    cap = cv2.VideoCapture(str(video_path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok:
        raise IOError(f"Kare okunamadi: {video_path} #{frame_idx}")
    if gray:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame


def iter_competition_frames(
    video_path: str | Path,
    step: int = FRAME_STEP,
    gray: bool = True,
    start: int = 0,
    stop: Optional[int] = None,
    progress: bool = False,
) -> Iterator[tuple[int, np.ndarray]]:
    """Videoyu sıralı okuyarak her ``step``. kareyi üret (yarışma temposu).

    Sıralı okuma + atlanan karelerde ``grab()`` kullanmak, her kareye
    seek yapmaktan kat kat hızlıdır. ``(frame_idx, frame)`` ikilileri üretir.
    """
    cap = cv2.VideoCapture(str(video_path))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stop = n_frames if stop is None else min(stop, n_frames)

    pbar = None
    if progress:
        from tqdm import tqdm

        # mininterval: defter çıktısını şişirmemek için ilerleme en fazla 5 sn'de bir yazılır
        pbar = tqdm(total=stop, desc=Path(str(video_path)).stem, unit="kare", mininterval=5)

    idx = 0
    try:
        while idx < stop:
            if idx >= start and idx % step == 0:
                ok, frame = cap.read()
                if not ok:
                    break
                if gray:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                yield idx, frame
            else:
                # grab: kareyi çözer ama numpy'a kopyalamaz -> ucuz atlama
                if not cap.grab():
                    break
            idx += 1
            if pbar is not None:
                pbar.update(1)
    finally:
        cap.release()
        if pbar is not None:
            pbar.close()
