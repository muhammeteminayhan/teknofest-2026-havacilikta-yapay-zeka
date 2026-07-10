"""Mid-Air veri setini (ULiège) bizim değerlendirme formatımıza uyarlama.

Mid-Air: sentetik drone uçuşları, aşağı bakan kamera (1024×1024, FOV 90°,
25 Hz), 100 Hz GT konum (yer değiştirme; ilk örnek (0,0,0)) — sis/günbatımı/
kış temalarında **GT'li E ölçümü** için tek kaynağımız.

Hizalama: GT 100 Hz, kare 25 Hz → kare i ↔ GT indeksi 4·i (tam oran).
Yarışma temposuna yaklaşmak için kareler ``step`` ile altörneklenir
(step=3 → 8,33 Hz ≈ yarışmanın 7,5 Hz'i; adım süresi 0,12 sn vs 0,133 sn).

Kamera: FOV 90° → f = W/2 = 512 px; sentetik görüntüde distorsiyon yok.
"""
from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd

from .calibration import CameraProfile
from .vo_pipeline import PairMeasurement

MIDAIR_PROFILE = CameraProfile(
    name="midair_down", width=1024, height=1024,
    fx=512.0, fy=512.0, cx=512.0, cy=512.0, k1=0.0, k2=0.0,
)

GT_PER_FRAME = 4          # 100 Hz GT / 25 Hz kare
VARSAYILAN_STEP = 3       # 25/3 = 8,33 Hz ~ yarışma temposu


def load_midair_gt(hdf5_path: str | Path, trajectory: str,
                   step: int = VARSAYILAN_STEP) -> pd.DataFrame:
    """Altörneklenmiş GT tablosu: frame_idx + translation_x/y/z (bizim format)."""
    with h5py.File(hdf5_path, "r") as f:
        pos = f[trajectory]["groundtruth"]["position"][:]
        n_frames = len(f[trajectory]["camera_data"]["color_down"])
    idx = np.arange(0, n_frames, step)
    gt = pos[idx * GT_PER_FRAME]
    return pd.DataFrame(dict(frame_idx=idx, translation_x=gt[:, 0],
                             translation_y=gt[:, 1], translation_z=gt[:, 2]))


def midair_frame_paths(weather_dir: str | Path, hdf5_path: str | Path,
                       trajectory: str, step: int = VARSAYILAN_STEP) -> list[Path]:
    """Altörneklenmiş kare dosya yolları (weather klasörüne göre)."""
    weather_dir = Path(weather_dir)
    with h5py.File(hdf5_path, "r") as f:
        rel = [p.decode() if isinstance(p, bytes) else str(p)
               for p in f[trajectory]["camera_data"]["color_down"][:]]
    return [weather_dir / rel[i] for i in range(0, len(rel), step)]


def build_midair_measurements(
    weather_dir: str | Path,
    trajectory: str,
    cache_csv: str | Path,
    tracker,
    step: int = VARSAYILAN_STEP,
) -> pd.DataFrame:
    """Bir trajectory için ardışık ölçüm önbelleği (bizim CSV formatı)."""
    cache_csv = Path(cache_csv)
    if cache_csv.exists():
        return pd.read_csv(cache_csv)
    hdf5_path = Path(weather_dir) / "sensor_records.hdf5"
    yollar = midair_frame_paths(weather_dir, hdf5_path, trajectory, step)

    t0 = time.perf_counter()
    rows, prev, prev_normal = [], None, None
    for i, yol in enumerate(yollar):
        gray = cv2.imread(str(yol), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            # eksik/bozuk kare: gecersiz olcum yaz, zinciri o noktadan yeniden baslat
            if prev is not None:
                row = asdict(PairMeasurement())
                row["idx0"], row["idx1"] = (i - 1) * step, i * step
                rows.append(row)
            prev = None
            continue
        if prev is not None:
            m = PairMeasurement.compute(prev, gray, MIDAIR_PROFILE, tracker, prev_normal)
            if m.hom_ok:
                prev_normal = m.hom_normal
            row = asdict(m)
            row["idx0"], row["idx1"] = (i - 1) * step, i * step
            rows.append(row)
        prev = gray
    df = pd.DataFrame(rows)
    df.to_csv(cache_csv, index=False)
    print(f"{trajectory}: {len(df)} olcum, {time.perf_counter() - t0:.0f} s -> {cache_csv.name}")
    return df
