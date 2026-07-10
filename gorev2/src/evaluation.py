"""Yarışma metriğiyle çok senaryolu değerlendirme altyapısı.

Akış: video bir kez taranıp ham ölçümler (``PairMeasurement``) CSV'ye
önbelleklenir; durum makinesi bu ölçümler üzerinde istenen sayıda rastgele
kesinti senaryosuyla saniyeler içinde koşturulur. Kıyas tabanları:

* **tut:** kesinti boyunca son sağlıklı konumu göndermeye devam et.
* **sabit_hiz:** kesinti anındaki hızla düz devam et.

Metrik, teknik şartname Denklem 2: tüm karelerde ortalama Öklid hatası.
Sağlıklı karelerde herkes GT'yi aynalar (hata 0) — puanı kesinti belirler.
"""
from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import data_io
from .calibration import CameraProfile
from .matching import FeatureTracker
from .vo_pipeline import (
    EstimatorConfig, PairMeasurement, VoLadderEstimator, VoPositionEstimator,
)

GT_COLS = ["translation_x", "translation_y", "translation_z"]


# --------------------------------------------------------------------------
# Kesinti senaryoları
# --------------------------------------------------------------------------
def make_outage_schedule(
    n_frames: int = 2250,
    healthy_prefix: int = 450,
    n_outages: tuple[int, int] = (1, 3),
    dur_range_s: tuple[float, float] = (20.0, 90.0),
    fps: float = 7.5,
    rng=None,
) -> np.ndarray:
    """Yarışma benzeri rastgele sağlık çizelgesi (True = sağlıklı)."""
    rng = np.random.default_rng(rng)
    healthy = np.ones(n_frames, dtype=bool)
    for _ in range(int(rng.integers(n_outages[0], n_outages[1] + 1))):
        dur = int(rng.uniform(*dur_range_s) * fps)
        start = int(rng.integers(healthy_prefix, n_frames - 20))
        healthy[start:start + dur] = False
    return healthy


# --------------------------------------------------------------------------
# Ölçüm önbelleği
# --------------------------------------------------------------------------
def build_measurements(
    video_path,
    profile: CameraProfile,
    cache_csv,
    progress: bool = True,
    tracker=None,
) -> pd.DataFrame:
    """Videodaki tüm ardışık 7,5 FPS çiftleri için ham ölçümleri üret (önbellekli).

    ``tracker`` verilmezse klasik LK motoru kullanılır; ``RaftTracker`` gibi
    aynı arayüzü sunan herhangi bir motor da geçirilebilir.
    """
    cache_csv = Path(cache_csv)
    if cache_csv.exists():
        print(f"olcum onbellekten yuklendi: {cache_csv.name}")
        return pd.read_csv(cache_csv)

    t0 = time.perf_counter()
    tracker = tracker if tracker is not None else FeatureTracker(profile)
    rows, prev, prev_normal = [], None, None
    for idx, gray in data_io.iter_competition_frames(video_path, progress=progress):
        if prev is not None:
            m = PairMeasurement.compute(prev[1], gray, profile, tracker, prev_normal)
            if m.hom_ok:
                prev_normal = m.hom_normal
            row = asdict(m)
            row["idx0"], row["idx1"] = prev[0], idx
            rows.append(row)
        prev = (idx, gray)
    df = pd.DataFrame(rows)
    df.to_csv(cache_csv, index=False)
    print(f"{len(df)} olcum, {time.perf_counter() - t0:.0f} s -> {cache_csv.name}")
    return df


def _meas_from_row(row) -> PairMeasurement:
    alanlar = {k: row[k] for k in PairMeasurement.__dataclass_fields__}
    for b in ("ok", "frozen", "sim_ok", "hom_ok"):
        alanlar[b] = bool(alanlar[b])
    return PairMeasurement(**alanlar)


def build_ladder_measurements(
    video_path,
    profile: CameraProfile,
    cache_csv,
    spacing: int = 8,
    progress: bool = True,
    tracker=None,
) -> pd.DataFrame:
    """Keyframe merdiveni ölçümleri: her kare için (son keyframe → kare) çifti.

    Keyframe'ler sabit ızgarada durur (0, spacing, 2·spacing, ...); böylece
    ölçümler kesinti çizelgesinden bağımsız önbelleklenebilir ve durum makinesi
    yüzlerce senaryoda yeniden koşturulabilir.
    """
    cache_csv = Path(cache_csv)
    if cache_csv.exists():
        print(f"merdiven olcumleri onbellekten yuklendi: {cache_csv.name}")
        return pd.read_csv(cache_csv)

    t0 = time.perf_counter()
    tracker = tracker if tracker is not None else FeatureTracker(profile)
    rows = []
    kf = None            # (ordinal, frame_idx, gray)
    prev_normal = None
    ord_ = 0
    for idx, gray in data_io.iter_competition_frames(video_path, progress=progress):
        if kf is None:
            kf = (0, idx, gray)
        else:
            m = PairMeasurement.compute(kf[2], gray, profile, tracker, prev_normal)
            if m.hom_ok:
                prev_normal = m.hom_normal
            row = asdict(m)
            row["idx0"], row["idx1"] = kf[1], idx
            row["kf_ord"], row["ord"] = kf[0], ord_
            rows.append(row)
            if ord_ % spacing == 0:
                kf = (ord_, idx, gray)
        ord_ += 1
    df = pd.DataFrame(rows)
    df.to_csv(cache_csv, index=False)
    print(f"{len(df)} merdiven olcumu, {time.perf_counter() - t0:.0f} s -> {cache_csv.name}")
    return df


# --------------------------------------------------------------------------
# Koşturucular
# --------------------------------------------------------------------------
def run_estimator(
    meas_df: pd.DataFrame,
    gt_sub: pd.DataFrame,
    healthy: np.ndarray,
    profile: CameraProfile,
    config: EstimatorConfig,
) -> np.ndarray:
    """Durum makinesini bir senaryo üzerinde koştur; (N,3) kestirim döndürür.

    Kare 0 her zaman sağlıklı kabul edilir (şartname: ilk 1 dk garanti).
    """
    gt = gt_sub[GT_COLS].to_numpy()
    n = len(gt)
    est = VoPositionEstimator(profile, config)
    preds = np.zeros((n, 3))
    preds[0] = gt[0]
    est.update(None, gt[0], True)  # ilk kare: ölçüm yok, GT aynalanır
    meas_rows = [_meas_from_row(r) for r in meas_df.to_dict("records")]
    for k in range(1, n):
        m = meas_rows[k - 1] if k - 1 < len(meas_rows) else None
        preds[k] = est.update(m, gt[k] if healthy[k] else None, bool(healthy[k]))
    return preds


def run_estimator_ladder(
    meas_cons_df: pd.DataFrame,
    meas_ladder_df: pd.DataFrame,
    gt_sub: pd.DataFrame,
    healthy: np.ndarray,
    profile: CameraProfile,
    config: EstimatorConfig,
    spacing: int = 8,
) -> np.ndarray:
    """Keyframe merdivenli durum makinesini bir senaryo üzerinde koştur."""
    gt = gt_sub[GT_COLS].to_numpy()
    n = len(gt)
    est = VoLadderEstimator(profile, config, spacing=spacing)
    preds = np.zeros((n, 3))
    preds[0] = gt[0]
    est.update_ladder(0, None, None, gt[0], True)
    cons_rows = [_meas_from_row(r) for r in meas_cons_df.to_dict("records")]
    ladder_rows = [_meas_from_row(r) for r in meas_ladder_df.to_dict("records")]
    for k in range(1, n):
        mc = cons_rows[k - 1] if k - 1 < len(cons_rows) else None
        ml = ladder_rows[k - 1] if k - 1 < len(ladder_rows) else None
        preds[k] = est.update_ladder(k, mc, ml, gt[k] if healthy[k] else None, bool(healthy[k]))
    return preds


def run_baseline(
    gt_sub: pd.DataFrame,
    healthy: np.ndarray,
    kind: str = "tut",
    vel_window: int = 8,
) -> np.ndarray:
    """Görsel bilgi kullanmayan kıyas tabanları."""
    gt = gt_sub[GT_COLS].to_numpy()
    n = len(gt)
    preds = np.zeros((n, 3))
    preds[0] = gt[0]
    vel = np.zeros(3)
    son_saglikli = gt[0].copy()
    gecmis = [gt[0].copy()]
    for k in range(1, n):
        if healthy[k]:
            preds[k] = gt[k]
            son_saglikli = gt[k].copy()
            gecmis.append(gt[k].copy())
            if len(gecmis) > vel_window + 1:
                gecmis.pop(0)
            if len(gecmis) >= 2:
                vel = (gecmis[-1] - gecmis[0]) / (len(gecmis) - 1)
        elif kind == "tut":
            preds[k] = son_saglikli
        else:  # sabit_hiz: kesintiye girerkenki hızla düz devam
            preds[k] = preds[k - 1] + vel
    return preds


# --------------------------------------------------------------------------
# Metrikler ve toplu değerlendirme
# --------------------------------------------------------------------------
def competition_error(preds: np.ndarray, gt: np.ndarray) -> float:
    """Teknik şartname Denklem 2: ortalama Öklid hatası [m]."""
    return float(np.linalg.norm(preds - gt, axis=1).mean())


def outage_error(preds: np.ndarray, gt: np.ndarray, healthy: np.ndarray) -> float:
    """Yalnızca kesinti karelerindeki ortalama hata [m] (tanı amaçlı)."""
    m = ~healthy
    if not m.any():
        return 0.0
    return float(np.linalg.norm(preds[m] - gt[m], axis=1).mean())


def batch_evaluate(
    meas_df: pd.DataFrame,
    gt_sub: pd.DataFrame,
    profile: CameraProfile,
    n_seeds: int = 20,
    methods: tuple[str, ...] = ("benzerlik", "homografi", "fuzyon"),
    schedule_kwargs: Optional[dict] = None,
    config_overrides: Optional[dict] = None,
    include_baselines: bool = True,
) -> pd.DataFrame:
    """Çok senaryolu değerlendirme: her tohumda aynı çizelgeyle tüm yöntemler."""
    gt = gt_sub[GT_COLS].to_numpy()
    n = len(gt)
    kw = dict(n_frames=n)
    if schedule_kwargs:
        kw.update(schedule_kwargs)
    over = config_overrides or {}
    rows = []
    for seed in range(n_seeds):
        healthy = make_outage_schedule(rng=seed, **kw)
        for yontem in methods:
            preds = run_estimator(meas_df, gt_sub, healthy, profile,
                                  EstimatorConfig(method=yontem, **over))
            rows.append(dict(seed=seed, yontem=yontem,
                             E=competition_error(preds, gt),
                             E_kesinti=outage_error(preds, gt, healthy),
                             kesinti_orani=float((~healthy).mean())))
        if include_baselines:
            for taban in ("tut", "sabit_hiz"):
                preds = run_baseline(gt_sub, healthy, kind=taban)
                rows.append(dict(seed=seed, yontem=taban,
                                 E=competition_error(preds, gt),
                                 E_kesinti=outage_error(preds, gt, healthy),
                                 kesinti_orani=float((~healthy).mean())))
    return pd.DataFrame(rows)
