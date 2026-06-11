"""
Batch (toplu) VO — pass-1 ve ham zincir kurma.

Yarışma arayüzü tüm kareleri ve health bitlerini baştan verdiğinden, problem
çevrimdışı bir smoothing problemine dönüşür. Bu modül:
  1) Tüm kareler için göreli hareketi (FrameMotion) toplar (pass-1),
  2) yaw-telafili ham VO zincirini kurar (per-frame artış d_i ve kümülatif r_i),
sonra `vo/optimize.py` global çapalı optimizasyonu uygular.

Ön-işleme/ön-yüz/hareket, çevrimiçi hat ile AYNI koddur (VOEstimator.compute_frame_motion).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .motion import FrameMotion


@dataclass
class BatchData:
    fms: list                # length N: FrameMotion (fm[0] = ilk kare, hareketsiz)
    healths: np.ndarray      # bool (N,)
    gts: np.ndarray          # (N,3) — sağlıksız karelerde NaN (optimizasyon kullanamaz)
    n_inliers: np.ndarray    # (N,) güven ağırlığı için


def collect_motions(estimator, frames, health_fn, gt_fn, progress=None) -> BatchData:
    """
    Pass-1: kareleri sırayla işle, FrameMotion + health + (maskeli) GT topla.

    frames     : (orig_idx, image) veren yineleyici (sıra önemlidir)
    health_fn  : health_fn(j) -> bool   (işlenmiş kare j sağlıklı mı = GT çapa)
    gt_fn      : gt_fn(j, orig_idx) -> (3,)  (sağlıklıyken kullanılacak GT/sunucu değeri)
    progress   : opsiyonel callback(j) ilerleme için

    Sağlıksız karelerde GT NaN'lanarak saklanır → optimizasyon hile yapamaz.
    """
    fms, hs, gts, ninl = [], [], [], []
    j = 0
    for orig_idx, frame in frames:
        fm = estimator.compute_frame_motion(frame)
        fms.append(fm)
        healthy = bool(health_fn(j))
        hs.append(healthy)
        gts.append(np.asarray(gt_fn(j, orig_idx), dtype=np.float64) if healthy
                   else np.full(3, np.nan))
        ninl.append(fm.n_inliers)
        if progress is not None:
            progress(j)
        j += 1
    return BatchData(fms, np.array(hs, dtype=bool), np.array(gts, dtype=np.float64),
                     np.array(ninl, dtype=np.float64))


def collect_frame_motions(estimator, frames, progress=None):
    """
    Pass-1 (yalnızca hareket): kareleri sırayla işle → (fms, orig_idxs).
    health/GT'den bağımsızdır → önbelleğe alınabilir (health deseni sonradan uygulanır).
    """
    fms, orig = [], []
    j = 0
    for orig_idx, frame in frames:
        fms.append(estimator.compute_frame_motion(frame))
        orig.append(orig_idx)
        if progress is not None:
            progress(j)
        j += 1
    return fms, orig


def motions_to_arrays(fms: list) -> dict:
    """FrameMotion listesini önbellek için dizilere çevir."""
    return {
        "rdx": np.array([f.raw_dx for f in fms]),
        "rdy": np.array([f.raw_dy for f in fms]),
        "rdz": np.array([f.raw_dz for f in fms]),
        "dyaw": np.array([f.dyaw for f in fms]),
        "ninl": np.array([f.n_inliers for f in fms]),
        "ok": np.array([f.ok for f in fms]),
    }


def arrays_to_motions(a: dict) -> list:
    """Önbellek dizilerinden FrameMotion listesini geri kur."""
    n = len(a["rdx"])
    return [FrameMotion(float(a["rdx"][i]), float(a["rdy"][i]), float(a["rdz"][i]),
                        float(a["dyaw"][i]), 1.0, int(a["ninl"][i]), 0.0, bool(a["ok"][i]))
            for i in range(n)]


def make_batch_data(fms: list, orig_idxs, health_fn, gt_fn) -> BatchData:
    """Hareketler + health deseni + GT'den BatchData kur (sağlıksızda GT=NaN)."""
    n = len(fms)
    hs = np.array([bool(health_fn(j)) for j in range(n)], dtype=bool)
    gts = np.full((n, 3), np.nan)
    for j in range(n):
        if hs[j]:
            gts[j] = np.asarray(gt_fn(j, orig_idxs[j]), dtype=np.float64)
    ninl = np.array([f.n_inliers for f in fms], dtype=np.float64)
    return BatchData(fms, hs, gts, ninl)


def build_raw_chain(fms: list) -> dict:
    """
    Yaw-telafili ham VO zinciri.

    Döndürür:
      inc    : (N,3) per-frame dünya(ham)-çerçevesi artış [rot(phi)·(dx,dy), dz]
      raw    : (N,3) kümülatif ham pozisyon (inc'in kümülatif toplamı)
      raw_dz : (N,) per-frame ham dz (irtifa proxy'si; z hizalama için)
      ok     : (N,) bool — kare güvenilir mi
    """
    n = len(fms)
    inc = np.zeros((n, 3))
    ok = np.zeros(n, dtype=bool)
    phi = 0.0
    last_inc = np.zeros(3)
    last_dyaw = 0.0
    for i in range(1, n):
        fm: FrameMotion = fms[i]
        if fm.ok:
            body = np.array([fm.raw_dx, fm.raw_dy, fm.raw_dz])
            dyaw = fm.dyaw
            last_inc, last_dyaw = body, dyaw
            ok[i] = True
        else:
            body, dyaw = last_inc, last_dyaw      # sabit-hız yedeği
        c, s = np.cos(phi), np.sin(phi)
        inc[i] = [c * body[0] - s * body[1], s * body[0] + c * body[1], body[2]]
        phi += dyaw
    raw = np.cumsum(inc, axis=0)
    return {"inc": inc, "raw": raw, "raw_dz": inc[:, 2], "ok": ok}
