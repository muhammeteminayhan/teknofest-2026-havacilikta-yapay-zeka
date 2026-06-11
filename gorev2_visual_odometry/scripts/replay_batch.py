"""
BATCH entegrasyon kuru provası (sunucu olmadan).

main_batch.py'nin veri yolunu doğrular: pass-1 → global smoothing → her kare için
GERÇEK FramePredictions + DetectedTranslation ile payload üretimi + uçtan uca metrik.
Mevcut pass-1 önbelleği varsa kullanır (hız).

Kullanım:
  python scripts/replay_batch.py --step 4 --health-mode intermittent --dropout 300
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_REPO = os.path.dirname(_ROOT)
_IFACE_SRC = os.path.join(_REPO, "TAKIM_BAGLANTI_ARAYUZU", "src")
for p in (_ROOT, _HERE, os.path.join(_ROOT, "configs"), _IFACE_SRC):
    sys.path.insert(0, p)

import config  # noqa: E402
from vo.frame_source import VideoFrameSource, load_translation_csv, read_video_meta  # noqa: E402
from vo.pipeline import build_estimator  # noqa: E402
from vo.batch import (collect_frame_motions, make_batch_data,  # noqa: E402
                      motions_to_arrays, arrays_to_motions)
from vo.optimize import smooth_trajectory  # noqa: E402
from evaluate import _is_healthy_frame  # noqa: E402

# Arayüzün GERÇEK veri sınıfları
from frame_predictions import FramePredictions  # noqa: E402
from detected_translation import DetectedTranslation  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=4)
    ap.add_argument("--healthy-prefix", type=int, default=config.HEALTHY_PREFIX_FRAMES)
    ap.add_argument("--health-mode", choices=["tail", "intermittent"], default="intermittent")
    ap.add_argument("--dropout", type=int, default=300)
    ap.add_argument("--recover", type=int, default=45)
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    video, csv_path = config.RGB_VIDEO, config.RGB_CSV
    gt_all = load_translation_csv(csv_path)
    meta = read_video_meta(video)
    est = build_estimator("rgb", meta.width, meta.height)

    cache = os.path.join(config.OUTPUTS_DIR,
                         f"_cache_pass1_rgb_step{args.step}_{config.FEATURE_BACKEND}_"
                         f"{config.MOTION_MODEL}_w{config.PROC_MAX_WIDTH}_f{config.MAX_FEATURES}_max{args.max_frames}.npz")
    if os.path.exists(cache):
        a = np.load(cache)
        fms = arrays_to_motions(a); orig_idxs = a["orig"].tolist()
        print(f"Pass-1 önbellekten ({len(fms)} kare).")
    else:
        src = VideoFrameSource(video, step=args.step, max_width=None, to_gray=False)
        maxj = args.max_frames or None

        def it():
            j = 0
            for oi, fr in src:
                if oi >= gt_all.shape[0] or (maxj and j >= maxj):
                    break
                yield oi, fr; j += 1
        print("Pass-1 çalışıyor (önbellek yok)...")
        fms, orig_idxs = collect_frame_motions(est, it())

    n = len(fms)
    batch = make_batch_data(fms, orig_idxs,
                            health_fn=lambda j: _is_healthy_frame(j, args),
                            gt_fn=lambda j, oi: gt_all[oi])
    preds = smooth_trajectory(batch, align_mode="interp")

    # ---- main_batch'in gönderim yolunu taklit et: FramePredictions + payload ----
    gts = gt_all[np.arange(n) * args.step]
    healths = batch.healths
    payload_ok = 0
    example = None
    pred_sent = np.zeros((n, 3))
    for j in range(n):
        gt = gts[j]
        prediction = FramePredictions(f"/frame/{j}/", f"/img/f_{j}.jpg", "replay",
                                      str(gt[0]), str(gt[1]), str(gt[2]))
        px, py, pz = preds[j]
        prediction.add_translation_object(DetectedTranslation(float(px), float(py), float(pz)))
        payload = prediction.create_payload("http://127.0.0.1:5000/")
        if isinstance(payload, dict) and len(payload.get("detected_translations", [])) == 1:
            payload_ok += 1
            if example is None:
                example = payload["detected_translations"][0]
        t = prediction.translations[-1]
        pred_sent[j] = [float(t.translation_x), float(t.translation_y), float(t.translation_z)]

    un = ~healths
    err = np.linalg.norm(pred_sent - gts, axis=1)
    print(f"\nKare: {n}, geçerli payload: {payload_ok}/{n}")
    print(f"Örnek translation payload: {json.dumps(example, ensure_ascii=False)}")
    print(f"Sağlıklı echo hatası: {err[healths].mean():.6f} m (≈0 olmalı)")
    print(f">>> RESMİ METRİK (health=0): E = {err[un].mean():.3f} m  (medyan {np.median(err[un]):.3f})")
    print("\nBATCH entegrasyon provası BAŞARILI ✓" if payload_ok == n else "\nUYARI: payload eksik!")


if __name__ == "__main__":
    main()
