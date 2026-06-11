"""
Entegrasyon kuru provası (sunucu olmadan).

`integration/object_detection_model.py` drop-in sürümünü, yarışma arayüzünün
GERÇEK veri sınıflarıyla (FramePredictions, DetectedTranslation) örnek video
üzerinde çalıştırır. Doğrular:
  - Drop-in ObjectDetectionModel arayüzle uyumlu mu (import, detect imzası),
  - Health anahtarlaması doğru mu (health=1 echo, health=0 VO kestirimi),
  - FramePredictions.create_payload() geçerli JSON payload üretiyor mu,
  - Uçtan uca pozisyon metriği (sunucu akışıyla aynı sonuç).

Kullanım:
  python scripts/replay_integration.py --step 4 --health-mode intermittent
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_REPO = os.path.dirname(_ROOT)
_IFACE_SRC = os.path.join(_REPO, "TAKIM_BAGLANTI_ARAYUZU", "src")

sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "configs"))
sys.path.insert(0, _IFACE_SRC)   # arayüzün veri sınıfları + integration'ın mutlak import yedeği

import config  # noqa: E402
from vo.frame_source import VideoFrameSource, load_translation_csv, read_video_meta  # noqa: E402

# Arayüzün GERÇEK veri sınıfı
from frame_predictions import FramePredictions  # noqa: E402  (TAKIM_BAGLANTI_ARAYUZU/src)


def load_integration_odm():
    """integration/object_detection_model.py dosyasını dosya yolundan yükle."""
    path = os.path.join(_ROOT, "integration", "object_detection_model.py")
    spec = importlib.util.spec_from_file_location("vo_integration_odm", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ObjectDetectionModel


def _is_healthy(j, prefix, mode, dropout, recover):
    if j < prefix:
        return True
    if mode == "tail":
        return False
    return (j - prefix) % (recover + dropout) < recover


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", choices=["rgb", "thermal"], default="rgb")
    ap.add_argument("--step", type=int, default=4)
    ap.add_argument("--healthy-prefix", type=int, default=config.HEALTHY_PREFIX_FRAMES)
    ap.add_argument("--health-mode", choices=["tail", "intermittent"], default="intermittent")
    ap.add_argument("--dropout", type=int, default=300)
    ap.add_argument("--recover", type=int, default=45)
    ap.add_argument("--max-frames", type=int, default=1200)
    args = ap.parse_args()

    video = config.RGB_VIDEO if args.modality == "rgb" else config.THERMAL_VIDEO
    csv_path = config.RGB_CSV if args.modality == "rgb" else config.THERMAL_CSV
    gt_all = load_translation_csv(csv_path)
    meta = read_video_meta(video)

    ObjectDetectionModel = load_integration_odm()
    model = ObjectDetectionModel("http://127.0.0.1:5000/")   # sahte url; ağ kullanılmaz
    print(f"Drop-in ObjectDetectionModel yüklendi. Video {meta.width}x{meta.height}.")

    src = VideoFrameSource(video, step=args.step, max_width=None, to_gray=False)
    preds, gts, healths = [], [], []
    payload_ok = 0
    payload_example = None
    j = 0
    for orig_idx, frame in src:
        if orig_idx >= gt_all.shape[0] or (args.max_frames and j >= args.max_frames):
            break
        gt = gt_all[orig_idx]
        healthy = _is_healthy(j, args.healthy_prefix, args.health_mode, args.dropout, args.recover)

        # main.py'deki gibi FramePredictions oluştur (GT string olarak gelir)
        prediction = FramePredictions(
            f"/frame/{orig_idx}/", f"/img/frame_{orig_idx:06d}.jpg", "replay_session",
            str(gt[0]), str(gt[1]), str(gt[2]),
        )
        # detect()'i doğrudan görüntüyle çağır (process() ağ indirme yapardı)
        prediction = model.detect(prediction, "1" if healthy else "0", image=frame)

        # Payload üretimini doğrula (sunucuya gönderilecek format)
        payload = prediction.create_payload("http://127.0.0.1:5000/")
        if (isinstance(payload, dict) and "detected_translations" in payload
                and len(payload["detected_translations"]) == 1):
            payload_ok += 1
            if payload_example is None:
                payload_example = payload

        t = prediction.translations[-1]
        preds.append([float(t.translation_x), float(t.translation_y), float(t.translation_z)])
        gts.append(gt)
        healths.append(healthy)
        j += 1

    preds, gts, healths = np.array(preds), np.array(gts), np.array(healths)
    un = ~healths
    err = np.linalg.norm(preds - gts, axis=1)

    print(f"\nİşlenen kare: {j}")
    print(f"Geçerli payload üretilen kare: {payload_ok}/{j}")
    print("Örnek payload (translation):")
    print("  " + json.dumps(payload_example["detected_translations"][0], ensure_ascii=False))
    print(f"\nSağlıklı segment hatası (echo, ~0 olmalı): {err[healths].mean():.6f} m")
    # Sağlıksızda kestirim GT'den FARKLI olmalı (gerçekten VO kullanıyoruz)
    diff_un = np.linalg.norm(preds[un] - gts[un], axis=1)
    print(f"Sağlıksız kare ort. |pred−GT| (>0 olmalı = VO aktif): {diff_un.mean():.3f} m")
    print(f"\n>>> RESMİ METRİK (health=0): E = {err[un].mean():.3f} m  (medyan {np.median(err[un]):.3f})")
    print(f"    Eksen MAE: X={np.abs(preds[un]-gts[un])[:,0].mean():.2f} "
          f"Y={np.abs(preds[un]-gts[un])[:,1].mean():.2f} "
          f"Z={np.abs(preds[un]-gts[un])[:,2].mean():.2f} m")
    print("\nEntegrasyon kuru provası BAŞARILI ✓" if payload_ok == j else "\nUYARI: bazı payload'lar üretilemedi!")


if __name__ == "__main__":
    main()
