"""
Faz 0 doğrulama scripti.

Yapar:
  1. Video meta bilgisini (çözünürlük, fps, kare sayısı) raporlar.
  2. Gerçek kare sayısını GT CSV satır sayısı ile karşılaştırır (1:1 hizalama).
  3. Çözünürlüğe uygun kamera kalibrasyonunu seçer.
  4. Birkaç örnek kareyi ham + undistort olarak outputs/ altına kaydeder
     (kalibrasyon/undistort görsel denetimi için).

Kullanım:
  python scripts/extract_frames.py --modality rgb
  python scripts/extract_frames.py --modality thermal --no-count
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

# Proje köklerini path'e ekle
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "configs"))

import config  # noqa: E402
from vo.calibration import parse_calibration_file, intrinsics_for_resolution, CameraModel  # noqa: E402
from vo.frame_source import read_video_meta, load_translation_csv, count_video_frames  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", choices=["rgb", "thermal"], default="rgb")
    ap.add_argument("--no-count", action="store_true",
                    help="Kareleri tek tek sayma (uzun sürer); kapsayıcı meta değerini kullan.")
    ap.add_argument("--samples", type=int, default=3, help="Kaydedilecek örnek kare sayısı.")
    args = ap.parse_args()

    if args.modality == "rgb":
        video, csv_path = config.RGB_VIDEO, config.RGB_CSV
    else:
        video, csv_path = config.THERMAL_VIDEO, config.THERMAL_CSV

    print(f"=== Modalite: {args.modality} ===")
    print(f"Video : {video}")
    print(f"CSV   : {csv_path}")

    meta = read_video_meta(video)
    print(f"\n[Video meta] {meta.width}x{meta.height}, fps={meta.fps:.4f}, "
          f"kapsayıcı kare sayısı={meta.frame_count}")
    print(f"  Süre ≈ {meta.frame_count / meta.fps:.1f} s")

    gt = load_translation_csv(csv_path)
    print(f"[GT CSV] {gt.shape[0]} satır (kare). "
          f"X[{gt[:,0].min():.2f},{gt[:,0].max():.2f}] "
          f"Y[{gt[:,1].min():.2f},{gt[:,1].max():.2f}] "
          f"Z[{gt[:,2].min():.2f},{gt[:,2].max():.2f}] (m)")

    if args.no_count:
        real_n = meta.frame_count
        print("\n[Kare sayımı atlandı — kapsayıcı değeri kullanılıyor]")
    else:
        print("\n[Gerçek kareler sayılıyor... (sıralı okuma)]")
        real_n = count_video_frames(video)
        print(f"  Gerçek okunan kare sayısı = {real_n}")

    aligned = (real_n == gt.shape[0])
    print(f"\n>>> HİZALAMA: video {real_n} kare vs CSV {gt.shape[0]} satır -> "
          f"{'TAMAM (1:1)' if aligned else 'UYUMSUZ! offset='+str(real_n-gt.shape[0])}")

    # Kalibrasyon seçimi
    cams = parse_calibration_file()
    intr = intrinsics_for_resolution(cams, meta.width, meta.height, args.modality)
    print(f"\n[Seçilen kalibrasyon] {intr}")
    if (intr.width, intr.height) != (meta.width, meta.height):
        print(f"  NOT: kalibrasyon boyutu video boyutundan farklı, ölçeklendi.")

    cam = CameraModel(intr)
    print(f"  Undistort sonrası new_K =\n{np.round(cam.new_K, 2)}")

    # Örnek kareleri kaydet (ham + undistort)
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    sample_dir = os.path.join(config.OUTPUTS_DIR, f"samples_{args.modality}")
    os.makedirs(sample_dir, exist_ok=True)

    cap = cv2.VideoCapture(video)
    sample_idxs = np.linspace(0, max(0, real_n - 1), args.samples).astype(int)
    sample_set = set(int(i) for i in sample_idxs)
    idx = 0
    saved = 0
    while saved < len(sample_set):
        ok, frame = cap.read()
        if not ok:
            break
        if idx in sample_set:
            raw_p = os.path.join(sample_dir, f"frame_{idx:06d}_raw.jpg")
            und_p = os.path.join(sample_dir, f"frame_{idx:06d}_undist.jpg")
            cv2.imwrite(raw_p, frame)
            cv2.imwrite(und_p, cam.undistort_image(frame))
            saved += 1
        idx += 1
    cap.release()
    print(f"\n[Örnek kareler kaydedildi] {saved} adet -> {sample_dir}")
    print("    (ham vs undist karşılaştırarak distorsiyon düzeltmesini gözle doğrulayın)")


if __name__ == "__main__":
    main()
