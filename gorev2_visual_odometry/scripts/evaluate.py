"""
Çevrimdışı değerlendirme — VO hattını örnek video + GT CSV üzerinde çalıştırır,
resmi metriği (ortalama Öklid hatası) hesaplar ve teşhis grafikleri kaydeder.

Yarışma sağlık deseni simüle edilir:
  - İlk `--healthy-prefix` (varsayılan 450) işlenmiş kare health=1 (GT verilir).
  - Sonrası health=0 (GT GİZLENİR; estimator kendi kestirimini üretir).
Metrik yalnızca health=0 kareler üzerinden hesaplanır (yarışmayı yansıtır).

Örnek video 29.97 fps; yarışma 7.5 fps. `--step 4` ile ~7.5 fps'e alt-örnekleyerek
yarışma koşulu taklit edilir (2255 kare ≈ 2250; 450 sağlıklı + ~1805 sağlıksız).

Kullanım:
  python scripts/evaluate.py --modality rgb --step 4
  python scripts/evaluate.py --modality rgb --step 1 --backend klt
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "configs"))

import config  # noqa: E402
from vo.frame_source import VideoFrameSource, load_translation_csv, read_video_meta  # noqa: E402
from vo.pipeline import build_estimator  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _is_healthy_frame(j: int, args) -> bool:
    """İşlenmiş kare j sağlıklı mı? (health deseni)."""
    if j < args.healthy_prefix:
        return True
    if args.health_mode == "tail":
        return False
    # intermittent: recover(sağlıklı) + dropout(sağlıksız) döngüsü
    pos = (j - args.healthy_prefix) % (args.recover + args.dropout)
    return pos < args.recover


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", choices=["rgb", "thermal"], default="rgb")
    ap.add_argument("--step", type=int, default=4, help="fps alt-örnekleme (4 ≈ 7.5fps).")
    ap.add_argument("--healthy-prefix", type=int, default=config.HEALTHY_PREFIX_FRAMES,
                    help="İlk N işlenmiş kare sağlıklı (GT verilir).")
    ap.add_argument("--backend", default=None, help="klt | orb | sift (varsayılan config).")
    ap.add_argument("--motion", default=None, help="homography | similarity (varsayılan config).")
    ap.add_argument("--no-undistort", action="store_true")
    ap.add_argument("--proc-width", type=int, default=config.PROC_MAX_WIDTH)
    ap.add_argument("--max-frames", type=int, default=0, help="Hızlı test için sınır (0=sınırsız).")
    ap.add_argument("--tag", default="", help="Çıktı dosyası etiketi.")
    # Sağlık deseni
    ap.add_argument("--health-mode", choices=["tail", "intermittent"], default="tail",
                    help="tail: prefix sonrası hep sağlıksız. intermittent: periyodik GPS düşüşü.")
    ap.add_argument("--dropout", type=int, default=300, help="intermittent: sağlıksız pencere boyu.")
    ap.add_argument("--recover", type=int, default=45, help="intermittent: araya giren sağlıklı (re-anchor) boyu.")
    # Parametre geçişleri (config override — hızlı tarama için)
    ap.add_argument("--alt-gain", type=float, default=None)
    ap.add_argument("--alt-smooth", type=float, default=None)
    ap.add_argument("--z-sign", type=float, default=None)
    ap.add_argument("--z-gain", type=float, default=None)
    args = ap.parse_args()

    # Config override'ları (estimator kurulmadan ÖNCE)
    if args.alt_gain is not None: config.ALT_PROP_GAIN = args.alt_gain
    if args.alt_smooth is not None: config.ALT_SMOOTH = args.alt_smooth
    if args.z_sign is not None: config.Z_SIGN = args.z_sign
    if args.z_gain is not None: config.Z_GAIN = args.z_gain

    if args.modality == "rgb":
        video, csv_path = config.RGB_VIDEO, config.RGB_CSV
    else:
        video, csv_path = config.THERMAL_VIDEO, config.THERMAL_CSV

    meta = read_video_meta(video)
    gt_all = load_translation_csv(csv_path)
    print(f"=== Değerlendirme: {args.modality} | {meta.width}x{meta.height} @ {meta.fps:.2f}fps ===")
    print(f"step={args.step} (≈{meta.fps/args.step:.1f}fps), healthy_prefix={args.healthy_prefix}, "
          f"backend={args.backend or config.FEATURE_BACKEND}, motion={args.motion or config.MOTION_MODEL}, "
          f"proc_width={args.proc_width}, undistort={not args.no_undistort}")

    est = build_estimator(
        args.modality, meta.width, meta.height, proc_max_width=args.proc_width,
        undistort=not args.no_undistort, feature_backend=args.backend,
        motion_model=args.motion,
    )
    print(f"İşleme çözünürlüğü: {est.proc_w}x{est.proc_h}, K=\n{np.round(est.K,1)}")

    src = VideoFrameSource(video, step=args.step, max_width=None, to_gray=False)

    preds, gts, healths = [], [], []
    t0 = time.perf_counter()
    j = 0
    for orig_idx, frame in src:
        if orig_idx >= gt_all.shape[0]:
            break
        gt = gt_all[orig_idx]
        healthy = _is_healthy_frame(j, args)
        out = est.process(frame, "1" if healthy else "0",
                          gt_xyz=gt if healthy else None)
        preds.append(out)
        gts.append(gt)
        healths.append(healthy)
        j += 1
        if j % 200 == 0:
            el = time.perf_counter() - t0
            print(f"  {j} kare işlendi ({j/el:.1f} kare/s)...", flush=True)
        if args.max_frames and j >= args.max_frames:
            break

    preds = np.array(preds)
    gts = np.array(gts)
    healths = np.array(healths)
    dt = time.perf_counter() - t0
    print(f"\nToplam {len(preds)} kare, {dt:.1f}s ({len(preds)/dt:.1f} kare/s)")

    # ---- Metrik (resmi: ortalama Öklid hatası, yalnızca health=0) ----
    err = np.linalg.norm(preds - gts, axis=1)
    unhealthy = ~healths
    n_un = int(unhealthy.sum())
    E = float(err[unhealthy].mean()) if n_un else float("nan")
    E_all = float(err.mean())
    ax_mae = np.abs(preds[unhealthy] - gts[unhealthy]).mean(axis=0) if n_un else np.array([np.nan]*3)

    print("\n" + "=" * 56)
    print(f"  RESMİ METRİK (health=0, {n_un} kare):")
    print(f"    Ortalama Öklid hatası E = {E:.3f} m")
    print(f"    Eksen MAE: X={ax_mae[0]:.3f}  Y={ax_mae[1]:.3f}  Z={ax_mae[2]:.3f} m")
    print(f"    Medyan={np.median(err[unhealthy]):.3f}  P90={np.percentile(err[unhealthy],90):.3f}  "
          f"Maks={err[unhealthy].max():.3f} m")
    print(f"  Sağlıklı segment hata (echo, ~0 olmalı): {err[healths].mean():.4f} m")
    print(f"  Tüm kareler ortalama hata: {E_all:.3f} m")
    print(f"  Son Umeyama ölçeği s (≈irtifa): {est.traj.scale:.2f}")
    print("=" * 56)

    # ---- Grafikler ----
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    tag = (args.tag + "_") if args.tag else ""
    base = os.path.join(config.OUTPUTS_DIR, f"eval_{args.modality}_{tag}step{args.step}")
    _plots(preds, gts, healths, err, base, args, E)
    print(f"\nGrafikler kaydedildi: {base}_*.png")


def _plots(preds, gts, healths, err, base, args, E):
    un = ~healths
    # 1) Üstten görünüm (X-Y) yörünge
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.plot(gts[:, 0], gts[:, 1], "g-", lw=1.6, label="GT")
    ax.plot(preds[un, 0], preds[un, 1], "r.", ms=2.5, label="Kestirim (health=0)")
    ax.plot(gts[healths, 0], gts[healths, 1], "b-", lw=2.4, label="Sağlıklı segment")
    ax.plot(gts[0, 0], gts[0, 1], "ks", ms=9, label="Başlangıç (0,0)")
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
    ax.set_title(f"Yörünge (üstten) — {args.modality} step{args.step} — E={E:.2f} m")
    ax.legend(); ax.axis("equal"); ax.grid(alpha=0.3)
    fig.savefig(base + "_xy.png", dpi=110, bbox_inches="tight"); plt.close(fig)

    # 2) Eksen-bazlı zaman serisi
    fig, axs = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    t = np.arange(len(preds))
    for k, name in enumerate("XYZ"):
        axs[k].plot(t, gts[:, k], "g-", lw=1.4, label="GT")
        axs[k].plot(t, preds[:, k], "r-", lw=0.9, label="Kestirim")
        axs[k].axvline(args.healthy_prefix, color="b", ls="--", alpha=0.6,
                       label="health=0 başlangıcı")
        axs[k].set_ylabel(f"{name} [m]"); axs[k].grid(alpha=0.3)
        if k == 0:
            axs[k].legend(ncol=3, fontsize=9)
    axs[-1].set_xlabel("İşlenmiş kare")
    axs[0].set_title(f"Eksen-bazlı pozisyon — {args.modality} step{args.step}")
    fig.savefig(base + "_axes.png", dpi=110, bbox_inches="tight"); plt.close(fig)

    # 3) Hata büyüklüğü zaman serisi
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(t, err, "m-", lw=0.9)
    ax.axvline(args.healthy_prefix, color="b", ls="--", alpha=0.6)
    ax.set_xlabel("İşlenmiş kare"); ax.set_ylabel("Öklid hatası [m]")
    ax.set_title(f"Hata büyüklüğü zamanla — health=0 ortalama E={E:.2f} m")
    ax.grid(alpha=0.3)
    fig.savefig(base + "_error.png", dpi=110, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
