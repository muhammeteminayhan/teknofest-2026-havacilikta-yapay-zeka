"""
Batch (toplu global optimizasyon) değerlendirmesi.

Çevrimiçi `evaluate.py` ile aynı veri/health-deseni/metrik kullanılır; fark:
tüm kareler önce işlenir (pass-1), sonra global çapalı smoothing uygulanır.
Sağlıksız segmentler iki uçtan GT çapalarıyla sınırlandığı için drift dağıtılır.

Kullanım:
  python scripts/evaluate_batch.py --modality rgb --step 4 --health-mode intermittent
  python scripts/evaluate_batch.py --modality rgb --step 4 --health-mode tail
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
sys.path.insert(0, _HERE)                       # evaluate.py importu için
sys.path.insert(0, os.path.join(_ROOT, "configs"))

import config  # noqa: E402
from vo.frame_source import VideoFrameSource, load_translation_csv, read_video_meta  # noqa: E402
from vo.pipeline import build_estimator  # noqa: E402
from vo.batch import (collect_frame_motions, make_batch_data,  # noqa: E402
                      motions_to_arrays, arrays_to_motions)
from vo.optimize import smooth_trajectory  # noqa: E402

# evaluate.py'den health deseni + grafik fonksiyonlarını yeniden kullan
from evaluate import _is_healthy_frame, _plots  # noqa: E402


def main():
    # Windows konsolu cp1254 → Unicode (→, ≈ vb.) çökmesini önle.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", choices=["rgb", "thermal"], default="rgb")
    ap.add_argument("--step", type=int, default=4)
    ap.add_argument("--healthy-prefix", type=int, default=config.HEALTHY_PREFIX_FRAMES)
    ap.add_argument("--health-mode", choices=["tail", "intermittent"], default="intermittent")
    ap.add_argument("--dropout", type=int, default=300)
    ap.add_argument("--recover", type=int, default=45)
    ap.add_argument("--w-anchor", type=float, default=1e4)
    ap.add_argument("--align-mode", choices=["interp", "global"], default="interp",
                    help="interp: segment-bazlı ölçek/yön interpolasyonu; global: tek Umeyama.")
    ap.add_argument("--stage2", action="store_true", help="Ek global yaw/ölçek düzeltmesi (genelde gereksiz).")
    ap.add_argument("--kf-step", type=int, default=50, help="Aşama-2 keyframe aralığı.")
    ap.add_argument("--backend", default=None)
    ap.add_argument("--motion", default=None)
    ap.add_argument("--proc-width", type=int, default=config.PROC_MAX_WIDTH)
    ap.add_argument("--max-features", type=int, default=None, help="Özellik sayısı (config override).")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--cache", action="store_true", help="Pass-1 sonuçlarını önbelleğe al/oku.")
    ap.add_argument("--tag", default="batch")
    ap.add_argument("--loop", action="store_true",
                    help="Loop closure: çapasız kuyruk için görsel revisit eşleme (DISK+LightGlue).")
    ap.add_argument("--loop-min-inliers", type=int, default=50)
    ap.add_argument("--loop-weight", type=float, default=50.0,
                    help="Sanal çapa ağırlığı (odometri ~1, gerçek GT 1e4; yaklaşık çapa için orta değer).")
    ap.add_argument("--loop-db-stride", type=int, default=10)
    ap.add_argument("--loop-query-stride", type=int, default=10)
    # Yoğun harita-tabanlı mutlak konumlandırma (AVL/DSMAC; --loop'un güçlü genellemesi)
    ap.add_argument("--map", action="store_true",
                    help="Yoğun harita konumlandırma: her sağlıksız kareyi GT'li harita karelerine konumlandır.")
    ap.add_argument("--map-db-stride", type=int, default=6)
    ap.add_argument("--map-query-stride", type=int, default=3)
    ap.add_argument("--map-min-inliers", type=int, default=40)
    ap.add_argument("--map-weight", type=float, default=50.0)
    ap.add_argument("--map-radius", type=float, default=120.0,
                    help="Uzamsal kapılama yarıçapı (m); 0 = tüm haritaya eşle.")
    args = ap.parse_args()

    if args.max_features is not None:
        config.MAX_FEATURES = args.max_features

    video = config.RGB_VIDEO if args.modality == "rgb" else config.THERMAL_VIDEO
    csv_path = config.RGB_CSV if args.modality == "rgb" else config.THERMAL_CSV
    meta = read_video_meta(video)
    gt_all = load_translation_csv(csv_path)
    print(f"=== BATCH değerlendirme: {args.modality} {meta.width}x{meta.height} @ {meta.fps:.2f}fps ===")
    print(f"step={args.step}, healthy_prefix={args.healthy_prefix}, mode={args.health_mode}"
          f"{f' (dropout={args.dropout}/recover={args.recover})' if args.health_mode=='intermittent' else ''}, "
          f"w_anchor={args.w_anchor:g}")

    est = build_estimator(args.modality, meta.width, meta.height, proc_max_width=args.proc_width,
                          feature_backend=args.backend, motion_model=args.motion)

    max_j = args.max_frames if args.max_frames else None
    t0 = time.perf_counter()

    # Pass-1 önbelleği (health/optimizer parametrelerinden bağımsız)
    cache_path = os.path.join(
        config.OUTPUTS_DIR,
        f"_cache_pass1_{args.modality}_step{args.step}_{args.backend or config.FEATURE_BACKEND}"
        f"_{args.motion or config.MOTION_MODEL}_w{args.proc_width}_f{config.MAX_FEATURES}_max{args.max_frames}.npz")

    if args.cache and os.path.exists(cache_path):
        a = np.load(cache_path)
        fms = arrays_to_motions(a)
        orig_idxs = a["orig"].tolist()
        print(f"Pass-1 ÖNBELLEKTEN okundu ({len(fms)} kare): {os.path.basename(cache_path)}")
    else:
        src = VideoFrameSource(video, step=args.step, max_width=None, to_gray=False)

        def frames_iter():
            j = 0
            for oi, fr in src:
                if oi >= gt_all.shape[0] or (max_j and j >= max_j):
                    break
                yield oi, fr
                j += 1

        def _prog(j):
            if j % 300 == 0 and j:
                print(f"  pass-1: {j} kare ({j/(time.perf_counter()-t0):.1f}/s)...", flush=True)

        fms, orig_idxs = collect_frame_motions(est, frames_iter(), progress=_prog)
        if args.cache:
            os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
            arr = motions_to_arrays(fms)
            arr["orig"] = np.array(orig_idxs)
            np.savez(cache_path, **arr)
            print(f"Pass-1 önbelleğe yazıldı: {os.path.basename(cache_path)}")
    t_pass1 = time.perf_counter() - t0

    batch = make_batch_data(fms, orig_idxs,
                            health_fn=lambda j: _is_healthy_frame(j, args),
                            gt_fn=lambda j, oi: gt_all[oi])
    n = len(batch.fms)

    t1 = time.perf_counter()
    pred, diag = smooth_trajectory(batch, w_anchor=args.w_anchor,
                                   stage2=args.stage2, kf_step=args.kf_step,
                                   align_mode=args.align_mode, return_diag=True)
    t_opt = time.perf_counter() - t1

    # İşlenmiş kare j → orijinal kare j*step (VideoFrameSource her step'te bir kare verir)
    gts = gt_all[np.arange(n) * args.step]
    healths = batch.healths
    un = ~healths
    err = np.linalg.norm(pred - gts, axis=1)
    E = float(err[un].mean())
    ax_mae = np.abs(pred[un] - gts[un]).mean(axis=0)

    # Karşılaştırma: hizalanmış ama smoothing'siz (p0) — drift dağıtımının katkısı
    p0 = diag["p0"]
    err0 = np.linalg.norm(p0 - gts, axis=1)
    E0 = float(err0[un].mean())

    print(f"\nPass-1 {t_pass1:.1f}s ({n/t_pass1:.1f}/s), optimizasyon {t_opt:.2f}s")
    print("=" * 60)
    print(f"  RESMİ METRİK (health=0, {int(un.sum())} kare):")
    print(f"    Ortalama Öklid hatası E = {E:.3f} m   (smoothing'siz baz: {E0:.1f} m)")
    print(f"    Eksen MAE: X={ax_mae[0]:.3f}  Y={ax_mae[1]:.3f}  Z={ax_mae[2]:.3f} m")
    print(f"    Medyan={np.median(err[un]):.3f}  P90={np.percentile(err[un],90):.3f}  "
          f"Maks={err[un].max():.3f} m")
    print(f"  Sağlıklı segment (echo): {err[healths].mean():.4f} m")
    print(f"  Hizalama olcegi s~irtifa={diag['sim2d'].s:.2f}, z-olcek cz={diag['cz']:.2f}")
    print("=" * 60)

    # ---- Loop closure (opsiyonel): çapasız kuyruk için görsel revisit eşleme ----
    if args.loop:
        from vo.loop_closure import LoopCloser, build_keyframes, LoopAnchor
        # Çapa önbelleği: (idx, xyz, n_inlier) ağırlıktan bağımsız → ağırlık taraması ucuz.
        lc_cache = os.path.join(
            config.OUTPUTS_DIR,
            f"_cache_loop_{args.modality}_step{args.step}_{args.health_mode}"
            f"_db{args.loop_db_stride}_q{args.loop_query_stride}_mi{args.loop_min_inliers}.npz")
        tlc = time.perf_counter()
        if args.cache and os.path.exists(lc_cache):
            la = np.load(lc_cache)
            a_idx, a_xyz, a_ninl = la["idx"], la["xyz"], la["ninl"]
            print(f"\nLoop closure: çapalar ÖNBELLEKTEN ({len(a_idx)} loop).")
        else:
            print("\nLoop closure: anahtar-kareler çıkarılıyor (DISK+LightGlue)...")
            closer = LoopCloser(proc_max_width=args.proc_width)
            src2 = VideoFrameSource(video, step=args.step, max_width=None, to_gray=False)

            def lc_iter():
                j = 0
                for oi, fr in src2:
                    if j >= n:
                        break
                    yield j, fr
                    j += 1

            db, queries = build_keyframes(closer, lc_iter(), healths, gts,
                                          db_stride=args.loop_db_stride,
                                          query_stride=args.loop_query_stride)
            det = closer.detect(db, queries, min_inliers=args.loop_min_inliers)
            a_idx = np.array([a.idx for a in det], dtype=int)
            a_xyz = np.array([a.xyz for a in det], dtype=float).reshape(-1, 3)
            a_ninl = np.array([a.n_inlier for a in det], dtype=int)
            print(f"  DB={len(db)} anahtar-kare, sorgu={len(queries)}, "
                  f"bulunan loop={len(a_idx)} ({time.perf_counter()-tlc:.0f}s)")
            if args.cache:
                os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
                np.savez(lc_cache, idx=a_idx, xyz=a_xyz, ninl=a_ninl)

        anchors = [LoopAnchor(int(i), x, args.loop_weight, int(ni))
                   for i, x, ni in zip(a_idx, a_xyz, a_ninl)]
        pred2 = smooth_trajectory(batch, w_anchor=args.w_anchor, align_mode=args.align_mode,
                                  loop_anchors=anchors)
        err2 = np.linalg.norm(pred2 - gts, axis=1)
        E2 = float(err2[un].mean())
        print(f"  loop sayısı={len(a_idx)}, ağırlık={args.loop_weight:g} "
              f"({time.perf_counter()-tlc:.0f}s)")
        if len(a_idx):
            # Yalnızca eşleşen (sanal çapalı) karelerdeki kazanç:
            print(f"  Loop kareleri E: {err[a_idx].mean():.1f}m -> {err2[a_idx].mean():.1f}m")
        print(f"  >>> RESMİ METRİK (loop closure): E = {E2:.3f} m  "
              f"(loop'suz: {E:.3f} m)")
        print(f"      Medyan={np.median(err2[un]):.3f}  Maks={err2[un].max():.3f} m")
        print("=" * 60)
        pred, err, E = pred2, err2, E2          # grafikler loop-düzeltilmiş sonucu göstersin

    # ---- Yoğun harita-tabanlı mutlak konumlandırma (AVL/DSMAC) ----
    if args.map:
        from vo.loop_closure import LoopCloser, LoopAnchor
        from vo.map_localize import MapLocalizer, build_map, localize_all
        from vo.batch import build_raw_chain
        mp_cache = os.path.join(
            config.OUTPUTS_DIR,
            f"_cache_map_{args.modality}_step{args.step}_{args.health_mode}"
            f"{('_d'+str(args.dropout)) if args.health_mode=='intermittent' else ''}"
            f"_db{args.map_db_stride}_q{args.map_query_stride}_mi{args.map_min_inliers}.npz")
        tmp = time.perf_counter()
        if args.cache and os.path.exists(mp_cache):
            ma = np.load(mp_cache)
            a_idx, a_xyz, a_ninl = ma["idx"], ma["xyz"], ma["ninl"]
            print(f"\nHarita konumlandırma: çapalar ÖNBELLEKTEN ({len(a_idx)} kare).")
        else:
            print("\nHarita konumlandırma: harita+sorgu çıkarılıyor (DISK+LightGlue)...")
            chain = build_raw_chain(batch.fms)
            theta = float(diag["sim2d"].theta); s = float(diag["sim2d"].s)
            closer = LoopCloser(proc_max_width=args.proc_width)
            src3 = VideoFrameSource(video, step=args.step, max_width=None, to_gray=False)

            def mp_iter():
                j = 0
                for oi, fr in src3:
                    if j >= n:
                        break
                    yield j, fr
                    j += 1

            map_kfs, queries = build_map(closer, mp_iter(), healths, gts, chain["phi"],
                                         theta, db_stride=args.map_db_stride,
                                         query_stride=args.map_query_stride)
            loc = MapLocalizer(closer, mpp0=max(s, 1.0) / 1000.0,
                               min_inliers=args.map_min_inliers)
            resid = loc.calibrate(map_kfs)
            print(f"  harita={len(map_kfs)} kare, sorgu={len(queries)}; "
                  f"kalibrasyon: flip={loc.flip.tolist()} mpp={loc.mpp:.4f} artık={resid}")
            anchors0 = localize_all(loc, map_kfs, queries, pred_prior=pred,
                                    radius=args.map_radius)
            a_idx = np.array([a.idx for a in anchors0], dtype=int)
            a_xyz = np.array([a.xyz for a in anchors0], dtype=float).reshape(-1, 3)
            a_ninl = np.array([a.n_inlier for a in anchors0], dtype=int)
            print(f"  konumlanan kare={len(a_idx)} ({time.perf_counter()-tmp:.0f}s)")
            if args.cache:
                os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
                np.savez(mp_cache, idx=a_idx, xyz=a_xyz, ninl=a_ninl)

        anchors = [LoopAnchor(int(i), x, args.map_weight, int(ni))
                   for i, x, ni in zip(a_idx, a_xyz, a_ninl)]
        predm = smooth_trajectory(batch, w_anchor=args.w_anchor, align_mode=args.align_mode,
                                  loop_anchors=anchors)
        errm = np.linalg.norm(predm - gts, axis=1)
        Em = float(errm[un].mean())
        if len(a_idx):
            print(f"  Konumlanan karelerde E: {err[a_idx].mean():.1f}m -> {errm[a_idx].mean():.1f}m")
        print(f"  >>> RESMİ METRİK (harita konumlandırma): E = {Em:.3f} m  (öncesi: {E:.3f} m)")
        print(f"      Medyan={np.median(errm[un]):.3f}  Maks={errm[un].max():.3f} m")
        print("=" * 60)
        pred, err, E = predm, errm, Em

    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    base = os.path.join(config.OUTPUTS_DIR, f"evalB_{args.modality}_{args.tag}_{args.health_mode}_step{args.step}")
    _plots(pred, gts, healths, err, base, args, E)
    print(f"Grafikler: {base}_*.png")


if __name__ == "__main__":
    main()
