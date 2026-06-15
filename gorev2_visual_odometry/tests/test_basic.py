"""
Temel birim testleri — düz `python tests/test_basic.py` ile veya pytest ile çalışır.

Kapsam: kalibrasyon ayrıştırma, Umeyama (3B/2B) geri-kurtarım, hareket kestirimi
(sentetik öteleme), ve VO hattı duman testi (sentetik kaydırılmış doku).
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "configs"))


def test_calibration_parse():
    from vo.calibration import parse_calibration_file
    cams = parse_calibration_file()
    assert set(cams) >= {"thermal", "rgb_4k", "rgb_1080p"}, cams.keys()
    th = cams["thermal"]
    assert (th.width, th.height) == (640, 512)          # ImageSize=[512 640]=[H W]
    assert abs(th.fx - 731.7965) < 1e-3
    K = cams["rgb_1080p"].K
    assert K[0, 0] > 1000 and K[2, 2] == 1.0 and K[0, 1] == 0.0
    print("test_calibration_parse PASS")


def test_umeyama_3d():
    from vo.aligner import umeyama
    rng = np.random.default_rng(0)
    X = rng.normal(size=(150, 3)) * 4
    theta = 0.6
    R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    s, t = 2.5, np.array([3.0, -1.0, 0.5])
    Y = (s * (R @ X.T)).T + t
    sim = umeyama(X, Y)
    assert abs(sim.s - s) < 1e-3
    assert np.linalg.norm(sim.apply(X) - Y) < 1e-6
    print("test_umeyama_3d PASS")


def test_umeyama_2d():
    from vo.aligner import umeyama_2d
    rng = np.random.default_rng(1)
    X = rng.normal(size=(120, 2)) * 5
    th = -0.9
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    s, t = 3.3, np.array([2.0, 7.0])
    Y = (s * (R @ X.T)).T + t
    sim = umeyama_2d(X, Y)
    assert abs(sim.s - s) < 1e-3 and abs(((sim.theta - th + np.pi) % (2 * np.pi)) - np.pi) < 1e-3
    print("test_umeyama_2d PASS")


def test_motion_translation():
    """Saf piksel ötelemesi → raw_dx/dy doğru işaret ve büyüklük (≈ −shift/f)."""
    from vo.motion import estimate_similarity_motion
    rng = np.random.default_rng(2)
    K = np.array([[900.0, 0, 640.0], [0, 900.0, 360.0], [0, 0, 1.0]])
    pts_prev = rng.uniform([100, 100], [1180, 620], size=(300, 2)).astype(np.float32)
    shift = np.array([12.0, -7.0], dtype=np.float32)   # sahne sağa-yukarı kayar
    pts_cur = pts_prev + shift
    fm = estimate_similarity_motion(pts_prev, pts_cur, K, min_inliers=30)
    assert fm.ok
    # raw_dx = -shift_x/fx (görüntü kayması kameranın tersi)
    assert abs(fm.raw_dx - (-shift[0] / K[0, 0])) < 1e-3
    assert abs(fm.raw_dy - (-shift[1] / K[1, 1])) < 1e-3
    assert abs(fm.dyaw) < 1e-2 and abs(fm.scale - 1.0) < 1e-2
    print("test_motion_translation PASS")


def test_pipeline_smoke():
    """Sentetik kaydırılmış doku ile VO hattı hatasız çalışır ve health mantığı doğru."""
    import cv2
    from vo.pipeline import VOEstimator
    from vo.calibration import CameraModel, parse_calibration_file, intrinsics_for_resolution

    rng = np.random.default_rng(3)
    W, H = 640, 480
    base = (rng.uniform(0, 255, size=(H + 80, W + 80))).astype(np.uint8)
    base = cv2.GaussianBlur(base, (3, 3), 0)

    cams = parse_calibration_file()
    intr = intrinsics_for_resolution(cams, W, H, "rgb")
    est = VOEstimator(CameraModel(intr), (W, H), undistort=False, feature_backend="klt",
                      motion_model="similarity")

    # Kamerayı +x yönünde kaydır → görüntü sola kayar; GT'yi de buna göre ver
    n = 40
    outs = []
    for i in range(n):
        dx = 2 * i            # piksel ofset (sahne penceresi)
        frame = base[40:40 + H, 40 + dx:40 + dx + W]
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        healthy = i < 25
        gt = (0.1 * i, 0.0, 0.0)          # sahte GT (yalnızca işaret/akış testi)
        out = est.process(frame, "1" if healthy else "0", gt_xyz=gt if healthy else None)
        outs.append(out)
    assert len(outs) == n
    # Sağlıklı karelerde echo: çıktı == GT
    assert abs(outs[10][0] - 1.0) < 1e-6
    # Sağlıksız karelerde bir kestirim üretilmiş (NaN/inf değil)
    assert np.all(np.isfinite(outs[-1]))
    print("test_pipeline_smoke PASS")


def test_linear_smoother():
    """Sabit-bias drift'li zincir + iki uç çapa → smoother drift'i dağıtıp gerçeği kurtarır."""
    from vo.optimize import _smooth_axis
    n = 100
    true_inc = np.ones(n)              # gerçek artış = 1 → p_true = 0..99
    true_inc[0] = 0
    p_true = np.cumsum(true_inc)
    v = true_inc + 0.1                 # her adımda +0.1 sistematik drift
    v[0] = 0
    healthy = np.zeros(n, dtype=bool)
    healthy[0] = healthy[-1] = True    # iki uç çapa
    g = np.zeros(n); g[0] = p_true[0]; g[-1] = p_true[-1]
    w_o = np.ones(n)
    p = _smooth_axis(v, healthy, g, w_o, w_a=1e5)
    assert abs(p[0] - p_true[0]) < 1e-2 and abs(p[-1] - p_true[-1]) < 1e-2
    # Drift dağıtıldığı için orta hata, ham drift'ten (mid ≈ 0.1*50=5) çok küçük olmalı
    assert np.max(np.abs(p - p_true)) < 0.6, np.max(np.abs(p - p_true))
    print("test_linear_smoother PASS")


def test_batch_smooth():
    """Sentetik BatchData → smooth_trajectory çalışır, sağlıklı echo, sonlu çıktı."""
    from vo.motion import FrameMotion
    from vo.batch import BatchData
    from vo.optimize import smooth_trajectory
    n = 200
    fms = [FrameMotion.zero()]
    for i in range(1, n):
        fms.append(FrameMotion(0.01, 0.002, 0.0005, 0.0, 1.0, 500, 0.9, True))
    healths = np.zeros(n, dtype=bool)
    healths[:60] = True; healths[150:] = True       # iki çapa bloğu, arada boşluk
    # GT: ham zincirin 50x ölçekli + döndürülmüş hali (bilinen benzerlik)
    from vo.batch import build_raw_chain
    raw = build_raw_chain(fms)["raw"]
    th = 0.5
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    gts = np.full((n, 3), np.nan)
    xy = 50.0 * (R @ raw[:, :2].T).T
    gts[healths, :2] = xy[healths]
    gts[healths, 2] = -5.0 * raw[healths, 2]
    batch = BatchData(fms, healths, gts, np.full(n, 500.0))
    pred = smooth_trajectory(batch, align_mode="interp")
    assert pred.shape == (n, 3) and np.all(np.isfinite(pred))
    assert np.linalg.norm(pred[healths] - gts[healths], axis=1).mean() < 1e-6  # echo
    # Sağlıksız blokta (60:150) kestirim gerçeğe yakın olmalı (bilinen benzerlikten)
    un = slice(60, 150)
    gt_un = 50.0 * (R @ raw[un, :2].T).T
    err = np.linalg.norm(pred[un, :2] - gt_un, axis=1)
    assert err.mean() < 1.0, err.mean()
    print("test_batch_smooth PASS")


def test_loop_anchor():
    """Sanal çapa (loop closure) mekanizması: bir kareyi verilen mutlak değere çeker."""
    from vo.optimize import _smooth_axis
    n = 100
    v = np.ones(n); v[0] = 0           # sabit hareket → çapasız p[99]≈99
    healthy = np.zeros(n, dtype=bool); healthy[0] = True
    g = np.zeros(n); g[0] = 0.0
    w_o = np.ones(n)
    p_no = _smooth_axis(v, healthy, g, w_o, 1e5)
    assert abs(p_no[99] - 99) < 1.0
    # 99. kareye sanal çapa (değer=50) → o kare 50'ye çekilmeli
    extra = (np.array([99]), np.array([50.0]), np.array([1e5]))
    p_ex = _smooth_axis(v, healthy, g, w_o, 1e5, extra=extra)
    assert abs(p_ex[99] - 50) < 1.0, p_ex[99]
    print("test_loop_anchor PASS")


def test_map_localize_geometry():
    """Harita konumlandırma geometrisi (torch'suz): homografi ofseti → doğru mutlak konum."""
    from vo.map_localize import MapLocalizer, MapKF
    from vo.loop_closure import _KF
    loc = MapLocalizer(closer=None, mpp0=1.0)   # __init__ yalnızca closer'ı saklar
    loc.mpp = 2.0
    loc.flip = np.array([1.0, -1.0])
    hw = (480, 640)
    m = MapKF(kf=_KF(5, None, None, hw, np.array([10.0, 20.0, 5.0])), heading=0.0)
    q = _KF(50, None, None, hw, None)
    # H: query merkezi keyframe merkezine göre (12,-7) piksel kayıyor
    H = np.array([[1, 0, 12.0], [0, 1, -7.0], [0, 0, 1.0]])
    g = loc._predict_from(q, m, H)
    # flip*d_px=[12,7]; off=2*R(0)*[12,7]=[24,14]; G=[10+24,20+14]
    assert abs(g[0] - 34.0) < 1e-6 and abs(g[1] - 34.0) < 1e-6, g
    # Heading=90°: R(pi/2)*[12,7]=[-7,12]; off=2*[-7,12]=[-14,24]; G=[10-14,20+24]
    m2 = MapKF(kf=_KF(5, None, None, hw, np.array([10.0, 20.0, 5.0])), heading=np.pi / 2)
    g2 = loc._predict_from(q, m2, H)
    assert abs(g2[0] - (-4.0)) < 1e-5 and abs(g2[1] - 44.0) < 1e-5, g2
    print("test_map_localize_geometry PASS")


def run_all():
    tests = [test_calibration_parse, test_umeyama_3d, test_umeyama_2d,
             test_motion_translation, test_pipeline_smoke,
             test_linear_smoother, test_batch_smooth, test_loop_anchor,
             test_map_localize_geometry]
    failed = 0
    for t in tests:
        try:
            t()
        except Exception as e:
            failed += 1
            print(f"{t.__name__} FAIL: {e}")
    print(f"\n{len(tests)-failed}/{len(tests)} test geçti.")
    return failed == 0


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
