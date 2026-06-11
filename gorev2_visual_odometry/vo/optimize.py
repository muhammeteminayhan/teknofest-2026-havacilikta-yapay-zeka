"""
Global yörünge optimizasyonu (smoothing) — batch çapalı.

Aşama-1 (doğrusal): per-eksen, çapa + odometri residual'larının ağırlıklı en küçük
kareleri. Sağlıksız segmentlerde biriken drift, iki uçtaki GT çapalarına ulaşacak
şekilde optimal (güven-ağırlıklı) dağıtılır.

Matematik (her eksen ayrı, çünkü residual'lar eksen-ayrık):
    min_p  Σ_{sağlıklı i} w_a (p_i − g_i)²  +  Σ_{i≥1} w_o,i ((p_i − p_{i−1}) − v_i)²
Normal denklemler tridiyagonal → scipy.sparse ile hızlı çözüm.

İki çapa arasında (anchor a, b; arada çapa yok) çözüm:
    p_i − p_{i−1} = v_i + λ_i  ;  Σλ = (g_b − g_a) − Σv   → drift ağırlıklı dağıtılır.

Aşama-2 (nonlineer, ayrı fonksiyon): yavaş-değişen yaw-bias + log-ölçek düzeltmesi.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import least_squares

from .aligner import umeyama_2d_ransac
from .batch import BatchData, build_raw_chain


def _rot2(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def _fit_z(raw_z_h: np.ndarray, gt_z_h: np.ndarray):
    """gt_z ≈ cz·raw_z + dz doğrusal fit; raw_z neredeyse sabitse cz=0 (dejenerasyon koruması)."""
    if np.std(raw_z_h) < 1e-9:
        return 0.0, float(np.mean(gt_z_h))
    cz, dz = np.polyfit(raw_z_h, gt_z_h, 1)
    return float(cz), float(dz)


def align_increments(inc: np.ndarray, raw: np.ndarray, healths: np.ndarray,
                     gts: np.ndarray):
    """
    Ham artışları metrik dünya çerçevesine getir (sağlıklı GT'den global hizalama).

    Döndürür: v (N,3) metrik artışlar, ve teşhis için (sim2d, c_z, d_z).
    """
    hi = np.where(healths)[0]
    raw_xy = raw[:, :2]
    sim2d, _ = umeyama_2d_ransac(raw_xy[hi], gts[hi, :2], thresh=3.0,
                                 min_inliers=max(10, len(hi) // 10))
    R = _rot2(sim2d.theta)
    v_xy = sim2d.s * (R @ inc[:, :2].T).T          # per-frame XY metrik artış

    # Z: kümülatif ham z (raw[:,2]) ile gt_z arasında doğrusal regresyon (işaret+ölçek)
    cz, dz = _fit_z(raw[hi, 2], gts[hi, 2])
    v_z = cz * inc[:, 2]
    v = np.column_stack([v_xy, v_z])
    return v, sim2d, float(cz), float(dz)


def correct_drift(v: np.ndarray, healths: np.ndarray, gts: np.ndarray,
                  w_o: np.ndarray, kf_step: int = 50, reg: float = 5.0,
                  max_nfev: int = 60):
    """
    Aşama-2: yavaş-değişen yaw-bias + log-ölçek düzeltmesi (nonlineer).

    VO'nun sistematik açısal/ölçek drift'i, pozisyonda KARESEL hata biriktirir;
    doğrusal smoother bunu (yalnızca ötelediği için) gideremez. Burada keyframe'lerde
    yaw-bias (δ) ve log-ölçek (γ) değişkenleri tanımlanıp, düzeltilmiş yörüngeyi TÜM
    çapalara en iyi oturtacak şekilde optimize edilir.

    Düzeltilmiş artış:  v'_xy_i = exp(γ_i)·R(Φ_i)·v_xy_i ,  v'_z_i = exp(γ_i)·v_z_i
    Φ_i = cumsum(δ_i)  (birikmiş heading düzeltmesi).  p_i = cumsum(v')_i (p_0=0).
    Çapa residual'i: sağlıklı i için (p_i − g_i).

    Döndürür: düzeltilmiş artışlar v' (N,3).
    """
    n = len(v)
    hi = np.where(healths)[0]
    if len(hi) < 5:
        return v.copy()
    # Keyframe konumları ve interpolasyon
    kf = np.arange(0, n, kf_step)
    if kf[-1] != n - 1:
        kf = np.append(kf, n - 1)
    nk = len(kf)
    wsq = np.sqrt(np.clip(w_o[hi], 0.05, None))

    v_xy = v[:, :2]
    v_z = v[:, 2]

    def forward(params):
        delta_kf = params[:nk]
        gamma_kf = params[nk:]
        delta = np.interp(np.arange(n), kf, delta_kf)
        gamma = np.interp(np.arange(n), kf, gamma_kf)
        Phi = np.cumsum(delta)
        eg = np.exp(gamma)
        c, s = np.cos(Phi), np.sin(Phi)
        # R(Phi)·v_xy, per-frame
        rx = c * v_xy[:, 0] - s * v_xy[:, 1]
        ry = s * v_xy[:, 0] + c * v_xy[:, 1]
        cx = eg * rx
        cy = eg * ry
        cz = eg * v_z
        px = np.cumsum(cx)
        py = np.cumsum(cy)
        pz = np.cumsum(cz)
        return px, py, pz, np.column_stack([cx, cy, cz])

    def resid(params):
        px, py, pz, _ = forward(params)
        rx = (px[hi] - gts[hi, 0]) * wsq
        ry = (py[hi] - gts[hi, 1]) * wsq
        rz = (pz[hi] - gts[hi, 2]) * wsq
        # Düzenlileştirme (küçük/yumuşak düzeltme): δ ve γ büyümesin
        rreg_d = reg * params[:nk]
        rreg_g = reg * params[nk:]
        return np.concatenate([rx, ry, rz, rreg_d, rreg_g])

    p0 = np.zeros(2 * nk)
    sol = least_squares(resid, p0, method="lm", max_nfev=max_nfev)
    _, _, _, vcorr = forward(sol.x)
    return vcorr


def _healthy_segments(healths: np.ndarray, min_len: int = 12):
    """Bitişik sağlıklı kare aralıklarını [(start, end_exclusive), ...] döndür."""
    segs = []
    i, n = 0, len(healths)
    while i < n:
        if healths[i]:
            j = i
            while j < n and healths[j]:
                j += 1
            if j - i >= min_len:
                segs.append((i, j))
            i = j
        else:
            i += 1
    return segs


def align_increments_interp(inc: np.ndarray, raw: np.ndarray, healths: np.ndarray,
                            gts: np.ndarray, min_seg: int = 12):
    """
    Segment-bazlı + interpolasyonlu hizalama (kare-başı ölçek & yön).

    Her sağlıklı segmentte 2B-Umeyama ile yerel (ölçek s_k, yön θ_k) hesaplanır;
    bunlar sağlıksız pencereler boyunca interpole edilir. İrtifa (ölçek) ve heading
    düzgün değiştiğinden bu, donmuş global değerin yarattığı pencere-içi ölçek/yön
    drift'ini büyük ölçüde giderir.
    """
    n = len(inc)
    segs = _healthy_segments(healths, min_seg)
    raw_xy = raw[:, :2]
    if len(segs) < 1:                       # yedek: global
        return align_increments(inc, raw, healths, gts)

    centers, ss, ths = [], [], []
    for (a, b) in segs:
        sim, _ = umeyama_2d_ransac(raw_xy[a:b], gts[a:b, :2], thresh=3.0,
                                   min_inliers=max(6, (b - a) // 5))
        centers.append(0.5 * (a + b - 1))
        ss.append(sim.s)
        ths.append(sim.theta)
    centers = np.array(centers)
    ss = np.array(ss)
    ths = np.unwrap(np.array(ths))          # açı interpolasyonu için sar-aç

    idx = np.arange(n)
    if len(centers) == 1:
        s_i = np.full(n, ss[0])
        th_i = np.full(n, ths[0])
    else:
        s_i = np.interp(idx, centers, ss)   # uçlarda düz ekstrapolasyon
        th_i = np.interp(idx, centers, ths)

    c, s = np.cos(th_i), np.sin(th_i)
    rx = c * inc[:, 0] - s * inc[:, 1]
    ry = s * inc[:, 0] + c * inc[:, 1]
    v_xy = np.column_stack([s_i * rx, s_i * ry])

    hi = np.where(healths)[0]
    cz, dz = _fit_z(raw[hi, 2], gts[hi, 2])
    v_z = cz * inc[:, 2]
    v = np.column_stack([v_xy, v_z])

    class _D:                                # teşhis için sahte sim2d
        pass
    d = _D(); d.s = float(np.median(ss)); d.theta = float(np.median(ths)); d.t = np.zeros(2)
    return v, d, float(cz), float(dz)


def _smooth_axis(v_axis: np.ndarray, healthy: np.ndarray, g_axis: np.ndarray,
                 w_o: np.ndarray, w_a: float) -> np.ndarray:
    """Tek eksen tridiyagonal smoother. v_axis[i] = i. kareye ait artış (i≥1)."""
    n = len(v_axis)
    main = np.zeros(n)
    lower = np.zeros(n - 1)      # M[i, i-1]
    rhs = np.zeros(n)
    # Odometri terimleri (i = 1..n-1): residual (p_i - p_{i-1} - v_i), ağırlık w_o[i]
    for i in range(1, n):
        w = w_o[i]
        main[i] += w
        main[i - 1] += w
        lower[i - 1] += -w
        rhs[i] += w * v_axis[i]
        rhs[i - 1] += -w * v_axis[i]
    # Çapa terimleri
    idx = np.where(healthy)[0]
    main[idx] += w_a
    rhs[idx] += w_a * g_axis[idx]
    M = sp.diags([lower, main, lower], offsets=[-1, 0, 1], format="csc")
    return spla.spsolve(M, rhs)


def smooth_trajectory(batch: BatchData, w_anchor: float = 1e4,
                      conf_floor: float = 0.15, stage2: bool = False,
                      kf_step: int = 50, align_mode: str = "interp",
                      return_diag: bool = False):
    """
    Global smoothing. align_mode="interp" (varsayılan): segment-bazlı GT-kalibreli
    ölçek/yön interpolasyonu (pencere-içi ölçek+heading drift'ini giderir); "global":
    tek global Umeyama. stage2=True ise ek yavaş-değişen yaw/ölçek düzeltmesi.
    Sonra doğrusal çapalı smoother. Sağlıklı karelerde GT echo edilir.
    """
    chain = build_raw_chain(batch.fms)
    inc, raw = chain["inc"], chain["raw"]
    healths, gts = batch.healths, batch.gts
    n = len(batch.fms)

    if align_mode == "interp":
        v, sim2d, cz, dz = align_increments_interp(inc, raw, healths, gts)
    else:
        v, sim2d, cz, dz = align_increments(inc, raw, healths, gts)

    # Per-frame güven ağırlığı (inlier ∝); dejenere kareler düşük ağırlık
    med = np.median(batch.n_inliers[batch.n_inliers > 0]) if np.any(batch.n_inliers > 0) else 1.0
    w_o = np.clip(batch.n_inliers / max(med, 1.0), conf_floor, 3.0)
    w_o[~chain["ok"]] = conf_floor

    v_aligned = v
    if stage2:
        v = correct_drift(v, healths, gts, w_o, kf_step=kf_step)

    pred = np.zeros((n, 3))
    for ax in range(3):
        pred[:, ax] = _smooth_axis(v[:, ax], healths, gts[:, ax], w_o, w_anchor)

    # Sağlıklı karelerde GT'yi aynen ver (echo; en iyi skor)
    pred[healths] = gts[healths]

    if return_diag:
        p0_xy = sim2d.s * (_rot2(sim2d.theta) @ raw[:, :2].T).T + sim2d.t
        p0 = np.column_stack([p0_xy, cz * raw[:, 2] + dz])
        return pred, {"p0": p0, "sim2d": sim2d, "cz": cz, "v": v_aligned,
                      "v_corr": v, "w_o": w_o}
    return pred
