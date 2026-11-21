# -*- coding: utf-8 -*-
"""D1+D2 — ORNEKLEM-DISI HIZALAMA TESTI (salt tani; uretim kodu degismez).

SORU
----
Kamera->dunya koprusu bugun Procrustes ile kuruluyor: **dik** M + tek skaler c.
Depo, kalan hatanin ANIZOTROPIK oldugunu ve Procrustes'in bunu soguramayacagini
kendi yaziyor (`vo_pipeline.py:225-243`): homografinin duzlem-ici bileseni kamera
xy-duzlemine dik izdusurulduğu icin yalniz egim ekseninde cos(egim) kadar buzuluyor.
Buna ragmen **genel (dik olmayan) 2x2 hizalama hic denenmemis**.

Bu betik onu UYGULAMADAN sorar: daha zengin bir harita GT'ye **orneklem disinda**
gercekten daha iyi mi oturuyor?

NEDEN ORNEKLEM-DISI
-------------------
Daha cok serbestlik orneklem ICINDE her zaman daha iyi oturur — o olcum anlamsizdir.
Ayrica bu depoda 4 serbestlikli bir hizalama denemesi (`merdiven_hizalama`) tam da
belirlenemezlik yuzunden coktu (sinav penceresi 2: 17,19 -> 25,97 m). Bu yuzden:
pencerede fit et, **sonraki 12 ornekte** puanla.

ETKIN ORNEK SAYISI (ESS) UYARISI
--------------------------------
`smooth_alpha=0.6` EMA'si ardisik e0'lari iliskilendirir; 12 ornek ~5 etkin ornege
duser. Dort parametreyi 5 etkin ornekle kestirmek `merdiven_hizalama`'yi olduren
sayisal rejimdir. Bu yuzden ESS de raporlanir.

ADAYLAR
-------
  P  : Procrustes (mevcut)                 — dik M + skaler c
  S  : det-normalize simetrik duzeltme     — M·(I + a·diag(1,-1) + b·antidiag) , +2 SD
  A  : genel 2x2 en kucuk kareler          — 4 SD (tavan olcumu, sevk adayi degil)

D2 — MEKANIZMA
--------------
S'nin ogrendigi anizotropi (eps, theta) gercekten EGIMDEN mi geliyor?
  eps ~ (1 - cos t)/2   ·   theta ~ hom_normal'in azimutu (atan2(n_y, n_x))
Iliski yoksa S egimi degil baska bir seyi (olcek, merdiven uyusmazligi) yutuyordur.

CALISTIR
--------
    .venv\\Scripts\\python gorev2\\tani_hizalama.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

GOREV2 = Path(__file__).resolve().parent
sys.path.insert(0, str(GOREV2))

import nokta_onbellek as NO                                        # noqa: E402
from nokta_onbellek import yatak_yukle, yatak_npz                  # noqa: E402
from kabul import yarisma_takvimi                                  # noqa: E402
from yontem_sinavi import oransal_takvim                           # noqa: E402
from gorev2_src import evaluation                                  # noqa: E402
from gorev2_src.vo_pipeline import EstimatorConfig, VoLadderEstimator, rot2d  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

TUM = ["video1", "video2", "yarisma", "v24",
       "v25_rgb_a", "v25_rgb_b", "v25_termal_a", "v25_termal_b"]

D1 = np.array([[1.0, 0.0], [0.0, -1.0]])      # saf uzama/kisalma
D2 = np.array([[0.0, 1.0], [1.0, 0.0]])       # 45 derece donmus uzama


class TamponluKestirici(VoLadderEstimator):
    """Kalibrasyon orneklerini (e0, g, psi, normal) sirasiyla kaydeder."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.ornekler = []                     # (e0, g, psi, n_xy)

    def update(self, meas, gt_xyz, healthy):
        onceki_n = len(self._buf_E)
        psi_once = self.psi
        r = super().update(meas, gt_xyz, healthy)
        # tamponda yeni ornek olustuysa yakala (temizlenme durumunu da kollayarak)
        if healthy and len(self._buf_E) == onceki_n + 1:
            nxy = (np.array([meas.hom_nx, meas.hom_ny])
                   if (meas is not None and meas.hom_ok) else np.array([np.nan, np.nan]))
            self.ornekler.append((np.array(self._buf_E[-1]), np.array(self._buf_G[-1]),
                                  psi_once, nxy))
        return r


def _procrustes(E, G):
    H = G.T @ E
    U, S, Vt = np.linalg.svd(H)
    M = U @ Vt
    c = float(S.sum() / max((E ** 2).sum(), 1e-12))
    return M, c


def _fit_hepsi(E, G):
    """Uc adayi da fit et; tahmin fonksiyonlarini ve (eps, theta) dondur."""
    M, c = _procrustes(E, G)
    P = lambda X: (M @ X.T).T * c                       # noqa: E731

    # S: G ~ c·M·(I + a·D1 + b·D2)·E  -> (a, b) lineer en kucuk kareler
    b1 = (c * (M @ D1) @ E.T).T
    b2 = (c * (M @ D2) @ E.T).T
    art = G - P(E)
    X = np.column_stack([b1.ravel(), b2.ravel()])
    try:
        ab, *_ = np.linalg.lstsq(X, art.ravel(), rcond=None)
        a, b = float(ab[0]), float(ab[1])
    except np.linalg.LinAlgError:
        a = b = 0.0
    Pm = np.eye(2) + a * D1 + b * D2
    det = np.linalg.det(Pm)
    if det > 1e-9:
        Pm = Pm / np.sqrt(det)                          # det-normalize: olcek yutmasin
    S_fn = lambda X: (c * (M @ Pm) @ X.T).T             # noqa: E731
    eps, theta = float(np.hypot(a, b)), float(0.5 * np.arctan2(b, a))

    # A: genel 2x2
    try:
        A = np.linalg.lstsq(E, G, rcond=None)[0].T      # G ~ E @ A.T
    except np.linalg.LinAlgError:
        A = c * M
    A_fn = lambda X: (A @ X.T).T                        # noqa: E731
    return dict(P=P, S=S_fn, A=A_fn), eps, theta


def _ess(x: np.ndarray) -> float:
    """Etkin ornek sayisi (lag-1 otokorelasyondan)."""
    if len(x) < 4:
        return float(len(x))
    v = x - x.mean(0)
    r = float((v[:-1] * v[1:]).sum() / max((v * v).sum(), 1e-12))
    r = float(np.clip(r, -0.99, 0.99))
    return len(x) * (1 - r) / (1 + r)


def olc(yatak: str, cfg: EstimatorConfig, pencere: int = 60, holdout: int = 12) -> dict:
    ib = NO.YATAKLAR[yatak].get("iki_boyut", False)
    gt, cons, lad, prof, n = yatak_yukle(yatak, n_kare=None if ib else 2250)
    if "gt_npz" in NO.YATAKLAR[yatak]:
        h = np.load(NO.YATAKLAR[yatak]["gt_npz"])["saglikli"].copy()
        n = min(n, len(h)); h = h[:n]; h[900:1051] = False
    elif yatak in ("video1", "video2"):
        h = yarisma_takvimi(n)
    else:
        h = oransal_takvim(n)
    gt = gt[:n]

    est = TamponluKestirici(prof, cfg, spacing=8)
    cr = [evaluation._meas_from_row(r) for r in cons.to_dict("records")]
    lr = [evaluation._meas_from_row(r) for r in lad.to_dict("records")]
    est.update_ladder(0, None, None, gt[0], True)
    for k in range(1, n):
        est.update_ladder(k, cr[k-1] if k-1 < len(cr) else None,
                          lr[k-1] if k-1 < len(lr) else None,
                          gt[k] if h[k] else None, bool(h[k]))

    O = est.ornekler
    if len(O) < pencere + holdout:
        return {}
    E_all = np.array([o[0] for o in O])
    G_all = np.array([o[1] for o in O])
    N_all = np.array([o[3] for o in O])

    skor = {"P": [], "S": [], "A": []}
    epsler, tetalar, tilt_tahmin, azimut, ess_ler = [], [], [], [], []
    for i in range(pencere, len(O) - holdout, holdout):
        E, G = E_all[i-pencere:i], G_all[i-pencere:i]
        Et, Gt = E_all[i:i+holdout], G_all[i:i+holdout]
        fns, eps, theta = _fit_hepsi(E, G)
        for ad, fn in fns.items():
            skor[ad].append(float(np.sqrt(((Gt - fn(Et)) ** 2).sum(1).mean())))
        epsler.append(eps); tetalar.append(theta); ess_ler.append(_ess(E))
        nxy = N_all[i-pencere:i]
        ok = np.isfinite(nxy).all(1)
        if ok.sum() > 5:
            nm = np.median(nxy[ok], axis=0)
            snt = float(np.clip(np.hypot(*nm), 0, 1))            # sin(tilt)
            tilt_tahmin.append((1 - np.sqrt(max(1 - snt**2, 0))) / 2)
            azimut.append(float(np.arctan2(nm[1], nm[0])))
        else:
            tilt_tahmin.append(np.nan); azimut.append(np.nan)

    return dict(
        n_pencere=len(skor["P"]), n_ornek=len(O),
        ess=float(np.median(ess_ler)),
        P=float(np.mean(skor["P"])), S=float(np.mean(skor["S"])), A=float(np.mean(skor["A"])),
        eps=np.array(epsler), theta=np.array(tetalar),
        tilt=np.array(tilt_tahmin), azim=np.array(azimut),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yatak", nargs="*", default=TUM)
    ap.add_argument("--pencere", type=int, default=60)
    ap.add_argument("--holdout", type=int, default=12)
    args = ap.parse_args()
    yataklar = [y for y in args.yatak
                if yatak_npz(y, None).exists() and yatak_npz(y, 8).exists()]

    cfg = EstimatorConfig(method="homografi")
    print("=" * 92)
    print("D1 — ORNEKLEM-DISI HIZALAMA TESTI")
    print(f"  pencere={args.pencere} · holdout={args.holdout} · "
          f"olcut = tutulan orneklerdeki RMS konum hatasi [m]")
    print("=" * 92)
    print(f"{'yatak':<15}{'pencere':>8}{'ESS':>7}"
          f"{'P (mevcut)':>12}{'S (+2 SD)':>12}{'A (4 SD)':>12}   kazanan")
    print("-" * 92)

    kazanan = {"P": 0, "S": 0, "A": 0}
    detay = {}
    for y in yataklar:
        r = olc(y, cfg, args.pencere, args.holdout)
        if not r:
            print(f"{y:<15}  yeterli ornek yok, atlandi")
            continue
        detay[y] = r
        en = min(("P", "S", "A"), key=lambda k: r[k])
        kazanan[en] += 1
        print(f"{y:<15}{r['n_pencere']:>8}{r['ess']:>7.1f}"
              f"{r['P']:>12.4f}{r['S']:>12.4f}{r['A']:>12.4f}   {en}", flush=True)

    print("-" * 92)
    print(f"kazanan sayisi: P={kazanan['P']} · S={kazanan['S']} · A={kazanan['A']}")
    print("\nKARAR (onceden yazili: S ya da A, 8 yatagin >=6'sinda P'yi gecmeli)")
    gecen_S = sum(1 for r in detay.values() if r["S"] < r["P"])
    gecen_A = sum(1 for r in detay.values() if r["A"] < r["P"])
    print(f"  S, P'yi gecen yatak: {gecen_S}/{len(detay)}")
    print(f"  A, P'yi gecen yatak: {gecen_A}/{len(detay)}")
    canli = gecen_S >= 6 or gecen_A >= 6
    print(f"  -> {'AILE CANLI: D2 mekanizma kontrolune gec' if canli else 'AILE OLDU: anizotropik hizalama kapanir'}")

    # ---------------- D2 ----------------
    print("\n" + "=" * 92)
    print("D2 — MEKANIZMA: ogrenilen anizotropi gercekten EGIMDEN mi geliyor?")
    print("=" * 92)
    print(f"{'yatak':<15}{'eps medyan':>12}{'egim tahmini':>14}{'kor(eps,egim)':>15}"
          f"{'kor(theta,azimut)':>19}")
    print("-" * 92)
    for y, r in detay.items():
        m = np.isfinite(r["tilt"]) & np.isfinite(r["eps"])
        ke = np.corrcoef(r["eps"][m], r["tilt"][m])[0, 1] if m.sum() > 5 else np.nan
        m2 = np.isfinite(r["azim"]) & np.isfinite(r["theta"])
        # aci karsilastirmasi: 2*theta ile azimut ayni periyotta
        kt = (np.corrcoef(np.cos(2*r["theta"][m2]), np.cos(r["azim"][m2]))[0, 1]
              if m2.sum() > 5 else np.nan)
        print(f"{y:<15}{np.median(r['eps']):>12.4f}"
              f"{np.nanmedian(r['tilt']):>14.4f}{ke:>15.3f}{kt:>19.3f}")
    print("\n  Beklenti: eps ~ (1-cos t)/2 ile POZITIF korele olmali (>0,3).")
    print("  Iliski yoksa S, egimi degil baska bir sistematigi yutuyordur.")


if __name__ == "__main__":
    main()
