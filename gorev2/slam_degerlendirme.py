# -*- coding: utf-8 -*-
"""GOREV 2 — SLAM/VO POZ DIZISI DEGERLENDIRME (ortak zemin).

Derin SLAM/VO sistemleri (VGGT, DPVO, DROID-SLAM, MASt3R-SLAM, ORB-SLAM3 ...) kendi
koordinat cercevesinde ve OLCEKSIZ poz uretir. Bu modul, boyle bir poz dizisini bizim
yarisma mimarimize baglar ve RESMI METRIKLE olcer:

  1. SAGLIKLI karelerde SLAM pozu <-> GT eslestirilir  -> Sim(3) (olcek + donme + oteleme)
  2. KESINTI karelerinde: konum = Sim(3) o SLAM_pozu   -> metrik kestirim
  3. `evaluation.competition_error` (sartname Denklem 2) ile puanlanir

Bu, mevcut `scale_alignment.py` mantiginin 3B genellemesidir.

ROLLING AKIS KISITI: yarismada gonderilen tahmin geri alinamaz; bu yuzden Sim(3) yalnizca
O ANA KADAR gorulen saglikli karelerden kurulur (nedensel/causal). Gelecege bakan bir
hizalama puanimizi yansitmaz — `--kahin` bayragi yalnizca ust sinir (oracle) merakı icindir.

Kullanim:
    # bir poz dizisini degerlendir (npz: 'pozlar' (N,4,4) veya 'konumlar' (N,3))
    .venv\\Scripts\\python gorev2\\slam_degerlendirme.py --poz gorev2/outputs/poz_vggt.npz --ad vggt

    # altyapi dogrulamasi: mevcut RAFT boru hattinin pozlari -> 3,693 m cikmali
    .venv\\Scripts\\python gorev2\\slam_degerlendirme.py --raft-dogrulama
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

GOREV2 = Path(__file__).resolve().parent
ROOT = GOREV2.parent
OUT = GOREV2 / "degerlendirme"
sys.path.insert(0, str(GOREV2))

from nokta_onbellek import onbellek_yukle, olcum_uret            # noqa: E402
from gorev2_src import data_io, evaluation                       # noqa: E402
from gorev2_src.calibration import RGB_1080P                     # noqa: E402
from gorev2_src.vo_pipeline import EstimatorConfig, VoLadderEstimator  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

GT_CSV = ROOT / "THYZ_2026_Ornek_Veri_Seti" / "THYZ_2026_Ornek_Veri_1_translation.csv"
KESINTI = [(451, 900), (1051, 1550), (1651, 2250)]
N_KARE = 2250
KARE_BUTCE_S = 2.4        # 90 dk / 2250 kare
# RAFT + homografi + smooth_alpha=1.0 + esik 1.0 (dusey kanal GERI ALINDI, bkz.
# degerlendirme/DUSEY_KANAL.md). OLCUM YOLU: canli sistem (`canli_dogrulama.py`,
# gercek video, onbelleksiz), ORNEK VERI SETI uzerinde.
# UYARI: bu esik yalniz ornek veri setinde gecerlidir. Bir SLAM adayi bu esigi gecse
# bile, uygulanmadan once `genelleme_sinavi.py` ile puanli oturumun kendi karelerinde
# de kazanmalidir — tek ucusta olculen kazanc genellemiyor (dusey kanal ornegi).
MEVCUT_EN_IYI = 3.757


def takvim(n=N_KARE):
    h = np.ones(n, dtype=bool)
    for b, s in KESINTI:
        h[b:s] = False
    return h


def uzun_takvim(n, rng):
    """Yarismadaki gibi UZUN kesintiler (450-600 kare)."""
    r = np.random.default_rng(rng)
    h = np.ones(n, dtype=bool)
    for _ in range(3):
        d = int(r.uniform(450, 600))
        b = int(r.integers(300, max(301, n - d - 10)))
        h[b:b + d] = False
    return h


# --------------------------------------------------------------------------
def sim3_fit(A: np.ndarray, B: np.ndarray):
    """Umeyama: B ~= s*R*A + t  (A: kaynak Nx3, B: hedef Nx3)."""
    if len(A) < 3:
        return None
    mu_a, mu_b = A.mean(0), B.mean(0)
    A0, B0 = A - mu_a, B - mu_b
    H = A0.T @ B0 / len(A)
    U, D, Vt = np.linalg.svd(H)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = Vt.T @ S @ U.T
    var_a = (A0 ** 2).sum() / len(A)
    s = float((D * np.diag(S)).sum() / var_a) if var_a > 1e-12 else 1.0
    t = mu_b - s * (R @ mu_a)
    return s, R, t


def sim3_uygula(P: np.ndarray, fit) -> np.ndarray:
    s, R, t = fit
    return (s * (R @ P.T).T) + t


def nedensel_hizala(slam_xyz: np.ndarray, gt: np.ndarray, healthy: np.ndarray,
                    min_ornek: int = 30, pencere: int | None = None) -> np.ndarray:
    """Her karede, YALNIZCA o ana kadarki saglikli karelerden Sim(3) kurup uygula.

    Saglikli karede GT aynalanir (yarismada da oyle yapiyoruz: hata 0).
    Kesintide en son gecerli hizalama kullanilir.
    """
    n = len(gt)
    preds = np.zeros((n, 3))
    fit = None
    idx_saglikli: list[int] = []
    for k in range(n):
        if healthy[k]:
            preds[k] = gt[k]
            idx_saglikli.append(k)
            if len(idx_saglikli) >= min_ornek:
                sec = idx_saglikli if pencere is None else idx_saglikli[-pencere:]
                f = sim3_fit(slam_xyz[sec], gt[sec])
                if f is not None:
                    fit = f
        else:
            preds[k] = sim3_uygula(slam_xyz[k][None], fit)[0] if fit is not None \
                else (preds[k - 1] if k else gt[0])
    return preds


def poz_yukle(npz_yolu) -> tuple[np.ndarray, np.ndarray | None]:
    """npz'den (N,3) konum dizisi + (varsa) kare sureleri dondur."""
    d = np.load(npz_yolu)
    if "konumlar" in d:
        xyz = d["konumlar"]
    elif "pozlar" in d:
        P = d["pozlar"]                      # (N,4,4) kamera->dunya
        xyz = P[:, :3, 3]
    else:
        raise SystemExit(f"npz icinde 'konumlar' veya 'pozlar' yok: {list(d.keys())}")
    sureler = d["sureler"] if "sureler" in d else None
    return np.asarray(xyz, float), sureler


def raft_pozlari(n: int) -> np.ndarray:
    """Altyapi dogrulamasi: mevcut RAFT boru hattini KESINTISIZ kosturup konum dizisi al.

    Tum kareler 'kesinti' gibi islenirse (GT capasi yok) ortaya cikan yorunge, SLAM'lerin
    urettigine benzer bicimde olceklidir; bunu Sim(3) hattindan gecirince mevcut sonucu
    (3,693 m) yeniden uretmeliyiz.
    """
    gt_sub = data_io.subsample_competition(data_io.load_translations(GT_CSV))
    gt = gt_sub.iloc[:n][evaluation.GT_COLS].to_numpy()
    obc = onbellek_yukle(GOREV2 / "outputs" / "nokta_raft_cons.npz")
    obl = onbellek_yukle(GOREV2 / "outputs" / "nokta_raft_sp8.npz")
    dc = olcum_uret(obc, RGB_1080P).iloc[:n - 1]
    dl = olcum_uret(obl, RGB_1080P).iloc[:n - 1]
    cr = [evaluation._meas_from_row(r) for r in dc.to_dict("records")]
    lr = [evaluation._meas_from_row(r) for r in dl.to_dict("records")]
    cfg = replace(EstimatorConfig(method="homografi"), smooth_alpha=1.0)
    est = VoLadderEstimator(RGB_1080P, cfg, spacing=8)
    preds = np.zeros((n, 3))
    preds[0] = gt[0]
    est.update_ladder(0, None, None, gt[0], True)
    healthy = takvim(n)
    for k in range(1, n):
        preds[k] = est.update_ladder(k, cr[k - 1] if k - 1 < len(cr) else None,
                                     lr[k - 1] if k - 1 < len(lr) else None,
                                     gt[k] if healthy[k] else None, bool(healthy[k]))
    return preds


def degerlendir(slam_xyz: np.ndarray, ad: str, sureler=None, senaryo: int = 15,
                pencere: int | None = None) -> dict:
    gt_sub = data_io.subsample_competition(data_io.load_translations(GT_CSV))
    n = min(N_KARE, len(gt_sub), len(slam_xyz))
    gt = gt_sub.iloc[:n][evaluation.GT_COLS].to_numpy()
    slam_xyz = slam_xyz[:n]

    yar = takvim(n)
    p_yar = nedensel_hizala(slam_xyz, gt, yar, pencere=pencere)
    y = evaluation.competition_error(p_yar, gt)

    uz = [uzun_takvim(n, s) for s in range(senaryo)]
    hh = np.array([evaluation.competition_error(
        nedensel_hizala(slam_xyz, gt, h, pencere=pencere), gt) for h in uz])

    from geometri_tarama import procrustes_tani
    aci, olcek = procrustes_tani(p_yar, gt, yar)

    L = [f"=== SLAM/VO DEGERLENDIRME — {ad} ===",
         f"kare: {n} | hizalama: NEDENSEL Sim(3) (yalniz gecmis saglikli kareler)"
         + (f", pencere={pencere}" if pencere else ""),
         "",
         f"  yarisma takvimi           : {y:.3f} m",
         f"  {senaryo} uzun-kesinti ort.: {hh.mean():.3f} m  (medyan {np.median(hh):.3f})",
         f"  yon hatasi                : {aci:.2f} derece",
         f"  olcek hatasi              : %{100*olcek:.2f}",
         "",
         f"  MEVCUT EN IYI (RAFT+homografi): {MEVCUT_EN_IYI:.3f} m"]
    if sureler is not None and len(sureler):
        ort = float(np.mean(sureler))
        L += [f"  kare basi sure            : {ort:.2f} s "
              f"({'BUTCE ICINDE' if ort <= KARE_BUTCE_S else 'BUTCE ASILDI'}; sinir {KARE_BUTCE_S} s)"]
    kazandi = y < MEVCUT_EN_IYI * 0.99
    L += ["", (">>> KAZANDI: mevcut sistemden iyi" if kazandi
               else ">>> KAZANAMADI: mevcut sistem (RAFT+homografi) daha iyi")]
    metin = "\n".join(L)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"slam_{ad}.txt").write_text(metin, encoding="utf-8")
    print(metin)
    return dict(ad=ad, yarisma=y, uzun=float(hh.mean()), aci=aci, olcek=olcek, kazandi=kazandi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poz", help="poz npz dosyasi")
    ap.add_argument("--ad", default="sistem")
    ap.add_argument("--senaryo", type=int, default=15)
    ap.add_argument("--pencere", type=int, default=None,
                    help="Sim(3) icin son N saglikli kare (None = tumu)")
    ap.add_argument("--raft-dogrulama", action="store_true",
                    help="altyapi dogrulamasi: RAFT boru hatti -> ~3,693 m bekleniyor")
    args = ap.parse_args()

    if args.raft_dogrulama:
        print("Altyapi dogrulamasi: mevcut RAFT boru hatti pozlari uretiliyor...")
        xyz = raft_pozlari(N_KARE)
        gt_sub = data_io.subsample_competition(data_io.load_translations(GT_CSV))
        gt = gt_sub.iloc[:len(xyz)][evaluation.GT_COLS].to_numpy()
        dogrudan = evaluation.competition_error(xyz, gt)
        print(f"  RAFT boru hatti DOGRUDAN (Sim(3)'suz): {dogrudan:.3f} m "
              f"(beklenen ~{MEVCUT_EN_IYI:.3f})")
        print("  -> altyapinin Sim(3) hatti bu diziyi bozmadan tasiyabiliyor mu:")
        degerlendir(xyz, "raft_dogrulama", senaryo=args.senaryo, pencere=args.pencere)
        return

    if not args.poz:
        sys.exit("--poz veya --raft-dogrulama verin")
    xyz, sureler = poz_yukle(args.poz)
    degerlendir(xyz, args.ad, sureler=sureler, senaryo=args.senaryo, pencere=args.pencere)


if __name__ == "__main__":
    main()
