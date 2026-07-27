# -*- coding: utf-8 -*-
"""GOREV 2 — NOKTA ONBELLEGI: esleştirme bir kez, geometri denemeleri saniyeler icinde.

SORUN: mevcut olcum CSV'leri geometri SONUCLARINI (hom_ux, sim_yaw, ...) sakliyor.
Geometri degisince (orn. RANSAC -> MAGSAC++) tum video yeniden taranmak zorunda (~25 dk).
Bu, onlarca varyanti denemeyi imkansiz kiliyor.

COZUM: tracker'in urettigi HAM esleşmeleri (pts0, pts1) bir kez cikarip .npz'ye yaz.
Sonra istenen geometri ayariyla `PairMeasurement.from_points(...)` uzerinden olcum
tablosu saniyeler icinde uretilir. Ayni altyapi, farkli eslestiriciler (RAFT, XFeat,
SuperPoint+LightGlue) icin de kullanilir — hepsi ayni `MatchResult` arayuzunu doner.

Kullanim:
    # 1) onbellek uret (tracker basina bir kez; ardisik + merdiven)
    .venv\\Scripts\\python gorev2\\nokta_onbellek.py --tracker raft
    .venv\\Scripts\\python gorev2\\nokta_onbellek.py --tracker raft --spacing 8

    # 2) baska modullerden kullan
    from nokta_onbellek import onbellek_yukle, olcum_uret
    ob = onbellek_yukle('gorev2/outputs/nokta_raft_cons.npz')
    df = olcum_uret(ob, RGB_1080P, robust_method=cv2.USAC_MAGSAC)
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GOREV2 = Path(__file__).resolve().parent
OUTPUTS = GOREV2 / "outputs"

_SRC = GOREV2 / "src"
if "gorev2_src" not in sys.modules:
    _spec = importlib.util.spec_from_file_location(
        "gorev2_src", _SRC / "__init__.py", submodule_search_locations=[str(_SRC)])
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules["gorev2_src"] = _mod
    _spec.loader.exec_module(_mod)

from gorev2_src import data_io                                   # noqa: E402
from gorev2_src.calibration import CameraProfile, RGB_1080P      # noqa: E402
from gorev2_src.vo_pipeline import PairMeasurement               # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

VIDEO = ROOT / "THYZ_2026_Ornek_Veri_Seti" / "THYZ_2026_Ornek_Veri_1.MP4"
FREEZE_DIFF = 0.35     # PairMeasurement.compute ile ayni donmus-kare esigi


def _donmus(gray0, gray1, esik=FREEZE_DIFF) -> bool:
    k0 = gray0[::8, ::8].astype(np.int16)
    k1 = gray1[::8, ::8].astype(np.int16)
    return float(np.abs(k0 - k1).mean()) < esik


def onbellek_uret(tracker, cikti_npz, video=VIDEO, spacing: int | None = None,
                  n_hedef: int | None = None, progress: bool = True) -> Path:
    """Kare ciftlerini esleştirip ham noktalari .npz'ye yaz.

    spacing=None -> ARDISIK ciftler (k-1, k)
    spacing=N    -> KEYFRAME MERDIVENI ciftleri (son keyframe, k); keyframe'ler
                    sabit izgarada (0, N, 2N, ...) durur — `build_ladder_measurements`
                    ile ayni kural, boylece olcumler kesinti cizelgesinden bagimsizdir.
    """
    cikti_npz = Path(cikti_npz)
    if cikti_npz.exists():
        print(f"nokta onbellegi zaten var: {cikti_npz.name}")
        return cikti_npz

    t0 = time.perf_counter()
    p0_par, p1_par, uzunluk, donmus, idx0, idx1 = [], [], [], [], [], []
    prev = None          # (ordinal, frame_idx, gray)
    kf = None            # (ordinal, frame_idx, gray) — merdiven icin
    ord_ = 0
    for fidx, gray in data_io.iter_competition_frames(video, progress=progress):
        if n_hedef is not None and ord_ >= n_hedef:
            break
        if spacing is None:
            ref = prev
        else:
            if kf is None:
                kf = (0, fidx, gray)
                prev = (ord_, fidx, gray)
                ord_ += 1
                continue
            ref = kf

        if ref is not None:
            g_ref = ref[2]
            if _donmus(g_ref, gray):
                p0_par.append(np.zeros((0, 2), np.float32))
                p1_par.append(np.zeros((0, 2), np.float32))
                uzunluk.append(0); donmus.append(True)
            else:
                mr = tracker.track_pair(g_ref, gray)
                if mr is None:
                    p0_par.append(np.zeros((0, 2), np.float32))
                    p1_par.append(np.zeros((0, 2), np.float32))
                    uzunluk.append(0); donmus.append(False)
                else:
                    p0_par.append(mr.pts0.astype(np.float32))
                    p1_par.append(mr.pts1.astype(np.float32))
                    uzunluk.append(len(mr.pts0)); donmus.append(False)
            idx0.append(ref[1]); idx1.append(fidx)

        if spacing is not None and ord_ % spacing == 0:
            kf = (ord_, fidx, gray)          # keyframe'i tasi
        prev = (ord_, fidx, gray)
        ord_ += 1

    pts0 = np.concatenate(p0_par) if p0_par else np.zeros((0, 2), np.float32)
    pts1 = np.concatenate(p1_par) if p1_par else np.zeros((0, 2), np.float32)
    offsets = np.concatenate([[0], np.cumsum(uzunluk)]).astype(np.int64)
    cikti_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cikti_npz, pts0=pts0, pts1=pts1, offsets=offsets,
                        donmus=np.array(donmus, bool),
                        idx0=np.array(idx0, np.int64), idx1=np.array(idx1, np.int64))
    mb = cikti_npz.stat().st_size / 1e6
    print(f"{len(uzunluk)} cift, {len(pts0)} nokta, {time.perf_counter()-t0:.0f} s, "
          f"{mb:.1f} MB -> {cikti_npz.name}")
    return cikti_npz


def onbellek_yukle(npz_yolu) -> dict:
    d = np.load(npz_yolu)
    return {k: d[k] for k in ("pts0", "pts1", "offsets", "donmus", "idx0", "idx1")}


def cift_noktalari(ob: dict, i: int):
    a, b = int(ob["offsets"][i]), int(ob["offsets"][i + 1])
    return ob["pts0"][a:b].astype(np.float64), ob["pts1"][a:b].astype(np.float64)


def olcum_uret(ob: dict, profile: CameraProfile = RGB_1080P,
               normal_zinciri: bool = True, **geo_kwargs) -> pd.DataFrame:
    """Nokta onbelleginden olcum tablosu uret (mevcut CSV ile ayni sutunlar).

    `geo_kwargs` geometri cagrilarina gecer: `robust_method`, `ransac_thresh`, ...
    `normal_zinciri`: homografi normalini onceki karenin normaliyle zincirle
    (build_*_measurements ile ayni davranis).
    """
    n = len(ob["offsets"]) - 1
    prev_normal = None
    satirlar = []
    for i in range(n):
        if ob["donmus"][i]:
            m = PairMeasurement()
            m.frozen = True
        else:
            p0, p1 = cift_noktalari(ob, i)
            if len(p0) < 4:
                m = PairMeasurement()
            else:
                m = PairMeasurement.from_points(p0, p1, profile,
                                                prev_normal=prev_normal, **geo_kwargs)
                if normal_zinciri and m.hom_ok:
                    prev_normal = m.hom_normal
        row = asdict(m)
        row["idx0"], row["idx1"] = int(ob["idx0"][i]), int(ob["idx1"][i])
        satirlar.append(row)
    return pd.DataFrame(satirlar)


def _tracker_kur(ad: str, profile, downscale=None):
    if ad == "raft":
        from gorev2_src.raft_matching import RaftTracker
        return RaftTracker(profile, model_size="large", downscale=downscale)
    if ad == "lk":
        from gorev2_src.matching import FeatureTracker
        return FeatureTracker(profile)
    if ad in ("xfeat", "splg", "superpoint"):
        from gorev2_src.derin_eslestirici import tracker_kur
        return tracker_kur(ad, profile)
    raise SystemExit(f"bilinmeyen tracker: {ad}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracker", default="raft", help="raft | lk | xfeat | splg")
    ap.add_argument("--spacing", type=int, default=None,
                    help="verilirse keyframe merdiveni ciftleri uretilir")
    ap.add_argument("--n", type=int, default=None, help="ilk N kare (hizli deneme)")
    ap.add_argument("--downscale", type=float, default=None,
                    help="RAFT calisma olcegi (None=otomatik 0.5; 1.0=tam cozunurluk)")
    args = ap.parse_args()

    ek = "cons" if args.spacing is None else f"sp{args.spacing}"
    if args.downscale is not None:
        ek += f"_ds{args.downscale:g}".replace(".", "")
    cikti = OUTPUTS / f"nokta_{args.tracker}_{ek}.npz"
    if cikti.exists():
        print(f"zaten var: {cikti}")
        return
    print(f"tracker={args.tracker} kuruluyor...")
    tracker = _tracker_kur(args.tracker, RGB_1080P, downscale=args.downscale)
    onbellek_uret(tracker, cikti, spacing=args.spacing, n_hedef=args.n)


if __name__ == "__main__":
    main()
