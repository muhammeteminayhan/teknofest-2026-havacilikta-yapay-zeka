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
from gorev2_src.calibration import (                             # noqa: E402
    CameraProfile, RGB_1080P, RGB_4K_VIDEO, THERMAL, profile_for_size,
)
from gorev2_src.vo_pipeline import PairMeasurement               # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

VERI = ROOT / "THYZ_2026_Ornek_Veri_Seti"
VERI25 = Path.home() / "Desktop" / "datasets" / "THYZ_2025_Ornek_Veri_Seti"
VERI24 = Path.home() / "Desktop" / "datasets" / "TUYZ_2024_Ornek_Veri"
VIDEO = VERI / "THYZ_2026_Ornek_Veri_1.MP4"
FREEZE_DIFF = 0.35     # PairMeasurement.compute ile ayni donmus-kare esigi


# --------------------------------------------------------------------------- test yataklari
# Uc bagimsiz ucus: iki modalite. Bir degisiklik ancak ucunde birden ayakta
# kalirsa uygulanir (bkz. kabul.py). "yarisma" yatagi puanli oturumun KENDI
# kareleridir — genelleme sinavinin hizli (onbellekli) yolu buna dayanir.
YATAKLAR = {
    "video1": dict(
        ad="video1-RGB (ornek veri 1 · test oturumu ucusu)",
        video=VERI / "THYZ_2026_Ornek_Veri_1.MP4",
        gt_csv=VERI / "THYZ_2026_Ornek_Veri_1_translation.csv",
        profile=RGB_1080P, ek="raft",
    ),
    "video2": dict(
        ad="video2-TERMAL (ornek veri 2 · bagimsiz ucus)",
        video=VERI / "THYZ_2026_Ornek_Veri_2_Termal.MP4",
        gt_csv=VERI / "THYZ_2026_Ornek_Veri_2_Termal_translation.csv",
        profile=THERMAL, ek="raft_v2",
    ),
    "yarisma": dict(
        ad="yarisma-RGB (puanli oturumun kendi kareleri)",
        kareler=ROOT / "sistem" / "_images" / "THYZ_2026_Online_Yarisma",
        gt_npz=GOREV2 / "outputs" / "yarisma_gt.npz",
        profile=RGB_1080P, ek="raft_yarisma",
    ),
    # --- 2025 ornek veri seti: BAGIMSIZ YIL, BAGIMSIZ UCUSLAR (yontem sinavi) ---
    # Kalibrasyon 2026 ile BIREBIR AYNI kamera (RGB 4K ve termal), yani donanim
    # degismemis; yalniz 4K video 3840x2160 geliyor (bkz. calibration.RGB_4K_VIDEO).
    # Dikkat: "Veri-1" RGB ile "Veri-1" Termal AYNI ucus DEGIL —
    #   A ucusu = RGB-1 + Termal-2 (yol 710 m, net 136 m, z araligi ~1 m)
    #   B ucusu = RGB-2 + Termal-1 (yol 1238 m, net 20 m, z araligi 28 m)
    # Her video kendi CSV'siyle eslesir.
    "v25_rgb_a": dict(
        ad="2025 RGB 4K · A ucusu (Ornek-Veri-1)",
        video=VERI25 / "Ornek-Veri-1-RGB.MP4",
        gt_csv=VERI25 / "Ornek-Veri-1-RGB-translation.csv",
        profile=RGB_4K_VIDEO, ek="raft_v25rgba",
    ),
    "v25_rgb_b": dict(
        ad="2025 RGB 4K · B ucusu (Ornek-Veri-2)",
        video=VERI25 / "Ornek-Veri-2-RGB.MP4",
        gt_csv=VERI25 / "Ornek-Veri-2-RGB-translation.csv",
        profile=RGB_4K_VIDEO, ek="raft_v25rgbb",
    ),
    "v25_termal_a": dict(
        ad="2025 TERMAL · A ucusu (Ornek-Veri-2-Termal)",
        video=VERI25 / "Ornek-Veri-2-Termal.MP4",
        gt_csv=VERI25 / "Ornek-Veri-2-Termal-translation.csv",
        profile=THERMAL, ek="raft_v25terma",
    ),
    "v25_termal_b": dict(
        ad="2025 TERMAL · B ucusu (Ornek-Veri-1-Termal)",
        video=VERI25 / "Ornek-Veri-1-Termal.MP4",
        gt_csv=VERI25 / "Ornek-Veri-1-Termal-translation.csv",
        profile=THERMAL, ek="raft_v25termb",
    ),
    # --- 2024 ornek veri: ZAYIF ama bagimsiz bir yatay-kanal sinavi ---
    # Sinirlari (dogrudan olculdu):
    #   * GT'de **z ekseni YOK** (yalniz translation_x, translation_y) -> hata 2B olculur
    #   * kalibrasyon dosyasi YOK -> varsayilan olarak 2026 1080p profili kullanilir;
    #     `yontem_sinavi_2024.py` ayrica distorsiyonsuz profille de kosar ki sonucun
    #     kalibrasyon varsayimina duyarli olup olmadigi gorulsun
    #   * yalniz 3322 kare (7,5 FPS'te 830 ornek) -> takvim oransal olarak kucultulur
    "v24": dict(
        ad="2024 ornek veri (1080p, GT yalniz x-y)",
        video=VERI24 / "TUYZ_2024_Ornek_Video.MP4",
        gt_csv=VERI24 / "GT_Translations.csv",
        profile=RGB_1080P, ek="raft_v24", iki_boyut=True,
    ),
    # --- olcum katmani varyantlari (ayni ucus, farkli on-isleme/ornekleme) ---
    "video2_clahe": dict(
        ad="video2-TERMAL + CLAHE/FPN on-isleme",
        video=VERI / "THYZ_2026_Ornek_Veri_2_Termal.MP4",
        gt_csv=VERI / "THYZ_2026_Ornek_Veri_2_Termal_translation.csv",
        profile=THERMAL, ek="raft_video2", ek_son="_clahe",
    ),
    "video2_yogun": dict(
        ad="video2-TERMAL + uyarlanabilir izgara (hedef 640 nokta)",
        video=VERI / "THYZ_2026_Ornek_Veri_2_Termal.MP4",
        gt_csv=VERI / "THYZ_2026_Ornek_Veri_2_Termal_translation.csv",
        profile=THERMAL, ek="raft_video2", ek_son="_g0",
    ),
}


def yatak_kareler(yatak: dict, progress: bool = True):
    """Yatagin kare akisini uret — video ise altornekli, dizin ise oldugu gibi."""
    if "kareler" in yatak:
        return data_io.iter_image_frames(yatak["kareler"], progress=progress)
    return data_io.iter_competition_frames(yatak["video"], progress=progress)


def yatak_npz(yatak_adi: str, spacing: int | None, tracker: str = "raft") -> Path:
    """Yatagin nokta onbellegi yolu. `tracker != "raft"` ise varyant adlandirmasi
    kullanilir (`main()` bu adi uretir): ``nokta_<tracker>_<yatak>_<cons|spN>.npz``."""
    y = YATAKLAR[yatak_adi]
    orta = "cons" if spacing is None else f"sp{spacing}"
    if tracker != "raft":
        return OUTPUTS / f"nokta_{tracker}_{yatak_adi}_{orta}.npz"
    return OUTPUTS / f"nokta_{y['ek']}_{orta}{y.get('ek_son', '')}.npz"


def _donmus(gray0, gray1, esik=FREEZE_DIFF) -> bool:
    k0 = gray0[::8, ::8].astype(np.int16)
    k1 = gray1[::8, ::8].astype(np.int16)
    return float(np.abs(k0 - k1).mean()) < esik


def onbellek_uret(tracker, cikti_npz, video=VIDEO, spacing: int | None = None,
                  n_hedef: int | None = None, progress: bool = True,
                  kareler=None) -> Path:
    """Kare ciftlerini esleştirip ham noktalari .npz'ye yaz.

    spacing=None -> ARDISIK ciftler (k-1, k)
    spacing=N    -> KEYFRAME MERDIVENI ciftleri (son keyframe, k); keyframe'ler
                    sabit izgarada (0, N, 2N, ...) durur — `build_ladder_measurements`
                    ile ayni kural, boylece olcumler kesinti cizelgesinden bagimsizdir.

    `kareler` verilirse video yerine o jeneratör kullanilir (yarisma oturumunun
    webp kareleri gibi, zaten 7,5 FPS olan kaynaklar icin).
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
    akis = kareler if kareler is not None else data_io.iter_competition_frames(
        video, progress=progress)
    for fidx, gray in akis:
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
               normal_zinciri: bool = True, tohum: int | None = 0,
               karistir: int | None = None, **geo_kwargs) -> pd.DataFrame:
    """Nokta onbelleginden olcum tablosu uret (mevcut CSV ile ayni sutunlar).

    `geo_kwargs` geometri cagrilarina gecer: `robust_method`, `ransac_thresh`, ...
    `normal_zinciri`: homografi normalini onceki karenin normaliyle zincirle
    (build_*_measurements ile ayni davranis).
    `tohum`: cift basina RANSAC tohumu (None = tohumsuz/eski davranis).

    `karistir`: verilirse her ciftin nokta SIRASI bu tohumla karistirilir.
    OpenCV RANSAC girdi verildiginde deterministiktir ama **sıraya duyarlidir**:
    ayni noktalar farkli sirada verilince farkli minimal ornek dizisi cekilir ve
    esit derecede iyi ama farkli bir model secilir (olculdu: inlier |fark| ort
    0,042 / maks 0,23; ortalama inlier degismiyor). Bu, sistemin **gercek gurultu
    tabani**dir — bir A/B farki bu tabandan kucukse anlamsizdir.
    """
    n = len(ob["offsets"]) - 1
    prev_normal = None
    satirlar = []
    rng = np.random.default_rng(karistir) if karistir is not None else None
    for i in range(n):
        if ob["donmus"][i]:
            m = PairMeasurement()
            m.frozen = True
        else:
            p0, p1 = cift_noktalari(ob, i)
            if rng is not None and len(p0) > 1:
                j = rng.permutation(len(p0))
                p0, p1 = p0[j], p1[j]
            if len(p0) < 4:
                m = PairMeasurement()
            else:
                m = PairMeasurement.from_points(
                    p0, p1, profile, prev_normal=prev_normal,
                    rng_tohum=None if tohum is None else tohum * 100003 + i,
                    **geo_kwargs)
                if normal_zinciri and m.hom_ok:
                    prev_normal = m.hom_normal
        row = asdict(m)
        row["idx0"], row["idx1"] = int(ob["idx0"][i]), int(ob["idx1"][i])
        satirlar.append(row)
    return pd.DataFrame(satirlar)


def yatak_yukle(yatak_adi: str, spacing: int = 8, n_kare: int | None = 2250,
                tohum: int | None = 0, karistir: int | None = None,
                geo_merdiven: dict | None = None, tracker: str = "raft",
                **geo_kwargs):
    """Bir test yatagini olcume hazir halde yukle.

    Doner: ``(gt, cons_df, ladder_df, profile, n)`` — `gt` (N,3) metre.
    `geo_kwargs` geometri cagrilarina gecer (ransac_thresh, robust_method, ...),
    boylece ayni nokta onbelleginden farkli geometri ayarlari saniyeler icinde
    olculebilir.
    """
    y = YATAKLAR[yatak_adi]
    prof = y["profile"]

    if "gt_npz" in y:
        d = np.load(y["gt_npz"])
        gt = d["konum"]
    else:
        gt_sub = data_io.subsample_competition(data_io.load_translations(y["gt_csv"]))
        if y.get("iki_boyut"):
            # GT'de z yok: sifir sutunu eklenir, hata 2B olculur (bkz. YATAKLAR["v24"])
            gt = np.column_stack([gt_sub["translation_x"], gt_sub["translation_y"],
                                  np.zeros(len(gt_sub))])
        else:
            gt = gt_sub[["translation_x", "translation_y", "translation_z"]].to_numpy()

    ob_c = onbellek_yukle(yatak_npz(yatak_adi, None, tracker))
    ob_l = onbellek_yukle(yatak_npz(yatak_adi, spacing, tracker))
    # Merdiven ve ardisik olcumler ayri geometri ayari alabilir: merdiven ciftlerinde
    # duzleme uyum cok daha dusuk (inlier medyani 0,54 vs 0,86), yani iki katman
    # farkli miktarda robustluk isteyebilir.
    cons = olcum_uret(ob_c, prof, tohum=tohum, karistir=karistir, **geo_kwargs)
    gl = geo_kwargs if geo_merdiven is None else {**geo_kwargs, **geo_merdiven}
    lad = olcum_uret(ob_l, prof, tohum=tohum, karistir=karistir, **gl)

    n = min(len(gt), len(cons) + 1, len(lad) + 1)
    if n_kare is not None:
        n = min(n, n_kare)
    return gt[:n], cons.iloc[:n - 1], lad.iloc[:n - 1], prof, n


def _tracker_kur(ad: str, profile, downscale=None, on_isle=None, grid_step=28,
                 hedef_nokta=640):
    if ad == "raft":
        from gorev2_src.raft_matching import RaftTracker
        return RaftTracker(profile, model_size="large", downscale=downscale,
                           on_isle=on_isle, grid_step=grid_step,
                           hedef_nokta=hedef_nokta)
    if ad == "lk":
        from gorev2_src.matching import FeatureTracker
        return FeatureTracker(profile)
    if ad in ("sift", "orb", "akaze", "brisk"):
        from gorev2_src.klasik_eslestirici import KlasikTracker
        return KlasikTracker(profile, yontem=ad, downscale=downscale)
    if ad in ("xfeat", "splg", "superpoint"):
        from gorev2_src.derin_eslestirici import tracker_kur
        return tracker_kur(ad, profile)
    raise SystemExit(f"bilinmeyen tracker: {ad}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracker", default="raft", help="raft | lk | xfeat | splg")
    ap.add_argument("--yatak", default="video1", choices=sorted(YATAKLAR),
                    help="video1 (RGB) | video2 (termal) | yarisma (puanli oturum)")
    ap.add_argument("--spacing", type=int, default=None,
                    help="verilirse keyframe merdiveni ciftleri uretilir")
    ap.add_argument("--n", type=int, default=None, help="ilk N kare (hizli deneme)")
    ap.add_argument("--downscale", type=float, default=None,
                    help="RAFT calisma olcegi (None=otomatik; 1.0=tam cozunurluk)")
    ap.add_argument("--on-isle", default=None, choices=[None, "clahe"],
                    help="kare on-islemesi (termal: clahe)")
    ap.add_argument("--grid-step", type=int, default=28,
                    help="ornekleme araligi; 0 -> hedef nokta sayisina gore uyarlanir")
    ap.add_argument("--hedef-nokta", type=int, default=640)
    args = ap.parse_args()

    y = YATAKLAR[args.yatak]
    grid = None if args.grid_step == 0 else args.grid_step
    varyant = (args.tracker == "raft" and args.downscale is None
               and args.on_isle is None and grid == 28)
    if varyant:
        cikti = yatak_npz(args.yatak, args.spacing)
    else:                                    # deneysel varyantlar ayri adlandirmayla
        ek = "cons" if args.spacing is None else f"sp{args.spacing}"
        if args.downscale is not None:
            ek += f"_ds{args.downscale:g}".replace(".", "")
        if args.on_isle:
            ek += f"_{args.on_isle}"
        if grid != 28:
            ek += f"_g{args.grid_step}"
        cikti = OUTPUTS / f"nokta_{args.tracker}_{args.yatak}_{ek}.npz"
    if cikti.exists():
        print(f"zaten var: {cikti}")
        return

    print(f"yatak={y['ad']}\ntracker={args.tracker} kuruluyor...")
    tracker = _tracker_kur(args.tracker, y["profile"], downscale=args.downscale,
                           on_isle=args.on_isle, grid_step=grid,
                           hedef_nokta=args.hedef_nokta)
    onbellek_uret(tracker, cikti, spacing=args.spacing, n_hedef=args.n,
                  kareler=yatak_kareler(y))


if __name__ == "__main__":
    main()
