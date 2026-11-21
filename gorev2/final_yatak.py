# -*- coding: utf-8 -*-
"""GOREV 2 — FINAL OTURUMLARI TEST YATAGI (Diyarbakir finali, 18-20 Eylul 2026).

Finalin bes oturumunu (Test, O1-O4) bu deponun olcum altyapisina baglar:

  1. LOG -> VERI. Her oturumun logundan kare basina saglik durumu, sunucunun verdigi GT
     (saglikli karelerde), BIZIM GONDERDIGIMIZ konum ve Gorev 1 kutulari cikarilir:
     ``outputs/final/<ad>.npz``.
  2. KARELER. ``final-diyarbakir/sistem/_images/<oturum>/frame_XXXXXX.webp``. Diskte
     eksik kareler var (O1: 8, O2: 6, O4: 1); canli sistem onlari gormustu. Eksik kare
     yerine bir onceki kare verilir (donmus-kare kapisi o adimi hareketsiz sayar), boylece
     kare sirasi kaymaz. Bu yuzden O1/O2/O4'te gonderilen yorunge ilk eksik kareden sonra
     birebir uretilemez; yontem kiyaslarinda temel cizgi YENIDEN KOSULAN uretimdir.
  3. ONBELLEK. RAFT ardisik + merdiven (sp8) nokta onbellekleri (``nokta_onbellek``).
  4. KONTROL NOKTALARI. Kesintiden sonra saglikli blok gelen oturumlarda (O1, O3) o blok
     gizlenip sistem kor kosturulur ve gizli blokta GERCEK GT ile hata olculur
     (``genelleme_sinavi`` yontemi).

Kullanim:
    .venv\\Scripts\\python gorev2\\final_yatak.py --hazirla
    .venv\\Scripts\\python gorev2\\final_yatak.py --onbellek final_o4 final_o1 ...
    .venv\\Scripts\\python gorev2\\final_yatak.py --temel            (onbellek yolu vs gonderilen)
    .venv\\Scripts\\python gorev2\\final_yatak.py --kontrol          (gercek-GT kontrol noktalari)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

GOREV2 = Path(__file__).resolve().parent
sys.path.insert(0, str(GOREV2))

import nokta_onbellek as no                                       # noqa: E402
from gorev2_src import evaluation                                 # noqa: E402
from gorev2_src.vo_pipeline import EstimatorConfig, VoLadderEstimator  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

FINAL = Path.home() / "Desktop" / "final-diyarbakir"
LOGLAR = FINAL / "sistem" / "_logs"
CIKTI = GOREV2 / "outputs" / "final"
N_KARE = 2250

# kisa ad -> (sunucudaki oturum klasoru, modalite)
OTURUMLAR = {
    "final_test": ("THYZ_2026_Test_Oturumu", "rgb"),
    "final_o1": ("THYZ_2026_Oturum1_alvsb", "rgb"),
    "final_o2": ("THYZ_2026_Oturum2_azeld", "rgb"),
    "final_o3": ("THYZ_2026_Oturum3_akrid", "termal"),
    "final_o4": ("THYZ_2026_Oturum4_aavwo", "rgb"),
}

# Gercek-GT kontrol noktalari: (ad, gizlenecek saglikli bloklar, puanlanan blok).
# Puanlanan blok da gizlenir; sistem onu kesinti sanar, biz sunucunun GT'siyle olceriz.
KONTROLLER = {
    "final_o1": [("O1 800-825 (349 kare kor)", [], (800, 825))],
    "final_o3": [("O3 1050-1070 (599 kare kor)", [], (1050, 1070)),
                 ("O3 1780-1800 (709 kare kor)", [], (1780, 1800)),
                 ("O3 1780-1800 (1329 kare kor)", [(1050, 1070)], (1780, 1800))],
}


# --------------------------------------------------------------------------- log -> veri
def log_bul(klasor: str) -> Path:
    """Oturumun 2250 karesini eksiksiz gonderen log (birden fazlaysa en yenisi)."""
    adaylar = []
    for f in sorted(LOGLAR.glob("*.log")):
        bas = f.read_text(encoding="utf-8", errors="ignore")
        if f"/{klasor}/frame_" not in bas:
            continue
        if bas.count("Prediction sent successfully") >= N_KARE:
            adaylar.append(f)
    if not adaylar:
        raise SystemExit(f"{klasor}: 2250 gonderimli log bulunamadi ({LOGLAR})")
    return adaylar[-1]


def _sinif(cls_url) -> int:
    return int(str(cls_url).rstrip("/").split("/")[-1])


def hazirla(ad: str) -> Path:
    """Logdan kare basina saglik, konum ve Gorev 1 kutularini cikar."""
    klasor, _ = OTURUMLAR[ad]
    log = log_bul(klasor)
    satirlar = log.read_text(encoding="utf-8", errors="ignore").splitlines()
    tur = np.full(N_KARE, -1, np.int8)            # 1 echo, 0 TAHMIN
    konum = np.full((N_KARE, 3), np.nan)
    kutu_kare, kutu_sinif, kutu_xyxy = [], [], []
    k = None
    for i, s in enumerate(satirlar):
        m = re.search(rf"get_current_frame: /{re.escape(klasor)}/frame_(\d+)\.webp", s)
        if m:
            k = int(m.group(1))
            continue
        m = re.search(r"\[Gorev2 (echo|TAHMIN)\]", s)
        if m and k is not None:
            tur[k] = 1 if m.group(1) == "echo" else 0
            continue
        if "Prediction sent successfully" in s and k is not None and i + 1 < len(satirlar):
            try:
                p = json.loads(satirlar[i + 1].strip())
            except json.JSONDecodeError:
                continue
            t = (p.get("detected_translations") or [None])[0]
            if t:
                konum[k] = [float(t["translation_x"]), float(t["translation_y"]),
                            float(t["translation_z"])]
            for o in p.get("detected_objects") or []:
                kutu_kare.append(k)
                kutu_sinif.append(_sinif(o["cls"]))
                kutu_xyxy.append([float(o["top_left_x"]), float(o["top_left_y"]),
                                  float(o["bottom_right_x"]), float(o["bottom_right_y"])])
    if (tur < 0).any() or np.isnan(konum).any():
        raise SystemExit(f"{ad}: eksik kare kaydi (tur {int((tur < 0).sum())}, "
                         f"konum {int(np.isnan(konum).any(1).sum())})")
    var = {int(p.name[6:12]) for p in (FINAL / "sistem" / "_images" / klasor).glob("frame_*.webp")}
    eksik = np.array([j for j in range(N_KARE) if j not in var], np.int64)
    CIKTI.mkdir(parents=True, exist_ok=True)
    yol = CIKTI / f"{ad}.npz"
    np.savez_compressed(
        yol, konum=konum, saglikli=tur == 1, eksik=eksik,
        kutu_kare=np.array(kutu_kare, np.int32), kutu_sinif=np.array(kutu_sinif, np.int8),
        kutu_xyxy=np.array(kutu_xyxy, np.float32).reshape(-1, 4), log=str(log.name))
    return yol


def veri(ad: str) -> dict:
    d = np.load(CIKTI / f"{ad}.npz")
    return {k: d[k] for k in d.files}


def bloklar(saglikli: np.ndarray) -> list[tuple[bool, int, int]]:
    out, bas = [], 0
    for i in range(1, len(saglikli) + 1):
        if i == len(saglikli) or saglikli[i] != saglikli[bas]:
            out.append((bool(saglikli[bas]), bas, i - 1))
            bas = i
    return out


# --------------------------------------------------------------------------- kosucu
def onbellek_olcumleri(ad: str, tohum: int | None = 0, karistir: int | None = None):
    """Oturumun ardisik + merdiven olcum satirlari (PairMeasurement listeleri)."""
    gt, cons, lad, prof, n = no.yatak_yukle(ad, spacing=8, n_kare=N_KARE,
                                            tohum=tohum, karistir=karistir)
    cr = [evaluation._meas_from_row(r) for r in cons.to_dict("records")]
    lr = [evaluation._meas_from_row(r) for r in lad.to_dict("records")]
    return gt, cr, lr, prof, n


def kos(cr, lr, gt, saglikli, prof, cfg: EstimatorConfig | None = None,
        spacing: int = 8, kanca=None) -> np.ndarray:
    """Merdivenli durum makinesini bir takvimde kostur (`run_estimator_ladder` esdegeri).

    `kanca(k, est, preds)` verilirse her kare `update_ladder`'dan ONCE cagrilir — dis
    gozlem (olcek capasi gibi) enjeksiyonu icin; `preds[:k]` o ana kadarki tahminlerdir.
    """
    cfg = cfg or EstimatorConfig()
    n = len(gt)
    est = VoLadderEstimator(prof, cfg, spacing=spacing)
    preds = np.zeros((n, 3))
    preds[0] = gt[0]
    est.update_ladder(0, None, None, gt[0], True)
    for k in range(1, n):
        if kanca is not None:
            kanca(k, est, preds)
        mc = cr[k - 1] if k - 1 < len(cr) else None
        ml = lr[k - 1] if k - 1 < len(lr) else None
        preds[k] = est.update_ladder(k, mc, ml, gt[k] if saglikli[k] else None,
                                     bool(saglikli[k]))
    return preds


def kontrol_hatalari(ad: str, kosucu) -> list[tuple[str, float, float]]:
    """Gercek-GT kontrol noktalari. `kosucu(saglikli_maskesi) -> preds`.

    Doner: (etiket, ortalama 3B hata, ortalama yatay hata) — puanlanan blok uzerinde.
    """
    d = veri(ad)
    out = []
    for etiket, gizli, (a, b) in KONTROLLER.get(ad, []):
        s = d["saglikli"].copy()
        for g0, g1 in gizli:
            s[g0:g1 + 1] = False
        s[a:b + 1] = False
        p = kosucu(s)
        e = p[a:b + 1] - d["konum"][a:b + 1]
        out.append((etiket, float(np.linalg.norm(e, axis=1).mean()),
                    float(np.linalg.norm(e[:, :2], axis=1).mean())))
    return out


# --------------------------------------------------------------------------- COLMAP girdisi
COLMAP_KOK = CIKTI / "colmap"


def colmap_hazirla(ad: str, genislik: int = 1280) -> Path:
    """Sozde-GT (sozde_gt.py, ayri venv) icin kareleri JPEG + kamera + GT olarak disa aktar.

    Final oturumlari: `saglikli` = sunucunun gercek takvimi, `konum` = saglikli karede GT.
    Dogrulama yataklari (video1, video2): tam GT var; `saglikli` = yalniz ilk 450 kare
    (O2/O4/Test'in birebir benzeri) ve `konum` = TUM karelerin GT'si (puanlama icin).
    Eksik final kareleri disa aktarilmaz (COLMAP'te yoklar; sonra aradegerlenir).
    """
    import cv2
    from gorev2_src import data_io

    y = no.YATAKLAR[ad]
    prof = y["profile"]
    hedef = COLMAP_KOK / ad
    (hedef / "images").mkdir(parents=True, exist_ok=True)
    if ad in OTURUMLAR:
        d = veri(ad)
        konum, saglikli, eksik = d["konum"], d["saglikli"], set(d["eksik"].tolist())
        kaynak = ((k, cv2.imread(str(y["kareler"] / f"frame_{k:06d}.webp"), cv2.IMREAD_GRAYSCALE))
                  for k in range(N_KARE) if k not in eksik)
        n = N_KARE
    else:
        gt_sub = data_io.subsample_competition(data_io.load_translations(y["gt_csv"]))
        konum = gt_sub[["translation_x", "translation_y", "translation_z"]].to_numpy()[:N_KARE]
        n = len(konum)
        saglikli = np.zeros(n, bool)
        saglikli[:450] = True
        # video akisi orijinal kare indeksini (0, 4, 8, ...) verir -> ordinal'e cevir
        kaynak = ((i, img) for i, (_, img) in enumerate(no.yatak_kareler(y, progress=False)))
    s = 1.0
    yazilan = []
    w = h = 0
    for k, img in kaynak:
        if k >= n:
            break
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img.shape[1] > genislik:
            s = genislik / img.shape[1]
            img = cv2.resize(img, (genislik, round(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(hedef / "images" / f"{k:06d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        yazilan.append(k)
        h, w = img.shape[:2]
    # OPENCV modeli: fx, fy, cx, cy, k1, k2, p1, p2 (distorsiyon normalize koordinatta -> olcekten bagimsiz)
    kamera = [prof.fx * s, prof.fy * s, (prof.cx + 0.5) * s - 0.5, (prof.cy + 0.5) * s - 0.5,
              prof.k1, prof.k2, 0.0, 0.0]
    np.savez(hedef / "girdi.npz", konum=konum[:n], saglikli=saglikli[:n],
             kareler=np.array(yazilan), kamera=np.array(kamera), boyut=np.array([w, h]),
             tam_gt=np.array(ad not in OTURUMLAR))
    print(f"{ad}: {len(yazilan)} kare -> {hedef} ({w}x{h}, olcek {s:.3f})")
    return hedef


# --------------------------------------------------------------------------- komutlar
def komut_hazirla():
    for ad in OTURUMLAR:
        yol = hazirla(ad)
        d = veri(ad)
        sinif, kare = d["kutu_sinif"], d["kutu_kare"]
        ped = np.isin(sinif, (3, 4))
        kes = ~d["saglikli"]
        print(f"{ad:11s} {str(d['log']):48s} saglikli {int(d['saglikli'].sum()):4d} | "
              f"kesinti {int(kes.sum()):4d} | eksik kare {len(d['eksik'])} | "
              f"tasit {int((sinif == 1).sum())} | ped {int(ped.sum())} "
              f"(kesintide {len(set(kare[ped & kes[kare]].tolist()))} kare)")
        print("            takvim: " + " | ".join(
            f"{'S' if s else 'K'} {a}-{b}" for s, a, b in bloklar(d["saglikli"])))


def komut_onbellek(adlar: list[str]):
    for ad in adlar:
        y = no.YATAKLAR[ad]
        tracker = None
        for sp in (None, 8):
            cikti = no.yatak_npz(ad, sp)
            if cikti.exists():
                print(f"{ad} {'cons' if sp is None else 'sp8'}: zaten var")
                continue
            if tracker is None:
                tracker = no._tracker_kur("raft", y["profile"])
            print(f"{ad} {'cons' if sp is None else 'sp8'} uretiliyor...", flush=True)
            no.onbellek_uret(tracker, cikti, spacing=sp, kareler=no.yatak_kareler(y))


def komut_temel(adlar: list[str]):
    """Onbellek yolundaki uretim kestiricisi, gonderdigimiz yorungeye ne kadar yakin?"""
    for ad in adlar:
        if not no.yatak_npz(ad, 8).exists():
            print(f"{ad}: onbellek yok, atlandi")
            continue
        d = veri(ad)
        gt, cr, lr, prof, n = onbellek_olcumleri(ad)
        p = kos(cr, lr, gt, d["saglikli"], prof)
        np.save(CIKTI / f"{ad}_temel.npy", p)
        kes = ~d["saglikli"]
        fark = np.linalg.norm(p - d["konum"], axis=1)
        ilk_eksik = int(d["eksik"].min()) if len(d["eksik"]) else N_KARE
        on = kes.copy()
        on[ilk_eksik:] = False
        print(f"{ad:11s} kesintide |yeniden - gonderilen|: ort {fark[kes].mean():8.2f} m, "
              f"maks {fark[kes].max():8.2f} m | ilk eksik kareye ({ilk_eksik}) kadar maks "
              f"{(fark[on].max() if on.any() else 0.0):.3f} m")


def komut_kontrol(adlar: list[str]):
    for ad in adlar:
        if ad not in KONTROLLER:
            continue
        gt, cr, lr, prof, n = onbellek_olcumleri(ad)
        for etiket, e3, e2 in kontrol_hatalari(ad, lambda s: kos(cr, lr, gt, s, prof)):
            print(f"{etiket:32s} 3B {e3:8.2f} m | yatay {e2:8.2f} m")


def main():
    ap = argparse.ArgumentParser(description="Final oturumlari test yatagi")
    ap.add_argument("--hazirla", action="store_true")
    ap.add_argument("--onbellek", nargs="*")
    ap.add_argument("--temel", nargs="*")
    ap.add_argument("--kontrol", nargs="*")
    ap.add_argument("--colmap-hazirla", nargs="+", metavar="AD",
                    help="sozde-GT girdisi: final_* oturumlari ve/veya video1 video2")
    a = ap.parse_args()
    if a.hazirla:
        komut_hazirla()
    for ad in a.colmap_hazirla or []:
        colmap_hazirla(ad)
    if a.onbellek is not None:
        komut_onbellek(a.onbellek or list(OTURUMLAR))
    if a.temel is not None:
        komut_temel(a.temel or list(OTURUMLAR))
    if a.kontrol is not None:
        komut_kontrol(a.kontrol or list(KONTROLLER))


if __name__ == "__main__":
    main()
