# -*- coding: utf-8 -*-
"""GOREV 2 — GENELLEME SINAVI: puanli oturumun KENDI kareleri uzerinde puan metrigi.

NEDEN VAR
---------
Tum ayar calismasi `THYZ_2026_Ornek_Veri_1.MP4` uzerinde yapildi. O video, cevrim ici
yarismanin TEST OTURUMUNUN ta kendisidir (GT'lerin ilk satiri birebir ayni). Puanli
oturum ise BASKA bir ucustur. "Rastgele kesinti takvimleri" ile yapilan asiri-uyum
kontrolu bu riski yakalayamaz: takvim cesitleniyor ama UCUS cesitlenmiyor.

Nitekim dusey kanal (`z_sim_agirlik=1.0`) ornek veri setinde −%25,6 kazandiriyordu;
bu sinavda iki pencerede de KAYBETTI ve sistemden kaldirildi (degerlendirme/DUSEY_KANAL.md).

YONTEM
------
Puanli oturumda kesinti karelerinin GT'si sunucu tarafindan hic verilmedi; puan metrigi
orada dogrudan olculemiyor. Cozum: SAGLIKLI bir blogu yapay olarak kesintiye cevirmek.
Sunucu o karelerin GT'sini vermisti (log'daki payload'lardan cikarilir), biz sisteme
vermiyoruz. Boylece gercek yarisma goruntulerinde, gercekci kesinti uzunlugunda, her
kare icin gercek hata olculebiliyor.

  pencere 1 : saglikli 0-450                -> puanlanan 900-1050   (449-599 karelik kesinti)
  pencere 2 : saglikli 0-450 ve 900-1050    -> puanlanan 1550-1650  (~500-600 karelik kesinti)

KABUL OLCUTU
------------
Bir degisiklik ancak HER IKI PENCEREDE de mevcut sistemi gecerse uygulanir.
Tek pencerede kazanip digerinde kaybeden degisiklik REDDEDILIR.

ON KOSUL
--------
`hazirlik_yarisma_gt()` bir kez calisir: puanli oturum logundan kare basina konum ve
saglik durumu cikarip `outputs/yarisma_gt.npz` olarak kaydeder.

CALISTIR
--------
    .venv\\Scripts\\python gorev2\\genelleme_sinavi.py
    .venv\\Scripts\\python gorev2\\genelleme_sinavi.py --pencere 1 --alpha 1.0 0.6
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import re
import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np

KOK = Path(__file__).resolve().parent.parent
KARELER = KOK / "sistem" / "_images" / "THYZ_2026_Online_Yarisma"
LOGLAR = KOK / "sistem" / "_logs"
CIKTI = Path(__file__).resolve().parent / "outputs"
GT_NPZ = CIKTI / "yarisma_gt.npz"

PENCERELER = {
    1: dict(son=1051, puan=(900, 1050), zorla_kesinti=[(900, 1050)]),
    2: dict(son=1651, puan=(1550, 1650), zorla_kesinti=[(1550, 1650)]),
}


# --------------------------------------------------------------------------- veri
def hazirlik_yarisma_gt() -> Path:
    """Puanli oturum logundan kare basina konum + saglik durumunu cikar."""
    if GT_NPZ.exists():
        return GT_NPZ
    adaylar = sorted(LOGLAR.glob("*.log"))
    if not adaylar:
        raise SystemExit(f"Puanli oturum logu bulunamadi: {LOGLAR}")

    for log in reversed(adaylar):          # en yeniden geriye
        satirlar = log.read_text(encoding="utf-8", errors="replace").splitlines()
        durumlar = [m.group(1) for s in satirlar
                    if (m := re.search(r"\[Gorev2 (echo|TAHMIN|yok)\]", s))]
        if len(durumlar) != 2250:
            continue
        konumlar = {}
        for s in satirlar:
            i = s.find('{"frame"')
            if i < 0:
                continue
            try:
                d = json.loads(s[i:])
            except json.JSONDecodeError:
                continue
            m = re.search(r"/frames/(\d+)/", d["frame"])
            if not m or not d["detected_translations"]:
                continue
            t = d["detected_translations"][0]
            konumlar[int(m.group(1))] = (float(t["translation_x"]),
                                         float(t["translation_y"]),
                                         float(t["translation_z"]))
        if len(konumlar) != 2250:
            continue
        konum = np.array([konumlar[f] for f in sorted(konumlar)])
        saglikli = np.array([d == "echo" for d in durumlar])
        CIKTI.mkdir(parents=True, exist_ok=True)
        np.savez(GT_NPZ, konum=konum, saglikli=saglikli)
        print(f"GT cikarildi: {log.name} -> {GT_NPZ} "
              f"({saglikli.sum()} saglikli / {(~saglikli).sum()} kesinti)")
        return GT_NPZ
    raise SystemExit("2250 karelik puanli oturum logu bulunamadi.")


def _gorev2_yukle():
    src = KOK / "gorev2" / "src"
    if "gorev2_src" not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            "gorev2_src", src / "__init__.py", submodule_search_locations=[str(src)])
        mod = importlib.util.module_from_spec(spec)
        sys.modules["gorev2_src"] = mod
        spec.loader.exec_module(mod)
    from gorev2_src.live_estimator import LivePositionEstimator
    from gorev2_src.vo_pipeline import EstimatorConfig
    return LivePositionEstimator, EstimatorConfig


# --------------------------------------------------------------------------- sinav
def kos(pencere: int, etiket: str, cfg, spacing: int = 8) -> dict:
    LivePositionEstimator, _ = _gorev2_yukle()
    p = PENCERELER[pencere]
    d = np.load(GT_NPZ)
    GT, gercek = d["konum"], d["saglikli"]

    beslenen = gercek.copy()
    for a, b in p["zorla_kesinti"]:
        beslenen[a:b + 1] = False          # GT'si bilinen blok yapay kesintiye cevrilir

    yollar = sorted(KARELER.glob("frame_*.webp"))[:p["son"]]
    if len(yollar) < p["son"]:
        raise SystemExit(f"Kare eksik: {len(yollar)} < {p['son']}  ({KARELER})")

    est = LivePositionEstimator(engine="raft", config=cfg, spacing=spacing)
    tahmin = np.zeros((p["son"], 3))
    t0 = time.perf_counter()
    for i, yol in enumerate(yollar):
        img = cv2.imread(str(yol))
        s = bool(beslenen[i])
        tahmin[i] = est.process_frame(img, GT[i] if s else None, s)
        if i and i % 400 == 0:
            print(f"    {i}/{p['son']}  {time.perf_counter() - t0:.0f} s", flush=True)
    if est.engine_used != "raft":
        print(f"    !!! UYARI: motor '{est.engine_used}' — RAFT kurulamadi, "
              f"bu kosu RAFT recetesini DOGRULAMAZ !!!")

    a, b = p["puan"]
    dil = slice(a, b + 1)
    hata = np.linalg.norm(tahmin[dil] - GT[dil], axis=1)
    eksen = np.abs(tahmin[dil] - GT[dil]).mean(axis=0)
    return dict(etiket=etiket, pencere=pencere, ort=float(hata.mean()),
                med=float(np.median(hata)), maks=float(hata.max()),
                eksen=eksen, sure=time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pencere", type=int, nargs="*", default=[1, 2], choices=[1, 2])
    ap.add_argument("--alpha", type=float, nargs="*", default=[1.0],
                    help="denenecek smooth_alpha degerleri")
    ap.add_argument("--spacing", type=int, nargs="*", default=[8],
                    help="denenecek keyframe araliklari")
    args = ap.parse_args()

    logging.disable(logging.CRITICAL)
    warnings.filterwarnings("ignore")
    os.chdir(KOK)
    hazirlik_yarisma_gt()
    _, EstimatorConfig = _gorev2_yukle()

    adaylar = [(f"alpha={a} spacing={sp}",
                EstimatorConfig(method="homografi", smooth_alpha=a), sp)
               for a in args.alpha for sp in args.spacing]

    sonuc: dict[str, dict[int, dict]] = {}
    for etiket, cfg, sp in adaylar:
        sonuc[etiket] = {}
        for w in args.pencere:
            print(f"\n=== pencere {w} · {etiket} ===", flush=True)
            r = kos(w, etiket, cfg, spacing=sp)
            sonuc[etiket][w] = r
            print(f"    ort {r['ort']:.3f} m | medyan {r['med']:.3f} | "
                  f"en kotu {r['maks']:.3f} | eksen x{r['eksen'][0]:.2f} "
                  f"y{r['eksen'][1]:.2f} z{r['eksen'][2]:.2f}  ({r['sure']:.0f}s)")

    print(f"\n{'=' * 72}\nGENELLEME SINAVI — ortalama Oklid hatasi (m)\n{'=' * 72}")
    basliklar = "".join(f"{'pencere ' + str(w):>13}" for w in args.pencere)
    print(f"  {'yapilandirma':<26}{basliklar}{'ortalama':>11}")
    for etiket in sonuc:
        satir = "".join(f"{sonuc[etiket][w]['ort']:13.2f}" for w in args.pencere)
        ort = np.mean([sonuc[etiket][w]["ort"] for w in args.pencere])
        print(f"  {etiket:<26}{satir}{ort:11.2f}")
    print("\n  KABUL: bir aday ancak TUM pencerelerde mevcut sistemi gecerse uygulanir.")


if __name__ == "__main__":
    main()
