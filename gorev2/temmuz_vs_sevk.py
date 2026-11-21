# -*- coding: utf-8 -*-
"""GOREV 2 — TEMMUZ (puanli oturum) kodu vs SEVK EDILEN kod, ayni karelerde.

NE OLCULUYOR
------------
Puanli oturumda kosan Gorev 2 uygulamasi hâlâ duruyor (`bugbuster_5202745_teslim`).
Bu betik onu ve sevk edilecek surumu 2026 ORNEK VERI SETI uzerinde, ayni 2250 karede,
ayni yarisma kesinti takviminde kosturur ve sartname Denklem 2 metrigini karsilastirir.

NEDEN LOCKSTEP (kare kare, iki kestirici birlikte)
--------------------------------------------------
Video bir kez cozulur ve HER kare iki kestiriciye de verilir. Boylece iki taraf
bit-birebir ayni girdiyi gorur; video kod cozucusunden ya da kare atlamadan gelen
hicbir fark sonuca karisamaz. Maliyeti RAFT'in kare basina iki kez kosmasidir.

KESINTI TAKVIMI  (puanli oturumun gercek takvimi)
    saglikli 0-450 | kesinti 451-899 | saglikli 900-1050
    kesinti 1051-1549 | saglikli 1550-1650 | kesinti 1651-2249

CALISTIR
    .venv\\Scripts\\python gorev2\\temmuz_vs_sevk.py                 # video1 (RGB)
    .venv\\Scripts\\python gorev2\\temmuz_vs_sevk.py --video 2       # video2 (termal)
    .venv\\Scripts\\python gorev2\\temmuz_vs_sevk.py --n 400         # hizli deneme
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np

GOREV2 = Path(__file__).resolve().parent
ROOT = GOREV2.parent
MASAUSTU = ROOT.parent
OUT = GOREV2 / "degerlendirme"
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

KESINTI = [(451, 900), (1051, 1550), (1651, 2250)]
ADIM = 4                       # 30 FPS -> yarisma temposu 7,5 FPS

SURUMLER = {
    "TEMMUZ": MASAUSTU / "bugbuster_5202745_teslim" / "gorev2" / "src",
    "SEVK":   MASAUSTU / "final-diyarbakir" / "gorev2" / "src",
}

VIDEOLAR = {
    1: ("THYZ_2026_Ornek_Veri_1.MP4", "THYZ_2026_Ornek_Veri_1_translation.csv", "RGB 1080p"),
    2: ("THYZ_2026_Ornek_Veri_2_Termal.MP4",
        "THYZ_2026_Ornek_Veri_2_Termal_translation.csv", "termal 640x512"),
}


def surum_yukle(ad: str, src: Path):
    """Bir uygulamayi kendi ust duzey paket adiyla yukle (goreli importlar calissin)."""
    paket = f"g2_{ad.lower()}"
    spec = importlib.util.spec_from_file_location(
        paket, src / "__init__.py", submodule_search_locations=[str(src)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[paket] = mod
    spec.loader.exec_module(mod)
    le = importlib.import_module(f"{paket}.live_estimator")
    return le.LivePositionEstimator


def gt_oku(csv: Path) -> np.ndarray:
    """30 FPS GT tablosundan yarisma temposunu (her 4. kare) al."""
    import csv as _csv
    with open(csv, newline="", encoding="utf-8") as f:
        satir = list(_csv.DictReader(f))
    # Sutun adi: 'frame_numbers' ve degeri 'frame_000123' bicimindedir
    # (bkz. gorev2/src/data_io.load_translations).
    out = []
    for r in satir:
        no = int(str(r["frame_numbers"]).replace("frame_", ""))
        if no % ADIM:
            continue
        out.append([float(r["translation_x"]), float(r["translation_y"]),
                    float(r.get("translation_z", 0.0) or 0.0)])
    return np.asarray(out, dtype=float)


def saglik_takvimi(n: int) -> np.ndarray:
    h = np.ones(n, bool)
    for a, b in KESINTI:
        h[a:b] = False
    return h


def kareler(video: Path, n_hedef: int):
    """Videoyu sirali oku, her ADIM'inci kareyi gri olarak uret."""
    cap = cv2.VideoCapture(str(video))
    i = k = 0
    try:
        while k < n_hedef:
            ok = cap.grab()
            if not ok:
                break
            if i % ADIM == 0:
                ok, kare = cap.retrieve()
                if not ok:
                    break
                yield cv2.cvtColor(kare, cv2.COLOR_BGR2GRAY) if kare.ndim == 3 else kare
                k += 1
            i += 1
    finally:
        cap.release()


def rapor(ad, preds, gt, saglik, sure_ms):
    d = np.linalg.norm(preds - gt, axis=1)
    kes = ~saglik
    dk = d[kes]                        # kisa denemelerde bos olabilir (kesinti 451'de baslar)
    sat = {
        "resmi": float(d.mean()),                 # sartname Denklem 2: TUM kareler
        "kesinti": float(dk.mean()) if dk.size else float("nan"),
        "medyan_kesinti": float(np.median(dk)) if dk.size else float("nan"),
        "p95_kesinti": float(np.percentile(dk, 95)) if dk.size else float("nan"),
        "maks": float(d.max()),
        "ms": float(np.mean(sure_ms)),
    }
    bloklar = []
    for i, (a, b) in enumerate(KESINTI, 1):
        b2 = min(b, len(d))
        if a < b2:
            bloklar.append((i, b2 - a, float(d[a:b2].mean()), float(d[b2 - 1])))
    return sat, bloklar, d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=int, default=1, choices=[1, 2])
    ap.add_argument("--n", type=int, default=None, help="ilk N kare (varsayilan: hepsi)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    vid_ad, csv_ad, aciklama = VIDEOLAR[a.video]
    video = ROOT / "THYZ_2026_Ornek_Veri_Seti" / vid_ad
    csv = ROOT / "THYZ_2026_Ornek_Veri_Seti" / csv_ad
    for p in (video, csv):
        if not p.exists():
            print(f"[HATA] bulunamadi: {p}")
            return 1

    gt = gt_oku(csv)
    n = len(gt) if a.n is None else min(a.n, len(gt))
    gt = gt[:n]
    saglik = saglik_takvimi(n)

    print("=" * 74)
    print(f"GOREV 2 — TEMMUZ vs SEVK   |   video{a.video} ({aciklama})")
    print("=" * 74)
    print(f"  kare      : {n}  (30 FPS videodan her {ADIM}. kare)")
    print(f"  saglikli  : {int(saglik.sum())}   kesinti: {int((~saglik).sum())}")
    print(f"  takvim    : {KESINTI}")

    kestiriciler, motorlar = {}, {}
    for ad, src in SURUMLER.items():
        if not src.exists():
            print(f"[HATA] {ad} kaynagi yok: {src}")
            return 1
        LPE = surum_yukle(ad, src)
        kestiriciler[ad] = LPE(engine="raft")
        print(f"  {ad:<7}  <- {src}")

    preds = {ad: np.zeros((n, 3)) for ad in kestiriciler}
    sureler = {ad: [] for ad in kestiriciler}

    t0 = time.perf_counter()
    for k, gray in enumerate(kareler(video, n)):
        h = bool(saglik[k])
        g = gt[k] if h else None
        for ad, est in kestiriciler.items():
            t = time.perf_counter()
            preds[ad][k] = est.process_frame(gray, g, h)
            sureler[ad].append((time.perf_counter() - t) * 1000)
        if k and k % 250 == 0:
            print(f"    {k}/{n} kare · {(time.perf_counter()-t0)/60:.1f} dk", flush=True)
    islenen = k + 1
    if islenen < n:
        print(f"  [!] video {islenen} karede bitti, kirpiliyor")
        n = islenen
        gt, saglik = gt[:n], saglik[:n]
        preds = {ad: v[:n] for ad, v in preds.items()}

    for ad, est in kestiriciler.items():
        if getattr(est, "engine_used", "raft") != "raft":
            print(f"  !!! {ad}: motor '{est.engine_used}' — RAFT kurulamadi, "
                  f"bu kosu RAFT recetesini DOGRULAMAZ !!!")

    sonuc = {ad: rapor(ad, preds[ad], gt, saglik, sureler[ad]) for ad in kestiriciler}

    print("\n" + "=" * 74)
    print("SONUC  —  sartname Denklem 2: ortalama Oklid hatasi [m], DUSUK IYI")
    print("=" * 74)
    b = ["resmi (tum kareler)", "kesinti kareleri", "kesinti medyan",
         "kesinti p95", "en kotu kare", "kare basi [ms]"]
    k = ["resmi", "kesinti", "medyan_kesinti", "p95_kesinti", "maks", "ms"]
    print(f"  {'olcut':<22}{'TEMMUZ':>12}{'SEVK':>12}{'fark':>12}{'':>8}")
    for ad, anah in zip(b, k):
        t, s = sonuc["TEMMUZ"][0][anah], sonuc["SEVK"][0][anah]
        f = s - t
        isaret = "" if anah == "ms" else ("  <- SEVK iyi" if f < -0.001 else
                                          ("  <- TEMMUZ iyi" if f > 0.001 else "  esit"))
        print(f"  {ad:<22}{t:>12.3f}{s:>12.3f}{f:>+12.3f}{isaret}")

    print("\n  KESINTI BLOKLARI (ortalama hata / blok sonu hatasi)")
    print(f"  {'blok':<18}{'TEMMUZ':>20}{'SEVK':>20}")
    for (i, uz, ot, st), (_, _, os_, ss) in zip(sonuc["TEMMUZ"][1], sonuc["SEVK"][1]):
        print(f"  blok{i} ({uz:3d} kare) {ot:>10.2f} /{st:>7.2f} {os_:>12.2f} /{ss:>7.2f}")

    yol = OUT / f"temmuz_vs_sevk_video{a.video}.txt"
    with open(yol, "w", encoding="utf-8") as f:
        f.write(f"GOREV 2 — TEMMUZ vs SEVK · video{a.video} ({aciklama}) · {n} kare\n")
        f.write(f"kesinti takvimi: {KESINTI}\n\n")
        for ad, anah in zip(b, k):
            t, s = sonuc["TEMMUZ"][0][anah], sonuc["SEVK"][0][anah]
            f.write(f"{ad:<22} TEMMUZ {t:9.3f}   SEVK {s:9.3f}   fark {s-t:+9.3f}\n")
    print(f"\n  kayit: {yol}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
