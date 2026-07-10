# -*- coding: utf-8 -*-
"""Toplu test koşumu — SADECE TEST/GÖRME amaçlı (çalışan sistemi BOZMAZ).

Tam sistemi (Görev 1+2+3) örnek videonun ~2250 karesi (7.5 FPS altörnek) boyunca
koşturur; her kare için GELEN veriyi (sunucunun göndereceği) ve BİZİM CEVABIMIZI
(sunucuya POST edeceğimiz tam payload) dosyalara toplar. Gerçek sunucuya İSTEK
ATMAZ — yerel videoyu aynı `detect()` boru hattından geçirir, yani birebir aynı
cevapları üretir ama hızlı ve sunucuyu meşgul etmeden.

main.py ve src/ HİÇ değişmez; bu tamamen ayrı bir araçtır.

Kullanım (sistem/ klasöründen):
    ..\\.venv\\Scripts\\python test_toplu_kosum.py                 # tam 2250, 3 görev
    ..\\.venv\\Scripts\\python test_toplu_kosum.py --limit 30      # hızlı deneme
    ..\\.venv\\Scripts\\python test_toplu_kosum.py --no-gorev3     # Görev 3'süz (hızlı)

Çıktılar: sistem/test_ciktilari/
    bizim_cevaplar.jsonl   — kare başı POST edeceğimiz tam payload
    gelen_veri.jsonl       — kare başı gelen (health + gt translation)
    ozet.csv               — okunur özet (tahmin vs GT + konum hatası + nesne/eşleşme sayısı)
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT / "THYZ_2026_Ornek_Veri_Seti" / "THYZ_2026_Ornek_Veri_1.MP4"
CSV = ROOT / "THYZ_2026_Ornek_Veri_Seti" / "THYZ_2026_Ornek_Veri_1_translation.csv"
REFDIR = ROOT / "THYZ_2026_Ornek_Veri_Seti" / "THYZ_2026_Ornek_Veri_1_Referans_Nesneler"
CIKTI = Path(__file__).resolve().parent / "test_ciktilari"
# create_payload'ın cls URL'i için taban (gerçek sunucu ile aynı görünüm)
SERVER_URL = "http://havaciliktayapayzeka.teknofest.org:1025/"


def _f(v):
    try:
        f = float(v)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="ilk N kare (hızlı deneme)")
    ap.add_argument("--no-gorev3", dest="gorev3", action="store_false",
                    help="Görev 3'ü atla (daha hızlı)")
    ap.add_argument("--healthy-prefix", type=int, default=450,
                    help="ilk kaç kare sağlıklı (gerisi kesinti)")
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT / "sistem"))
    from src.object_detection_model import ObjectDetectionModel
    from src.frame_predictions import FramePredictions
    # ObjectDetectionModel import edilince gorev2_src kayıtlı olur
    from gorev2_src import data_io

    # --- altörnekleme + health takvimi (simülatörle AYNI) ---
    gt_df = data_io.subsample_competition(data_io.load_translations(CSV)).reset_index(drop=True)
    kept_idx = gt_df["frame_idx"].to_numpy()
    gt = gt_df[["translation_x", "translation_y", "translation_z"]].to_numpy()
    n = len(gt_df) if args.limit is None else min(args.limit, len(gt_df))
    healthy = np.arange(n) < args.healthy_prefix   # ilk 450 sağlıklı, gerisi kesinti
    print(f"Toplam {n} kare | ilk {min(args.healthy_prefix, n)} sağlıklı, "
          f"gerisi kesinti | Görev 3: {'AÇIK' if args.gorev3 else 'kapalı'}")

    # --- modeli yükle (3 görev) ---
    model = ObjectDetectionModel(SERVER_URL)
    if not args.gorev3 and model.gorev3 is not None:
        model.gorev3 = None   # Görev 3'ü bu koşum için devre dışı bırak (dosya değişmez)
    if args.gorev3 and model.gorev3 is not None:
        refs = sorted(REFDIR.glob("*.*"))
        model.register_references([{"url": str(r), "object_id": i} for i, r in enumerate(refs)])

    # --- çıktı dosyaları ---
    CIKTI.mkdir(parents=True, exist_ok=True)
    f_cevap = open(CIKTI / "bizim_cevaplar.jsonl", "w", encoding="utf-8")
    f_gelen = open(CIKTI / "gelen_veri.jsonl", "w", encoding="utf-8")
    f_ozet = open(CIKTI / "ozet.csv", "w", encoding="utf-8", newline="")
    w = csv.writer(f_ozet)
    w.writerow(["kare", "native_kare", "health", "tahmin_x", "tahmin_y", "tahmin_z",
                "gt_x", "gt_y", "gt_z", "konum_hata_m", "gorev1_nesne", "gorev3_eslesme"])

    # --- video sıralı okunur, sadece altörnek kareler işlenir ---
    cap = cv2.VideoCapture(str(VIDEO))
    kept_set = {int(x): i for i, x in enumerate(kept_idx[:n])}
    hata_toplam, hata_say = 0.0, 0
    t0 = time.perf_counter()
    native = 0
    islenen = 0
    while islenen < n:
        ok, frame = cap.read()
        if not ok:
            break
        if native in kept_set:
            i = kept_set[native]
            saglikli = bool(healthy[i])
            gx, gy, gz = float(gt[i, 0]), float(gt[i, 1]), float(gt[i, 2])
            # GELEN veri: gerçek sunucu gibi — sağlıklıysa değer+1, kesintide NaN+0
            if saglikli:
                pred = FramePredictions("u", f"/{native:06d}.webp", "v", gx, gy, gz)
                hs = "1"
            else:
                pred = FramePredictions("u", f"/{native:06d}.webp", "v", "NaN", "NaN", "NaN")
                hs = "0"
            # BİZİM CEVAP: tam boru hattı
            pred = model.detect(pred, hs, frame)
            payload = pred.create_payload(SERVER_URL)

            # tahmin edilen konumu payload'dan çek
            tx = ty = tz = None
            if payload["detected_translations"]:
                dt = payload["detected_translations"][0]
                tx, ty, tz = _f(dt["translation_x"]), _f(dt["translation_y"]), _f(dt["translation_z"])
            hata = ""
            if None not in (tx, ty, tz):
                hata = float(np.linalg.norm([tx - gx, ty - gy, tz - gz]))
                hata_toplam += hata
                hata_say += 1

            # yaz (artımlı — koşarken bakılabilir)
            f_gelen.write(json.dumps({"kare": i, "native_kare": native, "health_status": hs,
                                      "gt_translation_x": gx, "gt_translation_y": gy,
                                      "gt_translation_z": gz}, ensure_ascii=False) + "\n")
            f_cevap.write(json.dumps({"kare": i, "payload": payload}, ensure_ascii=False) + "\n")
            w.writerow([i, native, hs,
                        f"{tx:.3f}" if tx is not None else "",
                        f"{ty:.3f}" if ty is not None else "",
                        f"{tz:.3f}" if tz is not None else "",
                        f"{gx:.3f}", f"{gy:.3f}", f"{gz:.3f}",
                        f"{hata:.3f}" if hata != "" else "",
                        len(payload["detected_objects"]),
                        len(payload["detected_undefined_objects"])])
            islenen += 1
            if islenen % 50 == 0:
                f_gelen.flush(); f_cevap.flush(); f_ozet.flush()
                hiz = islenen / (time.perf_counter() - t0)
                ort_hata = hata_toplam / hata_say if hata_say else 0
                print(f"  {islenen}/{n} kare | {hiz:.1f} kare/s | ort konum hata {ort_hata:.2f} m",
                      flush=True)
        native += 1
    cap.release()
    f_cevap.close(); f_gelen.close(); f_ozet.close()

    dk = (time.perf_counter() - t0) / 60
    ort_hata = hata_toplam / hata_say if hata_say else 0
    print(f"\nBİTTİ: {islenen} kare, {dk:.1f} dk | ort konum hatası {ort_hata:.2f} m")
    print(f"Çıktılar: {CIKTI}")
    print("  bizim_cevaplar.jsonl  (POST edeceğimiz tam payload'lar)")
    print("  gelen_veri.jsonl      (sunucunun göndereceği veri)")
    print("  ozet.csv              (tahmin vs GT + konum hatası — Excel'de aç)")


if __name__ == "__main__":
    main()
