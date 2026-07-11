# -*- coding: utf-8 -*-
"""SUNI TEST — Gorev 3 entegrasyonu + tum sistem dogru calisiyor mu?

Yakalanan GERCEK sunucu karelerini (sistem/sunucu_ciktilari/) kullanir. Her
referansin arama aralarigini BASLANGIC-3'ten BITIS+3'e kadar tam boru hattindan
(detect: Gorev 1+2+3) gecirir. Amac: entegrasyonun (referans kaydi + kare araligi
filtresi) ve tum sistemin (cokme yok, her gorev calisiyor) dogrulugunu gormek.

Beklenen: Gorev 3 SADECE referansin kendi araliginda "aktif" (aranir); aralik
disinda (once-3, sonra+3) aktif referans YOK -> aranmaz. Hicbir karede hata olmaz.

Calistir:  ..\\.venv\\Scripts\\python suni_test\\suni_test.py   (proje kokunden)
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAP = ROOT / "sistem" / "sunucu_ciktilari"       # yakalanan gercek veriler
OUT = Path(__file__).resolve().parent            # suni_test/
sys.path.insert(0, str(ROOT / "sistem"))
sys.stdout.reconfigure(encoding="utf-8")

import cv2
import numpy as np
from src.object_detection_model import ObjectDetectionModel
from src.frame_predictions import FramePredictions

FRAME0_ITER = 13   # iter 0 -> frame id 13 (yakalanan veri boyle)

# Referanslarin arama araliklari (gercek sunucudan gozlemlendi)
REFERANSLAR = {1: (165, 209), 2: (525, 656)}
BUFFER = 3


def iter_of(frame_id):
    return frame_id - FRAME0_ITER


def ref_indir_yerel(url):
    """Referans image_url'unu yerel kaydedilmis PNG'den oku (sunucuya gitmeden)."""
    ad = "reference_1" if "reference_1" in str(url) else "reference_2"
    return cv2.imread(str(CAP / f"{ad}.png"))


def main():
    print("=== SUNI TEST — Gorev 3 araliklari + tum sistem ===")
    model = ObjectDetectionModel("http://havaciliktayapayzeka.teknofest.org:1025/")
    print(f"gorev1={model.gorev1 is not None} gorev3={model.gorev3 is not None}\n")

    # test edilecek iter'ler: her referans araligi +- BUFFER
    test_frames = []
    for oid, (a, b) in REFERANSLAR.items():
        for fid in range(a - BUFFER, b + BUFFER + 1):
            test_frames.append((oid, fid))
    # (bazi kareler iki araliga da girebilir; sirali + benzersiz)
    gorulen = set()
    sirali = []
    for oid, fid in sorted(test_frames, key=lambda x: x[1]):
        if fid not in gorulen:
            sirali.append(fid); gorulen.add(fid)

    rows = []
    cevaplar = []   # bizim sunucuya gonderecegimiz TAM payload'lar
    hata = 0
    for fid in sirali:
        it = iter_of(fid)
        fpath = CAP / f"iter_{it:03d}_frames.json"
        tpath = CAP / f"iter_{it:03d}_translations.json"
        ipath = CAP / f"iter_{it:03d}.webp"
        if not (fpath.exists() and ipath.exists()):
            continue
        fr = json.load(open(fpath, encoding="utf-8"))[0]
        tr = (json.load(open(tpath, encoding="utf-8")) or [{}])[0]
        img = cv2.imread(str(ipath))

        # referans geldiyse kaydet (gercekte main.py de boyle yapiyor)
        if fr.get("references"):
            model.register_references(fr["references"], download=ref_indir_yerel)

        pred = FramePredictions(fr["url"], fr.get("image_url"), fr.get("video_name"),
                                tr.get("translation_x"), tr.get("translation_y"), tr.get("translation_z"))
        hs = str(tr.get("health_status", "0"))

        # bu kare icin Gorev 3 hangi referanslari ARAYACAK? (entegrasyon dogrulamasi)
        arliklar = model.gorev3.ref_araliklar if model.gorev3 else {}
        aktif = sorted(oid for oid, (a, b) in arliklar.items() if a <= fid <= b)

        pay = None
        try:
            pred = model.detect(pred, hs, img)
            pay = pred.create_payload("http://havaciliktayapayzeka.teknofest.org:1025/")
            g1 = len(pay["detected_objects"]); g3 = len(pay["detected_undefined_objects"])
            durum = "OK"
        except Exception as e:
            g1 = g3 = -1; durum = f"HATA: {e}"; hata += 1

        # bu kare hangi referans araliginda?
        icinde = [oid for oid, (a, b) in REFERANSLAR.items() if a <= fid <= b]

        # BIZIM sunucuya gonderecegimiz TAM cevap (payload) — kaydet
        if pay is not None:
            cevaplar.append({"frame_id": fid, "aralik_ici": icinde or None,
                             "health_status": hs, "payload": pay})
        rows.append(dict(frame_id=fid, health=hs, aralik_ici=icinde or "-",
                         g3_aranan_ref=aktif or "-", g3_eslesme=g3, g1_nesne=g1, durum=durum))
        isaret = "  <-- ARALIK ICI" if icinde else ""
        print(f"frame {fid} | health {hs} | G3 aranan={aktif or '[]'} | "
              f"G3 eslesme={g3} | G1={g1} | {durum}{isaret}")

    # kaydet: (1) ozet CSV, (2) BIZIM tam payload'lar (JSONL).
    # Dosya acik/kilitliyse (Excel'de aciksa) alternatif isme yaz — crash etme.
    def _yaz(path, fn):
        try:
            fn(path); return path
        except PermissionError:
            alt = path.with_name(path.stem + "_2" + path.suffix)
            fn(alt); print(f"UYARI: {path.name} kilitli (acik?), {alt.name}'e yazildi."); return alt

    def _csv(p):
        with open(p, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)

    def _jsonl(p):
        with open(p, "w", encoding="utf-8") as f:
            for c in cevaplar:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")

    csv_path = _yaz(OUT / "suni_test_sonuc.csv", _csv)
    jsonl_path = _yaz(OUT / "bizim_cevaplar.jsonl", _jsonl)

    # DOGRULAMA
    print("\n=== DOGRULAMA ===")
    yanlis = 0
    for r in rows:
        ici = r["aralik_ici"] != "-"
        arandi = r["g3_aranan_ref"] != "-"
        if ici != arandi:  # aralik ici ama aranmadi, ya da disi ama arandi
            yanlis += 1
            print(f"  ! frame {r['frame_id']}: aralik_ici={ici} ama g3_aranan={r['g3_aranan_ref']}")
    print(f"toplam test karesi   : {len(rows)}")
    print(f"detect() HATASI      : {hata}")
    print(f"aralik-filtre YANLISI: {yanlis}  (0 olmali: G3 sadece kendi araliginda aranmali)")
    print(f"Ozet CSV             : {csv_path}")
    print(f"Bizim cevaplar JSONL : {jsonl_path}  ({len(cevaplar)} kare, tam payload)")
    if hata == 0 and yanlis == 0:
        print("\nSONUC: ENTEGRASYON + TUM SISTEM DOGRU CALISIYOR [OK]")
    else:
        print("\nSONUC: DIKKAT - yukaridaki sorunlara bak.")


if __name__ == "__main__":
    main()
