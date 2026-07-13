# -*- coding: utf-8 -*-
"""GERCEK sunucu TAM VERI TOPLAYICISI — yeni rolling arayuz + gercek model.

Test oturumunu bastan sona (2250 kare) kosar ve HER SEYI `sistem/sunucu_ciktilari/`'na
kaydeder:
  references.json + references/reference_N.webp     (Gorev 3 referanslari + gorselleri)
  frames/<image>.webp                               (her karenin gorseli)
  sunucu_frames.jsonl / sunucu_translations.jsonl   (ham sunucu verisi, kare basi 1 satir)
  bizim_cevaplar.jsonl                              (POST ettigimiz TAM payload, kare basi)
  sunucu_yanitlari.jsonl                            (HTTP kod + govde, kare basi)
  ozet.csv                                         (okunur: frame_id/health/tahmin/G1/G3/kod)
  kosum.log + ozet.txt                             (calisma logu + final istatistik)

DIKKAT: Kareyi ilerletmek icin test oturumuna TAHMIN GONDERIR — PUANLI oturumda CALISTIRMA!
Sadece test oturumunda (gunluk sifirlanir) kullan.

Kullanim (sistem/ klasorunden, config/.env = .env.online):
    ..\\.venv\\Scripts\\python sunucu_topla.py            # tam kosu (session bitene kadar)
    ..\\.venv\\Scripts\\python sunucu_topla.py --max 50   # kisa deneme
"""
import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import requests
from decouple import config

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from src.connection_handler import ConnectionHandler          # noqa: E402
from src.frame_predictions import FramePredictions            # noqa: E402
from src.object_detection_model import ObjectDetectionModel   # noqa: E402

CIKTI = HERE / "sunucu_ciktilari"
MIN_FRAME_INTERVAL = 0.75   # <=80 gonderim/dk (guvenli)


def _indir(base, token, image_url):
    turl = image_url if str(image_url).startswith("http") else base + "media" + image_url
    r = requests.get(turl, headers={"Authorization": f"Token {token}"}, timeout=60)
    r.raise_for_status()
    return r.content


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=None,
                    help="en fazla kare (varsayilan: oturum bitene kadar ~2250)")
    args = ap.parse_args()

    config.search_path = "./config/"
    base = config("EVALUATION_SERVER_URL")
    team = config("TEAM_NAME")
    pw = config("PASSWORD")

    CIKTI.mkdir(parents=True, exist_ok=True)
    (CIKTI / "frames").mkdir(exist_ok=True)
    (CIKTI / "references").mkdir(exist_ok=True)

    logging.basicConfig(filename=str(CIKTI / "kosum.log"), level=logging.INFO, encoding="utf-8",
                        format="%(asctime)s - %(levelname)s - %(message)s")

    def yaz(m):
        print(m, flush=True)
        logging.info(m)

    yaz(f"=== SUNUCU VERI TOPLAMA (tam, gercek model) === {base}")

    # Model (gercek cevap icin) + baglanti
    model = ObjectDetectionModel(base)
    server = ConnectionHandler(base, username=team, password=pw)
    model.auth_token = server.auth_token
    if not server.auth_token:
        yaz("HATA: login basarisiz — sunucu kapali ya da kimlik yanlis.")
        return

    progress = server.get_progress()
    if not progress or not progress.get("session_name"):
        yaz(f"HATA: aktif oturum yok. progress={progress}")
        return
    session_name = progress["session_name"]
    total = int(progress.get("total_frames") or 0)
    yaz(f"Oturum: {session_name} | toplam {total} | baslangic frame_index={progress.get('frame_index')}")
    if progress.get("completed"):
        yaz("Oturum zaten TAMAMLANMIS (completed=True). Sifirlanmasini bekle. Cikiliyor.")
        return

    server.video_name = session_name + "/"
    server.create_img_folder(server.video_name)
    images_folder = os.path.join(server.img_save_path, server.video_name)

    # --- Referanslar (GET /reference/) + gorselleri indir + kaydet + kaydet(register) ---
    refs = server.get_reference_objects(force_download=True) or []
    (CIKTI / "references.json").write_text(
        json.dumps(refs, ensure_ascii=False, indent=2), encoding="utf-8")
    ref_image_paths = {}
    for r in refs:
        try:
            icerik = _indir(base, server.auth_token, r["image_url"])
            p = CIKTI / "references" / r["image_url"].split("/")[-1]
            p.write_bytes(icerik)
            ref_image_paths[r["url"]] = str(p)
        except Exception as e:
            yaz(f"UYARI: referans gorseli inemedi ({r.get('url')}): {e}")
    n_ref = model.register_references(refs, ref_image_paths=ref_image_paths)
    yaz(f"Referans: {len(refs)} cekildi, {n_ref} matcher'a kaydedildi.")

    # --- cikti dosyalari (append yazim) ---
    f_frames = open(CIKTI / "sunucu_frames.jsonl", "w", encoding="utf-8")
    f_trans = open(CIKTI / "sunucu_translations.jsonl", "w", encoding="utf-8")
    f_cevap = open(CIKTI / "bizim_cevaplar.jsonl", "w", encoding="utf-8")
    f_yanit = open(CIKTI / "sunucu_yanitlari.jsonl", "w", encoding="utf-8")
    f_ozet = open(CIKTI / "ozet.csv", "w", encoding="utf-8", newline="")
    w = csv.writer(f_ozet)
    w.writerow(["iter", "frame_id", "health", "tahmin_x", "tahmin_y", "tahmin_z",
                "gorev1", "gorev3", "sunucu_kodu"])

    kodlar = {}
    saglikli_say = kesinti_say = cevirisiz_say = refpred_say = 0
    stuck_url = None
    stuck_n = 0
    i = 0
    t0 = time.perf_counter()

    while True:
        if args.max is not None and i >= args.max:
            yaz(f"--max {args.max} ulasildi, duruyorum.")
            break
        f_start = time.monotonic()

        frame = server.get_current_frame()
        if frame is None:
            yaz("Oturum tamamlandi / kare yok — bitti.")
            break

        # stuck-detection (kare ilerlemiyorsa sonsuz donguden kac)
        if frame["image_url"] == stuck_url:
            stuck_n += 1
            if stuck_n >= 5:
                yaz(f"Kare 5 kez ilerlemedi ({frame['image_url']}) — duruyorum.")
                break
        else:
            stuck_url = frame["image_url"]
            stuck_n = 0

        tr = server.get_current_translation()
        if tr is None:
            hs = None
            gx = gy = gz = None
        else:
            hs = tr.get("health_status")
            gx, gy, gz = tr.get("translation_x"), tr.get("translation_y"), tr.get("translation_z")

        # gorseli indir + kaydet + oku
        img = None
        try:
            icerik = _indir(base, server.auth_token, frame["image_url"])
            (CIKTI / "frames" / frame["image_url"].split("/")[-1]).write_bytes(icerik)
            img = cv2.imdecode(np.frombuffer(icerik, np.uint8), cv2.IMREAD_COLOR)
        except Exception as e:
            yaz(f"iter {i}: gorsel inemedi: {e}")

        # bu kare icin aktif (pencere-ici) referanslar (resmi image_url kapisi)
        active_refs = [
            r for r in refs
            if r.get("frame_start_image_url") and r.get("frame_end_image_url")
            and r["frame_start_image_url"] <= frame["image_url"] <= r["frame_end_image_url"]
        ]

        # BIZIM CEVAP: tam boru hatti (Gorev 1+2+3)
        pred = FramePredictions(frame["url"], frame["image_url"], frame["video_name"], gx, gy, gz)
        pred = model.detect(pred, hs, img, active_refs=active_refs, ref_image_paths=ref_image_paths)
        payload = pred.create_payload(base)

        # gonder (201 = ilerle) + yaniti kaydet
        resp = server.send_prediction(pred)
        kod = getattr(resp, "status_code", None)
        govde = getattr(resp, "text", "") or ""
        kodlar[kod] = kodlar.get(kod, 0) + 1

        # --- kaydet ---
        f_frames.write(json.dumps(frame, ensure_ascii=False) + "\n")
        f_trans.write(json.dumps(tr, ensure_ascii=False) + "\n")
        f_cevap.write(json.dumps({"iter": i, "frame_id": frame.get("id"),
                                  "payload": payload}, ensure_ascii=False) + "\n")
        f_yanit.write(json.dumps({"iter": i, "frame_id": frame.get("id"),
                                  "kod": kod, "govde": govde[:300]}, ensure_ascii=False) + "\n")

        dt = payload["detected_translations"][0] if payload["detected_translations"] else {}
        g1 = len(payload["detected_objects"])
        g3 = len(payload["reference_predictions"])
        refpred_say += g3
        if hs is None:
            cevirisiz_say += 1
        elif str(hs) == "1":
            saglikli_say += 1
        else:
            kesinti_say += 1
        w.writerow([i, frame.get("id"), hs,
                    dt.get("translation_x", ""), dt.get("translation_y", ""), dt.get("translation_z", ""),
                    g1, g3, kod])

        if i % 25 == 0:
            for fh in (f_frames, f_trans, f_cevap, f_yanit, f_ozet):
                fh.flush()
            hiz = (i + 1) / (time.perf_counter() - t0)
            yaz(f"  iter {i} | frame {frame.get('id')} | health {hs} | kod {kod} | "
                f"G1={g1} G3={g3} | {hiz:.2f} kare/s | kodlar={kodlar}")

        i += 1
        elapsed = time.monotonic() - f_start
        if elapsed < MIN_FRAME_INTERVAL:
            time.sleep(MIN_FRAME_INTERVAL - elapsed)

    for fh in (f_frames, f_trans, f_cevap, f_yanit, f_ozet):
        fh.close()

    dk = (time.perf_counter() - t0) / 60
    ozet = [
        "=== TOPLAMA OZETI ===",
        f"oturum            : {session_name}",
        f"islenen kare      : {i} / {total}",
        f"sure              : {dk:.1f} dk",
        f"sunucu kodlari    : {kodlar}",
        f"saglikli (h=1)    : {saglikli_say}",
        f"kesinti (h=0)     : {kesinti_say}",
        f"cevirisiz (None)  : {cevirisiz_say}",
        f"toplam reference_prediction gonderimi : {refpred_say}",
        f"referans sayisi   : {len(refs)}",
        f"cikti klasoru     : {CIKTI}",
        "",
        "Format dogrulama icin:  ..\\.venv\\Scripts\\python format_dogrula.py",
    ]
    (CIKTI / "ozet.txt").write_text("\n".join(ozet), encoding="utf-8")
    for satir in ozet:
        yaz(satir)


if __name__ == "__main__":
    main()
