"""Hafif sunucu baglanti testi — Gorev 2 boru hattini/RAFT'i YUKLEMEDEN.

Amac (yarisma e-postasi): "saglikli sekilde sunucu baglantisi saglayabildiginizi
test etmek". Bu betik agir modelleri yuklemeden yalnizca AG katmanini dogrular:
  1. login (kimlik dogrulama)
  2. get_frames + get_translations
  3. GERCEK SUNUCU JSON SEMASINI OLDUGU GIBI yazdir (main.py'nin bekledigi
     anahtarlarla — url/image_url/video_name, translation_x/y/z, health_status —
     uyusuyor mu? Uyusmuyorsa main.py'yi ona gore uyarlariz.)
  4. bir gorsel indir (/media ucu testi)
  5. ilk ~10 kareye gecerli-ama-basit tahmin gonder, 201 dogrula
     (saglikli kare -> gelen translation'i echo; degilse 0,0,0)

Kullanim (sistem/ klasorunden, .env online'a ayarli):
    python baglanti_testi.py
"""
import json
import logging
import sys
from pathlib import Path

from decouple import config

from src.connection_handler import ConnectionHandler
from src.frame_predictions import FramePredictions
from src.detected_translation import DetectedTranslation

sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

N_TEST = 10   # kac kareye tahmin gonderilecek (limit 80/dk cok altinda)


def _sozluk_yazdir(ad, d):
    print(f"\n--- {ad} (gercek sunucu anahtarlari) ---")
    if isinstance(d, dict):
        for k, v in d.items():
            print(f"  {k!r}: {v!r}")
    else:
        print(f"  (tip: {type(d).__name__}) {str(d)[:200]}")


def run():
    config.search_path = "./config/"
    team = config("TEAM_NAME")
    pw = config("PASSWORD")
    url = config("EVALUATION_SERVER_URL")
    print(f"Sunucu: {url}  | takim: {team}")

    # 1) login
    server = ConnectionHandler(url, username=team, password=pw)
    if not server.auth_token:
        print("HATA: login basarisiz (auth_token yok). Kullanici/sifre/URL kontrol et.")
        return
    print(f"OK login. token: {str(server.auth_token)[:12]}...")

    # 2) frames + translations
    frames = server.get_frames(force_download=True)
    translations = server.get_translations(force_download=True)
    print(f"\nOK get_frames: {len(frames) if frames else 0} kare | "
          f"get_translations: {len(translations) if translations else 0} kayit")
    if not frames or not translations:
        print("HATA: kare/translation listesi bos. Oturum aktif mi?")
        return

    # 3) GERCEK SEMA — ilk kare + translation sozlugu
    _sozluk_yazdir("frames[0]", frames[0])
    _sozluk_yazdir("translations[0]", translations[0])

    # main.py'nin bekledigi anahtarlar var mi?
    beklenen_f = {"url", "image_url", "video_name"}
    beklenen_t = {"translation_x", "translation_y", "translation_z", "health_status"}
    eksik_f = beklenen_f - set(frames[0].keys())
    eksik_t = beklenen_t - set(translations[0].keys())
    if eksik_f or eksik_t:
        print(f"\n!!! SEMA UYUSMAZLIGI — main.py uyarlanmali. "
              f"Eksik frame anahtari: {eksik_f or '-'} | translation: {eksik_t or '-'}")
    else:
        print("\nOK sema: main.py'nin bekledigi tum anahtarlar mevcut.")

    # 4) bir gorsel indir (/media ucu)
    try:
        images_files, images_folder = server.get_listdir()
        ObjectDownload = __import__("src.object_detection_model", fromlist=["ObjectDetectionModel"])
        # sadece statik indirme fonksiyonunu kullan (model yuklemeden)
        img_url = url + "media" + frames[0]["image_url"]
        import requests
        r = requests.get(img_url, timeout=30)
        print(f"\nOK /media indirme: HTTP {r.status_code}, {len(r.content)} bayt "
              f"({frames[0]['image_url']})")
    except Exception as e:
        print(f"\nUYARI /media indirme denemesi: {type(e).__name__}: {e}")

    # 5) ilk N kareye gecerli-basit tahmin gonder
    print(f"\nIlk {N_TEST} kareye basit tahmin gonderiliyor...")
    basarili = 0
    for i in range(min(N_TEST, len(frames))):
        fr, tr = frames[i], translations[i]
        pred = FramePredictions(fr["url"], fr["image_url"], fr["video_name"],
                                tr.get("translation_x"), tr.get("translation_y"),
                                tr.get("translation_z"))
        # saglikli -> echo, degilse 0,0,0
        saglikli = str(tr.get("health_status")) == "1"
        if saglikli:
            tx, ty, tz = tr["translation_x"], tr["translation_y"], tr["translation_z"]
        else:
            tx, ty, tz = 0.0, 0.0, 0.0
        pred.add_translation_object(DetectedTranslation(tx, ty, tz))
        resp = server.send_prediction(pred)
        kod = getattr(resp, "status_code", None)
        if kod == 201:
            basarili += 1
        else:
            govde = getattr(resp, "text", "")[:200]
            print(f"  kare {i}: HTTP {kod} — {govde}")
    print(f"\nSONUC: {basarili}/{min(N_TEST, len(frames))} tahmin 201 (basarili).")
    print("Baglanti testi tamam." if basarili else "Gonderim basarisiz — yukaridaki yanitlara bak.")


if __name__ == "__main__":
    run()
