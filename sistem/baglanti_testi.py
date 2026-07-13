"""Hafif sunucu baglanti testi — SALT OKUNUR (tahmin GONDERMEZ), model YUKLEMEZ.

Resmi README (Bolum 4) uyarisi: sunucu her kareye YALNIZCA BIR kez tahmin kabul eder;
hazir olmadan puanli oturumda tahmin gonderirsen o kareler kalici yanar. Bu betik bu
yuzden HICBIR tahmin gondermez — yalnizca yeni rolling arayuzun okuma uclarini dogrular:

  1. login (kimlik dogrulama)
  2. get_progress        -> oturum var mi? session_name / total_frames / frame_index
  3. get_current_frame   -> siradaki kare + SEMASI (url/image_url/video_name/references)
  4. get_current_translation -> translation_x/y/z + health_status semasi
  5. get_reference_objects   -> Gorev 3 referans listesi + semasi (/reference/ ucu)
  6. bir gorsel indir (/media ucu, auth testi)

Kullanim (sistem/ klasorunden, .env online'a ayarli):
    ..\\.venv\\Scripts\\python baglanti_testi.py
"""
import sys

import requests
from decouple import config

from src.connection_handler import ConnectionHandler

sys.stdout.reconfigure(encoding="utf-8")


def _sema(ad, d, beklenen):
    print(f"\n--- {ad} (gercek sunucu anahtarlari) ---")
    if isinstance(d, dict):
        for k, v in d.items():
            print(f"  {k!r}: {str(v)[:80]!r}")
        eksik = set(beklenen) - set(d.keys())
        if eksik:
            print(f"  !!! EKSIK anahtar: {eksik}")
        else:
            print(f"  OK sema: beklenen anahtarlar mevcut ({', '.join(beklenen)})")
    else:
        print(f"  (tip: {type(d).__name__}) {str(d)[:200]}")


def run():
    config.search_path = "./config/"
    team = config("TEAM_NAME")
    pw = config("PASSWORD")
    url = config("EVALUATION_SERVER_URL")
    print(f"Sunucu: {url}  | takim: {team}")
    print("NOT: Bu test SALT OKUNUR — hicbir tahmin gonderilmez (puanli oturumda guvenli).")

    # 1) login
    server = ConnectionHandler(url, username=team, password=pw)
    if not server.auth_token:
        print("HATA: login basarisiz (auth_token yok). Kullanici/sifre/URL kontrol et.")
        return
    print(f"\nOK login. token: {str(server.auth_token)[:12]}...")

    # 2) get_progress — oturum durumu / nerede kaldik
    progress = server.get_progress()
    if progress is None:
        print("HATA: get_progress None (sunucuya ulasilamadi). Baglantiyi/sunucuyu kontrol et.")
        return
    _sema("progress", progress, ["session_name", "total_frames", "frame_index", "completed"])
    if not progress.get("session_name"):
        print("\nUYARI: aktif oturum yok (session_name bos). Sunucu oturumu acmis mi?")
        return
    print(f"\nOturum: {progress['session_name']} | "
          f"kare {progress['frame_index']}/{progress['total_frames']} | "
          f"tamam={progress['completed']}")

    # 3) get_current_frame — siradaki kare (GONDERIM YOK, ilerlemez)
    frame = server.get_current_frame()
    if not frame:
        print("\nUYARI: get_current_frame bos (oturum bitmis olabilir).")
        return
    _sema("current_frame", frame, ["url", "image_url", "video_name"])

    # 4) get_current_translation — konum + saglik
    tr = server.get_current_translation()
    if tr:
        _sema("current_translation", tr,
              ["translation_x", "translation_y", "translation_z", "health_status"])
    else:
        print("\nUYARI: get_current_translation bos (bu kare icin ceviri yok olabilir).")

    # 5) get_reference_objects — Gorev 3 referanslari (/reference/ ucu)
    # Referans onbellegi icin klasoru olustur (main.py da boyle yapar; yoksa zararsiz uyari).
    server.video_name = progress["session_name"] + "/"
    server.create_img_folder(server.video_name)
    refs = server.get_reference_objects(force_download=True)
    if refs:
        print(f"\nOK get_reference_objects: {len(refs)} referans")
        _sema("references[0]", refs[0],
              ["url", "image_url", "order", "frame_start_image_url", "frame_end_image_url"])
    else:
        print("\nBILGI: get_reference_objects bos — test oturumunda referans olmayabilir "
              "(gercek yarismada gelince Gorev 3 otomatik aranir).")

    # 6) bir gorsel indir (/media ucu, auth)
    try:
        img_url = url + "media" + frame["image_url"]
        r = requests.get(img_url, headers={"Authorization": f"Token {server.auth_token}"},
                         timeout=30)
        print(f"\nOK /media indirme: HTTP {r.status_code}, {len(r.content)} bayt "
              f"({frame['image_url']})")
    except Exception as e:
        print(f"\nUYARI /media indirme: {type(e).__name__}: {e}")

    print("\n=== BAGLANTI TESTI TAMAM (salt okunur) ===")
    print("Okuma uclari calisiyor. Gercek kosum icin: ..\\.venv\\Scripts\\python main.py")


if __name__ == "__main__":
    run()
