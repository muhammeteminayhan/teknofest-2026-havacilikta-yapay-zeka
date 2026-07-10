# -*- coding: utf-8 -*-
"""GERCEK sunucu kesif toplayicisi — SADECE TEST/GORME (calisan sistemi bozmaz).

Gercek competition sunucusuna dongüyle istek atar (rate limitlere uyarak) ve her
iterasyonun HER SEYINI kaydeder: sunucunun ham frames.json + translations.json,
indirilen gorsel, bizim POST ettigimiz cevap (payload) ve sunucunun yaniti (201/406/..).
Amac: test oturumu kac kare veriyor, kare nasil ilerliyor, limitler nasil davraniyor GORMEK.

Kullanim (sistem/ klasorunden, sunucu ACIK olmali):
    ..\\.venv\\Scripts\\python sunucu_topla.py --max 15
    ..\\.venv\\Scripts\\python sunucu_topla.py --max 15 --bekle 13 --gorevsiz  # model yuklemeden hizli

Cikti: sistem/sunucu_ciktilari/
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import requests
from decouple import Config, RepositoryEnv

CIKTI = Path(__file__).resolve().parent / "sunucu_ciktilari"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=15, help="en fazla iterasyon")
    ap.add_argument("--bekle", type=float, default=13.0, help="get_frames arasi sn (5/dk altinda)")
    ap.add_argument("--gorevsiz", action="store_true",
                    help="modelleri yukleme; tahmin yerine echo/0 gonder (mekanizmayi hizli gor)")
    ap.add_argument("--dur-tekrar", type=int, default=4,
                    help="kare id bu kadar kez ilerlemezse dur")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from src.connection_handler import ConnectionHandler
    from src.frame_predictions import FramePredictions
    from src.detected_translation import DetectedTranslation

    # --- online kimlik (.env.online'i DOGRUDAN oku; aktif .env'e dokunma) ---
    cfg = Config(RepositoryEnv(str(Path(__file__).resolve().parent / "config" / ".env.online")))
    base = cfg("EVALUATION_SERVER_URL")
    server = ConnectionHandler(base, username=cfg("TEAM_NAME"), password=cfg("PASSWORD"))
    if not server.auth_token:
        print("HATA: login basarisiz — sunucu kapali ya da kimlik yanlis.")
        return
    print(f"Login OK. Sunucu: {base}")

    # --- model (gercek cevap icin) ---
    model = None
    if not args.gorevsiz:
        from src.object_detection_model import ObjectDetectionModel
        model = ObjectDetectionModel(base)
        model.auth_token = server.auth_token

    CIKTI.mkdir(parents=True, exist_ok=True)
    toplu = open(CIKTI / "toplu.jsonl", "w", encoding="utf-8")

    gorulen_idler, son_id, tekrar = [], None, 0
    kodlar = {}
    ref_kaydedildi = False

    for k in range(args.max):
        if k > 0:
            time.sleep(args.bekle)   # get_frames 5/dk limiti
        try:
            frames = server.get_frames(force_download=True)
            translations = server.get_translations(force_download=True)
        except Exception as e:
            print(f"iter {k}: get_frames/translations hatasi: {e}")
            break
        if not frames:
            print(f"iter {k}: sunucu kare dondurmedi, duruyorum.")
            break

        fr, tr = frames[0], (translations[0] if translations else {})
        fid = fr.get("id")
        # ham sunucu yanitlarini kaydet
        (CIKTI / f"iter_{k:03d}_frames.json").write_text(json.dumps(frames, ensure_ascii=False, indent=2), encoding="utf-8")
        (CIKTI / f"iter_{k:03d}_translations.json").write_text(json.dumps(translations, ensure_ascii=False, indent=2), encoding="utf-8")

        # Gorev 3 referanslari (varsa, ilk kez)
        if model is not None and not ref_kaydedildi and fr.get("references"):
            def _ind(u):
                t = u if str(u).startswith("http") else base + "media" + u
                rr = requests.get(t, headers={"Authorization": f"Token {server.auth_token}"}, timeout=60)
                return cv2.imdecode(np.frombuffer(rr.content, np.uint8), cv2.IMREAD_COLOR)
            model.register_references(fr["references"], download=_ind)
            ref_kaydedildi = True

        # gorseli indir + kaydet
        img = None
        try:
            iu = fr["image_url"]
            turl = iu if str(iu).startswith("http") else base + "media" + iu
            resp = requests.get(turl, headers={"Authorization": f"Token {server.auth_token}"}, timeout=60)
            (CIKTI / f"iter_{k:03d}.webp").write_bytes(resp.content)
            img = cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"iter {k}: gorsel indirilemedi: {e}")

        # tahmin uret (gercek model ya da echo)
        pred = FramePredictions(fr["url"], fr.get("image_url"), fr.get("video_name"),
                                tr.get("translation_x"), tr.get("translation_y"), tr.get("translation_z"))
        hs = str(tr.get("health_status", "0"))
        if model is not None and img is not None:
            pred = model.detect(pred, hs, img)
        else:
            # gorevsiz: saglikliysa echo, degilse 0,0,0
            if hs == "1":
                pred.add_translation_object(DetectedTranslation(tr.get("translation_x"), tr.get("translation_y"), tr.get("translation_z")))
            else:
                pred.add_translation_object(DetectedTranslation(0.0, 0.0, 0.0))

        payload = pred.create_payload(base)
        (CIKTI / f"iter_{k:03d}_bizim_cevap.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        # gonder + sunucu yanitini kaydet
        r = server.send_prediction(pred)
        kod = getattr(r, "status_code", None)
        govde = getattr(r, "text", "")
        (CIKTI / f"iter_{k:03d}_sunucu_yanit.txt").write_text(f"HTTP {kod}\n{govde}", encoding="utf-8")
        kodlar[kod] = kodlar.get(kod, 0) + 1

        # bizim tahmin edilen konum (ozet icin)
        bt = payload["detected_translations"][0] if payload["detected_translations"] else {}
        toplu.write(json.dumps({"iter": k, "frame_id": fid, "image_url": fr.get("image_url"),
                                "health_status": hs, "bizim_translation": bt,
                                "gorev1_nesne": len(payload["detected_objects"]),
                                "gorev3_eslesme": len(payload["detected_undefined_objects"]),
                                "sunucu_kodu": kod}, ensure_ascii=False) + "\n")
        toplu.flush()

        gorulen_idler.append(fid)
        print(f"iter {k:02d} | frame id {fid} | health {hs} | sunucu {kod} | "
              f"G1 {len(payload['detected_objects'])} G3 {len(payload['detected_undefined_objects'])}", flush=True)

        # kare ilerliyor mu?
        if fid == son_id:
            tekrar += 1
            if tekrar >= args.dur_tekrar:
                print(f"Kare {args.dur_tekrar} kez ilerlemedi (id {fid}) — duruyorum.")
                break
        else:
            tekrar = 0
        son_id = fid

    toplu.close()
    farkli = sorted(set(x for x in gorulen_idler if x is not None))
    print("\n=== RAPOR ===")
    print(f"toplam iterasyon    : {len(gorulen_idler)}")
    print(f"FARKLI kare (id)    : {len(farkli)} -> {farkli}")
    print(f"sunucu kod dagilimi : {kodlar}")
    print(f"Cikti: {CIKTI}")
    print("Yorum: FARKLI kare cok azsa sunucu 2250'yi puanli oturumda verecek demektir.")


if __name__ == "__main__":
    main()
