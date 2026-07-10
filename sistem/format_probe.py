"""Gercek sunucu detected_objects FORMATINI dogrula.

Bir kareye Gorev 1 tespiti (DetectedObject, cls URL + motion_status) + translation
gonderip yaniti yazdirir. 201 = format kabul; 4xx/5xx = format sorunu (detay gelir);
406 = zaten gonderilmis (baska kareyi dene).
"""
import json
import sys

from decouple import config

from src.connection_handler import ConnectionHandler
from src.frame_predictions import FramePredictions
from src.detected_object import DetectedObject
from src.detected_translation import DetectedTranslation

sys.stdout.reconfigure(encoding="utf-8")
config.search_path = "./config/"
server = ConnectionHandler(config("EVALUATION_SERVER_URL"),
                           username=config("TEAM_NAME"), password=config("PASSWORD"))
frames = server.get_frames(force_download=True)
translations = server.get_translations(force_download=True)
print(f"{len(frames)} kare mevcut\n")

for i, (fr, tr) in enumerate(zip(frames, translations)):
    pred = FramePredictions(fr["url"], fr["image_url"], fr["video_name"],
                            tr.get("translation_x"), tr.get("translation_y"), tr.get("translation_z"))
    # ornek bir Gorev 1 tespiti: Tasit(0), hareketsiz, kare ortasinda kutu
    pred.add_detected_object(DetectedObject(cls=0, landing_status="-1",
                                            top_left_x=100, top_left_y=100,
                                            bottom_right_x=200, bottom_right_y=200,
                                            motion_status="1"))  # HAREKETLI: sunucu 1 saklamali
    pred.add_translation_object(DetectedTranslation(tr.get("translation_x", "0"),
                                                    tr.get("translation_y", "0"),
                                                    tr.get("translation_z", "0")))
    payload = pred.create_payload(server.base_url)
    print(f"--- kare {i} gonderilen detected_objects[0] ---")
    print(json.dumps(payload["detected_objects"][0], indent=2, ensure_ascii=False))
    resp = server.send_prediction(pred)
    kod = getattr(resp, "status_code", None)
    govde = getattr(resp, "text", "")
    print(f"YANIT: HTTP {kod} — {govde[:300]}\n")
    if kod == 201:
        print("OK: detected_objects formati KABUL EDILDI (cls URL + motion_status).")
        break
    elif kod == 406:
        print("(zaten gonderilmis, sonraki kareyi deniyorum...)\n")
        continue
    else:
        print("!!! FORMAT SORUNU — yukaridaki yanit detayina bak.")
        break
