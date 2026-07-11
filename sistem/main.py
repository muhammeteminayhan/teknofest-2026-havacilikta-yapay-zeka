import logging
import time
from collections import deque
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import requests
from decouple import config
from tqdm import tqdm
from src.connection_handler import ConnectionHandler
from src.frame_predictions import FramePredictions
from src.object_detection_model import ObjectDetectionModel


class RateLimiter:
    """Sunucu limiti: dakikada en fazla 80 tahmin.

    Limit asilirsa sunucu tahmini VERITABANINA YAZMAZ (connection_handler
    dokumantasyonu) — yani asim dogrudan puan kaybidir. Bu sinif, son 60
    saniyedeki gonderim sayisi 80'e ulastiginda gerektigi kadar bekler.
    """

    def __init__(self, max_per_minute=80, guvenlik_payi=0.25):
        self.zamanlar = deque(maxlen=max_per_minute)
        self.guvenlik_payi = guvenlik_payi

    def bekle(self):
        if len(self.zamanlar) == self.zamanlar.maxlen:
            gecen = time.time() - self.zamanlar[0]
            if gecen < 60.0:
                time.sleep(60.0 - gecen + self.guvenlik_payi)

    def kaydet(self):
        self.zamanlar.append(time.time())


def configure_logger(team_name):
    log_folder = "./_logs/"
    Path(log_folder).mkdir(parents=True, exist_ok=True)
    log_filename = datetime.now().strftime(log_folder + team_name + '_%Y_%m_%d__%H_%M_%S_%f.log')
    logging.basicConfig(filename=log_filename, level=logging.INFO, encoding='utf-8',
                        format='%(asctime)s - %(levelname)s - %(message)s')


def run():
    print("Started...")
    # .env dosyasından konfigurasyon icin gereken bilgileri cek
    config.search_path = "./config/"
    team_name = config('TEAM_NAME')
    password = config('PASSWORD')
    evaluation_server_url = config("EVALUATION_SERVER_URL")

    # Logger'a ilgili konfigurasyon atamasini yap
    configure_logger(team_name)

    # Takimlar kendi kodlarini ObjectDetectionModel sinifi icerisine implemente edebilirler. (OPSIYONEL)
    detection_model = ObjectDetectionModel(evaluation_server_url)

    # Sunucuya baglan
    server = ConnectionHandler(evaluation_server_url, username=team_name, password=password)

    # /media gorsel indirmesi kimlik dogrulama ister -> token'i modele ver
    detection_model.auth_token = server.auth_token

    # Mevcutta aktif olan oturumdan tum frameleri cek.
    frames_json = server.get_frames(force_download=True)

    # Mevcutta aktif olan oturumdan tum translation verilerini cek.
    translations_json = server.get_translations(force_download=True)  

    # Klasorun ve dosyanin yollarini cekelim
    images_files, images_folder = server.get_listdir()

    # Gorev 3 referans nesnelerini kaydet (varsa). Sunucu, frames-JSON'daki
    # `references` anahtarinda referans objeleri veriyor. DIKKAT: referanslar ilk
    # karede degil, ilgili kare araliginin BASINDA gorunuyor (or. frame 165) — bu
    # yuzden TUM kareler taranir, sadece frames_json[0] degil. register_from_server
    # ayni `order`i tekrar kaydetmez. references hic yoksa NO-OP: Gorev 2 korunur.
    referanslar = []
    for _fr in (frames_json or []):
        referanslar.extend(_fr.get("references", []) or [])
    if referanslar:
        def _ref_indir(url):
            tam = url if str(url).startswith("http") else evaluation_server_url + "media" + url
            r = requests.get(tam, headers={"Authorization": f"Token {server.auth_token}"}, timeout=60)
            return cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
        n_ref = detection_model.register_references(referanslar, download=_ref_indir)
        logging.info(f"Gorev 3: {n_ref} referans kaydedildi (sunucudan).")
        print(f"Gorev 3: {n_ref} referans kaydedildi.")

    # 80 tahmin/dk sunucu limiti icin tempo denetleyici
    limiter = RateLimiter(max_per_minute=80)

    # Nesne tespiti modelini frame frame olacak sekilde calistir
    for frame, translation in tqdm(zip(frames_json, translations_json), total=len(frames_json)):
        # Prediction objesini frame ve translation bilgilerini tutacak sekilde olustur
        predictions = FramePredictions(frame['url'], frame['image_url'], frame['video_name'],
                                       translation['translation_x'], translation['translation_y'], translation['translation_z'])
        # Healt status kontrolu ikinci gorevde sistemin ne zaman devreye girmesi gerektigini gosterir
        health_status = translation['health_status']
        # Tespit modelini calistir
        predictions = detection_model.process(predictions, evaluation_server_url, health_status, images_folder, images_files)
        # Limiti asmamak icin gerekiyorsa bekle, sonra gonder
        limiter.bekle()
        result = server.send_prediction(predictions)
        limiter.kaydet()

    # oturum sonu: kare basina islem suresi ozeti (yarisma butcesi ~1.3 s/kare)
    logging.info(f"Islem suresi ozeti: {detection_model.position_estimator.timing_stats}")
    print(f"Islem suresi ozeti: {detection_model.position_estimator.timing_stats}")


if __name__ == '__main__':
    run()
