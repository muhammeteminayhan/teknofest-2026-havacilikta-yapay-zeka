import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import requests

from .constants import classes, landing_statuses
from .detected_object import DetectedObject
from .detected_translation import DetectedTranslation
from .detected_undefined_object import DetectedUndefinedObject

# ---------------------------------------------------------------------------
# Gorev 2 (pozisyon kestirimi) boru hattini ice aktar: gorev2, bu deponun
# kardes klasorudur (havacilikta-yz-project/gorev2). Bu deponun kendi 'src'
# paketi ile isim cakismasini onlemek icin gorev2/src ayri bir paket adiyla
# (gorev2_src) yuklenir.
# ---------------------------------------------------------------------------
import importlib.util  # noqa: E402

GOREV2_SRC = Path(__file__).resolve().parents[2] / "gorev2" / "src"
if "gorev2_src" not in sys.modules:
    _spec = importlib.util.spec_from_file_location(
        "gorev2_src", GOREV2_SRC / "__init__.py",
        submodule_search_locations=[str(GOREV2_SRC)])
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules["gorev2_src"] = _mod
    _spec.loader.exec_module(_mod)

from gorev2_src.live_estimator import LivePositionEstimator, pre_flight_check  # noqa: E402

# Gorev 1 (nesne tespiti) + Gorev 3 (goruntu esleme) — proje kokundeki
# gorev1/ ve gorev3/ paketleri.
PROJE_KOK = Path(__file__).resolve().parents[2]
if str(PROJE_KOK) not in sys.path:
    sys.path.insert(0, str(PROJE_KOK))


def _guvenli_float(deger):
    """Sunucudan gelen degeri guvenle float'a cevir; 'NaN'/bozuk ise None."""
    try:
        f = float(deger)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


class ObjectDetectionModel:
    # Takim modeli: Gorev 2 pozisyon kestirimi entegre edildi.
    # Gorev 1 (nesne tespiti) icin isaretli bolume bakiniz.

    def __init__(self, evaluation_server_url):
        logging.info('Created Object Detection Model')
        self.evaulation_server = evaluation_server_url
        # Gercek sunucuda /media ucu KIMLIK DOGRULAMA ister (tokensiz 401).
        # main.py, server olusturduktan sonra bunu doldurur.
        self.auth_token = None

        # Yarisma gunu kontrol listesi: GPU / RAFT agirliklari / motor secimi
        kontrol = pre_flight_check()
        logging.info(f"On kontrol: {kontrol}")
        print(f"On kontrol: {kontrol}")

        # Pozisyon kestirici: motoru ilk karede (cozunurluk belli olunca) kurar.
        # RAFT (GPU) kurulamazsa kendiliginden LK yedegine duser ve loglar.
        self.position_estimator = LivePositionEstimator(engine="auto")

        # Gorev 1 tespit modeli (vanilla YOLOv8s). Yuklenemezse Gorev 2 yine calisir.
        self.gorev1 = None
        try:
            from gorev1 import Gorev1Detector
            self.gorev1 = Gorev1Detector()
            logging.info("Gorev 1 detektoru yuklendi.")
            print("Gorev 1 detektoru yuklendi.")
        except Exception as e:
            logging.error(f"Gorev 1 detektoru YUKLENEMEDI (Gorev 2 devam eder): {e}")
            print(f"UYARI: Gorev 1 detektoru yuklenemedi: {e}")

        # Gorev 3 eslestirici (DINOv3 omurga). Referanslar oturum basinda
        # register_references(...) ile kaydedilir; kayit yoksa match() bos doner.
        # Yuklenemezse Gorev 1/2 devam eder.
        self.gorev3 = None
        try:
            from gorev3 import Gorev3Matcher
            self.gorev3 = Gorev3Matcher(kind="dinov3")
            logging.info("Gorev 3 eslestirici yuklendi.")
            print("Gorev 3 eslestirici yuklendi.")
        except Exception as e:
            logging.error(f"Gorev 3 eslestirici YUKLENEMEDI (Gorev 1/2 devam eder): {e}")
            print(f"UYARI: Gorev 3 eslestirici yuklenemedi: {e}")

    def register_references(self, references, download=None):
        """Gorev 3 referans nesnelerini kaydet (main, frames-JSON `references`
        anahtarini gecirir). download(url) verilirse auth'lu indirir."""
        if self.gorev3 is None:
            return 0
        return self.gorev3.register_from_server(references, download=download)

    @staticmethod
    def download_image(img_url, images_folder, images_files, auth_token=None,
                       retries=3, initial_wait_time=0.1):
        t1 = time.perf_counter()
        wait_time = initial_wait_time
        # Gercek sunucu /media ucu icin Authorization sart (yerel simulatorde
        # zararsiz). Token yoksa header gonderilmez.
        headers = {"Authorization": f"Token {auth_token}"} if auth_token else {}
        image_name = img_url.split("/")[-1]
        if image_name not in images_files:
            for attempt in range(retries):
                    try:
                        response = requests.get(img_url, headers=headers, timeout=60)
                        response.raise_for_status()

                        img_bytes = response.content
                        with open(images_folder + image_name, 'wb') as img_file:
                            img_file.write(img_bytes)

                        t2 = time.perf_counter()
                        logging.info(f'{img_url} - Download Finished in {t2 - t1} seconds to {images_folder + image_name}')
                        return
                    except requests.exceptions.RequestException as e:
                        logging.error(f"Download failed for {img_url} on attempt {attempt + 1}: {e}")
                        logging.info(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                        wait_time *= 2

            logging.error(f"Failed to download image from {img_url} after {retries} attempts.")
        else:
            logging.info(f'{image_name} already exists in {images_folder}, skipping download.')

    def process(self, prediction, evaluation_server_url, health_status, images_folder, images_files):
        # Kareyi indir ve oku (gercek sunucu icin auth token ile)
        self.download_image(evaluation_server_url + "media" + prediction.image_url,
                            images_folder, images_files, auth_token=self.auth_token)
        image_name = prediction.image_url.split("/")[-1]
        image = cv2.imread(str(Path(images_folder) / image_name))
        if image is None:
            logging.error(f"Goruntu okunamadi: {image_name}")
        frame_results = self.detect(prediction, health_status, image)
        return frame_results

    def detect(self, prediction, health_status, image):
        # -------------------------------------------------------------------
        # GOREV 1 — NESNE TESPITI (best.pt / YOLOv8-DINOv3 + BoT-SORT)
        # Gorev1Detector cikti sozluklerini DetectedObject'e cevirir.
        # Bir karede hata olursa loglanir, oturum devam eder (bos liste = gecerli,
        # sahte kutu gonderilmez).
        # -------------------------------------------------------------------
        if self.gorev1 is not None and image is not None:
            try:
                for o in self.gorev1.detect(image):
                    prediction.add_detected_object(DetectedObject(
                        cls=int(o["cls"]),
                        landing_status=o["landing_status"],
                        top_left_x=o["top_left_x"], top_left_y=o["top_left_y"],
                        bottom_right_x=o["bottom_right_x"], bottom_right_y=o["bottom_right_y"],
                        motion_status=o["motion_status"],
                    ))
            except Exception as e:
                logging.error(f"Gorev 1 detect hatasi (bu kare atlandi): {e}")

        # -------------------------------------------------------------------
        # GOREV 3 — GORUNTU ESLEME (referanslar register edildiyse)
        # Referans yalnizca kendi kare araliginda aranir -> bu karenin frame id'sini
        # prediction.frame_url'den (".../frames/165/") cikarip match'e veriyoruz.
        # -------------------------------------------------------------------
        if self.gorev3 is not None and image is not None:
            try:
                frame_id = None
                try:
                    frame_id = int(str(prediction.frame_url).rstrip("/").split("/")[-1])
                except (ValueError, AttributeError, TypeError):
                    pass
                for u in self.gorev3.match(image, frame_id=frame_id):
                    prediction.add_detected_undefined_object(DetectedUndefinedObject(
                        object_id=u["object_id"],
                        top_left_x=u["top_left_x"], top_left_y=u["top_left_y"],
                        bottom_right_x=u["bottom_right_x"], bottom_right_y=u["bottom_right_y"],
                    ))
            except Exception as e:
                logging.error(f"Gorev 3 match hatasi (bu kare atlandi): {e}")

        # -------------------------------------------------------------------
        # GOREV 2 — POZISYON KESTIRIMI
        # DIKKAT: health_status sunucudan int (1/0) ya da str ('1'/'0')
        # gelebilir; ornek kod str karsilastiriyordu ve int gelirse sessizce
        # yanlis dala girerdi. str()'e cevirerek iki durumu da kapsiyoruz.
        # -------------------------------------------------------------------
        healthy = str(health_status) == '1'
        gt = None
        if healthy:
            gx = _guvenli_float(prediction.gt_translation_x)
            gy = _guvenli_float(prediction.gt_translation_y)
            gz = _guvenli_float(prediction.gt_translation_z)
            if None in (gx, gy, gz):
                # saglikli gorunup NaN gelirse guvenli taraf: kestirimle devam
                healthy = False
            else:
                gt = np.array([gx, gy, gz])

        if image is not None:
            tahmin = self.position_estimator.process_frame(image, gt, healthy)
        else:
            tahmin = None   # goruntu yoksa kestirici ilerletilemez

        if healthy:
            # Sunucudan gelen degerler AYNEN geri gonderilir (0 hata garantisi);
            # kestirici arka planda ogrenmeye devam eder.
            trans_obj = DetectedTranslation(prediction.gt_translation_x,
                                            prediction.gt_translation_y,
                                            prediction.gt_translation_z)
        elif tahmin is not None:
            trans_obj = DetectedTranslation(float(tahmin[0]), float(tahmin[1]), float(tahmin[2]))
        else:
            # en kotu durum yedegi: son kestirilen konumu tut
            son = getattr(self.position_estimator, "_est", None)
            konum = son.pos if son is not None else np.zeros(3)
            trans_obj = DetectedTranslation(float(konum[0]), float(konum[1]), float(konum[2]))
        prediction.add_translation_object(trans_obj)

        # Kare-kare gorunur ozet: log dosyasinda 3 gorevi tek satirda izlemek icin.
        # Gonderilen konum (saglikli=echo / kesinti=TAHMIN) + Gorev 1/3 sayilari.
        _x = _guvenli_float(trans_obj.translation_x)
        _y = _guvenli_float(trans_obj.translation_y)
        _z = _guvenli_float(trans_obj.translation_z)
        _konum = (f"({_x:.1f},{_y:.1f},{_z:.1f}) m"
                  if None not in (_x, _y, _z) else "(?)")
        logging.info(
            f"[Gorev2 {'echo' if healthy else 'TAHMIN'}] konum={_konum} | "
            f"[Gorev1] {len(prediction.detected_objects)} nesne | "
            f"[Gorev3] {len(prediction.detected_undefined_objects)} eslesme"
        )

        return prediction
