"""
TAKIM_BAGLANTI_ARAYUZU için DROP-IN `object_detection_model.py`.

Bu dosya, yarışma bağlantı arayüzündeki `src/object_detection_model.py` dosyasının
yerine konulacak sürümdür. Yalnızca İKİNCİ GÖREV (pozisyon kestirimi / visual
odometry) entegre edilmiştir; arayüzün diğer dosyaları (connection_handler,
frame_predictions, detected_translation, constants) DEĞİŞTİRİLMEDEN kullanılır.

Kurulum:
  1) Bu dosyayı `TAKIM_BAGLANTI_ARAYUZU/src/object_detection_model.py` olarak kopyalayın
     (orijinalini yedekleyin).
  2) `gorev2_visual_odometry` paketinin yolunu ayarlayın: aşağıdaki VO_PACKAGE_PATH
     otomatik bulamazsa `VO_PACKAGE_PATH` ortam değişkenini ayarlayın.
  3) Oturum termal ise `VO_MODALITY=thermal` ortam değişkenini ayarlayın (veya
     otomatik tespit: 640x512 çözünürlük → termal).

Sağlık (health) mantığı (arayüz koduyla uyumlu):
  health_status == '1' → GT pozisyon mevcut; echo edilir VE estimator'a beslenir
                         (yeniden çapalama + ölçek/eksen hizalama).
  health_status == '0' → VOEstimator'ın kendi kestirimi gönderilir.

NOT: Birinci görev (nesne tespiti) bu dosyada KAPSAM DIŞIDIR; takım kendi tespit
modelini `detect()` içindeki ilgili bölüme ekleyebilir. Şimdilik nesne tespiti
boş bırakılmıştır (yalnızca pozisyon kestirimine odaklanılmıştır).
"""
import logging
import os
import sys
import time

import requests

# Arayüz veri sınıfları: paket içindeyken (drop-in) göreli, standalone testte mutlak import
try:
    from .constants import classes, landing_statuses
    from .detected_object import DetectedObject
    from .detected_translation import DetectedTranslation
except ImportError:
    from constants import classes, landing_statuses
    from detected_object import DetectedObject
    from detected_translation import DetectedTranslation


# --------------------------------------------------------------------------
# VO paketini bul ve içe aktar
# --------------------------------------------------------------------------
def _locate_vo_package():
    # 1) Ortam değişkeni
    env = os.environ.get("VO_PACKAGE_PATH")
    cands = []
    if env:
        cands.append(env)
    # 2) Repo yapısına göre olası göreli yollar (this: <repo>/TAKIM_BAGLANTI_ARAYUZU/src/...)
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(here))          # <repo>/
    cands.append(os.path.join(repo_root, "gorev2_visual_odometry"))
    cands.append(os.path.join(os.path.dirname(repo_root), "gorev2_visual_odometry"))
    for c in cands:
        if c and os.path.isdir(os.path.join(c, "vo")):
            return c
    return None


_VO_PATH = _locate_vo_package()
if _VO_PATH and _VO_PATH not in sys.path:
    sys.path.insert(0, _VO_PATH)
    sys.path.insert(0, os.path.join(_VO_PATH, "configs"))

import cv2  # noqa: E402
from vo.pipeline import build_estimator  # noqa: E402


class ObjectDetectionModel:
    # Base class for team models

    def __init__(self, evaluation_server_url):
        logging.info('Created Object Detection Model (Visual Odometry / Görev 2)')
        self.evaulation_server = evaluation_server_url
        # VOEstimator ilk karede (çözünürlük bilinince) tembel kurulur
        self.vo = None
        self.modality = os.environ.get("VO_MODALITY", "rgb")
        self._frame_no = 0

    # ---------------------------------------------------------------- yardımcı
    @staticmethod
    def download_image(img_url, images_folder, images_files, retries=3, initial_wait_time=0.1):
        t1 = time.perf_counter()
        wait_time = initial_wait_time
        image_name = img_url.split("/")[-1]
        if image_name not in images_files:
            for attempt in range(retries):
                try:
                    response = requests.get(img_url, timeout=60)
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

    def _ensure_vo(self, image):
        if self.vo is not None:
            return
        h, w = image.shape[:2]
        # Otomatik modalite tespiti: termal kameranın çözünürlüğü 640x512
        modality = self.modality
        if (w, h) == (640, 512):
            modality = "thermal"
        self.vo = build_estimator(modality, w, h)
        logging.info(f"VOEstimator kuruldu: modality={modality}, giriş={w}x{h}, "
                     f"işleme={self.vo.proc_w}x{self.vo.proc_h}")

    # ---------------------------------------------------------------- ana akış
    def process(self, prediction, evaluation_server_url, health_status, images_folder, images_files):
        # Resmi indir
        self.download_image(evaluation_server_url + "media" + prediction.image_url,
                            images_folder, images_files)
        # İndirilen resmi yükle
        image_name = prediction.image_url.split("/")[-1]
        image_path = os.path.join(images_folder, image_name)
        image = cv2.imread(image_path)
        if image is None:
            logging.error(f"Resim okunamadı: {image_path} — bu kare için tahmin atlanıyor.")
        frame_results = self.detect(prediction, health_status, image)
        return frame_results

    def detect(self, prediction, health_status, image=None):
        # ----- BİRİNCİ GÖREV (nesne tespiti): KAPSAM DIŞI -----
        # Takım kendi tespit modelini buraya ekleyebilir. Şimdilik nesne eklenmez.
        # (İstenirse DetectedObject ile prediction.add_detected_object(...) çağrılabilir.)

        # ----- İKİNCİ GÖREV (pozisyon kestirimi / VO) -----
        gt_xyz = (prediction.gt_translation_x, prediction.gt_translation_y,
                  prediction.gt_translation_z)
        if image is not None:
            self._ensure_vo(image)
            out = self.vo.process(image, health_status, gt_xyz=gt_xyz)
            px, py, pz = float(out[0]), float(out[1]), float(out[2])
        else:
            # Resim yoksa: sağlıklıysa GT echo, değilse son güvenli kestirim/0
            if str(health_status).strip() == '1':
                px, py, pz = (float(gt_xyz[0]), float(gt_xyz[1]), float(gt_xyz[2]))
            elif self.vo is not None:
                last = self.vo.traj.predict()
                px, py, pz = float(last[0]), float(last[1]), float(last[2])
            else:
                px, py, pz = 0.0, 0.0, 0.0

        trans_obj = DetectedTranslation(px, py, pz)
        prediction.add_translation_object(trans_obj)
        self._frame_no += 1
        return prediction
