import importlib.util
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import requests

from .constants import classes, landing_statuses, moving_statuses
from .detected_object import DetectedObject
from .detected_translation import DetectedTranslation
from .reference_prediction import ReferencePrediction

# ---------------------------------------------------------------------------
# Gorev 2 (pozisyon kestirimi) boru hattini ice aktar: gorev2, bu deponun
# kardes klasorudur (havacilikta-yz-project/gorev2). Bu deponun kendi 'src'
# paketi ile isim cakismasini onlemek icin gorev2/src ayri bir paket adiyla
# (gorev2_src) yuklenir.
# ---------------------------------------------------------------------------
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
    """Takim modeli: uc gorevi (nesne tespiti / pozisyon / referans esleme) tek
    `detect()` icinde sirayla calistirir. Baglanti katmani (main.py,
    connection_handler.py, frame_predictions.py ...) resmi arayuzden BIREBIR
    alinmistir; yarismacinin duzenledigi tek dosya sartname geregi budur."""

    def __init__(self, evaluation_server_url):
        logging.info('Created Object Detection Model')
        self.evaulation_server = evaluation_server_url
        # Gercek sunucuda /media ucu KIMLIK DOGRULAMA ister (tokensiz 401).
        # main.py, server olusturduktan sonra bunu doldurur (ve process'e de gecer).
        self.auth_token = None

        # Yarisma gunu kontrol listesi: GPU / RAFT agirliklari / motor secimi
        kontrol = pre_flight_check()
        logging.info(f"On kontrol: {kontrol}")
        print(f"On kontrol: {kontrol}")

        # Pozisyon kestirici (Gorev 2): motoru ilk karede (cozunurluk belli olunca)
        # kurar. RAFT (GPU) kurulamazsa kendiliginden LK yedegine duser ve loglar.
        self.position_estimator = LivePositionEstimator(engine="auto")

        # Rolling sunucu ayni kareyi (tahmin ilerlemezse) tekrar verebilir. Gorev 2
        # durum-makinesi (VO) o kareyi IKI KEZ ilerletmemeli — son frame_url ve son
        # gonderilen konum tutulur; tekrar gelen karede VO ilerletilmeden ayni konum
        # gonderilir.
        self._last_frame_url = None
        self._last_trans = None  # (x, y, z) float

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
        # register_references(...) ile kaydedilir; kayit yoksa/aktif referans yoksa
        # bir sey uretmez. Yarisma ayari: rembg KAPALI (precision), tiling acik.
        self.gorev3 = None
        try:
            from gorev3 import Gorev3Matcher
            self.gorev3 = Gorev3Matcher(kind="dinov3", remove_bg=False)
            logging.info("Gorev 3 eslestirici yuklendi (rembg kapali, tiling acik).")
            print("Gorev 3 eslestirici yuklendi.")
        except Exception as e:
            logging.error(f"Gorev 3 eslestirici YUKLENEMEDI (Gorev 1/2 devam eder): {e}")
            print(f"UYARI: Gorev 3 eslestirici yuklenemedi: {e}")

    def register_references(self, references, ref_image_paths=None):
        """Gorev 3 referanslarini kaydet (main.py oturum basinda cagirir).
        `ref_image_paths`: {ref_url -> yerel_gorsel_yolu} (main onceden indirir)."""
        if self.gorev3 is None:
            return 0
        return self.gorev3.register_from_server(references, ref_image_paths=ref_image_paths)

    @staticmethod
    def download_image(img_url, images_folder, images_files, retries=3, initial_wait_time=0.1, auth_token=None):
        t1 = time.perf_counter()
        wait_time = initial_wait_time
        # Indirmek istedigimiz frame daha once indirilmis mi kontrol edelim
        image_name = img_url.split("/")[-1]
        if image_name not in images_files:
            # Gercek sunucu /media ucu icin Authorization sart (yerel simulatorde zararsiz).
            headers = {'Authorization': f'Token {auth_token}'} if auth_token else {}
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

    def process(self, prediction, evaluation_server_url, health_status, images_folder, images_files,
                active_refs=None, ref_image_paths=None, auth_token=None):
        # Kareyi indir ve oku (gercek sunucu icin auth token ile)
        token = auth_token or self.auth_token
        self.download_image(evaluation_server_url + "media" + prediction.image_url,
                            images_folder, images_files, auth_token=token)
        image_name = prediction.image_url.split("/")[-1]
        image = cv2.imread(str(Path(images_folder) / image_name))
        if image is None:
            logging.error(f"Goruntu okunamadi: {image_name}")
        return self.detect(prediction, health_status, image,
                           active_refs=active_refs or [], ref_image_paths=ref_image_paths or {})

    def detect(self, prediction, health_status, image, active_refs=None, ref_image_paths=None):
        active_refs = active_refs or []

        # -------------------------------------------------------------------
        # GOREV 1 — NESNE TESPITI
        # Gorev1Detector cikti sozluklerini DetectedObject'e cevirir. Resmi
        # DetectedObject: cls TUPLE olmali (payload'da cls[0] kullanilir).
        # Moving status yalnizca Tasit sinifinda anlamli; tasit-disi (UAP/UAI/
        # Insan) icin "-1" ("Tasit Degil") gonderilir (resmi not).
        # Bir karede hata olursa loglanir, oturum devam eder.
        # -------------------------------------------------------------------
        if self.gorev1 is not None and image is not None:
            try:
                for o in self.gorev1.detect(image):
                    cls_id = int(o["cls"])
                    if cls_id == classes["Tasit"]:
                        moving = str(o.get("motion_status", moving_statuses["Sabit"]))
                    else:
                        moving = moving_statuses["Tasit Degil"]  # "-1"
                    prediction.add_detected_object(DetectedObject(
                        (cls_id,),
                        o["landing_status"],
                        moving,
                        o["top_left_x"], o["top_left_y"],
                        o["bottom_right_x"], o["bottom_right_y"],
                    ))
            except Exception as e:
                logging.error(f"Gorev 1 detect hatasi (bu kare atlandi): {e}")

        # -------------------------------------------------------------------
        # GOREV 3 — REFERANS NESNE TESPITI (yalnizca aktif referans varken)
        # active_refs: main.py'nin bu kare icin pencere-ici sectigi referanslar.
        # Bos ise (kare hicbir referans araliginda degil) Gorev 3 hic calismaz.
        # Cikti resmi ReferencePrediction: referansin KENDI url'i + kare url'i + bbox.
        # -------------------------------------------------------------------
        if self.gorev3 is not None and image is not None and active_refs:
            try:
                frame_id = None
                try:
                    frame_id = int(str(prediction.frame_url).rstrip("/").split("/")[-1])
                except (ValueError, AttributeError, TypeError):
                    pass
                # Aktif referanslarin order'lari — matcher SADECE bunlari arasin
                # (pencere kapisi burada active_refs ile kesin belirlenir).
                active_ids = set()
                for ref in active_refs:
                    try:
                        active_ids.add(int(ref.get("order")))
                    except (TypeError, ValueError):
                        pass
                eslesmeler = {int(d["object_id"]): d
                              for d in self.gorev3.match(image, frame_id=frame_id,
                                                         active_ids=active_ids)}
                for ref in active_refs:
                    # Resmi pencere kontrolu: kare image_url'i [start, end] arasinda mi
                    start_img = ref.get("frame_start_image_url", "")
                    end_img = ref.get("frame_end_image_url", "")
                    if not (start_img and end_img and start_img <= prediction.image_url <= end_img):
                        continue
                    try:
                        oid = int(ref.get("order"))
                    except (TypeError, ValueError):
                        continue
                    d = eslesmeler.get(oid)
                    if d is None:
                        continue
                    prediction.add_reference_prediction(ReferencePrediction(
                        ref["url"], prediction.frame_url,
                        d["top_left_x"], d["top_left_y"],
                        d["bottom_right_x"], d["bottom_right_y"],
                    ))
            except Exception as e:
                logging.error(f"Gorev 3 match hatasi (bu kare atlandi): {e}")

        # -------------------------------------------------------------------
        # GOREV 2 — POZISYON KESTIRIMI
        # health_status: '1' saglikli (GT echo) / '0' kesinti (kestirim) /
        # None ceviri yok (Gorev 2 ciktisi uretme). Rolling sunucu ayni kareyi
        # tekrar verirse VO'yu ikinci kez ilerletmeyiz (_last_frame_url guard).
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

        same_frame = (prediction.frame_url is not None
                      and prediction.frame_url == self._last_frame_url)
        trans_obj = None

        if health_status is None:
            # Ceviri yok — Gorev 2 ciktisi uretmeyiz (kare yine de Gorev 1/3 ile gider).
            logging.info("Ceviri/health_status yok; Gorev 2 ciktisi uretilmiyor.")
        elif same_frame and self._last_trans is not None:
            # Ayni kare tekrar geldi (tahmin ilerlemedi) — VO'yu ilerletmeden tekrar gonder.
            trans_obj = DetectedTranslation(*self._last_trans)
        else:
            tahmin = (self.position_estimator.process_frame(image, gt, healthy)
                      if image is not None else None)
            if healthy:
                # Sunucudan gelen GT AYNEN geri gonderilir (0 hata garantisi).
                trans_obj = DetectedTranslation(prediction.gt_translation_x,
                                                prediction.gt_translation_y,
                                                prediction.gt_translation_z)
                self._last_trans = (float(gt[0]), float(gt[1]), float(gt[2]))
            elif tahmin is not None:
                trans_obj = DetectedTranslation(float(tahmin[0]), float(tahmin[1]), float(tahmin[2]))
                self._last_trans = (float(tahmin[0]), float(tahmin[1]), float(tahmin[2]))
            else:
                # en kotu durum yedegi: son kestirilen konum (goruntu yoksa)
                son = getattr(self.position_estimator, "_est", None)
                konum = getattr(son, "pos", None)
                konum = konum if konum is not None else np.zeros(3)
                trans_obj = DetectedTranslation(float(konum[0]), float(konum[1]), float(konum[2]))
                self._last_trans = (float(konum[0]), float(konum[1]), float(konum[2]))

        if trans_obj is not None:
            prediction.add_translation_object(trans_obj)
        self._last_frame_url = prediction.frame_url

        # Kare-kare gorunur ozet: log dosyasinda 3 gorevi tek satirda izlemek icin.
        durum = "yok" if health_status is None else ("echo" if healthy else "TAHMIN")
        konum_str = (f"({self._last_trans[0]:.1f},{self._last_trans[1]:.1f},{self._last_trans[2]:.1f}) m"
                     if self._last_trans is not None else "(-)")
        logging.info(
            f"[Gorev2 {durum}] konum={konum_str} | "
            f"[Gorev1] {len(prediction.detected_objects)} nesne | "
            f"[Gorev3] {len(prediction.reference_predictions)} eslesme"
        )

        return prediction
