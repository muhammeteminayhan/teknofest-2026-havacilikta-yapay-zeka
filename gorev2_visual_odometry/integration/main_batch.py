"""
Yarışma BATCH sürücüsü — Görev 2 (Visual Odometry) için.

Yarışma arayüzü tüm kareleri ve health bitlerini baştan verdiğinden, problem
çevrimdışı bir smoothing problemidir. Bu sürücü `main.py` yerine kullanılır
(arayüz dosyaları DEĞİŞTİRİLMEZ):

  1) Sunucuya bağlan, TÜM frame + translation listesini çek.
  2) TÜM kareleri sırayla indir + işle (pass-1: göreli hareket).
  3) Global çapalı optimizasyon (smoothing) — sağlıksız segmentler iki uçtaki GT
     çapalarıyla düzeltilir.
  4) Optimize pozisyonları 80 kare/dk sınırını gözeterek sunucuya gönder.

Kurulum: `TAKIM_BAGLANTI_ARAYUZU/config/.env` doldurulmuş olmalı (TEAM_NAME,
PASSWORD, EVALUATION_SERVER_URL, SESSION_NAME). Çalıştırma:
    python integration/main_batch.py
(Bu dosyayı TAKIM_BAGLANTI_ARAYUZU/ içine kopyalamak da olur; yollar otomatik bulunur.)
"""
from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# ---- Yol kurulumu: arayüz src + vo paketi ----
_HERE = os.path.dirname(os.path.abspath(__file__))
_VO_ROOT = os.path.dirname(_HERE)                      # gorev2_visual_odometry/
_REPO = os.path.dirname(_VO_ROOT)
_IFACE = os.path.join(_REPO, "TAKIM_BAGLANTI_ARAYUZU")
for p in (_VO_ROOT, os.path.join(_VO_ROOT, "configs"), os.path.join(_IFACE, "src"), _IFACE):
    if p not in sys.path:
        sys.path.insert(0, p)

from decouple import config as env          # noqa: E402
from connection_handler import ConnectionHandler   # noqa: E402 (arayüz, değişmez)
from frame_predictions import FramePredictions      # noqa: E402
from detected_translation import DetectedTranslation  # noqa: E402

from vo.pipeline import build_estimator     # noqa: E402
from vo.batch import make_batch_data        # noqa: E402
from vo.optimize import smooth_trajectory   # noqa: E402

SEND_RATE_PER_MIN = 80                       # send_prediction sınırı
SEND_INTERVAL = 60.0 / SEND_RATE_PER_MIN     # saniye/gönderim (~0.75s)


def configure_logger(team_name):
    log_folder = "./_logs/"
    Path(log_folder).mkdir(parents=True, exist_ok=True)
    fn = datetime.now().strftime(log_folder + team_name + "_batch_%Y_%m_%d__%H_%M_%S.log")
    logging.basicConfig(filename=fn, level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")


def download_image(url, folder, existing):
    """Resmi indir (zaten varsa atla). Dönüş: yerel yol veya None."""
    name = url.split("/")[-1]
    path = os.path.join(folder, name)
    if name in existing or os.path.exists(path):
        return path
    import requests
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            with open(path, "wb") as f:
                f.write(r.content)
            return path
        except requests.exceptions.RequestException as e:
            logging.error(f"İndirme hatası {url} (deneme {attempt+1}): {e}")
            time.sleep(0.2 * (attempt + 1))
    return None


def run():
    print("BATCH VO başlatılıyor...")
    env.search_path = os.path.join(_IFACE, "config")
    team = env("TEAM_NAME")
    password = env("PASSWORD")
    url = env("EVALUATION_SERVER_URL")
    modality = os.environ.get("VO_MODALITY", "rgb")
    configure_logger(team)

    server = ConnectionHandler(url, username=team, password=password)
    frames = server.get_frames(force_download=True)
    translations = server.get_translations(force_download=True)
    images_files, images_folder = server.get_listdir()
    existing = set(images_files)
    n = len(frames)
    print(f"{n} kare alındı. Pass-1 (indir + işle)...")

    # ---- Pass-1: sırayla indir + göreli hareket ----
    estimator = None
    fms, orig_idxs, healths_list, gts_list = [], [], [], []
    t0 = time.perf_counter()
    for j, (frame, tr) in enumerate(zip(frames, translations)):
        img_url = url + "media" + frame["image_url"]
        path = download_image(img_url, images_folder, existing)
        image = cv2.imread(path) if path else None
        if image is None:
            logging.error(f"Kare {j} okunamadı; sıfır-hareket varsayılıyor.")
            from vo.motion import FrameMotion
            fms.append(FrameMotion.zero())
        else:
            if estimator is None:
                h, w = image.shape[:2]
                if (w, h) == (640, 512):
                    modality = "thermal"
                estimator = build_estimator(modality, w, h)
                print(f"VOEstimator: modality={modality}, giriş={w}x{h}, "
                      f"işleme={estimator.proc_w}x{estimator.proc_h}")
            fms.append(estimator.compute_frame_motion(image))
        orig_idxs.append(j)
        healthy = str(tr.get("health_status", tr.get("gps_health_status", "1"))).strip() == "1"
        healths_list.append(healthy)
        gts_list.append((float(tr["translation_x"]), float(tr["translation_y"]),
                         float(tr["translation_z"])))
        if (j + 1) % 100 == 0:
            print(f"  pass-1 {j+1}/{n} ({(j+1)/(time.perf_counter()-t0):.1f}/s)", flush=True)

    # ---- Global optimizasyon ----
    print("Global optimizasyon (smoothing)...")
    batch = make_batch_data(fms, orig_idxs,
                            health_fn=lambda jj: healths_list[jj],
                            gt_fn=lambda jj, oi: gts_list[jj])
    preds = smooth_trajectory(batch, align_mode="interp")
    print(f"Optimizasyon tamam. {len(preds)} pozisyon hazır. Gönderiliyor (≤{SEND_RATE_PER_MIN}/dk)...")

    # ---- Gönderim (rate-limited) ----
    base_url = server.base_url
    last_send = 0.0
    sent = 0
    for j, frame in enumerate(frames):
        prediction = FramePredictions(
            frame["url"], frame["image_url"], frame["video_name"],
            gts_list[j][0], gts_list[j][1], gts_list[j][2])
        px, py, pz = preds[j]
        prediction.add_translation_object(DetectedTranslation(float(px), float(py), float(pz)))
        # 80/dk sınırı: gerekli kadar bekle
        dt = time.perf_counter() - last_send
        if dt < SEND_INTERVAL:
            time.sleep(SEND_INTERVAL - dt)
        last_send = time.perf_counter()
        resp = server.send_prediction(prediction)
        sent += 1
        if (j + 1) % 100 == 0:
            print(f"  gönderildi {j+1}/{n}", flush=True)
    print(f"Bitti. {sent} tahmin gönderildi.")


if __name__ == "__main__":
    run()
