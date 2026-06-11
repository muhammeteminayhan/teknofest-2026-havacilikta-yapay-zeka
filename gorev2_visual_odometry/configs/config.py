"""
Merkezi konfigürasyon — yollar, kamera kalibrasyon dosyası ve VO parametreleri.

Tüm modüller ve scriptler ayarları buradan okur. Yarışma koşulları
(şartname) ve örnek veri seti yolları tek yerde toplanmıştır.
"""
from __future__ import annotations

import os

# --------------------------------------------------------------------------
# Yollar
# --------------------------------------------------------------------------
# Bu dosya: gorev2_visual_odometry/configs/config.py
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(PROJECT_ROOT)  # havacilikta-yapay-zeka/

DATASET_DIR = os.path.join(REPO_ROOT, "THYZ_2026_Ornek_Veri_Seti")
CALIB_FILE = os.path.join(DATASET_DIR, "Kamera_Kalibrasyon_Parametreleri_2026.txt")

# Örnek veri: (video, ground-truth translation csv, modalite)
RGB_VIDEO = os.path.join(DATASET_DIR, "THYZ_2026_Ornek_Veri_1.MP4")
RGB_CSV = os.path.join(DATASET_DIR, "THYZ_2026_Ornek_Veri_1_translation.csv")
THERMAL_VIDEO = os.path.join(DATASET_DIR, "THYZ_2026_Ornek_Veri_2_Termal.MP4")
THERMAL_CSV = os.path.join(DATASET_DIR, "THYZ_2026_Ornek_Veri_2_Termal_translation.csv")

# Çıktı klasörleri (kareler, grafikler, ara dosyalar)
FRAMES_DIR = os.path.join(PROJECT_ROOT, "_frames")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

# --------------------------------------------------------------------------
# Yarışma senaryosu (teknik şartname)
# --------------------------------------------------------------------------
# Oturum başına 2250 kare (5 dk @ 7.5 fps). İlk 450 kare (1 dk) garanti sağlıklı.
SESSION_FPS = 7.5
HEALTHY_PREFIX_FRAMES = 450          # ilk 1 dk health=1 garanti
SESSION_TOTAL_FRAMES = 2250          # tipik oturum uzunluğu

# --------------------------------------------------------------------------
# VO parametreleri (Faz 1 varsayılanları — Faz 2'de ayarlanacak)
# --------------------------------------------------------------------------
# Ön-işleme
USE_CLAHE = False                    # termalde (Faz 3) True yapılacak
CLAHE_CLIP = 2.0
CLAHE_GRID = 8
# İşleme çözünürlüğü: büyük 4K kareleri hız/sağlamlık için küçültebiliriz.
# Kalibrasyon buna göre ölçeklenir. None => orijinal çözünürlük.
PROC_MAX_WIDTH = 1280                # işlemede kullanılacak azami genişlik (px)

# Özellik ön-yüzü
FEATURE_BACKEND = "klt"              # "klt" | "orb" | "sift"
MAX_FEATURES = 1500                  # takip/eşleştirme için azami nokta
KLT_MIN_TRACKS = 200                 # bu sayının altına düşünce yeniden tohumla
GFTT_QUALITY = 0.01
GFTT_MIN_DISTANCE = 12

# Hareket kestirimi (motion.py)
MOTION_MODEL = "hybrid"              # "hybrid" (öteleme+Z benzerlik, yaw homografi) | "homography" | "similarity"
RANSAC_REPROJ_THRESH = 2.0           # piksel (homografi/essential RANSAC eşiği)
MIN_INLIERS = 30                     # bu sayının altında kare "düşük güven"
HOMOGRAPHY_INLIER_RATIO = 0.6        # bu oranın altında essential'a düş

# Ölçek/hizalama (aligner.py, trajectory.py)
ALIGN_MIN_POINTS = 30                # Umeyama için gereken asgari sağlıklı kare
ALIGN_WINDOW = 450                   # kayan-pencere hizalama boyu (kare)

# İrtifa/ölçek takibi (trajectory.py) — kare-başı metrik ölçek
# Sağlıklı segmentte XY 2B-Umeyama ölçeği ≈ irtifa olarak öğrenilir.
# Sağlıksız segmentte irtifa, sahne ölçek-değişimi proxy'si ile yayılır:
#     a_i = a_{i-1} * exp(ALT_PROP_GAIN * dz_smooth)
# Not (tarama bulgusu): proxy yayılımı (α>0) 1806 karelik kesintisiz ölü-hesapta
# gürültü biriktirdiğinden donmuş ölçek (α=0) + sık yeniden-çapalama daha gürbüz.
ALT_PROP_GAIN = 0.0                  # irtifa yayılım kazancı α (0=dondur, 1=tam proxy)
ALT_SMOOTH = 0.8                     # dz proxy EMA katsayısı β (gürültü azaltma)
ALT_CLAMP_LO = 0.3                   # irtifa alt sınırı (referans ölçeğin çarpanı)
ALT_CLAMP_HI = 3.0                   # irtifa üst sınırı
# z-sign=-1: irtifa artışı (sahne küçülmesi) GT'de z AZALMASINA karşılık gelir (NED benzeri)
Z_SIGN = -1.0                        # irtifa→dünya-z işareti (tarama ile -1 seçildi)
Z_GAIN = 1.0                         # Z entegratör kazancı (irtifa proxy → metrik z)

# --------------------------------------------------------------------------
# Kalibrasyon kuralı
# --------------------------------------------------------------------------
# MATLAB cameraIntrinsics 1-tabanlı piksel kullanır; OpenCV 0-tabanlı.
# Principal point'e -1 ofset uygula (alt-piksel doğruluk; VO'da ihmal edilebilir).
MATLAB_TO_OPENCV_PP_OFFSET = True
