"""
Görev 1 — merkezi konfigürasyon (yollar, model, SAHI, eşikler).

Tüm yollar repo köküne görelidir; kullanıcı yalnızca dış veri setlerini `datasets/`
altına indirir (scriptlerle). Yarışmada internet olmadığından model ağırlıkları
`gorev1_nesne_tespiti/agirliklar/` altında çevrimdışı saklanır.
"""
from __future__ import annotations

import os

# --------------------------------------------------------------------------
# Yollar
# --------------------------------------------------------------------------
PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # gorev1_nesne_tespiti/
REPO_ROOT = os.path.dirname(PKG_ROOT)                                   # havacilikta-yapay-zeka/

DATASETS_DIR = os.path.join(REPO_ROOT, "datasets")          # dış veri (gitignore)
WEIGHTS_DIR = os.path.join(PKG_ROOT, "agirliklar")          # model ağırlıkları (gitignore)
OUTPUTS_DIR = os.path.join(PKG_ROOT, "outputs")             # görseller/sonuçlar

# Örnek veri (Görev 2 ile aynı veri seti) — baseline duman testi için.
ORNEK_VERI_DIR = os.path.join(REPO_ROOT, "THYZ_2026_Ornek_Veri_Seti")
ORNEK_RGB_VIDEO = os.path.join(ORNEK_VERI_DIR, "THYZ_2026_Ornek_Veri_1.MP4")
ORNEK_TERMAL_VIDEO = os.path.join(ORNEK_VERI_DIR, "THYZ_2026_Ornek_Veri_2_Termal.MP4")

# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
# Baseline: COCO-pretrained YOLOv11 (sanity). Fine-tune sonrası kendi ağırlığımız.
# 8GB VRAM: 'n'/'s' güvenli; 'm' fp16 ile sığar. Blackwell (sm_120) → torch cu128.
YOLO_MODEL = "yolo11n.pt"                  # baseline pretrained (ultralytics indirir)
YOLO_FINETUNED = os.path.join(WEIGHTS_DIR, "g1_finetuned.pt")  # eğitim çıktısı (sonra)
DEVICE = "cuda:0"                          # yoksa otomatik cpu'ya düşülür (yolo_sahi.py)
HALF = True                                # fp16 çıkarım (hız + VRAM)

# Çıkarım eşikleri
CONF_THRESH = 0.25
IOU_THRESH = 0.7                           # NMS

# --------------------------------------------------------------------------
# SAHI (Slicing Aided Hyper Inference) — küçük nesne için dilimleme
# --------------------------------------------------------------------------
USE_SAHI = True
SAHI_SLICE = 640                           # dilim boyutu (px)
SAHI_OVERLAP = 0.25                        # dilim örtüşme oranı
# 4K kareler için dilimleme şart; FHD'de de küçük nesnede kazanç sağlar.

# --------------------------------------------------------------------------
# Hareketli/sabit (ego-motion artık hareketi)
# --------------------------------------------------------------------------
# Ardışık karede ego-hareket telafisinden sonra bir taşıtın merkezi bu eşikten
# fazla kaydıysa "hareketli" sayılır (piksel). Çözünürlük/irtifaya göre ayarlanır.
HAREKET_ESIK_PX = 6.0
# Kaç ardışık karede tutarlı hareket aranacak (gürültü azaltma).
HAREKET_PENCERE = 3

# --------------------------------------------------------------------------
# UAP/UAİ iniş alanı
# --------------------------------------------------------------------------
# Daire içine/yakınına düşen herhangi bir nesne → iniş uygun DEĞİL.
INIS_DOLULUK_MARJI = 1.15                  # daire yarıçapının çarpanı (yakınlık marjı)
