"""
Görev 1 modeli — yarışma entegrasyonu.

Tespit (YOLO+SAHI) → hareket durumu (ego-motion) → iniş durumu (UAP/UAİ) →
yarışma sınıf id'leri → arayüzün `DetectedObject` payload'u.

Tasarım: çekirdek mantık arayüzden bağımsız `G1Sonuc` ara biçimini üretir; ayrı bir
dönüştürücü bunu `TAKIM_BAGLANTI_ARAYUZU/src/detected_object.py` ile uyumlu nesnelere
çevirir. Arayüz formatı taslak (motion_status alanı eklenince yalnızca dönüştürücü
güncellenir; çekirdek değişmez). Arayüz dosyaları DEĞİŞTİRİLMEZ.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from gorev1_nesne_tespiti.configs import config as C
from gorev1_nesne_tespiti.modeller import Detector, Detection
from gorev1_nesne_tespiti.hareket import HareketSiniflandirici
from gorev1_nesne_tespiti.inis import inis_durumu_belirle
from gorev1_nesne_tespiti.veri.siniflar import (
    SINIF_ID, coco_eslemesi, HAREKET_DURUMU, INIS_DURUMU,
)


@dataclass
class G1Sonuc:
    """Yarışma-uyumlu tek nesne sonucu (arayüzden bağımsız)."""
    cls_id: int            # yarışma sınıf id (0=Taşıt,1=İnsan,2=UAP,3=UAİ)
    motion_status: int     # 0/1 (yalnızca Taşıt), diğer -1
    landing_status: int    # 0/1 (UAP/UAİ), diğer -1
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float


class Gorev1Modeli:
    """
    Kare-kare çalışır; her kare için G1Sonuc listesi üretir.

    model_classes : "coco" (pretrained baseline; COCO→yarışma eşlemesi) ya da
                    "yarisma" (fine-tuned model; sınıf id'leri zaten 0..3).
    """

    def __init__(self, model_path: Optional[str] = None, model_classes: str = "coco",
                 use_sahi: Optional[bool] = None, arch: Optional[str] = None):
        self.model_classes = model_classes
        arch = arch or C.DETECTOR_ARCH
        if model_path is None:
            model_path = C.RTDETR_MODEL if arch == "rtdetr" else C.YOLO_MODEL
        self.detector = Detector(
            model_path=model_path, arch=arch,
            use_sahi=C.USE_SAHI if use_sahi is None else use_sahi,
            device=C.DEVICE, conf=C.CONF_THRESH, iou=C.IOU_THRESH, half=C.HALF,
            slice_size=C.SAHI_SLICE, overlap=C.SAHI_OVERLAP,
        )
        self.hareket = HareketSiniflandirici(esik_px=C.HAREKET_ESIK_PX,
                                             pencere=C.HAREKET_PENCERE)

    def _yarisma_cls(self, model_cls_id: int) -> Optional[int]:
        """Model sınıf id → yarışma sınıf id (0..3) veya None (atla)."""
        if self.model_classes == "yarisma":
            return int(model_cls_id) if 0 <= int(model_cls_id) <= 3 else None
        ad = coco_eslemesi(model_cls_id)          # 'Tasit'/'Insan'/None
        return SINIF_ID.get(ad) if ad is not None else None

    def isle(self, frame: np.ndarray) -> List[G1Sonuc]:
        h, w = frame.shape[:2]
        dets: List[Detection] = self.detector.detect(frame)

        # Yarışma sınıfına indir + grupla.
        kalan: List[Detection] = []
        yarisma_cls: List[int] = []
        for d in dets:
            yc = self._yarisma_cls(d.cls_id)
            if yc is None:
                continue
            kalan.append(d)
            yarisma_cls.append(yc)

        tasitlar = [d for d, yc in zip(kalan, yarisma_cls) if yc == SINIF_ID["Tasit"]]
        hareket_durumlari = self.hareket.guncelle(frame, tasitlar)
        tasit_hareket = {id(d): s for d, s in zip(tasitlar, hareket_durumlari)}

        # İniş alanları için engel kutuları: taşıt + insan.
        engeller = [d.xyxy for d, yc in zip(kalan, yarisma_cls)
                    if yc in (SINIF_ID["Tasit"], SINIF_ID["Insan"])]

        sonuc: List[G1Sonuc] = []
        for d, yc in zip(kalan, yarisma_cls):
            if yc == SINIF_ID["Tasit"]:
                ms = tasit_hareket.get(id(d), 0)
                ls = INIS_DURUMU["InisAlaniDegil"]
            elif yc == SINIF_ID["Insan"]:
                ms = HAREKET_DURUMU["Yok"]
                ls = INIS_DURUMU["InisAlaniDegil"]
            else:  # UAP / UAİ
                ms = HAREKET_DURUMU["Yok"]
                ls = inis_durumu_belirle(d.xyxy, engeller, w, h, C.INIS_DOLULUK_MARJI)
            sonuc.append(G1Sonuc(yc, ms, ls, d.x1, d.y1, d.x2, d.y2, d.conf))
        return sonuc


def to_detected_objects(sonuclar: List[G1Sonuc], server_url: str) -> list:
    """
    G1Sonuc listesi → arayüzün DetectedObject listesi (payload üretimine hazır).

    Not: Mevcut arayüz DetectedObject'i cls'i indekslenebilir (tuple) bekler ve
    create_payload'da cls[0]+1 kullanır. motion_status arayüz taslağında henüz yok;
    forward-compat için nesneye attribute olarak eklenir (gerçek format gelince
    payload'a dahil edilecek). Arayüz dosyası DEĞİŞTİRİLMEZ.
    """
    import sys, os
    _IFACE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "TAKIM_BAGLANTI_ARAYUZU", "src")
    if _IFACE not in sys.path:
        sys.path.insert(0, _IFACE)
    from detected_object import DetectedObject

    nesneler = []
    for s in sonuclar:
        d = DetectedObject(
            cls=(s.cls_id,),                 # cls[0] → sınıf id (arayüz beklentisi)
            landing_status=s.landing_status,
            top_left_x=s.x1, top_left_y=s.y1,
            bottom_right_x=s.x2, bottom_right_y=s.y2,
        )
        d.motion_status = str(s.motion_status)   # forward-compat (taslak formata hazır)
        nesneler.append(d)
    return nesneler
