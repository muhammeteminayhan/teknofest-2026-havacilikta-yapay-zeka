"""
Yarışma sınıfları ve dış veri seti → yarışma sınıf eşlemeleri.

Yarışma sınıfları (şartname Tablo 2 & 4; arayüz constants.py ile birebir):
    Taşıt=0, İnsan=1, UAP=2, UAİ=3
Ek nitelikler:
    - Taşıt için hareket_durumu: 0=Hareketsiz, 1=Hareketli (diğer sınıflarda -1)
    - UAP/UAİ için inis_durumu: 0=Uygun Değil, 1=Uygun, -1=İniş Alanı Değil
"""
from __future__ import annotations

# Arayüzdeki TAKIM_BAGLANTI_ARAYUZU/src/constants.py ile aynı (DEĞİŞTİRİLMEZ kaynak).
SINIF_ID = {
    "Tasit": 0,
    "Insan": 1,
    "UAP": 2,
    "UAI": 3,
}

YARISMA_SINIFLARI = {v: k for k, v in SINIF_ID.items()}

HAREKET_DURUMU = {
    "Hareketsiz": 0,
    "Hareketli": 1,
    "Yok": -1,          # taşıt dışı sınıflar
}

INIS_DURUMU = {
    "UygunDegil": 0,
    "Uygun": 1,
    "InisAlaniDegil": -1,   # taşıt/insan
}


# --- Dış veri seti sınıf eşlemeleri -------------------------------------------
# Dış setleri yarışmanın 4 sınıfına indirgeriz. UAP/UAİ dış sette yok → yalnızca
# örnek veri + sentetik ile öğrenilecek (None = atla/eşleşme yok).

# VisDrone sınıfları (0..9): ignored, pedestrian, people, bicycle, car, van,
# truck, tricycle, awning-tricycle, bus, motor
_VISDRONE = {
    0: None,          # ignored regions
    1: "Insan",       # pedestrian
    2: "Insan",       # people
    3: "Tasit",       # bicycle (sürücüsüyle bütün → taşıt; şartname: bisiklet sürücüsü insan değil)
    4: "Tasit",       # car
    5: "Tasit",       # van
    6: "Tasit",       # truck
    7: "Tasit",       # tricycle
    8: "Tasit",       # awning-tricycle
    9: "Tasit",       # bus
    10: "Tasit",      # motor (motosiklet+sürücü → taşıt)
}

# COCO (yalnızca baseline sanity için): person=0, bicycle=1, car=2, motorcycle=3,
# bus=5, truck=7, boat=8, train=6
_COCO = {
    0: "Insan",
    1: "Tasit",
    2: "Tasit",
    3: "Tasit",
    5: "Tasit",
    6: "Tasit",
    7: "Tasit",
    8: "Tasit",       # boat / deniz taşıtı
}


def visdrone_eslemesi(visdrone_cls: int):
    """VisDrone sınıf id → yarışma sınıf adı ('Tasit'/'Insan') veya None (atla)."""
    return _VISDRONE.get(int(visdrone_cls))


def coco_eslemesi(coco_cls: int):
    """COCO sınıf id → yarışma sınıf adı veya None. Yalnızca pretrained baseline içindir."""
    return _COCO.get(int(coco_cls))
