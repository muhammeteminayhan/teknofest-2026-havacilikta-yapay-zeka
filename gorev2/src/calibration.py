"""Kamera kalibrasyon profilleri ve distorsiyon düzeltme yardımcıları.

Değerler ``THYZ_2026_Ornek_Veri_Seti/Kamera_Kalibrasyon_Parametreleri_2026.txt``
dosyasından alınmıştır (MATLAB cameraIntrinsics formatı; MATLAB'da
PrincipalPoint = [cx, cy], ImageSize = [yükseklik, genişlik] sırasıyla verilir).

Kavramlar
---------
* **İç parametreler (intrinsics):** Kameranın 3B dünyayı 2B piksele nasıl
  izdüşürdüğünü tanımlar. ``fx, fy`` piksel cinsinden odak uzaklığı,
  ``cx, cy`` görüntü merkezinin (asal nokta) piksel koordinatıdır.
  Hepsi 3x3'lük **K matrisinde** toplanır.
* **Radyal distorsiyon (k1, k2):** Lensin görüntüyü kenarlara doğru bükmesi.
  Homografi gibi geometrik hesaplardan önce düzeltilmezse metrik kestirim
  sistematik hata alır. Termal kamerada k1=-0.35 ile bu etki oldukça büyüktür.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraProfile:
    """Tek bir kameranın iç parametreleri."""

    name: str
    width: int   # görüntü genişliği (piksel)
    height: int  # görüntü yüksekliği (piksel)
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float
    k2: float

    @property
    def K(self) -> np.ndarray:
        """3x3 kamera matrisi."""
        return np.array(
            [[self.fx, 0.0, self.cx],
             [0.0, self.fy, self.cy],
             [0.0, 0.0, 1.0]]
        )

    @property
    def dist_coeffs(self) -> np.ndarray:
        """OpenCV distorsiyon katsayı vektörü [k1, k2, p1, p2].

        Kalibrasyon dosyasında teğetsel distorsiyon [0 0] verildiği için
        p1 = p2 = 0'dır.
        """
        return np.array([self.k1, self.k2, 0.0, 0.0])

    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        """Tüm görüntünün distorsiyonunu düzelt (görselleştirme için)."""
        return cv2.undistort(image, self.K, self.dist_coeffs)

    def undistort_points(self, points: np.ndarray) -> np.ndarray:
        """Nx2 piksel koordinatlarının distorsiyonunu düzelt.

        Geometri hesaplarında tüm görüntüyü düzeltmek yerine yalnızca takip
        edilen noktaları düzeltmek çok daha ucuzdur. ``P=K`` verildiği için
        sonuç yine piksel koordinatında döner.
        """
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
        out = cv2.undistortPoints(pts, self.K, self.dist_coeffs, P=self.K)
        return out.reshape(-1, 2)


# Kalibrasyon dosyasındaki üç kamera profili
THERMAL = CameraProfile(
    name="termal", width=640, height=512,
    fx=731.7965, fy=732.0172, cx=319.2367, cy=251.2424,
    k1=-0.3507, k2=0.1137,
)
RGB_4K = CameraProfile(
    name="rgb_4k", width=4000, height=3000,
    fx=2792.2, fy=2795.2, cx=1988.0, cy=1562.2,
    k1=0.0798, k2=-0.1867,
)
RGB_1080P = CameraProfile(
    name="rgb_1080p", width=1920, height=1080,
    fx=1389.7, fy=1387.1, cx=954.007, cy=558.896,
    k1=0.1378, k2=-0.2564,
)

# 2025 örnek veri setindeki 4K videolar **3840×2160** çıkıyor, kalibrasyon dosyası ise
# 4000×3000 sensör için verilmiş. Tutarlı tek açıklama: 4:3 sensörden 16:9 video —
# dikeyde 3000 → 2250 kırpma, sonra 4000 → 3840 ölçekleme (s = 0,96).
#   fx' = fx·s,  fy' = fy·s,  cx' = cx·s,  cy' = (cy − (3000−2250)/2)·s
# Tutarlılık kontrolü: özgün cy=1562,2 sensör merkezinden 62,2 px aşağıda;
# türetilen cy'=1139,7 de video merkezinden (1080) 59,7 px aşağıda — aynı fiziksel
# kayma. Distorsiyon katsayıları normalize koordinatta tanımlı olduğu için değişmez.
#
# NOT: yatay kanalda odak uzaklığı hatası Procrustes ölçeği (`Alignment.height`)
# tarafından soğurulur; bu yüzden f'teki olası küçük sapma yöntem karşılaştırmasını
# bozmaz. Asıl önemli olan asal nokta ve distorsiyon düzeltmesidir.
_S25 = 3840.0 / 4000.0
RGB_4K_VIDEO = CameraProfile(
    name="rgb_4k_video", width=3840, height=2160,
    fx=2792.2 * _S25, fy=2795.2 * _S25,
    cx=1988.0 * _S25, cy=(1562.2 - 375.0) * _S25,
    k1=0.0798, k2=-0.1867,
)

PROFILES = (THERMAL, RGB_4K, RGB_1080P, RGB_4K_VIDEO)


def calisma_olcegi(width: int) -> float:
    """RAFT'ın bu genişlikte kullandığı işlem ölçeği (``RaftTracker.auto_scale``).

    Burada da tanımlı olması, geometri katmanının eşiği çalışma çözünürlüğüne
    göre normalize edebilmesi içindir (bkz. ``ransac_esigi``); ikisi aynı kuralı
    kullanmak zorunda.
    """
    if width <= 1024:
        return 1.0
    if width <= 2048:
        return 0.5
    return 0.25


# Eşiğin ölçüldüğü referans: 0,5 **çalışma pikseli**. İki modalitede bağımsız
# olarak doğrulandı (3 permütasyon, eşleştirilmiş, yarışma takvimi):
#   RGB 1080p (çalışma ölçeği 0,50) optimum yerel eşik 1,00 px -> 0,50 çalışma px
#   Termal    (çalışma ölçeği 1,00) optimum yerel eşik 0,50 px -> 0,50 çalışma px
# RAFT akış gürültüsü çalışma çözünürlüğünde sabittir; noktalar doğal piksele
# geri ölçeklendiği için doğal eşik 1/ölçek ile büyümelidir. Sabit 1,0 px
# kullanmak termalde eşiği 2× gevşetiyor, 4K'da 4× sıkıyordu.
REFERANS_CALISMA_ESIGI = 0.5


def ransac_esigi(profile: CameraProfile, calisma_esigi: float = REFERANS_CALISMA_ESIGI) -> float:
    """Çalışma pikselinde verilen eşiği bu kameranın doğal pikseline çevir."""
    return float(calisma_esigi / calisma_olcegi(profile.width))


def turet_profil(taban: CameraProfile, width: int, height: int) -> CameraProfile:
    """Bilinen bir sensör profilinden başka bir kare boyutu için profil türet.

    Kameralar sensörün tamamını video olarak vermez: tipik olarak istenen en-boy
    oranına **kırpar**, sonra hedef çözünürlüğe **ölçekler**. Bu iki adım iç
    parametreleri şöyle değiştirir (kırpma merkezli varsayılır):

        s   = width / kırpılmış_genişlik
        fx' = fx·s,  fy' = fy·s
        cx' = (cx - kırpma_x)·s,  cy' = (cy - kırpma_y)·s

    Radyal distorsiyon katsayıları normalize koordinatta tanımlı olduğu için
    **değişmez**.

    Neden gerekli: 2025 örnek verisinde RGB video 3840×2160, kalibrasyon ise
    4000×3000 (4:3 sensör). Eşleşme olmadığı için ``profile_for_size`` hata
    veriyordu. Finalde kamera/çözünürlük önceden bilinmediğinden bu türetme,
    bilinmeyen bir boyutta sistemin çökmemesi için de gereklidir.
    """
    # hedef en-boy oranını sensörden kırparak yakala (merkezli kırpma)
    hedef_oran = width / height
    if taban.width / taban.height > hedef_oran:      # sensör daha geniş -> yanlardan kırp
        kirp_w = taban.height * hedef_oran
        kirp_h = float(taban.height)
    else:                                            # sensör daha uzun -> üst/alttan kırp
        kirp_w = float(taban.width)
        kirp_h = taban.width / hedef_oran
    kirp_x = (taban.width - kirp_w) / 2.0
    kirp_y = (taban.height - kirp_h) / 2.0
    s = width / kirp_w
    return CameraProfile(
        name=f"{taban.name}_{width}x{height}",
        width=width, height=height,
        fx=taban.fx * s, fy=taban.fy * s,
        cx=(taban.cx - kirp_x) * s, cy=(taban.cy - kirp_y) * s,
        k1=taban.k1, k2=taban.k2,
    )


def profile_for_size(width: int, height: int, katı: bool = False) -> CameraProfile:
    """Görüntü boyutuna göre kamera profilini seç; eşleşme yoksa türet.

    ``katı=True`` eski davranış (eşleşme yoksa hata). Varsayılan olarak en yakın
    odak uzunluğuna sahip sensörden türetiriz — yarışma günü bilinmeyen bir
    çözünürlük gelirse sistemin durmaması için.
    """
    for profile in PROFILES:
        if profile.width == width and profile.height == height:
            return profile
    if katı:
        raise ValueError(f"Bilinmeyen goruntu boyutu: {width}x{height}")
    # aynı sensörden gelmiş olması en olası profil: genişliği en yakın olan
    taban = min(PROFILES, key=lambda p: abs(np.log(p.width / max(width, 1))))
    return turet_profil(taban, width, height)


def profile_for_frame(frame: np.ndarray) -> CameraProfile:
    """Bir video karesine (numpy dizisi) uyan kamera profilini seç."""
    h, w = frame.shape[:2]
    return profile_for_size(w, h)
