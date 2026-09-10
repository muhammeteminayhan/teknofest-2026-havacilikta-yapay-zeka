"""Sağlıklı penceredeki GT ile hizalama: eksen eşleme + metrik ölçek + z işareti.

Görüntüden çıkan hareketler *kamera ekseninde* ve *ölçeksizdir* (h=1
varsayımıyla). Yarışmanın istediği ise *referans koordinat sisteminde metre*.
Aradaki köprüyü, GT konumun sağlıklı geldiği dönemde kurarız — bu modülün işi:

* **M (2×2 dik matris):** başlangıç kamera ekseni ↔ dünya ekseni dönüşümü
  (gerekirse yansımayla; piksel ekseni "aşağı" baktığı için beklenir).
* **height (ölçek c):** birim-yükseklik deltalarını metreye çeviren çarpan.
  Fiziksel anlamı **yerden yükseklik** — GSD·f ile aynı büyüklük.
* **z_sign:** dünyanın z ekseni aşağı mı (+1, NED benzeri) yukarı mı (-1).

Kavram — **Procrustes/Kabsch hizalaması:** iki vektör kümesi arasındaki en iyi
dik dönüşümü kapalı formda (SVD ile) bulur: ``H = GᵀE``, ``M = U·Vᵀ``.
Ölçek de ``c = Σσᵢ / Σ|E|²`` ile gelir. RANSAC'a gerek yok; girdiler zaten
RANSAC'tan geçmiş adım vektörleridir.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Alignment:
    """Sağlıklı pencereden öğrenilen kamera→dünya köprüsü."""

    M: np.ndarray            # 2x2 dik matris (det ±1 olabilir)
    height: float            # metrik ölçek = yerden yükseklik [m]
    z_sign: float            # +1: dünya z aşağı, -1: yukarı
    mean_gt_z: float         # pencerenin ortalama GT z'si (h'yi güncel tutmak için)
    n_used: int
    residual_ratio: float    # kalan hata oranı |G - cME| / |G| (0'a yakın = iyi)

    @property
    def has_reflection(self) -> bool:
        return float(np.linalg.det(self.M)) < 0

    def current_height(self, gt_z: float) -> float:
        """Sağlıklı dönemde güncel yükseklik: GT z değişimiyle taşınır.

        +z aşağıysa z artınca araç alçalır → h azalır (ve tersi).
        """
        return self.height - self.z_sign * (gt_z - self.mean_gt_z)


def fit_alignment(
    E: np.ndarray,
    G: np.ndarray,
    dz_unit: np.ndarray,
    gt_dz: np.ndarray,
    gt_z: np.ndarray,
    prev_z_sign: float = 1.0,
    z_min_hareket: float = 2.0,
    z_min_kanit: float = 0.30,
) -> Optional[Alignment]:
    """Hizalamayı en küçük karelerle çöz.

    Parametreler
    ------------
    E : Nx2 — ψ=0 kamera çerçevesine döndürülmüş, h=1 varsayımlı yatay deltalar
    G : Nx2 — aynı adımların GT yatay deltaları [m]
    dz_unit : N — h=1 varsayımlı düşey deltalar (görüntü ölçeğinden)
    gt_dz : N — GT z deltaları [m]
    gt_z : N — adım sonlarındaki GT z değerleri (pencere ortalaması için)
    z_min_hareket : pencerede işaret kararı için gereken GERÇEK irtifa gezinmesi [m]
    z_min_kanit : işaret tutarlılığı eşiği |Σaᵢbᵢ| / Σ|aᵢbᵢ| ∈ [0, 1]
    """
    E = np.asarray(E, dtype=float)
    G = np.asarray(G, dtype=float)
    if len(E) < 3 or (E ** 2).sum() < 1e-12:
        return None

    H = G.T @ E
    U, S, Vt = np.linalg.svd(H)
    M = U @ Vt                       # yansımaya izin: det -1 çıkabilir
    c = float(S.sum() / (E ** 2).sum())
    if not np.isfinite(c) or c <= 0:
        return None

    pred = (M @ E.T).T * c
    g_norm = float(np.linalg.norm(G))
    residual = float(np.linalg.norm(G - pred) / g_norm) if g_norm > 0 else 1.0

    # ------------------------------------------------------------------ z işareti
    # Dünyanın z ekseni aşağı mı (+1) yukarı mı (-1) bakıyor. Kesintide düşey adım
    # `dz = z_sign · dz_unit · height` ile uygulandığı için bu TEK BİT yanlışsa
    # düşey kanal komple ters döner.
    #
    # ESKİ KURAL — `guc = Σ|dz_unit| · c > 1,0 m` ise `sign(Σ dz_unit·gt_dz)`.
    # HATALIYDI: `guc` ÖLÇÜMÜN büyüklüğünü ölçüyor, uçağın GERÇEK irtifa
    # hareketini değil. Düz uçuşta bile 60 örnekte |dz_unit| medyanı ~0,004 ve
    # c ~42 m olduğu için `guc` 1,4 m'ye çıkıyor — kapı pratikte HİÇ kapanmıyor.
    # Ölçüldü (genelleme sınavı, pencere 1'in kalibrasyon penceresi):
    #     t = Σ dz_unit·gt_dz = -1,04e-4    <- kendi gürültü tabanının 4 kat altında
    #     guc = 1,46 m  -> kapı GEÇER       <- ama GT z gezinmesi yalnız 0,188 m
    #     kor(dz_unit, gt_dz) = -0,071      <- yani hiçbir bilgi yok
    # İşaret böylece yazı-tura ile -1 seçiliyor ve puanlanan blokta uçak 10 m
    # irtifa değiştirince düşey hata 11 m oluyor. Yataklarda ölçülen kararsızlık:
    #     video1 %27 · video2 %38 (son karar YANLIŞ) · v25_rgb_b %26 · v25_termal_b %40
    #
    # YENİ KURAL — iki koşul birlikte:
    #   1) pencerede GERÇEK irtifa gezinmesi ≥ `z_min_hareket`  (GT'den, gürültüsüz)
    #   2) işaret tutarlılığı `|Σ aᵢbᵢ| / Σ|aᵢbᵢ|` ≥ `z_min_kanit`
    #      (1 = her örnek aynı yönde, 0 = tam sadeleşme — birbirini götüren
    #       gürültüyü "kanıt" saymaz)
    # Hiçbiri sağlanmazsa bilgi YOKTUR: önceki işaret korunur. Başlangıç değeri
    # +1'dir ve şartnamenin z-aşağı konvansiyonuyla uyumludur.
    gz_arr = np.asarray(gt_z, dtype=float)
    gercek_dusey = float(np.ptp(gz_arr)) if gz_arr.size else 0.0
    carpim = np.asarray(dz_unit, dtype=float) * np.asarray(gt_dz, dtype=float)
    t = float(np.sum(carpim))
    toplam = float(np.sum(np.abs(carpim)))
    kanit = abs(t) / toplam if toplam > 0 else 0.0
    z_sign = (float(np.sign(t))
              if (gercek_dusey >= z_min_hareket and kanit >= z_min_kanit and t != 0.0)
              else prev_z_sign)

    return Alignment(
        M=M,
        height=c,
        z_sign=z_sign,
        mean_gt_z=float(np.mean(gt_z)),
        n_used=len(E),
        residual_ratio=residual,
    )
