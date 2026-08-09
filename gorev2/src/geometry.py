"""İki kare arasındaki kamera hareketini kestiren geometri araçları.

Kamera çerçevesi uzlaşımı (nadir bakışta): **X sağa, Y görüntüde aşağı,
Z ileri = yere doğru.** Böylece Z, dünya "aşağı" yönüne yaklaşık hizalıdır
(veri keşfinde +z'nin aşağı baktığını bulmuştuk — NED benzeri).

İki kestirim yolu
-----------------
1. **Benzerlik (4 serbestlik):** ``p1 ≈ s·R(θ)·p0 + t``. Tam dik bakış
   varsayımı altında geçerli; çok gürbüz, yorumu kolay:

   * medyan akış ``(du, dv)`` → yatay hareket:  ``Δxy_cam = -(du, dv)·h/f``
   * ölçek ``s``  → düşey hareket:  ``h₂ = h₁/s``,  ``Δz_cam = h₁(1 - 1/s)``
     (alçalınca zemin büyür: s>1, Δz_cam>0 = aşağı)
   * görüntü dönmesi ``θ`` → yön değişimi: ``Δψ = -θ``
     (kamera +ψ dönerse zemin görüntüde -ψ döner)

2. **Homografi (8 serbestlik) + ayrıştırma:** Düzlemsel sahnede iki görüntü
   ``H = K(R - t·nᵀ/d)K⁻¹`` ile bağlıdır. ``cv2.decomposeHomographyMat``
   4 aday {R, t/d, n} verir; fiziksel çözüm, derinlik pozitifliği ve düzlem
   normalinin beklenen yöne (nadir'de ≈ [0,0,1]) yakınlığı ile seçilir.
   Eğik bakışta (70°) benzerlikten daha doğrudur; karşılığında çözüm seçimi
   ve gürültü hassasiyeti getirir.

Her iki yol da RANSAC ile aykırı eşleşmeleri (hareketli taşıtlar!) eler.
İşaret uzlaşımları ``notebooks/02_vo_cekirdegi.ipynb`` içindeki sentetik
senaryolarla sayısal olarak doğrulanmıştır.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


# --------------------------------------------------------------------------
# 1) Benzerlik yolu
# --------------------------------------------------------------------------
@dataclass
class SimilarityMotion:
    """İki kare arasındaki 4 serbestlikli görüntü hareketi."""

    du: float            # medyan akış x (undistorted px)
    dv: float            # medyan akış y (undistorted px)
    scale: float         # s (>1: zemin büyüyor = alçalma)
    angle_rad: float     # θ, görüntü düzleminde dönme
    n_inliers: int
    inlier_ratio: float

    def camera_delta(self, height: float, f: float) -> np.ndarray:
        """Kamera çerçevesinde metrik yer değiştirme (dx, dy, dz).

        ``height``: kameranın zemine dik uzaklığı (m), ``f``: odak (px).
        dz pozitif = aşağı (yere yaklaşma).

        **Çerçeve uzlaşımı:** dönen vektör **kamera-2 (adım sonrası)**
        ekseninde ifade edilidir — sentetik F senaryosuyla doğrulandı.
        Dünyaya çevirirken güncellenmiş yön kullanılmalı:
        ``Δp_dünya = R_dünya←kam(ψ_yeni) · Δp``.
        """
        gsd = height / f
        dx = -self.du * gsd
        dy = -self.dv * gsd
        dz = height * (1.0 - 1.0 / self.scale)
        return np.array([dx, dy, dz])

    def height_after(self, height: float) -> float:
        """Bu adımdan sonra zemine dik uzaklık."""
        return height / self.scale

    @property
    def yaw_delta_rad(self) -> float:
        """Kameranın yön (yaw) değişimi: Δψ = -θ."""
        return -self.angle_rad


    # ransac_thresh VARSAYILANI 2.0 -> 1.0 (kanıta dayalı; bkz. degerlendirme/OZET.md)
    # Kesinti hatası sistematikti (yön ~1,45°, ölçek %3,85). Yeniden-izdüşüm eşiğini
    # sıkılaştırmak, homografiye giren gevşek eşleşmeleri eleyerek bu yanlılığı düşürdü:
    #   yön 1,45° -> 1,27°,  ölçek %3,85 -> %2,11
    #   yarışma takvimi 4,343 -> 3,693 m (-%15)
    #   15 uzun-kesinti senaryosu 7,628 -> 6,439 m (-%15,6), 11/15 senaryoda daha iyi
    # (MAGSAC++ / USAC_ACCURATE denendi, her eşikte daha kötü çıktı.)
def estimate_similarity(
    pts0: np.ndarray,
    pts1: np.ndarray,
    ref_point: Optional[tuple[float, float]] = None,
    ransac_thresh: float = 1.0,
    min_inliers: int = 15,
    robust_method: int = cv2.RANSAC,
    n_deneme: int = 1,
) -> Optional[SimilarityMotion]:
    """Benzerlik hareketi; ``n_deneme>1`` ise topluluk (bkz.
    :func:`estimate_homography_motion` — aynı sıra-duyarlılığı gerekçesi)."""
    if n_deneme > 1:
        rng = np.random.default_rng(12345)
        adaylar = []
        for _ in range(n_deneme):
            j = rng.permutation(len(pts0))
            s = _sim_tek(pts0[j], pts1[j], ref_point, ransac_thresh,
                         min_inliers, robust_method)
            if s is not None:
                adaylar.append(s)
        if not adaylar:
            return None
        return SimilarityMotion(
            du=float(np.median([s.du for s in adaylar])),
            dv=float(np.median([s.dv for s in adaylar])),
            scale=float(np.median([s.scale for s in adaylar])),
            angle_rad=float(np.median([s.angle_rad for s in adaylar])),
            n_inliers=int(np.median([s.n_inliers for s in adaylar])),
            inlier_ratio=float(np.median([s.inlier_ratio for s in adaylar])),
        )
    return _sim_tek(pts0, pts1, ref_point, ransac_thresh, min_inliers, robust_method)


def _sim_tek(
    pts0: np.ndarray,
    pts1: np.ndarray,
    ref_point: Optional[tuple[float, float]] = None,
    ransac_thresh: float = 1.0,
    min_inliers: int = 15,
    robust_method: int = cv2.RANSAC,
) -> Optional[SimilarityMotion]:
    """RANSAC ile 4 serbestlikli benzerlik hareketi kestir; başarısızsa None.

    ``ref_point``: akışın değerlendirileceği piksel — **asal nokta (cx, cy)
    verilmelidir**. Optik eksenin zemini deldiği nokta olduğu için dönme ve
    ölçeğin akışa katkısı orada sıfırdır; böylece ``(du, dv)`` saf öteleme
    akışı olur (medyan akıştaki dönme/ölçek sapması ortadan kalkar).
    ``None`` verilirse inlier akışının medyanı kullanılır (daha az doğru).
    """
    if len(pts0) < min_inliers:
        return None
    # DIKKAT: estimateAffinePartial2D yalnizca RANSAC ve LMEDS destekler; USAC
    # aileleri (MAGSAC++ vb.) yalnizca findHomography/findFundamentalMat icin
    # gecerlidir. Desteklenmeyen bir yontem gelirse sessizce RANSAC'a duseriz —
    # boylece ayni `robust_method` her iki kestirime de guvenle gecirilebilir.
    if robust_method not in (cv2.RANSAC, cv2.LMEDS):
        robust_method = cv2.RANSAC
    M, inl = cv2.estimateAffinePartial2D(
        pts0, pts1, method=robust_method, ransacReprojThreshold=ransac_thresh
    )
    if M is None or inl is None:
        return None
    mask = inl.reshape(-1).astype(bool)
    if mask.sum() < min_inliers:
        return None
    if ref_point is not None:
        # akisi asal noktada degerlendir: M @ [cx, cy, 1] - [cx, cy]
        cx, cy = ref_point
        du = M[0, 0] * cx + M[0, 1] * cy + M[0, 2] - cx
        dv = M[1, 0] * cx + M[1, 1] * cy + M[1, 2] - cy
    else:
        flow = pts1[mask] - pts0[mask]
        du, dv = np.median(flow, axis=0)
    a, b = M[0, 0], M[1, 0]
    return SimilarityMotion(
        du=float(du),
        dv=float(dv),
        scale=float(np.hypot(a, b)),
        angle_rad=float(np.arctan2(b, a)),
        n_inliers=int(mask.sum()),
        inlier_ratio=float(mask.mean()),
    )


# --------------------------------------------------------------------------
# 2) Homografi yolu
# --------------------------------------------------------------------------
def _normalize_pts(p: np.ndarray):
    """Hartley normalizasyonu: merkez 0, ortalama uzaklık √2 (DLT koşullanması)."""
    mu = p.mean(axis=0)
    q = p - mu
    s = float(np.sqrt(2.0) / max(np.sqrt((q ** 2).sum(axis=1)).mean(), 1e-12))
    T = np.array([[s, 0.0, -s * mu[0]], [0.0, s, -s * mu[1]], [0.0, 0.0, 1.0]])
    return q * s, T


def _dlt_agirlikli(p0: np.ndarray, p1: np.ndarray, w: np.ndarray) -> Optional[np.ndarray]:
    """Ağırlıklı normalize DLT ile homografi çöz (p1 ≈ H·p0)."""
    n0, T0 = _normalize_pts(p0)
    n1, T1 = _normalize_pts(p1)
    sw = np.sqrt(np.maximum(w, 0.0))
    x, y = n0[:, 0], n0[:, 1]
    u, v = n1[:, 0], n1[:, 1]
    z = np.zeros_like(x)
    o = np.ones_like(x)
    A = np.empty((2 * len(x), 9))
    A[0::2] = np.column_stack([-x, -y, -o, z, z, z, u * x, u * y, u]) * sw[:, None]
    A[1::2] = np.column_stack([z, z, z, -x, -y, -o, v * x, v * y, v]) * sw[:, None]
    # Çözüm A'nın en küçük tekil değerine ait sağ tekil vektörü. Bunu A'nın tam
    # SVD'si yerine **normal denklemlerden** (AᵀA'nın en küçük özvektörü) alıyoruz:
    # AᵀA 9×9'dur, kurulumu O(N·81), özayrışması sabit maliyet — 1200×9'luk SVD'den
    # kat kat ucuz. GNC her çift için ~20 kez çözdüğü için bu fark canlı hızda
    # doğrudan görünür. Hartley normalizasyonu önce uygulandığı için A iyi
    # koşullanmıştır; AᵀA'nın koşul sayısını karesine çıkarması sorun yaratmaz.
    try:
        _, V = np.linalg.eigh(A.T @ A)
        Hn = V[:, 0].reshape(3, 3)
    except np.linalg.LinAlgError:
        return None
    H = np.linalg.inv(T1) @ Hn @ T0
    return H / H[2, 2] if abs(H[2, 2]) > 1e-12 else None


def _transfer_hata(H: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    """Simetrik transfer hatası [px] (ileri + geri izdüşüm)."""
    def ileri(M, a):
        h = np.column_stack([a, np.ones(len(a))]) @ M.T
        wz = np.where(np.abs(h[:, 2]) < 1e-12, 1e-12, h[:, 2])
        return h[:, :2] / wz[:, None]
    try:
        Hi = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return np.full(len(p0), np.inf)
    return (np.linalg.norm(ileri(H, p0) - p1, axis=1)
            + np.linalg.norm(ileri(Hi, p1) - p0, axis=1)) * 0.5


def gnc_homografi(p0: np.ndarray, p1: np.ndarray, H0: np.ndarray,
                  c: float = 1.0, tur: str = "tls", n_yineleme: int = 20,
                  faktor: float = 1.4) -> Optional[np.ndarray]:
    """Kademeli dışbükeysizleştirme (GNC) ile robust homografi.

    Kaynak alan: **robust uzaysal algılama / optimizasyon** (Yang & Carlone,
    arXiv:1909.08605; TEASER++). Bu boru hattına neden uygun:

    * **RANSAC** aykırıları *ayrık* seçer; merdiven çiftlerinde düzleme uyan oran
      ~%56 olduğu için birbirine yakın skorlu birden çok düzlem yarışır ve seçim
      nokta sırasına bağlı olur (ölçüldü: uçtan uca %18,7 yayılım).
    * **Sabit ölçekli IRLS** (bu dosyada, reddedildi) varyansı yok etti ama 3B yapıyı
      modele geri soktu → yanlılık (+0,36 … +1,82 m).

    GNC ikisinin arasını kapatır: dışbükeye yakın bir vekil maliyetle **tüm noktalardan**
    başlar (iyi bir havzaya düşer), sonra dışbükeysizlik parametresini kademeli
    keskinleştirerek **sert kesmeye** yakınsar (yapı noktaları dışarıda kalır).
    Black–Rangarajan ikiliği sayesinde her adım ağırlıklı en küçük karelerdir.

    ``tur="tls"``  — kesilmiş en küçük kareler vekili (yakınsamada sert kesme)
    ``tur="gm"``   — Geman-McClure (yakınsamada yumuşak, asla tam sıfırlamaz)
    ``c``          — kesme yarıçapı [px]; yeniden-izdüşüm eşiğinin muadili.
    """
    H = H0.copy()
    r = _transfer_hata(H, p0, p1)
    if not np.all(np.isfinite(r)):
        return None
    c2 = float(c) ** 2
    r2max = float(np.max(r ** 2))
    if r2max <= 1e-12:
        return H

    if tur == "tls":
        # mu kucukten buyuge: baslangicta neredeyse tum noktalar agirlikli (dışbukey)
        mu = c2 / max(2.0 * r2max - c2, 1e-9)
        mu = float(np.clip(mu, 1e-6, 1.0))
    else:
        mu = 2.0 * r2max / c2                      # buyukten kucuge

    for _ in range(n_yineleme):
        r2 = r ** 2
        if tur == "tls":
            ust = (mu + 1.0) / mu * c2
            alt = mu / (mu + 1.0) * c2
            w = np.zeros_like(r2)
            w[r2 <= alt] = 1.0
            orta = (r2 > alt) & (r2 < ust)
            if orta.any():
                w[orta] = (c * np.sqrt(mu * (mu + 1.0))
                           / np.maximum(np.sqrt(r2[orta]), 1e-9) - mu)
            w = np.clip(w, 0.0, 1.0)
        else:
            w = (mu * c2 / (r2 + mu * c2)) ** 2

        if w.sum() < 4:
            break
        Hy = _dlt_agirlikli(p0, p1, w)
        if Hy is None:
            break
        ry = _transfer_hata(Hy, p0, p1)
        if not np.all(np.isfinite(ry)):
            break
        H, r = Hy, ry
        mu = mu * faktor if tur == "tls" else mu / faktor
        if (tur == "tls" and mu > 1e6) or (tur != "tls" and mu < 1.0):
            break
    return H


def irls_homografi(p0: np.ndarray, p1: np.ndarray, H0: np.ndarray,
                   c: float = 2.0, n_yineleme: int = 8) -> Optional[np.ndarray]:
    """RANSAC çözümünü **tüm noktalar** üzerinde Cauchy M-kestirimiyle iyileştir.

    NEDEN: RANSAC inlier kümesini *ayrık* seçer. Merdiven çiftlerinde düzleme
    uyan nokta oranı ~%56 olduğu için birbirine yakın skorlu **birden çok düzlem**
    yarışır; hangisinin seçildiği nokta sırasına bağlıdır (ölçüldü: uçtan uca
    resmî metrikte 10 permütasyonda 3,67–4,41 m yayılım). Ayrık seçim yerine
    sürekli ağırlık ``w = 1/(1+(r/c)²)`` kullanmak:

    * seçim varyansını ortadan kaldırır (çözüm tek ve süreklidir),
    * 3B yapıyı (bina/ağaç) atmak yerine **söndürür** — zemin düzlemine
      yakın noktalar ağırlığını korur, uzaklar yumuşakça sıfırlanır.

    ``c`` piksel cinsinden ölçek parametresidir (yeniden-izdüşüm eşiğinin muadili).

    **DENENDİ ve REDDEDİLDİ.** Varyansı beklendiği gibi yok etti (aynı çiftin farklı
    permütasyonlarında hareket kestiriminin std'si **1e-3 → 1e-6**, yani çözüm tek ve
    kararlı) ama hatayı BÜYÜTTÜ (video1, 6 permütasyon, eşleştirilmiş):
        c=1,5 → **+0,360 m**,   c=2,0 → **+1,817 m**  (temel 3,917 m)
    Yorum: merdiven çiftlerinde düzleme uymayan ~%40'lık kesim **gürültü değil, gerçek
    3B yapı** (bina, ağaç). Yumuşak ağırlık onları modele geri sokuyor ve düzlemi
    zeminden yukarı çekiyor → ölçek yanlılığı. Bu sahnede **sert reddetme doğru araç**.
    Kayıt olarak bırakıldı; varsayılan kapalı (``irls_c=None``).
    """
    H = H0.copy()
    for _ in range(n_yineleme):
        r = _transfer_hata(H, p0, p1)
        if not np.all(np.isfinite(r)):
            return None
        w = 1.0 / (1.0 + (r / c) ** 2)
        if w.sum() < 4:
            return None
        Hy = _dlt_agirlikli(p0, p1, w)
        if Hy is None:
            return None
        if np.linalg.norm(Hy - H) < 1e-10:
            H = Hy
            break
        H = Hy
    return H


@dataclass
class HomographyMotion:
    """Homografi ayrıştırmasından seçilen fiziksel çözüm.

    * ``R``: kamera-1 koordinatındaki bir noktayı kamera-2 koordinatına
      taşıyan dönme (``p2 = R·p1 + t``).
    * ``cam_delta_over_d``: kamera-2'nin kamera-1'e göre konumu,
      kamera-1 çerçevesinde, düzlem uzaklığına bölünmüş (``c/d``).
    * ``n``: zemin düzleminin normali, kamera-1 çerçevesinde (birim,
      nadir bakışta ≈ [0, 0, 1]).
    """

    R: np.ndarray
    cam_delta_over_d: np.ndarray
    n: np.ndarray
    n_inliers: int
    inlier_ratio: float

    def camera_delta(self, plane_dist: float) -> np.ndarray:
        """Metrik yer değiştirme (d = düzleme dik uzaklık, m).

        **Çerçeve uzlaşımı:** dönen vektör **kamera-1 (adım öncesi)**
        ekseninde ifade edilir (benzerlik yolundan farklı!). Dünyaya
        çevirirken adım öncesi yön kullanılmalı:
        ``Δp_dünya = R_dünya←kam(ψ_eski) · Δp``.
        """
        return self.cam_delta_over_d * plane_dist

    def plane_dist_after(self, plane_dist: float) -> float:
        """Bu adımdan sonra düzleme dik uzaklık: d₂ = d₁(1 - nᵀ·c/d₁)."""
        return plane_dist * (1.0 - float(self.n @ self.cam_delta_over_d))

    @property
    def yaw_delta_rad(self) -> float:
        """Kameranın yön (yaw) değişimi (Z ekseni etrafında)."""
        return -float(np.arctan2(self.R[1, 0], self.R[0, 0]))

    @property
    def tilt_deg(self) -> float:
        """Bakışın nadir'den sapması: normal ile optik eksen arasındaki açı."""
        return float(np.degrees(np.arccos(np.clip(self.n[2], -1.0, 1.0))))


def estimate_homography_motion(
    pts0: np.ndarray,
    pts1: np.ndarray,
    K: np.ndarray,
    ransac_thresh: float = 1.0,
    min_inliers: int = 15,
    prev_normal: Optional[np.ndarray] = None,
    robust_method: int = cv2.RANSAC,
    n_deneme: int = 1,
    irls_c: Optional[float] = None,
    gnc_c: Optional[float] = None,
    gnc_tur: str = "tls",
    gnc_mad: Optional[float] = None,
) -> Optional[HomographyMotion]:
    """Homografi hareketi; ``n_deneme>1`` ise **topluluk (ensemble)** kestirimi.

    NEDEN TOPLULUK: OpenCV RANSAC girdi verildiğinde deterministiktir ama nokta
    **sırasına** duyarlıdır — aynı noktalar farklı sırada verilince farklı minimal
    örnekler çekilir ve *eşit derecede iyi ama farklı* bir model seçilir. Ölçüldü
    (merdiven çiftleri): inlier oranı |fark| ort 0,042 / maks 0,23; uçtan uca resmî
    metrikte 10 permütasyonda **ort 3,964 ± 0,226 m (yayılım 0,74 m = %18,7)**.

    Bu oynaklık kesinti boyunca bağımsız değil — sahne benzer olduğu için aynı
    "kötü düzlem" ardışık karelerde tekrar seçilir ve blok içinde *sistematik*
    sapma gibi görünür. Aynı çiftin birden çok permütasyonunu kestirip
    bileşen bazında **medyan** almak bu seçim varyansını söndürür.

    Maliyet yalnız geometri katmanındadır (RAFT akışı bir kez hesaplanır).

    **DENENDİ ve REDDEDİLDİ.** n=3 ile video1'de 3,917 → 3,974 m; ne hata ne varyans
    düştü. Gerekçe (IRLS denemesiyle aynı köke çıkıyor): permütasyonlar arası fark
    salt gürültü değil — farklı permütasyonlar farklı *düzlemler* seçiyor ve bunların
    medyanı fiziksel olarak anlamlı bir düzlem vermiyor. Varsayılan ``n_deneme=1``.
    """
    if n_deneme > 1:
        rng = np.random.default_rng(12345)
        adaylar = []
        for _ in range(n_deneme):
            j = rng.permutation(len(pts0))
            h = _hom_tek(pts0[j], pts1[j], K, ransac_thresh, min_inliers,
                         prev_normal, robust_method, irls_c, gnc_c, gnc_tur, gnc_mad)
            if h is not None:
                adaylar.append(h)
        if not adaylar:
            return None
        # Bileşen bazında medyan: normal ve yer değiştirme yönü aynı yarıkürede
        # olduğu için (fiziksel çözüm seçimi bunu garanti eder) doğrudan medyan
        # alınabilir; yaw zaten küçük açı, sarma sorunu yok.
        d = np.median(np.array([h.cam_delta_over_d for h in adaylar]), axis=0)
        nrm = np.median(np.array([h.n for h in adaylar]), axis=0)
        nrm = nrm / max(float(np.linalg.norm(nrm)), 1e-12)
        yaw = float(np.median([h.yaw_delta_rad for h in adaylar]))
        c, s = np.cos(-yaw), np.sin(-yaw)
        return HomographyMotion(
            R=np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]),
            cam_delta_over_d=d, n=nrm,
            n_inliers=int(np.median([h.n_inliers for h in adaylar])),
            inlier_ratio=float(np.median([h.inlier_ratio for h in adaylar])),
        )
    return _hom_tek(pts0, pts1, K, ransac_thresh, min_inliers, prev_normal,
                    robust_method, irls_c, gnc_c, gnc_tur, gnc_mad)


def _hom_tek(
    pts0: np.ndarray,
    pts1: np.ndarray,
    K: np.ndarray,
    ransac_thresh: float = 1.0,
    min_inliers: int = 15,
    prev_normal: Optional[np.ndarray] = None,
    robust_method: int = cv2.RANSAC,
    irls_c: Optional[float] = None,
    gnc_c: Optional[float] = None,
    gnc_tur: str = "tls",
    gnc_mad: Optional[float] = None,
) -> Optional[HomographyMotion]:
    """Homografi kestir, ayrıştır ve fiziksel çözümü seç; başarısızsa None.

    Çözüm seçimi iki kuralla yapılır:
    1. **Derinlik pozitifliği:** Görünen noktaların ışınları düzlemi kameranın
       önünde kesmelidir (``nᵀ·K⁻¹x > 0``). Bu, 4 adayın ikisini eler.
    2. **Normal tutarlılığı:** Kalan iki adaydan, normali referans yöne
       (önceki karenin normali; yoksa nadir beklentisi [0,0,1]) en yakın
       olan seçilir.
    """
    if len(pts0) < min_inliers:
        return None
    H, mask = cv2.findHomography(pts0, pts1, robust_method, ransac_thresh)
    if H is None or mask is None:
        return None
    mask = mask.reshape(-1).astype(bool)
    if mask.sum() < min_inliers:
        return None

    H_iyi = None
    if gnc_mad is not None:
        # KESME YARIÇAPINI VERİDEN TÜRET: c = k · 1,4826 · MAD(artıklar).
        # Sabit `c` modaliteler arasında taşınmıyor (termal büyük, RGB küçük c
        # istiyor); MAD ölçeği her kare çiftinde kendini ayarlar, böylece tek
        # ayar (k) üç kamerada da aynı istatistiksel anlamı taşır.
        r0 = _transfer_hata(H, pts0, pts1)
        if np.all(np.isfinite(r0)):
            mad = float(np.median(np.abs(r0 - np.median(r0)))) * 1.4826
            c_mad = max(gnc_mad * mad, 1e-3)
            H_iyi = gnc_homografi(pts0, pts1, H, c=c_mad, tur=gnc_tur)
    elif gnc_c is not None:
        H_iyi = gnc_homografi(pts0, pts1, H, c=gnc_c, tur=gnc_tur)
    elif irls_c is not None:
        # RANSAC yalnizca baslangic noktasi; asil cozum tum noktalar uzerinde
        # surekli agirlikli M-kestirimiyle bulunur (bkz. irls_homografi).
        H_iyi = irls_homografi(pts0, pts1, H, c=irls_c)
    if H_iyi is not None:
        H = H_iyi
        r = _transfer_hata(H, pts0, pts1)
        mask = r < ransac_thresh              # raporlanan inlier orani tutarli kalsin
        if mask.sum() < min_inliers:
            mask = r < max(ransac_thresh, float(np.median(r)))

    try:
        _, Rs, ts, ns = cv2.decomposeHomographyMat(H, K)
    except cv2.error:
        return None

    # Derinlik pozitifliği icin inlier isinlari (kamera-1 cerecevesi)
    pts_h = np.column_stack([pts0[mask], np.ones(int(mask.sum()))])
    rays = (np.linalg.inv(K) @ pts_h.T).T  # Nx3

    ref = prev_normal if prev_normal is not None else np.array([0.0, 0.0, 1.0])
    best, best_score = None, -np.inf
    for R, t, n in zip(Rs, ts, ns):
        n = n.reshape(3)
        # kural 1: noktalarin en az %95'i duzlemin gorunen tarafinda olmali
        if (rays @ n > 0).mean() < 0.95:
            continue
        # kural 2: referans normale yakinlik
        score = float(n @ ref)
        if score > best_score:
            best_score = score
            best = (R, t.reshape(3), n)

    if best is None:
        # Saf donme durumu: oteleme ~0 iken H ≈ K·R·K⁻¹ olur ve duzlem
        # ayristirmasi tanimsizdir (normal belirsiz, adaylar gecersiz cikar).
        # Bu durumda donmeyi dogrudan H'den cikarip otelemeyi 0 kabul ederiz;
        # boylece hover/donus anlarinda yon takibi kopmaz.
        A = np.linalg.inv(K) @ H @ K
        det = np.linalg.det(A)
        if det <= 0:
            return None
        A = A / np.cbrt(det)                      # olcegi normalize et (det=1)
        U, _, Vt = np.linalg.svd(A)
        R = U @ np.diag([1.0, 1.0, np.linalg.det(U @ Vt)]) @ Vt
        if np.linalg.norm(A - R) > 0.05:          # gercekten donmeye yakin mi?
            return None
        return HomographyMotion(
            R=R,
            cam_delta_over_d=np.zeros(3),
            n=ref.copy(),
            n_inliers=int(mask.sum()),
            inlier_ratio=float(mask.mean()),
        )

    R, t, n = best
    # p2 = R p1 + t (t = gercek_t/d)  =>  kamera-2'nin kamera-1'deki konumu: c/d = -Rᵀ t
    cam_delta_over_d = -(R.T @ t)
    return HomographyMotion(
        R=R,
        cam_delta_over_d=cam_delta_over_d,
        n=n,
        n_inliers=int(mask.sum()),
        inlier_ratio=float(mask.mean()),
    )
