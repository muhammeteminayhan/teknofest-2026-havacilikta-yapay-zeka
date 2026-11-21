"""Uçtan uca pozisyon kestirim boru hattı: ölçüm + durum makinesi.

İki katmandan oluşur:

1. **PairMeasurement** — iki kareden çıkarılan *ham görsel ölçüm* (pahalı kısım).
   Benzerlik ve homografi kestirimlerinin ikisini de taşır; CSV'ye
   yazılıp geri okunabilir. Böylece videolar bir kez işlenir, durum makinesi
   yüzlerce kesinti senaryosunda saniyeler içinde yeniden koşturulabilir.
   Yarışma günü de aynı ölçüm canlı üretilip aynı durum makinesine verilir.

2. **VoPositionEstimator** — yarışma akışına birebir uyan durum makinesi.
   Her karede ``update(ölçüm, gt, healthy)`` çağrılır:

   * **Sağlıklı:** çıktı = GT (0 hata). Bu sırada GT'den sürekli öğrenilir:
     eksen hizalaması M, metrik ölçek/yükseklik, z işareti (bkz.
     ``scale_alignment``) ve son hız. Konum GT'ye çapalanır → kesinti
     başladığında sürüklenme sıfırdan başlar.
   * **Kesinti:** görsel ölçüm dünyaya çevrilip konuma eklenir (dead
     reckoning). Yükseklik görüntü ölçeğiyle taşınır. Ölçüm kalite
     kapılarından geçemezse (az nokta, düşük inlier, donmuş kare...)
     **sabit-hız yedeğine** düşülür.

Çerçeve uzlaşımları (02 defterinde sentetik olarak kanıtlandı):
benzerlik deltası kamera-2 (adım sonrası), homografi deltası kamera-1
(adım öncesi) ekseninde → yön döndürmesi buna göre yapılır.
"""
from __future__ import annotations

import inspect
from collections import deque
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

import cv2
import numpy as np

from . import geometry
from .calibration import CameraProfile, ransac_esigi
from .matching import FeatureTracker
from .scale_alignment import Alignment, fit_alignment


@lru_cache(maxsize=None)
def _kabul_edilen(fn) -> frozenset:
    return frozenset(inspect.signature(fn).parameters)


def _suz(fn, kwargs: dict) -> dict:
    """`kwargs`tan yalnızca `fn`in kabul ettiği anahtarları geçir."""
    kabul = _kabul_edilen(fn)
    return {k: v for k, v in kwargs.items() if k in kabul}


# --------------------------------------------------------------------------
# 1) Ham görsel ölçüm
# --------------------------------------------------------------------------
@dataclass
class PairMeasurement:
    """İki ardışık kare arasındaki görsel hareket ölçümü (h=1 ölçeksiz)."""

    ok: bool = False
    frozen: bool = False          # kare donmuş/tekrarlanmış mı?
    n_kept: int = 0               # güvenilir eşleşme sayısı

    sim_ok: bool = False
    sim_unit_dx: float = np.nan   # kamera-2 ekseninde, h=1 için metre/metre
    sim_unit_dy: float = np.nan
    sim_dz_unit: float = np.nan   # 1 - 1/s  (h=1 için düşey delta, + = alçalma)
    sim_yaw: float = np.nan       # yön değişimi [rad]
    sim_inlier_ratio: float = 0.0

    hom_ok: bool = False
    hom_ux: float = np.nan        # kamera-1 ekseninde delta/d
    hom_uy: float = np.nan
    hom_uz: float = np.nan
    hom_nx: float = np.nan        # düzlem normali (kamera-1)
    hom_ny: float = np.nan
    hom_nz: float = np.nan
    hom_yaw: float = np.nan
    hom_inlier_ratio: float = 0.0

    # İKİNCİ (gevşek eşikli) homografi fitinden yalnız YÖN. Gerekçe ölçümle
    # bulundu: yön ve ölçek hataları zıt eşik istiyor. Termal videoda eşik
    # taraması (3 permütasyon, eşleştirilmiş):
    #     eşik 0,50 -> yön 19,39° · ölçek %2,34
    #     eşik 1,00 -> yön  3,70° · ölçek %14,56
    #     eşik 1,50 -> yön  3,12° · ölçek %16,91
    # Sıkı eşik yalnız gerçek zemin düzlemine uyan noktaları bırakıp ölçeği
    # (düzlem uzaklığı d) doğru veriyor, ama az ve dar dağılımlı nokta rotasyonu
    # kötü koşullandırıyor. Gevşek eşik tersi. RGB tarafında da aynı ayrışma
    # kayıtlı (ds0,75: yön 1,27°->0,65° ama ölçek %2,11->%2,53).
    # Çözüm: öteleme/ölçek sıkı fitten, yön gevşek fitten.
    hom_yaw_gevsek: float = np.nan
    hom_inlier_gevsek: float = 0.0

    @classmethod
    def compute(
        cls,
        gray0: np.ndarray,
        gray1: np.ndarray,
        profile: CameraProfile,
        tracker: FeatureTracker,
        prev_normal: Optional[np.ndarray] = None,
        freeze_diff_thresh: float = 0.35,
        rng_tohum: Optional[int] = None,
        **geo_kwargs,
    ) -> "PairMeasurement":
        """İki kareden ölçüm üret (yarışmada canlı, provada önbelleğe)."""
        m = cls()
        # Donmuş kare tespiti: kareler pratikte ayniysa akış da sahte 0 olur;
        # bu durumda ölçüm 'geçersiz' sayılır ki sabit-hız yedeği devreye girsin.
        kucuk0 = gray0[::8, ::8].astype(np.int16)
        kucuk1 = gray1[::8, ::8].astype(np.int16)
        if float(np.abs(kucuk0 - kucuk1).mean()) < freeze_diff_thresh:
            m.frozen = True
            return m

        mr = tracker.track_pair(gray0, gray1)
        if mr is None:
            return m
        return cls.from_points(mr.pts0, mr.pts1, profile, prev_normal=prev_normal,
                               n_kept=mr.n_kept, taban=m, rng_tohum=rng_tohum,
                               **geo_kwargs)

    @classmethod
    def from_points(
        cls,
        pts0: np.ndarray,
        pts1: np.ndarray,
        profile: CameraProfile,
        prev_normal: Optional[np.ndarray] = None,
        n_kept: Optional[int] = None,
        taban: Optional["PairMeasurement"] = None,
        rng_tohum: Optional[int] = None,
        **geo_kwargs,
    ) -> "PairMeasurement":
        """Hazır nokta eşleşmelerinden ölçüm üret (eşleştirme adımı atlanır).

        `compute()` bunun üstüne kuruludur; ayrıca **nokta önbelleği** ile geometri
        denemelerini (MAGSAC++ vb.) video yeniden taramadan koşturmayı sağlar.
        `geo_kwargs` doğrudan `geometry.*` çağrılarına geçer (`robust_method`,
        `ransac_thresh`, ...).

        ``rng_tohum``: OpenCV'nin **global** RANSAC üreteci verilen tohumla
        sabitlenir. Tohumsuzken sonuç, o ana kadar yapılmış tüm OpenCV
        çağrılarının geçmişine bağlıdır; aynı girdi farklı bağlamlarda farklı
        inlier maskesi verir (ölçüldü: merdiven inlier ortalaması 0,56 ↔ 0,61,
        resmî metrikte ±%1,3). A/B karşılaştırmasının anlamlı olması için
        ölçüm katmanının **yeniden üretilebilir** olması şarttır.
        """
        m = taban if taban is not None else cls()
        m.n_kept = int(n_kept if n_kept is not None else len(pts0))
        if rng_tohum is not None:
            cv2.setRNGSeed(int(rng_tohum))

        # Eşik çalışma pikselinde verildiyse bu kameranın doğal pikseline çevir.
        # (Kameradan bağımsız tek ayar; gerekçe ve ölçüm: calibration.ransac_esigi)
        geo_kwargs = dict(geo_kwargs)
        cal_esik = geo_kwargs.pop("calisma_esigi", None)
        if cal_esik is not None:
            geo_kwargs["ransac_thresh"] = ransac_esigi(profile, float(cal_esik))
        # NOT: eşiği kameraya göre normalize etmek (`calisma_esigi`, 0,5 çalışma
        # pikseli) 2026 verisinde doğru görünüyordu ama yıllar arası sınavda
        # taşınmadı: termalde ancak çok gevşek kapıyla (0,05) kazandırıyor, 0,30'da
        # kaybettiriyor. Varsayılan tarihsel davranışta bırakıldı (her kamerada
        # 1,0 px); `calisma_esigi` açıkça verilirse yine kullanılabilir.
        # Ayrıntı: degerlendirme/YONTEM_SINAVI.md

        f = profile.fx
        pp = (profile.cx, profile.cy)
        # geo_kwargs iki kestiriciye birden gider ama parametre kumeleri ayni
        # degil (orn. `irls_c` yalniz homografide anlamli). Imzaya gore suzuyoruz
        # ki cagiran taraf tek bir sozluk gecirebilsin.
        sim = geometry.estimate_similarity(
            pts0, pts1, ref_point=pp, **_suz(geometry.estimate_similarity, geo_kwargs))
        if sim is not None:
            delta_unit = sim.camera_delta(1.0, f)      # h=1 için
            m.sim_ok = True
            m.sim_unit_dx = float(delta_unit[0])
            m.sim_unit_dy = float(delta_unit[1])
            m.sim_dz_unit = float(delta_unit[2])
            m.sim_yaw = float(sim.yaw_delta_rad)
            m.sim_inlier_ratio = float(sim.inlier_ratio)

        hom = geometry.estimate_homography_motion(
            pts0, pts1, profile.K, prev_normal=prev_normal,
            **_suz(geometry.estimate_homography_motion, geo_kwargs)
        )
        if hom is not None:
            m.hom_ok = True
            m.hom_ux, m.hom_uy, m.hom_uz = (float(v) for v in hom.cam_delta_over_d)
            m.hom_nx, m.hom_ny, m.hom_nz = (float(v) for v in hom.n)
            m.hom_yaw = float(hom.yaw_delta_rad)
            m.hom_inlier_ratio = float(hom.inlier_ratio)

        # İkinci fit: yalnız yön için, kendi (gevşek) eşiğiyle. Bkz. alan açıklaması.
        esik_yon = geo_kwargs.get("ransac_thresh_yon")
        if esik_yon is not None:
            gk = dict(geo_kwargs)
            gk.pop("ransac_thresh_yon", None)
            gk["ransac_thresh"] = float(esik_yon)
            hom_g = geometry.estimate_homography_motion(
                pts0, pts1, profile.K, prev_normal=prev_normal,
                **_suz(geometry.estimate_homography_motion, gk))
            if hom_g is not None:
                m.hom_yaw_gevsek = float(hom_g.yaw_delta_rad)
                m.hom_inlier_gevsek = float(hom_g.inlier_ratio)

        m.ok = m.sim_ok or m.hom_ok
        return m

    @property
    def hom_normal(self) -> Optional[np.ndarray]:
        if not self.hom_ok:
            return None
        return np.array([self.hom_nx, self.hom_ny, self.hom_nz])


def rot2d(psi: float) -> np.ndarray:
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s], [s, c]])


def duzlem_tabani(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Zemin düzlemi içinde, kamera x eksenine dayalı ortonormal taban (e1, e2).

    NEDEN GEREKLİ: homografi ayrıştırmasından çıkan düzleme paralel bileşen
    ``yatay = delta - (n·delta)·n`` kamera çerçevesinde bir **3-vektördür**.
    Eski kod bunun ilk iki bileşenini (``yatay[:2]``) yatay hareket sayıyordu;
    bu, vektörü kamera xy-düzlemine dik izdüşürmek demektir ve eğim yönünde
    ``cos(eğim)`` kadar **küçültür**. Kalan hata izotropik değildir: yalnız
    eğim ekseninde büzülme olduğu için uçuş yönüne göre değişir, dolayısıyla
    Procrustes hizalamasının (dik ``M`` + tek ölçek ``c``) soğuramayacağı bir
    sistematik sapma bırakır.

    Ölçülen eğim (video 1): medyan 4,84°, %90 7,24° → %0,2–0,8 ölçek hatası.
    Şartname kamera açısını 70–90° serbest bıraktığı için finalde eğim 20°ye
    kadar çıkabilir; orada aynı hata **%6**'dır.

    Nadir bakışta (n ≈ [0,0,1]) e1 ≈ [1,0,0], e2 ≈ [0,1,0] olur; yani bu düzeltme
    eski davranışın **genelleştirilmiş** hâlidir, nadir limitinde onunla çakışır.
    """
    eksen = np.array([1.0, 0.0, 0.0])
    if abs(float(eksen @ n)) > 0.9:        # aşırı eğimde x ekseni düzleme dik kalır
        eksen = np.array([0.0, 1.0, 0.0])
    e1 = eksen - float(eksen @ n) * n
    nrm = float(np.linalg.norm(e1))
    if nrm < 1e-9:
        return np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    e1 = e1 / nrm
    return e1, np.cross(n, e1)


def duzlem_ici_xy(delta: np.ndarray, n: np.ndarray, duzlem_tabanli: bool):
    """Düzleme paralel bileşeni 2B düzlem koordinatına indir.

    ``duzlem_tabanli=False`` eski (``yatay[:2]``) davranışı korur — A/B için.
    """
    dz_unit = float(n @ delta)
    yatay = delta - dz_unit * n
    if not duzlem_tabanli:
        return yatay[:2], dz_unit
    e1, e2 = duzlem_tabani(n)
    return np.array([float(yatay @ e1), float(yatay @ e2)]), dz_unit


# --------------------------------------------------------------------------
# 2) Durum makinesi
# --------------------------------------------------------------------------
@dataclass
class EstimatorConfig:
    """Varsayılanlar, Aşama 4 taraması + taze-senaryo doğrulamasıyla seçildi:
    ortalama hatada fark yok (sistem parametrelere karşı gürbüz) ama
    calib_window=60 + smooth_alpha=0.6 en-kötü-durum hatasını belirgin
    küçültüyor (36→21 m) — yarışmada kuyruk riskine karşı tercih edildi."""

    method: str = "fuzyon"             # "benzerlik" | "homografi" | "fuzyon"
    calib_window: int = 60             # hizalama penceresi (adım; ~8 s)
    min_calib: int = 12                # hizalama için asgari örnek
    min_move_m: float = 0.10           # hizalamaya girecek asgari GT adımı
    # KALİTE KAPISI — 0,40 → 0,30 (kanıta dayalı; bkz. degerlendirme/YONTEM_SINAVI.md).
    # Eski 0,40 değeri ölçümlerin büyük kısmını atıp sistemi **sabit-hız yedeğine**
    # düşürüyordu. Gürültülü bir görsel ölçüm bile serbest sürüklenen sabit-hızdan
    # iyidir (aynı ilke sim↔hom kapısı kaldırılırken de ölçülmüştü).
    #
    # DEĞER SEÇİMİ — 2026 verisi 0,05'i işaret ediyordu, ama BAŞKA YILLARIN
    # uçuşlarında (2024 + 2025, 5 bağımsız yatak) 0,05 aşırı uyum çıktı:
    #   0,05 -> 2024 +2,4 m · 2025-RGB-A +0,5 m      (2026 sınavı −6,3)
    #   0,30 -> 2024 +1,2 m · 2025-RGB-A **−0,7 m**  (2026 sınavı −5,3)
    # 0,30 sınav kazancının neredeyse tamamını koruyup yıllar arası bedeli yarıya
    # indiriyor ve hiçbir yatakta felaket üretmiyor. Bilinmeyen final uçuşu için
    # tercih edilen denge budur.
    min_inlier_ratio: float = 0.30
    min_kept: int = 30                 # kalite kapısı (eşleşme sayısı)
    max_dz_unit: float = 0.15          # kalite kapısı (adım başı olanaksız ölçek)
    max_yaw_step: float = np.radians(10.0)
    cv_alpha: float = 0.30             # sabit-hız EMA katsayısı
    # sağlamlık sınırları
    h_bounds: tuple[float, float] = (5.0, 400.0)   # fiziksel yükseklik aralığı [m]
    max_step_xy: float = 3.5           # adım başı azami yatay hareket [m] (~26 m/s)
    # HİZALAMA KABUL EŞİĞİ — 0,50 → 0,30 (bütünlük denetimi / integrity monitoring).
    # 2026 yataklarında bu kapı hiç tetiklenmiyordu (residual medyanı video1'de 0,040,
    # video2'de 0,115) ve "ölü ayar" görünüyordu. BAŞKA YILLARIN verisinde asıl işlevi
    # ortaya çıktı: 2025 A uçuşunda (yavaş, alçak, %13,5'i durgun) hizalama residual'ı
    # medyan **0,404** — yani kamera→dünya köprüsü GT'ye hiç oturmuyor; öğrenilen ölçek
    # %55 oynuyor. Eski gevşek kapı bu bozuk hizalamayı kabul ediyor, sistem onunla ölü
    # hesap yapıyor ve sonuç **hiçbir şey yapmamaktan kötü** çıkıyordu:
    #     kesinti hatası 51,3 m · 'tut' tabanı 47,2 m · 'sabit_hız' tabanı 42,4 m
    # Kapı sıkılaşınca bozuk hizalama reddedilir, önceki (geçerli) hizalama korunur:
    #     0,50 -> 51,33 m     0,35 -> 28,23     0,25 -> 28,23     0,15 -> 28,16
    # ve **çalışan bütün yataklarda etkisi tam olarak sıfırdır** (video1/video2/sınav/2024
    # değişmiyor). 0,15'te video2 bozulmaya başlıyor (+1,0), 0,10'da v25 de geri kötüleşiyor.
    # 0,30 seçildi: video2'nin p90'ı (0,247) ile v25'in medyanı (0,404) arasında güvenli orta.
    residual_gate: float = 0.30
    smooth_alpha: float = 0.6          # ölçüm EMA katsayısı (1.0 = yumuşatma kapalı)
    # Düzleme paralel bileşeni kamera xy'sine izdüşürmek yerine düzlem içinde
    # ortonormal tabana açar (bkz. `duzlem_tabani`). Nadir bakışta davranış
    # değişmez; eğim büyüdükçe (şartname 70–90°) eski yol cos(eğim) kadar
    # sistematik küçültme yapar.
    # DENENDİ ve REDDEDİLDİ: sınav −0,136 ve video1 −0,007 kazandı ama **video2-termalde
    # +0,326 kaybetti** (6 permütasyon, eşleştirilmiş). Termalde düzlem normali daha az
    # noktayla kestirildiği için tabanın kendisi gürültülü; düzeltme gürültüyü yükseltiyor.
    # Üçüncü (termal) yatak olmasaydı yanlışlıkla uygulanacaktı.
    duzlem_tabanli: bool = False
    # Merdiven ölçümü için AYRI hizalama (eksen matrisi M + metrik ölçek/yükseklik)
    # öğren. Bugün `alignment` yalnız ARDIŞIK ölçümlerden fit ediliyor ama kesintide
    # konum MERDİVEN ölçümünden üretiliyor. İki ölçüm aynı ölçekte değil: 8 karelik
    # tabanda 3B yapı paralaksı devreye girdiği için düzleme uyum düşüyor
    # (homografi inlier medyanı 0,605 vs ardışıkta 0,923) ve ima ettikleri metrik
    # yükseklik farklı çıkıyor (53,536 m vs 53,880 m → %0,64 sistematik ölçek hatası).
    # DENENDİ ve REDDEDİLDİ: video1 −0,231 kazandı (yön 1,57°→1,10°) ama **sınavın
    # 2. penceresi çöktü** (17,19 → 25,97 m). Orada sağlıklı dönem parçalı ve merdiven
    # örnekleri bağımlı (aynı çapadan çıkan örtüşen hop'lar), 4 parametre belirlenemiyor.
    merdiven_hizalama: bool = False
    # Aynı uyuşmazlığın DÜŞÜK SERBESTLİKLİ hâli: eksen matrisi M ardışık zincirden
    # (çok sayıda bağımsız adımla iyi belirlenmiş) aynen alınır, yalnız **skaler
    # ölçek düzeltmesi** k öğrenilir. Gerekçe: `merdiven_hizalama` (4 serbestlik)
    # sınavın 2. penceresinde çöktü (17,19 → 25,97 m) çünkü orada sağlıklı dönem
    # parçalı ve merdiven örnekleri birbirine bağımlı (aynı çapadan çıkan
    # örtüşen hop'lar) — 4 parametre bu veriyle belirlenemiyor. 1 parametre
    # aynı veriyle çok daha kararlı kestirilir.
    # DENENDİ ve REDDEDİLDİ: video1'de en iyi sonucu verdi (3,292 m) ama sınav
    # 10,50 → 12,01 ve video2-termal 12,61 → 15,89 kaybetti. Ölçek düzeltmesi tek
    # uçuşun geometrisine uyum sağlıyor, genellemiyor.
    merdiven_olcek: bool = False
    # Yönü, ölçümdeki İKİNCİ (gevşek eşikli) homografi fitinden al; öteleme/ölçek
    # sıkı fitten gelmeye devam eder. Gerekçe: `PairMeasurement.hom_yaw_gevsek`.
    # DENENDİ ve REDDEDİLDİ: ETKİSİZ (termal, öteleme eşiği 0,5: yön 19,41° vs 19,39°).
    # Sıkı eşikteki yön hatası ölçümden değil KALİTE KAPISINDAN geliyordu — ölçümlerin
    # %30'u (RGB'de %61) `min_inlier_ratio` kapısına takılıp sabit-hıza düşüyordu.
    # Bu teşhis, turun asıl kazancına (`min_inlier_ratio` 0,40 → 0,05) götürdü.
    yon_gevsek: bool = False
    # YÖN KAYNAĞI: "hom" (homografi ayrıştırması) | "sim" (benzerlik kestirimi).
    # Gerekçe — yön ve ölçek hataları bu depoda ÜÇ KEZ bağımsız olarak zıt kestirici
    # tercih etti: (a) RAFT ds0,75 yönü 1,27°→0,65° iyileştirip ölçeği %2,11→%2,53
    # bozdu; (b) termal eşik taramasında eşik 0,5 ölçeği %2,34'e indirip yönü 19,4°'ye
    # çıkardı, 1,5 tersini yaptı; (c) SIFT yönü RAFT'tan iyi (1,30° vs 1,54°) ama
    # ölçeği kötü (%3,07 vs %2,42) verdi. Ötelemeyi/ölçeği homografide bırakıp yönü
    # ayrı bir kestiriciden almak bu ayrışmayı ilk kez doğrudan istismar eder.
    # Neden benzerlik: 4 serbestlikli benzerlik düzlemsel nadir sahnede daha iyi
    # koşullanmış — düzlem normali seçimi ve 4-aday belirsizliği yok. Ve `sim_yaw`
    # zaten her ölçümde var, ek hesap gerektirmez.
    # Uzlaşım doğrulandı: sim_yaw ↔ hom_yaw korelasyon 0,995–0,9996, eğim ~1,00,
    # medyan |fark| 0,02–0,07° (aynı işaret ve ölçek).
    #
    # DENENDİ ve REDDEDİLDİ (8 permütasyon, eşleştirilmiş):
    #   video1 3,885 → 18,310 m (+14,4) · sınav 12,115 → 15,065 · termal 14,99 → 16,97
    #   video1 yön hatası 1,69° → 9,91°
    # Neden: adım başına fark medyanda yalnız 0,034° ama **sistematik** (regresyon
    # eğimi 1,007, yani benzerlik yaw'ı ~%0,7 büyük). 600 karelik bir kesintide bu
    # ~20°'ye birikir. Yani homografinin dönmesi zaten doğru; **zayıf halka yön değil**.
    # Bu sonuç "yönü başka bir kestiriciden al" ailesinin önselini de zayıflatır
    # (esas matris dâhil): sorun kestiricinin koşullanması değil.
    # Varsayılan "hom" — kayıt olarak bırakıldı.
    yon_kaynak: str = "hom"
    # Merdiven kestirimini, ardışık zincir sonucuyla **inlier oranına göre harmanla**.
    # Gerekçe: `min_inlier_ratio` 0,40 → 0,05 net kazandırdı ama sınavın 1. penceresini
    # bir miktar kötüleştirdi (16,17 → 17,97) — orada gevşek kapıdan geçen zayıf
    # ölçümler zarar veriyor. Sert kapı (al/at) yerine sürekli ağırlık:
    #     w = clip(inlier / merdiven_harman_ref, 0, 1)
    #     pred = w · merdiven + (1−w) · ardışık
    # 0 = kapalı (sert kapı davranışı).
    merdiven_harman_ref: float = 0.0
    # --- Hampel tanımlayıcı (medyan + MAD) ile zamansal aykırı eleme ---
    # Kaynak alan: **endüstriyel ölçüm / sinyal işleme** (Hampel identifier; MATLAB
    # `hampel`, süreç izleme ve sensör temizlemede standart). Neden burada:
    # `min_inlier_ratio` 0,40 → 0,05 net kazandırdı ama artık zayıf ölçümler de
    # içeri giriyor; mevcut EMA (`smooth_alpha`) **doğrusal** bir filtre olduğu için
    # bir sıçramayı elemez, komşu karelere yayar. Hampel, kayan pencerede medyandan
    # `k · 1,4826 · MAD` kadar sapan örneği aykırı sayıp **medyanla değiştirir** —
    # gerçek manevrayı (medyan onunla birlikte kayar) bozmadan sıçramayı siler.
    # 0 = kapalı.
    hampel_pencere: int = 0
    hampel_k: float = 3.0
    # --- Yükseklik integratörü için anti-windup (kaynak alan: KONTROL MÜHENDİSLİĞİ) ---
    # Teşhis: kesintide `height` çarpımsal olarak taşınıyor (`h *= 1 - dz_unit`) ve
    # geri besleme olmadığı için sürükleniyor. Ölçüldü (yeni varsayılanlarla,
    # kestirilen/gerçek yükseklik oranı):
    #     video1-RGB    ort |hata| %7,6   · blok 1: 1,00 → 1,13 · blok 3: 1,00 → 0,89
    #     video2-TERMAL ort |hata| %25,8  · blok 3: 1,00 → **0,11** (yükseklik çöküyor)
    # `dxy = M·e0·height` olduğu için bu doğrudan yatay ölçek hatasına dönüşüyor —
    # kalan hatanın baskın kaynağı budur.
    #
    # İki klasik anti-windup aracı (ikisi de kesinti boyunca, çapa yüksekliğine göre):
    #   `h_bant`     — integratörü fiziksel banda kelepçele: h ∈ h0·[1−bant, 1+bant]
    #   `h_sizinti`  — sızıntılı integratör: her adım h'yi h0'a doğru bu oranda çek
    # Bu, REDDEDİLEN `height_dz_katsayi` sönümlemesinden farklıdır: o, artışı
    # (gerçek irtifa değişimini) kısıyordu; bunlar durumu sınırlıyor, değişimi değil.
    h_bant: float = 0.0
    h_sizinti: float = 0.0
    # --- YALNIZ MERDİVEN ölçümü için GNC kesme yarıçapı [px] ---
    # Kaynak alan: robust uzaysal algılama (Yang & Carlone, arXiv:1909.08605).
    # Neden yalnız merdiven: ardışık çiftlerde homografi inlier medyanı 0,86 —
    # düzlem modeli zaten oturuyor, ek robustluk yalnız yanlılık ekliyor. Merdiven
    # çiftlerinde ise 0,54: 8 karelik tabanda bina/ağaç paralaksı devreye giriyor,
    # RANSAC birbirine yakın skorlu düzlemler arasında **nokta sırasına göre**
    # seçim yapıyor. GNC dışbükeye yakın bir vekilden başlayıp sert kesmeye
    # tavlayarak bu seçimi belirlenimli hâle getiriyor.
    # DENENDİ, UYGULANDI, sonra BAŞKA YILLARIN VERİSİYLE GERİ ALINDI.
    # 2026'nın üç yatağında güçlü görünüyordu (8 permütasyon, eşleştirilmiş):
    #   sınav 10,92 -> 8,79 (her iki pencere de kazandı) · termal 10,49 -> 9,34
    #   video1 3,60 -> 3,73 · ölçek hatası video1 %2,55->%1,55, termal %6,33->%2,77
    # Ancak 2024+2025'in 5 bağımsız yatağında GENELLEMEDİ ve bir yatakta
    # **felaket** üretti:
    #   2025-RGB-B (4K): 12,80 -> 16,83 m tek başına, tam reçetede 21,19 m (+8,4)
    #   ortalama etki +0,75 m, 5 yatağın yalnız 2'sinde iyi
    # Bilinmeyen bir final uçuşunda böyle bir kuyruk riski kabul edilemez.
    # Kod ve gerekçe kayıt olarak duruyor; varsayılan KAPALI.
    # Ayrıntı: degerlendirme/YONTEM_SINAVI.md
    merdiven_gnc_c: Optional[float] = None
    # --- kesintide sürüklenme sönümleme: DENENDİ ve REDDEDİLDİ (varsayılan = kapalı) ---
    # Teşhis (gorev2/degerlendirme/): kesinti hatası rastgele değil sistematik —
    # tahmin yörüngesi GT'nin ~2° döndürülmüş, %3–4,5 ölçeklenmiş hâli; bu iki bileşen
    # bilinseydi hata 6,3 m yerine 1,5 m olurdu. Doğal fikir: `height *= (1-dz_unit)`
    # ve `psi += yaw` güncellemelerini sönümleyip birikimi yavaşlatmak.
    # SONUÇ (surukleme_tarama.py): sönümleme hatayı BÜYÜTÜYOR —
    #   yaw_katsayi=0 → 162,9 m,  height_dz_katsayi=0 → 13,6 m  (temel 4,34 m).
    # Çünkü araç gerçekten dönüyor ve alçalıyor; güncellemeler doğru ve gereklidir.
    # Sistematik sapma birikimden değil, ölçümün kendi yanlılığından geliyor.
    # Parametreler (varsayılanları davranışı DEĞİŞTİRMEZ) ileride yeniden denemek
    # isteyenler için bırakıldı.
    height_dz_katsayi: float = 1.0     # 0 = yükseklik kesintide sabit, 1 = mevcut davranış
    yaw_katsayi: float = 1.0           # 0 = yön kesintide sabit, 1 = mevcut davranış
    yaw_olu_bant: float = 0.0          # bu değerin altındaki |yaw| gürültü sayılır [rad]
    # NOT: "Düşey kanalı benzerlik ölçeğinden alma" (`z_sim_agirlik`) denendi ve
    # GERİ ALINDI. Örnek veri setinde (THYZ_2026_Ornek_Veri_1) ortalama hatayı
    # 3,767 → 2,803 m düşürüyordu; ancak PUANLI OTURUMUN kendi karelerinde ölçülünce
    # iki bağımsız pencerede de KAYBETTİ (16,04 → 20,07 m ve 18,46 → 18,62 m; zarar
    # tamamen z ekseninde: 10,25 → 16,91 m). Örnek veri seti = test oturumu videosu
    # olduğu için kazanç tek uçuşa aşırı uyumdu. Ayrıntı: degerlendirme/DUSEY_KANAL.md.
    #
    # NOT: Bir "sim↔hom uyuşmazlık kapısı" denendi ve GERİ ALINDI (defter 07 B).
    # Tez: uyuşmazlık yüksekken ölçüme güvenme, sabit-hıza düş. GT'li zor
    # verilerde (Mid-Air winter/foggy) test edilince kapı tetiklendiği 7 vakanın
    # 4'ünde E'yi BOZDU (net +88 m); düşük dokulu sahnede gürültülü ölçüm bile
    # serbest sürüklenen CV'den iyi çıktı. Kaldırıldı.
    # --- z işareti kanıt kapısı (bkz. scale_alignment.fit_alignment) ---
    # `z_sign` tek bittir ama yanlışsa düşey kanal komple ters döner. Eski kapı
    # ölçümün kendi gürültüsünü sinyal sanıyordu ve hiç kapanmıyordu; işaret
    # yataklarda %26–40 oranında kararsızdı. Bu iki eşik, kararın ancak pencerede
    # GERÇEK irtifa hareketi ve tutarlı kanıt varken verilmesini sağlar.
    # Aksi hâlde önceki işaret korunur (başlangıç +1 = şartnamenin z-aşağı yönü).
    z_min_hareket: float = 2.0     # penceredeki GT irtifa gezinmesi alt sınırı [m]
    z_min_kanit: float = 0.30      # işaret tutarlılığı |Σab|/Σ|ab| alt sınırı [0-1]


class VoPositionEstimator:
    """Yarışma akışını birebir izleyen pozisyon kestirim durum makinesi."""

    def __init__(self, profile: CameraProfile, config: Optional[EstimatorConfig] = None):
        self.profile = profile
        self.cfg = config or EstimatorConfig()

        self.psi = 0.0                       # birikimli yön (başlangıç ekseni = 0)
        self.pos = np.zeros(3)               # kestirilen konum (dünya, m)
        self.height: Optional[float] = None  # yerden yükseklik durumu
        self.alignment: Optional[Alignment] = None
        self.z_sign = 1.0

        n = self.cfg.calib_window
        self._buf_E = deque(maxlen=n)        # ψ=0 çerçevesinde birim deltalar
        self._buf_G = deque(maxlen=n)        # GT yatay deltalar
        self._buf_dzu = deque(maxlen=n)
        self._buf_gdz = deque(maxlen=n)
        self._buf_gz = deque(maxlen=n)

        self._hampel = deque(maxlen=max(self.cfg.hampel_pencere, 1))
        self._h0: Optional[float] = None     # kesinti başındaki yükseklik (anti-windup referansı)
        self._vel = np.zeros(3)              # sabit-hız yedeği (dünya, m/adım)
        self._have_vel = False
        self._last_gt: Optional[np.ndarray] = None
        self._prev_healthy = True            # bir önceki karenin sağlık durumu
        self._smooth = None                  # ölçüm yumuşatma durumu (e0, dz_unit)

        # tanılama sayaçları
        self.n_steps = 0
        self.n_invalid = 0
        self.n_frozen = 0

    # ---------------- iç yardımcılar ----------------
    def _sim_candidate(self, meas: PairMeasurement):
        cfg = self.cfg
        if not meas.sim_ok or meas.sim_inlier_ratio < cfg.min_inlier_ratio:
            return None
        e_cam = np.array([meas.sim_unit_dx, meas.sim_unit_dy])
        return e_cam, meas.sim_dz_unit, meas.sim_yaw, "cam2"   # adım-sonrası eksen

    def _hom_candidate(self, meas: PairMeasurement):
        cfg = self.cfg
        if not meas.hom_ok or meas.hom_inlier_ratio < cfg.min_inlier_ratio:
            return None
        delta = np.array([meas.hom_ux, meas.hom_uy, meas.hom_uz])
        xy, dz_unit = duzlem_ici_xy(delta, meas.hom_normal, cfg.duzlem_tabanli)
        return xy, dz_unit, self._yaw_sec(meas), "cam1"        # adım-öncesi eksen

    def _yaw_sec(self, meas: PairMeasurement) -> float:
        """Yön kaynağını seç (tek karar noktası; hem ardışık hem merdiven buradan geçer).

        `yon_kaynak="sim"` iken benzerlik yaw'ı kullanılır; ölçüm kendi kalite
        kapısından geçemezse homografiye düşülür — böylece bayrak asla ölçüm
        kaybettirmez.
        """
        if self.cfg.yon_kaynak == "sim":
            if (meas.sim_ok and np.isfinite(meas.sim_yaw)
                    and meas.sim_inlier_ratio >= self.cfg.min_inlier_ratio):
                return float(meas.sim_yaw)
        if self.cfg.yon_gevsek and np.isfinite(meas.hom_yaw_gevsek):
            return float(meas.hom_yaw_gevsek)
        return meas.hom_yaw

    def _extract(self, meas: Optional[PairMeasurement]):
        """Ölçümden yönteme uyan geçerli adayları çıkar (kalite kapılı).

        "fuzyon" yönteminde iki aday da toplanır ve ortalanır — bağımsız
        gürültüleri kısmen sönümler, biri kapıya takılırsa diğeri taşır.
        """
        cfg = self.cfg
        if meas is None or meas.frozen or not meas.ok or meas.n_kept < cfg.min_kept:
            return None
        if cfg.method == "benzerlik":
            adaylar = [self._sim_candidate(meas)]
        elif cfg.method == "homografi":
            adaylar = [self._hom_candidate(meas)]
        else:  # fuzyon
            adaylar = [self._sim_candidate(meas), self._hom_candidate(meas)]
        gecerli = []
        for aday in adaylar:
            if aday is None:
                continue
            e_cam, dz_unit, yaw, frame = aday
            if not np.isfinite(dz_unit) or abs(dz_unit) > cfg.max_dz_unit:
                continue
            if not np.isfinite(yaw) or abs(yaw) > cfg.max_yaw_step:
                continue
            gecerli.append((e_cam, dz_unit, yaw, frame))
        return gecerli or None

    def _integrate_candidates(self, adaylar) -> tuple[np.ndarray, float]:
        """Adayları ψ=0 çerçevesine döndürüp ortala; ψ'yi bir kez ilerlet.

        Not: yaw'ı GT hız yönüyle "düzeltme" fikri denendi ve GERİ ALINDI —
        drone yön değiştirmeden yana kayabildiği (crab) için GT hız yönü
        değişimi kamera yaw'ı ile aynı şey değildir; düzeltme sistematik hata
        ekliyordu.
        """
        yaw = self._yaw_isle(float(np.mean([a[2] for a in adaylar])))
        psi_eski = self.psi
        self.psi += yaw
        e0_list, dz_list = [], []
        for e_cam, dz_unit, _, frame in adaylar:
            psi = self.psi if frame == "cam2" else psi_eski
            e0_list.append(rot2d(psi) @ e_cam)
            dz_list.append(dz_unit)
        return np.mean(e0_list, axis=0), float(np.mean(dz_list))

    def _hampel_uygula(self, e0: np.ndarray, dz_unit: float):
        """Hampel tanımlayıcı: medyandan k·1,4826·MAD'dan fazla sapanı medyanla değiştir.

        Pencere (e0x, e0y, dz_unit) üçlülerini tutar; her bileşen ayrı test edilir.
        Pencere dolmadan (n < 5) müdahale edilmez — MAD anlamsız olur.
        """
        w = self.cfg.hampel_pencere
        if w <= 0:
            return e0, dz_unit
        ornek = np.array([float(e0[0]), float(e0[1]), float(dz_unit)])
        if len(self._hampel) >= 5:
            gecmis = np.array(self._hampel)                 # (n, 3)
            med = np.median(gecmis, axis=0)
            mad = np.median(np.abs(gecmis - med), axis=0) * 1.4826
            esik = self.cfg.hampel_k * mad
            # MAD sıfıra çökerse (sabit pencere) eleme yapma
            aykiri = (mad > 1e-12) & (np.abs(ornek - med) > esik)
            ornek = np.where(aykiri, med, ornek)
        self._hampel.append(ornek.copy())
        return ornek[:2], float(ornek[2])

    def _clamp_h(self, h: float) -> float:
        lo, hi = self.cfg.h_bounds
        return float(np.clip(h, lo, hi))

    def _yaw_isle(self, yaw: float) -> float:
        """Ölü bant + sönümleme uygula (kesintide yön sürüklenmesini sınırlar).

        Varsayılan ayarlarla (olu_bant=0, katsayi=1) davranış değişmez.
        """
        if abs(yaw) < self.cfg.yaw_olu_bant:
            return 0.0
        return yaw * self.cfg.yaw_katsayi

    def _h_guncelle(self, h: float, dz_unit: float) -> float:
        """Yükseklik durumunu ilerlet; katsayı 0 ise yükseklik sabit kalır.

        `h_sizinti`/`h_bant` verilmişse integratör, kesinti başındaki yüksekliğe
        (`self._h0`) göre sınırlanır — bkz. EstimatorConfig anti-windup notu.
        """
        k = self.cfg.height_dz_katsayi
        h = h * max(1.0 - k * dz_unit, 0.5)
        h0 = self._h0
        if h0 is not None and h0 > 0:
            if self.cfg.h_sizinti > 0.0:
                lam = self.cfg.h_sizinti
                h = (1.0 - lam) * h + lam * h0
            if self.cfg.h_bant > 0.0:
                b = self.cfg.h_bant
                h = float(np.clip(h, h0 * (1.0 - b), h0 * (1.0 + b)))
        return self._clamp_h(h)

    # ---------------- ana döngü ----------------
    def update(
        self,
        meas: Optional[PairMeasurement],
        gt_xyz: Optional[np.ndarray],
        healthy: bool,
    ) -> np.ndarray:
        """Bir kare ilerle: sağlıklıysa GT'yi aynala + öğren, değilse kestir."""
        self.n_steps += 1

        if healthy and not self._prev_healthy:
            # İyileşme anı: kesinti boyunca ψ görsel yaw'la (gürültülü) ilerledi;
            # kesinti öncesi kalibrasyon örnekleri artık ψ ile tutarsız olabilir.
            # Tamponlar temizlenir, taze pencereyle yeniden öğrenilir; yeterli
            # veri gelene dek eski hizalama korunur (yükseklik GT z'den yürür).
            self._buf_E.clear(); self._buf_G.clear()
            self._buf_dzu.clear(); self._buf_gdz.clear(); self._buf_gz.clear()

        cikarim = self._extract(meas)
        if cikarim is None:
            self.n_invalid += 1
            if meas is not None and meas.frozen:
                self.n_frozen += 1
            e0 = dz_unit = None
            self._smooth = None            # ölçüm zinciri koptu: yumuşatmayı sıfırla
        else:
            e0, dz_unit = self._integrate_candidates(cikarim)
            e0, dz_unit = self._hampel_uygula(e0, dz_unit)
            # isteğe bağlı zamansal yumuşatma (EMA) — gürültüyü sönümler,
            # karşılığında ani manevralarda küçük gecikme getirir
            a = self.cfg.smooth_alpha
            if a < 1.0:
                if self._smooth is not None:
                    e0 = a * e0 + (1 - a) * self._smooth[0]
                    dz_unit = a * dz_unit + (1 - a) * self._smooth[1]
                self._smooth = (e0, dz_unit)

        if healthy:
            gt_xyz = np.asarray(gt_xyz, dtype=float)
            # --- öğrenme (yalnız önceki kare de sağlıklıysa: GT adımı gerçek bir adım)
            if self._prev_healthy and self._last_gt is not None:
                g = gt_xyz[:2] - self._last_gt[:2]
                g_norm = float(np.hypot(*g))
                if e0 is not None and g_norm >= self.cfg.min_move_m:
                    self._buf_E.append(e0)
                    self._buf_G.append(g)
                    self._buf_dzu.append(dz_unit)
                    self._buf_gdz.append(gt_xyz[2] - self._last_gt[2])
                    self._buf_gz.append(gt_xyz[2])
                if len(self._buf_E) >= self.cfg.min_calib:
                    fit = fit_alignment(
                        np.array(self._buf_E), np.array(self._buf_G),
                        np.array(self._buf_dzu), np.array(self._buf_gdz),
                        np.array(self._buf_gz), prev_z_sign=self.z_sign,
                        z_min_hareket=self.cfg.z_min_hareket,
                        z_min_kanit=self.cfg.z_min_kanit,
                    )
                    if fit is not None and fit.residual_ratio < self.cfg.residual_gate:
                        self.alignment = fit
                        self.z_sign = fit.z_sign
                # sabit-hız yedeği GT'den beslenir (en güvenilir kaynak)
                adim = gt_xyz - self._last_gt
                self._vel = (1 - self.cfg.cv_alpha) * self._vel + self.cfg.cv_alpha * adim
                self._have_vel = True
            # --- durum çapalama
            if self.alignment is not None:
                self.height = self._clamp_h(self.alignment.current_height(float(gt_xyz[2])))
                self._h0 = self.height       # sağlıklı iken referans tazelenir
            self.pos = gt_xyz.copy()
            self._last_gt = gt_xyz
            self._prev_healthy = True
            return self.pos.copy()

        # ---------------- kesinti: dead reckoning ----------------
        self._prev_healthy = False
        delta = None
        if e0 is not None and self.alignment is not None and self.height is not None:
            dxy = self.alignment.M @ e0 * self.height
            if float(np.hypot(*dxy)) <= self.cfg.max_step_xy:   # fizik kapısı
                dz = self.z_sign * dz_unit * self.height
                delta = np.array([dxy[0], dxy[1], dz])
                self.height = self._h_guncelle(self.height, dz_unit)
                self._vel = (1 - self.cfg.cv_alpha) * self._vel + self.cfg.cv_alpha * delta
                self._have_vel = True
        if delta is None:
            self.n_invalid += 1 if cikarim is not None else 0
            delta = self._vel.copy() if self._have_vel else np.zeros(3)
        self.pos = self.pos + delta
        return self.pos.copy()


class VoLadderEstimator(VoPositionEstimator):
    """Keyframe (anahtar kare) merdiveniyle dead-reckoning.

    Kavram: ardışık zincirde her adımın hatası birikir (2250 halka). Merdivende
    ise konum, en fazla ``spacing`` adım geride duran bir **keyframe'e göre tek
    homografiyle** kestirilir; hata yalnızca keyframe atlamalarında birikir
    (halka sayısı ``spacing`` kat azalır) ve uzun taban çizgisi sinyal/gürültü
    oranını büyütür. Sağlıklı dönemde keyframe çapası doğrudan GT'ye oturur.

    Öğrenme (hizalama/ölçek/hız) ve geçersiz-ölçüm yedeği üst sınıftaki ardışık
    zincirden aynen devralınır; merdiven yalnızca kesintideki konum kestirimini
    değiştirir. Merdiven ölçümü kalite kapısına takılırsa o karede ardışık
    zincir sonucu kullanılır (kesintisiz düşüş).
    """

    def __init__(self, profile, config: Optional[EstimatorConfig] = None, spacing: int = 8):
        super().__init__(profile, config)
        self.spacing = spacing
        self._anchor: Optional[dict] = None   # ord, pos, h, psi, saglikli

        # Merdiven ölçümüne ait ayrı hizalama tamponları (bkz. `merdiven_hizalama`).
        n = self.cfg.calib_window
        self.alignment_ladder: Optional[Alignment] = None
        self._lbuf_E = deque(maxlen=n)
        self._lbuf_G = deque(maxlen=n)
        self._lbuf_dzu = deque(maxlen=n)
        self._lbuf_gdz = deque(maxlen=n)
        self._lbuf_gz = deque(maxlen=n)
        self._prev_healthy_ladder = True
        self.merdiven_k = 1.0                 # skaler ölçek düzeltmesi
        self._kbuf_p = deque(maxlen=n)        # ardışık hizalamayla kestirilen |dxy|
        self._kbuf_g = deque(maxlen=n)        # aynı adımın GT |dxy|'si

    def _merdiven_ogren(self, k_ord: int, meas_ladder, gt_xyz) -> None:
        """Sağlıklı dönemde keyframe→kare çiftinden merdiven hizalamasını öğren.

        Çapa sağlıklı dönemde doğrudan GT'ye oturduğu için ``gt - çapa`` gerçek bir
        hop yer değiştirmesidir; ardışık zincirdekiyle **aynı** en küçük kareler
        problemi, ama merdivenin kendi ölçüm istatistiğiyle çözülür.
        """
        a = self._anchor
        if a is None or not a.get("saglikli"):
            return
        hop = k_ord - a["ord"]
        if hop < 1:
            return
        cand = self._ladder_candidate(meas_ladder, hop)
        if cand is None:
            return
        e_cam, dz_unit, _ = cand
        gt_xyz = np.asarray(gt_xyz, dtype=float)
        g = gt_xyz[:2] - a["pos"][:2]
        e0 = rot2d(a["psi"]) @ e_cam

        # --- 1 serbestlikli yol: ardışık hizalamayla kestir, GT ile oranla ---
        if (self.cfg.merdiven_olcek and self.alignment is not None
                and a["h"] is not None and float(np.hypot(*g)) >= self.cfg.min_move_m):
            p = self.alignment.M @ e0 * a["h"]
            if np.all(np.isfinite(p)) and float(np.hypot(*p)) > 1e-6:
                self._kbuf_p.append(p)
                self._kbuf_g.append(g)
            if len(self._kbuf_p) >= self.cfg.min_calib:
                P = np.array(self._kbuf_p); G = np.array(self._kbuf_g)
                pay = float((G * P).sum())
                payda = float((P * P).sum())
                if payda > 1e-9:
                    k = pay / payda
                    if 0.5 < k < 2.0:          # fiziksel makullük kapısı
                        self.merdiven_k = k

        if not self.cfg.merdiven_hizalama:
            return
        if float(np.hypot(*g)) >= self.cfg.min_move_m:
            self._lbuf_E.append(e0)
            self._lbuf_G.append(g)
            self._lbuf_dzu.append(dz_unit)
            self._lbuf_gdz.append(float(gt_xyz[2] - a["pos"][2]))
            self._lbuf_gz.append(float(gt_xyz[2]))
        if len(self._lbuf_E) >= self.cfg.min_calib:
            fit = fit_alignment(
                np.array(self._lbuf_E), np.array(self._lbuf_G),
                np.array(self._lbuf_dzu), np.array(self._lbuf_gdz),
                np.array(self._lbuf_gz), prev_z_sign=self.z_sign,
                z_min_hareket=self.cfg.z_min_hareket,
                z_min_kanit=self.cfg.z_min_kanit,
            )
            if fit is not None and fit.residual_ratio < self.cfg.residual_gate:
                self.alignment_ladder = fit

    def _merdiven_hizalama(self) -> Optional[Alignment]:
        if self.cfg.merdiven_hizalama and self.alignment_ladder is not None:
            return self.alignment_ladder
        return self.alignment

    def _ladder_candidate(self, meas: Optional[PairMeasurement], hop: int):
        """Keyframe→şimdi homografi ölçümü; kapılar atlama uzunluğuyla ölçeklenir."""
        cfg = self.cfg
        if meas is None or meas.frozen or not meas.ok or meas.n_kept < cfg.min_kept:
            return None
        if not meas.hom_ok or meas.hom_inlier_ratio < cfg.min_inlier_ratio:
            return None
        delta = np.array([meas.hom_ux, meas.hom_uy, meas.hom_uz])
        xy, dz_unit = duzlem_ici_xy(delta, meas.hom_normal, cfg.duzlem_tabanli)
        if not np.isfinite(dz_unit) or abs(dz_unit) > min(0.6, cfg.max_dz_unit * hop):
            return None
        yaw = self._yaw_sec(meas)
        if not np.isfinite(yaw) or abs(yaw) > cfg.max_yaw_step * hop:
            return None
        # Düşey bileşen de saf homografi ayrıştırmasından gelir. Benzerlik ölçeğinden
        # alma denemesi ölçülüp GERİ ALINDI — gerekçe EstimatorConfig'te.
        return xy, dz_unit, float(yaw)

    def update_ladder(
        self,
        k_ord: int,
        meas_cons: Optional[PairMeasurement],
        meas_ladder: Optional[PairMeasurement],
        gt_xyz: Optional[np.ndarray],
        healthy: bool,
    ) -> np.ndarray:
        # önce ardışık akış: öğrenme, ψ, hız, sağlıklı aynalama + yedek kestirim
        pred = self.update(meas_cons, gt_xyz, healthy)

        if healthy:
            if self.cfg.merdiven_hizalama or self.cfg.merdiven_olcek:
                if not self._prev_healthy_ladder:      # iyileşme anı: ψ tutarsız olabilir
                    self._lbuf_E.clear(); self._lbuf_G.clear()
                    self._lbuf_dzu.clear(); self._lbuf_gdz.clear(); self._lbuf_gz.clear()
                    self._kbuf_p.clear(); self._kbuf_g.clear()
                self._merdiven_ogren(k_ord, meas_ladder, gt_xyz)
            self._prev_healthy_ladder = True
            if k_ord % self.spacing == 0:
                h_capa = self.height
                if self.cfg.merdiven_hizalama and self.alignment_ladder is not None:
                    h_capa = self._clamp_h(
                        self.alignment_ladder.current_height(float(np.asarray(gt_xyz)[2])))
                self._anchor = dict(ord=k_ord, pos=np.asarray(gt_xyz, float).copy(),
                                    h=h_capa, psi=self.psi, saglikli=True)
            return pred

        self._prev_healthy_ladder = False
        a = self._anchor
        hiz = self._merdiven_hizalama()
        if a is not None and a["h"] is not None and hiz is not None:
            hop = k_ord - a["ord"]
            cand = self._ladder_candidate(meas_ladder, max(hop, 1)) if hop >= 1 else None
            if cand is not None:
                e_cam, dz_unit, yaw = cand
                e0 = rot2d(a["psi"]) @ e_cam          # homografi: keyframe (adım öncesi) ekseni
                dxy = hiz.M @ e0 * a["h"]
                if self.cfg.merdiven_olcek:
                    dxy = dxy * self.merdiven_k
                if float(np.hypot(*dxy)) <= self.cfg.max_step_xy * hop:
                    dz = self.z_sign * dz_unit * a["h"]
                    pred = a["pos"] + np.array([dxy[0], dxy[1], dz])
                    ref = self.cfg.merdiven_harman_ref
                    if ref > 0.0 and meas_ladder is not None:
                        # zayıf ölçümü tamamen atmak yerine ardışık zincire doğru çek
                        w = float(np.clip(meas_ladder.hom_inlier_ratio / ref, 0.0, 1.0))
                        pred = w * pred + (1.0 - w) * self.pos
                    self.pos = pred.copy()            # ardışık durumu senkron tut
                    if k_ord % self.spacing == 0:     # yeni keyframe: çapayı taşı
                        self._anchor = dict(
                            ord=k_ord, pos=pred.copy(),
                            h=self._h_guncelle(a["h"], dz_unit),
                            psi=a["psi"] + self._yaw_isle(yaw), saglikli=False)
                    return pred.copy()
        # merdiven kullanılamadı: ardışık sonuçla devam; keyframe sınırındaysa çapayı yine taşı
        if k_ord % self.spacing == 0:
            self._anchor = dict(ord=k_ord, pos=self.pos.copy(), h=self.height,
                                psi=self.psi, saglikli=False)
        return pred
