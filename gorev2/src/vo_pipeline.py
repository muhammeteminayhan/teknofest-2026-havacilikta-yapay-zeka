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

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from . import geometry
from .calibration import CameraProfile
from .matching import FeatureTracker
from .scale_alignment import Alignment, fit_alignment


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

    @classmethod
    def compute(
        cls,
        gray0: np.ndarray,
        gray1: np.ndarray,
        profile: CameraProfile,
        tracker: FeatureTracker,
        prev_normal: Optional[np.ndarray] = None,
        freeze_diff_thresh: float = 0.35,
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
                               n_kept=mr.n_kept, taban=m)

    @classmethod
    def from_points(
        cls,
        pts0: np.ndarray,
        pts1: np.ndarray,
        profile: CameraProfile,
        prev_normal: Optional[np.ndarray] = None,
        n_kept: Optional[int] = None,
        taban: Optional["PairMeasurement"] = None,
        **geo_kwargs,
    ) -> "PairMeasurement":
        """Hazır nokta eşleşmelerinden ölçüm üret (eşleştirme adımı atlanır).

        `compute()` bunun üstüne kuruludur; ayrıca **nokta önbelleği** ile geometri
        denemelerini (MAGSAC++ vb.) video yeniden taramadan koşturmayı sağlar.
        `geo_kwargs` doğrudan `geometry.*` çağrılarına geçer (`robust_method`,
        `ransac_thresh`, ...).
        """
        m = taban if taban is not None else cls()
        m.n_kept = int(n_kept if n_kept is not None else len(pts0))

        f = profile.fx
        pp = (profile.cx, profile.cy)
        sim = geometry.estimate_similarity(pts0, pts1, ref_point=pp, **geo_kwargs)
        if sim is not None:
            delta_unit = sim.camera_delta(1.0, f)      # h=1 için
            m.sim_ok = True
            m.sim_unit_dx = float(delta_unit[0])
            m.sim_unit_dy = float(delta_unit[1])
            m.sim_dz_unit = float(delta_unit[2])
            m.sim_yaw = float(sim.yaw_delta_rad)
            m.sim_inlier_ratio = float(sim.inlier_ratio)

        hom = geometry.estimate_homography_motion(
            pts0, pts1, profile.K, prev_normal=prev_normal, **geo_kwargs
        )
        if hom is not None:
            m.hom_ok = True
            m.hom_ux, m.hom_uy, m.hom_uz = (float(v) for v in hom.cam_delta_over_d)
            m.hom_nx, m.hom_ny, m.hom_nz = (float(v) for v in hom.n)
            m.hom_yaw = float(hom.yaw_delta_rad)
            m.hom_inlier_ratio = float(hom.inlier_ratio)

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
    min_inlier_ratio: float = 0.40     # kalite kapısı
    min_kept: int = 30                 # kalite kapısı (eşleşme sayısı)
    max_dz_unit: float = 0.15          # kalite kapısı (adım başı olanaksız ölçek)
    max_yaw_step: float = np.radians(10.0)
    cv_alpha: float = 0.30             # sabit-hız EMA katsayısı
    # sağlamlık sınırları
    h_bounds: tuple[float, float] = (5.0, 400.0)   # fiziksel yükseklik aralığı [m]
    max_step_xy: float = 3.5           # adım başı azami yatay hareket [m] (~26 m/s)
    residual_gate: float = 0.5         # hizalama kabul eşiği (kalan hata oranı)
    smooth_alpha: float = 0.6          # ölçüm EMA katsayısı (1.0 = yumuşatma kapalı)
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
    # NOT: Bir "sim↔hom uyuşmazlık kapısı" denendi ve GERİ ALINDI (defter 07 B).
    # Tez: uyuşmazlık yüksekken ölçüme güvenme, sabit-hıza düş. GT'li zor
    # verilerde (Mid-Air winter/foggy) test edilince kapı tetiklendiği 7 vakanın
    # 4'ünde E'yi BOZDU (net +88 m); düşük dokulu sahnede gürültülü ölçüm bile
    # serbest sürüklenen CV'den iyi çıktı. Kaldırıldı.


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
        n = meas.hom_normal
        dz_unit = float(n @ delta)
        yatay = delta - dz_unit * n            # düzleme paralel bileşen
        return yatay[:2], dz_unit, meas.hom_yaw, "cam1"        # adım-öncesi eksen

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
        """Yükseklik durumunu ilerlet; katsayı 0 ise yükseklik sabit kalır."""
        k = self.cfg.height_dz_katsayi
        return self._clamp_h(h * max(1.0 - k * dz_unit, 0.5))

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
        self._anchor: Optional[dict] = None   # ord, pos, h, psi

    def _ladder_candidate(self, meas: Optional[PairMeasurement], hop: int):
        """Keyframe→şimdi homografi ölçümü; kapılar atlama uzunluğuyla ölçeklenir."""
        cfg = self.cfg
        if meas is None or meas.frozen or not meas.ok or meas.n_kept < cfg.min_kept:
            return None
        if not meas.hom_ok or meas.hom_inlier_ratio < cfg.min_inlier_ratio:
            return None
        delta = np.array([meas.hom_ux, meas.hom_uy, meas.hom_uz])
        n = meas.hom_normal
        dz_unit = float(n @ delta)
        if not np.isfinite(dz_unit) or abs(dz_unit) > min(0.6, cfg.max_dz_unit * hop):
            return None
        yaw = meas.hom_yaw
        if not np.isfinite(yaw) or abs(yaw) > cfg.max_yaw_step * hop:
            return None
        yatay = delta - dz_unit * n
        return yatay[:2], dz_unit, float(yaw)

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
            if k_ord % self.spacing == 0:
                self._anchor = dict(ord=k_ord, pos=np.asarray(gt_xyz, float).copy(),
                                    h=self.height, psi=self.psi)
            return pred

        a = self._anchor
        if a is not None and a["h"] is not None and self.alignment is not None:
            hop = k_ord - a["ord"]
            cand = self._ladder_candidate(meas_ladder, max(hop, 1)) if hop >= 1 else None
            if cand is not None:
                e_cam, dz_unit, yaw = cand
                e0 = rot2d(a["psi"]) @ e_cam          # homografi: keyframe (adım öncesi) ekseni
                dxy = self.alignment.M @ e0 * a["h"]
                if float(np.hypot(*dxy)) <= self.cfg.max_step_xy * hop:
                    dz = self.z_sign * dz_unit * a["h"]
                    pred = a["pos"] + np.array([dxy[0], dxy[1], dz])
                    self.pos = pred.copy()            # ardışık durumu senkron tut
                    if k_ord % self.spacing == 0:     # yeni keyframe: çapayı taşı
                        self._anchor = dict(
                            ord=k_ord, pos=pred.copy(),
                            h=self._h_guncelle(a["h"], dz_unit),
                            psi=a["psi"] + self._yaw_isle(yaw))
                    return pred.copy()
        # merdiven kullanılamadı: ardışık sonuçla devam; keyframe sınırındaysa çapayı yine taşı
        if k_ord % self.spacing == 0:
            self._anchor = dict(ord=k_ord, pos=self.pos.copy(), h=self.height, psi=self.psi)
        return pred
