# -*- coding: utf-8 -*-
"""Ölçek çapası: Görev 1 kutularından metre/piksel gözlemi ile yükseklik durumunu düzelt.

Kesintide konum adımı ``dxy = M · e0 · h``; ``h`` yalnız görüntü ölçeğiyle çarpımsal
taşınır ve mutlak bir gözlemi yoktur. Uzun kesintide küçük bir ölçüm yanlılığı üstel
birikir (finalde O4: h 55,7 → 14,8 m, kat edilen yol gerçeğin %52'si).

Yerdeki nesnelerin piksel boyu GSD'yi (metre/piksel) verir:  h ∝ 1 / boy.
  * Taşıt: fiziksel boyu bilinmiyor ama nüfus medyanı sabit → katsayı SAĞLIKLI dönemde
    öğrenilir (K = h · L_medyan). Böylece kalibrasyonun c-uzayı da otomatik taşınır.
  * UAP/UAİ: şartname gereği 4,5 m çaplı daire → fiziksel katsayı f·4,5; sağlıklı
    dönemde ped görülmüşse oradan doğrulanır, tutarsızsa kanal kapanır (yanlış pozitif).

Doğru kâhinle ölçüldü (kahin_olcek.py): kesintide yol uzunluğu oranıyla h düzeltilince
12 yatak×takvimin 11'inde hata düşüyor (ort. 16,70 → 12,01 m). Bu modül o kâhinin
gerçek gözlemle yapılabilen hâlidir.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

PED_CAP_M = 4.5                 # şartname 2.1.2: UAP/UAİ 4,5 m çaplı daire


@dataclass
class CapaAyar:
    kazanc: float = 0.5          # g: h ← h·(h_gözlem/h)^g
    olu_bant: float = 0.0        # |ln(h_gözlem/h)| bundan küçükse dokunma
    aralik: int = 8              # düzeltme kaç karede bir (kâhinle aynı)
    adim_sinir: tuple = (0.8, 1.25)   # tek düzeltmede en fazla oran
    arac_min: int = 3            # karede en az kaç taşıt
    arac_pencere: int = 15       # son kaç gözlemin medyanı
    arac_maks_yas: int = 40      # pencere gözlemleri en fazla kaç kare eski olabilir
    kal_pencere: int = 300       # katsayıyı öğrenmede son kaç sağlıklı kare
    kal_min: int = 20            # katsayı için en az kaç sağlıklı gözlem
    kal_maks_sapma: float = 0.30 # log(h·L) sağlam std'si bunu aşarsa kanal kapalı
    ped: bool = True
    arac: bool = True            # False: yalniz 4,5 m ped kanali (kullanici istegi: ped yontemi ayri olculsun)
    ped_tutarlilik: float = 0.15 # ardışık ped çaplarının oransal tutarlılığı
    ped_ardisik: int = 3
    ped_fiziksel_aralik: tuple = (0.7, 1.4)   # öğrenilen/fiziksel katsayı oranı
    kenar_px: float = 4.0


class OlcekCapasi:
    """Kare başına kutuları alır; sağlıklıda öğrenir, kesintide yükseklik hedefi üretir."""

    def __init__(self, fx: float, boyut: tuple[int, int], ayar: Optional[CapaAyar] = None):
        self.f = float(fx)
        self.W, self.H = boyut
        self.a = ayar or CapaAyar()
        self._kal_arac = deque(maxlen=self.a.kal_pencere)   # log(h·L)
        self._kal_ped = deque(maxlen=self.a.kal_pencere)    # log(h·D)
        self._arac = deque(maxlen=self.a.arac_pencere)      # (kare, h_gözlem)
        self._ped_gecmis = deque(maxlen=self.a.ped_ardisik) # (kare, D)
        self.K_arac: Optional[float] = None
        self.K_ped: Optional[float] = None
        self.durum = {}

    # ------------------------------------------------------------------ kutulardan ölçü
    def _icerde(self, x1, y1, x2, y2) -> bool:
        k = self.a.kenar_px
        return x1 > k and y1 > k and x2 < self.W - k and y2 < self.H - k

    def olcu(self, kutular) -> tuple[Optional[float], Optional[float]]:
        """(taşıt uzun kenarı medyanı, ped çapı) — yoksa None.

        `kutular`: [(sinif, x1, y1, x2, y2), ...]; sınıf sunucu numarası (1 taşıt, 3 UAP, 4 UAİ).
        """
        L, D = [], []
        for s, x1, y1, x2, y2 in kutular:
            if not self._icerde(x1, y1, x2, y2):
                continue
            w, h = x2 - x1, y2 - y1
            if s == 1:
                L.append(max(w, h))
            elif s in (3, 4) and 0.75 <= w / max(h, 1e-6) <= 1.33:
                D.append(max(w, h))
        L_med = None
        if len(L) >= self.a.arac_min:
            L = np.array(L)
            m = np.median(L)
            L = L[(L > 0.5 * m) & (L < 2.0 * m)]      # kamyon/otobüs ve kırpık kutuları ele
            if len(L) >= self.a.arac_min:
                L_med = float(np.median(L))
        return L_med, (float(max(D)) if D else None)

    # ------------------------------------------------------------------ öğrenme
    def saglikli(self, h: Optional[float], kutular):
        """Sağlıklı kare: o karedeki yükseklik durumuyla katsayıları öğren."""
        self._arac.clear()
        self._ped_gecmis.clear()
        if h is None or h <= 0:
            return
        L, D = self.olcu(kutular)
        if L:
            self._kal_arac.append(np.log(h * L))
        if D:
            self._kal_ped.append(np.log(h * D))
        self.K_arac = self._katsayi(self._kal_arac)
        Kp = self._katsayi(self._kal_ped, min_n=5)
        fiz = self.f * PED_CAP_M
        if Kp is not None:
            o = Kp / fiz
            self.K_ped = Kp if self.a.ped_fiziksel_aralik[0] <= o <= self.a.ped_fiziksel_aralik[1] else None
        else:
            self.K_ped = fiz if self.a.ped else None

    def _katsayi(self, kal, min_n: Optional[int] = None) -> Optional[float]:
        if len(kal) < (min_n or self.a.kal_min):
            return None
        v = np.array(kal)
        med = np.median(v)
        sapma = 1.4826 * np.median(np.abs(v - med))
        if sapma > self.a.kal_maks_sapma:
            return None
        return float(np.exp(med))

    # ------------------------------------------------------------------ kesinti
    def kesinti_hedefi(self, k: int, kutular) -> Optional[float]:
        """Kesinti karesi: o ana kadarki gözlemlerden yükseklik hedefi (yoksa None)."""
        L, D = self.olcu(kutular)
        if L and self.K_arac and self.a.arac:
            self._arac.append((k, self.K_arac / L))
        h_ped = None
        if D and self.K_ped and self.a.ped:
            self._ped_gecmis.append((k, D))
            g = list(self._ped_gecmis)
            if (len(g) == self.a.ped_ardisik and g[-1][0] - g[0][0] <= 2 * self.a.ped_ardisik
                    and (max(d for _, d in g) / min(d for _, d in g) - 1) <= self.a.ped_tutarlilik):
                h_ped = self.K_ped / np.median([d for _, d in g])
        taze = [h for kk, h in self._arac if k - kk <= self.a.arac_maks_yas]
        h_arac = float(np.median(taze)) if len(taze) >= 5 else None
        if h_ped is not None and h_arac is not None:
            # iki kaynak çelişiyorsa ped muhtemelen yanlış pozitif → araç
            return h_ped if abs(np.log(h_ped / h_arac)) < 0.25 else h_arac
        return h_ped if h_ped is not None else h_arac

    def duzeltme(self, h: float, h_hedef: float) -> float:
        """Yükseklik durumuna uygulanacak çarpan (1 = dokunma).

        Yumuşak eşik: |ln(h_hedef/h)| ölü bandın içindeyse dokunulmaz; dışındaysa yalnız
        BANDI AŞAN kısım düzeltilir. Gerekçe (video1 teşhisi): taşıt gözlemi bölgeden
        bölgeye ±%10–17 dalgalanıyor (araç karışımı, kamera eğikliği); RAFT'ın ölçeği zaten
        doğruyken gözlemi izlemek hata ekliyor. Finaldeki O4'te ise gereken düzeltme ×2–3.
        """
        r = float(np.log(h_hedef / h))
        b = self.a.olu_bant
        if abs(r) <= b:
            return 1.0
        r = r - np.sign(r) * b
        return float(np.clip(np.exp(self.a.kazanc * r), *self.a.adim_sinir))


def kanca_kur(capa: OlcekCapasi, kutular_kare: dict, saglikli: np.ndarray):
    """Önbellekli koşucu için kanca: `final_yatak.kos(..., kanca=...)`.

    Kare k güncellenmeden önce: k-1 karesinin kutuları ile k-1 sonrası durum eşleştirilir
    (canlıda Görev 1 aynı karede Görev 2'den önce koşar → nedensel).
    """
    def kanca(k, est, preds):
        j = k - 1
        kut = kutular_kare.get(j, [])
        if saglikli[j]:
            capa.saglikli(est.height, kut)
            return
        h_hedef = capa.kesinti_hedefi(j, kut)
        if h_hedef is None or est.height is None or j % capa.a.aralik:
            return
        c = capa.duzeltme(est.height, h_hedef)
        if c != 1.0:
            est.height = est._clamp_h(est.height * c)
            a = est._anchor
            if a is not None and a.get("h") is not None:
                a["h"] = est._clamp_h(a["h"] * c)
    return kanca
