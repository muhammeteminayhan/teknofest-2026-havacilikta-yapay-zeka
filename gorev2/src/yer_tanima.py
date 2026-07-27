"""Görsel yer tanıma (VPR) — kesintide biriken sürüklenmeyi düzeltmek için.

**Neden:** Yarışmada kesinti blokları çok uzun (449/499/599 kare ≈ 60–80 sn) ve
konum hatası blok boyunca doğrusal birikiyor (blok sonunda 12–15 m). Ancak
puanlı oturum kayıtlarında görüldü ki **araç aynı bölgeye geri dönüyor**
(354 m kat edilen yol → 51 m net yer değiştirme). Yani kesinti sırasında,
*sağlıklı dönemde GT ile görülmüş* alanların üzerinden yeniden geçiliyor.

**Fikir (loop closure):** Sağlıklı karelerde her karenin görsel betimleyicisi
(DINOv3 CLS token) ve **gerçek GT konumu** bir haritada saklanır. Kesinti
karelerinde aynı betimleyici çıkarılıp haritada aranır; yeterince ayırt edici
bir eşleşme bulunursa o karenin GT konumu bir "gözlem" olarak kullanılıp
kestirim yumuşakça o yöne çekilir. Böylece sürüklenme sıfırlanmasa da sınırlanır.

**Yanlış eşleşmeye karşı üç kapı** (hepsi geçilmeden düzeltme yapılmaz):
1. **Mutlak benzerlik:** kosinüs benzerliği ``min_benzerlik`` üstünde olmalı.
2. **Ayırt edicilik (Lowe oranı):** en iyi eşleşme, *farklı bir bölgedeki*
   ikinci en iyi eşleşmeden belirgin şekilde daha iyi olmalı. Havadan çekimde
   çok sayıda benzer görünen alan (tarla, yol) vardır; bu kapı olmadan sistem
   kolayca yanılır.
3. **Uzaysal makullük:** aday konum, mevcut VO kestiriminden ``max_sicrama``
   metreden uzaksa reddedilir — sürüklenme sınırlıyken uzak bir aday, doğru
   eşleşme değil neredeyse kesin bir hatadır.

Düzeltme **yumuşaktır** (``alpha`` ile karışım): konum durumu asla zıplatılmaz,
Görev 2 durum makinesinin sürekliliği korunur.

Betimleyici üretimi isteğe bağlıdır: sınıf, önceden hesaplanmış betimleyicilerle
(çevrimdışı A/B için) veya canlı kare vererek kullanılabilir.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np


class YerTanima:
    """Sağlıklı karelerden harita kurar, kesinti karelerinde konum önerir."""

    def __init__(
        self,
        backbone=None,
        min_benzerlik: float = 0.75,
        lowe_orani: float = 1.06,
        max_sicrama: float = 60.0,
        alpha: float = 0.25,
        ayrik_mesafe: float = 25.0,
        giris_araligi: int = 3,
        sorgu_araligi: int = 5,
        boyut: tuple[int, int] = (336, 224),
    ):
        """
        backbone      : DINOv3 sarmalayıcı (``forward`` sonrası ``last_cls_token``).
                        None ise yalnızca önceden hesaplanmış betimleyici kullanılabilir.
        min_benzerlik : kabul için asgari kosinüs benzerliği.
        lowe_orani    : en iyi / (farklı bölgedeki) ikinci en iyi oranı; büyük = seçici.
        max_sicrama   : aday konum ile VO kestirimi arasındaki azami mesafe [m].
        alpha         : düzeltme karışım katsayısı (0 = kapalı, 1 = tamamen adaya atla).
        ayrik_mesafe  : "farklı bölge" sayılması için asgari konum ayrımı [m] (Lowe kapısı).
        giris_araligi : haritaya kaç karede bir kayıt eklenecek (bellek/çeşitlilik).
        sorgu_araligi : kesintide kaç karede bir sorgu yapılacak (hesap bütçesi).
        boyut         : betimleyici için yeniden boyutlandırma (genişlik, yükseklik).
        """
        self.backbone = backbone
        self.min_benzerlik = min_benzerlik
        self.lowe_orani = lowe_orani
        self.max_sicrama = max_sicrama
        self.alpha = alpha
        self.ayrik_mesafe = ayrik_mesafe
        self.giris_araligi = giris_araligi
        self.sorgu_araligi = sorgu_araligi
        self.boyut = boyut

        self._desc: list[np.ndarray] = []     # harita betimleyicileri (L2 normalize)
        self._pos: list[np.ndarray] = []      # karşılık gelen GT konumlar
        self._n_ekle = 0
        self._n_sorgu = 0
        # tanılama
        self.n_kabul = 0
        self.n_red_benzerlik = 0
        self.n_red_lowe = 0
        self.n_red_mesafe = 0

    # ------------------------------------------------------------------
    def betimleyici(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Kareden L2-normalize DINOv3 CLS betimleyicisi üret (backbone gerekir)."""
        if self.backbone is None or frame_bgr is None:
            return None
        import cv2
        import torch

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, self.boyut, interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        ort = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        x = (x - ort) / std
        p = next(self.backbone.parameters())
        with torch.no_grad():
            self.backbone(x.to(p.device, p.dtype))
            cls = self.backbone.last_cls_token
        v = cls[0].float().cpu().numpy()
        n = np.linalg.norm(v)
        return (v / n) if n > 0 else None

    # ------------------------------------------------------------------
    def ekle(self, desc: Optional[np.ndarray], pos: np.ndarray) -> None:
        """Sağlıklı karede haritaya kayıt ekle (betimleyici + GT konum)."""
        if desc is None:
            return
        self._n_ekle += 1
        if self._n_ekle % self.giris_araligi:
            return
        self._desc.append(np.asarray(desc, dtype=np.float32))
        self._pos.append(np.asarray(pos, dtype=float).copy())

    # ------------------------------------------------------------------
    def sorgula(self, desc: Optional[np.ndarray], vo_pos: np.ndarray,
                zorla: bool = False) -> Optional[np.ndarray]:
        """Kesinti karesinde aday konum döndür (yoksa None).

        `zorla=True` sorgu aralığı sayacını atlar (test/tanı için).
        """
        if desc is None or len(self._desc) < 5:
            return None
        self._n_sorgu += 1
        if not zorla and (self._n_sorgu % self.sorgu_araligi):
            return None

        D = np.stack(self._desc)                      # (M, d)
        benzerlik = D @ np.asarray(desc, dtype=np.float32)
        i = int(np.argmax(benzerlik))
        s1 = float(benzerlik[i])
        if s1 < self.min_benzerlik:
            self.n_red_benzerlik += 1
            return None

        aday = self._pos[i]
        # Lowe kapısi: FARKLI bir bolgedeki en iyi ikinci eslesme
        P = np.stack(self._pos)
        uzak = np.linalg.norm(P - aday, axis=1) > self.ayrik_mesafe
        if uzak.any():
            s2 = float(benzerlik[uzak].max())
            if s2 > 0 and s1 / s2 < self.lowe_orani:
                self.n_red_lowe += 1
                return None

        # uzaysal makullük kapisi
        if np.linalg.norm(aday - np.asarray(vo_pos, dtype=float)) > self.max_sicrama:
            self.n_red_mesafe += 1
            return None

        self.n_kabul += 1
        return aday.copy()

    # ------------------------------------------------------------------
    def duzelt(self, vo_pos: np.ndarray, aday: Optional[np.ndarray],
               sadece_xy: bool = True) -> np.ndarray:
        """Yumuşak düzeltme: konumu adaya doğru ``alpha`` kadar çek."""
        if aday is None:
            return vo_pos
        yeni = np.asarray(vo_pos, dtype=float).copy()
        k = self.alpha
        yeni[0] = (1 - k) * yeni[0] + k * aday[0]
        yeni[1] = (1 - k) * yeni[1] + k * aday[1]
        if not sadece_xy:
            yeni[2] = (1 - k) * yeni[2] + k * aday[2]
        return yeni

    # ------------------------------------------------------------------
    @property
    def harita_boyu(self) -> int:
        return len(self._desc)

    def tani(self) -> dict:
        return dict(harita=self.harita_boyu, kabul=self.n_kabul,
                    red_benzerlik=self.n_red_benzerlik, red_lowe=self.n_red_lowe,
                    red_mesafe=self.n_red_mesafe)
