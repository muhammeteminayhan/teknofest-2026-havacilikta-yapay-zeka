"""
Hareketli/sabit taşıt ayrımı — ego-hareket telafili (şartname s.11).

Sorun: hava aracı sürekli hareket ettiğinden SABİT taşıtlar bile görüntüde kayar.
Çözüm: ardışık iki kare arasında küresel kamera hareketini (homografi H) `ortak`
modülünden al; bir taşıtın önceki konumunu H ile "sabit olsaydı" konumuna taşı ve
gerçek konumuyla karşılaştır. Artık (residual) kayma eşik üstündeyse → HAREKETLİ.

Bu, Görev 1 ile Görev 2'yi birbirine bağlayan paylaşılan ego-hareket hesabını kullanır
(`ortak.ego_hareket`). Baseline: IoU/merkez-mesafesi ile basit eşleştirme + zamansal
tutarlılık (N kare). İleride: gerçek çoklu-nesne takibi (BoT-SORT) ile değiştirilebilir.
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional

import numpy as np

from ortak.ego_hareket import kestir_homografi, artik_hareket_px


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (area_a + area_b - inter + 1e-9)


class HareketSiniflandirici:
    """
    Kare-kare beslenir; her taşıt tespiti için 0 (hareketsiz) / 1 (hareketli) verir.

    Kullanım:
        hs = HareketSiniflandirici(esik_px=6.0, pencere=3)
        durumlar = hs.guncelle(frame_bgr, tasit_deteksiyonlari)   # her kare
    """

    def __init__(self, esik_px: float = 6.0, pencere: int = 3, iou_esik: float = 0.2):
        self.esik_px = esik_px
        self.pencere = pencere
        self.iou_esik = iou_esik
        self._prev_img: Optional[np.ndarray] = None
        self._prev_dets: List = []                       # önceki kare taşıt kutuları
        self._gecmis: Dict[int, deque] = {}              # iz id → son N artık hareket
        self._sonraki_id = 0
        self._prev_ids: List[int] = []

    def guncelle(self, frame: np.ndarray, tasit_dets: List) -> List[int]:
        """
        frame      : mevcut kare (BGR/gri).
        tasit_dets : bu karedeki taşıt tespitleri (xyxy + .merkez olan nesneler;
                     gorev1.modeller.Detection ya da uyumlu).
        Döndürür   : tasit_dets ile aynı sırada hareket durumu listesi (0/1).
        """
        durumlar = [0] * len(tasit_dets)
        cur_ids = [-1] * len(tasit_dets)

        ego = None
        if self._prev_img is not None:
            ego = kestir_homografi(self._prev_img, frame)

        if ego is not None and ego.ok and self._prev_dets:
            for i, det in enumerate(tasit_dets):
                # Önceki karede en iyi eşleşen kutuyu bul (IoU).
                j_best, iou_best = -1, 0.0
                for j, pdet in enumerate(self._prev_dets):
                    v = _iou(det.xyxy, pdet.xyxy)
                    if v > iou_best:
                        iou_best, j_best = v, j

                if j_best >= 0 and iou_best >= self.iou_esik:
                    iz_id = self._prev_ids[j_best]
                    if iz_id < 0:
                        iz_id = self._sonraki_id; self._sonraki_id += 1
                    artik = artik_hareket_px(ego.H,
                                             self._prev_dets[j_best].merkez, det.merkez)
                    dq = self._gecmis.setdefault(iz_id, deque(maxlen=self.pencere))
                    dq.append(artik)
                    # Zamansal tutarlılık: penceredeki medyan artık eşik üstündeyse hareketli.
                    if len(dq) >= min(self.pencere, 2) and float(np.median(dq)) > self.esik_px:
                        durumlar[i] = 1
                    cur_ids[i] = iz_id
                else:
                    cur_ids[i] = self._sonraki_id; self._sonraki_id += 1

        # Durumu ilerlet.
        self._prev_img = frame
        self._prev_dets = list(tasit_dets)
        self._prev_ids = cur_ids
        # Kullanılmayan iz geçmişlerini temizle.
        canli = set(i for i in cur_ids if i >= 0)
        for k in list(self._gecmis.keys()):
            if k not in canli:
                del self._gecmis[k]
        return durumlar
