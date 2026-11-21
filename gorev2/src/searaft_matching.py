# -*- coding: utf-8 -*-
"""SEA-RAFT optik akış motoru — RAFT'ın birebir ikizi, yalnız ağ değişir.

NEDEN
-----
SEA-RAFT (ECCV 2024, oral): RAFT'a göre iki iddiası var —
  * **çapraz-veri genellemesi daha iyi** (KITTI ve Spring'de en iyi cross-dataset
    sonuç). Final uçuşu bilinmeyen bir kamera/modalite ile geleceği için bizim
    açımızdan asıl çekici yan bu.
  * mixture-of-Laplace kaybıyla eğitildiği için **piksel başına belirsizlik** üretiyor
    (`info` çıktısı). Bu tur belirsizliği KULLANMIYOR — karşılaştırmanın tek
    değişkenli kalması için.

TASARIM: `RaftTracker`'dan türetilir ve YALNIZ iki şey değişir —
modelin kurulumu (`__init__`) ve akışın hesaplanması (`_prepare` + `_flow`).
Izgara örnekleme, kenar payı, ileri-geri elemesi, çalışma ölçeği, undistort —
hepsi ebeveynden aynen gelir. Böylece A/B'de tek değişken akış ağıdır.

ÇÖZÜNÜRLÜK NOTU
---------------
Sevk edilen ağırlık `Tartan-C-T-TSKH-spring540x960-M` **540×960'ta** eğitildi.
Bizim `auto_scale` 1920 px'i 0,5 ile 960×540'a indiriyor — yani eğitim
çözünürlüğüyle birebir örtüşüyor. Bu yüzden referans `custom.py`'nin ek
`2**scale` küçültmesi (config'te `scale=-1`) BURADA UYGULANMAZ: girdi zaten
doğru boyutta, bir kez daha küçültmek eğitim dağılımından uzaklaştırırdı.

KURULUM
-------
Kaynak ağaç `vendor/SEA-RAFT` altında (BSD-3, özel CUDA kerneli yok).
Ağırlık HuggingFace'ten: `MemorySlices/Tartan-C-T-TSKH-spring540x960-M`.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Optional

import numpy as np

from .calibration import CameraProfile
from .raft_matching import RaftTracker

KOK = Path(__file__).resolve().parents[2]
VENDOR = KOK / "vendor" / "SEA-RAFT"
VARSAYILAN_AGIRLIK = "MemorySlices/Tartan-C-T-TSKH-spring540x960-M"
VARSAYILAN_CFG = "spring-M"

# --- Denenen varyantlar ------------------------------------------------------
# `spring-L` ile `spring-M` AYNI mimaridir; tek fark `iters` (12 vs 4). Yani "L"
# daha buyuk model degil, daha cok yineleme demektir — RAFT'in 12'siyle esitlemek
# icin kullanilir.
#
# `...-spring540x960-M` Spring veri setine INCE AYARLIDIR. Makalenin capraz-veri
# genelleme iddiasi ise ince ayarsiz `Tartan-C-T-TSKH432x960-M` icindir; bizim veri
# (nadir hava goruntusu) her ikisi icin de alan disi oldugundan ikisi de denenir.
VARYANTLAR = {
    "searaft":    dict(agirlik=VARSAYILAN_AGIRLIK, cfg="spring-M", iters=4),
    "searaft_l":  dict(agirlik=VARSAYILAN_AGIRLIK, cfg="spring-M", iters=12),
    "searaft_g":  dict(agirlik="MemorySlices/Tartan-C-T-TSKH432x960-M",
                       cfg="kitti-M", iters=4),
    "searaft_gl": dict(agirlik="MemorySlices/Tartan-C-T-TSKH432x960-M",
                       cfg="kitti-M", iters=12),
}


def _cfg_yukle(ad: str) -> types.SimpleNamespace:
    yol = VENDOR / "config" / "eval" / f"{ad}.json"
    if not yol.exists():
        raise FileNotFoundError(f"SEA-RAFT config bulunamadi: {yol}")
    d = json.loads(yol.read_text(encoding="utf-8"))
    # `scale` girdiyi 2**scale ile olcekler. Bizim girdimiz zaten egitim
    # cozunurlugunde geldigi icin ek olcekleme YAPILMAZ (bkz. modul dokumani).
    d["scale"] = 0
    return types.SimpleNamespace(**d)


class SeaRaftTracker(RaftTracker):
    """`RaftTracker` ile aynı arayüz; içeride SEA-RAFT koşar."""

    def __init__(
        self,
        profile: Optional[CameraProfile] = None,
        agirlik: str = VARSAYILAN_AGIRLIK,
        cfg_adi: str = VARSAYILAN_CFG,
        device: Optional[str] = None,
        downscale: Optional[float] = None,
        grid_step: Optional[int] = 28,
        hedef_nokta: int = 640,
        on_isle: Optional[str] = None,
        clahe_clip: float = 2.0,
        margin: int = 16,
        fb_check: bool = True,
        fb_threshold: float = 1.5,
        iters: Optional[int] = None,
        min_points: int = 12,
    ):
        import cv2
        import torch

        # --- ebeveynin __init__'i torchvision RAFT'i kurar; onu cagirmadan ayni
        #     nitelikleri birebir kuruyoruz ki gereksiz agirlik indirilmesin.
        self.torch = torch
        self.profile = profile
        self.downscale = None if downscale is None else float(downscale)
        self.grid_step = grid_step
        self.hedef_nokta = hedef_nokta
        self.on_isle = on_isle
        self._clahe = (cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
                       if on_isle == "clahe" else None)
        self.margin = margin
        self.fb_check = fb_check
        self.fb_threshold = fb_threshold
        self.min_points = min_points
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        çekirdek = str(VENDOR / "core")
        if çekirdek not in sys.path:
            sys.path.insert(0, çekirdek)
        from raft import RAFT                                   # noqa: E402

        self.args = _cfg_yukle(cfg_adi)
        self.num_flow_updates = int(iters if iters is not None else self.args.iters)
        self.model = RAFT.from_pretrained(agirlik, args=self.args)
        self.model = self.model.to(self.device).eval()
        self.son_belirsizlik: Optional[np.ndarray] = None   # (H,W) — bu tur kullanılmıyor

    # ------------------------------------------------------------------
    def _prepare(self, gray: np.ndarray, scale: float = None):
        """SEA-RAFT girdisi: HAM 0-255, 3 kanal. Normalizasyon ve doldurma ağın içinde.

        `RaftTracker._prepare` [-1,1]'e normalize edip 8'in katına dolduruyordu;
        SEA-RAFT `forward` içinde ikisini de kendi yapıyor ve çıktıyı geri kırpıyor,
        bu yüzden burada yalnız ölçekleme kalır.
        """
        import cv2
        img = gray
        scale = self._effective_scale(gray.shape[1]) if scale is None else scale
        if scale != 1.0:
            img = cv2.resize(img, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
        t = self.torch.from_numpy(img).float()          # 0-255
        t = t.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)  # 1x3xHxW
        return t, (h, w)

    def _flow(self, t0, t1):
        with self.torch.no_grad():
            cikti = self.model(t0.to(self.device), t1.to(self.device),
                               iters=self.num_flow_updates, test_mode=True)
        akis = cikti["flow"][-1][0].float().cpu().numpy()        # 2xHxW
        bilgi = cikti.get("info")
        if bilgi is not None:
            # info: (1,4,H,W) — ilk 2 kanal softmax agirligi, kalan 2 varyans.
            # Bu tur kullanilmiyor; ileride belirsizlik kapisi icin saklaniyor.
            self.son_belirsizlik = bilgi[-1][0, 2:].float().cpu().numpy().mean(0)
        return akis
