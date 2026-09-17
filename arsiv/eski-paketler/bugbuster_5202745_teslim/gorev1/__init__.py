"""Görev 1 — Nesne Tespiti paketi.

**DINOv3 gövdeli YOLO** (`agirliklar/best.pt`) ile taşıt / insan / iniş alanı (UAP-UAİ)
tespiti. Tespit katmanının etrafındaki mantık ayrıca yürütülür: ego-hareket telafili
taşıt hareket durumu, iniş alanı uygunluğu, BoT-SORT takibi ve UAP/UAİ sınıf takası.

**Model dosyası:** `agirliklar/best.pt` depoya dahil değildir (boyut sınırı); Google
Drive'dan indirilip bu klasöre konur — bkz. kök dizindeki `model.txt`.

**Özel fork gerekmez.** Ağırlık `ultralytics.nn.modules.dinov3.DINOv3` sınıfına pickle
bağımlıdır ve bu modül yalnız DINOV3-YOLO fork'unda bulunur. Fork'u kurmak yerine
(kurulu ultralytics'i geriye alır ve DINOv3 kullanan Görev 3'ü de riske atardı) modülün
kopyası `dinov3_modul.py` olarak pakette tutulur ve model yüklenirken `sys.modules`'e
enjekte edilir. Böylece standart `ultralytics` sürümüyle çalışır.

Kullanım:
    from gorev1 import Gorev1Detector
    det = Gorev1Detector()
    nesneler = det.detect(frame_bgr)   # list[dict], şartname şeması
"""
from pathlib import Path

_AGIRLIK = Path(__file__).resolve().parent / "agirliklar"
BEST_PT = _AGIRLIK / "best.pt"

from .dedektor import Gorev1Detector  # noqa: E402

__all__ = ["Gorev1Detector", "BEST_PT"]
