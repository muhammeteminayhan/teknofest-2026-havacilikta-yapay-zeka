"""Görev 1 — Nesne Tespiti paketi.

**DINOv3 gövdeli YOLO** (`agirliklar/best_dinov3.pt`) ile taşıt / insan / iniş alanı
(UAP-UAİ) tespiti. Tespit katmanı dışındaki her şey (ego-hareket telafili hareket
durumu, iniş uygunluğu, BoT-SORT takibi, UAP/UAİ sınıf takası) değişmeden korunur.

**Fork gerekmez:** ağırlık `ultralytics.nn.modules.dinov3.DINOv3` sınıfına pickle
bağımlıdır ve bu modül yalnız DINOV3-YOLO fork'unda bulunur. Fork'u kurmak yerine
(8.4.92 → 8.4.80 downgrade'i Görev 3'ü de riske atardı) modülün kopyası
`dinov3_modul.py` olarak proje içinde tutulur ve yükleme anında `sys.modules`'e
enjekte edilir — kurulu `ultralytics` hiç değiştirilmez, çevrimdışı indirme tuzağı
yoktur.

Ölçüm kaydı (`denetim/ab_model_*.txt`, 74 gerçek yarışma karesi): aday Taşıt (+54) ve
UAP'de (+12) üstün; İnsan (64→20) ve UAİ'de (12→7) geriler. Tam değişim kullanıcı
kararıdır; eski model `agirliklar/best.pt` olarak durur, geri dönüş tek satırlıktır.

Kullanım:
    from gorev1 import Gorev1Detector
    det = Gorev1Detector()
    nesneler = det.detect(frame_bgr)   # list[dict], şartname şeması
"""
from pathlib import Path

_AGIRLIK = Path(__file__).resolve().parent / "agirliklar"
BEST_PT = _AGIRLIK / "best_dinov3.pt"        # aktif model (DINOv3 gövdeli)
ESKI_YOLOV8S_PT = _AGIRLIK / "best.pt"       # önceki vanilla YOLOv8s — geri dönüş için

from .dedektor import Gorev1Detector  # noqa: E402

__all__ = ["Gorev1Detector", "BEST_PT", "ESKI_YOLOV8S_PT"]
