"""Görev 1 — Nesne Tespiti paketi.

**DINOv3 gövdeli YOLO** (`agirliklar/best.pt`, trial-8 koşusu) ile taşıt / insan /
iniş alanı (UAP-UAİ) tespiti. Tespit katmanının etrafındaki mantık ayrıca yürütülür:
ego-hareket telafili taşıt hareket durumu, iniş alanı uygunluğu ve BoT-SORT takibi.

Ağırlık dosyası (`agirliklar/best.pt`, ~206 MB) depoya konmaz; sevk paketinde
(`final-diyarbakir/`) bulunur.

**Sınıf eşlemesi ağırlığın kendi adlarından kurulur.** Eskiden sabit bir UAP/UAİ
takası vardı ve ağırlığa bağımlıydı: ağırlıklar arasında sınıf sırası değişiyor
(önceki ağırlıklarda 2=UAİ/3=UAP, trial-8'de 2=UAP/3=UAİ) ve yanlış eşleşme iki
sınıfı birden sıfırlardı. Artık `dedektor.spec_sinif_eslemesi()` model adlarını
şartname indeksine isimle bağlar; tanımsız bir sınıf adı kurulumda hata fırlatır.
Ayrıntı ve görsel kanıt: `denetim/TRIAL8_DEGISIMI.md`.

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
