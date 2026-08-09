# Görev 1 — tespit modeli değişimi: YOLOv8s → DINOv3-YOLO (28 Tem 2026)

**Değişen:** yalnız tespit katmanı (ağırlık + `imgsz`).
**Değişmeyen:** ego-hareket telafili hareket durumu, iniş uygunluğu, BoT-SORT takibi,
UAP/UAİ sınıf takası, şartname çıktı şeması — hepsi aynen korundu.

## Ölçüm (74 gerçek yarışma karesi, `ab_model.py`)

Aynı üretim yolu, aynı kareler, tek değişken model.

| Sınıf | MEVCUT (YOLOv8s) | ADAY (DINOv3) | ortak | yalnız aday | gerçek kayıp |
|---|---|---|---|---|---|
| **Taşıt** | 236 | **286** | 232 | **+54** | 4 |
| **UAP** | 12 | **24** | 12 | **+12** | 0 |
| **İnsan** | **64** | 20 | 0 | 20 | **−28** |
| **UAİ** | **12** | 7 | 0 | 7 | **−12** |

*(takipsiz/ham tespit — Görev 1 puanı mAP ile hesaplanır ve mAP ham tespitler üzerinden
ölçülür; BoT-SORT yalnız hareket durumu için gereklidir.)*

**Çözünürlük:** aday 640'ta eğitilmiş (checkpoint `train_args`) ve ölçüm bunu doğruladı —
640'ta net +86/−36, 1280'de +40/−56. Bu yüzden `imgsz` 1280 → **640** yapıldı. (Önceki
vanilla YOLOv8s için 1280 en iyiydi; model değişince optimum da değişti.)

## Dürüst değerlendirme

Aday **Taşıt ve UAP'de gerçekten üstün** — görsel doğrulamada mevcut modelin 0 tespit
yaptığı bir karede 3 park halindeki aracı doğru buldu, ve `ortak=232` kutu isabetinin de
iyi olduğunu gösteriyor.

Ama **İnsan ve UAİ'de geriliyor** ve bu sınıflarda `ortak = 0`: bulduğu birkaç nesnede
bile kutu isabetsiz (yeni tespit medyanı ~100×100 px, mevcut modelin insan kutuları
~29×43 px). Güven dağılımı nedeni açıklıyor: aday İnsan sınıfını medyan **0,065** güvenle
tespit ediyor (Taşıt'ta 0,814). Bu yüzden conf eşiği taraması da bir şey değiştirmedi —
düşük güvenli tespitler BoT-SORT'un `new_track_thresh=0.25` eşiğinde zaten eleniyordu.

Modelin kendi eğitim metrikleri de aynı yönde: mAP50-95 **0,728** (aday) vs **0,845**
(mevcut), recall 0,858 vs 0,943 — ancak iki model farklı veri seti yollarıyla eğitilmiş,
bu yüzden doğrudan kıyaslanamaz.

**Karar kullanıcıya aittir ve "tam değişim" yönünde verilmiştir.** Ölçüm, NT puanında
düşüş riskinin gerçek olduğunu söylüyor; bu risk bilinerek kabul edildi.

## Geri dönüş

Eski model duruyor: `agirliklar/best.pt` (`gorev1.ESKI_YOLOV8S_PT`). Geri almak için
`gorev1/__init__.py`'de `BEST_PT`'yi ona çevirmek ve `dedektor.py`'de `imgsz` varsayılanını
1280 yapmak yeterli.

## Fork bağımlılığı nasıl çözüldü

Ağırlık `ultralytics.nn.modules.dinov3.DINOv3` sınıfına pickle bağımlı; bu modül yalnız
DINOV3-YOLO fork'unda (8.4.80) var. Fork'u kurmak kurulu sürümü (8.4.92) geriye alır ve
Görev 3'ü de riske atardı. Bunun yerine modülün kopyası `gorev1/dinov3_modul.py` olarak
projede tutulur ve `dedektor._dinov3_enjekte()` ile yükleme anında `sys.modules`'e
enjekte edilir. Doğrulandı: model **fork olmadan**, kurulu ultralytics 8.4.92 ile yükleniyor.

## Hız

Görev 1 kare başı ~169 ms (önceki 1280'de ~114 ms). Tam sistem duman testinde üç görev
birlikte 562 ms/kare. Yarışma penceresi 90 dk, geçen oturum 46,5 dk — bütçe rahat.
