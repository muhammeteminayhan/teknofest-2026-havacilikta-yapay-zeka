# Görev 1 — Nesne Tespiti

Alt-görüş kamerası karelerinde **taşıt, insan ve iniş alanı (UAP / UAİ)** tespiti;
her nesne için sınıf, konum (kutu) ve durum bilgisi (taşıt hareketli mi, iniş alanı
uygun mu) üretilir.

## Nasıl çalışır

Arkadaşımızın eğittiği **vanilla YOLOv8s** modeli (`agirliklar/best.pt`) + BoT-SORT
takip. Standart `ultralytics` paketi kullanılır — **özel fork YOK, çevrimdışı indirme
tuzağı YOK** (model tek başına, `best.pt` yüklenmesi internet istemez).

`Gorev1Detector.detect(frame_bgr)` her kareyi işler ve şartname şemasında sözlük
listesi döndürür. Sınıf durumlu (stateful): takip ve hareket için önceki kareyi ve
takip edilen merkezleri hafızada tutar.

### Üretilen alanlar
| Alan | Anlam |
|---|---|
| `cls` | Sınıf (0 Taşıt, 1 İnsan, 2 UAP, 3 UAİ — şartname sırası) |
| `motion_status` | Yalnız taşıt: 0 sabit, 1 hareketli, aksi halde -1 |
| `landing_status` | Yalnız UAP/UAİ: 1 uygun, 0 uygun değil, aksi halde -1 |
| `top_left_x/y`, `bottom_right_x/y` | Kutu köşeleri (piksel) |

### İki önemli detay (arkadaşın script'inde eksikti, burada tamamlandı)

1. **Sınıf takası 2↔3.** Model `{0:Taşıt, 1:İnsan, 2:UAİ, 3:UAP}` sırasıyla eğitilmiş
   ama şartname `{2:UAP, 3:UAİ}` istiyor. `_spec_cls()` bu takası yapar. (Doğrulama:
   `YOLO(best.pt).names` → `{2:'UAİ', 3:'UAP'}`.)
2. **Hareket durumu.** Arkadaşın `teknofest.py`'sinde `motion_status` sabit 0 idi (TODO).
   Burada gerçek hesap var: ardışık kareler arasında **kamera ego-hareketi homografi ile
   telafi edilir** (`_calculate_homography`), takip edilen taşıtın beklenen konumu ile
   gerçek konumu arasındaki kayma `motion_thresh` (5 px) üstündeyse "hareketli".

## Dosyalar
```
gorev1/
├── __init__.py          # Gorev1Detector + BEST_PT yolunu dışa açar
├── dedektor.py          # Gorev1Detector sınıfı (tüm mantık)
├── agirliklar/best.pt   # vanilla YOLOv8s ağırlığı (21 MB)
├── kalite_denetim.py    # sahada gönderilen kutuları gerçek karelere çizer (denetim)
├── ab_cozunurluk.py     # çözünürlük A/B karşılaştırma + karar mekanizması
└── denetim/             # denetim çıktıları (paneller, karar dosyaları, CSV)
```

## Kalite denetimi ve çözünürlük kararı (kanıta dayalı)

Puanlı oturum sonrası, **sahada gerçekten ne gönderdiğimiz** denetlendi ve tespit
çözünürlüğü kanıta dayalı olarak seçildi. İki araç bunu tekrarlanabilir kılar:

**1) `kalite_denetim.py`** — yarışma logundaki her karenin `detected_objects` kutularını
o karenin gerçek görüntüsüne çizer; sınıf bazında paneller + 0 nesneli kare örnekleri
üretir (kaçırma / fazladan kutu gözle denetlenir).
```powershell
.venv\Scripts\python gorev1\kalite_denetim.py
```
*Bulgu:* şema %100 tutarlı (Taşıt sayısı = `moving_status` 0/1 toplamı), kutular doğru
nesnede, **uydurma kutu yok** — ancak `imgsz=640` ile **kaçırma var**: "0 nesne" gönderilen
karelerde gözle görülen park etmiş araçlar mevcut (bkz. `denetim/mevcut_bos_kareler.png`).

**2) `ab_cozunurluk.py`** — iki çözünürlüğü **üretim yolunun birebir aynısıyla**
(Gorev1Detector + BoT-SORT takip, ardışık kare blokları) karşılaştırır; kutuları IoU ile
eşleştirip *yeni / gerçek kayıp / yer değiştirme* ayrımı yapar ve **karar** üretir.
```powershell
.venv\Scripts\python gorev1\ab_cozunurluk.py --a 640 --b 1280
```

| Karşılaştırma | Yeni | Gerçek kayıp | Yeni kutu medyanı | Karar |
|---|---|---|---|---|
| 640 → **1280** | **+97** | −10 (%3,4) | 70×70 px (gerçek nesne) | ✅ **1280 önerilir** |
| 1280 → 1920 | +56 | −23 (%6,6) | **16×16 px (gürültü)** | ❌ reddedildi |

**1920 neden reddedildi:** görsel panelde (`denetim/ab_1280_vs_1920.png`) modelin **beyaz
yuvarlak taşları "UAP" sandığı** görüldü — yanlış pozitif precision'ı, dolayısıyla mAP'i
düşürür. Mekanizma bunu "yeni tespit boyutu gürültü seviyesinde" ölçütüyle otomatik yakalar.

**Sonuç:** `imgsz=1280` seçildi. Boş kare oranı %20→%16, maliyet +198 ms/kare
(tam koşum tahmini ~57 dk, yarışma penceresi 90 dk — güvenli).

> **Not (karar felsefesi):** sayısal tavsiye tek başına yeterli değildir; her iki A/B
> koşusunda da yeşil (yeni) kutular **gözle** doğrulanmıştır. 1920 örneğinde sayısal
> mekanizma önce "önerilir" demiş, görsel kanıt bunu çürütmüştür — ardından mekanizmaya
> gürültü ölçütü eklenmiştir.

## Tek başına test
```powershell
# Proje kökünden
.venv\Scripts\python -c "import sys; sys.path.insert(0,'.'); import cv2; from gorev1 import Gorev1Detector; d=Gorev1Detector(); cap=cv2.VideoCapture(r'THYZ_2026_Ornek_Veri_Seti\THYZ_2026_Ornek_Veri_1.MP4'); cap.set(1,5000); _,f=cap.read(); print(d.detect(f))"
```

## Ayar noktaları (müdahale etmek isterseniz)
`Gorev1Detector(...)` parametreleri (hepsinin makul varsayılanı var):
- `conf=0.25` — tespit güven eşiği. Düşürmek daha çok (ama daha gürültülü) tespit;
  yükseltmek daha az yanlış pozitif. **Yanlış pozitif mAP'i düşürür**, dikkatli ayarlayın.
- `iou=0.7` — NMS örtüşme eşiği.
- `motion_thresh=5.0` — taşıt "hareketli" sayılması için gereken piksel kayması.
- `imgsz=1280` — modele giren çözünürlük. **Kanıta dayalı seçildi** (yukarıdaki A/B
  bölümü): 640 küçük nesneleri kaçırıyor, 1920 uydurma kutu üretiyor. Değiştirmeden önce
  `ab_cozunurluk.py` ile ölçün.

Yeni bir model gelirse: `agirliklar/best.pt`'yi değiştirin. Sınıf sırası farklıysa
`_spec_cls()`'i güncelleyin (önce `YOLO(best.pt).names` ile sırayı kontrol edin).
