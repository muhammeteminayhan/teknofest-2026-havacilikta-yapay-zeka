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
└── agirliklar/best.pt   # vanilla YOLOv8s ağırlığı (21 MB)
```

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
- `imgsz=640` — modele giren çözünürlük.

Yeni bir model gelirse: `agirliklar/best.pt`'yi değiştirin. Sınıf sırası farklıysa
`_spec_cls()`'i güncelleyin (önce `YOLO(best.pt).names` ile sırayı kontrol edin).
