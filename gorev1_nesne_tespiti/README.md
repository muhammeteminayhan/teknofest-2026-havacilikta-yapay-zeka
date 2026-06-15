# Görev 1 — Nesne Tespiti (%25)

Alt-görüş hava kamerasından **Taşıt**, **İnsan**, **UAP** (Uçan Araba Park) ve **UAİ**
(Uçan Ambulans İniş) alanlarını tespit etme. Ek olarak: taşıtlarda **hareketli/sabit**,
UAP/UAİ'de **iniş uygunluğu**. Puan: **mAP @ IoU=0.5**.

## Yaklaşım

| Bileşen | Çözüm |
|---|---|
| **Tespit** | YOLOv11 (ultralytics) + **SAHI** dilimleme (FHD/4K'da küçük nesne) |
| **Hareketli/sabit** | Ego-hareket telafisi (`ortak/ego_hareket` — Görev 2 ile paylaşımlı homografi); artık piksel kayması eşik üstüyse hareketli |
| **UAP/UAİ iniş** | Daire geometrisi + doluluk: alan tam karede ve üstünde/yakınında cisim yoksa "uygun" |
| **RGB+Termal** | Modalite oto-tespit (640×512→termal); termal fine-tune (DroneVehicle/HIT-UAV) |

**Yarışmada internet yok** → tüm ağırlıklar önceden indirilip `agirliklar/` altında
çevrimdışı saklanır.

## Dış Veri Setleri (mutlaka — ön-eğitim/fine-tune)

| Set | İçerik | Rol | Script |
|---|---|---|---|
| **VisDrone** | ~10k görüntü, 2.6M kutu, RGB | Birincil araç/insan | `veri/indir_visdrone.py` |
| **DroneVehicle** | 28k RGB+**termal eşli** | Termal modalite (kilit) | `veri/indir_dronevehicle.py` |
| **HIT-UAV** | ~2.9k termal | Termal insan/araç | `veri/indir_hituav.py` |
| **TEKNOFEST 2022** | ~25k kuşbakışı (Türkçe yarışma) | En yakın dağılım | `veri/indir_teknofest2022.py` |

Sınıf eşleme: `veri/siniflar.py`. UAP/UAİ açık sette yok → örnek veri + sentetik artırma.

## Kurulum

```bash
# Blackwell GPU (RTX 5060) için CUDA 12.8'li torch:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r gorev1_nesne_tespiti/requirements.txt
```

## Çalıştırma

```bash
# Baseline duman testi (COCO-pretrained YOLOv11+SAHI; örnek karede tespit)
python gorev1_nesne_tespiti/baseline_demo.py --kare 8 --baslangic 900

# Dış veri indir (internetli makinede, yarışmadan önce)
python gorev1_nesne_tespiti/veri/indir_visdrone.py

# Fine-tune (4-sınıf birleşik veri yaml'ı ile)
python gorev1_nesne_tespiti/egitim/train.py --data datasets/g1_birlesik.yaml --epochs 100 --imgsz 1280

# Birim testler
python gorev1_nesne_tespiti/tests/test_g1.py
```

## Modüller

| Yol | Sorumluluk |
|---|---|
| `configs/config.py` | Yollar, model, SAHI, eşikler |
| `modeller/yolo_sahi.py` | YOLOv11 + SAHI sarıcı (`Detector`) |
| `hareket/durum.py` | Hareketli/sabit (ego-motion artık hareketi) |
| `inis/uap_uai.py` | UAP/UAİ daire + iniş-uygunluk (doluluk) |
| `veri/siniflar.py` | Yarışma sınıfları + dış set eşlemeleri |
| `veri/indir_*.py` | Dış veri indirme scriptleri |
| `egitim/train.py` | Fine-tune scripti |
| `entegrasyon/g1_model.py` | Tespit → `DetectedObject` payload |
| `baseline_demo.py` | Uçtan uca duman testi |

`ortak/ego_hareket` (homografi) Görev 2 VO ile paylaşılır — görevler bağlantılıdır.

## Durum

- ✅ Baseline boru hattı: YOLOv11+SAHI + hareket + iniş + payload (COCO-pretrained).
- ⏳ Sonraki: dış veri indir → fine-tune → mAP ölç → termal füzyon → UAP/UAİ sentetik.
