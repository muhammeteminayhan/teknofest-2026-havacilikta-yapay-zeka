# Görev 1 — trial-8 model değişimi (9 Eyl 2026)

**Değişen:** tespit ağırlığı (`trial-8/weights/best.pt`), `imgsz` 640 → **1280**, ve
sınıf eşlemesi sabit takas yerine **ağırlığın kendi adlarından** kuruluyor.
**Değişmeyen:** ego-hareket telafili hareket durumu, iniş uygunluğu, BoT-SORT takibi,
şartname çıktı şeması.

---

## 1. 🔴 Sınıf sırası değişti — sabit takas kaldırıldı

| ağırlık | sınıf 2 | sınıf 3 | şartnameye göre |
|---|---|---|---|
| `best_dinov3.pt` (önceki) | UAİ | UAP | **ters** |
| `best.pt` (YOLOv8s) | UAİ | UAP | **ters** |
| **`trial-8/weights/best.pt`** | **UAP** | **UAİ** | **birebir** |

Şartname (`sistem/src/constants.py`): `UAP: 2, UAİ: 3`.

Eski kodda sabit bir 2↔3 takası vardı ve **ağırlığa bağımlıydı**: trial-8 ile
kalsaydı UAP ve UAİ etiketleri ters gider, iki sınıf da sıfır alırdı.

**Çözüm:** `spec_sinif_eslemesi()` — ağırlığın `model.names`'ini okuyup şartname
indeksine **isimle** eşler. Türkçe karakterler normalize edilir (`UAİ`≡`UAI`,
`Taşıt`≡`Tasit`). Eksik/çakışık/tanımsız tablo **kurulumda `ValueError` fırlatır** —
sessizce yanlış etiketlemektense çökmek doğrudur.

Doğrulandı: eski ağırlıklarda eşleme `{0:0, 1:1, 2:3, 3:2}` (eski davranış **aynen
korunuyor**), trial-8'de `{0:0, 1:1, 2:2, 3:3}`. Üç bozuk tablo da reddedildi.

## 2. `imgsz` 640 → 1280

`trial-8/args.yaml` ve checkpoint `train_args`: **1280**. Önceki ağırlık 640'ta
eğitilmişti ve o dönem A/B ile 640 seçilmişti — model değişti, gerekçe de değişti.

Ölçülen süre: **125 ms/kare** (2400 ms bütçe). 2250 karelik oturumda ~5 dk.

## 3. A/B — 74 gerçek yarışma karesi, ham tespit (mAP'e yakın)

`ab_model.py --aday trial-8/weights/best.pt --imgsz-aday 1280 --mevcut
gorev1/agirliklar/best_dinov3.pt --imgsz 640 --takipsiz`

*(Harness'a kol başına ayrı `imgsz` ve `--mevcut` eklendi: her ağırlık KENDİ eğitim
çözünürlüğünde koşmalı.)*

| sınıf | MEVCUT | ADAY (trial-8) | ortak | yalnız aday | "gerçek kayıp" |
|---|---|---|---|---|---|
| Taşıt | 286 | 289 | 282 | 7 | 3 |
| **İnsan** | 20 | **54** | 1 | **53** | 17 |
| UAP | 24 | 12 | 12 | 0 | 12 |
| **UAİ** | 7 | **19** | 5 | 14 | 2 |
| **toplam** | 337 | **374** | | | |

Otomatik ölçüt iki RED verdi. **Görsel denetim ikisinin de yanlış alarm olduğunu
gösterdi** — deponun karar felsefesi gereği (sayısal öneri tek başına yeterli değil).

### ❌ RED 1: "kayıp oranı %10,1" → aslında **düzeltme**

Bütün UAP anlaşmazlığı tek bir 12 karelik blokta (670–681): mevcut 2 UAP, aday 1 UAP.
Ama **toplam iniş alanı sayısı iki modelde de 31.**

**Görsel kanıt:** [`trial8_kanit_uap_uai.png`](trial8_kanit_uap_uai.png) — sahnede iki
daire var ve **üzerlerinde yazı var**:
* **mavi daire, "UAP" yazılı** → her iki model de UAP ✓
* **kırmızı daire, "UAİ" yazılı** → mevcut model **UAP** diyor ❌ · trial-8 **UAİ** ✓

Yani mevcut model UAİ dairesini UAP sanıyor; trial-8 düzeltiyor. A/B'nin "12 gerçek
kayıp" saydığı şey **12 düzeltme**.

### ❌ RED 2: "yeni tespit boyutu 34×34 px" → gerçek insanlar

Ölçüt 16×16'lık gürültü kutularını yakalamak için konmuştu. Ama yeni tespitlerin 53'ü
**İnsan** ve bu irtifada bir insan zaten ~30×40 px.

| | İnsan kutu medyanı |
|---|---|
| trial-8 | **30 × 34,5 px** |
| best_dinov3 (mevcut) | 50 × 52 px |
| *(kayıtlı doğru boyut — YOLOv8s)* | *~29 × 43 px* |

**Görsel kanıt:** [`trial8_kanit_insan.png`](trial8_kanit_insan.png) — 12 kırpım,
hepsinde tepeden görünen gerçek insan (baş, omuz, uzuv seçiliyor).

## 4. Karar: trial-8 SEVK EDİLDİ

Önceki model değişiminde gerileyen **tam iki sınıfı onarıyor**:

| sınıf | YOLOv8s | best_dinov3 | **trial-8** |
|---|---|---|---|
| İnsan | 64 | 20 ⬇ | **54** ⬆ |
| UAİ | 12 | 7 ⬇ | **19** ⬆ |

Üstelik UAP/UAİ ayrımını doğru yapan **tek** model bu. Taşıt'ta da hafif önde (289 vs 286).

## Eğitim metrikleri (trial-8, kendi val kümesi)

mAP50 **0,814** · mAP50-95 **0,649** · precision **0,886** · recall **0,772**
(127 epoch, trial-3'ten devam, gövde donuk, `imgsz=1280`)

> Bu sayılar önceki ağırlıklarınkiyle **kıyaslanamaz** — farklı veri setleri
> (`combined_dataset`). Kıyas yalnız yukarıdaki 74 gerçek kare üzerinden yapılmıştır.

## Yeniden koşturmak için

```
.venv\Scripts\python gorev1\ab_model.py --aday trial-8\weights\best.pt --imgsz-aday 1280 ^
    --mevcut gorev1\agirliklar\best_dinov3.pt --imgsz 640 --takipsiz
```
