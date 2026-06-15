# TEKNOFEST 2026 — Havacılıkta Yapay Zeka

**Takım:** muhammeteminayhan  
**Yarışma:** TEKNOFEST 2026 Havacılıkta Yapay Zeka  
**Simülasyon tarihi:** 09.07.2026  

Alt-görüş hava kamerasından (FHD/4K, 7.5 fps, RGB *veya* termal) gelen kareler üzerinde
üç görev: nesne tespiti, pozisyon kestirimi ve görüntü eşleme. Üç görev de **aynı video
akışını ve bağlantı arayüzünü** paylaşır; kare başına **tek birleşik sonuç paketi** gönderilir.

---

## Görevler ve Puan

| Görev | Açıklama | Puan | Durum |
|-------|----------|------|-------|
| **Görev 1** | [Nesne Tespiti](gorev1_nesne_tespiti/) (taşıt/insan/UAP/UAİ + hareket/iniş) | %25 | 🟡 Baseline çalışıyor |
| **Görev 2** | [Pozisyon Tespiti / Visual Odometry](gorev2_visual_odometry/) | %40 | ✅ Tamamlandı |
| **Görev 3** | [Görüntü Eşleme](gorev3_goruntu_esleme/) (referans nesne) | %25 | ⚪ Planlandı |
| Rapor + Sunum | — | %10 | — |

> **Görevler bağlantılı:** ortak ego-hareket/kamera/veri katmanı `ortak/` altında.
> Görev 1'in "hareketli/sabit" ayrımı, Görev 2'nin görsel-odometri homografisini paylaşır.

---

## Repo Yapısı

```
ortak/                     # paylaşılan çekirdek (ego_hareket, veri, kamera)
gorev1_nesne_tespiti/      # YOLOv11 + SAHI; hareket + iniş + entegrasyon
gorev2_visual_odometry/    # ✅ hibrit VO + batch global optimizasyon
gorev3_goruntu_esleme/     # (planlandı) RoMa/MatchAnything eşleme
TAKIM_BAGLANTI_ARAYUZU/    # yarışma bağlantı arayüzü (değişmez)
THYZ_2026_Ornek_Veri_Seti/ # örnek veri (MP4'ler gitignore)
datasets/                  # dış veri setleri (gitignore; scriptlerle indirilir)
docs/                      # teknik şartname
```

---

## Görev 2 — Visual Odometry Sonuçları

GPS olmadan `x,y,z` pozisyon kestirimi. Batch global optimizasyon ile:

| GPS Kesinti Süresi | Çevrimiçi `E` | **Batch `E`** | Batch Medyan |
|--------------------|---------------|---------------|--------------|
| ~13 s (100 kare) | ~25 m | **4.2 m** | 1.6 m |
| ~27 s (200 kare) | ~38 m | **8.7 m** | 4.3 m |
| ~40 s (300 kare) | ~50 m | **22 m** | 10.5 m |
| ~60 s (450 kare) | — | **47 m** | 29.5 m |

Çapalanmış segmentlerde **5–50× iyileşme**. 7/7 birim test geçiyor.

---

## Kurulum

```bash
git clone https://github.com/muhammeteminayhan/teknofest-2026-havacilikta-yapay-zeka.git
cd teknofest-2026-havacilikta-yapay-zeka

# Görev 2 (klasik VO — GPU gerekmez)
pip install -r gorev2_visual_odometry/requirements.txt

# Görev 1 (YOLOv11+SAHI) — RTX 5060/Blackwell için CUDA 12.8'li torch:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r gorev1_nesne_tespiti/requirements.txt

# Örnek veri: MP4 dosyaları gitignored → THYZ_2026_Ornek_Veri_Seti/ altına manuel koy
# (CSV ve kalibrasyon repoda mevcut). MP4 yoksa baseline_demo örnek JPG'lere düşer.
```

> **Yarışmada internet YOK.** Tüm model ağırlıkları önceden indirilip `*/agirliklar/`
> altında çevrimdışı saklanmalıdır.

---

## Çalıştırma

```bash
# --- Görev 1: Nesne Tespiti ---
python gorev1_nesne_tespiti/baseline_demo.py --kare 8 --baslangic 900   # baseline duman testi
python gorev1_nesne_tespiti/veri/indir_visdrone.py                      # dış veri (internetli makine)
python gorev1_nesne_tespiti/tests/test_g1.py                            # birim testler

# --- Görev 2: Visual Odometry ---
python gorev2_visual_odometry/scripts/evaluate_batch.py --modality rgb --step 4 --health-mode intermittent --cache
python gorev2_visual_odometry/tests/test_basic.py
```

---

## Katkı Kılavuzu

```
main     ← stabil, yarışma günü kullanılacak
develop  ← entegrasyon dalı
feature/ ← bireysel geliştirme
```

- `main`'e sadece PR ile merge edilir.
- PR açmadan önce ilgili birim testler geçmeli:
  `gorev1_nesne_tespiti/tests/test_g1.py` ve `gorev2_visual_odometry/tests/test_basic.py`.

Detaylı şartname: [`docs/TEKNIK_SARTNAME.pdf`](docs/TEKNIK_SARTNAME.pdf)
