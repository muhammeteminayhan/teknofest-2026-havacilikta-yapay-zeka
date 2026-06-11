# TEKNOFEST 2026 — Havacılıkta Yapay Zeka

**Takım:** muhammeteminayhan  
**Yarışma:** TEKNOFEST 2026 Havacılıkta Yapay Zeka  
**Simülasyon tarihi:** 09.07.2026  

---

## Görevler

| Görev | Açıklama | Durum |
|-------|----------|-------|
| **Görev 1** | [Nesne Tespiti](gorev1_nesne_tespiti/) | Devam ediyor |
| **Görev 2** | [Pozisyon Tespiti / Visual Odometry](gorev2_visual_odometry/) | ✅ Tamamlandı |

---

## Görev 2 — Visual Odometry Sonuçları

Alt-görüş kameradan GPS olmadan `x,y,z` pozisyon kestirimi. Batch global optimizasyon ile:

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
# Repoyu klonla
git clone https://github.com/muhammeteminayhan/teknofest-2026-havacilikta-yapay-zeka.git
cd teknofest-2026-havacilikta-yapay-zeka

# Bağımlılıkları kur
pip install -r gorev2_visual_odometry/requirements.txt

# Veri setini kur — sadece MP4 dosyaları gitignored, manuel olarak koy:
#   THYZ_2026_Ornek_Veri_Seti/THYZ_2026_Ornek_Veri_1.MP4
#   THYZ_2026_Ornek_Veri_Seti/THYZ_2026_Ornek_Veri_2_Termal.MP4
# (CSV ve kalibrasyon dosyaları repoda zaten mevcut)

# Testleri çalıştır (veri seti gerekmez — tamamen sentetik)
python gorev2_visual_odometry/tests/test_basic.py
```

---

## Görev 2 Çalıştırma

```bash
# Batch değerlendirme (önerilen) — global optimizasyon + resmi metrik
python gorev2_visual_odometry/scripts/evaluate_batch.py \
    --modality rgb --step 4 --health-mode intermittent --dropout 300 --cache

# Entegrasyon kuru provası (sunucu olmadan)
python gorev2_visual_odometry/scripts/replay_batch.py \
    --step 4 --health-mode intermittent

# Yarışma günü (tüm kareleri indir → optimize → gönder)
python gorev2_visual_odometry/integration/main_batch.py
```

---

## Katkı Kılavuzu

```
main     ← stabil, yarışma günü kullanılacak
develop  ← entegrasyon dalı
feature/ ← bireysel geliştirme
```

- `main`'e sadece PR ile merge edilir (1 onay gerekli)
- PR açmadan önce: `python gorev2_visual_odometry/tests/test_basic.py` → 7/7 geçmeli
- Büyük değişikliklerde `evaluate_batch.py` ile E değeri kontrol edilmeli

---

## Bağımlılıklar

- Python 3.11+
- opencv-python ≥ 4.10, numpy ≥ 1.24, scipy, matplotlib, requests, tqdm, pillow
- GPU gerekmez (klasik hat CPU'da ~35 kare/s)

Detaylı şartname: [`docs/TEKNIK_SARTNAME.pdf`](docs/TEKNIK_SARTNAME.pdf)
