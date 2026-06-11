# Görev 2 — Visual Odometry (Pozisyon Kestirimi)

TEKNOFEST 2026 Havacılıkta Yapay Zekâ yarışması **İkinci Görev** (toplam puanın
**%40'ı**) için monoküler visual odometry çözümü.

Hava aracının konumlandırma sistemi (GPS) güvenilmez hale geldiğinde
(`health=0`), yalnızca alt-görüş kamera görüntülerinden aracın ilk kareye göre
`x, y, z` (metre) pozisyonu kestirilir. Puanlama, kestirim ile gerçek pozisyon
arasındaki **ortalama Öklid hatası** (`E`, düşük = iyi) ile yapılır.

## Yaklaşım (Hibrit klasik + öğrenilen-özellik-hazır)

1. **Kare-kare hareket** — KLT optik akış ile özellik takibi; ardından:
   - **Öteleme + irtifa** 2B benzerlikten (optik-eksen akışı; düşük gürültü),
   - **Yaw** homografi ayrıştırmasından (`decomposeHomographyMat`; perspektifi
     düzgün ayırır, açısal drift ↓). → `vo/motion.py` (`hybrid` model).
2. **Metrik ölçek + eksen hizalama** — monoküler VO ölçeksizdir. Sağlıklı
   (`health=1`) karelerdeki GT ile **2B Umeyama benzerlik hizalaması** yapılır →
   ölçek (≈irtifa) ve referans-eksen yönelimi öğrenilir. → `vo/aligner.py`
3. **XY / Z ayrıştırması** — sağlıklı segmentte GT-z neredeyse sabit olduğundan
   z gözlemlenemez; bu yüzden yer düzlemi (XY, 2B Umeyama ile iyi gözlemli) ve
   irtifa (Z, fiziksel işaretli proxy) **ayrı** ele alınır. → `vo/trajectory.py`
4. **Yeniden çapalama** — `health=1` olunca mutlak konum GT'ye sabitlenir, drift
   sıfırlanır; `health=0` boyunca son çapadan itibaren göreli VO entegre edilir.

## İki çalışma modu

- **Çevrimiçi (causal)** — her kareyi sadece geçmişle kestir, anında gönder.
  Basit; sunucu akış zorunlu kılarsa yedek. (`scripts/evaluate.py`, `integration/object_detection_model.py`)
- **Batch (toplu global optimizasyon) — ÖNERİLEN, maksimum doğruluk.** Yarışma
  arayüzü tüm kareleri + tüm health bitlerini baştan verdiğinden, tüm kareleri işle
  → **global çapalı smoothing** (her sağlıksız segment iki uçtaki GT çapalarıyla
  düzeltilir, drift dağıtılır) → sonra gönder. (`scripts/evaluate_batch.py`,
  `integration/main_batch.py`). Bkz. `vo/batch.py`, `vo/optimize.py`.

## Sonuçlar (örnek RGB veri, 7.5 fps ≈ `--step 4`)

İlk 450 kare sağlıklı (GT), sonrası senaryoya göre sağlıksız. Metrik yalnızca
`health=0` karelerde. **Batch**, çevrimiçiye göre belirgin üstün:

| Senaryo (GPS kesinti süresi) | Çevrimiçi `E` | **Batch `E`** | Batch medyan |
|---|---|---|---|
| Aralıklı, 100 kare (~13 s) | ~25 m | **4.2 m** | 1.6 m |
| Aralıklı, 200 kare (~27 s) | ~38 m | **8.7 m** | 4.3 m |
| Aralıklı, 300 kare (~40 s) | ~50 m | **22 m** | 10.5 m |
| Aralıklı, 450 kare (~60 s) | — | **47 m** | 29.5 m |
| Worst-case (1806 kare KESİNTİSİZ, çapasız kuyruk) | ~179 m | ~179 m | 136 m |

- Batch, çapalanmış (bracketed) segmentlerde **5–50× iyileşme** sağlar; hata GPS
  kesinti süresine bağlıdır. Z ekseni her durumda ~1–3 m.
- **Çapasız kuyruk** (oturum sağlıksız biterse, prefix sonrası hiç çapa yok) batch
  ile düzelmez; bu, monoküler VO'nun temel drift sınırıdır → öğrenilen özellikler /
  döngü kapama (sonraki faz) gerekir.

## Kurulum

```bash
pip install -r requirements.txt
```

GPU gerekmez (klasik hat CPU'da ~35 kare/s @ 1280px). Faz 2+ öğrenilen özellikler
(SuperPoint/LightGlue) için opsiyonel `torch`/`kornia`/`lightglue`.

## Çalıştırma

```bash
# Faz 0 — veri/kalibrasyon doğrulama (kare sayısı == CSV satırı, undistort görseli)
python scripts/extract_frames.py --modality rgb

# BATCH değerlendirme (ÖNERİLEN) — global optimizasyon, resmi metrik + grafikler
python scripts/evaluate_batch.py --modality rgb --step 4 --health-mode intermittent --dropout 300 --cache
python scripts/evaluate_batch.py --modality rgb --step 4 --health-mode tail --cache
#   --cache: pass-1'i (yavaş) önbelleğe alır; optimizer parametrelerini hızlı dener
#   --align-mode interp|global, --stage2, --w-anchor, --dropout/--recover

# Çevrimiçi değerlendirme (yedek mod)
python scripts/evaluate.py --modality rgb --step 4 --health-mode intermittent

# Entegrasyon kuru provaları (sunucu olmadan; gerçek FramePredictions ile)
python scripts/replay_batch.py --step 4 --health-mode intermittent       # batch yolu
python scripts/replay_integration.py --step 4 --health-mode intermittent  # çevrimiçi yolu

# Birim testler
python tests/test_basic.py
```

## Yarışma Sunucusuna Entegrasyon

İki seçenek (arayüz dosyaları **değişmez**):

**A) Batch (ÖNERİLEN) — `integration/main_batch.py`:** `main.py` yerine çalışır.
Tüm kareleri indir → işle → global smoothing → 80/dk throttle ile gönder.
```bash
# TAKIM_BAGLANTI_ARAYUZU/config/.env doldurulmuş olmalı
python integration/main_batch.py        # (VO_MODALITY=thermal termal oturumda)
```

**B) Çevrimiçi — `integration/object_detection_model.py`:** drop-in; orijinali
yedekleyip `TAKIM_BAGLANTI_ARAYUZU/src/` altına kopyalayın, `python main.py` çalıştırın.
Gerekirse `VO_PACKAGE_PATH` ile paket yolunu verin.

Her iki modda health mantığı: `health=1` → GT echo; `health=0` → VO/batch kestirimi.
Çıktı `DetectedTranslation(x, y, z)` payload'a eklenir.

## Modüller

| Dosya | Sorumluluk |
|---|---|
| `configs/config.py` | Tüm yollar ve parametreler |
| `vo/calibration.py` | MATLAB kalibrasyon → OpenCV K/dist; çözünürlük ölçekleme; undistort |
| `vo/frame_source.py` | MP4 → sıralı kareler; GT CSV; 1:1 hizalama |
| `vo/frontend.py` | Ön-işleme (gri+CLAHE); KLT / ORB / SIFT eşleştirme |
| `vo/motion.py` | Göreli hareket: hibrit / homografi / benzerlik |
| `vo/aligner.py` | Umeyama benzerlik hizalaması (3B ve 2B, RANSAC) |
| `vo/trajectory.py` | Çevrimiçi yörünge birikimi + health durum makinesi + çapalama |
| `vo/pipeline.py` | `VOEstimator` — `process()` ve `compute_frame_motion()` (paylaşılan) |
| `vo/batch.py` | Batch pass-1 (FrameMotion toplama) + ham zincir + önbellek |
| `vo/optimize.py` | Global smoothing: segment-interp hizalama + tridiyagonal çapalı çözüm |
| `integration/object_detection_model.py` | Çevrimiçi drop-in |
| `integration/main_batch.py` | **Batch yarışma sürücüsü** (önerilen) |

## Sonraki Adımlar (iyileştirme yol haritası)

Batch global optimizasyon (interp hizalama + çapalı smoothing) tamamlandı ve klasik
ön-yüzden çıkarılabilecek doğruluk büyük ölçüde alındı. Çapasız kuyruk ve uzun
kesintiler için kalan kaldıraçlar:

- **Öğrenilen özellikler** (SuperPoint + LightGlue, GPU): per-frame drift hızını
  düşürür → uzun pencere/kuyrukta daha iyi (`frontend.py` arkasına aynı arayüzle).
- **Döngü kapama (loop closure)**: yörünge önceden GT-çapalı alanları yeniden ziyaret
  ettiğinde mutlak düzeltme (çapasız kuyruğa yardımcı). Yer tanıma + pose-graph kısıtı.
- **Termal modalite** (Faz 3): CLAHE + AKAZE; ayrı termal intrinsics (hazır).
- Not: Yüksek çözünürlük/daha çok özellik test edildi — drift sistematik (ölçek/heading)
  olduğundan piksel hassasiyeti darboğaz DEĞİL; bu yüzden öncelik öğrenilen özellikler
  ve döngü kapama.
