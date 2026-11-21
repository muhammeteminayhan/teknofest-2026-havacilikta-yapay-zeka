# sistem/ — Canlı Bağlantı Arayüzü

Yarışma değerlendirme sunucusuyla (THYZDS) konuşan katman. TEKNOFEST'in verdiği örnek
arayüz temel alındı; üç görev buraya bağlandı ve gerçek sunucu bulgularına göre düzeltildi.

**Akış:** `main.py` → sunucudan kare al → `ObjectDetectionModel.process()` → görseli indir
→ `detect()` üç görevi çalıştırır → payload → `POST /prediction` (201).

## Çalıştırma

```powershell
# 1) kimlik bilgileri (bir kez) — gerçek .env depoda yok, .gitignore'da
Copy-Item config\example.env config\.env
notepad config\.env          # takım adı / şifre / sunucu URL'si

# 2) sistem/ klasöründen çalıştır
cd sistem
..\.venv\Scripts\python main.py
```
Loglar `_logs/` altına, indirilen görseller `_images/` altına yazılır (ikisi de otomatik
oluşur; silinebilir, yeniden oluşur).

## Görevlerin bağlandığı yer: `src/object_detection_model.py`

`ObjectDetectionModel.__init__` üç modeli **bir kez** yükler:
- `self.position_estimator` — Görev 2 (RAFT+homografi, `gorev2/`)
- `self.gorev1` — Görev 1 dedektörü (`gorev1/`)
- `self.gorev3` — Görev 3 eşleştirici (`gorev3/`)

Her biri try/except ile yüklenir: biri yüklenemezse **diğerleri devam eder** (örn. GPU
yoksa Görev 2 LK yedeğine düşer, DINOv3 önbelleği yoksa Görev 1/2 çalışmaya devam eder).

`detect()` sırayla: Görev 1 tespitleri → `DetectedObject`; Görev 3 eşleşmeleri →
`DetectedUndefinedObject`; Görev 2 → `DetectedTranslation`. Boş liste geçerlidir
(**sahte kutu asla gönderilmez** — yanlış pozitif skoru düşürür).

## Gelen ve giden veriyi nerede görürüm?

| Ne | Nerede |
|---|---|
| **Gelen: kare listesi** | `_images/<oturum>/frames.json` (url, image_url, video_name, references) |
| **Gelen: translation + health** | `_images/<oturum>/translations.json` (translation_x/y/z, health_status) |
| **Gelen: görüntüler** | `_images/<oturum>/*.webp` |
| **Giden: tam payload** | `_logs/<takım>_<tarih>.log` → `Prediction sent successfully` satırının altındaki JSON |
| **Kare-kare özet** | `_logs/*.log` → `[Gorev2 echo/TAHMIN] konum=(x,y,z) \| [Gorev1] N nesne \| [Gorev3] M eşleşme` |

**Görev 2'yi izlemek:** her karede log dosyasına bir özet satırı yazılır. `health_status='1'`
(sağlıklı) karelerde Görev 2 gelen konumu **echo** eder (0 hata garantisi); `'0'` (kesinti)
karelerde RAFT ile **TAHMIN** eder. `_logs` ve `_images` klasörleri `main.py` çalışınca otomatik
oluşur (yoksa boştur).

## Gerçek sunucu bulguları (kanıtlanmış — `format_probe.py` ile 201)

Örnek arayüz eski/eksikti; şunlar gerçek sunucuda test edilip düzeltildi:

1. **`/media` görsel indirmesi AUTH TOKEN ister.** Tokensiz 401. `download_image` artık
   `Authorization: Token ...` gönderir; `main.py`, `detection_model.auth_token`'ı doldurur.
2. **`cls` alanı URL formatında:** `classes/{id+1}/` (sunucu 1-endeksli; örn Taşıt=0 →
   `classes/1/`). `detected_object.py`'deki `PAYLOAD_CLS_URL` bayrağı bunu kontrol eder.
3. **Alan adı `moving_status` — `motion_status` DEĞİL!** Şartname Şekil 17 `motion_status`
   diyor ama sunucu `moving_status` bekliyor; yanlış anahtar sessizce düşüyordu (tüm taşıt
   hareket puanı kaybı). `detected_object.py` düzeltildi.
4. **Görev 3 referansları** frames-JSON'daki `references` anahtarından gelir (test
   oturumunda boştu). `object_detection_model.register_references(...)` bunları kaydeder.
5. Görseller `.webp` 1920×1080. `health_status` string ('1'/'0') gelebilir → `str()` ile
   karşılaştırılır. Gönderim hızı **80/dk** ile sınırlı (`main.py`'de RateLimiter).

## Dosyalar
```
sistem/
├── main.py                         # ana döngü + RateLimiter(80/dk)
├── baglanti_testi.py               # hafif ağ testi (RAFT'sız): login/şema/media/echo tahmin
├── format_probe.py                 # tespit payload formatını gerçek sunucuda doğrular
├── config/  .env (.online/example)             # sunucu bilgileri
└── src/
    ├── object_detection_model.py   # ★ üç görevin bağlandığı yer (buraya müdahale edersiniz)
    ├── connection_handler.py       # login/frames/translations/prediction (DEĞİŞTİRMEYİN)
    ├── frame_predictions.py        # payload birleştirme (+ detected_undefined_objects)
    ├── detected_object.py          # Görev 1 nesnesi (cls URL, moving_status)
    ├── detected_undefined_object.py# Görev 3 nesnesi
    ├── detected_translation.py     # Görev 2 çıktısı
    └── constants.py                # sınıf/URL sabitleri (DEĞİŞTİRMEYİN — uyarılı)
```

## Yardımcı test araçları
- `baglanti_testi.py` — modelleri yüklemeden sadece **ağ katmanını** test eder (hızlı,
  sunucu erişimi + şema + 201 doğrular). İlk denemede bunu çalıştırın.
- `format_probe.py` — tek bir tespit gönderip sunucunun kabul ettiği alan adlarını yazdırır.
