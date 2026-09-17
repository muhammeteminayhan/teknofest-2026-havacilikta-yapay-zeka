# TEKNOFEST 2026 — Havacılıkta Yapay Zeka Yarışması

Bu paket, yarışmanın **üç görevini tek bir canlı akış içinde** çalıştıran çözümün
kaynak kodudur.

| Görev | İş | Yaklaşım |
|---|---|---|
| **1 — Nesne Tespiti** | Taşıt / İnsan / UAP / UAİ tespiti, hareket durumu, iniş uygunluğu | DINOv3 gövdeli YOLO + BoT-SORT takibi |
| **2 — Konum Kestirimi** | GPS kesintisinde aracın konumunu görüntüden kestirmek | RAFT optik akış + düzlemsel homografi + keyframe merdiveni |
| **3 — Görüntü Eşleme** | Oturum başında verilen referans nesneleri karelerde bulmak | DINOv3 CLS token ↔ yama kosinüs benzerliği (çok ölçekli) |

Sunucudan gelen her kare için üçü sırayla çalışır ve tek bir tahmin paketi gönderilir.

---

## 1. Kurulum

### 1.1 Gereksinimler
- **Python 3.11+** (test edilen: 3.13)
- NVIDIA GPU + CUDA 12.8 sürücüsü **önerilir** (CPU'da da çalışır, bkz. §1.3 notu)

### 1.2 Sanal ortam ve paketler

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 1) ONCE torch + torchvision (CUDA surumu):
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128

# 2) SONRA diger bagimliliklar:
python -m pip install -r requirements.txt
```

> **Sıra önemlidir.** `torch`/`torchvision` CUDA sürümü ayrı bir indeksten kurulur;
> `requirements.txt` bunları içermez. Önce düz PyPI'dan kurulursa CPU sürümü gelir.

**CPU-only makine:** `--index-url` olmadan `pip install torch torchvision` yeterlidir.
Sistem GPU bulamazsa otomatik CPU'ya düşer; doğruluk aynıdır ancak kare başı süre
yarışma bütçesini aşabilir.

### 1.3 Model dosyaları

Model dosyaları boyut sınırı nedeniyle Google Drive'dadır — **`model.txt`** dosyasındaki
bağlantıyı ve yerleştirme talimatını izleyin. Özetle:

| Dosya | Konum |
|---|---|
| `best.pt` (Görev 1) | `gorev1/agirliklar/best.pt` |
| `raft_large_C_T_SKHT_V2-ff5fadd5.pth` | `~/.cache/torch/hub/checkpoints/` |
| `models--facebook--dinov3-vits16-pretrain-lvd1689m/` | `~/.cache/huggingface/hub/` |

Son ikisi normalde otomatik iner; **DINOv3 erişim onayı gerektiren (gated) bir model
olduğu için** ve yarışma ortamında internet garanti olmadığı için pakete dahil edilmiştir.

### 1.4 Sunucu bilgileri

Takım kimlik bilgileri **`sistem/config/.env` içinde hazır gelir** — ek düzenleme
gerekmez. İçerik:

```
TEAM_NAME=<takım adı>
PASSWORD=<takım şifresi>
EVALUATION_SERVER_URL="http://<yarışma-sunucusu>/"
SESSION_NAME=<oturum adı>
```

> Sunucu adresi veya oturum adı yarışma günü değişirse, yalnız `.env` içindeki ilgili
> satır güncellenir. Şablon olarak `sistem/config/example.env` de pakettedir.

### 1.5 Kurulum doğrulaması

Sunucuya bağlanmadan, her şeyin yerli yerinde olduğunu sınar:

```bash
python dogrula.py
```

Üç görevin de yüklendiğini, Görev 2'nin `motor=raft` seçtiğini ve ağırlıkların
bulunduğunu ekrana yazar. Hata varsa hangi dosyanın eksik olduğunu söyler.

---

## 2. Bağımlılıklar ve sürümler

Yarışmada kullanılan ve test edilen sürümler:

| Paket | Sürüm | Nerede kullanılıyor |
|---|---|---|
| torch | 2.11.0+cu128 | tüm modeller |
| torchvision | 0.26.0+cu128 | Görev 2 — RAFT optik akış |
| ultralytics | 8.4.92 | Görev 1 — YOLO + BoT-SORT |
| transformers | 5.13.0 | Görev 1/3 — DINOv3 omurgası |
| numpy | 2.4.6 | tümü |
| opencv-python | 4.11.0.86 | görüntü işleme, homografi |
| scipy | 1.18.0 | Görev 3 — ısı haritası ölçekleme |
| PyYAML | 6.0.3 | Görev 3 — yapılandırma |
| requests | 2.34.2 | sunucu iletişimi |
| python-decouple | 3.8 | `.env` okuma |
| tqdm | 4.68.3 | ilerleme çubuğu |

`ultralytics` **standart PyPI sürümüdür**; özel bir fork gerekmez (bkz. §4.1).

---

## 3. Çalıştırma

```bash
cd sistem
python main.py
```

Akış: sunucuya kimlik doğrulama → oturum referanslarını indirme → her kare için
üç görevi çalıştırıp tahmin gönderme. Kayıtlar `sistem/_logs/` altına yazılır.

`main.py` içindeki `MIN_FRAME_INTERVAL = 0.75` saniyelik alt sınır, sunucunun
dakikada 80 istek kuralını aşmamak içindir.

---

## 4. Kod akışı

```
main.py
  ├── ConnectionHandler ............ oturum açma, kare çekme, tahmin gönderme
  ├── register_references() ........ oturum başındaki referans nesneleri kaydet (Görev 3)
  └── her kare için:
        ObjectDetectionModel.process()
          ├── Görev 1: gorev1/dedektor.py       → DetectedObject listesi
          ├── Görev 2: gorev2/src/live_estimator → DetectedTranslation (x, y, z)
          └── Görev 3: gorev3/eslestirici.py    → ReferencePrediction listesi
```

Yarışmacı tarafından düzenlenen tek dosya `sistem/src/object_detection_model.py`'dir;
bağlantı katmanı (`connection_handler.py`, `frame_predictions.py`, `constants.py` …)
resmî arayüzden birebir alınmıştır.

### 4.1 Görev 1 — Nesne Tespiti (`gorev1/`)

| Dosya | İş |
|---|---|
| `dedektor.py` | Kare başına tespit + takip + durum üretimi |
| `dinov3_modul.py` | DINOv3 gövde modülü (fork bağımlılığını kaldırır) |

**Akış:** `model.track()` (BoT-SORT) ile tespit ve kimlik takibi → sınıfa göre işlem:

- **Taşıt** → hareket durumu. Kamera da hareket ettiği için önce **ego-hareket telafisi**
  yapılır: ardışık kareler arasında optik akış + RANSAC ile homografi bulunur, taşıtın
  merkezi bu homografiyle önceki kareye taşınır; kalan kayma eşiği aşarsa "hareketli".
- **UAP / UAİ** → iniş uygunluğu. Alan kare kenarına taşıyorsa veya içinde taşıt/insan
  varsa "inilemez", aksi hâlde "inilebilir".
- **Sınıf takası:** model `{0:Taşıt, 1:İnsan, 2:UAİ, 3:UAP}` sırasıyla eğitilmiştir;
  şartname `{2:UAP, 3:UAİ}` beklediği için `_spec_cls()` 2↔3 dönüşümü yapar.

**Fork bağımlılığı çözümü:** DINOv3 gövdeli ağırlık, `ultralytics.nn.modules.dinov3`
sınıfına pickle bağımlıdır ve bu modül yalnız özel bir ultralytics fork'unda bulunur.
Fork'u kurmak yerine (kurulu sürümü geriye alır ve DINOv3 kullanan Görev 3'ü de riske
atardı) modülün kopyası `dinov3_modul.py` olarak pakete konmuş, model yüklenirken
`sys.modules`'e enjekte edilmiştir. Böylece **standart ultralytics** ile çalışır.

**Çözünürlük:** `imgsz=1280`. Yarışma kareleri 1920×1080 ve hedefler küçüktür (uzaktaki
insan ~20×20 px; 640'a küçültülünce ~7×7 px'e iner ve kaybolur). Ölçtüğümüz kural: bir
model hangi çözünürlükte **eğitildiyse** çıkarımda da orada en iyi çalışır — özellikle
ViT tabanlı gövdelerde, çünkü konumsal gömme eğitim çözünürlüğüne göre öğrenilir.

### 4.2 Görev 2 — Konum Kestirimi (`gorev2/src/`)

| Dosya | İş |
|---|---|
| `live_estimator.py` | Canlı akış sarmalayıcısı, motor seçimi (RAFT / LK) |
| `vo_pipeline.py` | Ölçüm + durum makinesi (asıl mantık) |
| `raft_matching.py` | RAFT yoğun optik akış → nokta eşleşmeleri |
| `geometry.py` | Homografi / benzerlik kestirimi ve ayrıştırma |
| `scale_alignment.py` | Sağlıklı dönemde ölçek ve eksen hizalaması öğrenme |
| `calibration.py`, `matching.py` | Kamera profili, klasik LK yedeği |

**Temel fikir:** GPS varken (sağlıklı kare) sunucunun verdiği konum aynen döndürülür ve
bu sırada sistem **öğrenir**; GPS kesildiğinde görüntüden kestirim yapılır.

1. **Ölçüm.** İki kare arasında RAFT ile yoğun optik akış çıkarılır, ızgara noktaları
   örneklenir. Yerden bakan kamerada sahne büyük ölçüde düzlemsel olduğu için hareket
   **düzlemsel homografi** ile modellenir: `H = K(R − t·nᵀ/d)K⁻¹`. Ayrıştırma, yatay
   yer değiştirme ve yön değişimini verir.
2. **Ölçek.** Görüntüden gelen hareket ölçeksizdir. Sağlıklı dönemde görüntü ölçümleri
   ile gerçek GPS adımları eşleştirilip **Procrustes/Kabsch** ile eksen dönüşümü ve
   metrik ölçek öğrenilir (`scale_alignment.py`).
3. **Keyframe merdiveni.** Kesinti boyunca hata birikmesin diye her kare bir öncekine
   değil, sabit aralıklı bir **keyframe'e** göre konumlanır (`VoLadderEstimator`,
   spacing=8). Uzun taban çizgisi sürüklenmeyi kırar; iki ölçüm birbirini denetler.
4. **Yedekler.** Ölçüm kalite kapılarından geçemezse (az eşleşme, düşük inlier, donmuş
   kare) sabit-hız kestirimine düşülür; RAFT yüklenemezse klasik Lucas-Kanade'ye.

> **Geri alınan iki deneme — ve nedeni.** Çevrim içi oturum sonrası iki iyileştirme
> denendi: (1) düşey bileşeni homografi yerine benzerlik dönüşümünün ölçeğinden almak,
> (2) ölçüm yumuşatmasını kapatmak. İkisi de **örnek veri setinde** ölçülebilir kazanç
> veriyordu (sırasıyla −%25,6 ve −%4,2).
>
> Ancak bu ayar verisi, çevrim içi yarışmanın **test oturumu videosunun kendisiydi**;
> puanlı oturum başka bir uçuştur. İkisini puanlı oturumun kendi kareleriyle sınadık:
> sağlıklı bir bloğu yapay olarak kesintiye çevirip (sunucunun verdiği konumu sistemden
> saklayarak) gerçek hatayı ölçtük. Sonuç — düşey kanal iki bağımsız pencerede de
> kaybetti (16,04 → 20,07 m ve 18,46 → 18,62 m, zarar tamamen z ekseninde), yumuşatma
> kapatma ise pencerelerde ters yönde çıktı ve ortalamada geride kaldı (17,25 vs 16,71 m).
>
> **İkisi de geri alındı.** Teslim edilen sistem, puanlı oturumda koşan reçetenin
> kendisidir. Sınav aracı: `gorev2/genelleme_sinavi.py` (geliştirme deposunda).

### 4.3 Görev 3 — Görüntü Eşleme (`gorev3/`)

| Dosya | İş |
|---|---|
| `eslestirici.py` | Referans kaydı, kare-referans eşleme sarmalayıcısı |
| `template_matcher.py` | Eşleme çekirdeği (ısı haritası → kutu) |
| `dino_backbone.py` | DINOv3 omurgası (minimal, fork'suz) |
| `template_config.yaml` | Eşik ve ölçek ayarları |

**Zero-shot:** referans nesneler modelde eğitilmemiştir, görsel benzerlikle bulunur.

1. Oturum başında her referans görseli DINOv3'ten geçirilir, **CLS token**'ı saklanır.
2. Her karede görüntü patch'lere ayrılır; her patch'in referans CLS token'ıyla
   **kosinüs benzerliği** hesaplanır → ısı haritası.
3. Harita üç ölçekte (672/448/224) üretilip birleştirilir; ayrıca **SAHI tarzı tiling**
   ile kare karolara bölünüp küçük nesneler büyütülerek aranır.
4. Isı haritası istatistiksel eşikle (`ortalama + k·σ`) kesilir, morfolojik kapama
   uygulanır, tepe noktasını içeren kontur kutuya çevrilir.
5. **Arama penceresi:** sunucu her referans için `frame_start`–`frame_end` aralığı verir;
   referans yalnız kendi aralığında aranır. Bu, aralık dışı yanlış pozitifleri tamamen
   önler.

---

## 5. Tasarım kararları (özet)

- **Çevrimdışı çalışabilirlik.** Tüm model ağırlıkları yerelden yüklenir; yarışma
  sırasında internet gerekmez.
- **Çökmeyen tasarım.** Her görev kendi `try/except` sınırında kurulur: Görev 3 yüklenemezse
  Görev 1 ve 2 çalışmaya devam eder; RAFT yoksa LK'ye düşülür; ölçüm geçersizse
  sabit-hız yedeği devreye girer. Bir bileşenin arızası oturumu düşürmez.
- **Aynı karenin iki kez işlenmemesi.** Sunucu aynı kareyi tekrar verebildiği için son
  işlenen kare adresi tutulur; tekrar gelirse Görev 2 durum makinesi ilerletilmez ve
  önceki konum yeniden gönderilir.
- **Süre bütçesi.** 2250 karelik oturum 90 dakikalık pencerede tamamlanmalıdır; ölçülen
  toplam süre bu sınırın altındadır.
