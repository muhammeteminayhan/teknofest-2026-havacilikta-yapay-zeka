# TEKNOFEST 2026 — Havacılıkta Yapay Zeka Yarışması

Bu paket, yarışmanın **üç görevini tek bir canlı akış içinde** çalıştıran çözümün
kaynak kodudur.

> ⚡ **Yarışma günü buradan başla → [`YARISMA_KULLANIM.md`](YARISMA_KULLANIM.md)**
> Adım adım çalıştırma, ekranda ne görüleceği ve on maddelik sorun giderme listesi.
>
> Bu klasör **çalışır hâldedir**: model ağırlıkları (`gorev1/agirliklar/best.pt`,
> `modeller/`) ve sunucu bilgileri (`sistem/config/.env`) içindedir. Jüriye
> gönderilecek teslim paketi bunların çıkarılmış hâlidir.

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

**Çözünürlük:** `imgsz=640`. Ölçtüğümüz kural: bir model hangi çözünürlükte
**eğitildiyse** çıkarımda da orada en iyi çalışır — özellikle ViT tabanlı gövdelerde,
çünkü konumsal gömme eğitim çözünürlüğüne göre öğrenilir. Teslim edilen ağırlık
checkpoint'inin `train_args` alanına göre 640'ta eğitilmiştir ve A/B ölçümü bunu
doğruladı (74 gerçek yarışma karesi, aynı üretim yolu): 640'ta net **+86 / −36**,
1280'de **+40 / −56**.

### 4.2 Görev 2 — Konum Kestirimi (`gorev2/src/`)

| Dosya | İş |
|---|---|
| `live_estimator.py` | Canlı akış sarmalayıcısı, motor zinciri (RAFT → SIFT → LK) |
| `vo_pipeline.py` | Ölçüm + durum makinesi (asıl mantık) |
| `raft_matching.py` | RAFT yoğun optik akış → nokta eşleşmeleri (birincil) |
| `sift_matching.py` | SIFT takipçisi (GPU yoksa CPU yedeği) |
| `geometry.py` | Homografi / benzerlik kestirimi ve ayrıştırma |
| `scale_alignment.py` | Sağlıklı dönemde ölçek ve eksen hizalaması öğrenme |
| `calibration.py`, `matching.py` | Kamera profilleri (4 profil), klasik LK yedeği |

**Motor zinciri.** `raft` (GPU + yerel ağırlık) → `sift` (CPU) → `lk` (son çare).
CPU yedeği neden SIFT: üç yatakta ortalama hata LK füzyon **15,86 m**, LK
homografi+merdiven 12,36 m, **SIFT homografi+merdiven 9,21 m** (kıyas: RAFT 9,72 m).

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
4. **Hizalama bütünlüğü.** Öğrenilen köprünün GT'ye ne kadar oturduğu ölçülür
   (`residual_ratio`); **0,30**'un üstündeyse hizalama reddedilir ve önceki geçerli
   hizalama korunur. Bozuk kalibrasyonla ölü hesap yapmayı engelleyen ana koruma budur.
5. **Yedekler.** Ölçüm kalite kapılarından geçemezse (az eşleşme, düşük inlier, donmuş
   kare) sabit-hız kestirimine düşülür; RAFT yüklenemezse SIFT'e, o da olmazsa LK'ye.

**Sevk edilen ayarlar:** `method="homografi"` · `spacing=8` · `min_inlier_ratio=0,30` ·
`residual_gate=0,30` · `smooth_alpha=0,6` · `z_min_hareket=2,0` · `z_min_kanit=0,30`.

> **Nasıl doğrulandı.** Bütün eski ayar çalışması `THYZ_2026_Ornek_Veri_1.MP4` üzerinde
> yapılmıştı ve o video, çevrim içi yarışmanın **test oturumunun ta kendisi**. Tek uçuşta
> ölçülen kazancın genellenmediği ölçülerek görüldü (düşey kanal denemesi: örnek veride
> −%25,6, puanlı oturumun kendi karelerinde iki pencerede de kayıp → geri alındı).
>
> Bunun üzerine her değişiklik iki bağımsız kapıdan geçiriliyor:
> **(1)** puanlı oturumun kendi kareleri — sağlıklı bir blok yapay olarak kesintiye
> çevrilip gerçek hata ölçülür; **(2)** 2024 ve 2025 veri setlerinin beş uçuşu — ayarın
> hiç yapılmadığı yataklar. RANSAC'ın nokta sırasına duyarlılığı ölçüldüğü için (tek
> koşuda ±%18,7 gürültü) karşılaştırmalar **eşleştirilmiş permütasyonlarla** yapılır.
>
> Bu kapılardan geçen turlar: `residual_gate` 0,50 → 0,30 ve `min_inlier_ratio`
> 0,40 → 0,30; ardından **z işareti kanıt kapısı** (13 Ağu 2026).
>
> **z işareti kanıt kapısı.** Dünyanın z ekseninin aşağı mı yukarı mı baktığı tek bir
> bittir (`z_sign`), ama yanlışsa düşey kanal komple ters döner. Eski kural bu biti
> **ölçümün kendi büyüklüğüne** bakarak seçiyordu; düz uçuşta ölçüm gürültüsü eşiği
> geçtiği için karar pratikte yazı-turaydı ve yataklarda %26–40 oranında kararsızdı.
> Yeni kural kararı ancak kalibrasyon penceresinde **gerçek irtifa hareketi** (≥2 m)
> ve tutarlı kanıt varsa verir; yoksa önceki işaret korunur.
> Ölçüldü (3 permütasyon, eşleştirilmiş): sınavın 1. penceresi **17,26 → 11,46 m**,
> 2. pencere değişmedi; yedi yatağın ortalaması **8,12 → 6,69 m**; hiçbir yatakta
> kötüleşme ve felaket yok. Eşikler 0,5–5,0 m aralığında sonucu değiştirmiyor.
>
> ⚠️ Daha önce bu satırda yazan *"beş uçuşta 35,8 → 22,2 m"* rakamı **geçersizdir**:
> 2025 termal CSV'leri veri setinde çapraz adlandırılmış ve o iki yatak anlamsız
> ölçüyordu. Düzeltilmiş taban beş uçuşta 8,4 m'dir.

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
  Görev 1 ve 2 çalışmaya devam eder; RAFT kurulamazsa SIFT'e, o da olmazsa LK'ye düşülür;
  ölçüm geçersizse sabit-hız yedeği devreye girer; hizalama bozuksa reddedilip önceki
  kalibrasyon korunur. Bir bileşenin arızası oturumu düşürmez.
- **Her değişiklik iki bağımsız kapıdan geçer.** Puanlı oturumun kendi kareleri ve
  2024/2025 veri setlerinin beş uçuşu. Tek uçuşta ölçülen kazanç kazanç sayılmaz —
  bu kural, ölçülüp geri alınan bir iyileştirmeden sonra kondu (bkz. §4.2).
- **Aynı karenin iki kez işlenmemesi.** Sunucu aynı kareyi tekrar verebildiği için son
  işlenen kare adresi tutulur; tekrar gelirse Görev 2 durum makinesi ilerletilmez ve
  önceki konum yeniden gönderilir.
- **Süre bütçesi.** 2250 karelik oturum 90 dakikalık pencerede tamamlanmalıdır; ölçülen
  toplam süre bu sınırın altındadır.
