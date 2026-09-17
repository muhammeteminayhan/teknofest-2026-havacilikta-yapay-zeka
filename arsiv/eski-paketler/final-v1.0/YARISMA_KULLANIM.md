# Yarışma günü kullanım kılavuzu

> **Bu paket teslim paketi değildir.** Yarışma günü **çalıştırılacak** hâlidir:
> model ağırlıkları, takım şifresi ve sunucu bilgileri içindedir.
>
> Bu belge, sistemi **daha önce hiç çalıştırmamış birinin** tek başına
> koşturabilmesi için yazıldı. Adımları sırayla uygula; hiçbir adımı atlama.

---

## 0. Otuz saniyede özet

Önce **sanal ortamı aktif et**, sonra komutlar:

```
cd final-v1.0
.venv\Scripts\activate

python modelleri_yerlestir.py     (bir kez, kurulumdan sonra)
python dogrula.py                 (her açılışta — hepsi [OK] olmalı)

cd sistem
python main.py                    (yarışma başlayınca)
```

Satır başında `(.venv)` görüyorsan doğru ortamdasın.

Sistem başladıktan sonra **hiçbir şeye dokunma.** Ekranda ilerleme çubuğu akar,
oturum bitince kendi kapanır.

> ⚠️ Ortamı aktif etmeden `python dogrula.py` yazarsan "transformers KURULU DEGIL"
> gibi hatalar alırsın — bilgisayarda birden fazla Python var. Ayrıntı: Sorun 0.

---

## 1. Bu sistem ne yapıyor?

Yarışma sunucusu her karede bize bir görüntü ve bir "sağlık durumu" gönderir.
Biz o kareyi üç görevden geçirip **tek bir JSON paketiyle** cevabı geri yollarız:

| Görev | Ne üretir |
|---|---|
| 1 — Nesne Tespiti | taşıt / insan / iniş alanı kutuları + hareket ve iniş durumu |
| 2 — Konum Kestirimi | GPS kesikken aracın x, y, z konumu |
| 3 — Referans Eşleme | oturum başında verilen nesnenin karedeki yeri |

Oturum 2250 kare sürer, pencere 90 dakikadır. Sistem ölçülen hızıyla oturumu
**yaklaşık 37 dakikada** bitirir — bol zaman payı vardır.

---

## 2. Yarışmadan ÖNCE (aynı gün, en az 1 saat kala)

### 2.1 Python ortamı

Python **3.11 veya üstü** gerekir. Sanal ortamı **bu klasörün içine** kur:

```bash
cd final-v1.0
python -m venv .venv
.venv\Scripts\activate
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

> **Sıra önemli.** Önce CUDA'lı torch, sonra `requirements.txt`. Ters yaparsan
> torch'un CPU sürümü gelir ve sistem yavaşlar.

Ortam zaten kuruluysa kurulumu atla, ama **her yeni terminalde aktif etmeyi
unutma**:

```
.venv\Scripts\activate
```

Satır başında `(.venv)` yazmıyorsa aktif değildir ve komutlar yanlış Python'la
çalışır (bkz. Sorun 0).

### 2.2 Model ağırlıklarını yerine koy

```bash
python modelleri_yerlestir.py
```

Bu, pakete gömülü RAFT ve DINOv3 ağırlıklarını `~/.cache` altına kopyalar.
**Neden gerekli:** yarışma bilgisayarının internete çıkması garanti değildir;
ağırlıklar yerelde yoksa sistem onları indirmeye çalışır ve başarısız olur.

Görev 1'in kendi ağırlığı (`gorev1/agirliklar/best.pt`) zaten pakette, taşımaya
gerek yok.

### 2.3 Doğrulama

```bash
python dogrula.py
```

**Beklenen çıktı — dört bölüm, hepsi `[OK]`:**

```
1) PYTHON VE BAGIMLILIKLAR      → tüm paketler + "CUDA var: 12.8"
2) MODEL DOSYALARI              → Görev 1 ağırlığı 65 MB, RAFT var, DINOv3 var
3) SISTEM KURULUMU              → Görev 1 yüklendi, Görev 3 yüklendi
4) UCTAN UCA KARE ISLEME        → motor=raft
```

Son satırda **`*** SISTEM CALISMAYA HAZIR ***`** yazmalı.
Yazmıyorsa §5'e bak, yarışma başlamadan çöz.

### 2.4 Sunucu bilgilerini kontrol et

`sistem/config/.env` dosyası:

```
TEAM_NAME=bugbuster_5202745
PASSWORD=lJI6pABqrnQq
EVALUATION_SERVER_URL="http://havaciliktayapayzeka.teknofest.org:1025/"
SESSION_NAME=ONLINE_YARISMA_2026
```

- **Sunucu adresi yarışma günü değişirse** yalnız `EVALUATION_SERVER_URL` satırını
  güncelle. Sonundaki `/` kalsın.
- `SESSION_NAME` satırını **değiştirmene gerek yok** — `main.py` oturum adını
  sunucudan (`/progress`) alır, bu satırı okumaz. Sadece bilgi amaçlı durur.
- Şifre yanlışsa sisteme hiç giriş yapılamaz (bkz. Sorun 4).

---

## 3. Yarışma günü — çalıştırma

Oturum **açıldıktan sonra** başlat. Erken başlatırsan "No active session found"
deyip kapanır; sorun değil, oturum açılınca tekrar çalıştır.

```bash
cd sistem
python main.py
```

### Ekranda ne göreceksin

```
Started...
On kontrol: {'opencv': ..., 'cuda': True, 'gpu': 'NVIDIA ...', 'onerilen_motor': 'raft'}
Gorev 1 detektoru yuklendi.
Gorev 3 eslestirici yuklendi.
Session: ONLINE_YARISMA_2026 — resuming from frame 1 of 2250
Gorev 3: 7 referans kaydedildi.
Frames:   3%|██                    | 68/2250 [00:52<28:12, ...]
```

**Bunlar normaldir ve müdahale gerektirmez:**
- İlerleme çubuğunun ETA'sı dalgalanır.
- Bazı karelerde ilerleme bir saniye durur (sunucu aynı kareyi tekrar verir).
- Görev 3 yalnız referans aralıklarında çalışır; çoğu karede sessizdir.

### Koşarken neye bakılır

Sadece **iki şeye**:

1. **İlerleme çubuğu artıyor mu?** Artıyorsa her şey yolunda.
2. **Başlangıçtaki `'cuda': True` ve `motor=raft` var mıydı?** Varsa doğru
   yapılandırmadasın.

Detay isteyen olursa canlı log: `sistem/_logs/bugbuster_5202745_*.log`.
Her karede tek satır özet vardır:

```
[Gorev2 echo] konum=(109.3,-154.9,+0.0) m | [Gorev1] 3 nesne | [Gorev3] 0 eslesme
[Gorev2 TAHMIN] konum=(110.4,-155.5,+0.0) m | [Gorev1] 3 nesne | [Gorev3] 1 eslesme
```

`echo` = sunucu konumu verdi, aynen geri gönderdik (hata sıfır).
`TAHMIN` = GPS kesik, konumu görüntüden ürettik.

---

## 4. ⚠️ En önemli kural: çalışırken KAPATMA

Görev 2 bir **durum makinesidir** — konumu kare kare biriktirir ve GPS'in açık
olduğu dönemde ölçek kalibrasyonunu öğrenir. Bunların hepsi bellektedir.

**Programı kapatıp yeniden başlatırsan:**
- Sunucu tarafı kaldığı yerden devam eder (`resuming from frame N`) — bu iyi.
- Ama Görev 2 **sıfırdan başlar**: konum başlangıç noktasına döner, öğrenilmiş
  ölçek kaybolur. Bir sonraki sağlıklı blok gelene kadar Görev 2 çıktısı yanlış olur.

Yani:

> **Kesinti ortasında yeniden başlatmak, o kesintinin puanını yakar.**
> Sistem yavaş görünse bile bekle. 90 dakikalık pencerede 37 dakikalık iş var,
> acele etmeye gerek yok.

Yeniden başlatmak **gerçekten** zorunluysa (çökme, donma), sağlıklı bir bloğun
başlamasını beklemek en az zararlı seçenektir — logda arka arkaya `echo` satırları
görüyorsan o an sağlıklı bloktasın.

---

## 5. Sorun giderme

### Sorun 0 — ⭐ EN SIK: "transformers KURULU DEGIL" (ya da başka bir paket)

```
[UYARI] Bu bir SANAL ORTAM DEGIL — genel Python calisiyor.
[UYARI] ultralytics     8.4.67   ← test edilen: 8.4.92
[HATA]  transformers    KURULU DEGIL
```

**Sebep:** sanal ortam aktif değil. Bilgisayarda birden fazla Python var; ortam
aktif değilken `python` komutu genel olanı çalıştırıyor. Orada hem paketler eksik
hem de kurulu olanların sürümü farklı.

Gerçekten yaşanmış örnek:

| paket | ortam aktif değilken | doğru ortamda |
|---|---|---|
| transformers | **yok** | 5.13.0 |
| ultralytics | 8.4.67 | **8.4.92** |
| opencv | 4.13.0 | **4.11.0** |

**Çözüm:** ortamı aktif et, komutu tekrar çalıştır.

```
cd final-v1.0
.venv\Scripts\activate
python dogrula.py
```

Aktif olduğunu satır başındaki `(.venv)` yazısından anlarsın.

> **Eksik paketi genel Python'a kurup öyle çalıştırma.** transformers'ı kursan
> bile ultralytics ve opencv sürümleri farklı kalır; DINOv3 ağırlığını yükleyen
> mekanizma ve takip ayarları o sürümlerde **hiç denenmedi.** Yarışma günü test
> edilmemiş bir bileşim çalıştırmanın anlamı yok.

**Nasıl kontrol edersin:** `dogrula.py`'nin ikinci satırı hangi yorumlayıcının
çalıştığını yazar; içinde `.venv` geçmeli. Ayrıca sürümü test edilenden farklı
olan her paket için `[UYARI]` basar — hiç uyarı çıkmamalı.

---

### Sorun 1 — Sanal ortam yok / silinmiş
Ortamı bu klasörün içinde kur:

```
cd final-v1.0
python -m venv .venv
.venv\Scripts\activate
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

`requirements.txt` sürümleri sabitler; kurulum sonrası `python dogrula.py`
hiçbir `[UYARI]` vermemeli.

---

### Sorun 2 — `CUDA: YOK (CPU modu)`
GPU görünmüyor. Sistem **çalışır ama yavaşlar** ve Görev 2 otomatik olarak
SIFT yedeğine düşer (doğruluk bir miktar azalır, çökmez).

**Çözüm sırası:**
1. Başka bir program GPU'yu tutuyor olabilir (oyun, tarayıcı, başka Python) — kapat.
2. `nvidia-smi` komutu çalışıyor mu? Çalışmıyorsa sürücü sorunu var.
3. Torch'un CPU sürümü kurulmuş olabilir: `python -c "import torch; print(torch.__version__)"`
   → sonunda `+cu128` yoksa §2.1'deki torch komutunu tekrar çalıştır.

**Yarışma başlamışsa:** düzeltmeye kalkma, öyle devam et. CPU'da bile pencere yeter.

---

### Sorun 3 — `RAFT yok` veya `DINOv3 yok`
```
[UYARI] RAFT (Gorev 2)   yok — LK yedegine dusulur
[UYARI] DINOv3 (Gorev 1/3) yerel onbellekte yok
```
**Çözüm:** `python modelleri_yerlestir.py` çalıştır, sonra `python dogrula.py` tekrar.

Betik de bulamıyorsa ağırlıkları elle kopyala:
- `modeller/torch_hub_checkpoints/raft_large_*.pth`
  → `C:\Users\<KULLANICI>\.cache\torch\hub\checkpoints\`
- `modeller/huggingface_hub/models--facebook--dinov3-*` klasörünün **tamamı**
  → `C:\Users\<KULLANICI>\.cache\huggingface\hub\`

> DINOv3 yoksa **Görev 3 hiç çalışmaz** (Görev 1 ve 2 devam eder).
> RAFT yoksa Görev 2 SIFT'e düşer — kabul edilebilir, panik yok.

---

### Sorun 4 — Giriş yapılamıyor / "Could not reach the evaluation server"
Sistem sunucuya ulaşamıyor ya da kimlik doğrulayamıyor.

**Sırayla kontrol et:**
1. İnternet var mı? Tarayıcıdan sunucu adresini aç.
2. `.env` içindeki `EVALUATION_SERVER_URL` doğru mu? Port numarası (`:1025`) dâhil.
3. `TEAM_NAME` ve `PASSWORD` doğru mu? Boşluk veya tırnak kalmış olabilir.
4. Sunucu henüz açılmamış olabilir — birkaç dakika sonra tekrar dene.

---

### Sorun 5 — "No active session found. Exiting."
Sunucuda açık oturum yok. **Bu bir hata değil.** Oturum henüz başlamamıştır.
Yarışma görevlisi başlattığını söylediğinde `python main.py` komutunu tekrar çalıştır.

---

### Sorun 6 — "Aborting: current frame is not advancing"
Aynı kare beş kez üst üste geldi; sistem sonsuz döngüye girmemek için durdu.
Genellikle sunucu tarafında bir takılma vardır.

**Çözüm:** birkaç saniye bekle, `python main.py` ile tekrar başlat. Sunucu
kaldığı kareden devam eder. §4'teki uyarıyı hatırla — mümkünse sağlıklı blokta yap.

---

### Sorun 7 — Log'da çok sayıda "Rate limit hit"
Dakikada 80 tahmin sınırı aşılıyor. Normalde `main.py` içindeki 0,75 saniyelik
alt sınır bunu engeller.

**Çözüm:** `sistem/main.py` dosyasında `MIN_FRAME_INTERVAL = 0.75` satırını
`1.0` yap ve yeniden başlat. Bu, oturumu 2250 × 1 sn ≈ 38 dakikaya çıkarır —
pencere yine rahat yeter.

---

### Sorun 8 — Görev 3 hiç eşleşme üretmiyor
Başlangıçta `Gorev 3: 0 referans kaydedildi.` yazdıysa referans görselleri
indirilememiştir (sunucu, penceresi açılmamış referanslara 403 döndürebilir).

**Çözüm:** müdahale etme. Görev 1 ve 2 etkilenmez; referans pencereleri
açıldıkça sistem yine de çalışmayı dener. Oturumu yeniden başlatmak bu durumda
**Görev 2'ye zarar verir, Görev 3'ü kurtarmaz.**

---

### Sorun 9 — GPU belleği yetmiyor (`CUDA out of memory`)
Başka bir program VRAM tutuyordur.

**Çözüm:** tarayıcıyı, oyunları ve diğer Python süreçlerini kapat, yeniden başlat.
Yarışma sırasında olursa: kapat, bekle, tekrar başlat (§4'ü oku).

---

### Sorun 10 — Sistem beklenenden yavaş
Kare başı süreyi log'dan görebilirsin. Beklenen: Görev 1 ~100 ms, Görev 2 ~630 ms.

Çok daha yavaşsa muhtemelen CPU modundasın (bkz. Sorun 2) ya da GPU'yu paylaşan
başka bir iş var. **Oturum başladıysa dokunma** — 90 dakikalık pencerede CPU'da
bile bitirme şansı vardır ve yeniden başlatmanın maliyeti daha yüksektir.

---

## 6. Oturum bittiğinde

Ekranda `Session complete or no active session.` yazar ve program kapanır.

Saklaman gerekenler:
- `sistem/_logs/bugbuster_5202745_*.log` — tüm oturumun kaydı
- `sistem/_images/<oturum_adi>/` — indirilen kareler (büyük; istersen sil)

---

## 7. Hızlı komut kartı

| Ne zaman | Komut |
|---|---|
| Kurulumdan sonra bir kez | `python modelleri_yerlestir.py` |
| Her açılışta | `python dogrula.py` |
| Oturum açılınca | `cd sistem` → `python main.py` |
| Log'u canlı izlemek | `Get-Content sistem\_logs\*.log -Wait -Tail 20` |
| GPU durumu | `nvidia-smi` |

---

## 8. Kime ne sorulur

| Durum | Ne yap |
|---|---|
| Sunucu erişimi / oturum açılmıyor | Yarışma görevlisine sor |
| `dogrula.py` hata veriyor | Bu belgedeki §5'i uygula |
| Sistem koşarken tuhaf bir şey | **Kapatma**, log'un son 30 satırını kaydet, sonra sor |
