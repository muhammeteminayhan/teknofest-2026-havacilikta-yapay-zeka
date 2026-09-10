# Görev 2 — tanı turu (sevkiyat yok)

> Bu tur hiçbir üretim değişikliği yapmadı; `final-v1` teslim paketi donduruldu.
> Amaç, ölçek kanalı üzerine boru hattına dokunmadan tanı koşmaktı. **Turun en
> önemli çıktısı planlanan tanılardan biri değil, onların ortaya çıkardığı bir
> ölçüm-altyapısı hatasıdır.**

---

## 🔴 Bulunan sessiz hata: 2025 termal yataklarında GT ↔ video çapraz eşleşmesi

2025 veri setinde **termal CSV'ler çapraz adlandırılmış**: her termal video, kendi
adındaki CSV yerine diğeriyle eşleşiyor. Ölçümle kanıtlandı — birim yatay ölçüm
`|E|` ile aynı çiftin GT yatay adımı `|G|` korelasyonu:

| video | Veri‑1‑RGB.csv | Veri‑1‑Termal.csv | Veri‑2‑RGB.csv | Veri‑2‑Termal.csv |
|---|---|---|---|---|
| `Ornek-Veri-1-Termal.MP4` | **0,717** | 0,031 | 0,034 | **0,719** |
| `Ornek-Veri-2-Termal.MP4` | −0,008 | **0,523** | **0,520** | −0,013 |
| *(kıyas)* `Ornek-Veri-1-RGB.MP4` | **0,987** | 0,048 | 0,052 | **0,982** |

Doğru eşleşme: **A uçuşu = RGB‑1 + Termal‑1**, **B uçuşu = RGB‑2 + Termal‑2**;
yalnız *termal CSV adları* takas edilmiş.

### Neden fark edilmedi

Yatakları kurarken "tüm kareler sağlıklı" koşusunda hatanın **0,0000 m** çıkmasını
doğrulama saymıştım. **O kontrol bu hatayı yakalayamaz**: sağlıklı karede GT aynen
geri gönderildiği için, GT hangi uçuşa ait olursa olsun hata sıfırdır. Yakalayan tek
şey **görsel ölçümün GT ile korelasyonudur** — ve bu hiç ölçülmemişti.

### Ne geçersiz oldu

İki "bağımsız yatak" tamamen anlamsızdı. Onlara dayanan bulgular:

| bulgu | eski (bozuk) | düzeltilmiş | durum |
|---|---|---|---|
| "sistem *hiçbir şey yapmamaktan kötü*" (v25_termal_b: 76,5 vs tut 53,1) | ✗ | 11,04 vs tut 53,1 → **4,8× iyi** | **artifakt — geri çekildi** |
| v25_termal_a 27,2 m | ✗ | **3,86 m** (11,3× iyi) | artifakt |
| 5 yatak ortalaması 35,42 → 22,27 m | ✗ | **8,39 → 7,54 m** | rakam düzeltildi |
| rejim dedektörü tanısı (P4) | tamamen o iki yatağa dayanıyordu | — | **konu düşüyor** |

### Ne ayakta kaldı — ve güçlendi

`residual_gate` 0,50 → 0,30 değişikliğinin gerekçesi bozuk yataklardan geliyordu;
düzeltilmiş veride **yeniden ölçüldü ve doğrulandı**:

| yatak | residual_gate 0,50 → 0,30 |
|---|---|
| v25_termal_a | **−5,47 m** |
| v24 · v25_rgb_a · v25_rgb_b · v25_termal_b | +0,00 / +0,00 / +0,00 / +0,02 |

Yani kazanç gerçek ve mekanizma doğru: **bozuk hizalamayı reddet, çalışanı hiç
etkileme.** `min_inlier_ratio` 0,30 kararı da etkilenmedi (gerekçesi v24 ve
v25_rgb_a'ydı, ikisi de geçerli yataklar).

### Kalıcı koruma

`nokta_onbellek.yatak_dogrula()` / `yataklari_dogrula()` eklendi ve
`yontem_sinavi.py`'ye **zorunlu kapı** olarak bağlandı: eşleşmesi şüpheli yatak
otomatik atlanır. Şu anki durum — 8 yatak × 2 motor, hepsi temiz (kor 0,52–0,99).

### Düzeltilmiş yıllar-arası sınav

| yatak | ESKİ | +residual_gate | **YENİ** | fark |
|---|---|---|---|---|
| v25_rgb_a (4K) | 4,05 | 4,05 | **3,37** | **−0,68** ✓ |
| v25_rgb_b (4K) | 12,86 | 12,86 | 13,37 | +0,50 (eşit) |
| v25_termal_a | 9,09 | 4,41 | **3,86** | **−5,23** ✓ |
| v25_termal_b | 11,03 | 11,06 | 11,04 | +0,01 (eşit) |
| v24 (2B) | 4,90 | 4,90 | 6,08 | +1,17 ✗ |
| **ORTALAMA** | **8,39** | 7,46 | **7,54** | **−0,85** |

---

## D3 — Keyframe kullanılabilirlik bozunması ❌ (uzun çapa kapanıyor)

sp16 önbelleklerinden hop 1…16 eğrisi (daha önce yalnız iki nokta biliniyordu):

| yatak | hop 1 | hop 8 | hop 16 | hop16'da kapı altı | ima_h sapması (hop 1→16) |
|---|---|---|---|---|---|
| video1 (RGB) | 0,878 | 0,369 | 0,251 | **%71,4** | −1,3 … +1,5 % |
| yarışma (RGB) | 0,927 | 0,433 | 0,288 | **%58,7** | 0 … +2,3 % |
| video2 (termal) | 0,919 | 0,601 | 0,544 | **%9,1** | −4,7 … +0,04 % |

**Sonuç:** RGB'de keyframe hızla ölüyor — hop 16'da ölçümlerin %59–71'i
`min_inlier_ratio=0,30` kapısının altında. Uzun ömürlü çapa RGB'de **olanaksız**.
Termalde yaşıyor (%9,1) ama modaliteye özel yapısal değişiklik tek uçuşluk kanıtla
yapılmaz.

**Yan bulgu (değerli):** `ima_h` sapması hop ile **büyümüyor** (±%2,3 içinde). Yani
uzun taban çizgisi ek *yanlılık* getirmiyor; sınır tamamen ölçümün hayatta kalmasıdır.
Bu, `spacing=8`'in neden optimum olduğunu da açıklıyor: hop 4–8 arası hem yeterli
sinyal hem kabul edilebilir kapı-altı oranı veriyor.

*Not:* kayıtlardaki "merdiven inlier medyanı 0,54" değeri sp8 önbelleğinin **tüm
hop'ları (1–8)** üzerindeki medyandır, hop=8'in kendisi değil (hop 8 → 0,369).

## D4 — Düşey kanal ❌ (kalıcı kapandı)

`z_sim_agirlik` eski kapıyla (0,40/0,50), tek uçuş ailesinde reddedilmişti. Yeniden
**uygulamadan**, iki aday düşey sinyali sağlıklı karelerde GT `dz`'ye karşı puanladım
(açıklanan varyans R²):

| yatak | katman | hom `n·delta` | sim `1−1/ölçek` | kazanan |
|---|---|---|---|---|
| video1 | ardışık / merdiven | 0,833 / 0,856 | **0,908 / 0,916** | sim |
| video2 | ardışık / merdiven | 0,573 / 0,695 | **0,644 / 0,705** | sim |
| yarışma | ardışık / merdiven | 0,709 / 0,878 | **0,949 / 0,946** | sim |
| **v25_rgb_b** | ardışık / merdiven | **0,860 / 0,915** | 0,088 / 0,070 | **hom** |
| **v25_termal_b** | ardışık / merdiven | **0,547 / 0,849** | 0,142 / 0,077 | **hom** |
| v25_rgb_a · v25_termal_a | — | ~0 | ~0 | *(uçuşta düşey hareket yok)* |

Önceden yazılı kriter: benzerlik ≥5 yatakta her iki katmanda kazanmalı →
**3/7, kriter sağlanmadı.**

**Mekanizma net:** benzerlik ölçeği *sahne derinliği* değişimini okuyor, *yükseklik*
değişimini değil; ikisi ancak sahne düz ve bakış eksenine dik olduğunda çakışıyor.
2026 uçuşlarında çakışıyor, 2025'te çakışmıyor. Bu, `DUSEY_KANAL.md`'nin özgün
teşhisinin yıllar arası doğrulanmasıdır. **Kapı kalıcı kapandı.**

---

## D1/D2 — anizotropik hizalama ❌ (aile kapandı)

Depo, kalan hatanın anizotropik olduğunu ve Procrustes'in (dik `M` + skaler `c`)
bunu soğuramayacağını kendi yazıyordu. Genel/dik-olmayan hizalama hiç denenmemişti.
Uygulamadan önce **örneklem-dışı** sınandı: pencerede (60 örnek) fit, **sonraki 12
örnekte** puanla. Üç aday — `P` Procrustes (mevcut), `S` det-normalize simetrik
düzeltme (+2 SD), `A` genel 2×2 (4 SD).

### D1 — örneklem-dışı RMS konum hatası [m]

| yatak | pencere | **ESS** | P (mevcut) | S (+2 SD) | A (4 SD) | kazanan |
|---|---|---|---|---|---|---|
| video1 | 9 | 1,8 | 0,0742 | 0,0645 | **0,0641** | A |
| **video2** | 9 | 4,1 | **0,2128** | 0,2164 | 0,5353 | **P** |
| **yarışma** | 4 | 1,5 | **0,0931** | 0,0934 | 0,1037 | **P** |
| v24 | 7 | 2,8 | 0,2566 | **0,2418** | 0,2833 | S |
| v25_rgb_a | 9 | 2,7 | 0,0289 | **0,0289** | 0,0408 | S |
| v25_rgb_b | 9 | 1,7 | 0,1381 | 0,1137 | **0,1076** | A |
| v25_termal_a | 9 | 12,7 | 0,0909 | **0,0900** | 0,4353 | S |
| v25_termal_b | 9 | 3,8 | 0,1645 | 0,1384 | **0,1218** | A |

`S` Procrustes'i **6/8** yatakta geçiyor → önceden yazılı kriter (≥6) **kılpayı
sağlanıyor**. `A` yalnız 3/8 — daha çok serbestliğin aşırı uyduğunu doğruluyor.

**Ama üç uyarı:**
1. Kazanç payları çok küçük (video1 0,0742→0,0645), ve **kaybettiği iki yatak
   `video2` ile `yarışma`** — yani yarışmaya en yakın iki yatak.
2. **ESS 1,5–4,1.** 60 örneklik pencere yalnız ~2–4 *etkin* örnek taşıyor
   (lag-1 otokorelasyon ≈ 0,94; `smooth_alpha=0,6` EMA'sı + fiziksel süreklilik).
   İki ek parametreyi ~2 etkin örnekle kestirmek, `merdiven_hizalama`'yı öldüren
   sayısal rejimin ta kendisi. 6/8 bu koşulda yazı-tura'dan ayırt edilemez.
3. Bu, D2 vetosunun neden bağlayıcı olduğunu da açıklıyor.

### D2 — mekanizma: öğrenilen anizotropi eğimden mi geliyor? **HAYIR**

| yatak | ε̂ medyan | eğim tahmini (1−cos t)/2 | kor(ε̂, eğim) | kor(θ̂, azimut) |
|---|---|---|---|---|
| video1 | 0,0061 | 0,0023 | **−0,318** | −0,362 |
| video2 | 0,0223 | 0,0012 | **−0,430** | −0,500 |
| yarışma | 0,0411 | 0,0036 | — | — |
| v24 | 0,0031 | 0,0003 | **−0,392** | −0,512 |
| v25_rgb_a | 0,0041 | 0,0002 | −0,104 | 0,035 |
| v25_rgb_b | 0,0208 | 0,0308 | **−0,216** | 0,546 |
| v25_termal_a | 0,0221 | 0,0002 | +0,203 | −0,263 |
| v25_termal_b | 0,0248 | 0,0283 | **−0,270** | 0,360 |

Beklenti pozitif korelasyondu (>0,3). Ölçülen: **7 yatağın 6'sında NEGATİF**
(−0,10 … −0,43), yalnız birinde zayıf pozitif.

Dahası **büyüklük tutmuyor**: ε̂ çoğu yatakta eğimin öngördüğünden kat kat büyük
(video2: 0,0223 vs 0,0012 → 18×; v25_termal_a: 0,0221 vs 0,0002 → 100×).

**Sonuç:** `S` eğimi değil, **başka bir sistematiği** (büyük olasılıkla yükseklik
integratörü hatası ve merdiven↔ardışık ölçek uyuşmazlığı — ikisi de 2–70 kat büyük)
yutuyor. Önceden yazılı kriter gereği bu, D1 geçse bile **bağlayıcı vetodur**.

**AİLE KAPANDI.** Anizotropik/genel hizalama bir daha açılmayacak. Tasarım
incelemesinden çıkan iki tuzak da kayda geçti (ileride biri denerse diye):
genel `A` ölçeği yutar (= `height_dz_katsayi=0`, 4,34→13,6 m) ve düzeltme
**2ψ hızıyla döner** (kamera çerçevesinde fit edilmeliydi).

### Yan bulgu: ESS ~2, bu sistemde 4 serbestlikli hiçbir hizalama öğrenilemez

Kalibrasyon penceresi 60 örnek **taşıyor** ama yalnız ~2–4 *etkin* örnek
**içeriyor**. Bu, `merdiven_hizalama` (4 SD, sınav penceresi 2'de 17,19→25,97) ve
`merdiven_olcek` (1 SD, sınavda +1,51) başarısızlıklarının ortak kök nedenidir ve
gelecekteki her "hizalamaya serbestlik ekleyelim" fikri için ön kabul olmalıdır.

## Eski D1/D2 notu (tasarım incelemesi)

Tasarım incelemesi bu ailenin beklenen kazancını ölçülebilir yataklarda **~0**
olarak konumlandırmıştı (eğim anizotropisi medyanda %0,36, p90 %0,80; karıştırıcılar
2–70 kat büyük). Ayrıca iki ciddi tuzak belirlendi ve kayda geçti:

1. **Genel `A` ölçeği yutar** → `dxy = A·e0` yatay ölçeği fit penceresine dondurur;
   bu tam olarak `height_dz_katsayi=0` deneyidir (**4,34 → 13,6 m**). Herhangi bir
   zengin harita **determinant-normalize** edilmeli.
2. **Düzeltme yöne değişmez değil.** cos(eğim) büzülmesi *kamera* çerçevesinde
   oluşur; ψ=0 çerçevesinde sabit bir düzeltme **2ψ hızıyla döner** ve 90° yön
   değişiminde hatayı ikiye katlar. Fit kamera çerçevesinde yapılmalı.

Bu tur eşleşme hatasının düzeltilmesi ve etkilerinin yeniden ölçülmesiyle doldu.
D1/D2 açık iş olarak kalıyor; yapılırsa **2 serbestlikli, kamera çerçevesinde,
det-normalize, fiziksel sınırlı (|ε| ≤ 0,035), yön-çeşitliliği kapılı** biçimde ve
**sentetik eğim enjeksiyonuyla** doğrulanmalıdır — ortalama iyileşme beklenmemeli;
görülürse aşırı uyum kanıtı sayılmalıdır.

---

## Turun dersi

Üç turdur "tek veri ailesinde güçlü, bağımsız veride çöken" kazançlar elendi. Bu tur
bunun **bir katman altını** gösterdi: bağımsız verinin kendisi de sessizce bozuk
olabilir, ve bunu yakalayan kontrol "hata sıfır mı" değil, **"ölçüm GT ile korele mi"**
kontrolüdür. Artık zorunlu kapı.

## Yeniden koşturmak için

```
.venv\Scripts\python -c "import sys;sys.path.insert(0,'gorev2');from nokta_onbellek import yataklari_dogrula;yataklari_dogrula()"
.venv\Scripts\python gorev2\tani_hop.py
.venv\Scripts\python gorev2\tani_dusey.py
.venv\Scripts\python gorev2\yontem_sinavi.py --n-perm 5
```
