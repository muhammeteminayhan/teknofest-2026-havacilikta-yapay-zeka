# Görev 2 — ölçüm turu 2: gürültü tabanı, üç uçuşlu kapı ve inlier kapısı bulgusu

> ## ⚠️ BU BELGE KISMEN GEÇERSİZDİR — sonradan yapılan yöntem sınavına bakın
>
> Bu turun sonuçları 2026 verisinin üç yatağında ölçüldü. Sonraki tur bunları **2024 ve
> 2025** veri setlerinin 5 bağımsız yatağında sınadı ve bir kısmının **aşırı uyum**
> olduğunu buldu:
>
> | bu turda | yöntem sınavından sonra |
> |---|---|
> | `min_inlier_ratio = 0,05` | **0,30**'a çekildi (0,05 yıllar arası aşırı uyum) |
> | merdiven GNC'si `c=2,0` | **kaldırıldı** (2025-RGB-4K-B'de +8,4 m) |
> | kameraya göre normalize eşik | **kaldırıldı** (yıllar arası taşınmadı) |
> | — | `residual_gate` 0,50 → **0,30** eklendi (asıl kazanç) |
>
> Aşağıdaki metodoloji bulguları (gürültü tabanı, eşleştirilmiş karşılaştırma, üç uçuşlu
> kapı) **geçerlidir ve kullanılmaya devam ediyor**. Ayar sonuçları ise
> **[YONTEM_SINAVI.md](YONTEM_SINAVI.md)** ile değiştirilmiştir.

> **Bağlam:** finale gidiliyor, odak yalnız Görev 2. Final oturumunun uçuşu, teması ve
> kamerası (RGB 1080p / RGB 4K / **termal 640×512**) bilinmiyor; şartname kamera açısını
> 70–90° serbest bırakıyor. Hedef "örnek videoda en düşük sayı" değil, **bilinmeyen bir
> uçuşta gürbüzlük**.

## 0. Bu turun en önemli çıktısı: ölçümün gürültü tabanı

OpenCV RANSAC girdi verildiğinde deterministiktir **ama nokta sırasına duyarlıdır**: aynı
noktalar farklı sırada verilince eşit skorlu ama **farklı bir düzlem** seçilir. Ölçüldü
(video1, sevk edilen reçete, **aynı** nokta önbelleği, 10 permütasyon):

| | değer |
|---|---|
| ortalama | **3,964 m** |
| standart sapma | **0,226** |
| yayılım | 3,672 – 4,411 m (**%18,7**) |
| %95 aralığı | **±0,44 m** |

Aynı ölçüm sınav yatağında (puanlı oturum, 2 pencere): **±1,85 m**.

**Sonuçları:**
- Tek koşuyla ölçülen ~0,4 m'lik (sınavda ~2 m'lik) bir fark **gürültü olabilir**.
- Geçmiş turların bazı kararları bu tabanın altındadır ve kanıt değeri taşımaz —
  örneğin RAFT `ds0,75` varyantı **+0,034 m** ile reddedilmişti; bu saf gürültüdür.
- Bundan sonra her A/B **N permütasyonda, eşleştirilmiş (paired)** koşulur:
  aynı gürültü gerçeklemesinde iki yapılandırma karşılaştırılır, ortak gürültü düşer.

Ayrıca: `esik1_olcum.txt`'teki **3,693 m** `smooth_alpha=1.0` ile ölçülmüştü; o değer
sonradan 0,6'ya geri alındı. Sevk edilen reçetenin video1'deki gerçek değeri **3,88 m**
(tek permütasyon) / **3,92 ± 0,08 m** (6 permütasyon).

## 1. Altyapı

### Üç uçuşlu kabul kapısı — `kabul.py`
| yatak | veri | rol |
|---|---|---|
| `yarisma` | puanlı oturumun **kendi kareleri**, 2 sınav penceresi | **VETO** |
| `video1` | örnek veri 1 (RGB 1080p) — test oturumu uçuşu | kaybetmemeli |
| `video2` | örnek veri 2 (**termal**, 640×512) — bağımsız uçuş | kaybetmemeli |

`video2` bu turda **ilk kez** genelleme kanıtı olarak kullanıldı. Değerini hemen kanıtladı:
`duzlem_tabanli` adayı sınav + video1'i geçip **video2'de kaybetti** ve reddedildi.

### Sınavın hızlı yolu
Puanlı oturumun kareleri için nokta önbelleği üretildi
(`nokta_raft_yarisma_{cons,sp8}.npz`). Sınav **~56 dk/yapılandırma → ~30 sn**.
Doğrulama: önbellekli sonuç canlı sonuçla **birebir** aynı (16,61 / 16,80 m).

### Termal test yatağı
`video2` için nokta önbellekleri üretildi; **termal taban ilk kez ölçüldü**.

## SONUÇ TABLOSU (sevk edilen yeni reçete)

Üç değişiklik: `min_inlier_ratio` **0,40 → 0,05**, yeniden-izdüşüm eşiği **kameraya göre
normalize** (0,5 çalışma pikseli → RGB 1080p'de 1,0 px *değişmedi*, termalde 0,5, 4K'da 2,0),
ve **merdiven homografisinde GNC** (`merdiven_gnc_c=2,0`).

**Canlı, uçtan uca (önbelleksiz, gerçek `LivePositionEstimator`) — yarışmada gerçekleşecek koşu:**

| koşu | ESKİ (sevk edilen) | YENİ | kazanç |
|---|---|---|---|
| **genelleme sınavı** (puanlı oturumun kendi kareleri) | **16,71** | **8,61** | **−%48** |
|   · pencere 1 | 16,61 | **15,59** | ✓ |
|   · pencere 2 | 16,80 | **1,63** | ✓ |
| **video1-RGB** (`canli_dogrulama.py`) | 3,88 | **3,92** | ≈ aynı |
| **video2-TERMAL** (önbellekli, doğal sıra) | ~~32,66~~ → bkz. not | **10,22** | — |

> **Düzeltme:** buradaki 32,66 m yanlış bir taban. O ölçümdeki "ESKİ" tanımı normalize
> eşiği hâlâ açık taşıyordu (eşik o sırada `from_points` varsayılanıydı, config alanı
> değil), yani gerçek eski reçete değildi. Aynı takvimde gerçek eski reçetenin termal
> değeri **14,73 m**'dir. Nihai (yöntem sınavı sonrası) değer **14,30 m** — yani termal
> kazancı %69 değil, doğal çekilişte **%3**'tür.

**Permütasyon ortalamaları (8 permütasyon, eşleştirilmiş — beklenen değer):**

| yatak | önce | sonra |
|---|---|---|
| sınav | 16,98 ± 1,75 | **8,79 ± 0,74** |
| video1-RGB | 3,87 ± 0,08 | **3,73 ± 0,09** |
| video2-TERMAL | 15,24 ± 0,83 | **9,34 ± 0,51** |

Yayılım da 2–3 kat düştü — yarışma **tek çekiliş** olduğu için bu doğrudan risk azaltmasıdır.

**Hız:** 651 ms/kare, p95 671 ms, 2256 kare = **24,7 dk** (90 dk penceresi).
GNC'nin ilk hâli 1342 ms/kare ile bütçeyi zorluyordu; ağırlıklı DLT çözümü tam SVD yerine
**normal denklemlerden** (AᵀA'nın 9×9 özayrışması) alınınca hem GNC'li sistem hem taban
hızlandı — sonuç sayısal olarak birebir aynı (maks fark 2,7e-12).

Kod tarafında toplam üç ayar: `EstimatorConfig.min_inlier_ratio`, `calibration.ransac_esigi`,
`EstimatorConfig.merdiven_gnc_c`.

### ⚠️ Dürüst tablo: doğal çekiliş ≠ permütasyon ortalaması

RANSAC nokta sırasına duyarlı olduğu için her koşu bir dağılımdan **tek çekiliştir**.
Yarışmada gerçekleşecek olan "doğal sıra" çekilişidir, ama bilinmeyen bir uçuşta bunun
dağılımın hangi tarafına düşeceği önceden bilinemez — karar ölçütü **beklenen değer**
(permütasyon ortalaması), risk ölçütü ise yayılımdır.

| yatak | doğal çekiliş (0,40 → 0,05) | permütasyon ortalaması |
|---|---|---|
| **sınav** | 16,70 → **12,54** ✓ | 16,98 → **10,92** ✓ |
| **video2-termal** | 32,66 → **12,52** ✓ | 15,24 → **10,49** ✓ |
| video1-RGB | 3,88 → **4,38** ✗ | 3,87 → **3,60** ✓ |

İki uçuşta her iki ölçüt de kazanıyor. video1'de ortalama kazanıyor ama doğal çekiliş
kaybediyor — ve video1 üç yatak içinde **en az güvenilir olanı** (bütün eski ayar
çalışmasının aşırı uyduğu test-oturumu uçuşu). Değişiklik bu gerekçeyle uygulandı.

Ayrıca dikkat: eski yapılandırmada termalin doğal çekilişi **32,66 m** iken ortalaması
15,24 m — yani eski sistem termalde çok ağır kuyruklu. Yeni yapılandırma bu kuyruğu
kesiyor (12,52 vs 10,49).

## 2. ✅ UYGULANAN: `min_inlier_ratio` 0,40 → 0,05

**Nasıl bulundu.** Termalde RANSAC eşiği taranınca yön ve ölçek hatalarının **zıt eşik**
istediği görüldü (eşik 0,5: ölçek %2,34 ama yön 19,4°; eşik 1,5: yön 3,1° ama ölçek %16,9).
"İki kanalı ayrı eşikle kestir" denendi — **etkisiz** (yön 19,41 vs 19,39). Demek ki sıkı
eşikteki yön hatası ölçümden değil, **kalite kapısından** geliyordu. Doğrulandı:

| eşik | merdiven inlier medyanı | `<0,40` kapısına takılan |
|---|---|---|
| 0,50 | 0,524 (termal) / 0,332 (RGB) | **%29,9 / %60,8** |
| 1,00 | 0,728 / 0,537 | %7,7 / %25,1 |

Kapıya takılan her ölçüm atılıp **sabit-hız yedeğine** düşülüyordu. Depo bu ilkeyi zaten bir
kez bulmuştu ("düşük dokulu sahnede gürültülü ölçüm bile serbest sürüklenen CV'den iyi
çıktı" — sim↔hom kapısı kaldırılırken) ama `min_inlier_ratio`'ya uygulamamıştı.

**Sonuç (6 permütasyon, eşleştirilmiş):**

| yatak | temel (0,40) | aday (0,15) | fark |
|---|---|---|---|
| **sınav** | 16,677 ± 1,850 | **10,497 ± 0,689** | **−6,180 ±1,391** |
| video1-RGB | 3,917 ± 0,076 | **3,479 ± 0,196** | **−0,437 ±0,196** |
| video2-termal | 15,380 ± 0,834 | **12,607 ± 1,339** | **−2,773 ±0,521** |

İkincil ölçütler de kazandı: video1 uzun-kesinti −0,429; video2 uzun-kesinti −3,563,
yön −2,072°, ölçek −%5,231.

**Dürüst uyarı:** sınavın **1. penceresi kötüleşiyor** (16,17 → 17,97, +1,80 ±1,24) ama
2. penceresi çöküyor (17,19 → **3,02**, −14,16 ±3,25). Eski katı kural ("her pencerede
kazanacak") bu adayı reddederdi; kural tek koşulu/gürültüsüz bir dünya varsayıyordu.
Üç bağımsız uçuşun **hepsinde** birincil metrik kazandığı için aday uygulandı.

## 2b. ✅ İKİNCİ KAZANÇ: merdiven homografisinde GNC (`merdiven_gnc_c=2,0`)

**Nereden geldi.** Farklı sektörlerin robustluk araçları tarandı: robust optimizasyon
(GNC/TEASER), endüstriyel sinyal işleme (Hampel), kontrol mühendisliği (anti-windup,
sızıntılı integratör), jeodezi (DIA/veri gözetleme), robust istatistik (IRLS, MAD ölçeği),
torbalama (RANSAC topluluğu). **Yalnız biri işe yaradı** — ve ancak doğru katmana
uygulandığında.

**Neden yalnız merdivende.** Homografi inlier medyanı: ardışık çiftlerde **0,86**
(düzlem modeli oturuyor), merdiven çiftlerinde **0,54** (8 karelik tabanda bina/ağaç
paralaksı devrede). GNC'yi *her ikisine* uygulamak video1'i bozuyordu (3,48 → 5,10);
*yalnız merdivene* uygulamak kazandırdı. Zaten temiz olan katmanı "robustlaştırmak"
sadece yanlılık ekliyor.

**Yöntem.** RANSAC çözümünden başlanır; kesilmiş en küçük kareler vekili dışbükeye yakın
bir noktadan başlayıp (tüm noktalar ağırlıklı) **sert kesmeye tavlanır**. Her adım
ağırlıklı normalize DLT (Black–Rangarajan ikiliği). Böylece RANSAC'ın *ayrık* model
seçimi — nokta sırasına bağlı olan kısım — belirlenimli hâle gelir.

**Sonuç (8 permütasyon, eşleştirilmiş):**

| yatak | temel | GNC c=2,0 | fark |
|---|---|---|---|
| **sınav** | 10,920 ± 1,746 | **8,793 ± 0,740** | **−2,127 ±0,913** ✓ |
| video1-RGB | 3,599 ± 0,309 | 3,733 ± 0,092 | +0,134 ±0,237 (gürültü içinde) |
| video2-termal | 10,492 ± 1,680 | **9,340 ± 0,509** | −1,152 ±1,412 |

**Sınavın her İKİ penceresi de kazanıyor** (p1 17,94 → 15,45; p2 3,90 → 2,14) — eski katı
kuralı geçen ilk aday.

Asıl teşhis edilen hedefe (**ölçek yanlılığı**) doğrudan vuruyor:

| | önce | sonra |
|---|---|---|
| video1 ölçek hatası | %2,55 | **%1,55** |
| termal ölçek hatası | %6,33 | **%2,77** |

Ayrıca **yayılım her yatakta 2–3 kat düşüyor** (sınav ±1,75 → ±0,74). Yarışma tek çekiliş
olduğu için bu başlı başına risk azaltmasıdır.

`c` taraması: 1,5 / 2,0 / 2,5 üçü de kapıdan geçti; 2,0 en iyi dengeyi verdi
(2,5 termalde biraz daha iyi ama video1'e daha pahalı).

## 3. ❌ Denenip elenenler (hepsi ölçüldü)

| Aday | Sonuç | Gerekçe |
|---|---|---|
| **Sağlıklı→kesinti sapma kalibrasyonu** | ❌ **çürütüldü** | Sağlıklı dönemde ölçülen sistematik sapma sonraki kesintiye taşınmıyor: blok1 gölge −0,19° vs kesinti −2,22°; blok3 **+1,90° vs −2,01°** (ters işaret); ölçek 0,958 vs 1,010. Kapalı kapı. |
| **RANSAC topluluğu** (permütasyon + medyan) | ❌ | n=3: 3,917 → 3,974. Varyans düşmedi, hata düşmedi. |
| **IRLS / Cauchy M-kestirimi** | ❌ | Permütasyon varyansını **1e-3 → 1e-6** düşürüyor (kesin çözüm) ama hatayı büyütüyor: c=1,5 **+0,360**, c=2,0 **+1,817**. Yorum: merdivendeki %40 aykırı nokta gürültü değil **gerçek 3B yapı** (bina/ağaç); yumuşak ağırlıklandırma onları modele sokuyor. **Sert reddetme doğru araç.** |
| **Ayrı kanal** (öteleme sıkı, yön gevşek eşikten) | ❌ | Etkisiz (19,41 vs 19,39) — sorun ölçümde değil kapıdaydı (bkz. §2). |
| **`duzlem_tabanli`** (düzlem-içi ortonormal taban) | ❌ | Sınav −0,136 ve video1 −0,007 kazanıyor ama **video2'de +0,326 kaybediyor**. Üçüncü yatak olmasa uygulanacaktı. |
| **`merdiven_hizalama`** (merdivene ayrı 4 serbestlikli hizalama) | ❌ | video1 −0,231 kazanıyor, yön 1,57°→1,10°; ama **sınav pencere 2 çöküyor** (17,19 → 25,97). Parçalı sağlıklı dönemde 4 parametre belirlenemiyor. |
| **`merdiven_olcek`** (aynı fikrin 1 serbestlikli hâli) | ❌ | 0,15 üstünde: video1 **3,292** (en iyi) ama sınav 10,50 → 12,01 ve video2 12,61 → 15,89. |
| **Normalize eşik** (çalışma pikseli) | ⏸ nötr | 0,15 üstünde etkisiz (video2 12,61 → 13,06). Kural yine de doğru (§4) ve 4K için gerekli; ayrı değerlendirilecek. |
| **Loop closure / ortomozaik** | ❌ | Sınav pencerelerinde kapsama **%0**: en yakın sağlıklı konum medyanı 89,0 m (p1) ve 35,3 m (p2); kamera yer izdüşümü yarıçapı ~22 m. |
| **Termal CLAHE + FPN bastırma** | ❌ | Kazanan yapılandırma üzerinde 10,474 → **13,584** (+3,1 m); yön 1,26°→2,05°, ölçek %6,4→%11,4. Literatürdeki kazanç (arXiv 2603.02114) **16-bit ham** radyometrik veriden geliyor; elimizdeki termal video zaten 8-bit AGC'li, CLAHE orada sıkıştırma/kuantizasyon gürültüsünü büyütüyor. *Uyarı:* tek kare çiftinde inlier 0,872→0,888 ile umut verici görünmüştü — tek çiftlik sinyal aldatıcı. |
| **Uyarlanabilir ızgara** (396 → ~660 nokta) | ❌ | 10,474 → **11,798** (+1,3 m). Ek noktalar düşük dokulu bölgelerden geliyor; akış orada güvenilmez, yanlılık ekliyor. Nokta sayısı değil **nokta kalitesi** belirleyici. |
| **GNC — HER İKİ katmana** (ardışık + merdiven) | ❌ | c=3,0: sınav 10,65→8,62 ve termal 10,72→**6,96** ama video1 3,48→**5,10**. Ardışık katman zaten temiz (inlier 0,86); orada GNC yanlılık ekliyor. Yalnız merdivene uygulanınca kazanca döndü (§2b). |
| **GNC — MAD ile uyarlamalı `c`** | ❌ | `c = k·1,4826·MAD(artık)` ile modaliteler arası taşınabilirlik hedeflendi; k=1/2/3/4,5 hepsi kötü (+2,4 … +5,9). MAD, iyi oturmuş bir fitin artık ölçeğini ölçüyor — aradığımız "yapı ne kadar yüksekte" büyüklüğünü değil. |
| **Hampel filtresi** (medyan+MAD, zamansal) | ❌ | Endüstriyel ölçümden. **Çok kötü** (+4,8 … +18,2 m; w=5–15, k=2–3). Sebep: araç gerçekten manevra yapıyor — ölçüm dizisi durağan değil, medyanla değiştirme gerçek hareketi siliyor. Aynı kök neden depoda daha önce iki kez ölçülmüştü (sürüklenme sönümleme, EMA α). |
| **Anti-windup / sızıntılı integratör** (kontrol müh.) | ❌ | Yükseklik integratörünü çapa yüksekliğine bantla/çek: `h_bant` 0,1–0,5 ve `h_sizinti` 0,005–0,05 hepsi kötü (video1 3,51 → 5,61 … 12,04). Ölçülen yükseklik sapması windup değil, ölçümün kendi ölçek yanlılığının yansımasıymış. |
| **RAFT çalışma çözünürlüğü 0,75** (yeniden) | ❌ | Yeni gevşek kapıyla yeniden ölçüldü: 3,62 vs 3,51 — hâlâ kötü. *Ama* eski kapıyla 5,16 idi; kapı teşhisini doğruluyor. |
| **Keyframe aralığı** (spacing 4 / 16) | ❌ | Dokümanların "açık iş" maddesi üç yatakta kapatıldı: 4 → +2,82, 16 → +10,08. **spacing=8 korunuyor.** |

## 4. Ölçülen yan bulgular

**Eşiğin doğru normalizasyonu — iki modalitede bağımsız doğrulandı.**
Optimum yeniden-izdüşüm eşiği her iki kamerada da **0,5 çalışma pikseli**ne denk geliyor:

| kamera | çalışma ölçeği | optimum doğal eşik | çalışma pikselinde |
|---|---|---|---|
| RGB 1080p | 0,50 | 1,00 px | **0,50** |
| Termal 640 | 1,00 | 0,50 px | **0,50** |

RAFT akış gürültüsü çalışma çözünürlüğünde sabit; noktalar doğal piksele geri
ölçeklendiği için doğal eşik `1/ölçek` ile büyümeli. Sabit 1,0 px kullanmak termalde
eşiği 2× gevşetiyor, **4K'da 4× sıkardı** — final bilinmeyen kamerayla geleceği için
`calibration.ransac_esigi` olarak koda alındı.

**Termal ↔ RGB ölçüm farkları (aynı boru hattı):**

| | video1 RGB | video2 termal |
|---|---|---|
| n_kept (ardışık / merdiven) | 646 / 578 | **396 / 354** |
| homografi inlier (ardışık / merdiven) | 0,860 / 0,537 | 0,914 / 0,728 |
| düzlem eğimi (medyan) | 4,84° | 6,25° |
| ima edilen uçuş yüksekliği | 53,9 m | 32,8 m |
| **merdiven/ardışık ölçek oranı** | 0,9936 | **0,9844** |
| kesinti RMS x / y / z | 3,5 / 3,0 / 4,4 m | 10,7 / **19,6** / **20,4** m |

Termalde inlier oranı **daha iyi** (daha az 3B yapı) ama nokta sayısı %39 az ve
merdiven↔ardışık ölçek uyuşmazlığı 2,4 kat büyük. Hata ağırlıklı olarak y ve z'de;
uçuş baskın olarak +y yönünde 592 m ilerlediği için bu **ölçek hatasının** göstergesi.

## 4b. Kalan ayarlar: ölü kapılar ve doğrulanan değerler

Kalan bütün kalite kapıları 5 permütasyonda üç yatakta tarandı (yeni varsayılanlar üstünde):

| ayar | denenen | etki |
|---|---|---|
| `min_kept` | 10 / 60 | **hiç tetiklenmiyor** (n_kept medyanı 396–646) |
| `max_dz_unit` | 0,08 / 0,30 | **hiç tetiklenmiyor** |
| `max_step_xy` | 6,0 | **hiç tetiklenmiyor** |
| `max_yaw_step` | 20° | **hiç tetiklenmiyor** |
| `residual_gate` | 0,8 | **hiç tetiklenmiyor** |
| `smooth_alpha` | **1,0** | sınav −0,05 · video1 −0,14 · **video2 +1,15** → net kötü |
| `smooth_alpha` | 0,4 | net kötü (+0,87) |
| `calib_window` | 120 | net kötü (+3,73) |

Yani `smooth_alpha=0,6` ve `calib_window=60` artık **istatistiksel olarak doğrulandı**
(daha önce tek koşuya dayanıyordu). Diğer beş kapı pratikte ölü — ileride sadeleştirilebilir.

**İnlier-ağırlıklı yumuşak harmanlama** (`merdiven_harman_ref`) — sert kapı yerine
`w = clip(inlier/ref, 0, 1)` ile merdiven↔ardışık harmanı. 8 permütasyon, eşleştirilmiş:
sınav −0,191 ±0,240, video1 −0,000 ±0,013, video2 +0,010 ±0,156 → **ETKİSİZ**, sevk edilmedi.

## 5. Açık işler
- **Sınav 1. penceresinin teşhisi.** Denenen hiçbir `min_inlier_ratio` değeri p1'i
  iyileştirmiyor (0,25 → +1,04; 0,15 → +1,80; 0,05 → +1,50). Yumuşak harmanlama da
  çözmedi. O penceredeki 600 karelik kesintide zayıf ölçümlerin neden zarar verdiği
  ayrı bir inceleme konusu.
- **Varyansı düşürmek.** Yeni yapılandırma ortalamayı düşürdü ama yayılım hâlâ büyük
  (video2 ±1,7; sınav ±1,7–2,0). Yarışma tek çekiliş olduğu için yayılım doğrudan risktir.
  IRLS varyansı çözüyordu ama yanlılık ekliyordu — arada bir yol aranabilir
  (örn. sadece *ölçek* kanalını IRLS ile, yönü RANSAC ile kestirmek).
- **Termalin kalan ölçek hatası %6,3.** RGB'de %2,5. Kaynak hâlâ açık.
- Yapı-farkında homografi (uzaysal dengelenmiş inlier seçimi) — denenmedi.
- SEA-RAFT omurgası — denenmedi.
- 4K profili hiç test edilmedi (final 4K gelirse `ransac_esigi` 2,0 px verecek; kural
  doğru ama ölçülmedi).
