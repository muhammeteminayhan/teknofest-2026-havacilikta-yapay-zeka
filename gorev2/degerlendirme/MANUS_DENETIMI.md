# Görev 2 — Manus raporu denetimi ve tavan ölçümleri (14 Ağu 2026)

> `Gorev2_5_Cozum_Raporu.pdf` (Manus AI) beş çözüm ailesi öneriyor. Raporun kendi
> uyarısı: *"İncelenen makalelerin hiçbiri TEKNOFEST 2026 Görev 2 veri setinde doğrudan
> kıyaslanmış değildir"* — sıralaması ölçüm değil, uzman değerlendirmesi.
>
> **Raporun önerdiği nihai mimarinin büyük kısmı zaten sevk edilmiş durumda.** Rapor
> "Çözüm 1 + Çözüm 3" birleşimini öneriyor; sistemimiz tam olarak budur.

Bu tur, raporun **denenmemiş** kalemlerini ölçtü. Kalıcı kurala uyularak her kalem
kod yazılmadan önce bir **tavan ölçümüyle** başladı.

---

## 1. Denetim tablosu — rapordaki her yöntemin depodaki karşılığı

### Zaten sevk edilmiş

| rapordaki öneri | bizdeki karşılığı |
|---|---|
| Undistortion + `K` matrisi | `src/calibration.py` |
| Nokta seçimi ızgaraya dengeli | `raft_matching.py` `grid_step=28` meshgrid |
| Piramidal LK takip | `matching.py` (son çare yedeği) |
| `H = K(R − t·nᵀ/d)K⁻¹` | `geometry.py` |
| İlk sağlıklı karelerde Sim(3)/Procrustes | `scale_alignment.py` |
| Hız yumuşatma | `smooth_alpha=0,6` |
| Sabit-hız geri dönüşü | `_vel`, `cv_alpha` |
| Uzun aralıklı keyframe eşlemesi | merdiven, `spacing=8` |
| Yoğun optik akış → homografi | RAFT (Çözüm 3 = sistemimiz) |
| Donmuş kare tespiti | `PairMeasurement.frozen` |
| Offline replay altyapısı | `nokta_onbellek.py` + `kabul.py` |
| Stres testleri (gece/sis/blur) | `outputs/olcumler_tema_*` |
| Sentetik uçuş verisi | `outputs/olcumler_midair_*` |

### Zaten denenip ölçümle elenmiş

| rapordaki öneri | ölçüm |
|---|---|
| **MAGSAC++** | 9 kombinasyon; eşik 1,0'da 8,62 m / yön 6,32° — felaket |
| **ORB/LightGlue yeniden eşleme** | SuperPoint+LightGlue inlier %21 (RAFT %83); DISK+LightGlue 0/15 senaryo; XFeat 160 m |
| **Huber/Cauchy kaybı** | IRLS/Cauchy: varyansı 1e-3→1e-6 düşürüyor ama yanlılık ekliyor |
| **Ani sıçrama reddi** | Hampel: +4,8 … +18,2 m |
| **DPVO / DPV-SLAM / DROID** | derlendi ama `lietorch` torch 2.11 ile ABI çöküyor (CPU'da da) |
| **Metrik derinlik (Metric3D v2, DA V2)** | %28‒32 AbsRel — ölçek hatamızın 5‒10 katı |
| **VGGT** | tek parçada %2 (50 karede 0,97 m), zincirleme çöküyor (2250 karede 127 m) |

---

## 2. İş 1 — Tanı metrikleri ✅ **KURULDU**

Betik: [`gorev2/tani_metrik.py`](../tani_metrik.py). Üretim koduna dokunmaz; `kabul.py`
değiştirilmediği için kapı betikleri etkilenmez.

3 permütasyon, yalnız kesinti kareleri:

| | pencere1 | pencere2 | video1 | video2 | v25_rgb_a | v25_rgb_b | v25_termal_a | v25_termal_b | v24 |
|---|---|---|---|---|---|---|---|---|---|
| ortalama [m] | 11,46 | 5,60 | 5,68 | 15,05 | 4,99 | 15,49 | 6,08 | 11,92 | 10,40 |
| p95 [m] | 12,61 | 8,48 | 12,33 | **63,21** | 18,13 | 31,34 | 16,58 | 32,02 | 15,74 |
| en kötü tek kare | 12,66 | 10,16 | 14,78 | **85,27** | 26,52 | 45,09 | 26,31 | 48,69 | 16,99 |
| **kuyruğun hata payı** | **%5,8** | %11,1 | %12,0 | **%25,0** | %19,6 | %12,3 | %14,5 | %16,8 | %7,9 |

**Bulgu 1 — sınav penceresi 1'de kuyruk YOK.** Kuyruk payı %5,8 ≈ tekdüze dağılımın
%5'i; medyan (11,95) ≈ ortalama (11,46) ≈ en kötü kare (12,66). Yani oradaki hata bir
sıçrama değil, **düz birikmiş drift**. Sıçrama bastırma o pencerede bir şey kazandırmaz.

**Bulgu 2 — termalde kuyruk gerçek.** video2'de toplam hatanın **dörtte biri** en kötü
%5 kareden geliyor ve en kötü kare **85 m**. v25_termal_b (%16,8) ve v25_rgb_a (%19,6)
da kuyruklu. Yarışma tek çekiliş olduğu için bu doğrudan risktir.

**Bulgu 3 — sağlık geçişinde sıçrama yok.** Rapor (§11 Aşama 2) geçiş anındaki
hizalama sıçramasını ayrı ölçmeyi öneriyor. Ölçüldü: geçişten sonraki ilk 50 karede
hata **0,28‒2,41 m** (tüm yataklar), 100 karede 0,52‒4,16. Hata düzgün birikiyor,
devir teslimde sıçrama yok. **Bu bir doğrulama; iş çıkmıyor.**

**Eksen ayrışımı** tek bir baskın eksen göstermiyor: pencere1'de x+y (z yalnız 1,29),
video2'de y (MAE 11,72 / RMSE 20,65), v25_rgb_b'de x. *Not: v24 yatağının GT'sinde z
ekseni yok (sıfır sütunu), oradaki z rakamı kıyaslanabilir değildir.*

---

## 3. İş 4 — Uzaysal dengelenmiş inlier seçimi ❌ **KAPANDI**

Betik: [`gorev2/tavan_inlier_dagilimi.py`](../tavan_inlier_dagilimi.py)

Önce planın kendi öncülü çürütüldü: rapor "noktaları ızgaraya dengeli dağıt" diyor ama
**girdi tarafında bu zaten var** (`grid_step=28`). Geriye kalan gerçek soru RANSAC'ın
*seçtiği* inlier kümesinin kümelenip kümelenmediğiydi.

Ölçüm (yatak başına 150 çift, üretimdekiyle aynı `cv2.findHomography` çağrısı):

| yatak | katman | inlier | kapsama | merkez kayması | yayılım |
|---|---|---|---|---|---|
| yarışma | ardışık / merdiven | 0,928 / 0,617 | 1,000 / 1,000 | 0,014 / 0,046 | 0,953 / 0,796 |
| video1 | ardışık / merdiven | 0,862 / 0,518 | 1,000 / 0,938 | 0,029 / 0,048 | 0,917 / 0,817 |
| video2 | ardışık / merdiven | 0,903 / 0,688 | 1,000 / 0,938 | 0,018 / 0,041 | 0,943 / 0,847 |
| v25_rgb_b | ardışık / merdiven | 0,730 / **0,406** | 1,000 / 0,938 | 0,040 / 0,053 | 0,850 / 0,840 |
| v25_termal_b | ardışık / merdiven | 0,958 / 0,861 | 1,000 / 1,000 | 0,008 / 0,022 | 0,978 / 0,914 |

*(kapsama = 4×4 ızgarada inlier içeren hücre oranı · merkez kayması = inlier ağırlık
merkezinin sapması, köşegene normalize · yayılım = inlier bulutunun özdeğer izinin
tüm noktalarınkine oranı)*

**Önceden yazılı ölçüt:** kapsama ~1,00 ve merkez kayması ~0,00 ise iş kapanır.
**Ölçülen:** kapsama 0,938‒1,000, merkez kayması 0,008‒0,053. Kriter sağlandı.

En kötü durumda bile (v25_rgb_b merdiven, noktaların **%59'u atılıyor**) hayatta kalan
inlier'lar 16 hücrenin 15'ine yayılmış durumda. Mekanizma açık: nadir hava görüntüsünde
3B yapı (bina, ağaç) sahnenin **tamamına serpiştirilmiş**, tek bir bölgede toplu değil.
Uzaysal dengeleme düzeltecek bir şey bulamaz. **Aile kapandı.**

---

## 4. İş 3 — Dinamik nokta reddi ❌ **KAPANDI**

Betik: [`gorev2/tavan_dinamik_nokta.py`](../tavan_dinamik_nokta.py)

Rapor (§6.3, kaynak [13]) baskın homografiye uymayan akış kümelerinin dinamik nesne
sayılıp ağırlıklandırılmasını öneriyor. Bizim avantajımız: **Görev 1 dedektörü
hareketli/duran ayrımını zaten üretiyor** (`motion_status`), yani tahmin etmek yerine
ölçebiliyoruz.

Yarışma yatağı, 60 kare, hareketli araç kutuları %15 payla genişletildi:

| katman | aykırı oranı | hareketli kutuların kapladığı alan | **aykırıların kutuda olma oranı** | inlier'ların kutuda olma oranı |
|---|---|---|---|---|
| ardışık | %13,1 | %1,00 | **%1,03** | %1,09 |
| merdiven | %39,2 | %0,99 | **%1,36** | %0,63 |

**Önceden yazılı ölçüt:** aykırıların kutuda olma oranı, kutu alan oranından belirgin
yüksek ve mutlak olarak anlamlı (>%10) olmalı.
**Ölçülen:** %1,03 ve %1,36 — yani **kutu alanının kendisiyle (%1,0) aynı.** Aykırılar
hareketli araç üzerinde olmakta rastgele bir noktadan daha olasılıklı değil.

Merdiven katmanında noktaların **%39'u** aykırı, ama bunların yalnız **%1,4'ü** hareketli
araç üzerinde. Ayrıca kare başına hareketli tespit medyanı **0** ve karelerin yalnız
%30'unda hiç hareketli tespit var.

**Sonuç:** aykırılar hareketli nesnelerden değil **3B yapıdan** (bina/ağaç paralaksı)
geliyor. Bu, İş 4'ün bulgusuyla da tutarlı ve deponun merdiven inlier oranı (0,37‒0,62)
teşhisini açıklıyor. Dinamik nesne modeli bize bir şey kazandırmaz. **Aile kapandı.**

---

## 5. İş 2 — Kare kalite kapısı ❌ **KAPANDI**

Betikler: [`tavan_kare_kalitesi.py`](../tavan_kare_kalitesi.py) ·
[`tavan_doygun_piksel.py`](../tavan_doygun_piksel.py)

### 5a. Kare düzeyinde kalite: bozuk kare **yok**

8 yatak, yatak başına 450 örnek (her 5. kare):

| yatak | bulanık | düşük kontrast | kırpılmış >%1 | donmuş |
|---|---|---|---|---|
| yarışma | %0,0 | %0,0 | %0,0 | %0,0 |
| video1 | %0,0 | %0,0 | %0,0 | %0,0 |
| video2 | %0,0 | %0,0 | %0,0 | %0,0 |
| v25_rgb_a | %0,0 | %0,0 | %1,3 | %0,0 |
| v25_rgb_b | %0,4 | %0,0 | %10,9 | %0,0 |
| v25_termal_a | %0,7 | %0,0 | **%78,4** | %0,0 |
| v25_termal_b | %0,0 | %0,0 | **%66,0** | %0,0 |
| v24 | %6,6 | %0,0 | %3,0 | %0,0 |

**Bulanıklık, düşük kontrast ve donma pratikte yok.** Donmuş kareyi zaten
yakalıyoruz (`PairMeasurement.frozen`). Raporun önerdiği kare düzeyinde kalite kapısı
bu veride **ölü**dür.

Tek dolu sütun kırpılmış piksel, ama o da bir arıza değil: 8-bit AGC'li termal
görüntüde sıcak noktalar doyar. Termal karelerin üçte ikisini atmak saçma olurdu —
kare düzeyinde kapı burada **yanlış araçtır**.

### 5b. Piksel düzeyinde doygunluk maskesi: sinyal **yok**

Doğru soru şuydu: doygun bölgeye düşen örnekleme noktaları RANSAC tarafından
diğerlerinden daha sık aykırı sayılıyor mu? (120 kare/yatak, doygun bölge 3 px
genişletilmiş)

| yatak | katman | doygun alan | aykırı: doygun | aykırı: temiz | **oran** |
|---|---|---|---|---|---|
| video2 | ardışık / merdiven | %0,63 | %8,7 / %36,9 | %12,5 / %29,1 | **0,69 / 1,27** |
| v25_termal_a | ardışık / merdiven | %11,98 | %3,7 / %9,6 | %5,0 / %8,8 | **0,75 / 1,10** |
| v25_termal_b | ardışık / merdiven | %8,34 | %7,7 / %15,4 | %8,8 / %16,8 | **0,87 / 0,92** |

**Önceden yazılı ölçüt:** oran > 1,5 ise maske açılır.
**Ölçülen:** 0,69‒1,27 ve **altısının üçünde oran 1'in ALTINDA** — doygun bölgedeki
noktalar aykırı olmaya *daha az* eğilimli. Sistematik sinyal yok.

Mekanizma makul: termalde doyan yerler yol/çatı gibi büyük düz yüzeyler; kenarları
hâlâ dokulu ve RAFT oralarda sağlıklı akış üretiyor. **Aile kapandı.**

---

## 6. İş 5 — Kayan pencere backend'i ❌ **KAPANDI**

Betik: [`tavan_kayan_pencere.py`](../tavan_kayan_pencere.py)

Rapor (§4.2) backend için "10‒30 karelik kayan pencere, Huber/Cauchy kaybı, hız ve ivme
yumuşatması, ani sıçrama reddi" öneriyor. Listenin üçü depoda **zaten denenmiş**:
Huber/Cauchy → IRLS (yanlılık ekliyor), sıçrama reddi → Hampel (+4,8…+18,2 m), hız
yumuşatma → `smooth_alpha=0,6` (üretimde). Denenmemiş olan gerçek bir kayan-pencere
optimizasyonuydu.

Yazmadan önce sorulacak temel soru: **hatanın enerjisi yüksek frekansta mı?** Bir
yumuşatıcı ancak onu alabilir; düşük frekanslı sürüklenme gerçek hareketten ayırt
edilemez.

**Kâhin ölçümü** — hata sinyalinin yüksek frekanslı bileşeni *tamamen* silinseydi
(merkezi pencere; rolling bir çözümün **ulaşamayacağı** üst sınır), 2 permütasyon:

| yatak | MEVCUT | kâhin (w=10) | YF payı | kâhin (w=30) | YF payı |
|---|---|---|---|---|---|
| yarışma | 4,93 | 4,91 | **%0** | 4,89 | **%0** |
| video1 | 5,58 | 5,56 | %0 | 5,54 | %0 |
| video2 | 15,64 | 15,59 | %0 | 15,56 | %0 |
| v25_rgb_a | 4,94 | 4,93 | %0 | 4,92 | %0 |
| v25_rgb_b | 14,76 | 14,72 | %0 | 14,68 | %0 |
| v25_termal_a | 7,14 | 7,12 | %0 | 7,11 | %0 |
| v25_termal_b | 11,82 | 11,78 | %0 | 11,75 | %0 |
| v24 | 10,13 | 10,10 | %0 | 10,03 | %0 |
| **ORTALAMA** | **9,37** | **9,34** | | **9,31** | |

**Kâhin kazancı: −0,03 m (w=10) · −0,06 m (w=30).**

Hata enerjisinin **%0'ı** yüksek frekansta. Kesinti hatası saf **sürüklenmedir** —
hiçbir yumuşatıcı, filtre ya da kayan pencere ona dokunamaz. **Aile kapandı.**

> **Bu ölçüm tek başına bir aileyi değil, bir aileler kümesini açıklıyor.** Depoda
> ayrı ayrı reddedilmiş bütün son-işleme fikirleri — Hampel, EMA `smooth_alpha=1,0`,
> sürüklenme sönümleme, anti-windup `h_bant`/`h_sizinti` — aynı kök nedenden
> başarısız: **silinecek yüksek frekanslı hata yok.** Gelecekte "yörüngeyi
> yumuşatalım/filtreleyelim" biçiminde gelen her fikir, ölçüm yapılmadan bu maddeye
> yönlendirilebilir.

## Turun çıkarımı

Raporun denenmemiş kalemlerinden **üçü ölçümle kapandı** — dinamik nokta reddi,
uzaysal dengelenmiş inlier seçimi, kare kalite kapısı — ve hiçbiri için üretim kodu
yazılmadı. Ayrıca iki **doğrulama** çıktı: sağlık geçişinde sıçrama yok, sınav
penceresinde kuyruk yok.

Bu turun asıl kazancı bir yöntem değil, üç kapalı kapı ve bir teşhis: aykırıların
kaynağı **3B yapı** (bina/ağaç paralaksı). İki bağımsız ölçüm bunu söylüyor —
aykırılar hareketli nesnede değil (%1,4) ve uzaysal olarak kümelenmiyor (kapsama
0,94‒1,00). Merdiven katmanının 0,37‒0,62 inlier oranı buradan geliyor.

Açık kalan tek sağlam risk: **termal ve bir 4K yatağında kuyruk** (video2'de hatanın
%25'i en kötü %5 kareden, en kötü kare 85 m). Sınav pencerelerinde kuyruk olmadığı
için bu, sınavla yakalanamayan ayrı bir risktir.

## Yeniden koşturmak için

```
.venv\Scripts\python gorev2\tani_metrik.py --n-perm 3
.venv\Scripts\python gorev2\tavan_inlier_dagilimi.py
.venv\Scripts\python gorev2\tavan_dinamik_nokta.py --n-kare 200
.venv\Scripts\python gorev2\tavan_kare_kalitesi.py
.venv\Scripts\python gorev2\tavan_doygun_piksel.py --n-kare 120
```
