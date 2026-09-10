# Görev 2 — çoklu ön-uç ensemble turu (14 Ağu 2026)

> Manus raporunun **asıl mimari önerisi** (§10): tek ön uce güvenmek yerine iki ön uç
> eşzamanlı koşsun, her karede belirsizlik ağırlıklı birleştirilsin.
>
> Bizde şu an bir yedek **zinciri** var (RAFT → SIFT → LK): biri kurulamazsa diğerine
> geçilir. Raporun önerdiği ise eşzamanlı **seçim/harmanlama**.

Betikler: [`tavan_ensemble.py`](../tavan_ensemble.py) · [`kurtarma_sinavi.py`](../kurtarma_sinavi.py)

---

## 1. Önce tavan: kâhin seçici

Kod yazmadan önce üst sınır ölçüldü — her kesinti karesinde iki motorun ölçümünden
**GT'ye daha yakın** sonucu vereni seç. Kalibrasyon her zaman üretim motoruyla (RAFT)
yapılır; kâhin yalnız kesintide hangi ölçümün kullanılacağını seçer.

| yatak | RAFT | SIFT | **KÂHİN** |
|---|---|---|---|
| sınav pencere 1 | 11,29 | 16,19 | **11,27** |
| sınav pencere 2 | 5,12 | 5,29 | **3,65** |
| video1 | 3,84 | 4,07 | **2,83** |
| video2 (termal) | 10,75 | 9,52 | **5,88** |
| v24 | 4,63 | 6,13 | **3,24** |
| v25_rgb_a | 3,40 | 5,35 | 3,47 |
| v25_rgb_b | 10,15 | 12,29 | **8,31** |
| v25_termal_a | 4,91 | 6,25 | **4,55** |
| v25_termal_b | 8,12 | 10,06 | **6,96** |
| **7 yatak ort.** | **6,54** | 7,67 | **5,03** |

Tavan **−1,51 m (−%23)**. RAFT seçim oranı %50‒64, yani iki motor gerçekten farklı
hatalar yapıyor — aralarında seçilecek bilgi *var*.

**Ama sınav penceresi 1'de kâhin bile kazandırmıyor** (11,27 vs 11,29). Orada
seçilecek bir şey yok: SIFT 16,19 ile çok kötü ve ona her geçiş zarar veriyor.

*Not: kâhin açgözlüdür (o anki en yakın konumu seçer), bu yüzden katı bir üst sınır
değildir — miyop seçim durumu sonraki kareler için bozabilir.*

## 2. Tavana ulaşılabiliyor mu? Gerçekleştirilebilir seçiciler

Belirleyici soru bu: GT'ye bakmayan, yalnız ölçümün kendi istatistiklerine bakan bir
seçici tavanın ne kadarını yakalıyor?

| yatak | RAFT | KÂHİN | **inlier** | nkept | harman |
|---|---|---|---|---|---|
| **sınav pencere 1** | 11,29 | 11,27 | **15,17** | 12,58 | 14,88 |
| sınav pencere 2 | 5,12 | 3,65 | 2,75 | 5,06 | 2,70 |
| video1 | 3,84 | 2,83 | 3,92 | 4,02 | 3,86 |
| video2 | 10,75 | 5,88 | 8,48 | 10,94 | 8,48 |
| v24 | 4,63 | 3,24 | 4,16 | 6,82 | 4,21 |
| v25_rgb_a | 3,40 | 3,47 | 3,20 | 3,52 | 3,28 |
| v25_rgb_b | 10,15 | 8,31 | 10,34 | 12,34 | 10,67 |
| v25_termal_a | 4,91 | 4,55 | 4,85 | 5,40 | 4,83 |
| v25_termal_b | 8,12 | 6,96 | 7,75 | 9,42 | 7,76 |
| **7 yatak ort.** | **6,54** | 5,03 | **6,10** | 7,49 | 6,16 |
| | | *tavanın* | *%29'u* | *%−63* | *%26* |

### ❌ Her karede seçim — REDDEDİLDİ

`inlier` ortalamada kazandırıyor (−0,44 m) **ama sınav penceresi 1'i +3,88 m bozuyor.**
Deponun kendi kuralı gereği (>+3 m = felaket) ve kabul ölçütü gereği (sınavın iki
penceresinde de geçmeli) sevk edilemez. `merdiven_gnc_c` ile birebir aynı desen:
bir veri ailesinde güçlü, bir yatakta felaket.

**Mekanizma:** SIFT az sayıda ama iyi eşleşen nokta bulduğu için **yüksek inlier oranı
raporluyor**, oysa hareket kestirimi yanlı. İnlier oranı bu yüzden motorlar *arasında*
güvenilir bir hakem değil. Bu, `SuperPoint+LightGlue` turunda kayda geçen arıza
modunun aynısı (inlier %21 ama medyan akış doğru görünüyordu).

### ❌ `nkept` — REDDEDİLDİ

+0,95 m, tavanın **eksi** %63'ü. Nokta sayısı kalite göstergesi değil — aynı ders
"uyarlanabilir ızgara" maddesinde de kayıtlı (396 → 660 nokta, +1,3 m).

### ❌ `harman` (inlier ağırlıklı konum harmanı) — REDDEDİLDİ

`inlier` ile aynı davranış (pencere 1: 14,88). Yumuşak harmanlama sert seçimin
sorununu çözmüyor; yanlış motoru düşük ağırlıkla da olsa modele sokuyor.

## 3. Mekanizma değişikliği: `kurtarma`

Yukarıdakilerin hepsi **her karede** seçim yapıyor. Tanı metrikleri
([MANUS_DENETIMI.md](MANUS_DENETIMI.md)) ise farklı bir şey söylüyor: termalde gerçek
bir **kuyruk** var (video2'de hatanın %25'i en kötü %5 kareden, en kötü kare 85 m).
Doğru hedef her kare değil, **yalnız RAFT'ın çöktüğü kareler.**

`kurtarma`: SIFT'e yalnız RAFT merdiven inlier'i tabanın altındayken **ve** SIFT
belirgin üstünken geç; aksi hâlde RAFT'ta kal.

İlk ölçüm (2 permütasyon, taban 0,20 · üstünlük 1,5×):

| yatak | RAFT | **kurtarma** | fark |
|---|---|---|---|
| **sınav pencere 1** | 11,29 | **10,66** | **−0,63** ✓ |
| **sınav pencere 2** | 5,12 | **4,05** | **−1,07** ✓ |
| video1 | 3,84 | 3,84 | eşit |
| video2 | 10,75 | 11,09 | +0,34 |
| v24 | 4,63 | 3,95 | −0,68 ✓ |
| v25_rgb_a | 3,40 | 3,40 | eşit |
| v25_rgb_b | 10,15 | 10,08 | −0,07 |
| v25_termal_a | 4,91 | 4,91 | eşit |
| v25_termal_b | 8,12 | 8,12 | eşit |
| **7 yatak ort.** | **6,54** | **6,48** | −0,06 |

**Sınavın iki penceresi de kazanıyor, felaket yok** — üç kabul ölçütü de görünüşte
sağlanıyor. Ama kazanç küçük ve 2 permütasyon ince; ayrıca eşikler (0,20 / 1,5×) tek
bir denemeden geliyor.

*İronik nokta:* kurtarma, hedeflediği yatakta (video2) **kaybediyor** (+0,34).
Kazanç sınav pencerelerinden ve v24'ten geliyor.

### 3a. Kapı — 4 permütasyon, eşleştirilmiş (taban 0,20 · üstünlük 1,5×)

| yatak | MEVCUT | KURTARMA | fark | tetik | |
|---|---|---|---|---|---|
| **sınav pencere 1** | 12,20 ±1,51 | 11,36 ±1,13 | **−0,84 ±0,44** | %1,0 | KAZANDI |
| sınav pencere 2 | 6,09 ±2,07 | 5,35 ±2,30 | −0,74 **±1,19** | %0,9 | *gürültü içinde* |
| video1 | 3,91 | 3,91 | +0,00 | %0,1 | eşit |
| video2 | 10,20 | 10,36 | +0,16 ±0,33 | %0,2 | *gürültü içinde* |
| **v24** | 4,84 | 3,93 | **−0,92 ±0,33** | %7,8 | KAZANDI |
| v25_rgb_a | 3,31 | 3,31 | +0,00 | %0,0 | eşit |
| v25_rgb_b | 10,83 | 10,75 | −0,08 ±0,14 | %2,3 | marjinal |
| v25_termal_a | 3,94 | 3,94 | +0,00 | %0,0 | eşit |
| v25_termal_b | 8,14 | 8,14 | +0,00 | %0,0 | eşit |
| **7 yatak ort.** | **6,45** | **6,33** | **−0,12** | | |

Üç kabul ölçütü de sağlanıyor, felaket yok. **Ama dürüst okuma:**

* Yalnız **iki sonuç gürültü üstünde**: pencere 1 (~2σ) ve v24 (~2,8σ). Pencere 2'nin
  −0,74'ü ±1,19 eşleştirilmiş std ile gürültü içindedir, kazanç sayılmaz.
* **Yedi yatağın dördü hiç dokunulmuyor** (tetiklenme %0,0) — değişiklik büyük ölçüde
  atıl.
* En net kazanç **en zayıf yataktan** (v24: GT'de z yok, kalibrasyon dosyası yok) ve
  orada tetiklenme %7,8 ile diğerlerinin üç katı.
* **Mekanizma açıklaması tam oturmuyor:** kurtarma termal kuyruk için tasarlandı ama
  video2'de hafifçe kaybediyor ve üç termal yatağın ikisinde hiç tetiklenmiyor.
  İşe yarıyorsa öngörülen sebepten değil.

### 3b. Eşik taraması — ✅ aşırı uyum YOK

| taban | üstünlük | pencere1 | pencere2 | 7 yatak | tetik |
|---|---|---|---|---|---|
| *(mevcut)* | — | 11,29 | 5,12 | 6,54 | — |
| 0,10 | 1,5 | 11,29 | 5,12 | 6,52 | %0,1 |
| 0,15 | 1,5 | 11,02 | 4,25 | 6,43 | %0,7 |
| **0,20** | **1,5** | 10,66 | 4,05 | 6,48 | %1,4 |
| 0,30 | 1,5 | 11,15 | **3,73** | **6,31** | %2,5 |
| 0,40 | 1,5 | 11,15 | **3,73** | **6,27** | %3,3 |
| 0,20 | 1,0 | **9,87** | 3,87 | 6,54 | %4,1 |
| 0,20 | 2,0 | 11,36 | 4,09 | 6,47 | %0,7 |
| 0,20 | 3,0 | 11,29 | 5,12 | 6,47 | %0,4 |

**Tetiklenen her kombinasyon pencere 2'yi iyileştiriyor** ve 7 yatak ortalaması tabanla
**monoton** düşüyor (6,54 → 6,27). Tek noktada tepe yok, hiçbir kombinasyonda felaket
yok. Etki gerçek ve sağlam.

Ama tarama seçilen eşiğin **en iyisi olmadığını** da gösteriyor: taban 0,30‒0,40 yatak
ortalamasında belirgin daha iyi. Eşik kararı 4 permütasyonla yeniden veriliyor.

### 3c. Kapı — taban 0,30 · 4 permütasyon

| yatak | MEVCUT | KURTARMA | fark | tetik |
|---|---|---|---|---|
| sınav pencere 1 | 12,20 ±1,51 | 11,76 ±0,97 | −0,44 ±0,56 | %1,2 |
| sınav pencere 2 | 6,09 ±2,07 | 5,19 ±2,39 | −0,90 ±0,93 | %1,2 |
| video1 | 3,91 | 3,81 | −0,11 ±0,43 | %0,4 |
| video2 | 10,20 | 9,89 | **−0,31 ±0,15** | %0,6 |
| **v24** | 4,84 | 3,54 | **−1,30 ±0,31** | **%16,1** |
| v25_rgb_a | 3,31 | 3,31 | +0,00 | %0,0 |
| v25_rgb_b | 10,83 | 10,75 | −0,08 ±0,12 | %2,8 |
| v25_termal_a | 3,94 | 3,99 | +0,05 ±0,10 | %0,0 |
| v25_termal_b | 8,14 | 8,19 | +0,05 ±0,05 | %0,2 |
| **7 yatak ort.** | **6,45** | **6,21** | **−0,24** | |

## 4. ❌ KARAR: SEVK EDİLMEDİ

Üç kabul ölçütü de sağlanıyor ve hiçbir yatak +0,05 m'den fazla kaybetmiyor. Buna
rağmen sevk edilmedi. Belirleyici olan şu kontrol:

**v24'ün tetiklenme oranı %16,1 — diğer yatakların beş katı.** Ortalama v24 hariç
yeniden hesaplandığında:

| | MEVCUT | KURTARMA | fark |
|---|---|---|---|
| 7 yatak (v24 dâhil) | 6,45 | 6,21 | **−0,24** |
| **6 yatak (v24 hariç)** | **6,72** | **6,66** | **−0,065** |

**Kazancın %73'ü tek başına v24'ten geliyor** — ve v24 elimizdeki en zayıf yatak:
GT'sinde z ekseni yok, kalibrasyon dosyası yok, varsayılan 2026 profili kullanılıyor.

Redde gerekçeler:

1. **v24 hariç kazanç 0,065 m** — üretim yoluna ikinci bir ön uç sokmayı karşılamıyor.
2. **Sınav kazançları gürültüden ayrılmıyor.** İşaret iki eşikte de tutarlı biçimde
   negatif ama büyüklük oynak (pencere1: 0,20'de 2σ, 0,30'da 0,8σ).
3. **Mekanizma anlaşılmadı.** Termal kuyruk için tasarlandı; 0,20'de video2'de
   kaybediyor, 0,30'da 2σ ile kazanıyor. Neyin işe yaradığı açıklanamıyor.
4. **Mimari risk.** `live_estimator.py` tek motorlu; ikinci ön uç yarışma gününde yeni
   bir arıza yüzeyi demek. Finale beş hafta kala bu takas iyi değil.

**Lehine olan da kayda geçiyor:** hiçbir yatakta zarar yok, eşik taraması geniş bir
aralıkta monoton ve temiz, tetiklendiği her yerde işaret doğru yönde. Fikir yanlış
değil; bu hâliyle bedeli karşılamıyor. Ölçülebilir bir kazanç payı çıkarsa
(ör. v24 dışı yataklarda >0,3 m) yeniden açılabilir.

## 5. Bu turdan çıkan asıl iş

Kurtarma, RAFT merdiven inlier'i **0,30'un altına düştüğünde** devreye giriyor — yani
tam olarak `min_inlier_ratio` kapısının ölçümü **attığı** karelerde. Şu an o karelerde
sistem ardışık zincire düşüyor.

Demek ki doğru soru "SIFT mi kullanalım" değil:

> **Merdiven ölçümü reddedildiğinde RAFT'la daha iyi ne yapabiliriz?**

Örneğin daha kısa bir keyframe atlaması (hop 8 yerine 4) o karede hâlâ geçerli bir
ölçüm verebilir — ikinci motor gerektirmeyen, çalışma maliyeti sıfır bir çıkış.

### ❌ Ölçüldü ve bu da kapandı — [`tani_merdiven_red.py`](../tani_merdiven_red.py)

| yatak | kesinti karesi | **RED oranı** | RED karelerde hata | KABUL karelerde hata | sp4 kapıyı geçiyor |
|---|---|---|---|---|---|
| yarışma | 1698 | %8,7 | 5,76 | 4,65 | %64,9 |
| video1 | 1547 | %2,4 | 5,81 | 5,62 | %89,2 |
| video2 | 1547 | %3,9 | **14,29** | **14,92** | %66,7 |

İki şey söylüyor:

1. **Red nadir:** kesinti karelerinin yalnız %2,4‒8,7'si. (Daha önce kaydedilen %14,8
   yalnız sınav penceresi 1'e aitti; tüm takvimde oran çok daha düşük.)
2. **Reddedilen kareler daha kötü değil.** yarışmada +%24, video1'de +%3, video2'de
   **−%4** — yani termalde reddedilen karelerin hatası kabul edilenlerden *düşük*.
   Sistematik bir zarar kaynağı yok.

Kısa atlama teknik olarak mümkün (sp4 red anlarının %65‒89'unda kapıyı geçiyor) ama
**kurtarılacak bir şey yok.** Bu, ensemble bulgusuyla da tutarlı: SIFT kurtarması
yataklarda %0,0‒2,8 tetikleniyordu ve v24 dışında 0,065 m kazandırıyordu.

> **Aile kapandı.** `min_inlier_ratio` kapısının attığı ölçümler ne yeterince sık ne
> de yeterince zararlı; oraya bir kurtarma mekanizması koymak — hangi motorla olursa
> olsun — kazandırdığından fazlasını karmaşıklık olarak alıyor.

*Uyarı (yöntem notu): bir karedeki hata o karenin ölçümünü değil birikmiş geçmişi
yansıtır, dolayısıyla "RED vs KABUL kare hatası" karşılaştırması zayıf kanıttır. Karar
esas olarak **red oranının düşüklüğüne** ve ensemble turunun bağımsız sonucuna
dayanıyor.*

## Yeniden koşturmak için

```
.venv\Scripts\python gorev2\tavan_ensemble.py --n-perm 2
.venv\Scripts\python gorev2\kurtarma_sinavi.py --n-perm 4 --taban 0.30
.venv\Scripts\python gorev2\kurtarma_sinavi.py --tara --n-perm 2
```
