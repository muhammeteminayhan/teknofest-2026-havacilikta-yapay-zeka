# Görev 2 — sınav penceresi 1'in tanısı ve `z_sign` kanıt kapısı (13 Ağu 2026)

> Bu tur `OLCUM_TURU_2.md` §5'in **ilk açık işiyle** başladı: "sınav 1. penceresinin
> teşhisi". Teşhis bir ayar sorunu değil, **tek bitlik bir kestirim hatası** çıkardı.

---

## 1. Soru

Genelleme sınavı ortalaması 11,08 m ama hata neredeyse tamamen tek pencerede:

| | pencere 1 | pencere 2 |
|---|---|---|
| ölü hesap uzunluğu | 449 kare | 499 kare |
| puanlanan blok | 900–1050 | 1550–1650 |
| **hata** | **17,27 m** | **3,04 m** |

Ölü hesap süresi neredeyse aynı, yani fark **süre değil**. Denenen hiçbir
`min_inlier_ratio` değeri ve yumuşak harmanlama pencere 1'i düzeltmemişti.

Araç: `gorev2/tani_pencere1.py` — üretim yolunu (önbellekli, `kabul.py` ile aynı)
koşturur ama her karede durum makinesinin **içini** kaydeder.

## 2. Hata nerede? — düşey kanalda

| | pencere 1 | pencere 2 |
|---|---|---|
| yatay (xy) hata | 11,43 m | 3,03 m |
| **düşey (z) hata** | **10,99 m** | **0,20 m** |
| Procrustes ölçek `c` | 1,0555 | 0,9775 |
| Procrustes dönüklük | −0,94° | −0,65° |

Hata birikimi eğrisi olayı tek bir yere sabitliyor — 450 kare ölü hesaptan sonra
hata **hâlâ 4,4 m**, sonraki 70 karede 17,8 m'ye fırlıyor:

```
pencere 1:  k451:0,0  k599:2,7  k747:4,0  k895:4,4  k969:17,8  k1043:21,1
pencere 2:  k1051:0,2 k1199:2,3 k1347:3,8 k1495:6,4 k1569:2,3  k1643:6,5
```

GT'ye bakınca sebep görünüyor: uçak **940–1000 kareleri arasında ~10 m irtifa
değiştiriyor** (GT z: −0,42 → −9,35). Pencere 1'in puanlanan bloğu tam oraya denk
geliyor.

## 3. Ölçüm mü bozuk? — hayır, işaret ters

Merdiven ölçümünü gerçek birim yükseklik değişimiyle karşılaştırdım
(`dz_unit > 0` = yükseklik azaldı, çünkü boru hattı `h *= 1 − dz_unit` uyguluyor):

| küme | n | korelasyon | işaret uyumu |
|---|---|---|---|
| manevra dışı (451–940) | 66 | +0,367 | %80 |
| **MANEVRA (940–1010)** | **9** | **−0,981** | **%0** |

Büyüklük doğru (ölçüm gerçeğin %75'i), **işaret tamamen ters**. İlk iki hipotez
ölçümle elendi:

* **Homografi ayrıştırmasının iki-kollu belirsizliği mi?** Hayır — düzlem normalinin
  nadire göre eğimi manevrada **1,9° medyan** (manevra dışında 3,0°), yani normal
  gayet düzgün. Kol seçimi bozulmuyor.
* **Merdiven katmanına özgü mü? Ardışık katman kurtarır mı?** Hayır — ardışık katman
  da aynı yönde yanılıyor (kor −0,765, işaret uyumu %0) ve iki katman manevrada
  **%100 birbiriyle uyumlu**. Katmanlar arası tutarlılık kapısı bunu yakalayamaz.

Yani ölçümler doğru; onlara uygulanan **işaret** yanlış.

## 4. Kök neden: `z_sign` yazı-turayla seçiliyor

`z_sign`, dünyanın z ekseni aşağı mı (+1) yukarı mı (−1) baktığını söyleyen **tek
bit**. Kesintide düşey adım `dz = z_sign · dz_unit · height` ile uygulandığı için
yanlışsa düşey kanal komple ters döner. Sağlıklı dönemde de `current_height()`
üzerinden yükseklik durumunu — dolayısıyla **yatay ölçeği** — bozar.

Eski kural (`fit_alignment`):

```python
t    = sum(dz_unit * gt_dz)
guc  = sum(|dz_unit|) * c          # metre cinsinden "toplam düşey sinyal"
z_sign = sign(t) if (guc > 1.0 and t != 0.0) else prev_z_sign
```

Pencere 1'in kalibrasyon penceresinde ölçülen değerler:

| büyüklük | değer | yorum |
|---|---|---|
| `t = Σ dz_unit·gt_dz` | **−1,04e−4** | kendi gürültü tabanının kat kat altında |
| `guc` | 1,46 m | kapı **GEÇİYOR** |
| **GT z gezinmesi** | **0,188 m** | uçak düz uçuyor — **ortada bilgi yok** |
| `kor(dz_unit, gt_dz)` | −0,071 | ilişki sıfır |

**Kusur `guc` büyüklüğünde.** O, *ölçümün* büyüklüğünü ölçüyor, uçağın *gerçek*
irtifa hareketini değil. Düz uçuşta bile 60 örnekte `|dz_unit|` medyanı ~0,004 ve
`c ≈ 42 m` olduğu için `guc` 1,4 m'yi aşıyor; kapı pratikte **hiç kapanmıyor**.
İşaret, gürültünün işaretinden seçiliyor.

Pencere 2 aynı kapıdan **+1** ile çıkıyor (GT z gezinmesi 1,00 m, `t = +5,55e−4`) —
ama oradaki korelasyon da −0,004, yani o karar da yazı-tura; sadece doğru yüzü geldi.

### Bu tek uçuşa özgü değil

Sağlıklı karelerin yüzde kaçında `z_sign` hangi değerde:

| yatak | +1 | −1 | GT z aralığı | son karar |
|---|---|---|---|---|
| video1 | %73 | **%27** | 42,5 m | +1 |
| video2 | %62 | **%38** | 51,9 m | **−1** |
| v24 | %100 | %0 | 0,0 m | +1 |
| v25_rgb_a | %100 | %0 | 0,6 m | +1 |
| v25_rgb_b | %74 | **%26** | 22,3 m | +1 |
| v25_termal_a | %100 | %0 | 0,6 m | +1 |
| v25_termal_b | %60 | **%40** | 22,3 m | +1 |

Gerçek düşey hareketi olan dört yatağın hepsinde işaret **%26–40 oranında
çalkalanıyor**. `video2`'de son karar −1, yani teslim edilen sistem o uçuşta düşey
kanalı ters çalıştırıyor.

## 5. Doğrudan kanıt: işareti +1'e zorlarsak

Tek değişkenli, üretim yolunda:

| | mevcut | `z_sign=+1` zorlanmış |
|---|---|---|
| pencere 1 | **17,27 m** (z 10,99 · xy 11,43) | **11,52 m** (z **1,21** · xy 11,43) |
| pencere 2 | 3,04 m (z 0,20 · xy 3,03) | 3,04 m (değişmedi) |
| **sınav ortalaması** | **10,16 m** | **7,28 m** |

Düşey hata 10,99 → 1,21 m. Yatay kanal hiç değişmiyor (11,43 → 11,43), yani teşhis
tam yerinde: **hata düşey işaretin kendisiydi.**

## 6. Uygulanan düzeltme

`fit_alignment` içindeki kural, kanıtı doğru büyüklükle ölçecek şekilde değişti:

```python
gercek_dusey = ptp(gt_z)                       # penceredeki GERÇEK irtifa gezinmesi
carpim = dz_unit * gt_dz
kanit  = |Σ carpim| / Σ|carpim|                # ∈ [0,1]; 1 = her örnek aynı yönde
z_sign = sign(Σ carpim) if (gercek_dusey >= z_min_hareket
                            and kanit >= z_min_kanit) else prev_z_sign
```

İki koşul da sağlanmazsa **bilgi yoktur** ve önceki işaret korunur. Başlangıç değeri
+1'dir ve şartnamenin z-aşağı konvansiyonuyla uyumludur. Eşikler
`EstimatorConfig.z_min_hareket` (varsayılan 2,0 m) ve `z_min_kanit` (0,30).

`kanit` ölçütü bilinçli olarak korelasyon değil: birbirini götüren gürültüyü
(Σ|ab| büyük, |Σab| küçük) doğrudan cezalandırır ve ölçekten bağımsızdır.

## 7. Kabul ölçütü (ölçümden önce yazıldı)

1. Genelleme sınavı: **her iki pencerede de** kötüleşme yok, en az birinde kazanç
2. 7 bağımsız yatak ortalaması kötüleşmiyor
3. Hiçbir yatakta felaket (>+3 m) yok

Araç: `gorev2/zsign_sinavi.py` — eşleştirilmiş A/B; temel hat kolu eski kuralı
**birebir kopyalayarak** yamalar, yani karşılaştırma tek değişkenlidir.

## 8. A/B sonucu — ✅ UYGULANDI

3 permütasyon, eşleştirilmiş, tek değişken.

### Genelleme sınavı (puanlı oturumun kendi kareleri)

| | ESKİ | YENİ | fark | |
|---|---|---|---|---|
| pencere 1 | 17,26 ±0,16 | **11,46 ±0,30** | **−5,81 ±0,19** | KAZANDI |
| pencere 2 | 5,60 ±2,24 | 5,60 ±2,24 | +0,00 ±0,00 | eşit |
| **ortalama** | **11,43** | **8,53** | **−2,90** | |

### Yöntem sınavı — 7 bağımsız yatak

| yatak | ESKİ | YENİ | fark | |
|---|---|---|---|---|
| video1 | 3,90 ±0,12 | 3,90 ±0,12 | +0,00 | eşit |
| **video2** | 15,05 ±0,67 | **10,35 ±0,86** | **−4,70 ±0,25** | KAZANDI |
| v25_rgb_a | 3,43 ±0,08 | 3,43 ±0,08 | +0,00 | eşit |
| **v25_rgb_b** | 13,01 ±1,49 | **10,65 ±1,47** | **−2,37 ±0,02** | KAZANDI |
| v25_termal_a | 4,18 ±1,53 | 4,18 ±1,53 | +0,00 | eşit |
| **v25_termal_b** | 11,15 ±0,17 | **8,20 ±0,15** | **−2,96 ±0,03** | KAZANDI |
| v24 | 6,13 ±0,17 | 6,13 ±0,17 | +0,00 | eşit |
| **ORTALAMA** | **8,12** | **6,69** | **−1,43** | |

**Üç kabul ölçütü de sağlandı:** hiçbir pencerede/yatakta kötüleşme yok, ortalama
düşüyor, felaket yok. Kazanan dört yatak, §4'te işaret kararsızlığı ölçülen dört
yatağın ta kendisi — mekanizma tahmini tutuyor. Değişmeyen üç yatak düz uçuşlu
(GT z aralığı 0,0–0,6 m), yani orada zaten karar verilecek bir şey yoktu.

> Bu, deponun **tek uçuşa aşırı uyum** dersinin tersi bir vaka: kazanç ayar
> yapılmayan yıllarda (2025) ve ayar yapılmayan modalitede (termal) ortaya çıkıyor,
> çünkü düzeltilen şey bir ayar değil bir **kestirim hatası**.

## 9. Dürüstlük notu: kapı fiilen ne yapıyor?

"Yeni kural aslında `z_sign = +1` sabiti mi?" sorusunu ölçtüm:

| yatak | sağlıklı kare | kapı AÇIK | karar +1 | karar −1 | pencerede GT z gezinme medyanı |
|---|---|---|---|---|---|
| video1 | 681 | 117 (%17) | 117 | **0** | 0,39 m |
| video2 | 674 | 52 (%8) | 52 | **0** | 0,14 m |
| v25_rgb_a | 495 | 0 (%0) | 0 | 0 | 0,07 m |
| v25_rgb_b | 663 | 64 (%10) | 64 | **0** | 0,10 m |
| v25_termal_a | 497 | 0 (%0) | 0 | 0 | 0,07 m |
| v25_termal_b | 665 | 68 (%10) | 68 | **0** | 0,10 m |
| v24 | 411 | 0 (%0) | 0 | 0 | 0,00 m |
| yarışma | 525 | 83 (%16) | 83 | **0** | 0,23 m |

İki şey söylüyor:

1. **Kural gerçekten uyarlanabilir kalıyor** — düşey hareketi olan yataklarda kapı
   karelerin %8–17'sinde açılıyor ve işaret orada veriden kestiriliyor.
2. **Ama açıldığı 384 kerenin hepsinde +1 çıkıyor, hiç −1 yok.** Bu, +1'in bu
   şartnamenin z-aşağı konvansiyonu olduğunun bağımsız doğrulamasıdır. Pratikte
   elimizdeki bütün veride sonuç "+1"e denk geliyor; fark, artık bunun **kanıtla**
   olması ve gürültünün işareti çevirememesi.

Ayrıca kalibrasyon penceresindeki GT z gezinmesinin medyanı her yatakta **0,00–0,39 m**
— yani çoğu zaman gerçekten bilgi yok. Eski kuralın neden bu kadar sık yanıldığı
buradan görünüyor.

## 10. Eşik duyarlılığı — değer tek yatağa uydurulmadı

3 permütasyon, sınavın iki penceresi + değişen iki yatak:

| `z_min_hareket` | `z_min_kanit` | pencere 1 | pencere 2 | video2 | v25_termal_b |
|---|---|---|---|---|---|
| *(ESKİ kural)* | — | 17,26 | 5,60 | 15,05 | 11,15 |
| 0,50 | 0,30 | **11,46** | 5,60 | **10,35** | **8,20** |
| 1,00 | 0,30 | 11,46 | 5,60 | 10,35 | 8,20 |
| **2,00** | **0,30** | 11,46 | 5,60 | 10,35 | 8,20 |
| 5,00 | 0,30 | 11,46 | 5,60 | 10,35 | 8,20 |
| 2,00 | 0,00 | 11,46 | 5,60 | 10,35 | 8,20 |
| 2,00 | 0,15 | 11,46 | 5,60 | 10,35 | 8,20 |
| 2,00 | 0,50 | 11,46 | 5,60 | 10,35 | 8,20 |
| 2,00 | 0,80 | 11,46 | 5,60 | 10,35 | 8,20 |

**Dokuz kombinasyonun tamamı birebir aynı sonucu veriyor.** Yani kazanç bir eşik
ayarından gelmiyor; eski `guc` istatistiğinin karar vermesini engellemekten geliyor.
Hangi makul kapı konursa konsun sonuç aynı.

**Dürüstlük notu:** bu tabloda `z_min_kanit` **hiç bağlayıcı olmuyor** (0,00 ile 0,80
aynı sonucu veriyor) — ayırt edici olan tek koşul penceredeki gerçek irtifa
gezinmesi. `z_min_kanit` yine de kodda tutuluyor: pencerede ≥2 m gerçek düşey hareket
olup ölçümün çöp olduğu bir uçuş (ör. düşük dokulu termal) için ikinci bir koruma.
Elimizdeki veride tetiklenmediği bilinerek bırakılmıştır — `min_kept` ve `max_dz_unit`
gibi ölü kapılarla aynı statüde (bkz. `OLCUM_TURU_2.md` §4b).

## Yeniden koşturmak için

```
.venv\Scripts\python gorev2\tani_pencere1.py --n-perm 3
.venv\Scripts\python gorev2\zsign_sinavi.py --n-perm 3
.venv\Scripts\python gorev2\zsign_sinavi.py --tara --n-perm 3
```
