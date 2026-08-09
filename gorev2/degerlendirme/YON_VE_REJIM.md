# Görev 2 — yön kaynağı, rejim tanısı ve motor sorusu

> Bu tur **üç soruyu kapatmak** için yapıldı. Üçü de ölçüldü, üçü de **hayır** çıktı —
> ama üçü de gelecekte tekrar denenmesini önleyecek somut sayı bıraktı. Ayrıca bir
> önceki turda kendi yarattığım bir regresyon bulunup kapatıldı.

## 0. ✅ Kapatılan regresyon: `_meas_from_row`

Önceki turda reddedilen "ayrı yön kanalı" denemesi için `PairMeasurement`'a iki alan
eklenmişti (`hom_yaw_gevsek`, `hom_inlier_gevsek`). `evaluation._meas_from_row` alanları
`{k: row[k] for k in __dataclass_fields__}` ile okuduğu için **eski CSV önbelleklerinde
`KeyError`** veriyordu:

```
PairMeasurement alanlari: 20 · eski CSV sutunlari: 20
EKSIK: ['hom_yaw_gevsek', 'hom_inlier_gevsek']  ->  KeyError
```

Kırdığı araçlar: `test_yatagi.py`, `parametre_tarama.py`, `spacing_tarama.py`,
`surukleme_tarama.py`, `loop_closure_test.py`. (`kabul.py`/`yontem_sinavi.py` npz yolunu
kullandığı için etkilenmemişti — bu yüzden fark edilmemişti.)

**Düzeltme:** ölçüm artık **dataclass varsayılanlarından** kurulup yalnız mevcut sütunlar
üzerine bindiriliyor → bundan sonraki her yeni alan kendiliğinden geriye uyumlu.
Bool alanlar için NaN tuzağı da kapatıldı (`bool(nan) is True` olduğu için, eksik/NaN
bool sütunu varsayılana düşer). Doğrulama: `test_yatagi.py` yeniden koşuyor,
`kabul.py` sonuçları **birebir** korundu (11,08 / 3,77 / 14,30).

## 1. ❌ Yön kaynağı: benzerlik yaw'ı (`yon_kaynak="sim"`)

**Hipotez.** Yön ve ölçek hataları bu depoda üç kez bağımsız olarak zıt kestirici tercih
etti (ds0,75; termal eşik taraması; SIFT). Öyleyse yönü daha iyi koşullanmış bir
kestiriciden almak kazandırmalı. Benzerlik (4 serbestlik) düzlemsel nadir sahnede
homografi ayrıştırmasından iyi koşullanmıştır — düzlem normali seçimi ve 4-aday
belirsizliği yoktur. Ve `sim_yaw` **zaten her ölçümde var**: sıfır ek hesap.

**Ön kontrol (uzlaşım).** İşaret/ölçek uyumu doğrulandı:

| yatak | katman | korelasyon | eğim | medyan \|fark\| |
|---|---|---|---|---|
| video1 | ardışık / merdiven | 0,9990 / 0,9995 | 1,008 / 1,006 | 0,034° / 0,060° |
| video2 | ardışık / merdiven | 0,9953 / 0,9972 | 0,991 / 0,986 | 0,028° / 0,073° |
| yarışma | ardışık / merdiven | 0,9993 / 0,9996 | 0,999 / 1,000 | 0,019° / 0,046° |

**Sonuç — kesin ret** (8 permütasyon, eşleştirilmiş):

| yatak | temel | `yon_kaynak=sim` | fark |
|---|---|---|---|
| video1 | 3,885 | **18,310** | **+14,4** |
| sınav | 12,115 | 15,065 | +2,95 |
| video2 | 14,988 | 16,974 | +1,99 |

video1 yön hatası **1,69° → 9,91°**.

**Neden.** Adım başına fark medyanda yalnız 0,034° ama **sistematik** (regresyon eğimi
1,007 → benzerlik yaw'ı ~%0,7 büyük). 600 karelik bir kesintide bu ~20°'ye birikir.

**Asıl çıkarım:** homografinin dönmesi zaten doğru — **zayıf halka yön değil**. Bu,
"yönü başka bir kestiriciden al" ailesinin tamamının (esas matris dâhil) önselini
zayıflatır: sorun kestiricinin koşullanması değil. Kalan hata **ölçekte**.

## 2. ❌ Rejim dedektörü: kurulmayacak (tanı `rejim_tanisi.py`)

**Soru.** Sistem iki 2025 termal uçuşunda "hiçbir şey yapmamaktan kötü"ydü. Bunu
çalışma anında (GT'siz) tespit edip `tut`'a düşmek mümkün mü?

**Asimetri.** Kazanç ~8–23 m (tek yatak); yanlış tetikleme maliyeti ~60 m
(video1'de `tut` = 63,6 m, sistem 3,8 m). Bu yüzden anahtar ancak **ayrım çok genişse**
kurulmalıydı.

**Ölçüm:**

| yatak | sistem | `tut` | `sabit_hız` | fit denemesi | **red oranı** | **hizalama yaşı** | ölçek oynak% |
|---|---|---|---|---|---|---|---|
| video1 | 3,77 | 63,6 | 160,7 | 660 | **0,00** | **1** | 11,4 |
| yarışma | 6,18 | 46,1 | 193,8 | 514 | **0,00** | **1** | 10,9 |
| v24 | 6,46 | 37,0 | 89,6 | 200 | 0,05 | 1 | 20,6 |
| video2 | 14,30 | 78,9 | 106,4 | 653 | 0,19 | 1 | 12,8 |
| v25_termal_a | 27,20 | 47,2 | 42,4 | 430 | **0,78** | **601** | 12,7 |
| v25_termal_b | **76,51** | 53,1 | 87,3 | 493 | **0,78** | **604** | 8,1 |

**Sinyal mükemmel ayırıyor** — red oranı 0,00–0,19 & hizalama yaşı 1 **vs** red 0,78 &
yaş ~600. (Yaş 600 = kesintiye girerken kullanılan hizalama, o kesintiden 600 adım önce
kabul edilmiş; yani tüm sağlıklı dönem boyunca hiçbir hizalama kabul edilmemiş.)

**Ama yanlış soruyu ayırıyor.** İki termal yatak **birebir aynı imzaya** sahip, oysa:

| yatak | sistem | `tut` | `tut`'a düşmek |
|---|---|---|---|
| v25_termal_a | 27,20 | 47,2 | **−20,00 m (kaybettirir)** |
| v25_termal_b | 76,51 | 53,1 | **+23,44 m (kazandırır)** |

Sinyal "hizalama güvenilmez"i ayırıyor ama **"tutmak daha iyi olur"u ayırmıyor** —
bunlar farklı sorular. Net beklenen kazanç **+1,7 m**, buna karşılık çalışan bir uçuşta
yanlış tetiklenme maliyeti ~60 m.

**KARAR: anahtar kurulmadı.** Tanı aracı (`rejim_tanisi.py`) kayıt olarak kalıyor;
ileride "tutmak daha iyi olur"u ayıran bir sinyal bulunursa yeniden değerlendirilir.

## 3. ❌ SIFT birincil motor olmuyor (yedek olarak kalıyor)

Önceki tur SIFT'in **üç** yatakta ortalamasının RAFT'ınkinden düşük çıktığını göstermişti
(9,21 vs 9,72) — ama fark tamamen tek yataktan (termal) geliyordu. Bu yüzden aynı reçeteyle,
**sekiz yatakta**, eşleştirilmiş olarak ölçüldü (5 permütasyon):

| yatak | RAFT | SIFT | fark | karar |
|---|---|---|---|---|
| video1 | 3,91 | 4,01 | +0,10 ±0,48 | eşit |
| video2-termal | 15,04 | **11,45** | **−3,59** ±3,26 | KAZANDI |
| **sınav** | **11,88** | 15,08 | **+3,20** ±3,84 | eşit ama felaket eşiğinde |
| v24 | 6,08 | 13,27 | **+7,19** ±6,74 | KAYBETTİ |
| v25_rgb_a (4K) | 3,37 | **2,24** | **−1,13** ±0,41 | KAZANDI |
| v25_rgb_b (4K) | 13,37 | 27,12 | **+13,75** ±1,85 | KAYBETTİ |
| v25_termal_a | 27,16 | 30,42 | +3,27 ±0,40 | KAYBETTİ |
| v25_termal_b | 61,40 | 62,62 | +1,22 ±6,72 | eşit |
| **ORTALAMA** | **17,77** | **20,78** | **+3,00** | |

**2/8 yatakta iyi, 4 yatakta felaket (>+3 m).** Karar kuralı — "çoğunlukta kazanacak
VE hiçbir yatakta felaket üretmeyecek" — açıkça sağlanmıyor.

**SIFT birincil motor OLMUYOR; CPU yedeği olarak kalıyor** (o rolde 15,86 → 9,21 m
kazandırmıştı, bkz. [KLASIK_ESLESTIRICI.md](KLASIK_ESLESTIRICI.md)).

**Metodoloji notu:** üç yataklık sinyal (SIFT ortalaması RAFT'tan iyi) **tek-yatak
yanılsamasıydı** — sekiz yatakta işaret tersine döndü (−0,10 → +3,00). Geçen turda
GNC ile aynı hataya düşülmüştü; bu kez ölçüldüğü için karara dönüşmedi.
Termaldeki −3,59 m ise iki termal yatağın diğerinde (+3,27) tersine döndüğü için
modaliteye özel motor seçimini de desteklemiyor.

## Turun çıkarımı

Üç hipotez de aynı yerde tökezledi: **kalan hata yön kestiriminde ya da rejim
seçiminde değil, ölçekte.** Ve ölçek, ölçümün kendi yanlılığından geliyor — bu depoda
şimdiye kadar denenen hiçbir son-işleme (filtreleme, sönümleme, kısıtlama, robust
kestirim, model seçimi) onu düzeltemedi. Literatür de bir kapıyı kapattı: monoküler
metrik derinlik nadir bakışta zero-shot %28–32 AbsRel veriyor
([AerialMetric, ECCV 2026](https://arxiv.org/html/2606.29716v1)), yani mevcut %2,5–6,3'lük
ölçek hatamızın 5–10 katı.

## Yeniden koşturmak için

```
.venv\Scripts\python gorev2\rejim_tanisi.py
.venv\Scripts\python gorev2\kabul.py --cfg yon_kaynak=sim --n-perm 8
.venv\Scripts\python gorev2\yontem_sinavi.py --motor-kiyas sift --n-perm 5
```
