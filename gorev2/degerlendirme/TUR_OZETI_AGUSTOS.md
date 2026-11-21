# Görev 2 — Ağustos 2026 Ar-Ge turu: kapanış özeti

**Tarih aralığı:** 13‒16 Ağu 2026 · **Kapsam:** yalnız Görev 2 (PK, final formülünde %40)

---

## Sonuç

| | tur başında | **tur sonunda** |
|---|---|---|
| genelleme sınavı (puanlı oturumun kendi kareleri) | 11,43 m | **8,53 m** |
| 7 bağımsız yatak ortalaması | 8,12 m | **6,69 m** |

**Sevk edilen değişiklik sayısı: 1.** Kazancın tamamı ondan geliyor.

**Ölçümle kapatılan aile sayısı: 9.** Hiçbiri için üretim koduna dokunulmadı.

---

## 1. Sevk edilen: `z_sign` kanıt kapısı

Kayıt: [Z_ISARETI.md](Z_ISARETI.md)

Dünyanın z ekseninin aşağı mı yukarı mı baktığı **tek bir bit**tir (`z_sign`), ama
yanlışsa düşey kanal komple ters döner ve `current_height()` üzerinden yatay ölçeği de
bozar. Eski kural bu biti **ölçümün kendi büyüklüğüne** bakarak seçiyordu:

```python
guc = Σ|dz_unit| · c > 1,0 m   ->  z_sign = sign(Σ dz_unit·gt_dz)
```

`guc` uçağın gerçek irtifa hareketini değil ölçümün büyüklüğünü ölçtüğü için düz
uçuşta bile eşiği geçiyordu; karar pratikte yazı-turaydı ve yataklarda **%26‒40**
kararsızdı. Yeni kural kararı ancak pencerede gerçek irtifa hareketi (≥2 m) ve tutarlı
kanıt varken verir.

| | ESKİ | YENİ | fark |
|---|---|---|---|
| sınav pencere 1 | 17,26 | **11,46** | **−5,81 ±0,19** |
| sınav pencere 2 | 5,60 | 5,60 | eşit |
| video2 (termal) | 15,05 | **10,35** | −4,70 |
| v25_rgb_b (4K) | 13,01 | **10,65** | −2,37 |
| v25_termal_b | 11,15 | **8,20** | −2,96 |
| **7 yatak ort.** | **8,12** | **6,69** | **−1,43** |

Hiçbir yatakta kötüleşme yok. Kazanç **ayar yapılmayan yıllarda** (2025) ve **ayar
yapılmayan modalitede** (termal) çıkıyor — çünkü düzeltilen şey bir ayar değil bir
kestirim hatası. Eşikler 0,5‒5,0 m aralığında sonucu değiştirmiyor.

Sevkiyat doğrulandı: `denklik_testi.py` → tüm katmanlar bit-birebir; `final-v1.0`
içinde `dogrula.py` → tüm kontroller `[OK]`.

---

## 2. Kapatılan aileler

| aile | ön kabul ölçütü | ölçülen | kayıt |
|---|---|---|---|
| **Mutlak yükseklik gözlemi** | kâhin kazandırmalı | kusursuz bilgi **5/7 yatakta zarar veriyor** | [YUKSEKLIK_KAHINI.md](YUKSEKLIK_KAHINI.md) |
| **Kayan pencere / yumuşatma** | hata yüksek frekanslı olmalı | YF payı **%0**, kâhin kazancı −0,06 m | [MANUS_DENETIMI.md](MANUS_DENETIMI.md) §6 |
| **VGGT** | yerel geometri RAFT'a yakın olmalı | RAFT'ın **8 katı** kötü; ölçek %42‒52 tutarsız | [VGGT_TURU.md](VGGT_TURU.md) |
| **SEA-RAFT** | bir yatakta kazanmalı | 4 varyant, **hiçbiri kazanmadı** | [SEARAFT_TURU.md](SEARAFT_TURU.md) |
| **Çoklu ön-uç ensemble** | tavanın anlamlı payı yakalanmalı | kazancın **%73'ü en zayıf yataktan** | [ENSEMBLE.md](ENSEMBLE.md) |
| **Kısa atlama kurtarması** | red sık ve zararlı olmalı | red %2,4‒8,7 ve **zararsız** | [ENSEMBLE.md](ENSEMBLE.md) §5 |
| **Dinamik nokta reddi** | aykırıların >%10'u hareketli araçta | **%1,4** (kutu alanı %1,0) | [MANUS_DENETIMI.md](MANUS_DENETIMI.md) §4 |
| **Uzaysal dengelenmiş inlier** | kapsama <0,80 olmalı | kapsama **0,94‒1,00** | [MANUS_DENETIMI.md](MANUS_DENETIMI.md) §3 |
| **Kare kalite kapısı** | bozuk kare bulunmalı | bulanık/kontrast/donmuş **%0** | [MANUS_DENETIMI.md](MANUS_DENETIMI.md) §5 |

---

## 3. Turun asıl kazancı: dört yapısal teşhis

Bunlar tek tek fikirlerden daha değerli, çünkü **gelecekteki fikirleri önceden eliyorlar.**

### 3a. Yükseklik bir hata kaynağı değil, bir **telafi**

`dxy = M·e0·height`'ta yalnız sol taraf gözleniyor. Kesintide `height` serbest kalan
tek parametre ve `M·e0`'daki yanlılığı **yutuyor**. Gerçek değerine sabitlemek telafiyi
kaldırıp alttaki yanlılığı açığa çıkarıyor.

Bu, depoda **ayrı gerekçelerle** kaydedilmiş beş eski başarısızlığı tek açıklamaya
bağlıyor: `h_bant`/`h_sizinti`, `height_dz_katsayi=0`, `merdiven_olcek`,
`merdiven_hizalama`, genel `A` hizalama.

> **Kalıcı kural:** yükseklik/ölçek kanalına dokunan her fikir önce
> `kahin_yukseklik.py` ile sınanacak.

### 3b. Kesinti hatası **saf sürüklenmedir**

Hata enerjisinin **%0'ı** yüksek frekansta (8 yatak, w=10 ve w=30). Hiçbir yumuşatıcı,
filtre ya da kayan pencere ona dokunamaz. Bu, Hampel / EMA / sönümleme / anti-windup
ailelerinin tamamının ortak kök nedeni.

> **Kalıcı kural:** "yörüngeyi yumuşatalım" biçimindeki her fikir bu maddeye
> yönlendirilir.

### 3c. Aykırıların kaynağı **3B yapı**, hareketli nesne değil

İki bağımsız ölçüm: aykırıların yalnız %1,4'ü hareketli araç üzerinde (kutu alanı %1,0
— yani rastgeleyle aynı), ve inlier kümesi uzaysal olarak kümelenmiyor (kapsama
0,94‒1,00). Merdiven katmanının 0,37‒0,62 inlier oranı buradan geliyor ve "daha iyi
nokta seçimi" ile düzeltilebilecek bir şey değil.

### 3d. Yön ve ölçek kanalları **zıt yönde** çekiyor

SEA-RAFT yönü %21 daha iyi, ölçeği %31 daha kötü veriyor — üstelik akış yanlı değil ve
homografi uyumu daha iyi. Fark tamamen `(R, t, n)` ayrıştırmasında. Aynı gerilim
`hom_yaw_gevsek` mekanizmasında zaten kayıtlıydı; bir ön uç adayı bu takasın hangi
tarafında oturuyorsa sonucu o belirliyor. **Kesintide ölçek baskın olduğu için ölçek
kanalı belirleyicidir.**

---

## 4. Kalıcı altyapı

| araç | ne yapar |
|---|---|
| [`kahin_yukseklik.py`](../kahin_yukseklik.py) | yükseklik/ölçek fikirlerinin üst sınırını ölçer |
| [`tani_metrik.py`](../tani_metrik.py) | en kötü %5, geçiş sonrası drift, eksen MAE/RMSE |
| [`tani_pencere1.py`](../tani_pencere1.py) | hatayı ölçek/dönüklük/öteleme kanallarına ayırır |
| [`tani_akis_farki.py`](../tani_akis_farki.py) | iki ön ucu yanlılık/gürültü/inlier/artık olarak karşılaştırır |
| [`tani_kanal_farki.py`](../tani_kanal_farki.py) | iki ön ucu yön/ölçek kanallarına ayırır |
| [`tavan_*.py`](..) | 6 adet "önce tavanı ölç" betiği |
| `searaft` tracker ailesi | 4 varyant, `vendor/SEA-RAFT` altında kaynak |

**Yöntem notu:** bu turda 9 ailenin 9'u da **üretim kodu yazılmadan** kapandı, çünkü her
biri önce bir **tavan (kâhin) ölçümüyle** başladı. Deney tipik olarak 10‒30 dakika sürdü
ve haftalarca sürecek alt sistemleri baştan eledi. Bu kalıp turun en aktarılabilir
çıktısıdır.

---

## 5. Açık kalan tek risk

**Termal ve bir 4K yatağında kuyruk.** video2'de toplam hatanın **%25'i** en kötü %5
kareden geliyor ve en kötü kare **85 m**. v25_termal_b (%16,8) ve v25_rgb_a (%19,6) da
kuyruklu. Sınav pencerelerinde kuyruk **yok** (%5,8 ≈ tekdüze), dolayısıyla bu risk
genelleme sınavıyla **yakalanamıyor**.

Yarışma tek çekiliş olduğu için kuyruk doğrudan risktir. Bu turda kuyruğa yönelik iki
deneme yapıldı (ensemble, kurtarma seçicisi) ve ikisi de kabul ölçütünü geçemedi.
Sonraki tur için en anlamlı hedef budur.
