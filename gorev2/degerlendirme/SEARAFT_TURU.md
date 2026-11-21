# Görev 2 — SEA-RAFT turu (16 Ağu 2026)

> **Soru:** SEA-RAFT (ECCV 2024, oral) mevcut RAFT+homografi ön ucumuzu geçer mi?
>
> **Cevap: hayır.** Dört varyant denendi; **hiçbiri hiçbir yatakta kazanmadı.** En iyisi
> (`searaft_gl`) ayar yatağında berabere kalıyor ama **yarışma yatağında +1,36 m
> kaybediyor**. Mekanizma bulundu ve ilginç: SEA-RAFT **yönü daha iyi**, **ölçeği daha
> kötü** veriyor; kesintide hatamız ölçek-baskın olduğu için net sonuç kaybediyor.

Betikler: [`src/searaft_matching.py`](../src/searaft_matching.py) ·
[`tani_akis_farki.py`](../tani_akis_farki.py) · [`tani_kanal_farki.py`](../tani_kanal_farki.py) ·
[`tavan_kanal_bolusme.py`](../tavan_kanal_bolusme.py)

---

## 1. Fizibilite — temiz geçti

Deponun geçmişinde derin VO adaylarını öldüren şey hep **derleme duvarı** olmuştu
(DPVO/DROID/DPV-SLAM: `lietorch` ABI çökmesi). SEA-RAFT'ta o sorun yok:

| | sonuç |
|---|---|
| Özel CUDA kerneli | **gerekmiyor** (saf PyTorch) |
| torch 2.11 + Blackwell (sm_120) | çalışıyor |
| Lisans | BSD-3-Clause |
| Model | 19,7 M parametre · **121 ms/çift** (540×960) · **0,77 GB VRAM** |
| Ağırlık | HuggingFace (`MemorySlices/...`) |

Kıyas: VGGT 5+ GB VRAM tutuyor ve 518 px'te paging'e giriyordu.

## 2. Adil A/B nasıl kuruldu

`SeaRaftTracker`, `RaftTracker`'dan **türetildi** ve yalnız iki metot ezildi: modelin
kurulumu ve akışın hesaplanması. Izgara örnekleme, kenar payı, ileri-geri elemesi,
çalışma ölçeği, undistort — hepsi ebeveynden aynen geliyor. **Tek değişken akış ağı.**

Doğrulaması: çift başına 638 nokta (RAFT 646).

**Çözünürlük uyumu:** sevk edilen ağırlık 540×960'ta eğitilmiş; `auto_scale` 1920 px'i
0,5 ile tam 960×540'a indiriyor. Referans `custom.py`'nin ek `2**scale` küçültmesi
(config'te `scale=-1`) bu yüzden uygulanmadı — girdi zaten eğitim boyutunda.

### Zorunlu kapı: ölçüm GT ile korele mi?

| yatak | RAFT | SEA-RAFT |
|---|---|---|
| video1 | 0,789 | 0,792 |
| yarışma | 0,798 | 0,802 |

Geçti — entegrasyonda hata yok, karşılaştırma anlamlı.

## 3. Sonuç — dört varyant, üçü kaybediyor, en iyisi berabere

Aynı reçete, eşleştirilmiş permütasyon, video1:

| varyant | ağırlık / yineleme | hata [m] | RAFT'a fark |
|---|---|---|---|
| **RAFT** (mevcut) | torchvision, 12 yineleme | **3,90** | — |
| `searaft` | Spring ince ayarlı, 4 yineleme | 4,60 | **+0,69 ±0,06** |
| `searaft_l` | Spring ince ayarlı, 12 yineleme | 4,31 | **+0,41 ±0,12** |
| `searaft_g` | genelleme çekirdeği, 4 yineleme | 4,27 | **+0,37 ±0,27** |
| **`searaft_gl`** | **genelleme çekirdeği, 12 yineleme** | **3,91** | **−0,00 ±0,25** *(eşit)* |

### En iyi varyant, en önemli yatakta — kaybediyor

Beraberlik tek yatağa (video1) dayandığı için `searaft_gl` **yarışma yatağında** da
sınandı — puanlı oturumun kendi kareleri, finalin en iyi vekili (4 permütasyon):

| yatak | RAFT | `searaft_gl` | fark | |
|---|---|---|---|---|
| **yarışma** | **9,14** | 10,51 | **+1,36 ±1,32** | **KAYBETTİ** |
| video1 | 3,91 | 3,91 | −0,00 ±0,25 | eşit |
| **ORTALAMA** | **6,53** | 7,21 | **+0,68** | |

Yani en iyi yapılandırma bile yalnız ayar yatağında berabere kalıyor, yarışmaya en
yakın yatakta kaybediyor. (Temel varyant `searaft` da orada 8,53 → 9,79 vermişti.)

**Hiçbir SEA-RAFT varyantı hiçbir yatakta kazanmıyor.**

**Adil şans verildi.** İlk denemedeki iki tercih SEA-RAFT'ın aleyhineydi ve ikisi de
düzeltildi:
* `spring-L` ile `spring-M` **aynı mimari** — tek fark yineleme sayısı. 12'ye çıkarıp
  RAFT'la eşitledik.
* Spring'e ince ayarlı ağırlık yerine, makalenin çapraz-veri genelleme iddiasının ait
  olduğu **ince ayarsız** `Tartan-C-T-TSKH432x960-M` denendi.

İkisi birlikte açığı **kapattı** (`searaft_gl`, 4 permütasyonda −0,00 ±0,25) ama
**geçmedi**. Beraberlik, motor değiştirmeyi gerekçelendirmez: yeni bir bağımlılık,
~80 MB ağırlığın ve `vendor/` kaynak ağacının çevrimdışı paketlenmesi ve yarışma günü
yeni bir arıza yüzeyi demektir — hepsi sıfır ölçülen kazanç karşılığında.

## 4. Neden? — mekanizma bulundu

### 4a. Akış yanlı DEĞİL (XFeat'in arıza modu burada yok)

`tani_akis_farki.py`, video1 merdiven katmanı:

| motor | medyan \|akış\| px | ölçek oranı | açı farkı | nokta/çift | inlier | artık px |
|---|---|---|---|---|---|---|
| raft | 103,707 | 1,0000 | 0,00° | 575 | 0,532 | 0,385 |
| searaft | 103,016 | 0,9999 | 0,06° | 558 | 0,537 | 0,372 |
| searaft_l | 104,034 | 1,0004 | 0,05° | 574 | 0,530 | 0,373 |
| **searaft_g** | 103,792 | 0,9996 | 0,06° | 570 | **0,560** | **0,359** |

Ölçek oranı 1,0000'e %0,04 yakın, açı farkı 0,06°. XFeat'te kök neden "hareketi
sistematik olarak az ölçüyor" idi; **burada öyle bir şey yok.**

Dahası SEA-RAFT'ın **homografi uyumu daha iyi**: `searaft_g` inlier 0,560 (RAFT 0,532)
ve artık 0,359 px (RAFT 0,385). Yani düzlem modeline daha temiz oturuyor — ve yine de
yörünge daha kötü. Bu paradoks, farkın 2B yeniden-izdüşüm artığının **görmediği** bir
yerde olduğunu söylüyor: homografinin `(R, t, n)` ayrıştırmasında.

### 4b. Kanal ayrışımı — asıl bulgu

`tani_kanal_farki.py`, video1, 3 permütasyon:

| motor | hata [m] | **YÖN [°]** | **ÖLÇEK [%]** |
|---|---|---|---|
| raft | **3,904** | 1,707 | **2,989** |
| searaft | 4,598 | 1,434 | 3,284 |
| searaft_l | 4,312 | 1,721 | 3,401 |
| **searaft_g** | 4,271 | **1,341** | 3,930 |
| searaft_gl | 4,026 | 1,657 | 3,444 |

**SEA-RAFT yönü %21 daha iyi, ölçeği %31 daha kötü veriyor.**

Ve takas varyantlar arasında da görünüyor: `searaft_g` → `searaft_gl` (yineleme 4→12)
yönü kaybediyor (1,341 → 1,657) ama ölçeği kazanıyor (3,930 → 3,444). **Hiçbir SEA-RAFT
varyantı RAFT'ın ölçek doğruluğuna (%2,99) ulaşamıyor** — en iyisi %3,28. Bu tur boyunca
tutarlı olan tek şey bu: SEA-RAFT'ın ölçek kanalı bu veride yapısal olarak daha zayıf.

Bu, depoda zaten kayıtlı bir gerilimin başka bir noktası. `PairMeasurement` dokümanı:
*"yön ve ölçek hataları zıt eşik istiyor. Sıkı eşik yalnız gerçek zemin düzlemine uyan
noktaları bırakıp ölçeği doğru veriyor ama rotasyonu kötü koşullandırıyor. Gevşek eşik
tersi."* Sistem bunu iki **eşikle** çözüyor (`hom_yaw_gevsek`). SEA-RAFT bu takasın
farklı bir noktasında oturuyor — ve kesintide hata **ölçek-baskın** olduğu için yanlış
tarafında.

## 5. Mekanizmadan çıkan kurtarma denemesi — ❌ eşiğin altında

Doğal fikir: yönü SEA-RAFT'tan, ötelemeyi/ölçeği RAFT'tan al. Aynı desen sistemde
zaten var, sadece iki eşik yerine iki motorla.

`tavan_kanal_bolusme.py`, video1, 3 permütasyon:

| | MEVCUT | BÖLÜŞME | fark |
|---|---|---|---|
| hata [m] | 3,904 | 3,840 | **−0,064** |
| yön [°] | 1,707 | 1,548 | −0,159 |
| ölçek [%] | 2,989 | 2,965 | −0,024 |

Önceden yazılı ölçüt 0,2 m'ydi; ölçülen **−0,064 m**. Üstelik bu, her karede **iki
akış motoru** koşturmayı gerektirir (~2× maliyet). Ensemble turunda aynı takas zaten
reddedilmişti ([ENSEMBLE.md](ENSEMBLE.md)). **Kapandı.**

*(Yön 1,341°'ye değil 1,548°'ye iniyor çünkü boru hattı yaw'ı `_yaw_sec` ile sıkı/gevşek
fit arasından seçiyor; harman kazancı seyreltiyor.)*

## 6. Karar

**Aile kapandı. Sevkiyat yok.** RAFT birincil motor olarak kalıyor.

Kapı yeniden açılırsa hangi koşulda: SEA-RAFT'ın **ölçek** kanalı iyileşirse. Yön
tarafı zaten daha iyi; sorun `t/d` (düzlem uzaklığı) kestiriminin daha gürültülü
olması. Nadir hava görüntüsüne ince ayarlanmış bir SEA-RAFT ağırlığı bunu değiştirebilir
ama eğitim verisi elimizde yok.

**Kalıcı kazanım:** `searaft` ailesi `nokta_onbellek.py`'ye tracker olarak kayıtlı
(`searaft`, `searaft_l`, `searaft_g`, `searaft_gl`); kaynak `vendor/SEA-RAFT` altında.
Yeniden ölçmek dakikalar sürer. Ayrıca `tani_akis_farki.py` ve `tani_kanal_farki.py`
**herhangi iki motoru** karşılaştıran genel araçlar olarak kaldı — bundan sonraki her
ön uç adayı için doğrudan kullanılabilir.

## Yeniden koşturmak için

```
.venv\Scripts\python gorev2\nokta_onbellek.py --tracker searaft_g --yatak video1
.venv\Scripts\python gorev2\nokta_onbellek.py --tracker searaft_g --yatak video1 --spacing 8
.venv\Scripts\python gorev2\yontem_sinavi.py --yatak video1 --motor raft --motor-kiyas searaft_g --n-perm 3
.venv\Scripts\python gorev2\tani_akis_farki.py --katman sp8
.venv\Scripts\python gorev2\tani_kanal_farki.py --n-perm 3
.venv\Scripts\python gorev2\tavan_kanal_bolusme.py --yatak video1 --n-perm 3
```
