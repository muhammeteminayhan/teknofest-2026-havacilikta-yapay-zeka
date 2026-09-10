# Görev 2 — yükseklik kâhini: mutlak yükseklik gözlemi ailesi ❌ **KAPANDI**

> ## ⛔ SONUÇ: **KUSURSUZ** bir mutlak yükseklik gözlemi bile sistemi KÖTÜLEŞTİRİYOR
>
> Bu tur hiçbir üretim değişikliği yapmadı. Bir **kâhin (oracle) deneyi** ile, inşa
> edilmesi planlanan alt sistemin **ulaşabileceği en iyi sonuç** ölçüldü ve ölçüm
> fikri çürüttü. Alt sistem yazılmadan kapandı.

**Tarih:** 13 Ağu 2026 · **Betik:** [`gorev2/kahin_yukseklik.py`](../kahin_yukseklik.py)

---

## 1. Fikir neydi ve neden makuldü

Kesintide konum adımı:

```python
dxy = M @ e0 * height          # vo_pipeline.update / update_ladder
height = height * (1 - dz_unit)
```

Yükseklik **çarpımsal olarak taşınıyor** ve **mutlak bir gözlemi yok**. Ölçülen sapma
(z işareti düzeltildikten sonra, `tani_pencere1.py`):

| | sınav pencere 1 | sınav pencere 2 |
|---|---|---|
| toplam hata | 11,35 m | 3,04 m |
| yatay / düşey | **11,26** / 1,20 | 3,03 / 0,20 |
| ölçek hatası `c` | **%5,2** | %2,3 |
| yükseklik durumu ort. sapma | **%6,5** (en kötü %15,5) | %2,0 |

`dxy ∝ height` olduğu için %6,5'lik yükseklik sapması ile %5,2'lik ölçek hatasının
örtüşmesi güçlü bir işaretti: **yatay ölçek hatası aslında yükseklik hatasıdır.**

Planlanan çözüm: yerdeki duran taşıtları cetvel olarak kullanmak (Görev 1 dedektörü
zaten her karede koşuyor, ek maliyet ~sıfır) ve yüksekliğe mutlak bir gözlem eklemek.

## 2. Önce ödülü ölç: kâhin deneyi

Alt sistemi yazmadan önce **üst sınırı** ölçtük. Kesinti boyunca yükseklik durumu
(`self.height` ve merdiven çapasının `a["h"]`'si) **gerçek değere zorlandı**:

```python
h_gercek(i) = dondurulmus_hizalama.current_height(GT_z[i])
```

Bu, hayal edilebilecek **en iyi** nesne-ölçeği kanalıdır: gürültüsüz, yanlılıksız,
her karede mevcut. Gerçek bir kanal bundan ancak daha kötü olabilir.

## 3. Sonuç — 2 permütasyon, 7 yatak + sınavın iki penceresi

| yatak | MEVCUT | **KÂHİN** | fark | |
|---|---|---|---|---|
| **sınav pencere 1** | 11,29 | **21,19** | **+9,90** | **+%88** ❌ |
| sınav pencere 2 | 5,12 | **2,96** | −2,16 | −%42 ✓ |
| video1 (RGB) | 3,84 | 6,02 | +2,18 | +%57 ❌ |
| video2 (termal) | 10,75 | **7,89** | −2,86 | −%27 ✓ |
| v25_rgb_a (4K) | 3,40 | 4,12 | +0,72 | +%21 ❌ |
| v25_rgb_b (4K) | 10,15 | 9,56 | −0,59 | −%6 ✓ |
| v25_termal_a | 4,91 | 6,52 | +1,61 | +%33 ❌ |
| v25_termal_b | 8,12 | 8,95 | +0,83 | +%10 ❌ |
| v24 (2B) | 4,63 | 7,60 | +2,96 | +%64 ❌ |
| **7 yatak ortalaması** | **6,54** | **7,24** | **+0,69** | ❌ |

**Kusursuz bilgi, yedi yatağın beşinde ve sınavın en önemli penceresinde zarar
veriyor; ortalamada da kötü.** Ön kabul ölçütü (her iki pencerede kazanmalı, ortalama
kötüleşmemeli) sağlanmıyor. **Aile kapandı.**

## 4. Bu bir uygulama hatası değil — λ taraması

"Belki referans yükseklik yanlıştır" olasılığı ayrıca sınandı: kâhin yüksekliğine
λ çarpanı uygulanıp minimum arandı.

| λ | pencere 1 | pencere 2 | video2 |
|---|---|---|---|
| *(kâhin yok)* | **11,35** | 3,04 | 10,24 |
| 0,80 | 25,50 | 31,98 | 17,93 |
| 0,90 | 21,37 | 14,54 | 10,69 |
| 0,95 | *20,72* | 5,95 | 7,69 |
| **1,00** | 21,09 | ***3,11*** | ***7,37*** |
| 1,05 | 22,34 | 11,73 | 8,36 |
| 1,10 | 24,32 | 20,45 | 9,81 |
| 1,20 | 29,84 | 37,91 | 14,07 |

Eğri temiz bir parabol ve **pencere 2 ile video2'de minimum tam λ=1,00'de** — yani
referans doğru, kâhin doğru uygulanmış. Buna karşılık **pencere 1'de λ=0,80…1,20
aralığındaki hiçbir değer "hiçbir şey yapmamak"tan iyi değil.**

## 5. Mekanizma: yükseklik bir hata kaynağı değil, bir **telafi**

`dxy = M · e0 · height` denkleminde ölçülen tek şey soldaki `dxy`'nin GT'ye
uyumudur; `M`, `e0` ve `height` ayrı ayrı gözlenmiyor. Kesintide `height`
**serbest kalan tek parametredir** ve sürüklenirken `M · e0`'daki yanlılığı
kısmen yutuyor. Gerçek değerine sabitlemek bu telafiyi kaldırıyor ve alttaki
yanlılığı açığa çıkarıyor.

Bunu destekleyen bağımsız gözlem: **v24 yatağında GT'de z ekseni yok**, yani kâhin
orada yüksekliği tamamen **donduruyor** — ve hata %64 artıyor. Bu, depoda daha önce
`height_dz_katsayi=0` olarak ölçülen sonucun (4,34 → 13,6 m) aynısıdır.

Neden pencere 1 en çok zarar gören yatak: kalibrasyon **yavaş ve neredeyse düz** bir
parçada yapılıyor ama ölü hesap **çok dolambaçlı** bir uçuşa uygulanıyor.

| | pencere 1 | pencere 2 |
|---|---|---|
| kalibrasyonda adım medyanı | 0,587 m | 1,143 m |
| kalibrasyonda yön std | **6,2°** | 20,4° |
| kalibrasyonda özdeğer oranı λ1/λ2 | **47,9** | 31,3 |
| ölü hesapta doğrusallık (net/yol) | **0,227** | 0,569 |

Kötü koşullu bir kalibrasyondan gelen `M·e0` yanlılığı büyük; `height` onu taşıyor.

## 6. Bu bulgu geçmiş başarısızlıkları da açıklıyor

Depoda ayrı ayrı, farklı gerekçelerle kaydedilmiş **beş** başarısızlık aslında aynı
mekanizmanın örnekleriymiş — hepsi yükseklik/ölçek kanalını kısıtlıyor:

| denenen | sonuç | kayıt |
|---|---|---|
| `h_bant` / `h_sizinti` (anti-windup) | 3,51 → 5,61 … 12,04 | `EstimatorConfig` notu |
| `height_dz_katsayi = 0` (yükseklik sabit) | 4,34 → 13,6 m | `OZET.md` |
| `merdiven_olcek` (1 SD merdiven ölçeği) | sınav +1,51 | `OLCUM_TURU_2.md` §3 |
| `merdiven_hizalama` (4 SD) | sınav p2 17,19 → 25,97 | `OLCUM_TURU_2.md` §3 |
| genel `A` hizalama (ölçeği yutuyor) | 4,34 → 13,6 m | `TANI_TURU.md` D1/D2 |

Bunlar "ESS ~2" ya da "araç gerçekten alçalıyor" gibi ayrı gerekçelerle
açıklanmıştı. Kâhin deneyi **tek bir birleştirici açıklama** veriyor:
**yükseklik durumu serbest bir telafi olarak yük taşıyor ve onu kısıtlayan her şey —
doğru değerle kısıtlamak dâhil — kazandırdığından fazlasını kaybettiriyor.**

## 7. Kalıcı kural

> **Yükseklik/ölçek kanalını kısıtlayan, sabitleyen ya da doğrudan gözleyen her yeni
> fikir, önce `kahin_yukseklik.py` ile sınanmalıdır.** Kâhin (kusursuz bilgi)
> kazandırmıyorsa, gerçek hiçbir gözlem kazandıramaz. Deney ucuzdur (~10 dk) ve
> alt sistem yazmadan cevap verir.

## 8. Ne yapmalı — hedef değişti

Kalan hata `height`'ta değil, **`M · e0`'da**: kalibrasyondan gelen eksen/ölçek
köprüsü ve homografiden gelen birim yatay adım. Bir sonraki iş bu kanala gitmeli
(uzaysal dengelenmiş nokta seçimi — `OLCUM_TURU_2.md` §5'te "denenmedi" olarak duruyor).

Yan bulgu, ileride kullanılabilir: **kâhin, sınav penceresi 2 ve video2'de gerçekten
kazandırıyor** (−%42 ve −%27). Yani yükseklik gözlemi *bazı rejimlerde* değerli.
Ama hangi rejimde olduğunu **çalışma anında bilemiyoruz** (rejim dedektörü ailesi
`YON_VE_REJIM.md` §2'de zaten kapatıldı), ve yanlış rejimde bedeli çok ağır (+%88).

## Yeniden koşturmak için

```
.venv\Scripts\python gorev2\kahin_yukseklik.py --n-perm 2
.venv\Scripts\python gorev2\kahin_yukseklik.py --tara
```
