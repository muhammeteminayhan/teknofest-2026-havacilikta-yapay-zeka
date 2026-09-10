# Görev 3 — `min_confidence` 0,25 vs 0,10

**Tarih:** 10 Eylül 2026 · **Sonuç:** **0,25 korundu.** 0,10 denendi, sayıca daha iyi
göründü, **görsel kanıtla reddedildi.**

## Neden bakıldı

Final paketi hazırlanırken depo, puanlı oturumda **RNT 50,97 (8 takım içinde 2.)** alan
Temmuz teslim paketiyle dosya dosya karşılaştırıldı. Görev 3'ün üç Python dosyası
bit-birebir aynıydı; tek fark `template_config.yaml` içindeki bir satırdı:

| paket | `min_confidence` | puanlandı mı |
|---|---|---|
| `bugbuster_5202745_teslim` (Temmuz) | **0,1** | evet — RNT 50,97 |
| depo / sevk paketi | 0,25 | hayır |

Yani sevk edeceğimiz yapılandırma, puan alan yapılandırma değildi. Ayar ölü de değil;
`threshold_method: statistical` modunda istatistiksel kapıya **ek** bir mutlak taban
olarak duruyor (`template_matcher.py:235`):

```python
if max_val < mu + m.get('k_std', 3.0) * sd or max_val < self.min_confidence:
    return None
```

---

## ⚠️ Önce bir ölçüm hatası: sahte "ayar etkisiz" sonucu

İlk tarama `test_yatagi.py --ayar matching.min_confidence=...` ile yapıldı ve
**0,10 / 0,25 / 0,35 / 0,45 / 0,55 değerlerinin beşi de birebir aynı** çıktı. Bundan
"eşik bu veride hiç bağlayıcı değil" sonucu çıkarıldı — **yanlıştı.**

Sebep: `TemplateMatcher.__init__` bazı ayarları config sözlüğünden **kopyalayıp** kendi
özniteliğinde tutuyor (`cls_threshold`, `min_confidence`, `threshold_ratio`,
`noise_floor`, `fusion`, `augment`). `--ayar` yalnız sözlüğü değiştirdiği için bu beş
koşunun hepsi aslında dosyadaki 0,25 ile koştu. Tarama bir no-op'tu.

`test_yatagi.py > ayar_uygula()` düzeltildi: override sonrası kopyalanan öznitelikler
tazeleniyor. Bu tuzak `k_std`'yi etkilemiyor (o, çağrı anında sözlükten okunuyor) —
daha önceki `k_std` taramaları geçerlidir.

**Ders:** bir taramanın bütün satırları birebir aynı çıkıyorsa, ilk şüphe "parametre
etkisiz" değil, "parametre hiç uygulanmadı" olmalı.

---

## Gerçek ölçüm

Dosya değeri değiştirilerek koşuldu. Determinizm ayrıca doğrulandı: aynı ayarla iki
koşu birebir aynı sonucu verdi.

| | `0,25` | `0,10` |
|---|---|---|
| recall (aktif karede tespit var mı) | %81,9 (86/105) | **%85,7 (90/105)** |
| aralık dışı tespit (nesne sahnede YOK) | 6/14 | **11/14** |
| kutu alanı medyanı | 129 720 px² | 84 426 px² |

Sayıya bakan biri 0,10'u seçerdi: recall 3,8 puan artıyor.

## Görsel doğrulama — kararı tersine çeviren adım

`mc_ab.py` iki ayarı aynı karelerde koşturup **yalnız ayrıştıkları kareler** için yan
yana panel üretti:

```
a_var_b_yok    0      <- 0,10 hicbir tespiti KAYBETTIRMIYOR
b_var_a_yok    8      <- 0,10'un EKLEDIGI tespitler
ikisi_de      91
hicbiri       16
```

Eklenen 8 tespitin 4'ü aralık dışı (üretimde zaten imkânsız, pencere kontrolü var),
4'ü aktif aralıkta. **O dördü tek tek büyütülüp bakıldı** (`mc_ab_YAKIN.jpg`):

| kare | referans nesne | 0,10'un bulduğu yer | doğru mu |
|---|---|---|---|
| ref2 · 200 | basketbol sahası | yol kenarı / bordür | ❌ |
| ref4 · 1178 | siyah-beyaz yıldız motifi | kapı önü paspası | ❌ |
| ref4 · 1189 | siyah-beyaz yıldız motifi | kiremit çatı | ❌ |
| ref4 · 1200 | siyah-beyaz yıldız motifi | yaya geçidi çizgileri | ❌ |

**Dördü de yanlış nesnede.**

## Neden recall yanılttı

Test yatağının `recall` ölçütü *"o karede bir tespit **var mı**"* diye sorar,
*"**doğru yerde mi**"* diye değil — çünkü elimizde yer gerçeği kutusu yok (yatağın kendi
başlığında da yazılı). 0,10 daha çok kare için kutu üretiyor, ama ürettiği fazladan
kutular yanlış. Puan mAP ile hesaplandığı için bu **zarardır**:

* tespit üretmemek → TP yok, FP de yok
* yanlış yerde tespit üretmek → TP yok, **FP var** → kesinlik düşer

Aynı yönde ikinci işaret: nesnenin hiç olmadığı kontrol karelerinde ateşleme
6/14 → 11/14'e çıkıyor; yani sistem 0,10'da "bir şey bulmak" için daha istekli.

## Karar

**`min_confidence: 0.25` korundu.** Sevk edilen Görev 3, puanlı oturumda 50,97 alan
koddan yalnız bu tek değerde ayrılıyor ve ayrılma yönü ölçüm + görsel kanıtla
gerekçelendirilmiştir.

`dogrula.py` beş eşiği (`threshold_method`, `k_std`, `box_k_std`, `min_confidence`,
`max_box_area_ratio`) ekrana yazar ve sevk edilen değerlerden saparsa uyarır — yarışma
sabahı tek bakışta görülür.

## Üretilen dosyalar

- `mc_ab.py` — A/B + fark panelleri (bu klasöre yazar)
- `mc_ab_ref*.jpg` — ayrışan 8 karenin yan yana paneli
- `mc_ab_YAKIN.jpg` — dört aktif-aralık tespitinin yakın çekimi (**karar bu panelde verildi**)
