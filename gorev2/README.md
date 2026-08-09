# Görev 2 — GPS'siz Pozisyon Kestirimi

TEKNOFEST 2026 Havacılıkta Yapay Zeka yarışması, **İkinci Görev** (toplam puanın %40'ı):
GPS kesintisi sırasında, alt-görüş kamerası görüntülerinden hava aracının video
başlangıcına göre X/Y/Z yer değiştirmesini (metre) kestirmek.

## Klasör Yapısı

```
gorev2/
├── src/         # Asıl boru hattı modülleri (yarışmada .py olarak koşar)
│   ├── calibration.py   # kamera iç parametreleri + undistort
│   └── data_io.py       # CSV/video okuma, 7.5 FPS altörnekleme
├── reports/     # Analiz figürleri (aşağıdaki yol haritasında üretilenler)
└── outputs/     # Ara çıktılar (önbellek CSV'leri)
```

> **Geliştirme defterleri (notebook'lar) arşivlendi.** Çalışan sistem tamamen
> `src/*.py` ile koşar; aşağıdaki yol haritasında adı geçen `01_*.ipynb … 07_*.ipynb`
> defterleri (keşif, doğrulama, figür üretimi — **teknik tasarım raporu için**) proje
> dışındaki `Desktop/havacilikta-yz-arsiv/gorev2-notebooks/` klasöründedir. Ürettikleri
> figürler `reports/` altında durmaya devam eder.

## Ortam

Sanal ortam proje kökünde: `havacilikta-yz-project\.venv` (Python 3.13)

```powershell
# Kurulum (bir kere)
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

Not: `pip` komutlarını daima `python -m ...` üzerinden çağırın;
sistemde birden fazla Python sürümü var.

## Yol Haritası

- [x] Aşama 0 — İskelet + ortam
- [x] Aşama 1 — Veri keşfi (`01_veri_kesfi.ipynb`) — bulgular: +z aşağı (NED benzeri),
      sabit eksen eşlemesi yetersiz (yaw ~360° dönüyor) → sürekli yön takibi şart,
      GSD·fx ile yükseklik kestirimi doğrulandı
- [x] Aşama 2 — VO çekirdeği (`02_vo_cekirdegi.ipynb`, `src/matching.py`, `src/geometry.py`) —
      işaret/eksen uzlaşımları sentetik sahneyle kanıtlandı; benzerlik + homografi iki yol;
      20 sn'lik gerçek kesinti provası: z hatası 1,5-5 m, adım büyüklükleri GT'yi izliyor
- [x] Aşama 3 — Uçtan uca kestirim + değerlendirme (`03_degerlendirme.ipynb`,
      `src/vo_pipeline.py`, `src/scale_alignment.py`, `src/evaluation.py`) —
      20 senaryo × 2 video: **füzyon E ort. 7,9 m** (tabanlar: tut 55 m, sabit hız 67 m);
      figürler `reports/` altında
- [x] Aşama 4 — GPU yükseltmeleri (`04_gpu_raft.ipynb`, `src/raft_matching.py`) —
      karar: **birincil sistem RAFT+homografi** (taze senaryolarda E ort. 3,9 m / maks 18,9),
      yedek: LK+füzyon (GPU'suz, E 5,4 m); parametre ayarı ezber dersiyle doğrulandı
- [x] Aşama 4+ — 4K hazırlığı + keyframe zinciri (`05_4k_ve_keyframe.ipynb`) —
      çalışma çözünürlüğü politikası (4K otomatik; değişmezlik kanıtı <4 cm, bütçe ~0,4 s/kare);
      **keyframe merdiveni (8 adım): taze senaryolarda E ort. 3,56 / med 1,73 / maks 16,4 m**
      (iki bağımsız sette tutarlı kazanç); LK'de merdivenin işe yaramadığı da belgelendi
- [x] Aşama 5 — Sunucu entegrasyonu + tam prova (`06_entegrasyon_prova.ipynb`,
      `src/live_estimator.py`, `src/simulator.py`, `src/sim_eval.py`,
      `sistem/` (canlı bağlantı arayüzü) entegrasyonu + 80/dk RateLimiter) —
      **tam prova: 2256/2256 kare, E = 3,54 m, sağlıklıda 0 hata, tempo ihlalsiz,
      28,3 dk (sınır 60), canlı↔offline farkı 0,05 m**; yarışma günü kontrol listesi defterde
- [x] Tema dayanıklılığı (`07_tema_dayanikliligi.ipynb`, `src/degrade.py`, `src/midair_adapter.py`)
      — 3 kaynak: sentetik bozulma (E'li), MOBDrone deniz (66 klip), Mid-Air GT'li (sunny/sunset/
      foggy/winter). **Bulgu:** sis/günbatımı/bulanıklık/kar sistemi bozmuyor (RAFT taşıyor),
      sadece aşırı gece E'yi ~2× artırıyor. Denizde "sim↔hom uyuşmazlık kapısı" denendi ve
      **GT'li testte E'yi net bozduğu için KALDIRILDI** (negatif sonuç, dürüst belgelendi);
      sistem gate öncesi temiz haline döndü (E ort. 3,56 doğrulandı)

---

## Puanlı oturum sonrası iyileştirme çalışması (17 Tem 2026)

Çevrim içi yarışmada PK **55,29 puan (8/8)** ile en zayıf görevimiz oldu. Ardından resmî
metrik altında ölçüme dayalı iyileştirme turları yapıldı — **tam rapor:
[`degerlendirme/OZET.md`](degerlendirme/OZET.md)**.

**Ulaşılan nokta:** örnek veri setinde ortalama Öklid hatası 4,535 → 3,757 → 2,795 m
ölçüldü. Ancak bu kazançların **hiçbiri** puanlı oturumun kendi kareleriyle yapılan
genelleme sınavını geçemedi; **teslim edilen sistem, puanlı oturumda koşan reçetenin
kendisidir** (`method="homografi"`, varsayılan `smooth_alpha=0.6`, spacing=8, düşey
bileşen homografiden). Ayrıntı:
[`degerlendirme/DUSEY_KANAL.md`](degerlendirme/DUSEY_KANAL.md).

> **Alınan asıl ders:** bütün ayar çalışması `THYZ_2026_Ornek_Veri_1.MP4` üzerindeydi
> ve o video **çevrim içi yarışmanın test oturumunun ta kendisi**. Puanlı oturum başka
> bir uçuş. "Rastgele kesinti takvimi" ile yapılan aşırı-uyum kontrolü bunu yakalayamaz:
> takvim çeşitleniyor, uçuş çeşitlenmiyor. Artık her değişiklik
> [`genelleme_sinavi.py`](genelleme_sinavi.py) kapısından geçmek zorunda.

**Test yatağı** (`test_yatagi.py`): örnek veri seti (tam GT) + puanlı oturum logundan
çıkarılan **gerçek kesinti takvimi** (703 sağlıklı / 1547 kesinti, üç uzun blok:
449/499/599 kare) + resmî metrik (`competition_error`). Aşırı uyumu yakalamak için her
deneme ayrıca rastgele kesinti takvimlerinde de ölçüldü.

| Deneme | Sonuç |
|---|---|
| **`smooth_alpha=1.0`** (ölçüm EMA'sı kapalı) | ⛔ **geri alındı** — örnek veride kazançlı (4,535→4,343 m; rastgele −%21) ama genelleme sınavında ortalamada geride (17,25 vs 16,71 m) |
| **Düşey kanal** (`z_sim_agirlik=1.0`) | ⛔ **geri alındı** — örnek veride −%25,6, puanlı oturumun karelerinde iki pencerede de kayıp |
| Diğer 10 parametre | ❌ etkisiz ya da tek senaryoya aşırı uyum |
| Loop closure / görsel yer tanıma (`src/yer_tanima.py`) | ❌ kazanç yok → **varsayılan kapalı** |
| Sürüklenme sönümleme (yaw / yükseklik) | ❌ zararlı (`yaw_katsayi=0` → 162 m) |
| Keyframe aralığı (spacing 4–24) | ❌ **8 en iyi** (16 tek takvimde iyi göründü, 15 uzun-kesinti senaryosunda doğrulanamadı) |

**Öne çıkan teşhis:** kesinti hatası rastgele değil **sistematik** — tahmin yörüngesi GT'nin
~2° döndürülmüş ve %3–4,5 ölçeklenmiş hâli (bunlar bilinseydi blok hatası 6,28 m yerine
1,45 m olurdu). Ancak güncellemeleri sönümlemek durumu ağırlaştırıyor: araç gerçekten dönüp
alçaldığı için güncellemeler doğru ve gereklidir. Sapma birikimden değil, **ölçümün kendi
yanlılığından** geliyor → sonraki kazanç ölçüm katmanında (optik akış / yükseklik gözlemi)
aranmalı, parametre ayarında değil.

**Kıyas tabanları:** *tut* 63,6 m, *sabit hız* 160,7 m — sistem bunların **14×** altında.

---

## Ölçüm turu 2 — finale hazırlık (7 Ağu 2026)

**Tam rapor: [`degerlendirme/OLCUM_TURU_2.md`](degerlendirme/OLCUM_TURU_2.md)**

Finale gidiliyor; final oturumunun uçuşu, teması ve kamerası (RGB 1080p / 4K / **termal**)
bilinmiyor. Bu tur iki şey yaptı: ölçme yöntemini düzeltti ve tek büyük kazancı buldu.

### 1) Ölçme yöntemi — gürültü tabanı bulundu
OpenCV RANSAC nokta **sırasına** duyarlıdır: aynı noktalar farklı sırada verilince eşit
skorlu ama farklı bir düzlem seçilir. Aynı önbellekte 10 permütasyon → **3,964 ± 0,226 m,
yayılım %18,7 (±0,44 m, %95)**; sınav yatağında **±1,85 m**. Yani tek koşuyla ölçülen
~0,4 m'lik (sınavda ~2 m'lik) her fark gürültü olabilir — geçmiş turların bazı kararları
(örn. `ds0,75`'in +0,034 m ile reddi) bu tabanın altındadır. Artık her A/B **N permütasyonda,
eşleştirilmiş** koşuluyor (`kabul.py`).

*Not:* `esik1_olcum.txt`'teki **3,693 m** `smooth_alpha=1.0` ile ölçülmüştü; o değer sonradan
0,6'ya geri alındığı için sevk edilen reçetenin video1 değeri **3,88 m**'dir.

### 2) Üç uçuşlu kabul kapısı — `kabul.py`
`yarisma` (puanlı oturumun kendi kareleri, **VETO**) · `video1` (RGB) · `video2` (**termal**,
bağımsız uçuş, ilk kez genelleme kanıtı olarak kullanıldı). Sınav önbelleğe alındı:
**~56 dk → ~30 sn**, canlı yolla birebir doğrulandı.

### 3) ✅ Uygulanan kazanç: `min_inlier_ratio` 0,40 → 0,05 + eşik normalizasyonu
Kalite kapısı ölçümlerin büyük kısmını atıp sistemi **sabit-hız yedeğine** düşürüyordu
(merdiven çiftlerinin %30–61'i). Ayrıca yeniden-izdüşüm eşiği artık kameraya göre
normalize (0,5 **çalışma pikseli**; RGB 1080p'de 1,0 px = değişmedi, termalde 0,5, 4K'da 2,0).

| yatak | önce | sonra |
|---|---|---|
| **sınav** (canlı, uçtan uca) | 16,71 | **12,54** |
| **video2-TERMAL** | 15,24 | **10,49** (−%31) |
| video1-RGB | 3,87 | **3,60** |

*(Bu kazancın üstüne bir de merdiven GNC'si geldi — nihai tablo aşağıda.)*

### 4) ✅ İkinci kazanç: merdiven homografisinde GNC (`merdiven_gnc_c=2.0`)
Farklı sektörlerin robustluk araçları tarandı (robust optimizasyon, endüstriyel sinyal
işleme, kontrol mühendisliği, jeodezi, robust istatistik); **yalnız biri işe yaradı** —
ve ancak doğru katmana uygulanınca. Homografi inlier medyanı ardışık çiftlerde 0,86
(düzlem oturuyor), merdiven çiftlerinde 0,54 (bina/ağaç paralaksı). GNC her ikisine
uygulanınca video1'i bozdu; **yalnız merdivene** uygulanınca kazandırdı.

| yatak | önce | sonra |
|---|---|---|
| **sınav** | 10,92 ± 1,75 | **8,79 ± 0,74** (her iki pencere de kazandı) |
| video2-termal | 10,49 | **9,34** |
| video1-RGB | 3,60 | 3,73 (gürültü içinde) |

Teşhis edilen kök nedene doğrudan vuruyor: **ölçek hatası** video1 %2,55 → %1,55,
termal %6,33 → **%2,77**. Yayılım da her yatakta 2–3 kat düştü.

### 5) ❌ Ölçülüp elenenler
Sağlıklı→kesinti sapma kalibrasyonu (**çürütüldü**), IRLS/M-kestirimi, RANSAC topluluğu,
GNC (her iki katmana / MAD-uyarlamalı), **Hampel filtresi**, **anti-windup & sızıntılı
integratör**, ayrı yön kanalı, düzlem-içi taban, merdiven hizalaması (4 ve 1 serbestlikli),
termal CLAHE+FPN ön-işlemesi, uyarlanabilir ızgara, inlier-ağırlıklı harmanlama,
ds0,75, **keyframe aralığı 4/16** (spacing=8 korundu), loop closure.
Hepsinin sayıları ve gerekçeleri raporda.

> **Tekrarlanan ders:** bu sistemde *filtreleme / sönümleme / kısıtlama* eklemek neredeyse
> her seferinde zarar verdi (Hampel, anti-windup, IRLS, EMA, sürüklenme sönümleme).
> Kazandıran iki değişiklik de aynı yöndeydi: **daha çok ham ölçümü içeri almak**
> (kapı 0,40 → 0,05) ve **var olan ölçümün seçim belirsizliğini kaldırmak** (GNC).

> ⚠️ **Bu turun sonuçları YÖNTEM SINAVIYLA revize edildi.** Yukarıdaki `min_inlier=0,05`,
> normalize eşik ve merdiven GNC'si 2026 verisinde güçlüydü ama 2024/2025 verisinde
> aşırı uyum çıktı. Nihai reçete için aşağıya ve
> **[`degerlendirme/YONTEM_SINAVI.md`](degerlendirme/YONTEM_SINAVI.md)**'na bakın.

---

## Yöntem sınavı — başka yılların verisiyle bağımsız doğrulama (7 Ağu 2026)

**Tam rapor: [`degerlendirme/YONTEM_SINAVI.md`](degerlendirme/YONTEM_SINAVI.md)**

Turun bütün ayarları 2026 verisinde seçilmişti. 2024 ve 2025 örnek veri setleriyle
(5 bağımsız yatak; RGB **4K**, termal, ve 2B-GT'li 2024) yöntemin genellenip
genellenmediği sınandı.

### Bulunan asıl açık: sistemin çalışmadığı rejim
İki 2025 termal uçuşunda sistem **hiçbir şey yapmamaktan kötü** çıkıyordu
(53,5 m ve 101,7 m; `tut` tabanı 47,2 ve 53,1). Kök neden: kamera→dünya hizalamasının
GT'ye oturmadığı (residual medyanı **0,404**, video1'de 0,040) hâllerde eski
`residual_gate=0,50` bozuk hizalamayı kabul ediyordu.

### ✅ Uygulanan: `residual_gate` 0,50 → 0,30
Çalışan bütün yataklarda etkisi **tam olarak sıfır**; bozuk rejimde 53,5 → 27,0 ve
101,7 → 62,3. Bağımsız yataklarda ortalama **−13,19 m**.

### ⚠️ Kısmen geri alındı: `min_inlier_ratio` 0,05 → **0,30**
0,05 yıllar arası aşırı uyumdu (2024 +2,4 m). 0,30 sınav kazancını korur, bedeli yarılar.

### ❌ Geri alındı: merdiven GNC'si ve normalize eşik
GNC 2026'da güçlüydü ama 2025-RGB-4K-B'de **+8,4 m** felaket üretti. Normalize eşik
yıllar arası taşınmadı. İkisi de varsayılan kapalı, gerekçeleriyle kayıtlı.

### Nihai tablo

**5 bağımsız yatak (ayar orada yapılmadı):** ortalama **35,42 → 22,27 m**, 3/5 kazandı.

**2026 (6 permütasyon, eşleştirilmiş):**

| yatak | ESKİ | YENİ |
|---|---|---|
| **genelleme sınavı** | 16,68 ± 1,85 | **11,88 ± 1,17** (−4,80 ±0,95) |
| video1-RGB | 3,92 | 3,95 (eşit) |
| video2-termal | 15,38 | 15,05 (eşit) |

**Sevk edilen reçete:** RAFT + düzlemsel homografi + keyframe merdiveni (spacing=8),
`smooth_alpha=0.6`, `min_inlier_ratio=0.30`, `residual_gate=0.30`.

---

## Klasik eşleştiriciler (SIFT / ORB / AKAZE) — 8 Ağu 2026

**Tam rapor: [`degerlendirme/KLASIK_ESLESTIRICI.md`](degerlendirme/KLASIK_ESLESTIRICI.md)**

Daha önce yalnız *öğrenilmiş* seyrek eşleştiriciler denenmişti; klasik dedektörler kendi
boru hattımızda hiç sınanmamıştı. Yeni modül: [`src/klasik_eslestirici.py`](src/klasik_eslestirici.py)
(Lowe oran testi + karşılıklı en yakın komşu). Her eşleştirici **kendi eşik taramasıyla**.

**Birincil motor değişmedi — hiçbiri RAFT'ı geçemedi** (video1):
RAFT 3,77 · SIFT 4,48 · AKAZE 5,46 · **ORB 14,59** · SIFT tam çözünürlük 6,17.
SIFT'in yön hatası RAFT'tan iyi (1,30° vs 1,54°) ama ölçek hatası kötü — depoda
üçüncü kez görülen aynı ayrışma.

### ✅ Kazanç: GPU'suz yedek yolu 15,86 → 9,21 m

RAFT GPU ister; kurulamazsa sistem yedeğe düşüyordu. O yedek hem yanlış motordu
hem yanlış reçeteyle çalışıyordu. Üç yatakta ortalama:

| yedek | ortalama hata |
|---|---|
| LK, füzyon + merdiven kapalı *(eski)* | 15,86 |
| LK, homografi + merdiven | 12,36 |
| **SIFT, homografi + merdiven** *(yeni)* | **9,21** |

Motor zinciri artık **RAFT → SIFT → LK**; SIFT ~280 ms/kare (CPU), bütçenin çok altında.

---

## Yön, rejim ve motor soruları — 8 Ağu 2026

**Tam rapor: [`degerlendirme/YON_VE_REJIM.md`](degerlendirme/YON_VE_REJIM.md)**

Üç soru kapatıldı; üçünün de cevabı **hayır** çıktı, ama üçü de tekrar denenmesini
önleyecek sayı bıraktı. Ayrıca önceki turda yaratılan bir **regresyon** bulunup kapatıldı
(`evaluation._meas_from_row` eski CSV önbelleklerinde `KeyError` veriyordu; 5 aracı
kırıyordu — artık dataclass varsayılanlarından kuruluyor, gelecekteki her alan
kendiliğinden geriye uyumlu).

| soru | sonuç |
|---|---|
| **Yön benzerlik kestiriminden alınsın mı?** | ❌ video1 3,89 → **18,31 m**, yön hatası 1,69° → 9,91°. Adım başına fark yalnız 0,034° ama sistematik; 600 karede ~20°'ye birikiyor. **Homografinin dönmesi zaten doğru — zayıf halka yön değil.** |
| **Sistem "hiçbir şey yapmamaktan kötü" olduğunda `tut`'a düşülsün mü?** | ❌ Sinyal (hizalama red oranı 0,00–0,19 & yaş 1 **vs** 0,78 & yaş ~600) *hizalama güvenilirliğini* mükemmel ayırıyor ama *"tutmak daha iyi olur"u* ayırmıyor: aynı imzalı iki yataktan birinde −20 m, diğerinde +23 m. Net +1,7 m'ye karşı ~60 m yanlış-tetikleme riski. Anahtar kurulmadı. |
| **SIFT birincil motor olsun mu?** | ❌ 8 yatakta 2/8 kazanıyor, 4 felaket, ortalama **+3,00 m** kötü. Önceki 3 yataklık sinyal (SIFT 9,21 vs RAFT 9,72) **tek-yatak yanılsamasıymış**. CPU yedeği olarak kalıyor. |

**Turun çıkarımı:** üç hipotez de aynı yerde tökezledi — kalan hata yön kestiriminde ya
da rejim seçiminde değil, **ölçekte**, ve ölçek ölçümün kendi yanlılığından geliyor.
Literatür de bir kapı kapattı: monoküler metrik derinlik nadir bakışta zero-shot
%28–32 AbsRel ([AerialMetric, ECCV 2026](https://arxiv.org/html/2606.29716v1)) — mevcut
%2,5–6,3'lük ölçek hatamızın 5–10 katı.

**Sevk edilen reçete değişmedi.**
