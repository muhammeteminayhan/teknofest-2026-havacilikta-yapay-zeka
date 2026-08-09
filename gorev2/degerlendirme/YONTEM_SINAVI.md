# Görev 2 — YÖNTEM SINAVI: başka yılların verisiyle bağımsız doğrulama

> **Bu belge bir geri alma kaydıdır.** Ölçüm turu 2'nin iki "kazancı" 2026 verisinin
> üç yatağında güçlü görünüyordu; 2024 ve 2025 veri setlerinde sınanınca **aşırı uyum**
> oldukları ortaya çıktı ve biri tamamen, biri kısmen geri alındı. Buna karşılık,
> yalnız başka yılların verisi sayesinde görülebilen **çok daha büyük bir gürbüzlük
> açığı** bulundu ve kapatıldı.

## Neden bu sınav yapıldı

Turun bütün ayarları 2026 verisinde seçilmişti. Üç yatak vardı ama ikisi aynı yılın
aynı kamera/arazi koşullarından geliyordu. Sorulması gereken soru şuydu:

> **Yöntem mi çalışıyor, yoksa 2026 verisine mi uyduk?**

## Yataklar

| yatak | veri | modalite | notlar |
|---|---|---|---|
| `v25_rgb_a` / `v25_rgb_b` | 2025 örnek veri | **RGB 4K (3840×2160)** | 2026'da RGB 1080p'ydi |
| `v25_termal_a` / `v25_termal_b` | 2025 örnek veri | termal 640×512 | |
| `v24` | 2024 örnek veri | RGB 1080p | GT'de **z yok** → hata 2B; kalibrasyon dosyası yok |

**Kalibrasyon.** 2025 kalibrasyon dosyası 2026'nınkiyle **birebir aynı** (RGB 4K ve termal
iç parametreleri özdeş) — donanım değişmemiş. Tek fark: 4K videolar **3840×2160** geliyor,
kalibrasyon ise 4000×3000 için verilmiş. Tutarlı tek açıklama 4:3 sensörden 16:9 video:
dikeyde 3000→2250 kırpma, sonra 4000→3840 ölçekleme (s=0,96). Türetilen profil
`calibration.RGB_4K_VIDEO`. Tutarlılık kontrolü: özgün asal nokta sensör merkezinden
62,2 px aşağıda, türetilen de video merkezinden 59,7 px aşağıda — aynı fiziksel kayma.

**Uçuş eşleşmesi tuzağı.** "Veri-1" RGB ile "Veri-1" Termal **aynı uçuş değil**:
- A uçuşu = `Ornek-Veri-1-RGB` + `Ornek-Veri-2-Termal` (yol 710 m, net 136 m, z aralığı ~1 m)
- B uçuşu = `Ornek-Veri-2-RGB` + `Ornek-Veri-1-Termal` (yol 1238 m, net 20 m, z aralığı 28 m)

Her video kendi adıyla eşleşen CSV'yi kullanır. Yatakların doğru kurulduğu, **tüm kareler
sağlıklı** koşturulduğunda hatanın tam **0,0000 m** çıkmasıyla doğrulandı (GT↔video senkron).

**2024 verisi neden zayıf yatak:** GT'de yalnız `translation_x, translation_y` var (z yok),
kalibrasyon dosyası yok, video yalnız 3322 kare (7,5 FPS'te 830 örnek). Kalibrasyon
varsayımının sonucu etkilemediği ölçüldü: merdiven inlier medyanı 2026-1080p profiliyle de,
distorsiyonsuz profille de, f=1000 ile de **0,288** — yani düşük inlier sahnenin kendisinden.

## En büyük bulgu: sistemin çalışmadığı rejim

Kıyas tabanlarıyla ölçüldüğünde iki 2025 termal uçuşunda sistem **hiçbir şey
yapmamaktan kötü**:

| yatak | `tut` tabanı | `sabit_hız` | ESKİ reçete | oran |
|---|---|---|---|---|
| v25_termal_a | 47,2 | 42,4 | **53,5** | 0,9× |
| v25_termal_b | 53,1 | 87,3 | **101,7** | 0,5× |
| *(kıyas)* video1 | 63,6 | 161,0 | 3,99 | 15,9× |

Kök neden — **hizalama bütünlüğü**. Kamera→dünya köprüsünün (`Alignment`) sağlıklı
dönemde GT'ye ne kadar oturduğunu ölçen `residual_ratio`:

| yatak | residual medyan | öğrenilen ölçek oynaklığı | kesinti hatası |
|---|---|---|---|
| video1 | 0,040 | %11 | 3,4 m |
| video2 | 0,115 | %17 | 7,3 m |
| **v25_termal_a** | **0,404** | **%55** | **53 m** |

Eski `residual_gate = 0,50` bu bozuk hizalamayı kabul ediyor, sistem onunla ölü hesap
yapıyordu. 2026 yataklarında residual 0,04–0,12 olduğu için kapı hiç tetiklenmiyor ve
"ölü ayar" görünüyordu — **asıl işlevi ancak başka yılların verisinde ortaya çıktı.**

## ✅ Uygulanan: `residual_gate` 0,50 → 0,30

Sıkı kapı bozuk hizalamayı reddeder, önceki geçerli hizalama korunur.

| değer | v25_termal_a | v24 | video1 | video2 | sınav |
|---|---|---|---|---|---|
| 0,50 | 51,33 | 7,28 | 3,47 | 9,77 | 7,22 |
| **0,35 / 0,25** | **28,23** | 7,28 | 3,47 | 9,77 | 7,22 |
| 0,15 | 28,16 | 7,28 | 3,47 | 10,79 ✗ | 7,22 |
| 0,10 | 42,66 ✗ | 7,38 | 3,47 | 10,15 | 7,22 |

**Çalışan bütün yataklarda etkisi tam olarak sıfır** (fark 0,000). 0,30 seçildi:
video2'nin p90'ı (0,247) ile v25'in medyanı (0,404) arasında güvenli orta nokta.

Bağımsız yataklardaki toplam etkisi: **−13,19 m ortalama**.

## ⚠️ Kısmen geri alındı: `min_inlier_ratio` 0,05 → 0,30

2026 verisi 0,05'i işaret ediyordu; yıllar arası sınavda 0,05'in aşırı uyum olduğu çıktı:

| değer | 2024 | 2025-RGB-A | 2026 sınavı |
|---|---|---|---|
| 0,05 | **+2,4** | **+0,5** | −6,3 |
| **0,30** | +1,2 | **−0,7** | −5,3 |

0,30 sınav kazancının neredeyse tamamını koruyup yıllar arası bedeli yarıya indiriyor.

## ❌ Tamamen geri alındı: merdiven GNC'si

2026'da güçlüydü (sınav 10,92 → 8,79, her iki pencerede de kazanç; ölçek hatası
video1 %2,55→%1,55, termal %6,33→%2,77). Ama 5 bağımsız yatakta **genellemedi** ve
bir yatakta felaket üretti:

| yatak | ESKİ | GNC tek başına | tam reçetede |
|---|---|---|---|
| **v25_rgb_b (4K)** | 12,80 | **16,83 (+4,0)** | **21,19 (+8,4)** |
| v25_rgb_a | 4,18 | 4,28 | 4,60 |
| v24 | 4,90 | 4,51 | 5,70 |
| video1 | 3,90 | 4,28 | 3,75 |
| video2 | 14,94 | 14,58 | 9,76 |
| ortalama | — | **+0,75** | +0,85 |

Bilinmeyen bir final uçuşunda böyle bir kuyruk riski kabul edilemez. Kod ve gerekçe
`geometry.gnc_homografi` içinde kayıt olarak duruyor; varsayılan **kapalı**.

## ❌ Geri alındı: kameraya göre normalize eşik

0,5 çalışma pikseli kuralı 2026'da iki modalitede doğrulanmıştı, ama yıllar arası
taşınmadı: termalde ancak çok gevşek kapıyla (0,05) kazandırıyor, 0,30'da kaybettiriyor.
Varsayılan tarihsel davranışa döndü (her kamerada 1,0 px). `calisma_esigi` parametresi
ve `calibration.ransac_esigi` açıkça çağrılabilir hâlde duruyor.

## Nihai tablo — 5 bağımsız yatak (ayar bu yataklarda yapılmadı)

| yatak | ESKİ | +residual_gate | **YENİ** | fark |
|---|---|---|---|---|
| 2025 RGB-4K-A | 4,05 | 4,05 | **3,37** | **−0,68** ✓ |
| 2025 RGB-4K-B | 12,86 | 12,86 | 13,37 | +0,50 (eşit) |
| 2025 termal-A | 53,55 | 27,03 | **27,16** | **−26,40** ✓ |
| 2025 termal-B | 101,71 | 62,26 | **61,40** | **−40,31** ✓ |
| 2024 (2B) | 4,90 | 4,90 | 6,08 | +1,17 ✗ |
| **ortalama** | **35,42** | 22,22 | **22,27** | **−13,14** |

3/5 kazandı, 1 eşit, 1 kaybetti (en zayıf yatak).

## 2026 doğrulaması ve canlı uçtan uca koşu

**Önbellekli, 6 permütasyon, eşleştirilmiş:**

| yatak | ESKİ | YENİ | fark |
|---|---|---|---|
| **genelleme sınavı** | 16,677 ± 1,850 | **11,879 ± 1,173** | **−4,798 ±0,947** |
| video1-RGB | 3,917 ± 0,076 | 3,950 ± 0,128 | +0,033 (eşit) |
| video2-termal | 15,380 ± 0,834 | 15,052 ± 0,538 | −0,328 (eşit) |

**Canlı (önbelleksiz, gerçek `LivePositionEstimator`):**

| koşu | ESKİ | YENİ |
|---|---|---|
| `genelleme_sinavi.py` | **16,71** (p1 16,61 · p2 16,80) | **11,08** (p1 17,31 · p2 4,85) |
| `canli_dogrulama.py` (video1) | 3,88 | **3,760** |

Hız: **643 ms/kare**, p95 661 ms, 2256 kare = **24,5 dk** (90 dk penceresi).

## Çıkarılan ders

Tur boyunca üç kez aynı şey oldu: **tek bir veri ailesinde güçlü görünen bir kazanç,
bağımsız veride çöktü** (`z_sim_agirlik` geçen tur, `min_inlier=0,05` ve GNC bu tur).
Buna karşılık, bütün yataklarda **tam olarak nötr** olan tek değişiklik
(`residual_gate`) en büyük gerçek kazancı verdi — çünkü bir ayar değil, bir
**bütünlük denetimi**: yalnız sistem kendi kalibrasyonuna güvenemediğinde devreye giriyor.

Bilinmeyen bir final uçuşu için doğru öncelik budur: ortalamayı biraz daha düşüren
ayarlar değil, **felaket senaryosunu kapatan** korumalar.

## Yeniden koşturmak için

```
.venv\Scripts\python gorev2\yontem_sinavi.py --n-perm 5
```

Önbellekler: `outputs/nokta_raft_v25{rgba,rgbb,terma,termb}_{cons,sp8}.npz`,
`outputs/nokta_raft_v24_{cons,sp8}.npz`.
