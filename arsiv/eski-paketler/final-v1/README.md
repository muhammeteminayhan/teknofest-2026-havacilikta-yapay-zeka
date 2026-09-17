# Görev 2 — Pozisyon Kestirimi · **final-v1**

TEKNOFEST 2026 Havacılıkta Yapay Zekâ · İkinci Görev (genel puanın **%40'ı**).

GPS kesintisinde, yalnız aşağı bakan kamera görüntüsünden hava aracının video
başlangıcına göre x/y/z yer değiştirmesini kestirir.

Bu klasör **yarışma günü çalışacak kodun tamamıdır** — geliştirme deposundaki
deneme/ölçüm altyapısı ve reddedilen 14 ayar dâhil değildir. Temizlemenin
davranışı değiştirmediği [`denklik_testi.py`](denklik_testi.py) ile **bit‑birebir**
kanıtlanmıştır.

---

## Hızlı kullanım

```python
from konum import LivePositionEstimator, on_ucus_kontrolu

print(on_ucus_kontrolu())          # oturumdan önce
est = LivePositionEstimator(engine="auto")

for image, gt, healthy in sunucu_akisi:
    x, y, z = est.process_frame(image, gt, healthy)
```

* `healthy=True` → sunucunun gönderdiği konum **aynen** geri döner (hata 0).
* `healthy=False` → görsel ölçümle dead reckoning.
* Kamera profili ilk karenin boyutundan **otomatik** seçilir.

---

## Sevk edilen reçete

| bileşen | değer | neden |
|---|---|---|
| eşleştirme | **RAFT‑large** yoğun optik akış (GPU) | yoğun akış + ileri‑geri eleme, tekrarlayan havadan dokuda seyrek yöntemleri yener |
| çalışma ölçeği | otomatik (~1000 px): termal 1,0 · 1080p 0,5 · 4K 0,25 | ds 0,75 denendi: yönü iyileştirip **ölçeği bozuyor**, maliyeti 2,5× |
| hareket modeli | **düzlemsel homografi** + ayrıştırma | nadir bakışta benzerlikten doğru; eğik bakışta da geçerli |
| konum | **keyframe merdiveni**, `spacing=8` | hata yalnız keyframe atlamalarında birikir; 4 ve 16 üç yatakta denendi, 8 en iyi |
| `min_inlier_ratio` | **0,30** | 0,40 ölçümlerin çoğunu atıp sabit‑hıza düşürüyordu; 0,05 ise yıllar arası aşırı uyum |
| `residual_gate` | **0,30** | bütünlük denetimi: bozuk hizalamayı reddeder (aşağıya bakın) |
| `smooth_alpha` | **0,6** | 1,0 ve 0,4 üç yatakta denendi, ikisi de kötü |
| CPU yedeği | **SIFT** + homografi + merdiven | GPU yoksa kayıp neredeyse sıfır (aşağıya bakın) |

### Motor zinciri: RAFT → SIFT → LK

| motor | koşul | reçete | hız |
|---|---|---|---|
| **raft** | GPU + yerel ağırlık | homografi + merdiven | ~620 ms/kare |
| **sift** | RAFT kurulamazsa | homografi + merdiven | ~420 ms/kare |
| **lk** | SIFT de olmazsa / açık istek | füzyon, merdivensiz | ~50 ms/kare |

CPU yedeği neden SIFT (eskiden LK) — üç yatakta ortalama hata:
LK füzyon+merdivensiz **15,86 m** · LK homografi+merdiven 12,36 m ·
**SIFT homografi+merdiven 9,21 m** · (kıyas) RAFT/GPU 9,72 m.

---

## Ölçülen başarım

**Canlı, uçtan uca** (önbelleksiz, gerçek `LivePositionEstimator`;
gerçek kesinti takvimi 703 sağlıklı / 1547 kesinti):

| koşu | tur başında | **şimdi** |
|---|---|---|
| **Puanlı oturumun kendi kareleri** (genelleme sınavı) | 16,71 m | **8,53 m** |
| video1‑RGB (test oturumu uçuşu) | 3,88 m | **3,90 m** |
| video2‑TERMAL | 14,73 m | **10,35 m** |

**Bağımsız yıllar** (2024 + 2025, 5 uçuş; ayar bu yataklarda yapılmadı):
ortalama **7,58 → 6,52 m**. Yedi yatağın tamamında **8,12 → 6,69 m**.

> **İki düzeltme (13 Ağu 2026).**
> 1. Bu tablonun eski satırındaki *"bağımsız yıllar 35,42 → 22,27 m"* rakamı
>    **geçersizdi**: 2025 termal CSV'leri veri setinde çapraz adlandırılmış ve o iki
>    yatak anlamsız ölçüyordu (`degerlendirme/TANI_TURU.md`). Düzeltilmiş taban
>    5 yatakta 8,39 m'dir; yukarıdaki rakamlar düzeltilmiş yataklardandır.
> 2. Genelleme sınavının son kazancı `z_sign` kanıt kapısından gelir
>    (`degerlendirme/Z_ISARETI.md`): pencere 1 **17,26 → 11,46 m**, pencere 2
>    değişmedi. Aynı düzeltme video2, v25_rgb_b ve v25_termal_b yataklarını da
>    kazandırdı; **hiçbir yatakta kötüleşme yok**.
>
> Bütün rakamlar 3 permütasyonun ortalamasıdır. RANSAC nokta sırasına duyarlı
> olduğu için tek koşu ±%18,7 gürültü taşır; aynı yapılandırmanın tek-koşu
> ölçümleri 11,08–11,43 m arasında değişebiliyordu.

**Kıyas tabanları** (görsel bilgi kullanmayan): *tut* 63,6 m · *sabit hız* 160,7 m
→ sistem bunların **17×** altında.

**Süre:** ~620 ms/kare → 2250 karelik oturum ≈ **24 dk** (pencere 90 dk).

### Bu paketin kendi provaları (`prova.py`, 2250 kare, gerçek takvim)

| video | profil | resmî metrik | kesintide | `tut` tabanı | hız |
|---|---|---|---|---|---|
| `THYZ_2026_Ornek_Veri_1.MP4` | rgb_1080p | **3,770 m** | 5,483 m | 63,65 m (16,9×) | 614 ms · 23,3 dk |
| `THYZ_2026_Ornek_Veri_2_Termal.MP4` | termal | **14,302 m** | 20,801 m | 78,85 m (5,5×) | 415 ms · 15,6 dk |

Sağlıklı karelerde hata **0,0000 m** (sunucu değeri aynen geri gönderiliyor).
Bu sayılar geliştirme deposundaki referans uygulamayla **bit‑birebir** aynıdır
(`denklik_testi.py`: 3.769530 = 3.769530 · 14.301547 = 14.301547).

---

## Yarışma günü kontrol listesi

```bash
python on_ucus.py                                   # ortam + motor zinciri
python on_ucus.py --video ORNEK.MP4                 # üstüne gerçek karelerle hız
python prova.py --video ORNEK.MP4 --gt ORNEK.csv    # GT'li tam prova
```

1. **`RAFT agirligi yerel : EVET` olmalı.** Yarışma bilgisayarının internete
   çıkması yasaktır; ağırlık yoksa RAFT kurulamaz ve sistem SIFT'e düşer.
   Çözüm: internete bağlıyken bir kez `RaftTracker(...)` oluşturun.
2. **`cuda : True`** — yoksa SIFT yedeği devreye girer (kabul edilebilir).
3. **Kare boyutu tanınmalı.** Tanınan profiller: termal 640×512 · RGB 1080p ·
   RGB 4K 4000×3000 · RGB 4K video 3840×2160. Başka bir boyut gelirse
   `profile_for_size` **hata verir** — o durumda kalibrasyon dosyasından yeni bir
   `CameraProfile` eklenmelidir.
4. **Süre bütçesi**: `on_ucus.py` 2250 karelik oturum tahminini yazar.

---

## Dosyalar

```
konum/
  calibration.py      kamera profilleri (4 profil) + distorsiyon düzeltme
  geometry.py         benzerlik (4 SD) ve homografi (8 SD) hareket kestirimi
  scale_alignment.py  sağlıklı GT'den eksen hizalaması M + metrik ölçek + z işareti
  matching.py         MatchResult + LK takipçisi (son çare)
  raft_matching.py    RAFT takipçisi (birincil)
  sift_matching.py    SIFT takipçisi (CPU yedeği)
  vo_pipeline.py      PairMeasurement + durum makinesi (merdivenli)
  live_estimator.py   LivePositionEstimator + ön‑uçuş kontrolü
on_ucus.py            yarışma günü kontrol listesi
prova.py              GT'li uçtan uca prova (resmî metrik + kıyas tabanları)
denklik_testi.py      temiz paket ↔ geliştirme deposu bit‑birebirlik kanıtı
```

---

## Bilinen sınırlar (dürüst kayıt)

1. **Kalan hata ölçektedir.** Yön hatası ~1,5°, ölçek hatası RGB %2,5 / termal %6,3.
   Ölçek, `dxy = M·e0·height` bağıntısındaki `height` durumundan gelir; yükseklik
   kesintide çarpımsal olarak taşınır ve **mutlak bir gözlemi yoktur**.
2. **Son‑işleme çare değil.** Filtreleme, sönümleme, kısıtlama, robust kestirim,
   model seçimi ve rejim anahtarı ayrı ayrı ölçüldü; **hepsi zarar verdi ya da
   etkisiz kaldı**. Kazandıran iki değişiklik de ters yöndeydi: daha çok ham
   ölçümü içeri almak (`min_inlier_ratio` 0,40 → 0,30) ve güvenilmez kalibrasyonu
   reddetmek (`residual_gate`).
3. **Yavaş/alçak uçuşta zayıf.** 2025 termal uçuşlarından birinde sistem hâlâ
   "tut" tabanının gerisinde. `residual_gate` zararı büyük ölçüde kapattı
   (101,7 → 62,3 m) ama tamamen çözmedi. Bunu çalışma anında tespit edip
   "tut"a düşme fikri ölçüldü: sinyal *hizalamanın güvenilmezliğini* ayırıyor ama
   *tutmanın daha iyi olacağını* ayırmıyor (aynı imzalı iki uçuştan birinde
   −20 m, diğerinde +23 m) → anahtar **kurulmadı**.
4. **Monoküler metrik derinlik bir çözüm değil.** Nadir bakışta zero‑shot
   %28–32 AbsRel (AerialMetric, ECCV 2026) — mevcut ölçek hatamızın 5–10 katı.
5. **Süre bütçesi kodda zorlanmıyor.** Ölçülüyor (`timing_stats`) ama yavaşlama
   hâlinde otomatik bir hafifletme yolu yok.
