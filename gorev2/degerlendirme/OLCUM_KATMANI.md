# Görev 2 — ölçüm katmanı turu: 10 yöntemin değerlendirmesi

**Bağlam.** PK (Görev 2) puanlı oturumda 55,29 / 8 takım içinde 8. — genel puanın **%40'ı**
(final formülü 8/8 takımda doğrulandı: `Final = (NT·0,25 + RNT·0,25 + PK·0,40) / 0,90`).
Önceki tur parametre düzeyinde tüketilmişti; teşhis **ölçüm katmanının yanlılığını**
işaret ediyordu (kesinti hatası rastgele değil sistematik: yön ~1,45°, ölçek %3,85).
Bu tur, verilen 10 yöntemi bu katmana uyguladı.

## Sonuç özeti

| Yöntem | Karar | Gerekçe |
|---|---|---|
| **6. Homografi eşiği** | ✅ **UYGULANDI** | `ransac_thresh` 2,0 → **1,0** |
| 6. MAGSAC++ / USAC_ACCURATE | ❌ | Her eşikte RANSAC'tan kötü |
| 4. XFeat | ❌ | Hareketi sistematik az ölçüyor |
| 1. SuperPoint+LightGlue (→ DISK+LightGlue) | ❌ | En iyi 6,32 m; 0/15 senaryoda daha iyi |
| 3. CoTracker / TAPIR | ❌ | Mimari uyumsuz + merdiven zaten aynı işi yapıyor |
| 7. Orthomosaic | ❌ | Fizibilite ölçümü: kapsama %30,6 (eşik %40) |
| 2. DPVO, 5. DROID-SLAM, 9. GTSAM, 10. VINS-Fusion | ❌ | Baştan elendi (aşağıda) |

## ✅ Uygulanan kazanç: yeniden-izdüşüm eşiği 2,0 → 1,0

Homografi/benzerlik kestiriminde RANSAC eşiğinin sıkılaştırılması, gevşek eşleşmeleri
eleyerek **sistematik yanlılığı** düşürdü — aradığımız düzeltme buydu:

| Ölçüt | Önce (2,0) | Sonra (1,0) |
|---|---|---|
| **Ortalama Öklid hatası (yarışma takvimi)** | 4,343 m | **3,693 m (−%15)** |
| 15 uzun-kesinti senaryosu (ortalama) | 7,628 m | **6,439 m (−%15,6)**, 11/15'te daha iyi |
| Yön hatası (Procrustes) | 1,45° | **1,27°** |
| Ölçek hatası (Procrustes) | %3,85 | **%2,11** |
| Blok 1 ortalama / bitiş hatası | 9,17 / 12,33 m | **6,70 / 7,39 m** |

*Not:* Genel (karışık uzunlukta) rastgele takvimlerde küçük bir kayıp var (+%3,6). Kabul
gerekçesi: yarışmanın **yapısal** koşulu uzun kesintidir (449/499/599 kare — puanlı oturum
logundan), bu koşulda 15 bağımsız senaryoda net kazanç var ve fiziksel açıklaması nettir.

**Toplam ilerleme (iki tur):** 4,535 m → **3,693 m (−%18,6)**.

## ❌ Elenenler ve gerekçeleri

**Baştan elenenler (donanım/mimari):**
- **VINS-Fusion** — IMU gerektirir; sistemde yalnız monoküler RGB var.
- **DROID-SLAM** — offline bundle adjustment; rolling akışta gönderilen tahmin geri
  alınamadığı için geriye dönük düzeltme puana yansımaz. Ayrıca 8,5 GB VRAM'de sınırda.
- **DPVO** — CUDA eklenti derlemesi gerektirir; monoküler ölçek belirsizliği GT-kalibrasyonlu
  mimarimizle çakışır.
- **GTSAM Factor Graph** — füzyon için ikinci sensör yok; graf optimizasyonu gürültüyü
  azaltır ama **sistematik yanlılığı düzeltmez** (teşhisimiz tam olarak yanlılıktı).
- **TartanVO/DFormer** — monoküler ölçek belirsiz; yedek olarak bırakıldı, sıra gelmedi.

**Denenip elenenler (ölçümle):**

**MAGSAC++ / USAC_ACCURATE** — 9 kombinasyon tarandı; hiçbiri kazanmadı
(MAGSAC++ eşik 1,0'da 8,62 m / yön 6,32° ile felaket). MAGSAC++'ın esnek eşik yaklaşımı,
bu düzlemsel-nadir sahnede RANSAC'ın sabit sıkı eşiğinden daha kötü.

**XFeat** (yöntem 4) — kurulup çalıştırıldı, önbellek 8× hızlı üretildi (88 s vs 721 s,
2700 nokta/çift). Ama sonuç felaket (160 m, yön 130°). Kök neden bulundu: **hareketi
sistematik olarak az ölçüyor** — aynı kare çiftinde medyan akış RAFT (+2,53, +25,21) iken
XFeat (+0,02, +16,71); eşik sıkılaştıkça daha da düşüyor (16,7 → 10,3 → 2,4). Yüksek
akışlı bölgelerde eşleşme kuramıyor → ölçek yanlılığı. Havadan tarım/arazi görüntüsündeki
tekrarlayan dokuda seyrek özellik eşleştirmenin bilinen zaafı.

**DISK + LightGlue** (yöntem 1 muadili) — *SuperPoint çıkarıcısı kornia'da yok, `lightglue`
paketi PyPI'da yayınlanmamış; LightGlue'nun resmi partner çifti DISK kullanıldı.*
En iyi 6,32 m (RAFT 3,69 m), **0/15** senaryoda daha iyi. XFeat ile aynı kök neden.

**CoTracker / TAPIR** (yöntem 3) — yüklendi; `cotracker3_online` 8 karelik video penceresi
bekliyor (`step=8`), ikili `track_pair` arayüzüne durumlu sarmalayıcı gerekir. Üstelik
**keyframe merdivenimiz (spacing=8) zaten uzun-menzilli keyframe→kare eşleştirmesi yapıyor**
ve nadir hava görüntüsünde CoTracker'ın ana avantajı (occlusion dayanıklılığı) devrede değil.

**Orthomosaic** (yöntem 7) — fizibilite ölçüldü: uçuş yüksekliği medyan 55 m → kamera yer
izdüşümü 76×43 m (yarıçap ~22 m). Kesinti karelerinin yalnızca **%30,6'sı** bir sağlıklı
kareye bu mesafede (kabul eşiği %40). Mozaik kesinti karelerinin çoğunu kapsayamaz —
loop closure turundaki bulguyla aynı kök neden (araç yeni alanlara gidiyor).

## Ek deneme: RAFT çalışma çözünürlüğü (ds 0,5 → 0,75)

RAFT görüntüyü 1920 → 960 px'e küçültüp çalışıyor (`auto_scale`). Daha yüksek çözünürlüğün
(1440 px) akış hassasiyetini artırıp yanlılığı azaltması beklenirdi. Kendi eşik taramasıyla
denendi (piksel ölçeği değiştiği için eşik de tarandı):

| Ölçek | Eşik | Yarışma | 15 uzun-kesinti | Yön | Ölçek |
|---|---|---|---|---|---|
| **0,50 (mevcut)** | **1,0** | **3,693 m** | **6,439 m** | 1,27° | **%2,11** |
| 0,75 | 0,75 | 3,727 m | 7,917 m | **0,65°** | %2,53 |
| 0,75 | 1,0 | 3,722 m | 6,606 m | 1,20° | %2,49 |
| 0,75 | 1,5 / 2,0 | 4,42 / 4,28 m | 8,13 / 8,06 m | ~1,1° | %3,5–3,9 |

**Reddedildi.** Hiçbir varyant mevcut ayarı geçemedi (en iyisi 8/15 senaryoda iyi ama
ortalamada −1,48 m kötü). İlginç bulgu: yüksek çözünürlük **yön** doğruluğunu belirgin
artırıyor (1,27° → 0,65°) ama **ölçek** hatasını kötüleştiriyor — ölçek yanlılığının
kaynağı akış çözünürlüğü değil, muhtemelen düzlemsellik varsayımı ve yükseklik durumu.
Ayrıca maliyet 2,5× (önbellek üretimi 721 s → 1781 s), gerçek zamanlı bütçeyi zorlar.

*Sonraki tur için ipucu:* yön ve ölçek hataları **farklı kaynaklardan** geliyor; ölçek
tarafı için homografi düzlem-normali / yükseklik gözlemi hattı incelenmeli.

## Neden RAFT üstün kaldı

Ölçülen fark seyrek vs yoğun eşleştirmede: RAFT **yoğun optik akış + ileri-geri (cycle)
tutarlılık elemesi** yapıyor; tekrarlayan dokuda yanlış eşleşmeleri baştan eliyor.
Seyrek yöntemler (XFeat, DISK) ayırt edici köşe bulamadığı bölgelerde ya eşleşme
kuramıyor ya da yanlış kuruyor — ikisi de sistematik yanlılık üretiyor.

## Altyapı kazanımı: nokta önbelleği

`nokta_onbellek.py` — tracker'ın ham eşleşmeleri (`pts0`,`pts1`) bir kez `.npz`'ye yazılır;
geometri denemeleri **video yeniden taranmadan saniyeler içinde** koşar (25 dk → 3 sn).
Doğrulama: önbellekten üretilen ölçümler eski CSV ile **birebir aynı** (maks fark 0,000000).
Bu altyapı olmasaydı 9+7+5 = 21 varyantlık tarama pratik olmazdı.

## Dosyalar
- `nokta_onbellek.py` — eşleşme önbelleği + `olcum_uret()`
- `geometri_tarama.py` — robust kestirici / eşik taraması (+ Procrustes teşhisi)
- `src/derin_eslestirici.py` — XFeatTracker, DiskLightGlueTracker (`MatchResult` arayüzü)
- `test_yatagi.py` — resmî metrik + yarışmanın gerçek kesinti takvimi
