# Görev 2 — düşey kanal denemesi: **ÖLÇÜLDÜ, YANLIŞLANDI, GERİ ALINDI**

> **Durum (29 Tem 2026): bu yöntem sistemden kaldırıldı.**
> `EstimatorConfig.z_sim_agirlik`, `_ladder_candidate` içindeki harmanlama, `final_ab.py`
> ve buna ait ölçüm çıktıları silindi. Bu dosya *neden denendiğini* ve *neden geri
> alındığını* kayda geçirmek için duruyor — aynı yola tekrar girilmesin diye.

---

## 1. Fikir ve ilk ölçüm (27–28 Tem)

Kesinti hatasının eksen bazında dökümü, **örnek veri setinde** en büyük bileşenin düşey
eksen olduğunu gösteriyordu: x 3,14 · y 2,85 · **z 4,47 m**.

Düşey hareket, homografi ayrıştırmasının `n · delta` bileşeninden geliyordu; monoküler
düzlemsel homografide bu bileşen yatayın yanında zayıftır. Buna karşılık **benzerlik
dönüşümünün ölçeği, nadir bakan kamerada doğrudan yükseklik oranını verir** ve
`PairMeasurement.from_points` bunu zaten her karede hesaplıyordu — bedava, bağımsız
ikinci bir kanal.

`z_sim_agirlik=1.0` ile ölçülen (kaynak: `THYZ_2026_Ornek_Veri_1.MP4`, yarışmanın gerçek
kesinti takvimi):

| ölçüt | eski | z_sim_agirlik=1.0 |
|---|---|---|
| yarışma takvimi | 3,767 m | **2,803 m (−%25,6)** |
| 15 uzun-kesinti ort. | 6,562 m | 5,785 m (−%11,8) |
| z RMS (kesinti) | 4,47 | 2,45 |
| ölçek hatası (Procrustes) | %2,06 | %1,52 |

Uçtan uca canlı doğrulama da aynı kazancı verdi (2,795 m). Bu tabloya bakınca karar
"uygula" idi ve uygulandı.

---

## 2. Genelleme sınavı (29 Tem) — kazanç **yok**, üstelik zarar

Yukarıdaki bütün ölçümler **tek bir uçuş** üzerinde yapılmıştı:
`THYZ_2026_Ornek_Veri_1.MP4`. Bu video, çevrim içi yarışmanın **test oturumunun ta
kendisidir** — iki kaynağın GT'lerinin ilk satırı birebir aynı
(`0,04443544… · 0,00306853… · −0,00023451…`). **Puanlı oturum ise başka bir uçuştur.**

"Rastgele kesinti takvimleri" ile yapılan aşırı-uyum kontrolü bu riski yakalayamazdı:
takvim çeşitleniyordu ama **uçuş çeşitlenmiyordu**.

### Sınav yöntemi

Puanlı oturumda kesinti karelerinin GT'si sunucu tarafından hiç verilmedi; puan metriği
orada doğrudan ölçülemiyor. Çözüm: **sağlıklı bir bloğu yapay olarak kesintiye
çevirmek.** Sunucu o karelerin GT'sini vermişti, biz sisteme vermiyoruz. Böylece gerçek
yarışma görüntülerinde, gerçekçi kesinti uzunluğunda, her kare için gerçek hata
ölçülebiliyor.

| pencere | beslenen sağlıklı | puanlanan | kesinti uzunluğu |
|---|---|---|---|
| 1 | kare 0–450 | **900–1050** (151 kare) | 449–599 kare |
| 2 | kare 0–450 ve 900–1050 | **1550–1650** (101 kare) | ~500–600 kare |

### Sonuç — ortalama Öklid hatası (puan metriğinin kendisi)

| yapılandırma | pencere 1 | pencere 2 | ortalama |
|---|---|---|---|
| α=1,0 · **z=0** | **16,04 m** | 18,46 m | 17,25 m |
| α=1,0 · z=1 | 20,07 m | 18,62 m | 19,35 m |
| α=0,6 · **z=0** | 16,61 m | **16,80 m** | **16,71 m** |
| α=0,6 · z=1 | 20,59 m | 16,95 m | 18,77 m |

`z=1` **dört karşılaştırmanın dördünde de kaybetti.** Zarar tam olarak değişikliğin etki
ettiği eksende yoğunlaşıyor:

| pencere 1, eksen bazında ort. \|hata\| | x | y | z |
|---|---|---|---|
| z=0 | 5,39 | 7,92 | **10,25** |
| z=1 | 5,94 | 5,83 | **16,91** |

> Not: bu mutlak değerler (16–20 m) örnek veri setindeki 2,8 m ile kıyaslanamaz — burada
> puanlanan pencere kesintinin **en kötü ucu** (449–599. karesi), örnek veri setindeki
> ortalama ise kesintinin tamamını (başındaki düşük hatalı kısım dahil) kapsıyor.
> Karşılaştırma yalnız yapılandırmalar arasında geçerlidir.

### Neden

Benzerlik ölçeği ↔ yükseklik ilişkisi, sahnenin **düzlemsel ve görüş eksenine dik**
olmasına dayanır. Örnek veri setinin uçuşunda bu varsayım iyi tutuyor; puanlı oturumun
uçuşunda (farklı irtifa profili, daha katmanlı sahne) tutmuyor — ölçek değişimi yükseklik
yerine sahne derinliğini okuyor. Homografi ayrıştırması bu koşulda daha gürbüz.

---

## 3. Bu turdan kalıcı olarak alınan dersler

1. **Tek uçuşta ölçülen kazanç, kazanç değildir.** Aşırı uyum kontrolü değişkenin
   *kendisini* çeşitlendirmelidir — takvimi değil, uçuşu.
2. **Genelleme sınavı artık zorunlu bir kapı.** Herhangi bir Görev 2 değişikliği, puanlı
   oturumun kendi karelerindeki iki pencerede de kazanmadan uygulanmaz.
   Araç: [`gorev2/genelleme_sinavi.py`](../genelleme_sinavi.py).
3. **Hangi eksenin baskın olduğu uçuşa göre değişiyor.** Pencere 1'de hata düşeyde
   (z 10,25 m), pencere 2'de neredeyse tamamen yatayda (z 0,13 m; y 15,65 m). Tek bir
   eksene özel çözüm aramak yanlış zemin.

---

## 4. Bu turdan geçerliliğini koruyan bulgular

- **`smooth_alpha=1.0`** (ölçüm EMA'sı kapalı) — ⛔ **bu da geri alındı (29 Tem).**
  Örnek veri setinde net kazançlıydı (4,535 → 4,343 m yarışma takvimi; 5,280 → 4,186 m
  rastgele takvimler; en büyük hata 15,04 → 12,42 m). Ama o kanıt da aynı tek uçuştan
  geliyor. Genelleme sınavında pencereler ters yönde ve **ortalamada 0,6 önde**
  (16,71 vs 17,25 m). Teslim, puanlı oturumda koşan değere (`0.6`, yani varsayılan)
  döndürüldü.
- **`spacing=8`.** Bir ara spacing=16 kazanır göründü (3,584 m); z kanalı devredeyken
  avantajı kayboldu ve 8 yeniden en iyi oldu. z kanalı da kalktığına göre spacing'in
  genelleme sınavında yeniden ölçülmesi gerekir — **açık iş.**
- **`ransac_thresh=1.0`** — 15 senaryoda kimse geçemedi, keskin optimum (1,5'te y RMS 12,8).
- **`sim↔hom uyuşmazlık kapısı`** — GT'li zor verilerde zararlı, kaldırıldı (defter 07 B).
- **Sürüklenme sönümleme** (`yaw_katsayi`, `height_dz_katsayi`) — zararlı
  (yaw_katsayi=0 → 162,9 m). Araç gerçekten dönüp alçalıyor; güncellemeler doğru.
- **Çoklu taban çizgisi** (`VoCokluMerdiven`) — −%1,5 kazanç, +0,30 s/kare maliyet; elendi.
- **Merdiven yanlılık düzeltmesi** (`bias_*`) — uzun kesintide geriye düşüyor; elendi.
- **VGGT track eşleştirici** — kalite umut vericiydi ama 2250 karede 129 dk (pencere 90 dk).
- **Loop closure / yer tanıma** — kazanç yok.

---

## 5. Asıl teşhis hâlâ geçerli — ve sıradaki iş burada

Kesinti hatası **rastgele değil sistematik**: tahmin yörüngesi GT'nin ~2° döndürülmüş ve
%3–4,5 ölçeklenmiş hâli. Bu iki bileşen bilinseydi blok hatası 6,28 m yerine **1,45 m**
olurdu. Sönümleme denendi ve zararlı çıktı, çünkü sapma birikimden değil **ölçümün kendi
yanlılığından** geliyor.

Yani sıradaki kazanç parametre ayarında değil, **ölçüm katmanında** (yön ve ölçek
yanlılığının kendisini kestirmekte) aranmalı — ve artık genelleme sınavından geçmek
zorunda.
