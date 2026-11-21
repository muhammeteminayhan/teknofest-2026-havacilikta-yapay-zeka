# TEKNOFEST 2026 — Havacılıkta Yapay Zeka · Takım **bugbuster**

İnsansız hava aracının alt-görüş kamerasından gelen canlı video akışında **üç görevi aynı
oturumda** çalıştıran sistem: nesne tespiti, GPS'siz pozisyon kestirimi ve zero-shot
görüntü eşleme. Yarışma sunucusuna saniyeler içinde yanıt vermek zorunda olan, tek GPU
üzerinde çalışan bütünleşik bir boru hattı.

**Sonuç:** 35 takım arasında **16.** (toplam 0,45102 · FTR 85,0 · sunum 75,34).
Selçuk Üniversitesi · Diyarbakır finali, Eylül 2026.

> Bu depo yarışma sonrası **olduğu gibi** açıldı: çalışan sistemin yanı sıra denenip
> **elenen** yöntemlerin ölçüm kaydını da içeriyor. Neyin işe yaramadığı, neyin
> yaradığı kadar ayrıntılı yazılmıştır.

---

## Üç görev

| Görev | İş | Puan payı | Klasör | Yaklaşım |
|---|---|---|---|---|
| **Görev 1** | Nesne tespiti — taşıt / insan / iniş alanı (UAP, UAİ) + durum | — | [`gorev1/`](gorev1/) | DINOv3 gövdeli YOLO + BoT-SORT |
| **Görev 2** | GPS'siz pozisyon kestirimi | **%40** | [`gorev2/`](gorev2/) | RAFT optik akış + düzlemsel homografi |
| **Görev 3** | Görüntü eşleme — referans nesneyi karede bulma | — | [`gorev3/`](gorev3/) | DINOv3 patch benzerliği (zero-shot) |
| — | Sunucu bağlantısı, üç görevin birleştirilmesi | — | [`sistem/`](sistem/) | — |

Görev 2 baştan sona bu depoda geliştirildi. Görev 1'in ağırlığı ve Görev 3'ün çekirdek
eşleme mantığı takım arkadaşımıza ait; buradaki kod onları şartname şemasına bağlar ve
eksik kalan parçaları (hareket durumu hesabı, sınıf eşlemesi) tamamlar — ayrıntısı her
görevin kendi README'sinde.

---

## Resmî sonuçlar

Dört puanlı oturum, görev başına normalize skor (1,0 = en iyi):

| Oturum | Görev 1 | Görev 2 | Görev 3 | Oturum skoru |
|---|---|---|---|---|
| O1 | 0,408 | 0,057 | 0,502 | 0,250 |
| O2 | 0,567 | 0,872 | 0,587 | 0,637 |
| O3 (termal) | 0,335 | 0,546 | 0,128 | 0,334 |
| O4 | 0,528 | 0,110 | 0,344 | 0,262 |
| **Ortalama** | **0,460** | **0,396** | **0,390** | **0,371** |

Görev 2 oturumlar arasında çok oynak: O2'de 0,872, O1'de 0,057. Bu dalgalanma finalden
sonra incelendi ve sebebi bulundu.

### Final sonrası bulgu: ORB-SLAM3

Görev 2'de 22. olduktan sonra, kaybın nereden geldiğini ölçmek için bir tur daha yapıldı
([`FINAL_YENIDEN_DEGERLENDIRME.md`](gorev2/degerlendirme/FINAL_YENIDEN_DEGERLENDIRME.md)).
Teşhis: tek-çift homografi + düzlem varsayımı **ölçek** tarafında tavan yapmış; kötü
oturumlarda hem ölçek hem yön çökmüş.

ORB-SLAM3 daha önceki turda "Windows'ta Pangolin/DBoW2/g2o derlemesi" gerekçesiyle
*denenmeden* elenmişti. Docker'a taşınıp gerçekten koşturulduğunda:

| Yöntem | O1 | O2 | O3 (termal) | O4 |
|---|---|---|---|---|
| Gönderilen (RAFT) | 156,3 m | 8,6 m | 58,5 m | 224,8 m |
| ORB-SLAM3 + Umeyama | **9,0 m** | **5,5 m** | 197,6 m ✗ | **13,1 m** |
| Kural: RGB→ORB, termal→RAFT | 9,0 m | 5,5 m | 81,9 m | 13,1 m |

RGB uçuşlarında hata 13–25 kat düşüyor, termalde ORB-SLAM3 çöküyor. Modalite kuralı
GT'si olan 6 bağımsız yataktan çıkarıldı; final oturumu O3 bunu bağımsız doğruladı.
Bu kuralla tahmini Görev 2 ortalaması 0,77–0,81 (**Görev 2'de 4.–5.**), toplam
0,600–0,616 (**genel 2.**).

Ölçümün sınırı da kayıtta: sözde-GT'nin kendi hatası 2–6 m, yani en üst sıraların
(1. mi 3. mü) bu belirsizliğe duyarlı olduğu, 150 m'ye karşı 9 m gibi büyük farkların
olmadığı açıkça yazılmış. **Bu tur üretim hattına bağlanmadı** — `sistem/` altında
ORB-SLAM referansı yoktur, çevrimdışı bir ölçümdür.

---

## Yöntem

### Görev 1 — nesne tespiti
DINOv3 gövdeli YOLO (`trial-8`, 127 dönem, en iyi 117. dönem, `imgsz=1280`):
**mAP50 0,813 · mAP50-95 0,649**. BoT-SORT ile takip. İki nokta şartname uyumu için
ayrıca hesaplanır:

- **Hareket durumu** — ardışık kareler arasındaki kamera ego-hareketi homografi ile
  telafi edilir; takip edilen taşıtın beklenen ve gerçek konumu arasındaki kayma eşiği
  (5 px) geçerse "hareketli".
- **Sınıf eşlemesi** — model `{2:UAİ, 3:UAP}` sırasıyla eğitilmiş, şartname
  `{2:UAP, 3:UAİ}` istiyor; takas kodda yapılır.

### Görev 2 — GPS'siz pozisyon kestirimi
GPS kesintisi boyunca, yalnız monoküler alt-görüş görüntüsünden video başlangıcına göre
X/Y/Z yer değiştirmesi. Sevk edilen reçete: **RAFT optik akış + düzlemsel homografi
ayrıştırması + keyframe merdiveni** (`spacing=8`), `smooth_alpha=0.6`.

Resmî metrik yalnızca kesinti karelerindeki hatayı ölçer — sağlıklı karelerde herkes
GT'yi aynaladığı için hata sıfırdır. Görsel bilgi kullanmayan kıyas tabanları: *konumu
tut* 63,6 m, *sabit hız* 160,7 m. Sistem bunların ~14 katı altında.

Kritik metodolojik ders depoda kayıtlı: bütün ayar çalışması başta
`THYZ_2026_Ornek_Veri_1.MP4` üzerinde yapılmıştı ve **o video test oturumunun ta
kendisiydi**. Örnek veride %25 kazandıran düşey kanal, bağımsız pencerelerde kaybetti
ve geri alındı. Sonrasında her değişiklik için zorunlu bir **genelleme sınavı** kapısı
kondu: bağımsız iki pencerede de kazanmayan değişiklik girmiyor.

### Görev 3 — görüntü eşleme
Oturum başında verilen referans nesneler modelde eğitilmemiştir; **görsel benzerlikle**
bulunur. DINOv3 ViT-S/16 CLS token'ı ile karenin patch token'ları kosinüs benzerliğine
sokulur → ısı haritası → morfoloji + eşik ile kutu. Üç ölçek (224/168/112) max füzyonla
birleştirilir. Referansın arka planı rembg ile temizlenir. DINOv3 kapılı bir model
olduğundan kapısız **dinov2** yedeği tek satırla devreye alınabilir.

---

## Elenen yöntemler kaydı

Görev 2 için denenip **reddedilen** her yöntem, neden reddedildiği ve ölçüm sayılarıyla
[`gorev2/degerlendirme/`](gorev2/degerlendirme/) altında duruyor (17 belge). Öne çıkanlar:

| Belge | Konu |
|---|---|
| [`OZET.md`](gorev2/degerlendirme/OZET.md) | Genel özet ve yürürlükteki reçete |
| [`SLAM_TURU.md`](gorev2/degerlendirme/SLAM_TURU.md) | DPVO, VGGT, DROID, MASt3R, ORB-SLAM3 turu |
| [`FINAL_YENIDEN_DEGERLENDIRME.md`](gorev2/degerlendirme/FINAL_YENIDEN_DEGERLENDIRME.md) | Final sonrası ORB-SLAM3 ölçümü |
| [`OLCUM_KATMANI.md`](gorev2/degerlendirme/OLCUM_KATMANI.md) | SuperPoint/LightGlue, DISK, XFeat denemeleri |
| [`DUSEY_KANAL.md`](gorev2/degerlendirme/DUSEY_KANAL.md) | Örnek veriye aşırı uyum vakası, geri alma kaydı |
| [`YONTEM_SINAVI.md`](gorev2/degerlendirme/YONTEM_SINAVI.md) | Bağımsız yataklarda yöntem karşılaştırması |

---

## Kurulum

Python 3.13 · CUDA'lı PyTorch 2.11 (cu128) · NVIDIA GPU önerilir.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

Sürümler çalışan ortama sabitlenmiştir (`requirements.txt`). İki not:

1. **`ultralytics` standart pip'ten kurulur** — özel fork gerekmez.
2. **DINOv3 kapılı (gated) bir modeldir.** Bir kez HuggingFace erişim onayı + token ile
   indirilip `~/.cache/huggingface`'e önbelleklenir; sonrasında token gerekmez,
   çevrimdışı çalışır. Erişim yoksa `Gorev3Matcher(kind="dinov2")` ile kapısız yedeğe
   geçilir.

### Kimlik bilgileri

Gerçek sunucu bilgileri depoda **yok** (`.gitignore`'da). Şablondan kendi dosyanızı
oluşturup doldurun:

```powershell
Copy-Item sistem\config\example.env sistem\config\.env
notepad sistem\config\.env
```

---

## Çalıştırma

```powershell
cd sistem
..\.venv\Scripts\python main.py
```

Yarışma günü adım adım işletim kılavuzu: **[YARISMA_KULLANIM.md](YARISMA_KULLANIM.md)**.

### Doğrulama

Sunucuya hiçbir tahmin göndermeyen, salt okunur bağlantı testi:

```powershell
cd sistem
..\.venv\Scripts\python baglanti_testi.py
```

Modellerin internetsiz yüklendiğini doğrulamak için:

```powershell
$env:HF_HUB_OFFLINE="1"; $env:TRANSFORMERS_OFFLINE="1"; cd sistem; ..\.venv\Scripts\python -c "import sys; sys.path.insert(0,'.'); from src.object_detection_model import ObjectDetectionModel; m=ObjectDetectionModel('http://x/'); print('gorev1:', m.gorev1 is not None, '| gorev3:', m.gorev3 is not None)"
```

---

## Depo yapısı

```
├── gorev1/                      Nesne tespiti (ağırlık + dedektör)
├── gorev2/                      Pozisyon kestirimi
│   ├── src/                       çalışan boru hattı modülleri
│   ├── degerlendirme/             ölçüm ve eleme kayıtları (17 belge)
│   ├── orbslam3_docker/           ORB-SLAM3 imajı + nedensel poz sürücüsü
│   └── outputs/final/             final sonrası ölçüm sonuçları
├── gorev3/                      Görüntü eşleme
├── sistem/                      Canlı sunucu arayüzü, üç görevin birleşimi
├── trial-8/                     Görev 1 eğitim kaydı (grafikler, results.csv)
├── docs/                        Şartnameler + teknik tasarım raporu (FTR)
├── THYZ_2026_Ornek_Veri_Seti/   Örnek referans nesneler + kalibrasyon
└── YARISMA_KULLANIM.md          Yarışma günü işletim kılavuzu
```

Örnek veri setinin videoları (`*.MP4`) ve Görev 1 eğitim ağırlıkları
(206 MB × 2) GitHub dosya sınırı nedeniyle depoda değildir; sevk edilen
`gorev1/agirliklar/best.pt` (21 MB) dâhildir. Görev 2'nin COLMAP ara verisi
(12 GB) de hariç — sonuç dosyaları `gorev2/outputs/final/` altında durur.

---

## Doğrulanmış kaynak bütçesi

RTX 5060 8 GB üzerinde ölçülmüş:

| Senaryo | Kare başı | Tepe VRAM |
|---|---|---|
| Görev 1 + 2 | ~0,7 s | ~1,1 GB |
| Görev 1 + 2 + 3 (aktif) | ~1,0 s | ~1,1 GB |

Sunucunun 80 gönderim/dakika sınırı kare başına ~0,75 s taban koyduğundan süre bütçesi
rahattır. Görev 3'ün maliyeti referans sayısından bağımsızdır — kare öznitelikleri bir
kez çıkarılır.

---

## Lisans ve kullanım

Kod için henüz bir lisans dosyası eklenmemiştir; bu durumda varsayılan olarak tüm
haklar saklıdır. Yeniden kullanmak isterseniz iletişime geçin.

`THYZ_2026_Ornek_Veri_Seti/` ve `docs/` altındaki şartnameler TEKNOFEST'e aittir ve
yalnızca referans amacıyla bulunur.
