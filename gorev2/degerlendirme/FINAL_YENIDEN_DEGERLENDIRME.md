# Görev 2 — Diyarbakır finali sonrası yeniden değerlendirme (Ekim 2026)

**Soru:** Finalde Görev 2'de 22. olduk (ort. 0,396). ORB-SLAM3, UAP/UAİ ölçeği, deniz
sabitleme gibi tekniklerle 5 oturumun hepsinde ne kadar hata yapardık, kaçıncı olurduk?

**Kısa cevap:** Önceden sabitlenmiş bir modalite kuralıyla — **RGB'de ORB-SLAM3 + Umeyama,
termalde mevcut RAFT** — tahmini Görev 2 ortalaması 0,77–0,81 (**Görev 2'de 4.–5.**),
toplam 0,600–0,616 (**genel 2.**; resmî 16.). RGB oturumlarında hata 13–25 kat düştü
(O1 156 → 9 m, O4 225 → 12 m). Kalan açık termal oturum (O3).

---

## 1. Ölçüm altyapısı (yeni)

| araç | iş |
|---|---|
| `final_yatak.py` | 5 oturumun logundan takvim, sağlıklı GT, **gönderdiğimiz** konum, Görev 1 kutuları; eksik kare güvenli okuyucu (O1 8, O2 6, O4 1 kare diskte yok); RAFT önbellekleri; gerçek-GT kontrol noktaları (O1 800–825, O3 1050–1070 / 1780–1800) |
| `sozde_gt.py` (`.venv_colmap`, pycolmap 4.2.1) | kesinti karelerinin konumu için COLMAP sözde-GT: her 3. kare, 2048 SIFT, ardışık + **döngü** çiftleri (2. geçiş), artımlı SfM, sabit iç parametre, Umeyama (**yansıma serbest**) + tek sağlıklı blokta **yatış düzeltmesi** |
| `final_puan.py` | resmî tablodan hata→puan kalibrasyonu (üstel / Gauss), oturum sırası, G2 sırası, toplam, genel sıra |
| `kahin_olcek.py` | doğru tanımlanmış metre/piksel kâhini |
| `src/olcek_capasi.py`, `capa_deney.py` | taşıt boyu + 4,5 m ped ölçek çapası |
| `orbslam3_docker/`, `orbslam3_poz.py` | ORB-SLAM3 v1.0 Docker'da, **nedensel** poz kaydı (`mono_nedensel.cc`); Umeyama bağlama, RAFT'a düşme, hibrit |
| `final_degerlendir.py`, `final_rapor.py` | 5 oturum × yöntem × 3 RANSAC permütasyonu; tablo + `outputs/final/final_rapor.png` |

### Sözde-GT doğruluğu (gerçek GT'ye karşı)

| sınav | hata |
|---|---|
| video1 (tam GT), yalnız ilk 450 kareyle oturtma → 1800 kesinti karesi | **ort 2,43 m** (medyan 2,21, maks 5,48) |
| O1, yalnız ilk blokla → gizli 800–825 bloğu | 6,20 m |
| O3 (termal), üç sağlıklı blokla oturtma kalıntısı | 4,26 m RMS |
| O1 799. karede bizim tahmin ↔ sözde-GT | 20,5 m (ölçülen dönüş sıçraması ~21,6 m) |

Düzeltmeler sırasıyla: döngü çiftleri 33,4 → 17,0 → 3,3 m; yansıma serbest + yatış 2,4 m.

## 2. Bulunan iki hata (depodaki eski sonuçları etkiliyor)

**a) Umeyama'da yansıma yasağı.** Yarışma GT çerçevesi sağ-el bir kamera/SfM çerçevesine göre
**aynalı** (en iyi oturtmada det R = −1). `slam_degerlendirme.sim3_fit` yansımayı yasaklıyordu:
COLMAP'te kâhin hizalamada bile düşey hata 17,3 m ve z, GT z ile −0,50 korelasyonlu çıkıyordu.
ORB-SLAM3 video1: 16,41 → **2,52 m**. `aynalama=True` bayrağı eklendi. **SLAM_TURU'ndaki VGGT /
ORB-çekirdeği sonuçları bu hatayla ölçüldü — yeniden ölçülmeye değer.**

**b) Yükseklik kâhini yanlış tanımlıydı.** `YUKSEKLIK_KAHINI.md` "kusursuz yükseklik bile zarar
verir" sonucunu `current_height(gt_z)` ile, yani **GT irtifası + düz arazi** ile elde etmişti.
Görüntü ölçeğini belirleyen yerden yüksekliktir (O4 sağlıklı döneminde uçak 47,7 m inerken AGL
sabit kaldı). Metre/piksel kâhini (kesintide yol uzunluğu oranıyla h düzeltme): **12 yatak×
takvimin 11'inde kazanç, ort. 16,70 → 12,01 m**. Ölçek gözlemi ailesi yeniden açık.

## 3. Yöntemler — final oturumları (sözde-GT'ye göre oturum E'si, m; 3 permütasyon ort.)

| yöntem | O1 | O2 | O3 (termal) | O4 | Test |
|---|---|---|---|---|---|
| gönderilen (yarışmada) | 156,3 | 8,6 | 58,5 | 224,8 | 15,3 |
| temel (RAFT, yeniden koşum) | 152,5 | 10,3 | 81,9 | 216,7 | 15,0 |
| araç çapası (ölü bant 0,3, g 0,5) | 61,4 | 9,6 | 83,0 | 241,5 | 15,4 |
| ped çapası (yalnız 4,5 m ped) | 152,5 | 9,9 | 81,9 | 232,6 | 15,0 |
| hibrit (RAFT yönü + ORB ölçeği) | 17,0 | 8,2 | 174,4 | 279,4 | 13,8 |
| **ORB-SLAM3 + Umeyama** | **9,0** | **5,5** | 197,6 | **13,1** | **2,2** |
| ORB-SLAM3 + z limiti 20 m | 9,0 | 5,5 | 196,5 | 12,2 | 2,3 |
| **KURAL: RGB→ORB, termal→RAFT** | **9,0** | **5,5** | **81,9** | **13,1** | **2,2** |

Gerçek-GT kontrol noktalarında (O1 349 kare, O3 599/709/1329 kare kör) RGB'de yöntemler yakın
(O1: 18–22 m — erken nokta, arazi kaynaklı düşey); termal O3'te ORB-SLAM3 152–559 m ile çöküyor.

**ORB-SLAM3 tekrarlanabilirliği** (çok iş parçacıklı): O1 9,00 / 9,33 / 8,93 m; O4 13,05 /
14,93 / 10,04 m.

### Neden ölçek düzeltmeleri O4'te yetmedi
Gönderilen O4 yörüngesi sözde-GT'ye göre: yol oranı 0,75 → 0,31 (ölçek çöküşü, araç boyuyla
bağımsız ölçümle aynı) **ve** yön farkı −54°, −65°, … **+92°, +95°**. RAFT'ın yönü de çökmüş;
yalnız ölçeği düzeltmek yanlış yöne giden yolu uzatıyor. ORB-SLAM3 ikisini birlikte çözüyor.

### GT'li 6 yatakta (video1, video2, v25 ×4; çevrim içi + final-tipi takvim)
| | RAFT | ORB-SLAM3 | hibrit |
|---|---|---|---|
| RGB uçuşları ort. | 12,42 | **7,26 (−%42)** | 11,87 |
| termal uçuşları ort. | 20,85 | 31,33 | **19,90** |

Modalite kuralı bu tablodan, final oturumlarına bakılmadan çıktı; finalde O3 (termal) ORB'nin
çöktüğünü bağımsız olarak doğruladı.

## 4. Tahmini puan ve sıra

Resmî tablodan kalibrasyon: her oturumda (gönderilen E, resmî skor) → τ. Diğer takımların
skorları sabit tutuldu.

| yöntem | biçim | O1 | O2 | O3 | O4 | G2 ort. | G2 sırası | toplam | genel |
|---|---|---|---|---|---|---|---|---|---|
| gönderilen | — | 0,057 (24.) | 0,872 (13.) | 0,546 (11.) | 0,110 (27.) | 0,396 | 22. | 0,4510 | 16. |
| ORB-SLAM3 (her yerde) | üstel | 0,848 (10.) | 0,916 (2.) | 0,129 (19.) | 0,880 (1.) | 0,693 | 5. | 0,5699 | 5. |
| ORB-SLAM3 (her yerde) | Gauss | 0,991 (1.) | 0,945 (1.) | 0,001 (33.) | 0,993 (1.) | 0,732 | 5. | 0,5856 | 3. |
| **KURAL** | **üstel** | 0,848 (10.) | 0,916 (2.) | 0,428 (12.) | 0,880 (1.) | **0,768** | **5.** | **0,5998** | **2.** |
| **KURAL** | **Gauss** | 0,991 (1.) | 0,945 (1.) | 0,305 (16.) | 0,993 (1.) | **0,808** | **4.** | **0,6160** | **2.** |
| araç çapası | üstel / Gauss | | | | | 0,425 / 0,464 | 18. / 15. | | 16. / 15. |

Birincinin toplamı 0,6366; aradaki 0,02–0,04'ün tamamı **termal O3**'te (kural orada 0,43 / 0,31).

τ tutarlılığı: üstel 54,5 / 62,7 / 96,6 / 101,8 m (2 kat aralık); Gauss 92,3 / 23,2 / 75,2 /
151,3 m (6,5 kat). Organizasyon sabit bir formül kullandıysa **üstel biçim daha olası**.

## 5. İstenen ama elenen / ertelenenler

- **UAP/UAİ 4,5 m ped ölçeği:** kuruldu ve ölçüldü. Yanlış pozitif pedler (O4'te medyan 34 px;
  50 m'de gerçek ped ≈ 125 px) ve tutarsız kalibrasyon yüzünden video1'de +%44, v25_termal_a'da
  +%73 zarar, O4'te 216,7 → 232,6 m. Kanal **kapalı**; araç çapası yalnız sağlıklı dönemde
  öğrenilen taşıt boyuyla ve ölü bantla güvenli.
- **Deniz/su sabitleme:** 5 oturumun 150'şer karesinde su yüzeyi yok (en yüksek aday mavi
  şemsiyeler). Puanı etkileyemez, ölçülemez → bu turda yapılmadı.
- **DPVO:** sonraki adım (Docker'da CUDA 12.8 + torch cu128 ile lietorch yeniden derleme + `dpvo.pth`).

## 6. Sınırlar

- Sözde-GT hatası (2–6 m), ORB-SLAM3'ün en iyi oturumlardaki hatasıyla (2–13 m) aynı mertebede:
  en üstteki sıralar (1. mi 3. mü) bu belirsizliğe duyarlı; büyük farklar (RAFT 150–225 m'ye
  karşı ORB 9–13 m) değil.
- Hata→puan dönüşümü yayınlanmadı; iki biçim aralık olarak verildi. Diğer takımların skorları
  sabit tutuldu (bizim hatamız değişince formül onları da etkileyebilir).
- Temel yöntemin yeniden koşumu, RANSAC rastgeleliği yüzünden gönderilenden farklı
  (O3: gönderilen 58,5 m, yeniden koşumların ort. 81,9 m — yarışmada şanslı bir koşuydu).
- ORB-SLAM3 çevrimdışı (Docker) koşuldu ama **nedensel** poz kaydıyla; canlı sisteme bağlanmadı.
  Hız uygun (~0,08 s/kare, 1280×720).

## 7. Sonraki adımlar
1. Termal: ORB-SLAM3 termalde çöküyor; termal için RAFT + (video2'de çok kazandıran) araç çapası,
   DPVO ve CLAHE'li ORB denenmeli — birinciliğe kalan fark tamamen burada.
2. ORB-SLAM3'ü canlı yola bağlamak (Docker'da soket/paylaşımlı klasör ile kare başına) ve iz
   kaybında RAFT'a düşmeyi canlı sınamak.
3. Yansıma düzeltmesiyle SLAM_TURU (VGGT, DPVO) yeniden ölçülmeli.
