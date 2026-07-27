# Görev 2 — puanlı oturum sonrası iyileştirme çalışması (özet)

**Bağlam:** Puanlı oturumda PK (Pozisyon Kestirimi) **55,29 puan / 8 takım içinde 8.** sırada
kaldı. Bu çalışma, resmî metrik altında ölçüme dayalı iyileştirme arayışıdır.

**Resmî metrik** (`src/evaluation.py:competition_error`, teknik şartname Denklem 2):
*tüm karelerde ortalama Öklid hatası*. Sağlıklı karelerde herkes GT'yi aynaladığı için hata 0;
**puanı yalnızca kesinti karelerindeki hata belirler.**

## Test yatağı

`test_yatagi.py` — sahadaki koşulun birebir kopyası:
- **Veri:** örnek veri seti video 1 (tam GT), yarışma temposuna altörneklenmiş (2250 kare)
- **Kesinti takvimi:** puanlı oturum logundan çıkarılan **gerçek** yapı —
  `echo 0-450 │ KESİNTİ 451-899 │ echo 900-1050 │ KESİNTİ 1051-1549 │ echo 1550-1650 │ KESİNTİ 1651-2249`
  (703 sağlıklı + 1547 kesinti; yarışma logu ile birebir aynı)
- **Genelleme kontrolü:** ayrıca rastgele kesinti takvimleri (aşırı uyumu yakalamak için)

**Kıyas tabanları:** görsel bilgi kullanmayan yöntemler — *tut* 63,6 m, *sabit hız* 160,7 m.
Sistemimiz bunların **14×** altında; yani çekirdek yaklaşım sağlam.

## Sonuçlar

| Deneme | Araç | Sonuç |
|---|---|---|
| **Ölçüm yumuşatma kapalı** (`smooth_alpha=1.0`) | `parametre_tarama.py` | ✅ **UYGULANDI** |
| Diğer 10 parametre (calib_window, eşikler, cv_alpha…) | `parametre_tarama.py` | ❌ etkisiz / aşırı uyum |
| Loop closure (DINOv3 yer tanıma) | `loop_closure_test.py` | ❌ kazanç yok — kapalı |
| Sürüklenme sönümleme (yaw/yükseklik) | `surukleme_tarama.py` | ❌ zararlı |
| Keyframe aralığı (spacing 4–24) | `spacing_tarama.py` | ❌ 8 en iyi |

### ✅ Kazanç: `smooth_alpha=1.0`

| Metrik | Önce (0,6) | Sonra (1,0) |
|---|---|---|
| Ortalama Öklid hatası — yarışma takvimi | 4,535 m | **4,343 m** |
| Ortalama hata — 6 rastgele takvim | 5,280 m | **4,186 m (−%21)** |
| En büyük hata (kuyruk riski) | 15,04 m | **12,42 m** |

Ölçüm EMA'sı uzun kesintide gecikme ve sistematik sapma ekliyordu. `live_estimator.py`
içindeki RAFT reçetesine işlendi.

### ❌ Loop closure neden çalışmadı

Fikir sağlamdı: puanlı oturumda araç kesinti sırasında **aynı bölgeye dönüyordu**
(354 m yol → 51 m net). Sağlıklı karelerin DINOv3 betimleyicisi + GT konumu haritalanıp
kesintide tanınırsa sürüklenme sınırlanabilirdi. Kuruldu (`src/yer_tanima.py`, üç güvenlik
kapısı: mutlak benzerlik, Lowe oranı, uzaysal makullük) ama:

| Veri | Kesintide net/yol oranı | Sağlıklı alana yakınlık |
|---|---|---|
| **Örnek video** (tek GT'li veri) | 0,41–0,71 → **ileri gidiyor** | %25'i 20 m içinde |
| Yarışma (gerçek) | 0,11–0,14 → daireler çiziyor | %37'si 20 m içinde |

**Ölçebildiğimiz veride o davranış yok**; yarışma verisinde ise kesinti karelerinin GT'si
bizde yok. 18 eşik kombinasyonu denendi: yüksek eşikte neredeyse hiç tetiklenmedi (8/1547),
düşük eşikte yanlış eşleşip hatayı artırdı. **Varsayılan kapalı**; kod ve test korunuyor.

### ❌ Sürüklenme sönümleme neden çalışmadı

Procrustes teşhisi: hata sistematik (yön ~2°, ölçek %3–4,5); bu bileşenler bilinseydi blok
hatası 6,28 m yerine 1,45 m olurdu. Ancak güncellemeleri sönümlemek felaketle sonuçlandı
(`yaw_katsayi=0` → 162,9 m). Araç gerçekten dönüp alçaldığı için güncellemeler **doğru ve
gerekli**; sapma birikimden değil, ölçümün kendi yanlılığından geliyor. Kazanç için ölçümü
(optik akış/homografi) iyileştirmek gerekir — parametre ayarı değil.

### ❌ Keyframe aralığı

`spacing=16` yarışma takviminde çok iyi göründü (3,375 m, −%22) **ama** 15 uzun-kesintili
rastgele senaryoda üstünlüğü doğrulanamadı (7/15, ortalama fark +0,04 m = gürültü).
Tek takvime aşırı uyum olduğu anlaşıldı; **spacing=8 korundu**.

## Çıkarım

Sistem, geliştirme aşamasında zaten iyi optimize edilmiş; parametre düzeyinde kolay kazanç
kalmamış. Elde edilen tek gerçek kazanç `smooth_alpha=1.0` (genel senaryolarda −%21).
Daha büyük sıçrama için **ölçüm katmanının** iyileştirilmesi gerekiyor (daha iyi optik akış /
doğrudan yükseklik gözlemi / harici sensör), ya da yarışma benzeri (dönüşlü) GT'li veri
elde edilip loop closure'ın yeniden değerlendirilmesi.

## Dosyalar

- `test_yatagi.py` — baseline ölçüm + hata eğrisi
- `parametre_tarama.py` — 11 parametre × 2 senaryo ailesi
- `loop_closure_test.py` — betimleyici önbelleği + 18 eşik kombinasyonu
- `surukleme_tarama.py` — yaw/yükseklik sönümleme
- `spacing_tarama.py` — keyframe aralığı (her değer için taze RAFT ölçümü)
- `src/yer_tanima.py` — VPR modülü (varsayılan kapalı)
