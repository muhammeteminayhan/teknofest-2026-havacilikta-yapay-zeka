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
metrik altında ölçüme dayalı bir iyileştirme turu yapıldı — **tam rapor:
[`degerlendirme/OZET.md`](degerlendirme/OZET.md)**.

**Test yatağı** (`test_yatagi.py`): örnek veri seti (tam GT) + puanlı oturum logundan
çıkarılan **gerçek kesinti takvimi** (703 sağlıklı / 1547 kesinti, üç uzun blok:
449/499/599 kare) + resmî metrik (`competition_error`). Aşırı uyumu yakalamak için her
deneme ayrıca rastgele kesinti takvimlerinde de ölçüldü.

| Deneme | Sonuç |
|---|---|
| **`smooth_alpha=1.0`** (ölçüm EMA'sı kapalı) | ✅ **uygulandı** — ort. hata 4,535→4,343 m (yarışma takvimi), 5,280→**4,186 m** (rastgele, −%21), en büyük hata 15,04→12,42 m |
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
