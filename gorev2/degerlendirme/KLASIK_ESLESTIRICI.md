# Görev 2 — klasik eşleştiriciler (SIFT / ORB / AKAZE) sınavı

## Neden yapıldı

Ölçüm katmanı turlarında **öğrenilmiş** seyrek eşleştiriciler denenmişti (XFeat,
DISK+LightGlue, gerçek SuperPoint+LightGlue) ve hepsi RAFT'a yenilmişti; ayrıca
ORB-SLAM'in **tamamı** bir sistem olarak denenmişti (monoküler ölçek üçgenlemesi nadir
sahnede dejenere oldu). Ama **klasik dedektörler kendi boru hattımızda bir
`MatchResult` üreticisi olarak hiç sınanmamıştı.** Bu boşluk kapatıldı.

Yeni modül: [`src/klasik_eslestirici.py`](../src/klasik_eslestirici.py) —
SIFT / ORB / AKAZE / BRISK, `FeatureTracker` arayüzüyle.
Eşleştirme politikası: k=2 + **Lowe oran testi** (0,75) + **karşılıklı en yakın komşu**;
RANSAC'a yalnız ikisinden de geçenler girer.

**Her eşleştirici kendi eşiğiyle taranmıştır** — yeniden-izdüşüm eşiği eşleştiriciye
özgüdür (geçmişte SuperPoint'in optimumu RAFT'ınkinin 8 katı çıkmıştı).

## Sonuç: hiçbiri RAFT'ı geçemedi

video1, yarışma takvimi + 15 uzun-kesinti senaryosu, sevk edilen durum makinesi:

| eşleştirici | en iyi eşik | yarışma | 15 uzun | yön° | ölçek% | inlier | karar |
|---|---|---|---|---|---|---|---|
| **RAFT (mevcut)** | 1,0 | **3,770** | **7,492** | 1,54 | 2,42 | 0,826 | — |
| SIFT | 1,0 | 4,480 | 7,965 | **1,30** | 3,07 | 0,729 | ❌ 5/15 |
| AKAZE | 2,0 | 5,458 | 9,461 | 1,86 | 5,94 | 0,873 | ❌ 4/15 |
| **ORB** | 2,0 | 14,589 | 28,457 | 5,58 | 19,75 | 0,511 | ❌ 1/15 |
| SIFT tam çözünürlük | 4,0 | 6,166 | 10,991 | 2,26 | 6,60 | 0,951 | ❌ 0/15 |

**Notlar.**
- **SIFT en yakın rakip** (+%19). İlginç: **yön hatası RAFT'tan iyi** (1,30° vs 1,54°)
  ama **ölçek hatası kötü** (%3,07 vs %2,42) — depoda üçüncü kez görülen aynı ayrışma:
  yön ve ölçek farklı kaynaklardan geliyor.
- **ORB elenmeli**: homografi inlier oranı yalnız %27–51. İkili betimleyici havadan
  çekimdeki tekrarlayan dokuda ayırt edici değil.
- **Tam çözünürlük SIFT'i kötüleştiriyor** (4,48 → 6,17). RAFT'ın `ds0,75` denemesiyle
  aynı örüntü: bu boru hattında daha yüksek çözünürlük ölçek yanlılığını azaltmıyor.
- İlk bakışta SIFT umut vericiydi (RAFT'tan çok nokta: 1784 vs 627; daha yüksek inlier:
  0,742 vs 0,696; daha hızlı: 214 vs 286 ms). **Nokta sayısı ve inlier oranı sonucu
  öngörmüyor** — daha önce uyarlanabilir ızgara denemesinde de aynı ders çıkmıştı.

## ✅ Yan ürün: CPU (GPU'suz) yedek yolu 4,4× iyileşiyor

RAFT GPU ister. `LivePositionEstimator` GPU/ağırlık bulamazsa **LK yedeğine** düşüyor —
ve o yedek `method="fuzyon"` + **merdiven kapalı** olarak yapılandırılmış. Ölçüldü:

| motor | reçete | yarışma | 15 uzun |
|---|---|---|---|
| **LK — sevk edilen yedek** (füzyon, merdiven yok) | | **19,810** | 13,505 |
| LK | homografi + merdiven(8) | 6,365 | 13,092 |
| **SIFT** | homografi + merdiven(8) | **4,480** | **7,965** |
| AKAZE | homografi + merdiven(8) | 5,458 | 9,461 |
| *(kıyas)* RAFT/GPU | homografi + merdiven(8) | 3,770 | 7,492 |

### Üç yatakta doğrulama (tek uçuşa güvenmeme dersi gereği)

| yatak | LK — sevk edilen yedek | LK + merdiven | **SIFT + merdiven** | *(kıyas)* RAFT/GPU |
|---|---|---|---|---|
| video1 | 20,39 | 6,36 | **4,48** | 3,77 |
| sınav (puanlı oturum) | 15,07 | **12,54** | 13,76 | 11,08 |
| video2-termal | 12,13 | 18,19 | **9,40** | 14,30 |
| **ortalama** | **15,86** | 12,36 | **9,21** | 9,72 |

- **SIFT 3 yatağın 2'sinde en iyi**, hiçbirinde felaket yok, en kötü hâli 13,76 m
  (eski yedeğin en kötüsü 20,39 m).
- **LK + merdiven** ortalamada eski yedekten iyi ama **termalde bozuluyor** (12,13 → 18,19).
- SIFT'in ortalaması GPU'lu RAFT'ınkinden bile düşük — ama bu tek bir yatağın
  (termal, 9,40 vs 14,30) etkisi; birincil motor değişikliği için yeterli kanıt değil.

## ✅ UYGULANDI: CPU yedeği LK → SIFT

`live_estimator.py` motor zinciri **RAFT → SIFT → LK** oldu:

| istek | seçilen motor | reçete | merdiven | hız |
|---|---|---|---|---|
| `auto` / `raft` | raft (GPU) | homografi | açık | ~650 ms/kare |
| (RAFT kurulamazsa) | **sift** (CPU) | **homografi** | **açık** | ~280 ms/kare |
| `lk` (açık istek / son çare) | lk | füzyon | kapalı | ~40 ms/kare |

Aynı anda iki hata düzeltildi: yedek motor (LK → SIFT) **ve** yedek reçete
(füzyon/merdivensiz → homografi/merdivenli). Açık `engine="lk"` isteği hâlâ LK verir.

## Yeniden koşturmak için

```
.venv\Scripts\python gorev2\nokta_onbellek.py --tracker sift --yatak video1
.venv\Scripts\python gorev2\nokta_onbellek.py --tracker sift --yatak video1 --spacing 8
.venv\Scripts\python gorev2\klasik_ab.py --tracker sift akaze orb
```
