# Görev 2 — VGGT turu: **aile kapandı** (14 Ağu 2026)

> ## ⛔ SONUÇ: sorun parça birleştirmede DEĞİL. VGGT'nin bu veri türündeki yerel
> geometrisi mevcut ön ucumuzun 3‒8 katı kötü, parça ölçeği %42‒52 tutarsız ve
> derinlikle normalize etmek bunu **hiç** düzeltmiyor.

Betikler: [`tavan_vggt_parca.py`](../tavan_vggt_parca.py) ·
[`vggt_cozunurluk.py`](../vggt_cozunurluk.py) · [`vggt_derinlik_normu.py`](../vggt_derinlik_normu.py)

---

## 0. Önceki turun tespiti ve neden yeniden açıldı

`SLAM_TURU.md` şunu yazmıştı:

> "Kâhin (oracle) Sim(3) ile ölçülen yörünge kalitesi, sorunun **VGGT'de değil parça
> birleştirmede** olduğunu gösteriyor. İlk 50 kare: 0,97 m / 45 m yol = %2. … ileride
> zincirleme düzgün kurulursa **en umut verici aday** budur."

Bu tespit **tek bir pencereden** geliyordu. Bu tur onu bütün diziye yaydı ve
**çürüttü**.

## 1. Parça içi kalite — kâhin Sim(3) ile, 563 parça

Her 8 karelik parça GT'ye ayrı ayrı kâhin Sim(3) ile oturtuldu. Bu, **herhangi bir
birleştirme yönteminin ulaşabileceği en iyi** sonuçtur.

| | artık hata medyan | artık/GT yolu medyan | p95 |
|---|---|---|---|
| VGGT (294 px) | 0,479 m | **%8,1** | %13,1 |

Parça başına GT yolu medyan 6,39 m. Yani parça *içinde* bile hata yolun %8'i.

### Kıyas — **aynı ölçüt** mevcut sistemde

Aynı 8 karelik pencerelerde, aynı kâhin Sim(3) fiti, ama bu kez RAFT+homografi'nin
**gerçek ölü hesap** yörüngesi üzerinde (yalnız ilk 60 kare kalibrasyona açık):

| | artık/yol medyan | p95 |
|---|---|---|
| **RAFT + homografi** | **%1,0** | %2,7 |
| **VGGT** | **%8,1** | %13,1 |

**VGGT, kâhin hizalamayla bile, kâhinsiz çalışan mevcut sistemin 8 katı kötü.**

## 2. Çözünürlük sınırlayıcı mı? Şekilde evet, ölçekte hayır

294 px bir tercih değil zorunluluktu (8,5 GB GPU; 518 px paging → 40× yavaşlama).
25 parçada üç çözünürlük denendi:

| girdi | artık/yol medyan | p95 | **parça ölçeği değişim kats.** | s/parça | tepe VRAM |
|---|---|---|---|---|---|
| 294 px | %6,2 | %13,0 | **%52** | 0,84 | 7,79 GB |
| 392 px | **%3,4** | %9,1 | **%47** | 1,62 | 8,52 GB |
| 518 px | — | — | — | *paging* | >8,5 GB ✗ |

* **Şekil hatası çözünürlükle iyileşiyor** (%6,2 → %3,4) ama hâlâ RAFT'ın %1,0'ının
  3 katı.
* **Parça ölçeğinin tutarsızlığı iyileşmiyor** (%52 → %47). Zincirlemeyi öldüren şey
  budur ve daha fazla piksel çözmüyor.

518 px 8,5 GB'ı aşıp paging'e girdi ve durduruldu — bu donanımda erişilemez.

## 3. Zincirleme neden onarılamaz

| ölçüm | değer |
|---|---|
| parça ölçeği `s_c` değişim katsayısı | **%78** (563 parça, 294 px) |
| ardışık parça ölçek oranı, ort. \|sapma\| | **%14,8** |
| tüm diziye tek Sim(3) | 127,61 m |
| global ölçek+dönme, parça başına yalnız öteleme *(kâhin)* | 1,710 m |

Ardışık parçalar arasında ölçek her eklemde ortalama %14,8 sapıyor. 563 eklemde bu,
bir global graf optimizasyonunun geri getirebileceği bir gürültü değil.

Son satır ilgi çekici (1,71 m) ama **563 adet GT'den gelen öteleme** kullanıyor —
kesintide 1800 kare boyunca GT olmadığı için gerçekleştirilemez.

## 4. Son şans: derinlikle normalizasyon — ❌ **etkisiz**

**Fikir.** Parça ölçeği keyfi olsa bile VGGT derinliği de *aynı keyfi birimde* tahmin
ediyor. Öyleyse

```
e = |kamera merkezi adımı| / (parça içi medyan derinlik)
```

parça ölçeğinden bağımsız olmalı — ve bu tam olarak boru hattımızın beklediği
büyüklük (homografi de `delta/d` üretir, metriğe çevirmeyi öğrenilmiş `height` yapar).
Doğru olsaydı VGGT'yi global yörünge için değil, **mevcut doğrulanmış durum
makinesine yerel ölçüm besleyen bir ön uç** olarak kullanabilirdik ve parça zincirleme
sorunu tamamen ortadan kalkardı.

**Ölçüm** (60 parça, 392 px):

| büyüklük | değişim katsayısı |
|---|---|
| ham parça ölçeği `s_c` | %42 |
| `s_c × medyan derinlik` | **%43** |
| `g/e` oranı (= ima edilen uçuş yüksekliği) | **%42** |

**Normalizasyon hiçbir şey değiştirmiyor.** Derinlik, pozlarla tam olarak aynı
tutarsızlığı taşıyor.

Üstelik ima edilen yükseklik **medyan 31,09 m** (p05‒p95: 22,6‒55,3) — oysa video1
için ölçülen gerçek uçuş yüksekliği ~53,9 m (`OLCUM_TURU_2.md` §4). Yani ölçüm hem
%42 oynak hem sistematik olarak yanlış.

Kıyas: RAFT+homografi'nin ölçek hatası **%2,5**. VGGT'ninki **%42** — 17 katı.

## 5. Karar

**Aile kapandı.** Üç bağımsız kapı da kapalı:

1. **Global zincirleme** — parça ölçeği %47‒52 tutarsız, çözünürlükle düzelmiyor.
   Daha iyi bir birleştirici yazmak, daha kötü parçaları birleştirmek olurdu.
2. **Yerel geometri** — kâhin hizalamayla bile RAFT'ın 3‒8 katı kötü.
3. **Derinlikle normalizasyon** — parça ölçeği gürültüsünü hiç azaltmıyor (%42→%43).

`SLAM_TURU.md`'deki "en umut verici aday" notu bu ölçümlerle **geçersizdir** ve orada
düzeltildi.

**Kapı yeniden açılırsa hangi koşulda:** yalnız (a) belirgin daha büyük VRAM'li bir
GPU'da 518 px+ çalıştırılabilirse **ve** (b) parça ölçeği değişim katsayısı %10'un
altına inerse. Ölçüm betikleri duruyor, tekrar koşturmak dakikalar sürer.

## Yeniden koşturmak için

```
.venv\Scripts\python gorev2\tavan_vggt_parca.py
.venv\Scripts\python gorev2\vggt_cozunurluk.py --n-parca 25
.venv\Scripts\python gorev2\vggt_derinlik_normu.py --n-parca 60
```
