# FTR raporu — üretim ve bakım

## Dosyalar

| dosya | ne |
|---|---|
| `Teknofest_FTR_Sablon_2026_d2Drs.docx` | Teknofest'in verdiği şablon — **değiştirilmez** |
| `ftr_grafik.py` | rapora giren grafikleri üretir → `sekiller/` |
| `ftr_uret.py` | şablonu doldurur → `BugBuster_FTR_2026.docx` |
| `BugBuster_FTR_2026.docx` | teslim edilecek rapor (8 sayfa, 8 şekil, 2 tablo) |
| `BugBuster_FTR_2026_onizleme.pdf` | son hâlin PDF önizlemesi (Word'den dışa aktarıldı) |

## Yeniden üretmek

```bash
cd docs
python ftr_grafik.py    # grafikler (sekiller/*.png)
python ftr_uret.py      # BugBuster_FTR_2026.docx
```

`ftr_uret.py` şablonu her seferinde sıfırdan açar; çıktı dosyası elle düzenlenirse
bir sonraki koşuda kaybolur. Metin değişikliği **betikten** yapılmalıdır.

## Doldurulacak yer tutucular

Raporda göze çarpsın diye `[[BÜYÜK HARF]]` biçiminde bırakıldı:

- `[[TAKIM ID]]` — kapak
- `[[AD SOYAD]]` ×2 ve `[[GÖREV]]` ×4 — kapak, üye satırları
- `[[NT]]` ve `[[SIRA]]` — Tablo 1, nesne tespiti puanı ve sırası

## Şablonun bilinmesi gereken üç özelliği

Bunlar deneyerek bulundu; rapora şekil/tablo eklerken tekrar sorun çıkarabilir.

1. **Gövde iki sütunludur** (sütun genişliği 4866 twip ≈ 8,6 cm). Sütuna sığmayan
   tablo sayfadan taşar. `TABLO()` ve `SEKIL()` genişliği buna göre ölçekler.

2. **Tam genişlikli şekiller bölüm adasına konur.** İki sütunun ortasında geçici
   olarak tek sütunlu bir bölüm açılır (`_bolum_sonu(2) … _bolum_sonu(1)`). Önce
   çapalı resim (`<wp:anchor>` + üst-alt kaydırma) denendi; Word şekli sayfa
   başına taşırken altyazı paragrafı yerinde kaldığı için ikisi bambaşka
   sütunlara düşüyordu.

3. **Gövde bölümünün tipi `nextPage`'dir** (kapaktan sonra yeni sayfada başlasın
   diye). Ada açınca gövde parçalara ayrılıyor; bu tip **yalnız ilk parçada**
   kalmalı, kalanlar `continuous` olmalı. Aksi hâlde son parça da yeni sayfaya
   atlıyor ve ondan önceki ada tek başına yarım sayfada kalıyor.

Ek olarak kaynakça paragrafları numaralandırılmış listedir (`numId 2`): `[1]`,
`[2]` … ön ekini Word kendisi basar, metne elle yazılmaz.
