# suni_test — Görev 3 entegrasyonu + tüm sistem doğrulaması

Gerçek yarışma sunucusundan yakalanan kareler üzerinde, **Görev 3'ün ilgili olduğu
referans aralıklarını** (her referansın `frame_start`..`frame_end` penceresini
**başlangıç−3'ten bitiş+3'e** kadar) tam boru hattından (Görev 1+2+3) geçirir.

**Amaç:** entegrasyonun ve tüm sistemin doğru çalıştığını göstermek —
- Referanslar geldiğinde kaydediliyor mu (order 1 ve 2),
- Görev 3 **yalnızca kendi kare aralığında** aranıyor mu (aralık dışında değil),
- Her karede `detect()` üç görevi de hatasız koşturuyor mu.

## Dosyalar
- `suni_test.py` — test scripti.
- `suni_test_sonuc.csv` — 189 karenin kare-kare özeti (frame, aralık, aranan referans, eşleşme, durum).
- `bizim_cevaplar.jsonl` — 189 karenin **sunucuya göndereceğimiz tam payload'ı**
  (detected_objects + detected_translations + detected_undefined_objects).

## Sonuç (kaydedilen koşumdan)
- **189 test karesi, 0 detect() hatası, 0 aralık-filtre yanlışı.**
- Sınır davranışı: `164→165` aranan `[]→[1]` (aralık başladı), `209→210` `[1]→[]` (bitti),
  `524→525` `[]→[2]`. ±3 tampon karelerinde Görev 3 hiç aranmadı.
- Görev 3 aralık içinde **doğru object_id ile** çıktı üretti (40 kare). **Ancak kutular
  hâlâ "tüm-kare" yanlış pozitif** — bu **algoritma/eşleşme kalitesi** meselesi
  (`gorev3/template_matcher.py`), entegrasyon değil.
- Görev 2 konum değerleri bu **izole** replay'de kalibre olmadığından (aralığa "atlanıyor")
  doğru değildir — sistem hatası değil, test kurgusu yan etkisi.

## Tekrar çalıştırmak için
`sistem/sunucu_ciktilari/` (yakalanan gerçek kareler + referans görselleri) gerekir; o klasör
repoya dahil DEĞİL (büyük/geçici). Buradaki `.csv` ve `.jsonl` çıktıları **inceleme için** eklidir.

```
# proje kökünden
.venv\Scripts\python suni_test\suni_test.py
```
