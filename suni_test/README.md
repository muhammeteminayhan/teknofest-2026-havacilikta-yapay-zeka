# suni_test — Görev 3 entegrasyonu + yeni arayüz doğrulaması (çevrimdışı)

Gerçek yarışma sunucusundan yakalanan kareler + referans görselleri üzerinde, **13 Tem resmî
arayüze uyarlanmış** boru hattını (Görev 1+2+3) sunucu olmadan doğrular.

## Dosyalar
- **`entegrasyon_testi.py`** — YENİ akış uçtan uca testi. `register_references(YENİ format,
  ref_image_paths)` → `detect(pred, health, img, active_refs=...)` → çıktı resmî
  **`reference_predictions`** şemasında (`reference`=ref url, `frame`=kare url, bbox).
  Doğrular: (a) referanslar kaydoluyor, (b) Görev 1+2+3 ateşleniyor, (c) Görev 3 **doğru url +
  şema** üretiyor, (d) **pencere dışı temiz** (yanlış pozitif yok).
- `gorev3_gorsel_test.py` — düzeltilmiş eşleştiriciyi görsel doğrulama (biçerdöver + halı saha
  kutuları çizili figür üretir: `gorev3_fix_reference_1/2.png`).

## Çalıştırmak için
`sistem/sunucu_ciktilari/` (yakalanan gerçek kareler + `reference_1/2.png`) gerekir; büyük/geçici
olduğu için repoya dahil DEĞİL.
```
# proje kökünden (çevrimdışı da çalışır)
..\.venv\Scripts\python suni_test\entegrasyon_testi.py
```
**Beklenen çıktı:** `YENI ARAYUZ ENTEGRASYONU CALISIYOR` — Görev 1+2+3 ateşlendi;
`reference_predictions` doğru url+şema; pencere dışı temiz; sema/yanlış-pozitif hatası 0.

## Not
Referansın penceresinin **her** karesinde eşleşmesi beklenmez (görünürlük/güven eşiğine bağlı);
test bunu tespit-farkında değerlendirir: eşleşme olursa url+şema doğru olmalı, pencere dışında
kesinlikle boş olmalı, ve her referans en az bir karede eşleşmeli.
