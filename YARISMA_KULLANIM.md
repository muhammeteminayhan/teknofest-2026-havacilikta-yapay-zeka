# 🏁 Yarışma Günü Kullanım Kılavuzu

Bu dosya, yarışma günü sistemi **adım adım** nasıl çalıştıracağını anlatır. Sistem üç görevi
(Görev 1 nesne tespiti + Görev 2 konum + Görev 3 görüntü eşleme) tek komutla, otomatik yapar.

> **Bu makinede her şey hazır.** Ek kurulum yok; sadece aşağıdaki adımları izle.

---

## 1) Ön-uçuş kontrol listesi (yarışmadan önce bir kez)

- [ ] **Sanal ortam var:** `havacilikta-yz-project\.venv` (Python 3.13, CUDA'lı torch)
- [ ] **GPU çalışıyor:** aşağıdaki komut `cuda: True` yazmalı
  ```
  .venv\Scripts\python -c "import torch; print('cuda:', torch.cuda.is_available())"
  ```
- [ ] **Çevrimdışı önbellekler yerinde** (internetsiz de çalışsın diye):
  - DINOv3 (Görev 3): `C:\Users\<sen>\.cache\huggingface\hub\models--facebook--dinov3-vits16-...`
  - RAFT (Görev 2): torchvision önbelleği (otomatik)
  - *(Not: rembg/u2net artık gerekmiyor — Görev 3 rembg kapalı çalışıyor.)*
- [ ] **Sunucu bilgileri doğru:** `sistem\config\.env.online` içinde takım adı + şifre + URL

---

## 2) (Önce) Hızlı bağlantı testi — model yüklemeden

Sunucuya erişimi + tahmin gönderimini saniyeler içinde doğrular:
```
cd sistem
copy config\.env.online config\.env
..\.venv\Scripts\python baglanti_testi.py
```
**Beklenen:** login başarılı + birkaç örnek tahmin **201** döner. (Sunucu kapalıysa bağlantı
hatası verir — normal, sunucunun açılmasını bekle.)

---

## 3) ASIL KOŞUM (yarışma)

```
cd sistem
copy config\.env.online config\.env       :: online sunucuyu aktif et ("Overwrite?" -> Y)
..\.venv\Scripts\python main.py
```

Bu kadar. Sistem otomatik olarak:
1. Sunucuya login olur,
2. Kareleri tek tek çeker,
3. Her karede **Görev 1 + Görev 2 + Görev 3**'ü çalıştırır,
4. Tahmini sunucuya gönderir (**201** = kabul),
5. Görev 3 referansları `references` anahtarından **otomatik** kaydolur (elle bir şey yapmana gerek yok).

---

## 4) Çalışırken ne izlemelisin

Açılışta şunları görmelisin:
```
On kontrol: {... 'cuda': True, 'raft_agirliklari_yerel': True, 'onerilen_motor': 'raft'}
Gorev 1 detektoru yuklendi.
Gorev 3 eslestirici yuklendi (rembg kapali, tiling acik).
```
Sonra her kare için ilerleme çubuğu + sunucudan **201**.

**Loglar:** `sistem\_logs\<takım>_<tarih>.log` — her karede tek satır özet:
```
[Gorev2 echo]   konum=(x,y,z) m | [Gorev1] N nesne | [Gorev3] M eslesme     (saglikli kare)
[Gorev2 TAHMIN] konum=(x,y,z) m | ...                                        (kesinti karesi)
```
- `echo` = sağlıklı kare (gelen konumu aynen gönderiyor, 0 hata)
- `TAHMIN` = kesinti karesi (RAFT ile konum kestiriyor)

---

## 5) Bilmen gerekenler

- **80 tahmin/dk sınırı:** sistem otomatik yönetiyor (RateLimiter). Aşım = ceza, ama sen bir şey yapmıyorsun.
- **Sunucu günlük sıfırlanır:** her gün taze oturum.
- **İnternetsiz de çalışır:** modeller önbellekte. İstersen doğrulamak için Wi-Fi kapalıyken çalıştır.
- **Görev sürümü:** Görev 3 = DINOv3 + **tiling** (minik nesneleri de yakalar) + **rembg kapalı**
  (yanlış pozitif az). Görev 2 = RAFT+homografi (en güçlü, ~3.5 m). Görev 1 = YOLOv8s.

---

## 6) Sorun giderme

| Belirti | Sebep / Çözüm |
|---|---|
| `Login Failed` / bağlantı hatası | Sunucu kapalı ya da `.env.online` yanlış. Bilgileri kontrol et, sunucunun açık olduğundan emin ol. |
| `Copy-Item` tanınmıyor | cmd kullanıyorsun; `Copy-Item` yerine `copy` yaz (yukarıdaki komutlarda `copy` var). |
| HTTP **406** | O kareye zaten tahmin gönderilmiş (tekrar). Sistem otomatik geçer, sorun değil. |
| `cuda: False` | GPU görünmüyor; Görev 2 otomatik LK yedeğine düşer (biraz daha düşük doğruluk) ama sistem çalışır. |
| Görev 3 `0 eslesme` | Sunucu o oturumda referans göndermediyse normal (test oturumunda `references=[]` gelir). Gerçek referans gelince otomatik aranır. |

---

## Yerel test (yarışma öncesi prova, sunucusuz)

Gerçek sunucu olmadan tüm sistemi örnek videoyla denemek için:
```
:: 1. terminal — yerel simülatör (2250 kare sunar)
.venv\Scripts\python gorev2\src\simulator.py
:: 2. terminal — sistemi simülatöre bağla
cd sistem
copy config\.env.simulator config\.env
..\.venv\Scripts\python main.py
```

Daha fazla ayrıntı: kök [README.md](README.md) ve alt klasör README'leri (`gorev1/`, `gorev2/`, `gorev3/`, `sistem/`).
