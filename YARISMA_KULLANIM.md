# 🏁 Yarışma Günü Kullanım Kılavuzu-

Bu dosya, yarışma günü sistemi **adım adım** nasıl çalıştıracağını anlatır. Sistem üç görevi
(Görev 1 nesne tespiti + Görev 2 konum + Görev 3 görüntü eşleme) tek komutla, otomatik yapar.

Bağlantı katmanı **resmî Takım Bağlantı Arayüzü (13 Tem sürümü)** ile birebir uyumludur:
sunucu kareleri **teker teker** verir, sistem **kaldığı yerden** devam eder, referanslar
**`/reference/` ucundan** otomatik gelir.

> **Bu makinede her şey hazır.** Ek kurulum yok; sadece aşağıdaki adımları izle.

> ⚠️ **EN ÖNEMLİ KURAL:** Sunucu her kareye **yalnızca BİR kez** tahmin kabul eder (tekrar =
> `406`). Bu yüzden **puanlı yarışma oturumunda** sistem hazır değilken `main.py` **çalıştırma** —
> yanlış tahminler kalıcı kaydolur, geri alınamaz. Denemeleri **test oturumunda** yap.

---

## 1) Ön-uçuş kontrol listesi (yarışmadan önce bir kez)

- [ ] **Sanal ortam var:** `havacilikta-yz-project\.venv` (Python 3.13, CUDA'lı torch)
- [ ] **GPU çalışıyor:** `.venv\Scripts\python -c "import torch; print('cuda:', torch.cuda.is_available())"` → `True`
- [ ] **Çevrimdışı önbellekler yerinde:** DINOv3 (`~/.cache/huggingface`), RAFT (torchvision cache).
      *(rembg/u2net gerekmiyor — Görev 3 rembg kapalı.)*
- [ ] **Sunucu bilgileri doğru:** `sistem\config\.env.online`
      → `EVALUATION_SERVER_URL="http://havaciliktayapayzeka.teknofest.org:1025/"` + takım adı + şifre

---

## 2) (Önce) Bağlantı testi — SALT OKUNUR, tahmin GÖNDERMEZ

Sunucuya erişimi ve yeni okuma uçlarını (progress / kare / referans) **hiçbir tahmin
göndermeden** doğrular — puanlı oturumda bile güvenlidir:
```
cd sistem
copy config\.env.online config\.env
..\.venv\Scripts\python baglanti_testi.py
```
**Beklenen:** `OK login` + `progress` (oturum adı / toplam kare / kalınan kare) + `current_frame`
ve `current_translation` şeması + `get_reference_objects` (referans varsa) + `/media` indirme
`HTTP 200`. Hata verirse sunucu kapalı/erişilemez ya da `.env` yanlış demektir.

---

## 3) ASIL KOŞUM (yarışma)

```
cd sistem
copy config\.env.online config\.env       :: online sunucuyu aktif et ("Overwrite?" -> Y)
..\.venv\Scripts\python main.py
```

Sistem otomatik olarak:
1. Sunucuya login olur, **`get_progress` ile nerede kaldığını bulur** (bağlantı koparsa aynı
   komutla **kaldığı kareden devam eder**),
2. Tüm referansları `/reference/`'ten çeker, görüntülerini indirir, Görev 3 eşleştiricisine kaydeder,
3. Kareleri **teker teker** çeker; her karede **Görev 1 + Görev 2 + Görev 3**'ü çalıştırır,
4. Tahmini gönderir (**201** = kabul) ve bir sonraki kareye geçer.

---

## 4) Çalışırken ne izlemelisin

Açılışta:
```
On kontrol: {... 'cuda': True, 'raft_agirliklari_yerel': True, 'onerilen_motor': 'raft'}
Gorev 1 detektoru yuklendi.
Gorev 3 eslestirici yuklendi.
Session: <oturum> — resuming from frame N of 2250
Gorev 3: K referans kaydedildi.
```
Sonra kare kare ilerleyen bir çubuk (`Frames: ... / 2250`).

**Loglar:** `sistem\_logs\<takım>_<tarih>.log` — her karede tek satır özet:
```
[Gorev2 echo]   konum=(x,y,z) m | [Gorev1] N nesne | [Gorev3] M eslesme     (saglikli kare)
[Gorev2 TAHMIN] konum=(x,y,z) m | ...                                        (kesinti karesi)
[Gorev2 yok]    ...                                                          (ceviri gelmedi)
```

---

## 5) Bilmen gerekenler

- **Gönderim hızı:** sistem her kare arasında en az ~0,75 sn bekler → **≤80 gönderim/dk**
  (güvenli taraf). Yarışma 90 dk / 2250 kare için bol zaman payı var.
- **Kaldığı yerden devam:** koşu koparsa aynı `main.py` komutunu tekrar çalıştır — `get_progress`
  ile kalınan kareden sürer (bitmiş kareler tekrar gönderilmez).
- **Görev 2 ilk 450 kare:** sağlıklı GT ile ölçek + koordinat hizalaması otomatik kurulur;
  sonra kesinti karelerinde (health=0) kestirim üretilir. Kamera: 1080p parametreleri otomatik.
- **İnternetsiz de çalışır:** modeller önbellekte (DINOv3 + RAFT).

---

## 6) Sorun giderme

| Belirti | Sebep / Çözüm |
|---|---|
| `Login Failed` / `progress None` | Sunucu kapalı ya da `.env.online` yanlış. Bilgileri ve sunucunun açık olduğunu kontrol et. |
| `No active session found` | Sunucu henüz oturum açmamış. Duyurulan saati bekle. |
| `Copy-Item` tanınmıyor | cmd'de `copy` kullan (komutlarda zaten `copy` var). |
| HTTP **406** | O kareye zaten tahmin gönderilmiş. Sistem stuck-sayaçla korunur; 5 kez ilerlemezse durur. |
| HTTP **403** | Hız limiti aşıldı (bu tahmin kaydedilmez). Hız zaten ≤80/dk; tekrarlarsa `MIN_FRAME_INTERVAL`'i artır. |
| `cuda: False` | GPU görünmüyor; Görev 2 otomatik LK yedeğine düşer (biraz düşük doğruluk) ama çalışır. |
| Görev 3 `0 referans` | Test oturumunda referans olmayabilir (normal). Gerçek yarışmada `/reference/`'ten gelince otomatik aranır. |

---

## Çevrimdışı model kontrolü (yarışma öncesi prova)

İnternet olmadan üç modelin de önbellekten yüklendiğini doğrula:
```
$env:HF_HUB_OFFLINE="1"; $env:TRANSFORMERS_OFFLINE="1"
cd sistem
..\.venv\Scripts\python -c "import sys; sys.path.insert(0,'.'); from src.object_detection_model import ObjectDetectionModel; m=ObjectDetectionModel('http://x/'); print('gorev1:', m.gorev1 is not None, '| gorev3:', m.gorev3 is not None)"
$env:HF_HUB_OFFLINE=$null; $env:TRANSFORMERS_OFFLINE=$null
```
**Beklenen:** `cuda: True`, "Gorev 1/3 yuklendi", `gorev1: True | gorev3: True`.
(Tam boru hattı doğrulaması gerçek test oturumuyla yapılır — Bölüm 2 bağlantı testi + Bölüm 3 koşum.)

Daha fazla ayrıntı: kök [README.md](README.md) ve alt klasör README'leri.
