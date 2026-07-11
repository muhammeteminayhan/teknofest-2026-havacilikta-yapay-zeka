# TEKNOFEST 2026 — Havacılıkta Yapay Zeka (Takım: bugbuster)

İnsansız hava aracının alt-görüş kamerasından gelen kareler üzerinde **üç görevi** aynı
canlı oturumda çalıştıran sistem:

| Görev | İş | Klasör | Omurga |
|---|---|---|---|
| **Görev 1** | Nesne tespiti (taşıt/insan/iniş alanı + durum) | `gorev1/` | YOLOv8s (vanilla) |
| **Görev 2** | GPS'siz pozisyon kestirimi (%40 puan) | `gorev2/` | RAFT + homografi |
| **Görev 3** | Görüntü eşleme (referans nesne bulma) | `gorev3/` | DINOv3 |
| — | Sunucu bağlantısı + üç görevi birleştirme | `sistem/` | — |

Görev 2 bize ait ve tam prova edilmiş; Görev 1 ve 3 arkadaşımızın eğittiği model/koddan
uyarlandı. Hepsi tek `sistem/` oturumunda birleşik çalışır.

> 🏁 **Yarışma günü:** adım adım çalıştırma için → **[YARISMA_KULLANIM.md](YARISMA_KULLANIM.md)**

## Klasör yapısı
```
havacilikta-yz-project/
├── README.md                  ← buradasınız (kurulum + test rehberi + yarışma kontrol listesi)
├── requirements.txt           ← Python bağımlılıkları
├── docs/                      ← yarışma şartnameleri (PDF)
├── THYZ_2026_Ornek_Veri_Seti/ ← örnek veri (2 video + translation CSV + referans nesneler + kalibrasyon)
├── gorev1/                    ← Görev 1 paketi (kendi README'si var)
├── gorev2/                    ← Görev 2 paketi (kendi README'si var)
├── gorev3/                    ← Görev 3 paketi (kendi README'si var)
├── sistem/                    ← canlı bağlantı arayüzü (kendi README'si var)
└── eski-dosyalar/             ← arşiv (eski modeller, fork, yedekler) — silinebilir
```
Her alt klasörün kendi README'si o görevin ayrıntısını ve ayar noktalarını anlatır.

## Kurulum

Sanal ortam proje kökünde hazır: `.venv` (Python 3.13, torch 2.11+cu128, CUDA'lı).
Sıfırdan kurulması gerekirse:
```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

**İki önemli kurulum notu:**
1. **`ultralytics` standart pip'ten kurulur** (Görev 1 vanilla YOLOv8s). Eski çözümdeki
   özel "fork" artık **gerekmez** — sadeleştirildi.
2. **DINOv3 (Görev 3) kapılı bir model.** Bir kez erişim onayı + HuggingFace token ile
   indirilip önbelleğe alınır (`~/.cache/huggingface`); **sonrasında token gerekmez,
   çevrimdışı çalışır.** Bu makinede önbellek zaten dolu. (Token yoksa/erişim yoksa
   `gorev3/` içinde `kind="dinov2"` ile kapılı olmayan yedeğe geçilebilir.)

## Çalıştırma
```powershell
# 1) Hangi sunucu? online = gerçek yarışma, simulator = yerel taklit
Copy-Item sistem\config\.env.online sistem\config\.env

# 2) sistem/ klasöründen başlat
cd sistem
..\.venv\Scripts\python main.py
```

---

# 🧪 Sistemi Kendim Nasıl Test Ederim?

Aşağıdaki testleri **sırayla** yaparsanız, hafiften ağıra doğru her katmanı ayrı ayrı
doğrulamış olursunuz. Hepsi proje kökünden çalıştırılır (aksi belirtilmedikçe).

### Test 1 — Ağ katmanı (en hızlı, model yüklemez)
Gerçek sunucuya erişimi, şemayı ve tahmin gönderimini (201) doğrular.
```powershell
Copy-Item sistem\config\.env.online sistem\config\.env
cd sistem
..\.venv\Scripts\python baglanti_testi.py
```
**Beklenen:** login başarılı, kare/translation şeması yazdırılır, ~10 tahmin `201` döner.
Hata alırsanız sunucu kapalı olabilir (sunucu 09-15 Tem arası açıktı, sonuçlar günlük sıfırlanır).

### Test 2 — Görev 1 tek başına (nesne tespiti)
```powershell
.venv\Scripts\python -c "import sys; sys.path.insert(0,'.'); import cv2; from gorev1 import Gorev1Detector; d=Gorev1Detector(); cap=cv2.VideoCapture(r'THYZ_2026_Ornek_Veri_Seti\THYZ_2026_Ornek_Veri_1.MP4'); cap.set(1,5000); _,f=cap.read(); print('tespit:', d.detect(f))"
```
**Beklenen:** birkaç taşıt sözlüğü (`cls`, `motion_status`, kutu koordinatları).

### Test 3 — Görev 3 tek başına (görüntü eşleme + figür)
```powershell
.venv\Scripts\python -c "import sys; sys.path.insert(0,'.'); import cv2; from gorev3 import Gorev3Matcher; m=Gorev3Matcher(); import glob; [m.register_reference(cv2.imread(r), i) for i,r in enumerate(sorted(glob.glob(r'THYZ_2026_Ornek_Veri_Seti\THYZ_2026_Ornek_Veri_1_Referans_Nesneler\*.*')))]; cap=cv2.VideoCapture(r'THYZ_2026_Ornek_Veri_Seti\THYZ_2026_Ornek_Veri_1.MP4'); cap.set(1,800); _,f=cap.read(); print('eslesme:', m.match(f))"
```
**Beklenen:** eşleşme listesi. (Not: örnek referanslar yakın-çekim olduğundan eşleşme
marjinaldir — `gorev3/README.md`'deki "eşik ayarı" bölümüne bakın.)

### Test 4 — Üç görev birlikte, yerel simülatörle (tam prova, sunucusuz)
Gerçek sunucuya bağlanmadan tüm boru hattını örnek videoyla koşturur.
```powershell
# 1. terminal: yerel simülatörü başlat
.venv\Scripts\python gorev2\src\simulator.py
# 2. terminal: arayüzü simülatöre bağla
Copy-Item sistem\config\.env.simulator sistem\config\.env
cd sistem
..\.venv\Scripts\python main.py
```
**Beklenen:** kareler işlenir, `_logs/`'ta 201'ler, hata yok.

### Test 5 — Çevrimdışı doğrulama (yarışma günü provası)
İnternet olmadan tüm modeller önbellekten yüklenmeli. İnterneti kapatın (Wi-Fi off) veya:
```powershell
$env:HF_HUB_OFFLINE="1"; $env:TRANSFORMERS_OFFLINE="1"
cd sistem
..\.venv\Scripts\python main.py     # (simülatör .env ile)
$env:HF_HUB_OFFLINE=$null; $env:TRANSFORMERS_OFFLINE=$null
```
**Beklenen:** "Gorev 1/3 yuklendi" + kareler işlenir. İnternet gerektirmez.

### Test 6 — Gerçek sunucuda tam koşu
Test 1 geçtiyse: `.env.online` ile `python main.py`. `_logs/` içinde 201'leri izleyin.

---

# ✅ Yarışma Günü Kontrol Listesi

- [ ] **Çevrimdışı hazır mı?** Test 5'i çalıştır. Gerekli önbellekler: DINOv3
      (`~/.cache/huggingface`), RAFT ağırlığı (torchvision cache). *(rembg/u2net **gerekmez** —
      Görev 3 yarışmada rembg kapalı çalışıyor.)* Görev 1 internet istemez.
- [ ] **`sistem/main.py` başlarken "On kontrol"** satırında `cuda: True`, `raft_agirliklari_yerel: True`
      görünmeli (GPU + RAFT hazır).
- [ ] **Doğru `.env`** (`.env.online`) kopyalandı mı?
- [ ] **80/dk gönderim sınırı** — `main.py`'deki RateLimiter açık (aşım → ceza).
- [ ] **Sunucu günlük sıfırlanır** — her gün taze oturum.
- [ ] Görev 3 gerçek referanslar gelince eşik ayarı gerekebilir (`gorev3/README.md`).

## Doğrulanmış kaynak bütçesi (bu makine, RTX 5060 8 GB)
| Senaryo | Kare başı | Tepe VRAM |
|---|---|---|
| Görev 1 + 2 | ~0,7 s | ~1,1 GB |
| Görev 1 + 2 + 3 (aktif) | ~1,0 s | ~1,1 GB |

80/dk sınırı zaten kare başına ~0,75 s taban koyduğundan süre bütçesi rahat. Görev 3
maliyeti referans sayısından bağımsızdır (kare öznitelikleri bir kez çıkarılır).
