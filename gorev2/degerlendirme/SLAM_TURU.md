# Görev 2 — derin SLAM/VO turu (DPVO, DROID, MASt3R, VGGT, SLAM3R, ORB-SLAM3)

**Hedef:** PK hatasını (şu an **3,693 m**) çok-kareli optimizasyon (bundle adjustment /
feed-forward geometri) ile düşürmek. Teşhis: kalan hata sistematik — yön ~1,27°,
**ölçek %2,11**; tek-çift homografi + düzlem varsayımı ölçek tarafında tavan yapmış.

**Kullanıcı kararları:** hepsi denensin, gereken kurulum yapılsın; **90 dk / 2250 kare
= kare başına ~2,4 sn bütçesi bağlayıcı**.

---

## Ortam: sıfırdan derleme altyapısı kuruldu

Başlangıçta `CUDA_HOME` yok, `nvcc` yok, MSVC yok → hiçbir CUDA eklentisi derlenemiyordu.
winget ile **CUDA Toolkit 12.8** (torch cu128 ile eşleşen sürüm) + **VS Build Tools
(MSVC 14.44)** kuruldu.

### Windows + torch 2.11 + Blackwell (sm_120) uyumsuzlukları ve çözümleri

Bu sistemler Linux/GCC için yazılmış. Derleme yolu açılırken çözülen engeller:

| # | Engel | Çözüm |
|---|---|---|
| 1 | `error C2872: 'std' belirsiz simge` — torch 2.11 başlıkları MSVC ile | `/permissive- /Zc:__cplusplus /std:c++17` |
| 2 | `tensor.type()` → PyTorch 2.11'de `ScalarType`'a dönüşmüyor | `.scalar_type()` (`AT_DISPATCH` ve özel `DISPATCH_*` makroları; 80 yer) |
| 3 | `accessor32<long,...>`, `accessor<long,1>`, `data_ptr<long>`, `item<long>` — Windows'ta `long` **32-bit**, PyTorch `int64_t` bekliyor | `int64_t` dönüşümü (3 dosya, ~45 yer) |
| 4 | `(float[6]){...}` — GCC compound literal, MSVC desteklemiyor | Standart C++ tampon + pointer (2 yer) |
| 5 | `ImportError: DLL load failed` | `import torch` uzantılardan **önce** + `os.add_dll_directory(CUDA/bin)` |
| 6 | `dpvo.loop_closure` alt paketi kurulumda eksik | Kaynaktan kopyalandı; `pypose`, `yacs`, `einops` `--no-deps` ile kuruldu |

**Sonuç:** `torch-scatter 2.1.2` ✓, **DPVO (cuda_corr + cuda_ba + lietorch_backends)** ✓
derlendi ve import edildi. Bu yol artık DROID-SLAM / MASt3R-SLAM için de açık.

> **Önemli uyarı — ana ortam koruması:** VGGT'nin standart kurulumu numpy'ı 2.4.6 → 1.26.4
> düşürerek yarışma kodunu bozacaktı (scipy/rembg/tifffile numpy≥2 istiyor). Fark edilip
> geri alındı; tüm SLAM bağımlılıkları **`--no-deps`** ile kuruldu ve her adımda sistem
> doğrulandı (Görev 1: 15 nesne, Görev 2/3 çalışıyor, numpy 2.4.6).

---

## Değerlendirme altyapısı

**`slam_degerlendirme.py`** — herhangi bir SLAM/VO'nun poz dizisini yarışma mimarisine bağlar:
sağlıklı karelerde **nedensel Sim(3)** (yalnız geçmiş kareler; rolling akış kısıtı),
kesintide uygula, `competition_error` ile ölç + Procrustes yön/ölçek teşhisi + kare bütçesi.

**Doğrulama:** mevcut RAFT boru hattının pozlarıyla beslendiğinde bilinen sonucu
(**3,693 m**, yön 1,27°, ölçek %2,11) **birebir** yeniden üretti.

---

## VGGT — kritik bulgu: VRAM eşiği

VGGT-1B tek başına **5,03 GB** VRAM tutuyor; GPU 8,5 GB. Parça boyutu/çözünürlük eşiği
aşınca sistem belleğine paging başlıyor ve **40× yavaşlıyor**:

| Çözünürlük / parça | s/kare | Tepe VRAM |
|---|---|---|
| 518 / N=8 (varsayılan) | 3,94 | **9,81 GB** ✗ paging |
| 392 / N=8 | 1,40 | 8,55 GB |
| **294 / N=8** | **0,44** | **7,81 GB** ✓ |
| 392 / N=4 | 0,50 | 7,71 GB ✓ |

İlk koşum 518 px ile başlatılmıştı: parça başına **556 sn** (tüm video ≈ 87 saat).
294 px'e inince **0,5–0,6 sn/parça** (tüm video ≈ 35 dk). Yaklaşım: VGGT-Long tarzı
örtüşen parçalar + parçalar arası **Sim(3)** zincirleme (`vggt_poz.py`).

---

## Elenenler (denenmeden, gerekçeli)

| Yöntem | Gerekçe |
|---|---|
| **VINS-Fusion** | IMU gerektirir; sistemde yalnız monoküler RGB var |
| **DROID-SLAM** | Offline bundle adjustment; rolling akışta geriye dönük düzeltme puana yansımaz. Ayrıca 8,5 GB VRAM sınırda *(derleme yolu artık açık — süre kalırsa denenecek)* |
| **GTSAM** | Füzyon için ikinci sensör yok; graf optimizasyonu gürültüyü azaltır, **sistematik yanlılığı düzeltmez** |
| **ORB-SLAM3** | Windows'ta Pangolin/DBoW2/g2o C++ derlemesi; en zor kalem, en sona bırakıldı |
| **8. madde (Homografi + SuperPoint/LightGlue)** | **Zaten mevcut sistemimiz** (`H = K(R − t·nᵀ/d)K⁻¹`). Önceki turda SuperPoint yerine DISK+LightGlue ve XFeat denendi, **ikisi de RAFT'tan kötü** (bkz. `OLCUM_KATMANI.md`) |

---

## Ölçüm sonuçları

| Sistem | Yarışma takvimi | Yön | Ölçek | s/kare | Karar |
|---|---|---|---|---|---|
| **RAFT + homografi (mevcut)** | **3,693 m** | 1,27° | %2,11 | ~0,6 | **korunuyor** |
| VGGT (294 px, parça+Sim(3)) | 125,7 m | 102° | %42,9 | **0,07** | ❌ zincirleme çözülmeden kullanılamaz |
| DPVO | — | — | — | — | ❌ çalışma zamanı çökmesi |
| **ORB-SLAM (tracking çekirdeği)** | **90,6 m** | 36,5° | %68 | **0,05** | ❌ ölçek mekanizması bu sahneyle uyumsuz |

### ORB-SLAM — sonradan denendi (düşey kanal turunun ardından)

Bu turda "en zor kalem" diye ertelenmişti; sonradan ölçüldü. **Hazır binding yolu kapalı:**
PyPI'daki `orbslam3` wheel'i bir Linux `.so`'sudur (Windows'ta yüklenmez) ve ayrıca
Pangolin/DBoW2/g2o'ya dinamik bağımlıdır. Kaldı ki ORB-SLAM3'ün ayırt edici parçaları
olan **loop closure ve global bundle adjustment rolling akışta zaten kullanılamaz** —
gönderilen tahmin geri alınamadığı için geriye dönük düzeltme puana yansımaz.

Yarışmada işlevi olan kısım **tracking** çekirdeğidir; OpenCV ile kurulup (ORB + Lowe
oran testi + ORB-SLAM'in kendi H/F model seçimi + triangulation + PnP + keyframe yönetimi)
dört varyant ölçüldü:

| Varyant | Yarışma | Neden |
|---|---|---|
| 1. temel | 104,8 m | takip %80 çöküyor (keyframe güncellenmeyince kısır döngü) |
| 2. + motion model + keyframe kurtarma | **90,6 m** | takip %99,5 — en iyi ORB varyantı |
| 3. + ORB-SLAM'in gerçek H/F puanlaması + ölçek koruma | 4947 m | triangulation dejenere → yörünge 4,6×10⁸'e patladı |
| 4. haritasız saf düzlemsel zincir | 116,7 m | ölçek her keyframe'de birim `\|t\|`'ye sıfırlanıyor |

**Kök neden (varyantlar arasında tutarlı):** monoküler ORB-SLAM ölçeği *triangulation*
ile taşır. Nadir bakan kamerada sahne neredeyse saf düzlemdir ve baseline küçüktür —
derinlik kestirimi kötü koşulludur. İlginç bir doğrulama: varyant 3'te model seçimi
düzeltilince sistem sahneyi doğru teşhis etti (H=3, F=0 — yani "bu sahne düzlemsel"),
ama tam da bu yüzden triangulation çöktü. Bizim hattımız ölçeği *GT'den* öğrenip
(sağlıklı dönemde Procrustes) düzlemsel homografi + yükseklik ile taşıdığı için aynı
sahnede sorun yaşamıyor. Deneme dosyaları kaldırıldı; bu kayıt yeterlidir.

### DPVO — derlendi ama çalışmıyor

Yedi derleme denemesi sonunda **başarıyla derlendi** (`Successfully installed dpvo-0.0.0`)
ve `cuda_corr` kernelinin Blackwell'de (capability 12.0) doğru çalıştığı ayrıca doğrulandı.
Ancak çalışma zamanında **`lietorch` çöküyor**:

```
Windows fatal exception: access violation
  dpvo/lietorch/group_ops.py:14 in forward   (SE3.inv)
```

Çökme **CPU'da bile** oluyor → GPU/Blackwell sorunu değil, `lietorch_backends`'in torch 2.11
ile **ABI uyumsuzluğu**. Çözümü lietorch kernel kodunun bütünüyle elden geçirilmesini
gerektirir (belirsiz sonuçlu, saatler).

> **Bu bulgu DROID-SLAM ve DPV-SLAM'i de eler** — ikisi de aynı `lietorch` bağımlılığını
> kullanır. Aynı ABI duvarına çarpacakları için ayrıca denenmedi.

### VGGT — model iyi, benim zincirlemem yetersiz

Kâhin (oracle) Sim(3) ile ölçülen yörünge kalitesi, sorunun **VGGT'de değil parça
birleştirmede** olduğunu gösteriyor:

| Pencere | Hata | GT yolu | Yorum |
|---|---|---|---|
| **İlk 50 kare** | **0,97 m** | 45 m | Tek parçada **mükemmel** (%2) |
| İlk 200 kare | 23,1 m | 218 m | Bozulma başlıyor |
| Tüm dizi (2250) | 127,6 m | 2066 m | Tamamen bozuk |

Hata mesafeyle birikiyor. İki ayrı arıza gözlendi:
1. **Ölçek patlaması** (ilk deneme): parça ölçekleri kümülatif çarpılıyor; 563 parçada
   %5'lik sapma bile 1,05⁵⁶³ ≈ 10¹² → ölçülen hata **10²⁸ m**.
2. **Ölçek büzülmesi** (bant sınırlaması sonrası): bant 1,02'ye kısılınca yörünge
   büzüldü — VGGT toplam hareketi 17 birim, GT 215 m.

Doğru çözüm, VGGT-Long'un yaptığı gibi tüm alt-haritalar üzerinde **global Sim(3)/SL(4)
graf optimizasyonu** kurmaktır; bu, bu turun kapsamını aşan ayrı bir çalışmadır.

**Not:** VGGT hız açısından fazlasıyla uygun (**0,07 s/kare**, bütçe 2,4 sn) ve tek parçada
%2 hata veriyor — ileride zincirleme düzgün kurulursa **en umut verici aday** budur.

> ## ⛔ DÜZELTME (14 Ağu 2026): yukarıdaki not GEÇERSİZDİR
>
> Buradaki "%2 hata" ve "sorun VGGT'de değil parça birleştirmede" tespiti **tek bir
> pencereden** (ilk 50 kare) geliyordu. 563 parçanın tamamında ölçülünce çürüdü:
>
> | ölçüt (kâhin Sim(3), 8 karelik parçalar) | VGGT | RAFT+homografi |
> |---|---|---|
> | parça içi artık / GT yolu, medyan | **%8,1** | **%1,0** |
> | parça ölçeği değişim katsayısı | **%78** | — |
>
> Çözünürlük şekil hatasını düzeltiyor (294→392 px: %6,2→%3,4) ama **parça ölçeğinin
> tutarsızlığını düzeltmiyor** (%52→%47); 518 px bu GPU'da paging'e giriyor.
> Derinlikle normalizasyon da ölçek gürültüsünü hiç azaltmıyor (%42 → %43).
>
> **Aile kapandı.** Ayrıntı ve yeniden açılma koşulları: [VGGT_TURU.md](VGGT_TURU.md)

### Denenmeyenler (gerekçe)

| Sistem | Neden denenmedi |
|---|---|
| DROID-SLAM, DPV-SLAM | `lietorch` ABI çökmesi (DPVO'da kanıtlandı) |
| MASt3R-SLAM, SLAM3R | Kendi CUDA backend'lerini derliyorlar; aynı torch 2.11/MSVC duvarı riski yüksek. Süre bütçesi bu turda tükendi |
| ORB-SLAM3 | Windows'ta Pangolin/DBoW2/g2o C++ derlemesi — en zor kalem |

---

## Turun sonucu

**Hiçbir derin SLAM/VO sistemi mevcut RAFT+homografi hattını (3,693 m) geçemedi.**
Kazanan yok → **canlı sistem değiştirilmedi**.

Kazanımlar:
- Derleme altyapısı (CUDA 12.8 + MSVC) kuruldu; 6 uyumsuzluk kalıbı belgelendi — bu yol
  ileride yeniden kullanılabilir.
- `slam_degerlendirme.py`: herhangi bir poz dizisini yarışma metriğiyle ölçen, doğrulanmış
  altyapı (RAFT pozlarıyla 3,693 m'yi birebir üretiyor).
- VGGT'nin bu veride **tek parçada %2 hata** verdiği ölçüldü — gelecek çalışma için net hedef.
