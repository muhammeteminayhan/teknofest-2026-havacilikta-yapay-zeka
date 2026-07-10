# Görev 3 — Görüntü Eşleme

Oturum başında **referans nesneler** verilir; her karede bu nesnelerin görüntüde olup
olmadığı bulunur ve konumları (kutu) raporlanır. Zero-shot: nesneler modelde eğitilmiş
değildir, **görsel benzerlikle** eşleştirilir.

## Nasıl çalışır

**DINO patch özellikleri + FPN çok-ölçek kosinüs benzerliği** (`template_matcher.py`).
Referansın DINO **CLS token**'ı, karenin her **patch token**'ıyla kosinüs benzerliğine
sokulur → ısı haritası → morfolojik işlem + eşikle kutu çıkarılır. Üç ölçekte (224/168/112)
yapılıp birleştirilir (max füzyon).

### Omurga: DINOv3 (fork'suz)
- Varsayılan omurga **`facebook/dinov3-vits16-pretrain-lvd1689m`** (HuggingFace, patch16).
  Bu model **kapılı (gated)** — bir kez erişim onayı + token ile indirilip önbelleğe
  alınır, **sonrasında token gerekmeden çevrimdışı çalışır** (`~/.cache/huggingface`).
- `dino_backbone.py` fork'un ağır projeksiyon katmanlarını (P3/P4/P5) **atlar** — eşleme
  için yalnızca ham `last_cls_token` / `last_patch_tokens` / `mask_dim` gerekir.
- **Yedek omurga "dinov2"** (torch.hub `dinov2_vits14_reg`, patch14, kapılı değil):
  `Gorev3Matcher(kind="dinov2")` ile tek satırda geçilir. DINOv3 önbelleği yoksa güvenli liman.

## Dosyalar
```
gorev3/
├── __init__.py            # Gorev3Matcher'ı dışa açar
├── eslestirici.py         # Gorev3Matcher: referans kaydı + per-frame match (sarmalayıcı)
├── dino_backbone.py       # minimal DINOv3/DINOv2 omurgası (fork'suz, ~40 satır)
├── template_matcher.py    # eşleme çekirdeği (arkadaşın dosyası; kare-özniteliği önbelleği eklendi)
├── template_config.yaml   # eşik/ölçek ayarları
└── reports/               # görsel eşleme çıktıları (ısı haritası figürleri)
```

## Kullanım
```python
from gorev3 import Gorev3Matcher
m = Gorev3Matcher(kind="dinov3")
m.register_reference(ref_bgr, object_id=0)     # BGR görüntü, obje kimliği
eslesmeler = m.match(frame_bgr)                # [{object_id, top_left_x/y, bottom_right_x/y}, ...]
```
Sunucudan gelen referanslar için: `register_from_server(references, download)` —
frames-JSON'daki `references` listesini savunmacı okur (url + id alanlarını arar).

## Önemli: eşleşme kalitesi ve eşik ayarı

Örnek referans nesneler (`THYZ_2026_Ornek_Veri_Seti/..._Referans_Nesneler/`) **yer
seviyesi yakın-çekim** (ör. 3019×2145); hava görüntüsünde nesne çok küçük kaldığından
ısı haritaları **dağınık** çıkıyor ve eşleşmeler marjinal (güven ~0,25-0,4). Bu bir hata
değil, **ölçek uyuşmazlığı** — sistem doğru çalışıyor.

**Gerçek yarışma referansları geldiğinde** `template_config.yaml` içindeki eşikleri
ayarlayın:
- `matching.min_confidence` (0.25) — eşleşme kabul eşiği. **Yanlış pozitif skoru düşürür**;
  referanslar zayıf eşleşiyorsa düşürün, çok gürültü varsa yükseltin.
- `matching.threshold_ratio` (0.60), `matching.noise_floor` (0.15) — kutu sınırı.
- `fpn.levels` (224/168/112) — nesne ölçeğine göre. Referans hava-görüşüne yakınsa
  daha küçük ölçekler işe yarar.

Görsel kontrol: `template_matcher.py`'nin `match(rgb, save_path=...)` çağrısı ısı haritası
figürü üretir (`reports/`'a bakın).

## Performans
- Kare başı ~0,7 s (referans sayısından **bağımsız** — kare öznitelikleri ölçek başına bir
  kez çıkarılıp tüm referanslarda yeniden kullanılır; bu optimizasyon `match()` içinde).
- VRAM ~0,1 GB (küçük ViT-S omurga).
