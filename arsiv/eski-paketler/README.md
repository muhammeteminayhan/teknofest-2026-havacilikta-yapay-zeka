# Eski paketler — arşiv

Bu klasör, `final-diyarbakir` öncesindeki paket sürümlerinin **kodunu** saklar.
Paketlerin kendisi 17 Eylül 2026'da bilgisayardan kaldırıldı; geri gerekirse
buradan çıkarılabilir.

**Yarışma günü çalıştırılacak paket burada değildir** — o `final-diyarbakir`
klasörüdür ve bu arşivin hiçbir parçası onun yerine kullanılamaz.

## İçerik

| Klasör | Tarih | Neydi | Neden emekli |
|---|---|---|---|
| `bugbuster_5202745_teslim` | 29 Tem 2026 | TEKNOFEST'e **fiilen teslim edilen** paket | Teslim tamamlandı; sonraki turlar sistemi geliştirdi |
| `TESLIM_PAKETI` | 29 Tem 2026 | Teslim paketinin ağırlıklı, çalıştırılabilir kopyası | Görev 2 katmanı 28 Tem'de donmuş; Ağustos'taki düzeltmeleri almadı (aşağıya bak) |
| `final-v1` | 13 Ağu 2026 | Küçük koşu/denetim betikleri (`on_ucus.py`, `prova.py`, `konum/`) | `final-diyarbakir` içindeki `dogrula.py` + `prova.py` bunların yerini aldı |
| `final-v1.0` | 13 Ağu 2026 | İlk tam paket kuşağı (kurulum + model yerleştirme) | `final-diyarbakir` (10 Eyl) bunun halefi |

## Neden arşivlendi, neden kaldırıldı

`TESLIM_PAKETI`'nin Görev 2 katmanı **28–29 Temmuz** tarihliydi. Projedeki
karşılıkları 7–13 Ağustos'ta güncellendi ve bu güncellemeler pakete hiç
kopyalanmadı. Aradaki farklar:

| | Eski paket | Güncel (`final-diyarbakir`) |
|---|---|---|
| z işareti kuralı | tek koşullu — düz uçuşta gürültüyü kanıt sayıyordu | iki koşullu kanıt kapısı |
| `min_inlier_ratio` | 0,40 | 0,30 |
| `residual_gate` | 0,50 | 0,30 |
| Kamera profilleri | 3 profil; bilinmeyen çözünürlükte `ValueError` | 4 profil (3840×2160 dâhil) |

Yani bu paketler yalnız eski değil, **ölçülmüş biçimde daha kötü**. Arşiv değeri
tarihseldir: hangi sürümle neyin teslim edildiğini gösterir.

## Çıkarılanlar

Arşiv yalnız kaynak kodu içerir. Şunlar bilerek alınmadı:

- `.venv/`, `__pycache__/` — yeniden üretilebilir
- Model ağırlıkları (`.pt`, `.pth`, `.safetensors`) — GitHub 100 MB sınırı;
  çalışan ağırlık `final-diyarbakir/gorev1/agirliklar/best.pt` içindedir
- `modeller/` önbelleği — `modelleri_yerlestir.py` yeniden kurar
- `.env` — takım şifresi; arşivde işlevi yok (`example.env` bırakıldı)
- 5 MB üstü dosyalar

## Sürüm zinciri

```
bugbuster_5202745_teslim  (29 Tem, teslim)
        │
   TESLIM_PAKETI          (29 Tem, ağırlıklı kopya)
        │
   final-v1 / final-v1.0  (13 Ağu, ilk paket kuşağı)
        │
   final-diyarbakir       (10 Eyl, YARIŞMADA KULLANILAN)
```
