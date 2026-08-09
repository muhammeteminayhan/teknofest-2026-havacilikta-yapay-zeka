# Görev 3 — uzamsal patch kanalı turu (28 Tem 2026)

> ## ⛔ SONUÇ: BU TURDAKİ DEĞİŞİKLİKLER **GERİ ALINDI** — sistem eski haline döndürüldü
>
> `template_matcher.py` ve `template_config.yaml` `git checkout` ile özgün haline
> döndürüldü; doğrulandı (recall %78,6 / kutu 371×371 — turdan önceki değerlerle birebir).
> Yarışma sistemi bu turdan **etkilenmemiştir**.
>
> **Neden geri alındı.** Toplu sayılar iyi görünüyordu (recall %76,1 → %93,0, kutu alanı
> 332×332 → 179×179) ama görsel denetim, kazancın bir kısmının **sahte** olduğunu gösterdi:
>
> * **ref5** (referans: mavi mozaikli bina) — yeni ayar kare 1390'da yolun ortasına
>   **30×42 px**'lik bir kutu çiziyordu; eski sistem orada hiç tespit üretmiyordu. Bu
>   referansın "%100 recall"ü büyük ölçüde bu tür sahte kutulardan geliyordu (kutu alanı
>   medyanı 4593 px² = 68×68, oysa referans 337×790 px'lik bir bina).
> * **ref4** (çatıdaki yıldız deseni) — eski sistem yıldızı tam sarıyordu (123×127); yeni
>   sistem yanındaki kubbeyi de içine alıyordu (198–220×127…158). GT yıldızın kendisiyse
>   IoU ≈ 0,45 olur ve **mAP@0,5 eşiğini geçemez**.
>
> Yani patch kanalı büyük nesnelerde (spor sahası, amfi) kutuyu iyileştiriyor ama küçük
> nesnelerde komşu dokuya taşıyor ve zayıf eşleşmelerde sahte kutu üretiyor. Kullanılan
> recall metriği "tespit **var mı**" ölçtüğü için bu bozulmayı **ödüllendiriyordu** —
> ölçütün kendisi yanıltıcıydı.
>
> **Kalıcı kazanım:** `test_yatagi.py` ve buradaki ölçümler duruyor. Bir sonraki denemede
> önce ölçütün düzeltilmesi gerekir (ör. referans en-boy oranı / makul ölçek aralığı
> kontrolü, ve "doğru yerde mi" sorusunu sorabilen bir değerlendirme).
>
> Aşağıdaki bölümler o turda ne ölçüldüğünün kaydıdır; **uygulanan ayar değildir.**

**Bağlam.** RNT puanlı oturumda **50,97 / 8 takım içinde 2.** — en güçlü görevimiz ve final
formülünde %25 ağırlığı var. Şartname 9.3: puanlama **mAP**, yani hem nesneyi bulmak hem
**kutu isabeti** (IoU) sayılıyor.

## Teşhis: kullanılmayan bir kanal

`TemplateMatcher` referans kaydında hem `mean_cls` hem `mean_patches` hesaplıyordu, ama
`_fpn_heatmap` **yalnızca `mean_cls`** kullanıyordu. CLS token tek bir global vektördür:
"bu bölge referansa benziyor mu" sorusuna cevap verir, ama nesnenin *nerede bittiğini*
söylemez. Ölçülen sonucu buydu — **kutu alanı medyanı 371×371 px**.

Eklenen: referansın patch matrisi ↔ karenin patch matrisi kosinüs benzerliği; her kare
patch'i için **en iyi eşleşen referans patch'i** (max havuzlama). Son benzerlik
`(1-w)·CLS + w·patch`. Kod: `template_matcher.py:_fpn_heatmap`, ayar
`matching.patch_weight`.

## Ölçüm — `test_yatagi.py`

GT kutu yok, ama sunucunun referans kayıtlarında **kare aralığı** (`frame_start`–`frame_end`)
var: o aralıkta nesne sahnededir. Test yatağı bunu zayıf-etiket olarak kullanıp recall,
aralık-dışı tespit, kutu alanı, kutu sıçraması ve süreyi ölçer.

**Yatak doğrulaması:** mevcut sistemle koşulunca recall %78,6 ve 970 ms/kare üretti;
puanlı oturumdaki gerçek değerler %77 ve ~1014 ms. Yatak sistemi doğru temsil ediyor.

### patch_weight taraması (7 referans × 12 kare)

| w | recall | kutu alanı | süre |
|---|---|---|---|
| 0,0 (eski) | %78,6 | 371×371 | 970 ms |
| 0,2 / 0,25 | %85,7 | 304×304 / 295×295 | — |
| **0,3 (seçildi)** | **%86,9** | **287×287** | 894 ms |
| 0,35 / 0,4 | %85,7 | 284×284 / 282×282 | — |
| 0,5 | %81,0 | 239×239 | 929 ms |
| 0,7 | %65,5 | 235×235 | 922 ms |

Ağırlık arttıkça kutu daralıyor ama recall çöküyor; 0,3 hem recall hem IoU yönünde kazanç
veren nokta. `patch_topk` (referansın arka plan patch'lerini eleme) denendi — kutu büyüdü,
faydası yok.

### Eşiklerin yeni zeminde yeniden taranması

Görev 2'de öğrenilen ders burada da uygulandı: *kaynak düzelince eski optimumlar geçersiz
olabilir.*

| ayar | recall | kutu | aralık dışı |
|---|---|---|---|
| k_std 3,0 (eski) | %86,9 | 287×287 | 9/14 |
| **k_std 2,5 (seçildi)** | **%95,2** | 268×268 | 12/14 |
| k_std 2,0 | %100 | 238×238 | 13/14 |
| box_k_std 2,5 | %86,9 | **198×198** | 9/14 |
| reference.augment=true | %86,9 | 287×287 | — |

## Seçilen ayar ve tam koşu (tek değişkenli karşılaştırma)

`patch_weight = 0,3` + `k_std = 2,5` (diğerleri değişmedi). Aynı 686 aktif karede iki ayar:

| | ESKİ (`patch=0`, `k_std=3,0`) | **SEÇİLEN** (`patch=0,3`, `k_std=2,5`) |
|---|---|---|
| **recall** | **522**/686 = %76,1 | **638**/686 = **%93,0** |
| kutu alanı medyanı | 332×332 px | **241×241 px** (alan −%47) |
| aralık dışı tespit | 6/14 | 12/14 |
| süre | 993 ms | 1073 ms (+%8) |

**Yatağın güvenilirliğinin en güçlü kanıtı:** eski ayar tam koşuda **522** kare üretti —
puanlı oturumda da tam **522** karede tahmin göndermiştik. Aynı sayı, bağımsız iki yoldan.

## Ek düzeltme: referans başına tek kutu

Görsel denetimde bir kare fark edildi: aynı spor sahasına **iki kutu** çiziliyordu. Koda
bakınca sebebi çıktı — `match()` her referans için **iki ayrı aday** üretiyor (tam-kare
geçiş + SAHI tiling geçişi) ve bunlar kısmen örtüştüğünde NMS eşiğini (0,5) aşamayıp
ikisi de hayatta kalıyordu. Oysa **her referans tek bir nesnedir**; ikinci kutu tanımı
gereği yanlış pozitiftir ve mAP'te precision'ı düşürür.

`matching.one_box_per_reference = true` eklendi (en yüksek güvenli aday tutulur).

| | çok kutu | **tek kutu** |
|---|---|---|
| recall | %93,0 | **%93,0** (değişmedi) |
| süre | 1073 ms | **979 ms** |
| kutu alanı medyanı | 241×241 | 269×269 |

Recall hiç düşmeden fazladan kutular elendi. Kutu medyanının büyümesi beklenen: artık
küçük tiling kutuları sayıma girmiyor.

**İki adayı BIRLESTIRME (union) fikri ölçümle reddedildi:** iki aday olduğunda IoU medyanı
yalnız **0,03**, ve 18/26 karede IoU < 0,1 — yani adaylar *farklı yerleri* gösteriyor
(biri doğru nesne, diğeri başka bir yer). Birleştirmek devasa yanlış kutu üretirdi. Bu veri
aynı zamanda elenen ikinci kutuların gerçekten yanlış pozitif olduğunu doğruluyor.

## Reddedilenler ve gerekçeleri

- **`box_k_std = 2,5`** — kutuyu 287→198 px daraltıyor ve recall'ı düşürmüyordu, yani
  sayısal olarak cazipti. **Görsel denetimde reddedildi:** spor sahasının yalnız yeşil
  yarısını alıyor, kırmızı bölümü kesiyordu. Daha dar kutu her zaman daha isabetli değil.
- **`k_std = 2,0`** — recall'ı %100 yapıyor ve kurtardığı kareler **gerçek** (görselde
  kamera kayınca kadraj kenarında kalan sahayı doğru yakalıyor). Reddetme sebebi:
  kare başına fazladan kutu üretiyor ve çıktımızda **güven skoru yok**
  (`ReferencePrediction` yalnız kutu gönderir), dolayısıyla fazladan kutu doğrudan yanlış
  pozitif sayılır ve mAP'te precision'ı düşürür.
- **`reference.augment`** — hiçbir ölçütü değiştirmedi.
- **`patch_topk`** — kutu büyüdü.

## Görsel doğrulama

`degerlendirme/panel_SECILEN_*.jpg`. Gözle doğrulananlar:
- **ref2** (spor sahası): kutu sahayı sarıyor ✓
- **ref4** (çatıdaki yıldız/pusula deseni): k1100/k1125/k1150'de desen **isabetle**
  kutulanıyor ✓
- **ref6** (amfi basamakları): doğru ✓

## Bilinen sınırlar (dürüstlük notu)

1. **Recall metriği "tespit var mı" ölçer, "doğru mu" değil.** Aralık başında/sonunda nesne
   henüz kadraja girmemiş olabiliyor; orada model yine de bir şeye kutu çiziyor (ör. ref4
   aralığının ilk karesinde havuza). Bu kareler recall'a *olumlu* yansıyor ama gerçekte
   yanlış pozitif. Gerçek mAP bu yüzden buradaki %93'ten düşük olacaktır.
2. **Model "burada yok" diyemiyor.** Aralık dışı kontrol karelerinde (nesne sahnede yokken)
   12/14 karede tespit üretiyor — bir keresinde alakasız bir arabaya. Yarışmada aralık
   kapısı (`active_ids`) bunu gönderilmekten alıkoyduğu için puana yansımıyor, ama model
   ayrım gücünün zayıf olduğunu gösteriyor.
3. Bazı karelerde tek nesneye **iki kutu** çıkıyor (NMS tam ayıramıyor) — mAP'te fazladan
   kutu maliyetlidir.
