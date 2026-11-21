# -*- coding: utf-8 -*-
"""GOREV 3 — TEST YATAGI: gercek yarisma kareleri + resmi referans araliklari.

NEDEN: Gorev 3'un puani mAP ile hesaplanir (sartname 9.3) ama elimizde GT KUTU YOK,
dolayisiyla mAP dogrudan olculemez. Buna karsilik sunucunun verdigi referans kayitlari
her referans icin bir **kare araligi** (`frame_start` – `frame_end`) iceriyor. Bu, zayif
ama GERCEK bir etikettir: o aralikta nesne sahnede VARDIR, disinda yoktur.

Bu yatak o etiketten cikarilabilecek her seyi olcer:

  * **recall**        : aktif karelerin kacinda tespit uretiyoruz  (puanli oturum: %77)
  * **aralik disi**   : nesnenin OLMADIGI karelerde uretilen tespit (yanlis pozitif gostergesi)
  * **kutu alani**    : cok buyuk/kucuk kutu egilimi (IoU'yu dogrudan bozar)
  * **kutu sicramasi**: ardisik karelerde merkez kaymasi — localization kararliligi proxy'si
  * **sure**          : kare basi ms (yarisma penceresi 90 dk)

Kutu ISABETI otomatik olculemez; bunun icin gorsel panel uretilir ve GOZLE dogrulanir
(Gorev 1'de ayni yol kullanildi).

Calistir:
    .venv\\Scripts\\python gorev3\\test_yatagi.py                  # ornekleme (hizli)
    .venv\\Scripts\\python gorev3\\test_yatagi.py --tam            # tum aktif kareler
    .venv\\Scripts\\python gorev3\\test_yatagi.py --etiket patch   # varyanti adlandir
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

GOREV3 = Path(__file__).resolve().parent
ROOT = GOREV3.parent
OUT = GOREV3 / "degerlendirme"
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

OTURUM = ROOT / "sistem" / "_images" / "THYZ_2026_Online_Yarisma"
REFS_JSON = OTURUM / "references.json"


def _kare_no(image_url: str) -> int | None:
    """'/.../frame_000010.webp' -> 10 (dosya adi 0-tabanli)."""
    m = re.search(r"frame_(\d+)", image_url or "")
    return int(m.group(1)) if m else None


def referanslari_oku():
    """references.json -> [{order, img, bas, son}] (bas/son = 0-tabanli kare no)."""
    refs = json.loads(REFS_JSON.read_text(encoding="utf-8"))
    out = []
    for r in refs:
        bas = _kare_no(r.get("frame_start_image_url"))
        son = _kare_no(r.get("frame_end_image_url"))
        img = OTURUM / "references" / Path(r["image_url"]).name
        if bas is None or son is None or not img.exists():
            print(f"  (atlandi: order={r.get('order')} — aralik/gorsel eksik)")
            continue
        out.append(dict(order=int(r["order"]), img=img, bas=bas, son=son,
                        url=r["url"], image_url=r["image_url"]))
    return out


def kare_yolu(no: int) -> Path:
    return OTURUM / f"frame_{no:06d}.webp"


def ayar_uygula(m, ayar: str | None):
    """'matching.patch_weight=0.5,matching.k_std=2.5' -> matcher.config'e yaz.

    Config dosyaya yazilmadan, YALNIZ bu kosu icin degistirilir; boylece varyantlar
    yan yana olculebilir ve yarisma ayari kazara degismez.
    """
    if not ayar:
        return {}
    uygulanan = {}
    for parca in ayar.split(","):
        yol, _, deg = parca.partition("=")
        dugum = m.matcher.config
        anahtarlar = yol.strip().split(".")
        for a in anahtarlar[:-1]:
            dugum = dugum.setdefault(a, {})
        try:
            v = yaml_deger(deg.strip())
        except Exception:
            v = deg.strip()
        dugum[anahtarlar[-1]] = v
        uygulanan[yol.strip()] = v
    # DIKKAT — SESSIZ TUZAK: TemplateMatcher.__init__ bazi ayarlari config'ten
    # KOPYALAYIP kendi ozniteliginde tutuyor (cls_threshold, min_confidence,
    # threshold_ratio, noise_floor, fusion, augment). Sadece config sozlugunu
    # degistirmek bunlari ETKILEMIYOR; bu yuzden 10 Eyl 2026'da yapilan bir
    # min_confidence taramasi bes farkli degerde de AYNI sonucu verdi ve
    # "ayar etkisiz" gibi gorundu. Kopyalanan oznitelikler burada tazeleniyor.
    _m = m.matcher
    for _ad, _yol in (("cls_threshold", ("matching", "cls_threshold")),
                      ("min_confidence", ("matching", "min_confidence")),
                      ("threshold_ratio", ("matching", "threshold_ratio")),
                      ("noise_floor", ("matching", "noise_floor")),
                      ("fusion", ("fpn", "fusion_method")),
                      ("augment", ("reference", "augment"))):
        try:
            setattr(_m, _ad, _m.config[_yol[0]][_yol[1]])
        except KeyError:
            pass
    return uygulanan


def yaml_deger(s: str):
    import yaml as _y
    return _y.safe_load(s)


def matcher_kur(ayar: str | None = None, **kw):
    from gorev3 import Gorev3Matcher
    m = Gorev3Matcher(kind="dinov3", remove_bg=False, **kw)
    uyg = ayar_uygula(m, ayar)          # referans KAYDINDAN ONCE (augment gibi ayarlar icin)
    if uyg:
        print(f"  ayar override: {uyg}")
    refs = referanslari_oku()
    sunucu_kayit = [{"url": r["url"], "image_url": str(r["img"]), "order": r["order"],
                     "frame_start": f"http://x/frames/{r['bas']+1}/",
                     "frame_end": f"http://x/frames/{r['son']+1}/"} for r in refs]
    yollar = {r["url"]: str(r["img"]) for r in refs}
    n = m.register_from_server(sunucu_kayit, ref_image_paths=yollar)
    return m, refs, n


def _merkez(d):
    return ((d["top_left_x"] + d["bottom_right_x"]) / 2.0,
            (d["top_left_y"] + d["bottom_right_y"]) / 2.0)


def _alan(d):
    return max(0, d["bottom_right_x"] - d["top_left_x"]) * \
           max(0, d["bottom_right_y"] - d["top_left_y"])


def kos(m, refs, ornek: int | None, disari: int):
    """Aktif aralik + (kontrol icin) aralik disi kareleri isle."""
    sonuc = {}          # order -> {kare_no: [det, ...]}
    sureler = []
    for r in refs:
        aralik = list(range(r["bas"], r["son"] + 1))
        if ornek is not None and len(aralik) > ornek:
            idx = np.linspace(0, len(aralik) - 1, ornek).astype(int)
            aralik = [aralik[i] for i in idx]
        # aralik DISI kontrol kareleri (nesne sahnede yok -> tespit cikmamali)
        dis = []
        if disari > 0:
            aday = [r["bas"] - 200, r["bas"] - 120, r["son"] + 120, r["son"] + 200]
            dis = [k for k in aday if 0 <= k < 2250 and kare_yolu(k).exists()][:disari]
        kayit = {}
        for k in aralik + dis:
            p = kare_yolu(k)
            if not p.exists():
                continue
            fr = cv2.imread(str(p))
            t0 = time.perf_counter()
            dets = m.match(fr, frame_id=k + 1, active_ids={r["order"]})
            sureler.append((time.perf_counter() - t0) * 1000)
            kayit[k] = dets
        sonuc[r["order"]] = dict(kayit=kayit, aralik=set(aralik), dis=set(dis))
    return sonuc, sureler


def rapor(sonuc, refs, sureler, etiket, m=None, panel=2):
    OUT.mkdir(parents=True, exist_ok=True)
    L = [f"=== GOREV 3 — TEST YATAGI ({etiket}) ===",
         f"kaynak: {OTURUM.name} | {len(refs)} referans (resmi frame_start/end araliklari)",
         "olcut: recall = aktif karelerin kacinda tespit var; aralik disi = nesne YOKKEN uretilen tespit",
         "",
         f"{'ref':>4}{'aralik':>14}{'kare':>7}{'tespit':>8}{'recall':>9}"
         f"{'alan medyan':>13}{'sicrama px':>12}{'disi':>7}"]
    top_akt = top_bul = top_dis_kare = top_dis_bul = 0
    alanlar_tum = []
    for r in refs:
        s = sonuc.get(r["order"])
        if not s:
            continue
        akt = [k for k in s["kayit"] if k in s["aralik"]]
        bul = [k for k in akt if s["kayit"][k]]
        alanlar = [_alan(d) for k in akt for d in s["kayit"][k]]
        alanlar_tum += alanlar
        # ardisik kare merkez sicramasi (yalniz ardisik ornekler arasinda)
        sic = []
        sirali = sorted(bul)
        for a, b in zip(sirali, sirali[1:]):
            if b - a <= 3:                     # gercekten ardisik sayilacak kadar yakin
                ca, cb = _merkez(s["kayit"][a][0]), _merkez(s["kayit"][b][0])
                sic.append(float(np.hypot(ca[0] - cb[0], ca[1] - cb[1])))
        dis_kare = [k for k in s["kayit"] if k in s["dis"]]
        dis_bul = [k for k in dis_kare if s["kayit"][k]]
        top_akt += len(akt); top_bul += len(bul)
        top_dis_kare += len(dis_kare); top_dis_bul += len(dis_bul)
        aralik_str = f"{r['bas']}-{r['son']}"
        disi_str = f"{len(dis_bul)}/{len(dis_kare)}"
        alan_med = int(np.median(alanlar)) if alanlar else 0
        sic_med = float(np.median(sic)) if sic else 0.0
        L.append(f"{r['order']:>4}{aralik_str:>14}"
                 f"{len(akt):>7}{len(bul):>8}{100*len(bul)/max(len(akt),1):>8.0f}%"
                 f"{alan_med:>13}{sic_med:>12.1f}{disi_str:>7}")
    L += ["",
          f"TOPLAM recall: {top_bul}/{top_akt} = %{100*top_bul/max(top_akt,1):.1f}",
          f"ARALIK DISI tespit: {top_dis_bul}/{top_dis_kare} kare "
          f"(dusuk olmali — nesne sahnede yok)",
          f"kutu alani medyan: {int(np.median(alanlar_tum)) if alanlar_tum else 0} px2 "
          f"({np.sqrt(np.median(alanlar_tum)) if alanlar_tum else 0:.0f}x"
          f"{np.sqrt(np.median(alanlar_tum)) if alanlar_tum else 0:.0f})",
          f"sure: {np.mean(sureler):.0f} ms/kare (p95 {np.percentile(sureler,95):.0f})",
          ""]
    metin = "\n".join(L)
    (OUT / f"yatak_{etiket}.txt").write_text(metin, encoding="utf-8")
    print(metin)
    return dict(recall=top_bul / max(top_akt, 1), disi=top_dis_bul,
                alan=float(np.median(alanlar_tum)) if alanlar_tum else 0.0,
                ms=float(np.mean(sureler)))


def paneller(sonuc, refs, etiket, adet=2):
    """Her referans icin kutu cizilmis ornek kare (gozle dogrulama)."""
    OUT.mkdir(parents=True, exist_ok=True)
    yazildi = 0
    for r in refs:
        s = sonuc.get(r["order"])
        if not s:
            continue
        bulunan = sorted(k for k in s["kayit"] if k in s["aralik"] and s["kayit"][k])
        if not bulunan:
            continue
        for k in bulunan[:: max(1, len(bulunan) // adet)][:adet]:
            fr = cv2.imread(str(kare_yolu(k)))
            for d in s["kayit"][k]:
                cv2.rectangle(fr, (d["top_left_x"], d["top_left_y"]),
                              (d["bottom_right_x"], d["bottom_right_y"]), (0, 255, 0), 3)
            ref_im = cv2.imread(str(r["img"]))
            h = 220
            ref_im = cv2.resize(ref_im, (int(ref_im.shape[1] * h / ref_im.shape[0]), h))
            fr[10:10 + h, 10:10 + ref_im.shape[1]] = ref_im     # referans kucuk resim
            cv2.rectangle(fr, (10, 10), (10 + ref_im.shape[1], 10 + h), (0, 200, 255), 3)
            cv2.putText(fr, f"ref{r['order']}  kare {k}", (10, h + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 255), 3)
            kucuk = cv2.resize(fr, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(OUT / f"panel_{etiket}_ref{r['order']}_k{k}.jpg"), kucuk,
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
            yazildi += 1
    print(f"gorsel: {yazildi} panel -> {OUT.name}/panel_{etiket}_*.jpg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ornek", type=int, default=15, help="referans basi kare ornegi")
    ap.add_argument("--tam", action="store_true", help="tum aktif kareler (yavas)")
    ap.add_argument("--disari", type=int, default=2, help="aralik disi kontrol karesi")
    ap.add_argument("--etiket", default="mevcut")
    ap.add_argument("--ayar", default=None,
                    help="gecici config override: 'matching.patch_weight=0.5,matching.k_std=2.5'")
    ap.add_argument("--panel", type=int, default=2)
    args = ap.parse_args()

    m, refs, n = matcher_kur(ayar=args.ayar)
    print(f"{n} referans kayitli | ornek={'TAM' if args.tam else args.ornek}\n")
    sonuc, sureler = kos(m, refs, None if args.tam else args.ornek, args.disari)
    rapor(sonuc, refs, sureler, args.etiket, m)
    if args.panel:
        paneller(sonuc, refs, args.etiket, args.panel)


if __name__ == "__main__":
    main()
