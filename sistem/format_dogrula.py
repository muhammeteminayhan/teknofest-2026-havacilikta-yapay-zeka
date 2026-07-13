# -*- coding: utf-8 -*-
"""BIZIM CEVABIN FORMATINI dogrula (toplanan sunucu_ciktilari/ uzerinden).

Iki kat dogrulama:
  1) YAPISAL — her `bizim_cevaplar.jsonl` payload'i resmi semaya uyuyor mu:
       ust anahtarlar {frame, detected_objects, detected_translations, reference_predictions};
       detected_object -> cls(URL) + landing_status + moving_status + bbox;
       detected_translation -> x/y/z; reference_prediction -> reference(url) + frame(url) + bbox.
  2) SUNUCU-KABUL — `sunucu_yanitlari.jsonl`'de kod dagilimi (201 = kabul); ozellikle
     reference_prediction GONDERILEN karelerin hepsi 201 mi (Gorev 3 formati kabul edildi mi).

Cikti: FORMAT DOGRU / eksik listesi.
Kullanim (sistem/ klasorunden):  ..\\.venv\\Scripts\\python format_dogrula.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CIKTI = HERE / "sunucu_ciktilari"
sys.stdout.reconfigure(encoding="utf-8")

UST = {"frame", "detected_objects", "detected_translations", "reference_predictions"}
OBJ = {"cls", "landing_status", "moving_status",
       "top_left_x", "top_left_y", "bottom_right_x", "bottom_right_y"}
TRANS = {"translation_x", "translation_y", "translation_z"}
REFP = {"reference", "frame", "top_left_x", "top_left_y", "bottom_right_x", "bottom_right_y"}


def sema_hatalari(payload):
    h = []
    if set(payload.keys()) != UST:
        h.append(f"ust anahtarlar yanlis: {sorted(payload.keys())}")
        return h
    for o in payload["detected_objects"]:
        if set(o.keys()) != OBJ:
            h.append(f"detected_object anahtar: {sorted(o.keys())}")
        elif "/classes/" not in str(o["cls"]):
            h.append(f"cls URL degil: {o['cls']}")
    for t in payload["detected_translations"]:
        if set(t.keys()) != TRANS:
            h.append(f"translation anahtar: {sorted(t.keys())}")
    for rp in payload["reference_predictions"]:
        if set(rp.keys()) != REFP:
            h.append(f"reference_prediction anahtar: {sorted(rp.keys())}")
        elif "/reference/" not in str(rp["reference"]):
            h.append(f"reference URL degil: {rp['reference']}")
        elif "/frames/" not in str(rp["frame"]):
            h.append(f"frame URL degil: {rp['frame']}")
    return h


def _satirlar(p):
    if not p.exists():
        return []
    return [json.loads(s) for s in p.read_text(encoding="utf-8").splitlines() if s.strip()]


def main():
    cevaplar = _satirlar(CIKTI / "bizim_cevaplar.jsonl")
    yanitlar = _satirlar(CIKTI / "sunucu_yanitlari.jsonl")
    if not cevaplar:
        print(f"HATA: {CIKTI / 'bizim_cevaplar.jsonl'} yok/bos. Once sunucu_topla.py calistir.")
        sys.exit(1)

    print(f"Kayit: {len(cevaplar)} bizim_cevap | {len(yanitlar)} sunucu_yaniti\n")

    # 1) YAPISAL
    toplam_hata = 0
    ornek_hatalar = []
    n_obj = n_trans = n_refp = 0
    for c in cevaplar:
        pl = c["payload"]
        n_obj += len(pl["detected_objects"])
        n_trans += len(pl["detected_translations"])
        n_refp += len(pl["reference_predictions"])
        h = sema_hatalari(pl)
        if h:
            toplam_hata += len(h)
            if len(ornek_hatalar) < 8:
                ornek_hatalar.append((c.get("iter"), h))
    print("--- 1) YAPISAL SEMA ---")
    print(f"detected_objects: {n_obj} | detected_translations: {n_trans} | reference_predictions: {n_refp}")
    if toplam_hata == 0:
        print("OK: TUM payload'lar resmi semaya uyuyor (0 hata).")
    else:
        print(f"!!! {toplam_hata} sema hatasi. Ornekler:")
        for it, h in ornek_hatalar:
            print(f"   iter {it}: {h}")

    # 2) SUNUCU-KABUL
    kodlar = {}
    for y in yanitlar:
        kodlar[y.get("kod")] = kodlar.get(y.get("kod"), 0) + 1
    kod201 = kodlar.get(201, 0)
    print("\n--- 2) SUNUCU-KABUL ---")
    print(f"kod dagilimi: {kodlar}")
    if yanitlar:
        print(f"201 (kabul) orani: {kod201}/{len(yanitlar)} = %{100*kod201/len(yanitlar):.1f}")

    # reference_prediction gonderilen karelerin sunucu kodu 201 mi?
    yanit_by_iter = {y.get("iter"): y.get("kod") for y in yanitlar}
    refp_gonderilen = 0
    refp_201 = 0
    ref_sayac = {}   # reference url -> gonderim sayisi
    for c in cevaplar:
        rps = c["payload"]["reference_predictions"]
        if rps:
            refp_gonderilen += 1
            if yanit_by_iter.get(c.get("iter")) == 201:
                refp_201 += 1
            for rp in rps:
                ref_sayac[rp["reference"]] = ref_sayac.get(rp["reference"], 0) + 1
    print(f"\nreference_prediction iceren kare: {refp_gonderilen} | bunlarin 201 olani: {refp_201}")
    if ref_sayac:
        print("referans basi gonderim (Gorev 3 formati canli kabul edildi):")
        for u in sorted(ref_sayac):
            print(f"   {u}  -> {ref_sayac[u]} kare")
    else:
        print("(hic reference_prediction gonderilmemis — referans penceresine girilmemis olabilir)")

    # SONUC
    sema_ok = (toplam_hata == 0)
    kabul_ok = (len(yanitlar) == 0) or (kod201 == len(yanitlar)) or \
               (refp_gonderilen == 0) or (refp_201 == refp_gonderilen)
    print("\n=== SONUC ===")
    if sema_ok and (refp_gonderilen == 0 or refp_201 == refp_gonderilen):
        print("FORMAT DOGRU: payload'lar resmi semaya uyuyor" +
              (" ve gonderilen reference_prediction'larin hepsi 201 (kabul)." if refp_gonderilen else "."))
    else:
        print("DIKKAT: yukarida sema hatasi ve/veya reference_prediction icin 201-disi yanit var.")
        sys.exit(1)


if __name__ == "__main__":
    main()
