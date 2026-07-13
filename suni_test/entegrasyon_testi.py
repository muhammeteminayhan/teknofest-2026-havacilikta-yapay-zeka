# -*- coding: utf-8 -*-
"""YENI arayuz (13 Tem resmi) entegrasyon testi — CEVRIMDISI, gercek yakalanan veriyle.

Amac: resmi rolling akisinin bizim modele bagli kismini uctan uca dogrulamak:
  - register_references(YENI format refs, ref_image_paths)  -> DINOv3 ozniteligi cikar
  - detect(pred, health, img, active_refs=...)              -> Gorev 1 + 2 + 3
  - Gorev 3 ciktisi resmi ReferencePrediction semasi: reference=ref['url'], frame=frame_url

Sunucu YOK: yakalanan kareler (sunucu_ciktilari/iter_*.webp) + yakalanan frames/translations
JSON'lari + gercek referans gorselleri (reference_1/2.png) kullanilir. Referans pencereleri
gozlemlenen degerlerdir (ref1: frame 165-209, ref2: 525-656; iter = frame_id - 13).

Calistir: proje kokunden  .venv\\Scripts\\python suni_test\\entegrasyon_testi.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAP = ROOT / "sistem" / "sunucu_ciktilari"
sys.path.insert(0, str(ROOT / "sistem"))
sys.stdout.reconfigure(encoding="utf-8")

import cv2  # noqa: E402
from src.frame_predictions import FramePredictions  # noqa: E402
from src.object_detection_model import ObjectDetectionModel  # noqa: E402

FRAME0 = 13  # iter 0 -> frame id 13


def _iter_of(frame_id):
    return frame_id - FRAME0


def _frames_json(it):
    p = CAP / f"iter_{it:03d}_frames.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    return d[0] if isinstance(d, list) else d


def _trans_json(it):
    p = CAP / f"iter_{it:03d}_translations.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    return (d[0] if isinstance(d, list) else d) if d else None


def _ref_dict(order, ref_png, start_fid, end_fid):
    """YENI /reference/ formatinda referans sozlugu uret (yakalanan sinir karelerinden)."""
    fs = _frames_json(_iter_of(start_fid))
    fe = _frames_json(_iter_of(end_fid))
    return {
        "url": f"http://x/reference/{order}/",
        "image_url": f"/media/reference_{order}.png",
        "order": order,
        "frame_start": fs["url"], "frame_end": fe["url"],
        "frame_start_image_url": fs["image_url"],
        "frame_end_image_url": fe["image_url"],
    }, str(ref_png)


def _active_refs(refs, frame_image_url):
    """main.py'nin pencere kapisi: image_url ∈ [start, end]."""
    return [r for r in refs
            if r.get("frame_start_image_url") and r.get("frame_end_image_url")
            and r["frame_start_image_url"] <= frame_image_url <= r["frame_end_image_url"]]


def main():
    print("=== YENI arayuz entegrasyon testi (cevrimdisi) ===\n")
    print("Model yukleniyor (Gorev 1/2/3)...")
    model = ObjectDetectionModel("http://x/")

    # --- YENI format referanslari + yerel gorsel yollari ---
    refs = []
    ref_image_paths = {}
    for order, png, (a, b) in [(1, CAP / "reference_1.png", (165, 209)),
                               (2, CAP / "reference_2.png", (525, 656))]:
        rd, path = _ref_dict(order, png, a, b)
        refs.append(rd)
        ref_image_paths[rd["url"]] = path
    n = model.register_references(refs, ref_image_paths=ref_image_paths)
    print(f"register_references -> {n} referans kaydedildi (beklenen 2)\n")
    assert n == 2, "Referanslar kaydedilemedi!"

    # --- test kareleri: her referansin penceresinden + pencere disindan ---
    testler = [
        ("ref1 penceresi", 178, 1),   # frame 178 (iter 165) -> ref1 aktif
        ("ref1 penceresi", 191, 1),
        ("ref2 penceresi", 568, 2),   # frame 568 -> ref2 aktif
        ("ref2 penceresi", 613, 2),
        ("pencere DISI", 313, None),  # frame 313 -> hicbir referans aktif degil
    ]

    # DIKKAT: bir referans, penceresinin HER karesinde eslesmez (gorunurluk/guven
    # esigine bagli). Dogru dogrulama semantigi:
    #   - pencere DISI  -> reference_predictions BOS olmali (yanlis pozitif yok),
    #   - pencere ICI   -> eslesme olursa DOGRU url + resmi sema (yoksa kabul),
    #   - genel         -> her referans en az bir karede eslesmeli (wiring atesleniyor).
    hata = 0
    eslesen_order = set()
    for etiket, fid, beklenen_order in testler:
        it = _iter_of(fid)
        fj = _frames_json(it)
        webp = CAP / f"iter_{it:03d}.webp"
        if fj is None or not webp.exists():
            print(f"[ATLA] {etiket} frame {fid}: yakalanan veri yok")
            continue
        img = cv2.imread(str(webp))
        tj = _trans_json(it) or {}
        hs = tj.get("health_status")
        pred = FramePredictions(fj["url"], fj["image_url"], fj["video_name"],
                                tj.get("translation_x"), tj.get("translation_y"),
                                tj.get("translation_z"))
        active = _active_refs(refs, fj["image_url"])
        pred = model.detect(pred, hs, img, active_refs=active, ref_image_paths=ref_image_paths)

        n_obj = len(pred.detected_objects)
        n_tr = len(pred.translations)
        rp = pred.reference_predictions
        urls = [p.reference_url for p in rp]

        if beklenen_order is None:
            # pencere disi: KESINLIKLE bos olmali
            ok = (len(rp) == 0)
            if not ok:
                hata += 1
            bilgi = f"referans YOK (beklenen) | reference_predictions={len(rp)} aktif={len(active)}"
        else:
            beklenen_url = f"http://x/reference/{beklenen_order}/"
            # eslesme varsa: url + sema dogru olmali (yanlis url = HATA)
            sema_ok = True
            if rp:
                pl = rp[0].create_payload()
                sema_ok = (set(pl.keys()) == {"reference", "frame", "top_left_x", "top_left_y",
                                              "bottom_right_x", "bottom_right_y"}
                           and all(u == beklenen_url for u in urls))
                if sema_ok:
                    eslesen_order.add(beklenen_order)
            if rp and not sema_ok:
                hata += 1
            ok = sema_ok
            durum_ic = "eslesme VAR" if rp else "eslesme yok (kabul, tespit-bagli)"
            bilgi = f"{durum_ic} | url={urls} aktif={len(active)}"
        print(f"[{'OK ' if ok else 'HATA'}] {etiket:16s} frame {fid}: "
              f"Gorev1={n_obj} Gorev2={n_tr} | {bilgi}")

    # genel: her referans en az bir karede eslesmeli (Gorev 3 gercekten atesleniyor)
    ates_ok = eslesen_order == {1, 2}
    print(f"\nEslesen referanslar: {sorted(eslesen_order)} (beklenen [1, 2]) -> "
          f"{'OK' if ates_ok else 'EKSIK'}")
    print(f"Sema/yanlis-pozitif hatasi: {hata} (beklenen 0)")

    if hata == 0 and ates_ok:
        print("\n=== YENI ARAYUZ ENTEGRASYONU CALISIYOR ===")
        print("Gorev 1+2+3 ateslendi; reference_predictions dogru url+sema; pencere disi temiz.")
    else:
        print("\nDIKKAT: entegrasyonda sorun — yukariyi incele.")
        sys.exit(1)


if __name__ == "__main__":
    main()
