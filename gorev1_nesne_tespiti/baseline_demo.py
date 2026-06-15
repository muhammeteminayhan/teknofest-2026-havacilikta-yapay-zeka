"""
Görev 1 baseline duman testi.

Örnek RGB videodan ardışık kareler çıkarır (yoksa Görev 2'nin commit'li örnek
karelerine düşer), YOLOv11+SAHI (COCO-pretrained) ile tespit yapar, hareketli/sabit
ve iniş durumunu hesaplar, kutuları çizip outputs/ altına kaydeder ve sonuçları
arayüzün DetectedObject payload'una çevirip doğrular.

Amaç: boru hattının uçtan uca çalıştığını göstermek (COCO car/person ≈ Taşıt/İnsan).
Gerçek doğruluk fine-tune sonrası gelir.

Kullanım:
    python gorev1_nesne_tespiti/baseline_demo.py --kare 8 --baslangic 900
    python gorev1_nesne_tespiti/baseline_demo.py --no-sahi      # hızlı (dilimsiz)
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import cv2
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from gorev1_nesne_tespiti.configs import config as C  # noqa: E402
from gorev1_nesne_tespiti.entegrasyon.g1_model import Gorev1Modeli, to_detected_objects  # noqa: E402
from gorev1_nesne_tespiti.veri.siniflar import YARISMA_SINIFLARI  # noqa: E402

_RENK = {0: (0, 200, 0), 1: (0, 140, 255), 2: (255, 0, 0), 3: (0, 0, 255)}


def kareleri_al(n: int, baslangic: int):
    """Ardışık kareler döndür (orig_idx, BGR). Önce MP4, yoksa örnek JPG'ler."""
    if os.path.exists(C.ORNEK_RGB_VIDEO):
        cap = cv2.VideoCapture(C.ORNEK_RGB_VIDEO)
        cap.set(cv2.CAP_PROP_POS_FRAMES, baslangic)
        out = []
        for i in range(n):
            ok, fr = cap.read()
            if not ok:
                break
            out.append((baslangic + i, fr))
        cap.release()
        if out:
            return out
    # Yedek: Görev 2 örnek kareleri (repoda commit'li).
    ornek = sorted(glob.glob(os.path.join(
        _REPO, "gorev2_visual_odometry", "outputs", "samples_rgb", "*_raw.jpg")))
    print("MP4 yok → örnek JPG kareler kullanılıyor (hareket durumu anlamlı olmaz).")
    return [(i, cv2.imread(p)) for i, p in enumerate(ornek)]


def ciz(frame, sonuclar):
    im = frame.copy()
    for s in sonuclar:
        renk = _RENK.get(s.cls_id, (255, 255, 255))
        cv2.rectangle(im, (int(s.x1), int(s.y1)), (int(s.x2), int(s.y2)), renk, 2)
        etiket = YARISMA_SINIFLARI.get(s.cls_id, "?")
        if s.cls_id == 0:
            etiket += f" {'hareketli' if s.motion_status == 1 else 'sabit'}"
        if s.cls_id in (2, 3):
            etiket += f" inis={s.landing_status}"
        cv2.putText(im, etiket, (int(s.x1), max(0, int(s.y1) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, renk, 1, cv2.LINE_AA)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kare", type=int, default=8)
    ap.add_argument("--baslangic", type=int, default=900)
    ap.add_argument("--no-sahi", action="store_true")
    args = ap.parse_args()

    os.makedirs(C.OUTPUTS_DIR, exist_ok=True)
    kareler = kareleri_al(args.kare, args.baslangic)
    if not kareler:
        print("HATA: kare bulunamadı (ne MP4 ne örnek JPG).")
        return

    print(f"Model: {C.YOLO_MODEL} | SAHI: {not args.no_sahi} | cihaz: {C.DEVICE}")
    model = Gorev1Modeli(model_classes="coco", use_sahi=not args.no_sahi)

    toplam = 0
    for oi, fr in kareler:
        sonuclar = model.isle(fr)
        toplam += len(sonuclar)
        # Arayüz payload doğrulaması (replay):
        nesneler = to_detected_objects(sonuclar, "http://127.0.0.1:5000/")
        gecerli = sum(1 for d in nesneler
                      if isinstance(d.create_payload("http://127.0.0.1:5000/"), dict))
        out_path = os.path.join(C.OUTPUTS_DIR, f"baseline_{oi:06d}.jpg")
        cv2.imwrite(out_path, ciz(fr, sonuclar))
        sayim = {}
        for s in sonuclar:
            sayim[YARISMA_SINIFLARI.get(s.cls_id, "?")] = sayim.get(YARISMA_SINIFLARI.get(s.cls_id, "?"), 0) + 1
        print(f"  kare {oi}: {len(sonuclar)} tespit {dict(sayim)} | "
              f"payload {gecerli}/{len(nesneler)} ✓ -> {os.path.basename(out_path)}")

    print(f"\nToplam {toplam} tespit, {len(kareler)} karede. Görseller: {C.OUTPUTS_DIR}")
    print("Baseline boru hattı çalışıyor ✓ (tespit→hareket→iniş→payload).")


if __name__ == "__main__":
    main()
