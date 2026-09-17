# -*- coding: utf-8 -*-
"""YARISMA GUNU ON-UCUS KONTROLU — her oturumdan once calistir.

Denetlenenler:
  1. Ortam: OpenCV / torch / CUDA / GPU adi
  2. **Cevrimdisi hazirlik**: RAFT agirliklari makinede inmis mi? (yarisma
     bilgisayarinin internete cikmasi YASAK; agirlik yoksa RAFT kurulamaz)
  3. Motor zinciri gercekten calisiyor mu (RAFT -> SIFT -> LK)
  4. Kamera profili tanindi mi (termal 640x512 · 1080p · 4K 4000x3000 · 4K video 3840x2160)
  5. Kare basi sure butcesi (sartname: saniyede 1 kare yeterli; oturum 2250 kare)

Kullanim:
    python on_ucus.py                       # yalniz ortam + motor kontrolu
    python on_ucus.py --video YOL.MP4       # ustune gercek karelerle hiz olcumu
    python on_ucus.py --kareler KLASOR      # kare klasoru (webp/jpg/png)
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2                                             # noqa: E402
from konum import (LivePositionEstimator, PROFILES,    # noqa: E402
                   on_ucus_kontrolu, profile_for_size)

BUTCE_S_KARE = 1.6          # 90 dk pencere / 2250 kare ~ 2,4 s; guvenli hedef 1,6 s
OTURUM_KARE = 2250


def _kare_akisi(video: str | None, kareler: str | None, n: int):
    """Videodan (her 4. kare) ya da klasorden (oldugu gibi) kare uret."""
    if video:
        cap = cv2.VideoCapture(video)
        try:
            verilen = i = 0
            while verilen < n:
                ok, f = cap.read()
                if not ok:
                    break
                if i % 4 == 0:            # 30 FPS video -> 7,5 FPS yarisma temposu
                    yield f
                    verilen += 1
                i += 1
        finally:
            cap.release()
    else:
        for p in sorted(glob.glob(str(Path(kareler) / "*")))[:n]:
            im = cv2.imread(p)
            if im is not None:
                yield im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=None)
    ap.add_argument("--kareler", default=None)
    ap.add_argument("--n", type=int, default=60, help="hiz olcumu icin kare sayisi")
    args = ap.parse_args()

    print("=" * 72)
    print("GOREV 2 — ON-UCUS KONTROLU")
    print("=" * 72)

    d = on_ucus_kontrolu()
    print(f"\n[1] ORTAM")
    print(f"    opencv          : {d['opencv']}")
    print(f"    torch           : {d['torch']}")
    print(f"    cuda            : {d['cuda']}   gpu: {d['gpu']}")

    print(f"\n[2] CEVRIMDISI HAZIRLIK")
    ok_agirlik = d["raft_agirligi_yerel"]
    print(f"    RAFT agirligi yerel : {'EVET' if ok_agirlik else '>>> HAYIR <<<'}")
    if not ok_agirlik:
        print("    !!! Yarisma bilgisayari cevrimdisi olacak. Agirlik yoksa RAFT")
        print("    !!! KURULAMAZ ve sistem SIFT yedegine duser (ort. +0,5 m).")
        print("    !!! Cozum: internete bagliyken bir kez RaftTracker olustur.")
    print(f"    SIFT (yedek) var    : {'EVET' if d['sift_var'] else 'HAYIR'}")
    print(f"    onerilen motor      : {d['onerilen_motor']}")

    print(f"\n[3] TANINAN KAMERA PROFILLERI")
    for p in PROFILES:
        print(f"    {p.name:14s} {p.width:>5}x{p.height:<5} fx={p.fx:8.2f} "
              f"k1={p.k1:+.4f}")

    if not (args.video or args.kareler):
        print("\n[4] MOTOR ZINCIRI — kare verilmedi, sentetik kareyle test ediliyor")
        kareler = [np.random.default_rng(i).integers(0, 255, (1080, 1920), np.uint8)
                   for i in range(3)]
    else:
        print("\n[4] MOTOR ZINCIRI + HIZ — gercek karelerle")
        kareler = list(_kare_akisi(args.video, args.kareler, args.n))
        if not kareler:
            print("    !!! Kare okunamadi.")
            return 1

    boy = kareler[0].shape
    try:
        prof = profile_for_size(boy[1], boy[0])
        print(f"    kare boyutu {boy[1]}x{boy[0]} -> profil '{prof.name}'  OK")
    except ValueError as e:
        print(f"    !!! PROFIL TANINMADI: {e}")
        return 1

    for motor in ("auto", "sift", "lk"):
        try:
            est = LivePositionEstimator(engine=motor)
            for i, im in enumerate(kareler):
                est.process_frame(im, np.array([i * 0.1, i * 0.05, 0.0]),
                                  i < len(kareler) // 2)
            t = est.timing_stats
            butce = t["ort_ms"] / 1000 <= BUTCE_S_KARE
            print(f"    istek={motor:<5} -> motor={est.engine_used:<5} "
                  f"yontem={est.config.method:<9} "
                  f"ort={t['ort_ms']:6.0f} ms  p95={t['p95_ms']:6.0f} ms  "
                  f"{'OK' if butce else '>>> BUTCE ASIMI <<<'}")
            if motor == "auto":
                tahmin_dk = t["ort_ms"] / 1000 * OTURUM_KARE / 60
                print(f"    {'':13} -> {OTURUM_KARE} karelik oturum tahmini: "
                      f"{tahmin_dk:.1f} dk")
        except Exception as e:
            print(f"    istek={motor:<5} -> HATA: {type(e).__name__}: {e}")

    print("\n" + "=" * 72)
    hazir = ok_agirlik and d["cuda"]
    print(f"DURUM: {'HAZIR (RAFT/GPU)' if hazir else 'GPU ya da agirlik EKSIK — yedekle kosar'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
