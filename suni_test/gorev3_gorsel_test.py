# -*- coding: utf-8 -*-
"""Gorev 3 DUZELTME testi (gorsel): duzeltilmis matcher'i gercek karelerde kostur.

Yakalanan gercek referanslar (reference_1/2.png) + kendi kare araliklarindaki kareler.
Her referans icin: referans gorseli + aralik karelerinde bizim bbox (varsa) cizili figur.
Amac: whole-image cop kutular gitti mi, kutular lokal mi, statistical esik + guard calisiyor mu.

Calistir: proje kokunden  .venv\\Scripts\\python suni_test\\gorev3_gorsel_test.py
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAP = ROOT / "sistem" / "sunucu_ciktilari"
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from gorev3 import Gorev3Matcher

FRAME0 = 13  # iter 0 -> frame id 13
REFS = {
    1: {"png": CAP / "reference_1.png", "range": (165, 209), "iters": [152, 165, 178, 192]},
    2: {"png": CAP / "reference_2.png", "range": (525, 656), "iters": [512, 555, 600, 640]},
}


def main():
    print("Duzeltilmis matcher yukleniyor (DINOv3, statistical esik, yuksek-res FPN)...")
    m = Gorev3Matcher(kind="dinov3", remove_bg=False)
    for oid, r in REFS.items():
        m.register_reference(cv2.imread(str(r["png"])), object_id=oid, frame_range=r["range"])
    print(f"araliklar: {m.ref_araliklar}\n")

    for oid, r in REFS.items():
        iters = r["iters"]
        fig, axes = plt.subplots(1, 1 + len(iters), figsize=(5.2 * (1 + len(iters)), 4.6))
        ref = cv2.cvtColor(cv2.imread(str(r["png"])), cv2.COLOR_BGR2RGB)
        axes[0].imshow(ref); axes[0].set_title(f"reference_{oid}"); axes[0].axis("off")
        for j, it in enumerate(iters):
            fid = FRAME0 + it
            frame = cv2.imread(str(CAP / f"iter_{it:03d}.webp"))
            if frame is None:
                axes[j + 1].set_title(f"frame {fid} YOK"); axes[j + 1].axis("off"); continue
            t0 = time.perf_counter()
            res = m.match(frame, frame_id=fid)
            dt = (time.perf_counter() - t0) * 1000
            vis = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            bilgi = "eslesme YOK"
            if res:
                d = res[0]
                cv2.rectangle(vis, (d["top_left_x"], d["top_left_y"]),
                              (d["bottom_right_x"], d["bottom_right_y"]), (255, 0, 0), 5)
                area = (d["bottom_right_x"] - d["top_left_x"]) * (d["bottom_right_y"] - d["top_left_y"])
                oran = area / (frame.shape[0] * frame.shape[1]) * 100
                bilgi = f"kutu (alan %{oran:.0f})"
            axes[j + 1].imshow(vis)
            axes[j + 1].set_title(f"frame {fid}\n{bilgi} | {dt:.0f}ms", fontsize=10)
            axes[j + 1].axis("off")
            print(f"  ref{oid} frame {fid}: {bilgi} ({dt:.0f}ms)")
        plt.tight_layout()
        out = OUT / f"gorev3_fix_reference_{oid}.png"
        plt.savefig(out, dpi=80, bbox_inches="tight"); plt.close()
        print(f"kaydedildi: {out}\n")
    print("GOREV3 DUZELTME GORSEL TESTI TAMAM")


if __name__ == "__main__":
    main()
