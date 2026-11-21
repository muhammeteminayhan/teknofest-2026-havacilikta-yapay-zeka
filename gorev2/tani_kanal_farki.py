# -*- coding: utf-8 -*-
"""GOREV 2 — TANI: motorlar arasindaki fark HANGI KANALDA?

NEDEN VAR
---------
`tani_akis_farki.py` bir paradoks buldu. SEA-RAFT (ozellikle `searaft_g`):
  * akis yanliligi YOK        (olcek orani 0,9996 · aci farki 0,06°)
  * nokta sayisi ayni         (570 vs 575)
  * homografi uyumu DAHA IYI  (inlier 0,560 vs 0,532 · artik 0,359 vs 0,385 px)

...ama yorunge hatasi daha kotu (video1: 4,27 vs 3,90).

Demek ki fark, 2B yeniden-izdusum artiginin GORMEDIGI bir yerde: homografinin
(R, t, n) ayristirmasinda. Bu betik `procrustes_tani` ile hatayi kanallara ayirir —
YON (derece) ve OLCEK (%) — ve hangisinin bozuldugunu soyler.

CALISTIR
--------
    .venv\\Scripts\\python gorev2\\tani_kanal_farki.py
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

GOREV2 = Path(__file__).resolve().parent
sys.path.insert(0, str(GOREV2))

import kabul                                                        # noqa: E402
from kabul import yatak_olc, yatak_npz                              # noqa: E402
from gorev2_src.vo_pipeline import EstimatorConfig                  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yatak", default="video1")
    ap.add_argument("--motor", nargs="*",
                    default=["raft", "searaft", "searaft_l", "searaft_g", "searaft_gl"])
    ap.add_argument("--n-perm", type=int, default=3)
    args = ap.parse_args()

    cfg = EstimatorConfig(method="homografi")
    print("=" * 78)
    print(f"KANAL FARKI — yatak {args.yatak}, {args.n_perm} permutasyon")
    print("=" * 78)
    print(f"\n  {'motor':<14}{'hata [m]':>12}{'uzun kesinti':>15}"
          f"{'YON [deg]':>12}{'OLCEK [%]':>12}")
    print("  " + "-" * 65)

    for m in args.motor:
        if not (yatak_npz(args.yatak, None, m).exists()
                and yatak_npz(args.yatak, 8, m).exists()):
            print(f"  {m:<14}{'onbellek yok':>12}")
            continue
        kabul.MOTOR = m
        R = [yatak_olc(args.yatak, cfg, 8, k, {}) for k in range(args.n_perm)]
        g = lambda k: float(np.mean([r[k] for r in R]))
        print(f"  {m:<14}{g('ana'):>12.3f}{g('uzun'):>15.3f}"
              f"{g('yon'):>12.3f}{g('olcek'):>12.3f}", flush=True)
    kabul.MOTOR = "raft"

    print("\n  YORUM ANAHTARI")
    print("   YON bozuluyorsa: homografinin donme ayristirmasi daha gurultulu")
    print("   OLCEK bozuluyorsa: duzlem uzakligi (t/d) kestirimi daha gurultulu")
    print("   Ikisi de esitse: fark zamansal tutarlilikta (kare kare sacilim)")


if __name__ == "__main__":
    main()
