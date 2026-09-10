# -*- coding: utf-8 -*-
"""GOREV 2 — VGGT: parca ici kalite COZUNURLUGE mi takili?

NEDEN VAR
---------
`tavan_vggt_parca.py` sunu olctu: VGGT'nin parca ici artik hatasi GT yolunun %8,1'i
(kahin Sim(3) ile), RAFT'inki %1,0. Yani VGGT'nin YEREL geometrisi de yetersiz —
sorun yalniz parca birlestirmede degil.

Ama o olcum **294 px** girdiyle yapildi ve bu bir tercih degil ZORUNLULUKTU:
8,5 GB GPU'da 518 px paging'e giriyor ve 40x yavasliyor (`SLAM_TURU.md`).

Bu betik su ayrimi yapar:
  * "VGGT bu veride kotu"                   -> aile kapanir
  * "VGGT iyi ama bizim donanimda calismaz" -> baska bir karar (ve baska bir GPU'da
                                               yeniden acilabilir)

Sinirli sayida parcada birkac cozunurluk denenir; olcut ayni: GT'ye kahin Sim(3) ile
oturtulan parcanin artik hatasi / parcanin GT yolu.

CALISTIR
--------
    .venv\\Scripts\\python gorev2\\vggt_cozunurluk.py
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np

GOREV2 = Path(__file__).resolve().parent
ROOT = GOREV2.parent
sys.path.insert(0, str(GOREV2))

import kabul                                                       # noqa: E402,F401
from gorev2_src import data_io                                     # noqa: E402
from vggt_poz import sim3_fit, _hazirla, _kamera_merkezleri, VIDEO  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

GT_CSV = ROOT / "THYZ_2026_Ornek_Veri_Seti" / "THYZ_2026_Ornek_Veri_1_translation.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-parca", type=int, default=25, help="cozunurluk basina parca")
    ap.add_argument("--parca", type=int, default=8)
    ap.add_argument("--adim", type=int, default=40, help="parcalar arasi kare atlamasi")
    ap.add_argument("--cozunurluk", type=int, nargs="*", default=[294, 392, 518])
    args = ap.parse_args()

    import torch
    from vggt.models.vggt import VGGT
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    gt_sub = data_io.subsample_competition(data_io.load_translations(GT_CSV))
    G = gt_sub[["translation_x", "translation_y", "translation_z"]].to_numpy()

    gerek = args.adim * args.n_parca + args.parca
    kareler = []
    for _, g in data_io.iter_competition_frames(VIDEO, progress=False):
        kareler.append(g)
        if len(kareler) >= gerek:
            break
    print(f"{len(kareler)} kare okundu · {args.n_parca} parca x {args.parca} kare\n")

    print("VGGT yukleniyor...")
    model = VGGT.from_pretrained("facebook/VGGT-1B").to("cuda").eval()

    print(f"\n  {'cozunurluk':>11}{'artik/yol medyan':>20}{'p95':>9}"
          f"{'s_c dgs.kats.':>15}{'s/parca':>10}{'tepe VRAM':>12}")
    print("  " + "-" * 77)

    for boyut in args.cozunurluk:
        oranlar, olcekler, sureler = [], [], []
        torch.cuda.reset_peak_memory_stats()
        patladi = None
        for i in range(args.n_parca):
            bas = i * args.adim
            son = bas + args.parca
            if son > len(kareler) or son > len(G):
                break
            try:
                imgs = _hazirla(kareler[bas:son], boyut).to("cuda")
                t0 = time.perf_counter()
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    pred = model(imgs[None])
                extri, _ = pose_encoding_to_extri_intri(pred["pose_enc"], imgs.shape[-2:])
                sureler.append(time.perf_counter() - t0)
                P = _kamera_merkezleri(extri[0].float().cpu().numpy())
                del imgs, pred
                torch.cuda.empty_cache()
            except RuntimeError as e:
                patladi = type(e).__name__
                break
            fit = sim3_fit(P, G[bas:son])
            if fit is None:
                continue
            s, R, t = fit
            if not np.isfinite(s) or s <= 0:
                continue
            Q = (s * (R @ P.T).T) + t
            yol = float(np.linalg.norm(np.diff(G[bas:son], axis=0), axis=1).sum())
            oranlar.append(float(np.linalg.norm(Q - G[bas:son], axis=1).mean()) / max(yol, 1e-9))
            olcekler.append(s)
        tepe = torch.cuda.max_memory_allocated() / 1e9
        if not oranlar:
            print(f"  {boyut:>11}   ATLANDI ({patladi})")
            continue
        o = np.array(oranlar); sc = np.array(olcekler)
        print(f"  {boyut:>11}{100*np.median(o):>19.1f}%{100*np.percentile(o,95):>8.1f}%"
              f"{100*sc.std()/max(abs(sc.mean()),1e-9):>14.0f}%"
              f"{np.mean(sureler):>10.2f}{tepe:>11.2f}G"
              + (f"  ({patladi} sonrasi {len(o)} parca)" if patladi else ""), flush=True)

    print("\n  KIYAS: RAFT + homografi ayni olcutte medyan %1,0 (p95 %2,7)")
    print("  Cozunurluk arttikca VGGT %1-2'ye yaklasmiyorsa sorun donanim degil,")
    print("  modelin bu veri turundeki yerel geometrisidir -> aile kapanir.")


if __name__ == "__main__":
    main()
