# -*- coding: utf-8 -*-
"""GOREV 2 — TAVAN OLCUMU: VGGT'nin PARCA ICI geometrisi ne kadar iyi?

NEDEN VAR
---------
`SLAM_TURU.md`: VGGT zincirlenmis yorungesi tamamen bozuk (125,7 m; yon hatasi 102°,
olcek hatasi %42,9) ama **ilk 50 karede kahin Sim(3) ile 0,97 m** — yani sorun modelde
degil, PARCA BIRLESTIRMEDE. Bu tespit tek bir pencereden geliyordu.

Bu betik onu butun diziye yayiyor ve iki sey olcuyor:

  1. **Parca ici kalite** — her parcayi GT'ye ayri ayri kahin Sim(3) ile oturt, artik
     hatayi olc. Bu, HERHANGI bir birlestirme yonteminin ULASABILECEGI EN IYI sonuctur.
     Buyukse VGGT'nin yerel geometrisi yetersiz demektir ve is kapanir.

  2. **Parcalar arasi olcek tutarliligi** — her parcanin kendi kahin olcegi `s_c`.
     `s_c` parcadan parcaya ne kadar oynuyor? Sabite yakinsa zincirleme hic gerekmez,
     tek bir global olcek yeter. Cok oynuyorsa zincirlemenin savasmasi gereken sey budur.

NOT: `poz_vggt.npz` zincirlenmis GLOBAL konumlari tutar, ama parca basina Sim(3)
uygulamak icin bu yeterlidir — benzerligin benzerligi yine benzerliktir, parca ICI
geometri bozulmadan kalir.

CALISTIR
--------
    .venv\\Scripts\\python gorev2\\tavan_vggt_parca.py
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

GOREV2 = Path(__file__).resolve().parent
ROOT = GOREV2.parent
sys.path.insert(0, str(GOREV2))

import kabul                                                       # noqa: E402,F401  (gorev2_src'i sys.modules'e kurar)
from gorev2_src import data_io                                     # noqa: E402
from vggt_poz import sim3_fit                                      # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

NPZ = GOREV2 / "outputs" / "poz_vggt.npz"
GT_CSV = ROOT / "THYZ_2026_Ornek_Veri_Seti" / "THYZ_2026_Ornek_Veri_1_translation.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parca", type=int, default=8)
    ap.add_argument("--adim", type=int, default=4, help="parca - ortusme")
    args = ap.parse_args()

    d = np.load(NPZ)
    P = d["konumlar"]
    gt_sub = data_io.subsample_competition(data_io.load_translations(GT_CSV))
    G = gt_sub[["translation_x", "translation_y", "translation_z"]].to_numpy()
    n = min(len(P), len(G))
    P, G = P[:n], G[:n]
    print(f"VGGT pozlari: {n} kare  ·  GT: {len(G)} kare\n")

    print("=" * 76)
    print("1) PARCA ICI KALITE — her parca GT'ye kahin Sim(3) ile oturtuldu")
    print("=" * 76)
    artiklar, olcekler, yollar, basla = [], [], [], []
    for bas in range(0, n - args.parca + 1, args.adim):
        son = bas + args.parca
        fit = sim3_fit(P[bas:son], G[bas:son])
        if fit is None:
            continue
        s, R, t = fit
        if not np.isfinite(s) or s <= 0:
            continue
        Q = (s * (R @ P[bas:son].T).T) + t
        artiklar.append(float(np.linalg.norm(Q - G[bas:son], axis=1).mean()))
        olcekler.append(s)
        yollar.append(float(np.linalg.norm(np.diff(G[bas:son], axis=0), axis=1).sum()))
        basla.append(bas)
    a = np.array(artiklar); sc = np.array(olcekler); yol = np.array(yollar)
    print(f"  parca sayisi              : {len(a)}  (parca={args.parca}, adim={args.adim})")
    print(f"  parca ici artik hata [m]  : medyan {np.median(a):.3f} · "
          f"ort {a.mean():.3f} · p95 {np.percentile(a, 95):.3f} · maks {a.max():.3f}")
    print(f"  parca basina GT yolu [m]  : medyan {np.median(yol):.2f}")
    oran = a / np.maximum(yol, 1e-9)
    print(f"  artik / yol               : medyan %{100*np.median(oran):.1f} · "
          f"p95 %{100*np.percentile(oran, 95):.1f}")

    print("\n" + "=" * 76)
    print("2) PARCALAR ARASI OLCEK TUTARLILIGI — her parcanin kahin olcegi s_c")
    print("=" * 76)
    print(f"  s_c medyan                : {np.median(sc):.4f}")
    print(f"  s_c degisim katsayisi     : %{100*sc.std()/max(abs(sc.mean()),1e-9):.1f}")
    print(f"  s_c araligi (p05-p95)     : {np.percentile(sc,5):.4f} - {np.percentile(sc,95):.4f}")
    orann = sc[1:] / np.maximum(sc[:-1], 1e-12)
    print(f"  ardisik parca olcek orani : medyan {np.median(orann):.4f} · "
          f"ort |sapma| %{100*np.abs(orann-1).mean():.1f}")

    print("\n" + "=" * 76)
    print("3) TEK GLOBAL OLCEK YETER MI? (zincirleme olmadan)")
    print("=" * 76)
    s_tek = float(np.median(sc))
    fit_g = sim3_fit(P, G)
    if fit_g is not None:
        sg, Rg, tg = fit_g
        Qg = (sg * (Rg @ P.T).T) + tg
        print(f"  TUM diziye tek Sim(3)     : {np.linalg.norm(Qg - G, axis=1).mean():.2f} m"
              f"  (olcek {sg:.4f})")
    # parca basina yalniz OTELEME serbest, olcek+donme global
    if fit_g is not None:
        hata = []
        for bas, in zip(basla):
            son = bas + args.parca
            Qc = (sg * (Rg @ P[bas:son].T).T)
            Qc = Qc + (G[bas:son].mean(0) - Qc.mean(0))
            hata.append(float(np.linalg.norm(Qc - G[bas:son], axis=1).mean()))
        print(f"  global olcek+donme, parca basina yalniz oteleme: "
              f"{np.mean(hata):.3f} m")

    # ------------------------------------------------------------------ kiyas
    print("\n" + "=" * 76)
    print("4) KIYAS — AYNI OLCUT mevcut sistemde (RAFT + homografi)")
    print("=" * 76)
    try:
        from kabul import _yukle, _geo_birlestir, _kos
        from gorev2_src.vo_pipeline import EstimatorConfig
        cfg = EstimatorConfig(method="homografi")
        gtr, cons, lad, prof, nn = _yukle("video1", 8, 0, _geo_birlestir(cfg, {}))
        # tum kareler SAGLIKLI kosulsa sistem GT'yi aynalar; olcum kalitesini gormek icin
        # tamami KESINTI olan bir takvim kullaniyoruz (ilk 60 kare kalibrasyon icin acik)
        h = np.zeros(nn, bool); h[:60] = True
        pr = _kos(cons, lad, gtr, h, prof, cfg, 8)
        ar = []
        for bas in range(60, nn - args.parca + 1, args.adim):
            son = bas + args.parca
            f = sim3_fit(pr[bas:son], gtr[bas:son])
            if f is None:
                continue
            s2, R2, t2 = f
            if not np.isfinite(s2) or s2 <= 0:
                continue
            Q2 = (s2 * (R2 @ pr[bas:son].T).T) + t2
            yl = float(np.linalg.norm(np.diff(gtr[bas:son], axis=0), axis=1).sum())
            ar.append(float(np.linalg.norm(Q2 - gtr[bas:son], axis=1).mean()) / max(yl, 1e-9))
        ar = np.array(ar)
        print(f"  RAFT parca ici artik/yol  : medyan %{100*np.median(ar):.1f} · "
              f"p95 %{100*np.percentile(ar, 95):.1f}   ({len(ar)} parca)")
        print(f"  VGGT parca ici artik/yol  : medyan %{100*np.median(oran):.1f} · "
              f"p95 %{100*np.percentile(oran, 95):.1f}   ({len(oran)} parca)")
    except Exception as e:
        print(f"  kiyas kosulamadi ({type(e).__name__}: {e})")

    print("\n  YORUM ANAHTARI")
    print("   (1) kucukse (yol'un %5'inden az): VGGT yerel geometrisi SAGLAM,")
    print("       sorun yalniz birlestirmede -> is ACIK")
    print("   (2) s_c cok oynuyorsa: zincirlemenin savastigi sey bu; tek global olcek")
    print("       yetmez. Oynamiyorsa zincirleme GEREKSIZ, tek olcek yeter.")


if __name__ == "__main__":
    main()
