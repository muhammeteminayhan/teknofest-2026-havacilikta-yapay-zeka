# -*- coding: utf-8 -*-
"""GOREV 2 — LOOP CLOSURE (yer tanima) A/B testi, resmi metrikle.

Iki asama:
  1) BETIMLEYICI ONBELLEGI: ornek video karelerinden DINOv3 CLS betimleyicileri
     bir kez cikarilip `outputs/desc_video1.npy` dosyasina yazilir (~2250 x 384).
     Bir daha video okunmaz -> parametre A/B'leri saniyeler icinde kosar.
  2) A/B: ayni test yatagi (gercek yarisma kesinti takvimi) uzerinde
     loop closure KAPALI vs ACIK karsilastirilir; ayrica esik taramasi yapilir.

KABUL OLCUTU (asiri uyuma ve yanlis eslesmeye karsi):
  - yarismanin gercek takviminde hata dusmeli,
  - farkli rastgele takvimlerin ortalamasinda da dusmeli,
  - HICBIR kesinti blogunda hata artmamali (yanlis eslesme koruması).

Calistir (proje kokunden):
    .venv\\Scripts\\python gorev2\\loop_closure_test.py            # onbellek + A/B
    .venv\\Scripts\\python gorev2\\loop_closure_test.py --sadece-onbellek
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GOREV2 = Path(__file__).resolve().parent
OUT = GOREV2 / "degerlendirme"

_SRC = GOREV2 / "src"
if "gorev2_src" not in sys.modules:
    _spec = importlib.util.spec_from_file_location(
        "gorev2_src", _SRC / "__init__.py", submodule_search_locations=[str(_SRC)])
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules["gorev2_src"] = _mod
    _spec.loader.exec_module(_mod)
sys.path.insert(0, str(ROOT))

from gorev2_src import data_io, evaluation                        # noqa: E402
from gorev2_src.calibration import RGB_1080P                      # noqa: E402
from gorev2_src.vo_pipeline import EstimatorConfig, VoLadderEstimator  # noqa: E402
from gorev2_src.yer_tanima import YerTanima                       # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

VIDEO = ROOT / "THYZ_2026_Ornek_Veri_Seti" / "THYZ_2026_Ornek_Veri_1.MP4"
GT_CSV = ROOT / "THYZ_2026_Ornek_Veri_Seti" / "THYZ_2026_Ornek_Veri_1_translation.csv"
CACHE_CONS = GOREV2 / "outputs" / "olcumler_video1_raft.csv"
CACHE_LADDER = GOREV2 / "outputs" / "merdiven_video1_raft.csv"
DESC_NPY = GOREV2 / "outputs" / "desc_video1.npy"

KESINTI_BLOKLARI = [(451, 900), (1051, 1550), (1651, 2250)]
N_KARE = 2250
SPACING = 8
SMOOTH_ALPHA = 1.0     # parametre taramasinda kazanan ayar


def yarisma_takvimi(n=N_KARE):
    h = np.ones(n, dtype=bool)
    for bas, son in KESINTI_BLOKLARI:
        h[bas:son] = False
    return h


def betimleyici_onbellegi(n_hedef=N_KARE) -> np.ndarray:
    """Video karelerinden DINOv3 CLS betimleyicilerini bir kez cikar (onbellekli)."""
    if DESC_NPY.exists():
        d = np.load(DESC_NPY)
        print(f"betimleyici onbellekten yuklendi: {DESC_NPY.name} {d.shape}")
        return d
    print("Betimleyiciler cikariliyor (video bir kez taranacak)...")
    from gorev3.dino_backbone import build_backbone
    yt = YerTanima(backbone=build_backbone("dinov3"))
    t0 = time.perf_counter()
    descs = []
    for i, (_, frame) in enumerate(data_io.iter_competition_frames(VIDEO, gray=False,
                                                                   progress=True)):
        if i >= n_hedef:
            break
        d = yt.betimleyici(frame)
        descs.append(d if d is not None else np.zeros(descs[0].shape if descs else 384,
                                                      dtype=np.float32))
    D = np.stack(descs).astype(np.float32)
    DESC_NPY.parent.mkdir(parents=True, exist_ok=True)
    np.save(DESC_NPY, D)
    print(f"{D.shape} betimleyici, {time.perf_counter()-t0:.0f} s -> {DESC_NPY.name}")
    return D


def kos(cons_rows, ladder_rows, gt, healthy, cfg, desc=None, yt_kwargs=None):
    """Kestirici kosumu; `desc` verilirse loop closure ACIK."""
    n = len(gt)
    est = VoLadderEstimator(RGB_1080P, cfg, spacing=SPACING)
    preds = np.zeros((n, 3))
    preds[0] = gt[0]
    est.update_ladder(0, None, None, gt[0], True)
    yt = YerTanima(**(yt_kwargs or {})) if desc is not None else None
    if yt is not None and healthy[0]:
        yt.ekle(desc[0], gt[0])
    for k in range(1, n):
        mc = cons_rows[k - 1] if k - 1 < len(cons_rows) else None
        ml = ladder_rows[k - 1] if k - 1 < len(ladder_rows) else None
        p = est.update_ladder(k, mc, ml, gt[k] if healthy[k] else None, bool(healthy[k]))
        if yt is not None:
            if healthy[k]:
                yt.ekle(desc[k], gt[k])          # saglikli: haritaya ekle
            else:
                aday = yt.sorgula(desc[k], p)    # kesinti: yer tanima sorgusu
                if aday is not None:
                    p = yt.duzelt(p, aday)
                    est.pos = p.copy()           # durum makinesini senkron tut
        preds[k] = p
    return preds, (yt.tani() if yt is not None else None)


def blok_hatalari(preds, gt, healthy):
    hata = np.linalg.norm(preds - gt, axis=1)
    out = []
    bas = None
    for i in range(len(healthy) + 1):
        icinde = i < len(healthy) and not healthy[i]
        if icinde and bas is None:
            bas = i
        elif not icinde and bas is not None:
            out.append((bas, i - 1, float(hata[bas:i].mean()), float(hata[i - 1])))
            bas = None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sadece-onbellek", action="store_true")
    ap.add_argument("--senaryo", type=int, default=4)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    desc = betimleyici_onbellegi()
    if args.sadece_onbellek:
        return

    gt_sub = data_io.subsample_competition(data_io.load_translations(GT_CSV))
    cons = pd.read_csv(CACHE_CONS)
    ladder = pd.read_csv(CACHE_LADDER)
    n = min(N_KARE, len(gt_sub), len(cons) + 1, len(ladder) + 1, len(desc))
    gt_sub = gt_sub.iloc[:n].reset_index(drop=True)
    gt = gt_sub[evaluation.GT_COLS].to_numpy()
    desc = desc[:n]
    cons_rows = [evaluation._meas_from_row(r) for r in cons.iloc[:n - 1].to_dict("records")]
    ladder_rows = [evaluation._meas_from_row(r) for r in ladder.iloc[:n - 1].to_dict("records")]

    cfg = replace(EstimatorConfig(method="homografi"), smooth_alpha=SMOOTH_ALPHA)
    yarisma = yarisma_takvimi(n)
    rastgele = [evaluation.make_outage_schedule(n_frames=n, rng=s) for s in range(args.senaryo)]

    # --- TEMEL: loop closure KAPALI ---
    p0, _ = kos(cons_rows, ladder_rows, gt, yarisma, cfg)
    t_yar = evaluation.competition_error(p0, gt)
    t_ras = float(np.mean([evaluation.competition_error(
        kos(cons_rows, ladder_rows, gt, h, cfg)[0], gt) for h in rastgele]))
    t_blok = blok_hatalari(p0, gt, yarisma)
    print(f"=== TEMEL (loop closure KAPALI, smooth_alpha={SMOOTH_ALPHA}) ===")
    print(f"  yarisma {t_yar:.3f} m | {args.senaryo} rastgele ort {t_ras:.3f} m\n")

    # --- ACIK: esik taramasi ---
    print(f"{'min_benz':>9} {'lowe':>6} {'alpha':>6} {'yarisma':>10} {'rastgele':>10} "
          f"{'kabul':>7} {'karar':>8}")
    print("-" * 68)
    satirlar = []
    en_iyi = None
    for min_b in (0.70, 0.80, 0.88):
        for lowe in (1.02, 1.08):
            for alpha in (0.15, 0.30, 0.50):
                kw = dict(min_benzerlik=min_b, lowe_orani=lowe, alpha=alpha)
                p, tani = kos(cons_rows, ladder_rows, gt, yarisma, cfg, desc, kw)
                y = evaluation.competition_error(p, gt)
                r = float(np.mean([evaluation.competition_error(
                    kos(cons_rows, ladder_rows, gt, h, cfg, desc, kw)[0], gt)
                    for h in rastgele]))
                blok = blok_hatalari(p, gt, yarisma)
                blok_bozdu = any(b[2] > t[2] + 1e-6 for b, t in zip(blok, t_blok))
                iyi = (y < t_yar * 0.99) and (r < t_ras * 0.99) and not blok_bozdu
                karar = "KABUL" if iyi else ("blok+" if blok_bozdu else "-")
                print(f"{min_b:>9.2f} {lowe:>6.2f} {alpha:>6.2f} {y:>9.3f}m {r:>9.3f}m "
                      f"{tani['kabul']:>7} {karar:>8}")
                satirlar.append([min_b, lowe, alpha, y, r, tani, karar])
                if iyi and (en_iyi is None or (y + r) < (en_iyi[3] + en_iyi[4])):
                    en_iyi = [min_b, lowe, alpha, y, r, tani]

    L = ["=== GOREV 2 — LOOP CLOSURE (yer tanima) A/B ===",
         f"temel (KAPALI): yarisma {t_yar:.3f} m | {args.senaryo} rastgele ort {t_ras:.3f} m",
         f"config: homografi, spacing={SPACING}, smooth_alpha={SMOOTH_ALPHA}",
         "",
         "Kabul olcutu: yarisma VE rastgele ortalamasinda >=%1 kazanc, ayrica",
         "HICBIR kesinti blogunda ortalama hata artmamali (yanlis eslesme koruması).",
         ""]
    L.append(f"{'min_benz':>9} {'lowe':>6} {'alpha':>6} {'yarisma':>10} {'rastgele':>10} {'karar':>8}")
    for s in satirlar:
        L.append(f"{s[0]:>9.2f} {s[1]:>6.2f} {s[2]:>6.2f} {s[3]:>9.3f}m {s[4]:>9.3f}m {s[6]:>8}")
    L.append("")
    if en_iyi:
        mb, lo, al, y, r, tani = en_iyi
        L += ["--- EN IYI AYAR ---",
              f"  min_benzerlik={mb}, lowe_orani={lo}, alpha={al}",
              f"  yarisma  : {y:.3f} m  (temel {t_yar:.3f} m, {100*(y-t_yar)/t_yar:+.1f}%)",
              f"  rastgele : {r:.3f} m  (temel {t_ras:.3f} m, {100*(r-t_ras)/t_ras:+.1f}%)",
              f"  tani     : {tani}",
              "",
              ">>> TAVSIYE: loop closure ACIK (yukaridaki ayarla)"]
    else:
        L += [">>> TAVSIYE: loop closure KAPALI kalsin — hicbir ayar tum kapilardan gecmedi.",
              "    (yer tanima bu veride guvenilir kazanc saglamiyor; yanlis eslesme riski",
              "     kazanctan buyuk)"]
    metin = "\n".join(L)
    (OUT / "loop_closure.txt").write_text(metin, encoding="utf-8")
    print()
    print(metin)
    print(f"\nCikti: {OUT}")


if __name__ == "__main__":
    main()
