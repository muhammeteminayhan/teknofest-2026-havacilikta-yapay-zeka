# -*- coding: utf-8 -*-
"""GOREV 2 — TAVAN OLCUMU: RANSAC inlier kumesi uzaysal olarak KUMELENIYOR MU?

NEDEN VAR
---------
Manus raporu (Cozum 1, nokta secimi) "mumkunse goruntu izgarasina dengeli dagitim"
oneriyor. **Girdi tarafinda bu zaten var:** `raft_matching.py` noktalari `grid_step=28`
ile duzgun bir meshgrid'den orneklüyor.

Geriye kalan gercek soru sudur: RANSAC'in SECTIGI inlier kumesi kadrajin tamamina
yayiliyor mu, yoksa bir bolgede mi topluyor? Merdiven katmaninda inlier orani yalniz
0,369 — kalan noktalar tek bir kosede kumelenmisse duzlem fiti yanlidir ve
"yapi-farkinda / uzaysal dengelenmis inlier secimi" (OLCUM_TURU_2.md §5'te
"denenmedi") anlamli bir is olur. Kumelenme yoksa o is kapanir.

BU BETIK URETIM KODUNA DOKUNMAZ. Nokta onbelleginden okur ve uretimdekiyle AYNI
cagriyi (`cv2.findHomography`, ayni esik) yapip mask'i disaridan olcer.

OLCUTLER
--------
  kapsama      : 4x4 izgarada inlier iceren hucre orani (1,00 = tam yayilim)
  merkez kaymasi: inlier agirlik merkezinin TUM noktalarin merkezinden sapmasi,
                  kadraj kosegenine bolunerek normalize (0 = sapma yok)
  yayilim orani : inlier bulutunun ozdeger izinin tum noktalarinkine orani
                  (1,00 = inlier'lar girdiyle ayni kadar yayilmis, <1 = daralmis)

CALISTIR
--------
    .venv\\Scripts\\python gorev2\\tavan_inlier_dagilimi.py
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np

GOREV2 = Path(__file__).resolve().parent
sys.path.insert(0, str(GOREV2))

from nokta_onbellek import YATAKLAR, onbellek_yukle, yatak_npz    # noqa: E402
from gorev2_src.vo_pipeline import EstimatorConfig                # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")


def dagilim_olc(pts0: np.ndarray, mask: np.ndarray, izgara: int = 4) -> dict:
    """Inlier kumesinin uzaysal dagilim olcutleri."""
    inl = pts0[mask]
    if len(inl) < 4:
        return {}
    x0, y0 = pts0.min(axis=0)
    x1, y1 = pts0.max(axis=0)
    w, h = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)

    # 4x4 hucre kapsamasi — girdinin doldurdugu hucrelere gore
    gi = np.clip(((inl[:, 0] - x0) / w * izgara).astype(int), 0, izgara - 1)
    gj = np.clip(((inl[:, 1] - y0) / h * izgara).astype(int), 0, izgara - 1)
    ti = np.clip(((pts0[:, 0] - x0) / w * izgara).astype(int), 0, izgara - 1)
    tj = np.clip(((pts0[:, 1] - y0) / h * izgara).astype(int), 0, izgara - 1)
    dolu_inl = len(set(zip(gi.tolist(), gj.tolist())))
    dolu_tum = len(set(zip(ti.tolist(), tj.tolist())))

    kayma = np.linalg.norm(inl.mean(axis=0) - pts0.mean(axis=0)) / np.hypot(w, h)
    iz_inl = float(np.trace(np.cov(inl.T)))
    iz_tum = float(np.trace(np.cov(pts0.T)))
    return dict(kapsama=dolu_inl / max(dolu_tum, 1),
                merkez_kaymasi=float(kayma),
                yayilim_orani=iz_inl / max(iz_tum, 1e-9),
                inlier_orani=float(mask.mean()))


def yatak_olc(ad: str, katman: str, esik: float, n_ornek: int) -> list[dict]:
    y = YATAKLAR[ad]
    spacing = 8 if katman == "merdiven" else None
    ob = onbellek_yukle(yatak_npz(ad, spacing, "raft"))
    pts0, pts1, off = ob["pts0"], ob["pts1"], ob["offsets"]
    n_cift = len(off) - 1
    idx = np.linspace(0, n_cift - 1, min(n_ornek, n_cift)).astype(int)
    out = []
    for i in idx:
        a, b = off[i], off[i + 1]
        if b - a < 20:
            continue
        p0, p1 = pts0[a:b].astype(np.float64), pts1[a:b].astype(np.float64)
        H, mask = cv2.findHomography(p0, p1, cv2.RANSAC, esik)
        if H is None or mask is None:
            continue
        m = dagilim_olc(p0, mask.reshape(-1).astype(bool))
        if m:
            out.append(m)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yatak", nargs="*",
                    default=["yarisma", "video1", "video2", "v25_rgb_b", "v25_termal_b"])
    ap.add_argument("--n-ornek", type=int, default=150)
    ap.add_argument("--esik", type=float, default=1.0)
    args = ap.parse_args()

    # Sevk edilen deger: her kamerada 1,0 dogal piksel. (`calisma_esigi` ile kameraya
    # gore normalize etme denendi ve yillar arasi sinavda tasinmadi — YONTEM_SINAVI.md;
    # varsayilan tarihsel davranista birakildi, bkz. vo_pipeline.from_points.)
    esik = args.esik

    print("=" * 88)
    print(f"INLIER UZAYSAL DAGILIMI — RANSAC esigi {esik}, yatak basina {args.n_ornek} cift")
    print("=" * 88)
    print(f"\n  {'yatak':<15}{'katman':<11}{'inlier':>9}{'kapsama':>10}"
          f"{'merkez kay.':>13}{'yayilim':>10}")
    print("  " + "-" * 68)
    for ad in args.yatak:
        for katman in ("ardisik", "merdiven"):
            try:
                r = yatak_olc(ad, katman, esik, args.n_ornek)
            except Exception as e:
                print(f"  {ad:<15}{katman:<11} ATLANDI ({type(e).__name__})")
                continue
            if not r:
                print(f"  {ad:<15}{katman:<11} olcum yok")
                continue
            g = lambda k: np.median([x[k] for x in r])
            print(f"  {ad:<15}{katman:<11}{g('inlier_orani'):>9.3f}"
                  f"{g('kapsama'):>10.3f}{g('merkez_kaymasi'):>13.3f}"
                  f"{g('yayilim_orani'):>10.3f}", flush=True)

    print("\n  YORUM ANAHTARI")
    print("   kapsama ~1,00 ve merkez kaymasi ~0,00  -> inlier'lar kadraja yayilmis,")
    print("                                            'uzaysal dengeleme' isi KAPANIR")
    print("   kapsama <0,80 ya da merkez kaymasi >0,05 -> kumelenme var, is ACIK")


if __name__ == "__main__":
    main()
