# -*- coding: utf-8 -*-
"""GOREV 2 — METRE/PIKSEL KAHINI: dogru tanimlanmis yukseklik kahini.

NEDEN VAR
---------
`kahin_yukseklik.py` (YUKSEKLIK_KAHINI.md) "kusursuz yukseklik bile 5/7 yatakta zarar
veriyor" sonucunu verdi ve mutlak yukseklik gozlemi ailesini kapatti. Ama o kahin
yuksekligi `current_height(gt_z)` ile uretiyor: kalibrasyondaki olcek `c` + GT IRTIFA
degisimi (duz arazi varsayimi). Goruntu olcegini belirleyen ise YERDEN yukseklik (AGL):
finalin O4 oturumunda saglikli donemde ucak 47,7 m inerken AGL sabit kaldi (arazi de
dusuyordu). Yani eski kahin, arac/ped gozleminin olctugu seyi degil, irtifayi enjekte etti.

Arac boyu ya da 4,5 m'lik ped bir METRE/PIKSEL (GSD) gozlemidir. Bunun dogru kahini:
kesintide son W karede GT YOL UZUNLUGU / kestiricinin KENDI yol uzunlugu oranina gore
yukseklik durumunu duzeltmek. Duz arazi varsayimi yok, yon bilgisi yok — yalniz olcek.

MODLAR
------
  yok    : mevcut sistem
  eski   : kahin_yukseklik.py'nin kahini (current_height(gt_z))
  olcek  : metre/piksel kahini (bu betik)

TAKVIMLER
---------
  cevrimici : cevrim ici puanli oturumun takvimi (3 kesinti, 450-600 kare)
  final     : 450 saglikli + 1800 kesinti (finalde Test/O2/O4'un birebir yapisi)

CALISTIR
--------
    .venv\\Scripts\\python gorev2\\kahin_olcek.py
    .venv\\Scripts\\python gorev2\\kahin_olcek.py --n-perm 2 --yatak video1 video2
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

GOREV2 = Path(__file__).resolve().parent
sys.path.insert(0, str(GOREV2))

from kabul import _yukle, _geo_birlestir, yarisma_takvimi              # noqa: E402
from final_yatak import kos                                            # noqa: E402
from gorev2_src.evaluation import _meas_from_row, competition_error    # noqa: E402
from gorev2_src.vo_pipeline import EstimatorConfig                     # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

YATAKLAR = ["video1", "video2", "v25_rgb_a", "v25_rgb_b", "v25_termal_a", "v25_termal_b"]


def final_takvimi(n: int) -> np.ndarray:
    h = np.zeros(n, bool)
    h[:450] = True
    return h


TAKVIMLER = {"cevrimici": yarisma_takvimi, "final": final_takvimi}


def kanca_kur(mod: str, gt: np.ndarray, saglikli: np.ndarray, W: int = 40, aralik: int = 8,
              sinir: tuple[float, float] = (0.8, 1.25)):
    """Kesinti karelerinde yukseklik durumunu kahinle duzelten kanca."""
    if mod == "yok":
        return None

    def kanca(k, est, preds):
        if saglikli[k]:
            return
        a = est._anchor
        if mod == "eski":
            hiz = est.alignment
            if hiz is None:
                return
            if est.height is not None:
                est.height = est._clamp_h(hiz.current_height(float(gt[k - 1, 2])))
            if a is not None and a.get("h") is not None:
                a["h"] = est._clamp_h(hiz.current_height(float(gt[int(a["ord"]), 2])))
            return
        # mod == "olcek": yalniz kesinti karelerinden olusan son W karelik pencere
        if k % aralik or est.height is None:
            return
        bas = k - 1
        while bas > max(0, k - 1 - W) and not saglikli[bas - 1]:
            bas -= 1
        if k - 1 - bas < 10:
            return
        L_gt = np.linalg.norm(np.diff(gt[bas:k, :2], axis=0), axis=1).sum()
        L_pr = np.linalg.norm(np.diff(preds[bas:k, :2], axis=0), axis=1).sum()
        if L_gt < 3.0 or L_pr < 0.5:
            return
        rho = float(np.clip(L_gt / L_pr, *sinir))
        est.height = est._clamp_h(est.height * rho)
        if a is not None and a.get("h") is not None:
            a["h"] = est._clamp_h(a["h"] * rho)

    return kanca


def olc(yatak: str, takvim: str, mod: str, karistir: int, cfg: EstimatorConfig) -> float:
    gt, cons, lad, prof, n = _yukle(yatak, 8, karistir, _geo_birlestir(cfg, {}))
    cr = [_meas_from_row(r) for r in cons.to_dict("records")]
    lr = [_meas_from_row(r) for r in lad.to_dict("records")]
    s = TAKVIMLER[takvim](n)
    p = kos(cr, lr, gt, s, prof, cfg, kanca=kanca_kur(mod, gt, s))
    return competition_error(p, gt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=2)
    ap.add_argument("--yatak", nargs="*", default=YATAKLAR)
    ap.add_argument("--takvim", nargs="*", default=list(TAKVIMLER))
    a = ap.parse_args()
    cfg = EstimatorConfig(method="homografi")
    modlar = ("yok", "eski", "olcek")
    print(f"{'yatak':14s} {'takvim':10s} " + " ".join(f"{m:>9s}" for m in modlar)
          + "   eski fark   olcek fark")
    ozet = {m: [] for m in modlar}
    for y in a.yatak:
        for t in a.takvim:
            r = {m: float(np.mean([olc(y, t, m, i, cfg) for i in range(a.n_perm)])) for m in modlar}
            for m in modlar:
                ozet[m].append(r[m])
            print(f"{y:14s} {t:10s} " + " ".join(f"{r[m]:9.2f}" for m in modlar)
                  + f"   {100 * (r['eski'] / r['yok'] - 1):+8.0f}%  {100 * (r['olcek'] / r['yok'] - 1):+9.0f}%",
                  flush=True)
    print("-" * 80)
    print(f"{'ORTALAMA':25s} " + " ".join(f"{np.mean(ozet[m]):9.2f}" for m in modlar))
    for m in ("eski", "olcek"):
        kaz = sum(b < y for b, y in zip(ozet[m], ozet["yok"]))
        print(f"  {m}: {kaz}/{len(ozet[m])} yatak-takvimde mevcuttan iyi")


if __name__ == "__main__":
    main()
