# -*- coding: utf-8 -*-
"""GOREV 2 — TANI: merdiven olcumu reddedildiginde ne oluyor?

NEDEN VAR
---------
Ensemble turu (degerlendirme/ENSEMBLE.md) sunu ortaya cikardi: SIFT "kurtarmasi"
yalniz RAFT merdiven inlier'i `min_inlier_ratio`nun altina dustugunde devreye giriyor
— yani kapinin olcumu ATTIGI karelerde. Su an o karelerde sistem ardisik zincire
dusuyor (hop 1).

Dogru soru "SIFT mi kullanalim" degil:
    **Merdiven reddedildiginde RAFT'la daha iyi ne yapabiliriz?**

Bu betik once problemin BUYUKLUGUNU olcer, sonra kisa atlamanin (hop 4) o karelerde
gecerli olup olmadigina bakar. Iki sey ogrenilir:

  1. Red orani ve o karelerin HATA PAYI — sorun kayda deger mi?
  2. Red anlarinda sp4 merdiveni kapiyi geciyor mu — kisa atlama bir CIKIS mi?

sp4 onbellegi yalniz 3 yatakta var (yarisma, video1, video2); diger yataklar icin
uretilmesi gerekir. Once bu ucunde olcup devam etmeye deger mi ona bakiyoruz.

CALISTIR
--------
    .venv\\Scripts\\python gorev2\\tani_merdiven_red.py
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

GOREV2 = Path(__file__).resolve().parent
sys.path.insert(0, str(GOREV2))

from kabul import PENCERELER, _yukle, _geo_birlestir, yarisma_takvimi   # noqa: E402
from nokta_onbellek import YATAKLAR                                     # noqa: E402
from gorev2_src.evaluation import _meas_from_row                        # noqa: E402
from gorev2_src.vo_pipeline import EstimatorConfig, VoLadderEstimator   # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")


def olc(ad, cfg, karistir, sp_kisa=4):
    gt, cons, lad8, prof, n = _yukle(ad, 8, karistir, _geo_birlestir(cfg, {}))
    try:
        _, _, lad4, _, n4 = _yukle(ad, sp_kisa, karistir, _geo_birlestir(cfg, {}))
        n = min(n, n4)
    except Exception:
        lad4 = None
    gt = gt[:n]

    if ad == "yarisma":
        gercek = np.load(YATAKLAR["yarisma"]["gt_npz"])["saglikli"]
        h = gercek[:n].copy()
        h[900:1051] = False          # pencere 1 takvimi
    else:
        h = yarisma_takvimi(n)

    est = VoLadderEstimator(prof, cfg, spacing=8)
    est.update_ladder(0, None, None, gt[0], True)
    cr = [_meas_from_row(r) for r in cons.iloc[:n - 1].to_dict("records")]
    l8 = [_meas_from_row(r) for r in lad8.iloc[:n - 1].to_dict("records")]
    l4 = ([_meas_from_row(r) for r in lad4.iloc[:n - 1].to_dict("records")]
          if lad4 is not None else None)

    preds = np.zeros((n, 3)); preds[0] = gt[0]
    kayit = []
    _orij = est._ladder_candidate
    kutu = {"k": 0, "kabul": {}}

    def _sarmal(meas, hop):
        r = _orij(meas, hop)
        kutu["kabul"][kutu["k"]] = (r is not None)
        return r

    est._ladder_candidate = _sarmal

    for k in range(1, n):
        kutu["k"] = k
        preds[k] = est.update_ladder(k, cr[k - 1], l8[k - 1],
                                     gt[k] if h[k] else None, bool(h[k]))
        if h[k]:
            continue
        kabul = kutu["kabul"].get(k)
        m8, m4 = l8[k - 1], (l4[k - 1] if l4 else None)
        kayit.append(dict(
            k=k, kabul=bool(kabul),
            hata=float(np.linalg.norm(preds[k] - gt[k])),
            i8=(m8.hom_inlier_ratio if m8 and m8.hom_ok else 0.0),
            i4=(m4.hom_inlier_ratio if m4 and m4.hom_ok else 0.0),
            n4=(m4.n_kept if m4 else 0),
        ))
    return kayit, cfg.min_inlier_ratio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yatak", nargs="*", default=["yarisma", "video1", "video2"])
    ap.add_argument("--n-perm", type=int, default=2)
    args = ap.parse_args()

    cfg = EstimatorConfig(method="homografi")
    print("=" * 92)
    print("MERDIVEN REDDI — buyukluk ve kisa atlama (sp4) cikisi")
    print("=" * 92)
    print(f"\n  {'yatak':<12}{'kesinti':>9}{'RED %':>8}{'RED kare hata':>15}"
          f"{'KABUL kare hata':>17}{'sp4 kapiyi geciyor':>21}")
    print("  " + "-" * 82)

    for ad in args.yatak:
        A = []
        for i in range(args.n_perm):
            try:
                r, esik = olc(ad, cfg, i)
            except Exception as e:
                print(f"  {ad:<12} ATLANDI ({type(e).__name__}: {e})")
                r = None
                break
            A.extend(r)
        if not A:
            continue
        kabul = np.array([x["kabul"] for x in A])
        hata = np.array([x["hata"] for x in A])
        i4 = np.array([x["i4"] for x in A])
        var4 = (i4 > 0).any()
        red = ~kabul
        kurtar = (i4[red] >= esik).mean() if var4 and red.sum() else np.nan
        print(f"  {ad:<12}{len(A):>9}{100*red.mean():>7.1f}%"
              f"{hata[red].mean() if red.sum() else np.nan:>15.2f}"
              f"{hata[kabul].mean() if kabul.sum() else np.nan:>17.2f}"
              f"{(100*kurtar if np.isfinite(kurtar) else np.nan):>20.1f}%", flush=True)

    print("\n  YORUM ANAHTARI")
    print("   RED kare hatasi KABUL karelerinden belirgin yuksekse: red anlari gercek")
    print("     bir zarar kaynagi -> is ACIK")
    print("   'sp4 kapiyi geciyor' yuksekse: kisa atlama o anlarda gecerli bir olcum")
    print("     verebiliyor -> ikinci motorsuz cikis VAR")


if __name__ == "__main__":
    main()
