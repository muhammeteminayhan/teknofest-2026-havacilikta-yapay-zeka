# -*- coding: utf-8 -*-
"""GOREV 2 — TAVAN OLCUMU: kayan pencere / yumusatma backend'i ne kazandirabilir?

NEDEN VAR
---------
Manus raporu (§4.2) backend icin "10-30 karelik kayan pencere, Huber/Cauchy kaybi, hiz
ve ivme yumusatmasi, ani sicrama reddi" oneriyor. Bu listenin ucu depoda ZATEN denendi:

  * Huber/Cauchy  -> IRLS/Cauchy: varyansi dusuruyor, YANLILIK ekliyor (reddedildi)
  * sicrama reddi -> Hampel: +4,8 ... +18,2 m (reddedildi)
  * hiz yumusatma -> `smooth_alpha=0,6` (uretimde, istatistiksel dogrulandi)

Denenmemis olan tek sey gercek bir kayan-pencere OPTIMIZASYONU. Ama onu yazmadan once
sorulacak temel soru su:

    **Hatanin enerjisi nerede — yuksek frekansta mi, dusuk frekansta mi?**

Bir yumusatici/kayan pencere ancak YUKSEK FREKANSLI bileseni alabilir. Dusuk frekansli
surukleme, gercek hareketten ayirt edilemez; hicbir nedensel yumusatici onu kaldiramaz.

BU BETIK bir KAHIN olcumu yapar: hata sinyalinin yuksek frekansli kismini TAMAMEN
kaldirir (hicbir gercek yumusatici bundan iyisini yapamaz) ve puani yeniden hesaplar.

  * Kazanc buyukse -> yumusatma ailesi ACIK
  * Kazanc kucukse -> hata surukleme; aile KAPANIR

EK KISIT (onemli): yarisma ROLLING akistir — gonderilen tahmin geri alinamaz. Yani
gercek bir cozum ancak NEDENSEL (yalniz gecmise bakan) olabilir. Kahin burada
merkezi (iki yonlu) pencere kullanir, yani gercekten ulasilamaz bir ust sinirdir.

CALISTIR
--------
    .venv\\Scripts\\python gorev2\\tavan_kayan_pencere.py
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

GOREV2 = Path(__file__).resolve().parent
sys.path.insert(0, str(GOREV2))

from kabul import PENCERELER, _yukle, _geo_birlestir, _kos, yarisma_takvimi  # noqa: E402
from nokta_onbellek import YATAKLAR                                          # noqa: E402
from gorev2_src.vo_pipeline import EstimatorConfig                           # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

TUM = ["yarisma", "video1", "video2", "v25_rgb_a", "v25_rgb_b",
       "v25_termal_a", "v25_termal_b", "v24"]


def _hareketli_ort(x: np.ndarray, w: int) -> np.ndarray:
    """Merkezi hareketli ortalama (kenarlar tekrarla doldurulur)."""
    if w < 2:
        return x.copy()
    pad = w // 2
    out = np.empty_like(x)
    for j in range(x.shape[1]):
        d = np.pad(x[:, j], (pad, pad), mode="edge")
        c = np.convolve(d, np.ones(w) / w, mode="valid")
        out[:, j] = c[:len(x)]
    return out


def blok_ayristir(preds, gt, healthy, w):
    """Her kesinti blogunda hatayi dusuk/yuksek frekansa ayir.

    Doner: (ham_hata, kahin_hata, yuksek_frekans_payi)
      kahin_hata = yuksek frekansli bilesen TAMAMEN kaldirilsaydi olusan hata
    """
    h = np.asarray(healthy, bool)
    ham, kahin, paylar = [], [], []
    bas = None
    for i in range(len(h) + 1):
        icinde = i < len(h) and not h[i]
        if icinde and bas is None:
            bas = i
        elif not icinde and bas is not None:
            e = preds[bas:i] - gt[bas:i]
            if len(e) >= max(w, 8):
                e_low = _hareketli_ort(e, w)
                e_high = e - e_low
                ham.append(np.linalg.norm(e, axis=1))
                kahin.append(np.linalg.norm(e_low, axis=1))   # yuksek frekans silindi
                paylar.append(float((e_high ** 2).sum() / max((e ** 2).sum(), 1e-12)))
            bas = None
    if not ham:
        return np.nan, np.nan, np.nan
    return (float(np.concatenate(ham).mean()),
            float(np.concatenate(kahin).mean()),
            float(np.mean(paylar)))


def yatak_olc(ad, cfg, spacing, karistir, w):
    gt, cons, lad, prof, n = _yukle(ad, spacing, karistir, _geo_birlestir(cfg, {}))
    if ad == "yarisma":
        gercek = np.load(YATAKLAR["yarisma"]["gt_npz"])["saglikli"]
        out = []
        for _, p in PENCERELER.items():
            son = p["son"]
            h = gercek[:son].copy()
            a, b = p["puan"]
            h[a:b + 1] = False
            pr = _kos(cons.iloc[:son - 1], lad.iloc[:son - 1], gt[:son], h,
                      prof, cfg, spacing)
            out.append(blok_ayristir(pr[:son], gt[:son], h, w))
        return tuple(float(np.mean([o[k] for o in out])) for k in range(3))
    h = yarisma_takvimi(n)
    pr = _kos(cons, lad, gt, h, prof, cfg, spacing)
    return blok_ayristir(pr, gt, h, w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yatak", nargs="*", default=TUM)
    ap.add_argument("--pencere", type=int, nargs="*", default=[10, 30],
                    help="raporun onerdigi 10-30 kare")
    ap.add_argument("--n-perm", type=int, default=2)
    args = ap.parse_args()

    cfg = EstimatorConfig(method="homografi")
    print("=" * 86)
    print("KAYAN PENCERE TAVANI — yuksek frekansli hata TAMAMEN silinseydi")
    print(f"  {args.n_perm} permutasyon · KAHIN: merkezi pencere kullanir, yani")
    print("  nedensel (rolling) bir cozumun ULASAMAYACAGI bir ust sinirdir")
    print("=" * 86)
    bas = "".join(f"{'w=' + str(w):>20}" for w in args.pencere)
    print(f"\n  {'yatak':<14}{'MEVCUT':>10}{bas}")
    print(f"  {'':<14}{'':>10}" + "".join(f"{'kahin   YF payi':>20}" for _ in args.pencere))
    print("  " + "-" * (24 + 20 * len(args.pencere)))

    ort_m, ort_k = [], {w: [] for w in args.pencere}
    for ad in args.yatak:
        try:
            satir = f"  {ad:<14}"
            ham0 = None
            for w in args.pencere:
                R = [yatak_olc(ad, cfg, 8, k, w) for k in range(args.n_perm)]
                ham = float(np.mean([r[0] for r in R]))
                kah = float(np.mean([r[1] for r in R]))
                pay = float(np.mean([r[2] for r in R]))
                if ham0 is None:
                    ham0 = ham
                    satir += f"{ham:>10.2f}"
                satir += f"{kah:>13.2f}{100*pay:>6.0f}%"
                ort_k[w].append(kah)
            ort_m.append(ham0)
            print(satir, flush=True)
        except Exception as e:
            print(f"  {ad:<14} ATLANDI ({type(e).__name__}: {e})")

    print("  " + "-" * (24 + 20 * len(args.pencere)))
    satir = f"  {'ORTALAMA':<14}{np.mean(ort_m):>10.2f}"
    for w in args.pencere:
        satir += f"{np.mean(ort_k[w]):>13.2f}{'':>7}"
    print(satir)
    for w in args.pencere:
        print(f"\n  w={w}: kahin kazanci {np.mean(ort_k[w]) - np.mean(ort_m):+.2f} m")

    print("\n  YORUM ANAHTARI")
    print("   'YF payi' = hata enerjisinin yuzde kaci yuksek frekansta.")
    print("   Kucukse (<%10) hata SURUKLEMEDIR: gercek hareketten ayirt edilemez,")
    print("     hicbir yumusatici/kayan pencere kaldiramaz -> aile KAPANIR.")


if __name__ == "__main__":
    main()
