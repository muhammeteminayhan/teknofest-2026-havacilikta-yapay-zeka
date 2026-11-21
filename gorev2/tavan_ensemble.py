# -*- coding: utf-8 -*-
"""GOREV 2 — TAVAN OLCUMU: coklu on-uc ensemble ne kazandirabilir? (UST SINIR)

NEDEN VAR
---------
Manus raporunun asil mimari onerisi (§10) su: tek bir on uce guvenmek yerine iki on uc
(ogrenilmis homografi + optik akis) esZAMANLI kossun, her karede belirsizlik agirlikli
birlestirilsin, uyusmazlikta klasik geometriye dusulsun.

Bizde su an bir yedek ZINCIRI var (RAFT -> SIFT -> LK): biri kurulamazsa digerine
gecilir. Raporun onerdigi ise es zamanli SECIM/HARMANLAMA.

Boyle bir sistemi yazmadan once ust sinirini olcuyoruz: **kahin secici** — her kesinti
karesinde iki motorun olcumunden GT'ye daha yakin sonucu vereni sec. Hicbir gercek
belirsizlik olcutu bundan iyi secemez.

  * Kazanc buyukse -> ensemble ailesi ACIK, gercek bir belirsizlik olcutu aranir
  * Kazanc kucukse -> aile KAPANIR (secilecek bir sey yok; motorlar ayni hatayi
                      yapiyor ya da biri her zaman daha iyi)

Kalibrasyon (saglikli kareler) HER ZAMAN uretim motoruyla (RAFT) yapilir; kahin yalniz
kesintide hangi olcumun kullanilacagini secer. Boylece olculen sey tam olarak
"on uc secimi" olur, kalibrasyon farki degil.

CALISTIR
--------
    .venv\\Scripts\\python gorev2\\tavan_ensemble.py
    .venv\\Scripts\\python gorev2\\tavan_ensemble.py --yatak yarisma video2
"""
from __future__ import annotations

import argparse
import copy
import sys
import warnings
from pathlib import Path

import numpy as np

GOREV2 = Path(__file__).resolve().parent
sys.path.insert(0, str(GOREV2))

from kabul import PENCERELER, yarisma_takvimi                       # noqa: E402
from nokta_onbellek import YATAKLAR, yatak_yukle                    # noqa: E402
from gorev2_src.evaluation import _meas_from_row, competition_error  # noqa: E402
from gorev2_src.vo_pipeline import EstimatorConfig, VoLadderEstimator  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

# yatak adi -> onbellek adi (raft ve sift farkli eklerle uretilmis)
TUM = ["yarisma", "video1", "video2", "v24",
       "v25_rgb_a", "v25_rgb_b", "v25_termal_a", "v25_termal_b"]


def _satirlar(ad, motor, spacing, karistir, n_kare=2250):
    gt, cons, lad, prof, n = yatak_yukle(ad, spacing=spacing, karistir=karistir,
                                         tracker=motor, n_kare=n_kare)
    return (gt, [_meas_from_row(r) for r in cons.to_dict("records")],
            [_meas_from_row(r) for r in lad.to_dict("records")], prof, n)


SABIT = ("raft", "sift")
# GERCEKLESTIRILEBILIR seciciler: yalniz olcumun kendi istatistiklerine bakar, GT'ye DEGIL.
SECICI = ("inlier", "nkept", "harman", "kurtarma")

# `kurtarma` esikleri: RAFT'i YALNIZ acikca coktugunde birak.
# Gerekce: 'inlier' karsilastiricisi her karede secim yapiyor ve sinav penceresi 1'i
# +3,88 m bozuyor. Ama tani metrikleri termalde gercek bir KUYRUK oldugunu soyluyor
# (video2: hatanin %25'i en kotu %5 kareden). Dogru hedef her kare degil, yalniz
# RAFT'in coktugu kareler.
KURTARMA_RAFT_TABAN = 0.20     # RAFT merdiven inlier'i bunun altindaysa supheli
KURTARMA_SIFT_USTUNLUK = 1.50  # ve SIFT bu kat kadar iyiyse SIFT'e gec


def _skor(ml, mc, olcut):
    """Bir motorun o karedeki guven skoru (buyuk = iyi)."""
    if olcut == "inlier":
        return (ml.hom_inlier_ratio if ml is not None and ml.hom_ok else -1.0)
    if olcut == "nkept":
        return float(ml.n_kept) if ml is not None else -1.0
    return -1.0


def kos(gt, motorlar, healthy, prof, cfg, spacing, mod="raft",
        kurtarma_esik=(KURTARMA_RAFT_TABAN, KURTARMA_SIFT_USTUNLUK)):
    """`mod`: 'raft'|'sift' (sabit) · 'kahin' (GT gorur) · 'inlier'/'nkept'/'harman'."""
    n = len(gt)
    est = VoLadderEstimator(prof, cfg, spacing=spacing)
    preds = np.zeros((n, 3))
    preds[0] = gt[0]
    est.update_ladder(0, None, None, gt[0], True)
    secim = {k: 0 for k in motorlar}

    def _al(m, k):
        cr, lr = motorlar[m]
        return (cr[k - 1] if k - 1 < len(cr) else None,
                lr[k - 1] if k - 1 < len(lr) else None)

    for k in range(1, n):
        s = bool(healthy[k])
        if s or mod in SABIT:
            m = "raft" if s else mod          # kalibrasyon her zaman uretim motoruyla
            mc, ml = _al(m, k)
            preds[k] = est.update_ladder(k, mc, ml, gt[k] if s else None, s)
            continue

        # her motoru ayri bir kopyada ilerlet, sonra sec
        adaylar = {}
        for ad in motorlar:
            e2 = copy.deepcopy(est)
            mc, ml = _al(ad, k)
            adaylar[ad] = (e2, e2.update_ladder(k, mc, ml, None, False), mc, ml)

        if mod == "kahin":
            ad = min(adaylar, key=lambda a: np.linalg.norm(adaylar[a][1] - gt[k]))
        elif mod == "kurtarma":
            r = _skor(adaylar["raft"][3], adaylar["raft"][2], "inlier")
            s2 = _skor(adaylar["sift"][3], adaylar["sift"][2], "inlier")
            ad = ("sift" if (r < kurtarma_esik[0]
                             and s2 > kurtarma_esik[1] * max(r, 1e-6))
                  else "raft")
        elif mod == "harman":
            # inlier-agirlikli konum harmani; durum icin daha guclu olani tut
            w = {a: max(_skor(adaylar[a][3], adaylar[a][2], "inlier"), 0.0)
                 for a in adaylar}
            top = sum(w.values())
            ad = max(w, key=w.get)
            if top > 0:
                preds[k] = sum(w[a] * adaylar[a][1] for a in adaylar) / top
                est = adaylar[ad][0]
                est.pos = preds[k].copy()      # durumu harmanla senkron tut
                secim[ad] += 1
                continue
        else:
            ad = max(adaylar, key=lambda a: _skor(adaylar[a][3], adaylar[a][2], mod))

        est, preds[k] = adaylar[ad][0], adaylar[ad][1]
        secim[ad] += 1
    return preds, secim


def yatak_olc(ad, cfg, spacing, karistir):
    motorlar, gt, prof, n = {}, None, None, None
    for m in ("raft", "sift"):
        g, cr, lr, p, nn = _satirlar(ad, m, spacing, karistir)
        motorlar[m] = (cr, lr)
        if gt is None or nn < n:
            gt, prof, n = g, p, nn
    gt = gt[:n]

    out = {}
    if ad == "yarisma":
        gercek = np.load(YATAKLAR["yarisma"]["gt_npz"])["saglikli"]
        for w, pw in PENCERELER.items():
            son = pw["son"]
            h = gercek[:son].copy()
            a, b = pw["puan"]
            h[a:b + 1] = False
            mot = {k: (c[:son - 1], l[:son - 1]) for k, (c, l) in motorlar.items()}
            for mod in SABIT + ("kahin",) + SECICI:
                pr, sec = kos(gt[:son], mot, h, prof, cfg, spacing, mod)
                out[(f"pencere{w}", mod)] = float(
                    np.linalg.norm(pr[a:b + 1] - gt[a:b + 1], axis=1).mean())
                if mod == "kahin":
                    out[(f"pencere{w}", "sec_raft")] = sec["raft"] / max(sum(sec.values()), 1)
        return out

    h = yarisma_takvimi(n)
    for mod in SABIT + ("kahin",) + SECICI:
        pr, sec = kos(gt, motorlar, h, prof, cfg, spacing, mod)
        out[(ad, mod)] = competition_error(pr, gt)
        if mod == "kahin":
            out[(ad, "sec_raft")] = sec["raft"] / max(sum(sec.values()), 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yatak", nargs="*", default=TUM)
    ap.add_argument("--n-perm", type=int, default=2)
    ap.add_argument("--spacing", type=int, default=8)
    args = ap.parse_args()

    cfg = EstimatorConfig(method="homografi")
    ham: dict = {}
    for i in range(args.n_perm):
        for ad in args.yatak:
            try:
                for k, v in yatak_olc(ad, cfg, args.spacing, i).items():
                    ham.setdefault(k, []).append(v)
            except Exception as e:
                print(f"  {ad}: ATLANDI ({type(e).__name__}: {e})")
        print(f"  permutasyon {i} bitti", flush=True)

    etiketler = []
    for k in ham:
        if k[0] not in etiketler:
            etiketler.append(k[0])

    print("\n" + "=" * 78)
    print(f"ENSEMBLE TAVANI — kahin secici (RAFT vs SIFT), {args.n_perm} permutasyon")
    print("BU SEVK EDILEBILIR BIR YOL DEGIL; yalniz UST SINIR")
    print("=" * 78)
    basliklar = ("RAFT", "SIFT", "KAHIN") + tuple(s.upper() for s in SECICI)
    print(f"\n  {'yatak':<15}" + "".join(f"{b:>10}" for b in basliklar))
    print("  " + "-" * (15 + 10 * len(basliklar)))
    ort = {m: [] for m in SABIT + ("kahin",) + SECICI}
    for e in etiketler:
        g = lambda m: float(np.mean(ham[(e, m)])) if (e, m) in ham else float("nan")
        print(f"  {e:<15}" + "".join(
            f"{g(m):>10.2f}" for m in SABIT + ("kahin",) + SECICI))
        if not e.startswith("pencere"):
            for m in ort:
                ort[m].append(g(m))
    print("  " + "-" * (15 + 10 * len(basliklar)))
    print(f"  {'7 yatak ort.':<15}" + "".join(
        f"{np.mean(ort[m]):>10.2f}" for m in SABIT + ("kahin",) + SECICI))

    tavan = np.mean(ort["kahin"]) - np.mean(ort["raft"])
    print(f"\n  KAHIN tavani: {tavan:+.2f} m")
    for m in SECICI:
        k = np.mean(ort[m]) - np.mean(ort["raft"])
        pay = 100 * k / tavan if abs(tavan) > 1e-9 else 0.0
        print(f"  {m:<8} gercek kazanc: {k:+.2f} m  -> tavanin %{pay:.0f}'i")

    print("\n  YORUM ANAHTARI")
    print("   KAHIN GT gorur, sevk edilemez. Belirleyici olan GERCEKLESTIRILEBILIR")
    print("   secicilerin tavanin ne kadarini yakalayabildigidir. Hicbiri anlamli bir")
    print("   pay yakalayamiyorsa aile KAPANIR: tavan var ama ulasilamiyor.")


if __name__ == "__main__":
    main()
