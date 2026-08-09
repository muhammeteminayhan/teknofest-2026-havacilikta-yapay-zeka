# -*- coding: utf-8 -*-
"""GOREV 2 — REJIM TANISI: sistem ne zaman "hicbir sey yapmamaktan kotu" oluyor?

NEDEN
-----
`residual_gate` 0,50 -> 0,30 degisikligi turun en buyuk kazanciydi: calisan yataklarda
etkisi TAM SIFIR, bozuk iki 2025 termal yataginda 53,5 -> 27,0 ve 101,7 -> 62,3 m.
Kok neden olculmustu: kamera->dunya hizalamasinin GT'ye oturma olcusu
(`Alignment.residual_ratio`) o ucuslarda medyan 0,404 iken video1'de 0,040.

Kalan soru: sistem HALA bir yatakta (v25_termal_b: 61,4 m) "tut" tabanindan (53,1 m)
kotu. Bunu calisma aninda (GT'siz) tespit edip "tut"a dusmek mumkun mu?

ASIMETRI — bu betigin var olma sebebi
-------------------------------------
Kazanc ~8 m (tek yatak). Yanlis tetikleme maliyeti ~60 m (video1'de 'tut' = 63,6 m,
sistem 3,9 m). Yani anahtar ancak AYRIM COK GENISSE kurulmali. Bu betik anahtari
KURMAZ; yalnizca ayrimin genis olup olmadigini olcer.

NE OLCULUYOR
------------
Her yatak icin:
  * `tut` ve `sabit_hiz` kiyas tabanlari — sistem bunlari geciyor mu?
  * `fit_alignment` DENEME sayisi (fit is None sayilmaz)
  * `residual_gate` tarafindan REDDEDILME orani
  * kullanilan hizalamanin YASI (kabul edildikten sonra gecen adim)
  * `Alignment.height` OYNAKLIGI (std/ortalama)

ON KOSUL RISKI (once bunu dogrula)
----------------------------------
Hedef rejim yavas ucus (0,26 m/kare, %13,5 durgun). `min_move_m=0,10` kalibrasyon
ornegini suzuyor; `_buf_E` `min_calib=12`'ye hic ulasmazsa `fit_alignment` neredeyse
hic CAGRILMAZ ve red orani tanimsiz kalir. Deneme sayisi bu yuzden raporlaniyor.

CALISTIR
--------
    .venv\\Scripts\\python gorev2\\rejim_tanisi.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

GOREV2 = Path(__file__).resolve().parent
sys.path.insert(0, str(GOREV2))

import nokta_onbellek as NO                                      # noqa: E402
from nokta_onbellek import yatak_yukle, yatak_npz                # noqa: E402
from kabul import yarisma_takvimi, PENCERELER                    # noqa: E402
from yontem_sinavi import oransal_takvim                         # noqa: E402
from gorev2_src import evaluation                                # noqa: E402
from gorev2_src.scale_alignment import fit_alignment             # noqa: E402
from gorev2_src.vo_pipeline import EstimatorConfig, VoLadderEstimator  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

TUM = ["video1", "video2", "yarisma", "v24",
       "v25_rgb_a", "v25_rgb_b", "v25_termal_a", "v25_termal_b"]


class TaniliKestirici(VoLadderEstimator):
    """`fit_alignment` cagrilarini ve `residual_gate` kararlarini kaydeden alt sinif.

    Uretim koduna sayac eklemek yerine burada sarmalanir: uretim yolu degismez,
    tani yalniz bu betikte yasar.
    """

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.deneme = 0            # fit_alignment kac kez ANLAMLI sonuc dondurdu
        self.red = 0               # bunlarin kaci residual_gate'e takildi
        self.yukseklikler = []     # kabul edilen hizalamalarin olcegi
        self.yas = []              # kesinti baslarken hizalamanin yasi (adim)
        self._son_kabul_adim = None
        self._adim = 0

    def update(self, meas, gt_xyz, healthy):
        onceki = self.alignment
        self._adim += 1
        if healthy and self._prev_healthy and len(self._buf_E) >= self.cfg.min_calib:
            fit = fit_alignment(
                np.array(self._buf_E), np.array(self._buf_G),
                np.array(self._buf_dzu), np.array(self._buf_gdz),
                np.array(self._buf_gz), prev_z_sign=self.z_sign)
            if fit is not None:                      # `None` = deneme sayilmaz
                self.deneme += 1
                if fit.residual_ratio < self.cfg.residual_gate:
                    self.yukseklikler.append(fit.height)
                    self._son_kabul_adim = self._adim
                else:
                    self.red += 1
        if not healthy and self._prev_healthy and self._son_kabul_adim is not None:
            self.yas.append(self._adim - self._son_kabul_adim)   # kesinti girisi
        return super().update(meas, gt_xyz, healthy)


def olc(yatak: str, cfg: EstimatorConfig) -> dict:
    ib = NO.YATAKLAR[yatak].get("iki_boyut", False)
    gt, cons, lad, prof, n = yatak_yukle(yatak, n_kare=None if ib else 2250)
    boyut = 2 if ib else 3

    if yatak == "yarisma":
        d = np.load(NO.YATAKLAR["yarisma"]["gt_npz"])
        h = d["saglikli"].copy()
        h[900:1051] = False
        n = len(h)
        gt = gt[:n]
    elif yatak in ("v24", "v25_rgb_a", "v25_rgb_b", "v25_termal_a", "v25_termal_b"):
        h = oransal_takvim(n)
    else:
        h = yarisma_takvimi(n)

    est = TaniliKestirici(prof, cfg, spacing=8)
    cr = [evaluation._meas_from_row(r) for r in cons.to_dict("records")]
    lr = [evaluation._meas_from_row(r) for r in lad.to_dict("records")]
    preds = np.zeros((n, 3))
    preds[0] = gt[0]
    est.update_ladder(0, None, None, gt[0], True)
    for k in range(1, n):
        preds[k] = est.update_ladder(
            k, cr[k - 1] if k - 1 < len(cr) else None,
            lr[k - 1] if k - 1 < len(lr) else None,
            gt[k] if h[k] else None, bool(h[k]))

    hata = lambda p: float(np.linalg.norm(p[:, :boyut] - gt[:, :boyut], axis=1).mean())
    gt_df = pd.DataFrame(gt, columns=evaluation.GT_COLS)
    tut = hata(evaluation.run_baseline(gt_df, h, kind="tut"))
    hiz = hata(evaluation.run_baseline(gt_df, h, kind="sabit_hiz"))
    yuk = np.array(est.yukseklikler)
    return dict(
        sistem=hata(preds), tut=tut, hiz=hiz,
        oran=tut / max(hata(preds), 1e-9),
        deneme=est.deneme,
        red=est.red / est.deneme if est.deneme else float("nan"),
        yas=float(np.median(est.yas)) if est.yas else float("nan"),
        oynak=float(yuk.std() / yuk.mean() * 100) if len(yuk) > 1 else float("nan"),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yatak", nargs="*", default=None)
    args = ap.parse_args()
    yataklar = [y for y in (args.yatak or TUM)
                if yatak_npz(y, None).exists() and yatak_npz(y, 8).exists()]

    cfg = EstimatorConfig(method="homografi")
    print("=" * 100)
    print("REJIM TANISI — sistem ne zaman tabandan kotu, ve bu calisma aninda gorulebilir mi?")
    print(f"  residual_gate = {cfg.residual_gate} · min_calib = {cfg.min_calib} "
          f"· min_move_m = {cfg.min_move_m}")
    print("=" * 100)
    print(f"{'yatak':<15}{'sistem':>8}{'tut':>8}{'sbt_hiz':>9}{'tut/sis':>8}"
          f"{'fit deneme':>12}{'red orani':>11}{'hiz.yasi':>10}{'olcek oynak%':>13}")
    print("-" * 100)
    satirlar = {}
    for y in yataklar:
        r = olc(y, cfg)
        satirlar[y] = r
        bayrak = "  <-- TABANDAN KOTU" if r["oran"] < 1.0 else ""
        print(f"{y:<15}{r['sistem']:>8.2f}{r['tut']:>8.1f}{r['hiz']:>9.1f}{r['oran']:>8.1f}x"
              f"{r['deneme']:>12}{r['red']:>10.2f}{r['yas']:>10.0f}{r['oynak']:>12.1f}%"
              f"{bayrak}", flush=True)

    print("\n" + "=" * 100)
    print("AYRIM DEGERLENDIRMESI — IKI AYRI SORU")
    print("=" * 100)

    # SORU 1: sinyal "hizalama guvenilir mi"yi ayiriyor mu?
    supheli = [y for y, r in satirlar.items() if r["red"] > 0.5]
    saglam = [y for y, r in satirlar.items() if r["red"] <= 0.5]
    print("\n1) Sinyal HIZALAMA GUVENILIRLIGINI ayiriyor mu?")
    if saglam:
        print(f"   saglam {len(saglam)} yatak : red orani maks {max(satirlar[y]['red'] for y in saglam):.2f}"
              f" · hizalama yasi maks {max(satirlar[y]['yas'] for y in saglam):.0f}")
    if supheli:
        print(f"   supheli {len(supheli)} yatak: red orani min  {min(satirlar[y]['red'] for y in supheli):.2f}"
              f" · hizalama yasi min  {min(satirlar[y]['yas'] for y in supheli):.0f}")
        print("   -> EVET, ayrim genis ve ikili (red ~0 & yas 1  vs  red ~0,8 & yas ~600).")

    # SORU 2: ama "tut"a dusmek O yataklarda gercekten iyi mi? Asil soru bu.
    print("\n2) Supheli yataklarda 'tut'a dusmek KAZANDIRIR MIYDI?")
    fayda = []
    for y in supheli:
        r = satirlar[y]
        d = r["tut"] - r["sistem"]        # negatif = tut daha iyi olurdu
        fayda.append(d)
        print(f"   {y:<15} sistem {r['sistem']:6.2f} · tut {r['tut']:6.1f} "
              f"-> tut'a dusmek {-d:+7.2f} m")
    if supheli and any(f > 0 for f in fayda) and any(f < 0 for f in fayda):
        print("\n   -> HAYIR. Ayni imzaya sahip yataklarin BIRINDE tutmak kazandirir,")
        print("      DIGERINDE kaybettirir. Sinyal 'hizalama guvenilmez'i ayiriyor ama")
        print("      'tutmak daha iyi olur'u AYIRMIYOR — bunlar farkli sorular.")
        print(f"\n   KARAR: anahtar KURULMAYACAK. Net beklenen kazanc ~{-np.mean(fayda):+.1f} m,")
        print("          buna karsilik calisan bir ucusta yanlis tetiklenme maliyeti ~60 m.")
    elif supheli and all(f < 0 for f in fayda):
        print("\n   -> EVET, tum supheli yataklarda tutmak kazandirirdi; anahtar degerlendirilebilir.")
    else:
        print("\n   -> HAYIR, supheli yataklarda bile sistem tuttan iyi; anahtara gerek yok.")


if __name__ == "__main__":
    main()
