# -*- coding: utf-8 -*-
"""GOREV 2 — KAHIN (ORACLE) DENEYI: yukseklik bilinseydi ne kazanirdik?

NEDEN VAR
---------
Faz 0 tanisi (degerlendirme/Z_ISARETI.md ve tani_pencere1.py) su tabloyu verdi —
z isareti duzeltildikten SONRA, sinav penceresi 1:

    toplam hata 11,35 m  ·  yatay 11,26  ·  dusey 1,20
    olcek hatasi %5,2    ·  yukseklik durumu ort %6,5 geziniyor (en kotu %15,5)

Kesintide adim `dxy = M · e0 · height` oldugu icin yukseklik hatasi DOGRUDAN yatay
olcek hatasidir; olculen %5,2 ile %6,5 ortusuyor. Yani kalan hatanin baskin kaynagi
yuksekligin MUTLAK BIR GOZLEMI OLMAMASI.

Bir sonraki faz (nesne-olcegi cetveli: Gorev 1'in duran tasit kutulari) bu gozlemi
eklemeyi hedefliyor. Ama once ODULUN BUYUKLUGUNU bilmek gerekir: mukemmel bir
yukseklik gozlemi bile hatayi 11,35'ten yalniz 10'a indiriyorsa o is yapilmaz.

BU BETIK NE YAPAR
-----------------
Kesinti boyunca yukseklik durumunu GT'den beslenen GERCEK degere zorlar (kahin) ve
puani yeniden olcer. **Bu sevk edilebilir bir yol DEGILDIR** — GT kullanir. Yalnizca
ust siniri (ceiling) verir.

Ayrica gercekci bir gozlemin ne kadar iyi olmasi gerektigini de olcer:
  --gurultu    gozleme % cinsinden gurultu ekler   (dedektor kutu gurultusu)
  --aralik     gozlemi her N karede bir uygular    (arac her karede yok)
  --yanlilik   gozleme sabit % yanlilik ekler      (arac boyu populasyon kaymasi)

CALISTIR
--------
    .venv\\Scripts\\python gorev2\\kahin_yukseklik.py
    .venv\\Scripts\\python gorev2\\kahin_yukseklik.py --tara
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
from gorev2_src.evaluation import _meas_from_row, competition_error     # noqa: E402
from gorev2_src.vo_pipeline import EstimatorConfig, VoLadderEstimator   # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")


def kos_kahin(cons, lad, gt, healthy, prof, cfg, spacing,
              kahin: bool, gurultu=0.0, aralik=1, yanlilik=0.0, tohum=0):
    """Uretim kosusu; `kahin=True` ise kesintide yukseklik GERCEK degere zorlanir."""
    n = len(gt)
    est = VoLadderEstimator(prof, cfg, spacing=spacing)
    preds = np.zeros((n, 3))
    preds[0] = gt[0]
    est.update_ladder(0, None, None, gt[0], True)
    cr = [_meas_from_row(r) for r in cons.to_dict("records")]
    lr = [_meas_from_row(r) for r in lad.to_dict("records")]
    rng = np.random.default_rng(tohum)

    hiz = None
    for k in range(1, n):
        if kahin and not healthy[k] and hiz is not None:
            # gercek yukseklik = dondurulmus hizalamanin GT z ile tasinmis hali
            def _h(i):
                h0 = hiz.current_height(float(gt[i, 2]))
                if yanlilik:
                    h0 *= (1.0 + yanlilik)
                if gurultu:
                    h0 *= (1.0 + rng.normal(0.0, gurultu))
                return h0
            if (k - 1) % aralik == 0:          # gozlem yalniz her `aralik` karede
                if est.height is not None:
                    est.height = est._clamp_h(_h(k - 1))
                a = est._anchor
                if a is not None and a.get("h") is not None:
                    a["h"] = est._clamp_h(_h(int(a["ord"])))
        preds[k] = est.update_ladder(
            k, cr[k - 1] if k - 1 < len(cr) else None,
            lr[k - 1] if k - 1 < len(lr) else None,
            gt[k] if healthy[k] else None, bool(healthy[k]))
        if healthy[k] and est.alignment is not None:
            hiz = est.alignment
    return preds


def sinav(cfg, spacing, karistir, **kw):
    gt_t, cons, lad, prof, n = _yukle("yarisma", spacing, karistir, _geo_birlestir(cfg, {}))
    gercek = np.load(YATAKLAR["yarisma"]["gt_npz"])["saglikli"]
    out = {}
    for w, p in PENCERELER.items():
        son = p["son"]
        h = gercek[:son].copy()
        a, b = p["puan"]
        h[a:b + 1] = False
        preds = kos_kahin(cons.iloc[:son - 1], lad.iloc[:son - 1], gt_t[:son], h,
                          prof, cfg, spacing, **kw)
        out[w] = float(np.linalg.norm(preds[a:b + 1] - gt_t[a:b + 1], axis=1).mean())
    return out


def yatak(ad, cfg, spacing, karistir, **kw):
    gt, cons, lad, prof, n = _yukle(ad, spacing, karistir, _geo_birlestir(cfg, {}))
    h = yarisma_takvimi(n)
    return competition_error(kos_kahin(cons, lad, gt, h, prof, cfg, spacing, **kw), gt)


def olc(cfg, n_perm, yataklar, **kw):
    s = {w: [] for w in (1, 2)}
    y = {a: [] for a in yataklar}
    for i in range(n_perm):
        r = sinav(cfg, 8, i, **kw)
        for w in (1, 2):
            s[w].append(r[w])
        for a in yataklar:
            y[a].append(yatak(a, cfg, 8, i, **kw))
    return ({w: float(np.mean(s[w])) for w in (1, 2)},
            {a: float(np.mean(y[a])) for a in yataklar})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=3)
    ap.add_argument("--yatak", nargs="*", default=["video2", "v25_termal_b", "v25_rgb_b"])
    ap.add_argument("--tara", action="store_true")
    args = ap.parse_args()

    cfg = EstimatorConfig(method="homografi")
    Y = args.yatak
    bas = "".join(f"{a:>14}" for a in Y)
    print("=" * (44 + 14 * len(Y)))
    print("KAHIN DENEYI — kesintide yukseklik GERCEK degerinden besleniyor")
    print(f"  {args.n_perm} permutasyon · BU SEVK EDILEBILIR BIR YOL DEGIL, yalniz UST SINIR")
    print("=" * (44 + 14 * len(Y)))
    print(f"\n  {'kosu':<26}{'pencere1':>10}{'pencere2':>10}{bas}")
    print("  " + "-" * (46 + 14 * len(Y)))

    def _yaz(ad, s, y):
        print(f"  {ad:<26}{s[1]:>10.2f}{s[2]:>10.2f}"
              + "".join(f"{y[a]:>14.2f}" for a in Y), flush=True)

    s0, y0 = olc(cfg, args.n_perm, Y, kahin=False)
    _yaz("MEVCUT (kahin yok)", s0, y0)
    s1, y1 = olc(cfg, args.n_perm, Y, kahin=True)
    _yaz("KAHIN — kusursuz", s1, y1)

    if args.tara:
        print()
        for g in (0.02, 0.05, 0.10):
            s, y = olc(cfg, args.n_perm, Y, kahin=True, gurultu=g)
            _yaz(f"KAHIN + %{100*g:.0f} gurultu", s, y)
        print()
        for a in (8, 25, 50):
            s, y = olc(cfg, args.n_perm, Y, kahin=True, aralik=a)
            _yaz(f"KAHIN + her {a} karede", s, y)
        print()
        for b in (0.03, 0.05, 0.10):
            s, y = olc(cfg, args.n_perm, Y, kahin=True, yanlilik=b)
            _yaz(f"KAHIN + %{100*b:.0f} yanlilik", s, y)
        print()
        s, y = olc(cfg, args.n_perm, Y, kahin=True, gurultu=0.05, aralik=25, yanlilik=0.03)
        _yaz("GERCEKCI (%5 g, 25 kare, %3 y)", s, y)

    print("\n  'MEVCUT' ile 'KAHIN' arasindaki fark, nesne-olcegi kanalinin")
    print("  ULASABILECEGI EN IYI sonuctur. Gercekci satir beklenen sonuca daha yakindir.")


if __name__ == "__main__":
    main()
