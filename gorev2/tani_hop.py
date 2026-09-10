# -*- coding: utf-8 -*-
"""D3 — KEYFRAME KULLANILABILIRLIK BOZUNMASI (salt tani; uretim kodu degismez).

SORU
----
Bir keyframe, uzerinden kac kare gectikten sonra kullanilamaz hale geliyor?

NEDEN ONEMLI
------------
Kesintide yukseklik HER keyframe atlamasinda carpimsal guncelleniyor
(`_h_guncelle`: h *= 1 - dz_unit). 600 karelik bir kesintide `spacing=8` ile
**75 carpim** yapiliyor; olculen yukseklik surukienmesi (video1 %7,6 · termal %25,8,
bir blokta oran 0,11'e cokuyor) bu birikimden besleniyor. Uzun omurlu bir capa
carpim sayisini 10-15'e indirebilir.

Bugune dek yalnizca IKI nokta olculmus: hop 1 -> inlier 0,86 · hop 8 -> 0,54 (RGB),
0,914 -> 0,728 (termal). Aradaki egri hic cizilmemis. Uzun capa fikri de yalnizca
bir satir olarak var (`VoCokluMerdiven`, -%1,5 / +0,30 s/kare, video1, eski
yapilandirma, hic commit edilmemis).

YONTEM (bedava)
---------------
`spacing=16` nokta onbelleklerinde keyframe sabit izgarada durur, dolayisiyla ayni
dosyada hop = 1..16 arasi butun mesafeler bulunur:
    hop = (idx1 - idx0) / FRAME_STEP
Her hop icin homografi inlier orani, n_kept ve **ima edilen yukseklik** olculur.
Ima edilen yukseklik = GT yer degistirme / birim-yukseklik olcum buyuklugu; hop
buyudukce bunun hop=1 degerinden sapmasi, olcumun sistematik yanliliginin
hop ile nasil degistigini gosterir (asil ilgilendigimiz sey budur).

DOGRULAMA
---------
Egri bilinen iki noktayi yeniden uretmeli (hop 1 ~0,86 · hop 8 ~0,54 RGB).
Uretmiyorsa betik hatalidir.

CALISTIR
--------
    .venv\\Scripts\\python gorev2\\tani_hop.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

GOREV2 = Path(__file__).resolve().parent
sys.path.insert(0, str(GOREV2))

import nokta_onbellek as NO                                   # noqa: E402
from nokta_onbellek import onbellek_yukle, olcum_uret, yatak_npz   # noqa: E402
from gorev2_src import data_io                                # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

# sp16 onbellegi olan yataklar (hop 1..16 hepsi bu dosyalarda)
YATAKLAR = ["video1", "video2", "yarisma"]


def _gt(yatak: str) -> np.ndarray:
    y = NO.YATAKLAR[yatak]
    if "gt_npz" in y:
        return np.load(y["gt_npz"])["konum"]
    g = data_io.subsample_competition(data_io.load_translations(y["gt_csv"]))
    return g[["translation_x", "translation_y", "translation_z"]].to_numpy()


def egri(yatak: str, spacing: int = 16) -> pd.DataFrame:
    p = yatak_npz(yatak, spacing)
    if not p.exists():
        return pd.DataFrame()
    y = NO.YATAKLAR[yatak]
    prof = y["profile"]
    # DIKKAT: kare DIZININDEN gelen yataklar (yarisma) zaten 7,5 FPS'tir, idx = ordinal.
    # Videodan gelenler 30 FPS'tir ve her 4. kare alinir, idx = dogal kare numarasi.
    adim = 1 if "kareler" in y else data_io.FRAME_STEP

    ob = onbellek_yukle(p)
    d = olcum_uret(ob, prof)
    d["idx0"], d["idx1"] = ob["idx0"], ob["idx1"]
    d["hop"] = ((d.idx1 - d.idx0) / adim).round().astype(int)

    gt = _gt(yatak)
    n = len(gt)
    ok = d[(d.hom_ok) & (~d.frozen) & (d.hop >= 1)].copy()

    # duzleme paralel bilesenin buyuklugu (h=1) ve ayni hop'un GT yer degistirmesi
    nn = ok[["hom_nx", "hom_ny", "hom_nz"]].to_numpy()
    dl = ok[["hom_ux", "hom_uy", "hom_uz"]].to_numpy()
    dzu = (nn * dl).sum(1)
    E = np.linalg.norm((dl - dzu[:, None] * nn)[:, :2], axis=1)
    i0 = np.clip((ok.idx0 // adim).to_numpy(), 0, n - 1)
    i1 = np.clip((ok.idx1 // adim).to_numpy(), 0, n - 1)
    G = np.linalg.norm(gt[i1, :2] - gt[i0, :2], axis=1)
    ok["E"], ok["G"] = E, G

    satir = []
    for hop, grup in ok.groupby("hop"):
        g = grup[grup.G > 0.05 * hop]          # anlamli hareket
        ima = g.G.sum() / max(g.E.sum(), 1e-9) if len(g) else np.nan
        satir.append(dict(
            hop=hop, n=len(grup),
            inlier=float(grup.hom_inlier_ratio.median()),
            n_kept=float(grup.n_kept.median()),
            kapi_alti=float((grup.hom_inlier_ratio < 0.30).mean()),
            ima_h=ima,
        ))
    return pd.DataFrame(satir).sort_values("hop")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yatak", nargs="*", default=YATAKLAR)
    ap.add_argument("--spacing", type=int, default=16)
    args = ap.parse_args()

    print("=" * 84)
    print("D3 — KEYFRAME KULLANILABILIRLIK BOZUNMASI (hop 1..%d)" % args.spacing)
    print("=" * 84)
    print("  ima_h = GT yer degistirme / birim-olcum buyuklugu — hop ile SAPMASI")
    print("          olcumun sistematik yanliliginin hop bagimliligini gosterir.\n")

    ozet = {}
    for y in args.yatak:
        df = egri(y, args.spacing)
        if df.empty:
            print(f"--- {y}: sp{args.spacing} onbellegi yok, atlandi ---\n")
            continue
        ozet[y] = df
        taban = df[df.hop == 1].ima_h.iloc[0] if (df.hop == 1).any() else np.nan
        print(f"--- {y} ({NO.YATAKLAR[y]['profile'].name}) ---")
        print(f"{'hop':>4}{'n':>7}{'inlier':>9}{'n_kept':>9}"
              f"{'kapi<0.30':>11}{'ima_h [m]':>11}{'hop1 sapma':>12}")
        for _, r in df.iterrows():
            sap = (r.ima_h / taban - 1) * 100 if np.isfinite(taban) else np.nan
            print(f"{int(r.hop):>4}{int(r.n):>7}{r.inlier:>9.3f}{r.n_kept:>9.0f}"
                  f"{100*r.kapi_alti:>10.1f}%{r.ima_h:>11.2f}{sap:>+11.2f}%")
        print()

    # --- dogrulama
    # DIKKAT: kayitlardaki "merdiven inlier medyani 0,54", sp8 onbelleginin TUM
    # hop'lari (1..8) uzerindeki medyandir — hop=8'in kendisi degil. Dogru kontrol,
    # hop 1..8 medyanlarinin ortalamasinin o degere yakin olmasidir.
    print("=" * 84)
    print("DOGRULAMA — kayitli degerlerle tutarlilik")
    for y, bekle_h1, bekle_karisik in (("video1", 0.86, 0.54), ("video2", 0.914, 0.728)):
        if y not in ozet:
            continue
        d = ozet[y]
        h1 = d[d.hop == 1].inlier.iloc[0] if (d.hop == 1).any() else np.nan
        karisik = d[(d.hop >= 1) & (d.hop <= 8)].inlier.mean()
        ok = abs(h1 - bekle_h1) < 0.06 and abs(karisik - bekle_karisik) < 0.09
        print(f"  {y:<8} hop1={h1:.3f} (kayit ~{bekle_h1:.2f}) · "
              f"hop1-8 ort={karisik:.3f} (kayit ~{bekle_karisik:.2f})"
              f"  -> {'TUTARLI' if ok else '!!! TUTARSIZ'}")

    # --- karar
    print("\n" + "=" * 84)
    print("KARAR (onceden yazili kriter: hop 16-24'e kadar makul kalirsa uzun capa acilir)")
    for y, d in ozet.items():
        d16 = d[d.hop == args.spacing]
        d8 = d[d.hop == 8]
        if d16.empty or d8.empty:
            continue
        i8, i16 = d8.inlier.iloc[0], d16.inlier.iloc[0]
        k16 = d16.kapi_alti.iloc[0]
        taban = d[d.hop == 1].ima_h.iloc[0]
        s8 = (d8.ima_h.iloc[0] / taban - 1) * 100
        s16 = (d16.ima_h.iloc[0] / taban - 1) * 100
        print(f"  {y:<9} inlier {i8:.3f}(hop8) -> {i16:.3f}(hop{args.spacing}) · "
              f"kapi alti %{100*k16:.1f} · ima_h sapmasi {s8:+.2f}% -> {s16:+.2f}%")


if __name__ == "__main__":
    main()
