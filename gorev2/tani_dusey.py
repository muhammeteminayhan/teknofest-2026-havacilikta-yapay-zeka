# -*- coding: utf-8 -*-
"""D4 — DUSEY KANAL YENIDEN DEGERLENDIRMESI (salt tani; uretim kodu degismez).

SORU
----
Kesintide dikey hareket homografi ayristirmasinin `dz_unit = n·delta` teriminden
geliyor. Alternatif — benzerlik olceginden gelen `sim_dz_unit` — 29 Tem'de
`z_sim_agirlik` adiyla denenip REDDEDILMISTI. Ama o karar:
  * ESKI kapiyla alindi (`min_inlier_ratio=0,40` · `residual_gate=0,50`),
  * yalniz TEK ucus ailesinde olculdu (video1 + puanli oturum),
  * termalde ve 2024/2025 yataklarinda HIC olculmedi.
O gunden beri sinav tabani 16,71 -> 11,08 m indi ve `residual_gate` tam onun
basarisizlik kipine benzeyen "bozuk hizalama" rejimini duzeltti.

Bu betik kanali YENIDEN UYGULAMADAN sorar: iki aday sinyalden hangisi GT'nin
dikey hareketini daha iyi aciklar?

YONTEM
------
Saglikli ardisik kare ciftlerinde (GT gercek bir adim) topla:
    hom kanali : dz_unit = n·delta
    sim kanali : sim_dz_unit = 1 - 1/olcek
    hedef      : GT dz [m]
Ikisi de "dikey yerdegistirme / yukseklik" birimindedir; GT dz'ye en kucuk
karelerle olceklenir (egim ~ z_sign·yukseklik). Karsilastirma olcutleri:
    * korelasyon (isaret ve tutarlilik)
    * aciklanan varyans R² (asil olcut)
    * artik saclim [m]
Ayrica MERDIVEN ciftlerinde de ayni olcum yapilir — kesintide konumu ureten
katman merdivendir, karar oraya bakmali.

CIKTI KARARI (onceden yazili)
-----------------------------
Benzerlik kanali 8 yatagin COGUNDA (>=5) hem ardisik hem merdiven katmaninda
daha yuksek R² verirse, dusey kanal bir sonraki turda tam kabul kapisiyla
yeniden acilir. Aksi halde kapi KALICI kapanir.

CALISTIR
--------
    .venv\\Scripts\\python gorev2\\tani_dusey.py
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
from nokta_onbellek import yatak_yukle, yatak_npz             # noqa: E402
from gorev2_src import data_io                                # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

TUM = ["video1", "video2", "yarisma", "v24",
       "v25_rgb_a", "v25_rgb_b", "v25_termal_a", "v25_termal_b"]


def _puanla(x: np.ndarray, y: np.ndarray) -> dict:
    """y ~ a·x + b en kucuk kareler; korelasyon, R², artik saclim."""
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 30 or x.std() < 1e-12:
        return dict(n=len(x), kor=np.nan, r2=np.nan, artik=np.nan, egim=np.nan)
    a, b = np.polyfit(x, y, 1)
    tahmin = a * x + b
    ss_res = float(((y - tahmin) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return dict(n=len(x), kor=float(np.corrcoef(x, y)[0, 1]),
                r2=float(1 - ss_res / max(ss_tot, 1e-12)),
                artik=float(np.sqrt(ss_res / len(x))), egim=float(a))


def olc(yatak: str) -> dict:
    y = NO.YATAKLAR[yatak]
    ib = y.get("iki_boyut", False)
    if ib:
        return {}                       # 2024'te GT'de z yok
    adim = 1 if "kareler" in y else data_io.FRAME_STEP
    gt, cons, lad, prof, n = yatak_yukle(yatak, n_kare=None)

    # sadece SAGLIKLI donemde GT gercek; yarisma yataginda saglik maskesi var
    if "gt_npz" in y:
        saglik = np.load(y["gt_npz"])["saglikli"]
        n = min(n, len(saglik))
    else:
        saglik = np.ones(n, bool)
    gt = gt[:n]

    sonuc = {}
    for ad, d in (("ardisik", cons), ("merdiven", lad)):
        d = d.iloc[:n - 1]
        ok = (d.hom_ok.to_numpy()) & (d.sim_ok.to_numpy()) & (~d.frozen.to_numpy())
        nn = d[["hom_nx", "hom_ny", "hom_nz"]].to_numpy()
        dl = d[["hom_ux", "hom_uy", "hom_uz"]].to_numpy()
        hom_dz = (nn * dl).sum(1)
        sim_dz = d.sim_dz_unit.to_numpy()

        i0 = np.clip(d.idx0.to_numpy() // adim, 0, n - 1)
        i1 = np.clip(d.idx1.to_numpy() // adim, 0, n - 1)
        gt_dz = gt[i1, 2] - gt[i0, 2]
        # iki ucu da saglikli olan ciftler (GT adimi gercek)
        gecerli = ok & saglik[i0] & saglik[i1] & (np.abs(gt_dz) > 1e-6)

        sonuc[ad] = dict(hom=_puanla(hom_dz[gecerli], gt_dz[gecerli]),
                         sim=_puanla(sim_dz[gecerli], gt_dz[gecerli]))
    return sonuc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yatak", nargs="*", default=TUM)
    args = ap.parse_args()
    yataklar = [y for y in args.yatak
                if yatak_npz(y, None).exists() and yatak_npz(y, 8).exists()]

    print("=" * 96)
    print("D4 — DUSEY KANAL: hangi sinyal GT'nin dikey hareketini daha iyi acikliyor?")
    print("=" * 96)
    print("  R² = aciklanan varyans (asil olcut) · artik = olceklendikten sonraki saclim [m]\n")
    print(f"{'yatak':<15}{'katman':<10}{'kanal':<6}{'n':>7}{'kor':>8}{'R²':>8}"
          f"{'artik[m]':>10}{'kazanan':>9}")
    print("-" * 96)

    skor = {"hom": 0, "sim": 0}
    detay = {}
    for y in yataklar:
        r = olc(y)
        if not r:
            print(f"{y:<15}{'— GT z yok, atlandi':<40}")
            continue
        detay[y] = r
        for kat in ("ardisik", "merdiven"):
            h, s = r[kat]["hom"], r[kat]["sim"]
            if not (np.isfinite(h["r2"]) and np.isfinite(s["r2"])):
                continue
            kazanan = "sim" if s["r2"] > h["r2"] else "hom"
            skor[kazanan] += 1
            for ad, v in (("hom", h), ("sim", s)):
                yildiz = " <<<" if ad == kazanan else ""
                print(f"{y if (kat=='ardisik' and ad=='hom') else '':<15}"
                      f"{kat if ad=='hom' else '':<10}{ad:<6}{v['n']:>7}"
                      f"{v['kor']:>8.3f}{v['r2']:>8.3f}{v['artik']:>10.3f}"
                      f"{yildiz:>9}")
        print()

    print("=" * 96)
    print(f"TOPLAM: hom {skor['hom']} · sim {skor['sim']}  "
          f"(yatak x katman karsilastirmasi)")
    print("\nKARAR (onceden yazili: benzerlik >=5 yatakta HER IKI katmanda kazanmali)")
    tam_kazanan = [y for y, r in detay.items()
                   if all(np.isfinite(r[k]['sim']['r2']) and np.isfinite(r[k]['hom']['r2'])
                          and r[k]['sim']['r2'] > r[k]['hom']['r2']
                          for k in ("ardisik", "merdiven"))]
    print(f"  benzerlik her iki katmanda kazanan yatak: {len(tam_kazanan)}/{len(detay)} "
          f"{tam_kazanan if tam_kazanan else ''}")
    if len(tam_kazanan) >= 5:
        print("  -> KAPI ACILIR: dusey kanal sonraki turda tam kabul kapisiyla denenir.")
    else:
        print("  -> KAPI KALICI KAPANIR: benzerlik kanali genellemiyor.")


if __name__ == "__main__":
    main()
