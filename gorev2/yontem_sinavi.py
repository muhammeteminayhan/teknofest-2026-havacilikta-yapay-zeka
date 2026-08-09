# -*- coding: utf-8 -*-
"""GOREV 2 — YONTEM SINAVI: BASKA YILLARIN verisinde bagimsiz dogrulama.

NEDEN
-----
Olcum turu 2'nin iki kazanci — `min_inlier_ratio` 0,40 -> 0,05 (+ kameraya gore
normalize esik) ve merdiven homografisinde GNC — 2026 verisinin uc yataginda
dogrulandi. Ama o uc yatagin ikisi ayni yilin ayni arazi/kamera kosullarindan
geliyor. Gercek soru:

    **Yontem mi calisiyor, yoksa 2026 verisine mi uyduk?**

Bu betik yontemi, ayarin HIC yapilmadigi baska yillarin ucuslarinda kosar.

YATAKLAR
--------
2025 ornek veri seti (tam 3B GT, kalibrasyon 2026 ile ayni donanim):
    v25_rgb_a / v25_rgb_b        RGB **4K (3840x2160)** — 2026'da 1080p'ydi
    v25_termal_a / v25_termal_b  termal 640x512
  DIKKAT: "Veri-1" RGB ile "Veri-1" Termal ayni ucus DEGIL —
    A ucusu = Ornek-Veri-1-RGB + Ornek-Veri-2-Termal (yol 710 m,  net 136 m)
    B ucusu = Ornek-Veri-2-RGB + Ornek-Veri-1-Termal (yol 1238 m, net  20 m)

2024 ornek veri (ZAYIF yatak, yalniz yatay kanal):
    v24   GT'de z YOK -> hata **2B** olculur; kalibrasyon dosyasi yok ->
          hem 2026-1080p profiliyle hem DISTORSIYONSUZ profille kosulur ki
          sonucun kalibrasyon varsayimina duyarli olup olmadigi gorulsun.
          Yalnizca 830 ornek -> takvim oransal kucultulur.

CALISTIR
--------
    .venv\\Scripts\\python gorev2\\yontem_sinavi.py --n-perm 5
    .venv\\Scripts\\python gorev2\\yontem_sinavi.py --yatak v24
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

GOREV2 = Path(__file__).resolve().parent
sys.path.insert(0, str(GOREV2))

from kabul import yatak_olc, procrustes_tani, _kos             # noqa: E402
from nokta_onbellek import YATAKLAR, yatak_npz, yatak_yukle    # noqa: E402
from gorev2_src import evaluation                              # noqa: E402
from gorev2_src.vo_pipeline import EstimatorConfig             # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

# Turun degisiklikleri, geriye dogru soyulmus hali
YENI = EstimatorConfig(method="homografi")            # sevk edilen
ARA = replace(YENI, min_inlier_ratio=0.40)            # yalniz `residual_gate` degisikligi
ESKI = replace(ARA, residual_gate=0.50)               # tur oncesi recete

V25 = ["v25_rgb_a", "v25_rgb_b", "v25_termal_a", "v25_termal_b"]
V24 = ["v24"]


def oransal_takvim(n: int) -> np.ndarray:
    """Yarisma takviminin ORANLARINI koruyan, n kareye olceklenmis saglik cizelgesi.

    2026 yapisi (2250 kare): saglikli %0-20 · kesinti %20-40 · saglikli %40-47 ·
    kesinti %47-69 · saglikli %69-73 · kesinti %73-100.
    Kisa videolarda (2024: 830 ornek) sabit blok sinirlari kullanilamaz.
    """
    h = np.ones(n, bool)
    for a, b in ((0.200, 0.400), (0.467, 0.689), (0.734, 1.000)):
        h[int(a * n):int(b * n)] = False
    return h


def olc_2b(yatak: str, cfg: EstimatorConfig, karistir: int, geo: dict,
           motor: str = "raft") -> float:
    """GT'de z olmayan yataklar icin: 2B ortalama Oklid hatasi."""
    gt, cons, lad, prof, n = yatak_yukle(yatak, spacing=8, karistir=karistir,
                                         n_kare=None, tracker=motor, **geo)
    h = oransal_takvim(n)
    p = _kos(cons, lad, gt, h, prof, cfg, 8)
    return float(np.linalg.norm(p[:, :2] - gt[:, :2], axis=1).mean())


def _motor_kiyas(yataklar, motor_a: str, motor_b: str, n_perm: int):
    """AYNI (sevk edilen) recete ile iki eslestiriciyi karsilastir.

    Amac: "SIFT birincil motor olsun mu" sorusunu kapatmak. Uc yatakta SIFT
    ortalamasi RAFT'inkinden dusuktu ama fark tamamen TEK yataktan (termal)
    geliyordu; gecen tur tam boyle bir tek-yatak sinyaline kanilip geri alinmisti.
    """
    import kabul
    print("=" * 88)
    print(f"MOTOR KARSILASTIRMASI — {motor_a} vs {motor_b} (ayni recete, {n_perm} permutasyon)")
    print("=" * 88)
    print(f"\n{'yatak':<15}{motor_a:>12}{motor_b:>12}{'fark':>16}{'karar':>10}")
    print("-" * 88)
    ozet = {}
    for y in yataklar:
        iki_b = YATAKLAR[y].get("iki_boyut", False)
        R = {}
        for m in (motor_a, motor_b):
            if not (yatak_npz(y, None, m).exists() and yatak_npz(y, 8, m).exists()):
                R[m] = None
                continue
            kabul.MOTOR = m
            if iki_b:
                R[m] = np.array([olc_2b(y, YENI, k, {}, m) for k in range(n_perm)])
            else:
                R[m] = np.array([yatak_olc(y, YENI, 8, k, {})["ana"]
                                 for k in range(n_perm)])
        if R[motor_a] is None or R[motor_b] is None:
            print(f"{y:<15}{'onbellek yok — atlandi':>50}")
            continue
        ozet[y] = R
        d = R[motor_b] - R[motor_a]
        se = 2 * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
        karar = ("KAZANDI" if d.mean() + se < 0 else
                 "KAYBETTI" if d.mean() - se > 0 else "esit")
        print(f"{y:<15}{R[motor_a].mean():>12.2f}{R[motor_b].mean():>12.2f}"
              f"{d.mean():>+11.2f} ±{se:4.2f}{karar:>10}", flush=True)
    if len(ozet) > 1:
        print("-" * 88)
        a = np.mean([ozet[y][motor_a].mean() for y in ozet])
        b = np.mean([ozet[y][motor_b].mean() for y in ozet])
        kaz = sum(1 for y in ozet if ozet[y][motor_b].mean() < ozet[y][motor_a].mean())
        felaket = [y for y in ozet
                   if ozet[y][motor_b].mean() - ozet[y][motor_a].mean() > 3.0]
        print(f"{'ORTALAMA':<15}{a:>12.2f}{b:>12.2f}{b - a:>+16.2f}")
        print(f"\n  {motor_b} kac yatakta iyi : {kaz}/{len(ozet)}")
        print(f"  felaket (>+3 m) yatak : {felaket if felaket else 'yok'}")
        print(f"\n  KARAR KURALI: {motor_b} ancak COGUNLUKTA kazanir VE hicbir yatakta")
        print("                felaket uretmezse birincil motor olur.")
        uygun = kaz > len(ozet) / 2 and not felaket
        print(f"  -> {motor_b} birincil olabilir mi: {'EVET' if uygun else 'HAYIR'}")
    kabul.MOTOR = "raft"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=5)
    ap.add_argument("--yatak", nargs="*", default=None)
    ap.add_argument("--motor", default="raft",
                    help="eslestirici onbellegi: raft | sift | akaze ...")
    ap.add_argument("--motor-kiyas", default=None,
                    help="verilirse iki motoru AYNI recete ile karsilastirir")
    args = ap.parse_args()

    import kabul
    kabul.MOTOR = args.motor

    istenen = args.yatak or (V25 + V24)
    yataklar = [y for y in istenen
                if yatak_npz(y, None, args.motor).exists()
                and yatak_npz(y, 8, args.motor).exists()]
    eksik = [y for y in istenen if y not in yataklar]

    if args.motor_kiyas:
        _motor_kiyas(yataklar, args.motor, args.motor_kiyas, args.n_perm)
        return

    print("=" * 88)
    print("YONTEM SINAVI — baska yillarin verisi (ayar BU YATAKLARDA YAPILMADI)")
    print(f"  {args.n_perm} permutasyon · eslestirilmis")
    if eksik:
        print(f"  !!! onbellegi olmayan yatak (atlandi): {', '.join(eksik)}")
    print("=" * 88)
    print(f"\n{'yatak':<15}{'ESKI':>13}{'+residual':>13}{'+kapi (YENI)':>13}"
          f"{'YENI-ESKI':>18}")
    print("-" * 88)

    ozet = {}
    for y in yataklar:
        t0 = time.perf_counter()
        iki_b = YATAKLAR[y].get("iki_boyut", False)
        R = {}
        for ad, cfg in (("eski", ESKI), ("ara", ARA), ("yeni", YENI)):
            if iki_b:
                R[ad] = np.array([olc_2b(y, cfg, k, {}, args.motor)
                                  for k in range(args.n_perm)])
            else:
                R[ad] = np.array([yatak_olc(y, cfg, 8, k, {})["ana"]
                                  for k in range(args.n_perm)])
        ozet[y] = R
        d = R["yeni"] - R["eski"]
        se = 2 * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
        karar = ("KAZANDI" if d.mean() + se < 0 else
                 "KAYBETTI" if d.mean() - se > 0 else "esit")
        etiket = y + (" (2B)" if iki_b else "")
        print(f"{etiket:<15}{R['eski'].mean():>8.2f}±{R['eski'].std(ddof=1):<4.2f}"
              f"{R['ara'].mean():>8.2f}±{R['ara'].std(ddof=1):<4.2f}"
              f"{R['yeni'].mean():>8.2f}±{R['yeni'].std(ddof=1):<4.2f}"
              f"{d.mean():>+11.2f} ±{se:4.2f} {karar:>8}  ({time.perf_counter()-t0:.0f}s)",
              flush=True)

    if len(ozet) > 1:
        print("-" * 88)
        e = np.mean([ozet[y]["eski"].mean() for y in ozet])
        a = np.mean([ozet[y]["ara"].mean() for y in ozet])
        nn = np.mean([ozet[y]["yeni"].mean() for y in ozet])
        kazanan = sum(1 for y in ozet if ozet[y]["yeni"].mean() < ozet[y]["eski"].mean())
        print(f"{'ORTALAMA':<15}{e:>13.2f}{a:>13.2f}{nn:>13.2f}{nn - e:>+18.2f}")
        print(f"\n  residual_gate katkisi : {a - e:+.2f} m")
        print(f"  min_inlier katkisi    : {nn - a:+.2f} m")
        print(f"  yeni recete kac yatakta iyi : {kazanan}/{len(ozet)}")

    print("\nNOT: ayarlar 2026 verisinde secildi; buradaki sonuc yontemin")
    print("     genellenip genellenmedigini gosterir, yeni bir ayar turu degildir.")


if __name__ == "__main__":
    main()
