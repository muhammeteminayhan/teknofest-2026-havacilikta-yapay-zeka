# -*- coding: utf-8 -*-
"""GOREV 2 — KABUL KAPISI: bir degisikligi UC BAGIMSIZ UCUS uzerinde, gurultu
tabanini hesaba katarak olcer.

NEDEN VAR
---------
1) **Genelleme.** Butun eski ayar calismasi `THYZ_2026_Ornek_Veri_1.MP4` uzerinde
   yapildi ve o video cevrim ici TEST OTURUMUNUN ta kendisi. Tek ucusta olculen
   kazanc puanli oturumda iki kez yeniden uretilemedi (bkz. degerlendirme/DUSEY_KANAL.md).

2) **Gurultu.** OpenCV RANSAC girdi verildiginde deterministiktir ama nokta
   SIRASINA duyarlidir: ayni noktalar farkli sirada verilince esit derecede iyi
   ama FARKLI bir duzlem secilir. Olculdu (video1, sevk edilen recete, ayni nokta
   onbellegi, 10 permutasyon):

       ort 3,964 m · std 0,226 · yayilim 3,672–4,411 m  (= ortalamanin %18,7'si)

   Yani TEK kosuyla olculen ~0,4 m'lik bir fark tamamen gurultu olabilir. Gecmis
   turlarin bircok karari (orn. ds0,75'in +0,034 m ile reddedilmesi) bu tabanin
   altindadir ve kanit degeri tasimaz.

COZUM
-----
Her aday, temel hatla **AYNI permutasyon tohumlarinda** kosulur ve karsilastirma
**eslestirilmis fark** uzerinden yapilir (paired): ayni gurultu gerceklemesinde
iki yapilandirma karsilastirildigi icin ortak gurultu duser, kucuk ama gercek
farklar gorunur hale gelir.

UC KAPI
-------
  yarisma  : puanli oturumun KENDI kareleri, iki sinav penceresi   -> VETO
  video1   : ornek veri 1 (RGB, test oturumu ucusu)                -> kaybetmemeli
  video2   : ornek veri 2 (TERMAL, bagimsiz ucus)                  -> kaybetmemeli

KULLANIM
--------
    .venv\\Scripts\\python gorev2\\kabul.py --geo irls_c=2.0
    .venv\\Scripts\\python gorev2\\kabul.py --cfg smooth_alpha=1.0 --n-perm 8
    .venv\\Scripts\\python gorev2\\kabul.py --geo irls_c=2.0 --yatak video1 video2
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

GOREV2 = Path(__file__).resolve().parent
ROOT = GOREV2.parent
sys.path.insert(0, str(GOREV2))

_SRC = GOREV2 / "src"
if "gorev2_src" not in sys.modules:
    _spec = importlib.util.spec_from_file_location(
        "gorev2_src", _SRC / "__init__.py", submodule_search_locations=[str(_SRC)])
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules["gorev2_src"] = _mod
    _spec.loader.exec_module(_mod)

from nokta_onbellek import YATAKLAR, yatak_yukle, yatak_npz     # noqa: E402
from gorev2_src import evaluation                               # noqa: E402
from gorev2_src.vo_pipeline import EstimatorConfig              # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

# Puanli oturum logundan cikarilan GERCEK kesinti takvimi
KESINTI = [(451, 900), (1051, 1550), (1651, 2250)]
N_UZUN = 15                       # uzun-kesinti senaryo sayisi (OZET.md ile ayni)

# genelleme_sinavi.py ile ayni pencereler: saglikli bir blok yapay kesintiye cevrilir
PENCERELER = {1: dict(son=1051, puan=(900, 1050)),
              2: dict(son=1651, puan=(1550, 1650))}


# --------------------------------------------------------------------------- takvimler
def yarisma_takvimi(n: int) -> np.ndarray:
    h = np.ones(n, bool)
    for a, b in KESINTI:
        h[a:b] = False
    return h


def uzun_takvim(n: int, rng) -> np.ndarray:
    """Yarismadaki gibi UZUN kesintiler (450-600 kare) — slam_degerlendirme ile ayni."""
    r = np.random.default_rng(rng)
    h = np.ones(n, dtype=bool)
    for _ in range(3):
        d = int(r.uniform(450, 600))
        b = int(r.integers(300, max(301, n - d - 10)))
        h[b:b + d] = False
    return h


def procrustes_tani(preds, gt, healthy):
    """Kesinti bloklarinda ortalama yon (derece) ve olcek hatasi."""
    aci, olcek, bas = [], [], None
    for i in range(len(healthy) + 1):
        icinde = i < len(healthy) and not healthy[i]
        if icinde and bas is None:
            bas = i
        elif not icinde and bas is not None:
            p = preds[bas:i, :2] - preds[bas, :2]
            g = gt[bas:i, :2] - gt[bas, :2]
            if len(p) > 10 and (p ** 2).sum() > 1e-9:
                U, S, Vt = np.linalg.svd(g.T @ p)
                R = U @ Vt
                if np.linalg.det(R) < 0:
                    U[:, -1] *= -1
                    R = U @ Vt
                aci.append(abs(np.degrees(np.arctan2(R[1, 0], R[0, 0]))))
                olcek.append(abs(float(S.sum() / max((p ** 2).sum(), 1e-9)) - 1.0))
            bas = None
    return (float(np.mean(aci)) if aci else 0.0), (float(np.mean(olcek)) if olcek else 0.0)


# --------------------------------------------------------------------------- kosucular
def _kos(cons, lad, gt, healthy, prof, cfg, spacing):
    gt_sub = pd.DataFrame(gt, columns=evaluation.GT_COLS)
    return evaluation.run_estimator_ladder(cons, lad, gt_sub, healthy, prof, cfg,
                                           spacing=spacing)


_OLCUM_ONBELLEK: dict = {}


MOTOR = "raft"          # A/B icin degistirilebilir: "raft" | "sift" | "akaze" | ...


def _yukle(yatak: str, spacing: int, karistir: int, geo: dict):
    """Olcum uretimi pahali (~30 s); ayni (yatak, perm, geometri) icin bir kez yap.

    Temel hat ile aday cogu zaman AYNI geometriyi kullanir (yalniz EstimatorConfig
    degisir); bu durumda olcumler paylasilir ve kosu suresi yariya iner.
    """
    anahtar = (yatak, spacing, karistir, MOTOR, repr(sorted(geo.items())))
    if anahtar not in _OLCUM_ONBELLEK:
        _OLCUM_ONBELLEK[anahtar] = yatak_yukle(
            yatak, spacing=spacing, karistir=karistir, tracker=MOTOR, **geo)
    return _OLCUM_ONBELLEK[anahtar]


def _geo_birlestir(cfg: EstimatorConfig, geo: dict) -> dict:
    """EstimatorConfig icindeki OLCUM katmani ayarlarini geo sozluguene tasi.

    `merdiven_gnc_c` canli yolda `LivePositionEstimator` tarafindan merdiven
    olcumune uygulaniyor; onbellekli yolda ayni etkiyi almak icin `olcum_uret`e
    `geo_merdiven` olarak gecmeli — yoksa sinav/A-B bu ayari HIC gormez.
    """
    c = getattr(cfg, "merdiven_gnc_c", None)
    if not c:
        return geo
    gm = dict(geo.get("geo_merdiven") or {})
    gm["gnc_c"] = float(c)
    return {**geo, "geo_merdiven": gm}


def yatak_olc(yatak: str, cfg: EstimatorConfig, spacing: int, karistir: int,
              geo: dict) -> dict:
    """Bir yatakta bir permutasyon icin butun metrikleri uret."""
    gt, cons, lad, prof, n = _yukle(yatak, spacing, karistir, _geo_birlestir(cfg, geo))

    if yatak == "yarisma":
        # Sinav: puanli oturumun kendi kareleri. Kesinti karelerinin GT'si YOK
        # (log'daki degerler bizim tahminimiz), o yuzden yalniz SAGLIKLI bir blogu
        # yapay kesintiye cevirip orada puanliyoruz — genelleme_sinavi.py ile ayni.
        d = np.load(YATAKLAR["yarisma"]["gt_npz"])
        gercek = d["saglikli"]
        sonuc = {}
        for w, p in PENCERELER.items():
            son = p["son"]
            h = gercek[:son].copy()
            a, b = p["puan"]
            h[a:b + 1] = False
            preds = _kos(cons.iloc[:son - 1], lad.iloc[:son - 1], gt[:son], h,
                         prof, cfg, spacing)
            dil = slice(a, b + 1)
            sonuc[f"pencere{w}"] = float(
                np.linalg.norm(preds[dil] - gt[dil], axis=1).mean())
        sonuc["ana"] = float(np.mean(list(sonuc.values())))
        return sonuc

    yar = yarisma_takvimi(n)
    p_yar = _kos(cons, lad, gt, yar, prof, cfg, spacing)
    uz = [evaluation.competition_error(
        _kos(cons, lad, gt, uzun_takvim(n, s), prof, cfg, spacing), gt)
        for s in range(N_UZUN)]
    yon, olcek = procrustes_tani(p_yar, gt, yar)
    return dict(ana=evaluation.competition_error(p_yar, gt),
                uzun=float(np.mean(uz)), yon=yon, olcek=100 * olcek)


# --------------------------------------------------------------------------- rapor
def _ozet(vals: np.ndarray) -> str:
    return f"{vals.mean():6.3f} ± {vals.std(ddof=1) if len(vals) > 1 else 0:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geo", nargs="*", default=[],
                    help="geometri parametreleri, orn. irls_c=2.0 ransac_thresh=1.5")
    ap.add_argument("--cfg", nargs="*", default=[],
                    help="EstimatorConfig alanlari, orn. smooth_alpha=1.0")
    ap.add_argument("--spacing", type=int, default=8)
    ap.add_argument("--n-perm", type=int, default=6,
                    help="permutasyon sayisi (gurultu tabani icin >=6 onerilir)")
    ap.add_argument("--yatak", nargs="*", default=None, choices=sorted(YATAKLAR))
    ap.add_argument("--etiket", default=None)
    args = ap.parse_args()

    def _ayrist(cift):
        k, v = cift.split("=", 1)
        try:
            v = int(v) if v.isdigit() or (v[0] == "-" and v[1:].isdigit()) else float(v)
        except ValueError:
            pass
        return k, v

    geo = dict(_ayrist(s) for s in args.geo)
    cfg_over = dict(_ayrist(s) for s in args.cfg)
    etiket = args.etiket or (" ".join(args.geo + args.cfg) or "TEMEL")

    temel_cfg = EstimatorConfig(method="homografi")
    aday_cfg = replace(temel_cfg, **cfg_over) if cfg_over else temel_cfg

    yataklar = args.yatak or [y for y in ("yarisma", "video1", "video2")
                              if yatak_npz(y, None).exists() and yatak_npz(y, args.spacing).exists()]
    eksik = [y for y in ("yarisma", "video1", "video2") if y not in yataklar]

    print(f"{'=' * 78}\nKABUL KAPISI — aday: {etiket}")
    print(f"  {args.n_perm} permutasyon · eslestirilmis karsilastirma · spacing={args.spacing}")
    if eksik:
        print(f"  !!! ONBELLEGI OLMAYAN YATAK (atlandi): {', '.join(eksik)}")
    print("=" * 78)

    tum = {}
    for yatak in yataklar:
        t0 = time.perf_counter()
        T, A = [], []
        for k in range(args.n_perm):
            T.append(yatak_olc(yatak, temel_cfg, args.spacing, k, {}))
            A.append(yatak_olc(yatak, aday_cfg, args.spacing, k, geo))
            print(f"    {yatak} perm {k}: temel {T[-1]['ana']:.3f} -> aday {A[-1]['ana']:.3f}",
                  flush=True)
        anahtarlar = [a for a in T[0] if a != "ana"] or []
        t_ana = np.array([r["ana"] for r in T])
        a_ana = np.array([r["ana"] for r in A])
        fark = a_ana - t_ana
        se = fark.std(ddof=1) / np.sqrt(len(fark)) if len(fark) > 1 else 0.0
        tum[yatak] = dict(t=t_ana, a=a_ana, fark=fark, se=se,
                          detay=(T, A, anahtarlar), sure=time.perf_counter() - t0)
        print(f"    {yatak} bitti ({tum[yatak]['sure']:.0f}s)\n", flush=True)

    print(f"\n{'=' * 78}\n{'yatak':<10}{'TEMEL':>16}{'ADAY':>16}{'fark (eslestirilmis)':>26}{'karar':>10}")
    print("=" * 78)
    kararlar = {}
    for yatak, r in tum.items():
        f, se = r["fark"].mean(), r["se"]
        # 2 standart hata = kabaca %95 guven araligi
        if f + 2 * se < 0:
            karar = "KAZANDI"
        elif f - 2 * se > 0:
            karar = "KAYBETTI"
        else:
            karar = "esit"
        kararlar[yatak] = karar
        print(f"{yatak:<10}{_ozet(r['t']):>16}{_ozet(r['a']):>16}"
              f"{f:>+16.3f} ±{2*se:5.3f}{karar:>10}")

    print("-" * 78)
    print("ikincil olcutler (hepsi eslestirilmis fark ± 2 standart hata)")
    for yatak, r in tum.items():
        T, A, anah = r["detay"]
        for k in anah:
            t = np.array([x[k] for x in T])
            a = np.array([x[k] for x in A])
            d = a - t
            s = 2 * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
            im = "KAZANC" if d.mean() + s < 0 else ("KAYIP" if d.mean() - s > 0 else "esit")
            print(f"  {yatak:<9}{k:<9}{t.mean():8.3f} -> {a.mean():7.3f}"
                  f"   {d.mean():+7.3f} ±{s:5.3f}  {im}")

    print("=" * 78)
    if "yarisma" in kararlar and kararlar["yarisma"] == "KAYBETTI":
        sonuc = "REDDEDILDI (sinav vetosu)"
    elif any(v == "KAYBETTI" for v in kararlar.values()):
        sonuc = "REDDEDILDI (bir yatakta kayip)"
    elif any(v == "KAZANDI" for v in kararlar.values()):
        sonuc = "UYGULANABILIR"
    else:
        sonuc = "ETKISIZ (gurultu icinde)"
    print(f"SONUC: {sonuc}")
    print("KURAL: sinav VETO; hicbir yatakta kaybetmeyecek; en az birinde kazanacak.")


if __name__ == "__main__":
    main()
