# -*- coding: utf-8 -*-
"""GOREV 2 — TANI METRIKLERI: ortalama hatanin ARKASINDA ne var?

NEDEN VAR
---------
Yarisma metrigi tum kareler uzerindeki ortalama Oklid hatasi. Ortalama, tek bir buyuk
sicramadan guclu bicimde etkilenir; ama bugune kadar yalnizca ortalamaya bakiyorduk.
`Gorev2_5_Cozum_Raporu.pdf` §11 Asama 2 bu yuzden ayri kayitlar tutulmasini oneriyor:

  * en kotu %5 karedeki hata      -> tek sicrama mi, yaygin drift mi?
  * saglik gecisinden sonraki
    ilk 50/100/300 kare           -> gecis aninda hizalama sicramasi var mi?
  * eksen bazli MAE/RMSE          -> hangi eksen toplami buyutuyor?
  * inlier orani / n_kept         -> geometrik on ucun guvenilirligi

BU BETIK SEVKIYAT KARARI URETMEZ. Sonraki islerin okuma zeminidir: bir degisiklik
ortalamayi ayni biraksa bile kuyrugu duzeltiyorsa bunu gormek isteriz (yarisma TEK
CEKILIS oldugu icin kuyruk dogrudan risktir).

Uretim kosusuna DOKUNMAZ — `kabul.py`'nin kendisi degistirilmedi, yalnizca oradaki
yukleyiciler ve kosucu yeniden kullanildi. Boylece kapi betikleri (`yontem_sinavi.py`,
`genelleme_sinavi.py`) etkilenmez.

CALISTIR
--------
    .venv\\Scripts\\python gorev2\\tani_metrik.py
    .venv\\Scripts\\python gorev2\\tani_metrik.py --yatak yarisma video2 --n-perm 3
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

TUM_YATAKLAR = ["yarisma", "video1", "video2", "v25_rgb_a", "v25_rgb_b",
                "v25_termal_a", "v25_termal_b", "v24"]


# --------------------------------------------------------------------------- metrikler
def gecis_noktalari(healthy: np.ndarray) -> list[int]:
    """Saglik 1 -> 0 gecislerinin kare numaralari."""
    h = np.asarray(healthy, bool)
    return [i for i in range(1, len(h)) if h[i - 1] and not h[i]]


def tani_metrikleri(preds: np.ndarray, gt: np.ndarray, healthy: np.ndarray,
                    puan_dilimi: slice | None = None) -> dict:
    """Bir kosunun tani metrikleri.

    `puan_dilimi` verilirse metrikler yalniz o dilimde (sinav penceresi) hesaplanir;
    verilmezse butun KESINTI kareleri kullanilir — iki durumda da saglikli kareler
    disaridadir, cunku orada GT aynen geri gonderiliyor ve hata sifir.
    """
    h = np.asarray(healthy, bool)
    if puan_dilimi is not None:
        maske = np.zeros(len(h), bool)
        maske[puan_dilimi] = True
    else:
        maske = ~h
    if maske.sum() == 0:
        return {}

    fark = preds[maske] - gt[maske]
    e = np.linalg.norm(fark, axis=1)

    # en kotu %5: kuyrugun agirligi
    k = max(int(np.ceil(0.05 * len(e))), 1)
    en_kotu = np.sort(e)[-k:]
    kuyruk_payi = en_kotu.sum() / max(e.sum(), 1e-12)

    m = dict(
        ort=float(e.mean()), medyan=float(np.median(e)), maks=float(e.max()),
        p95=float(np.percentile(e, 95)),
        kotu5_ort=float(en_kotu.mean()),
        kuyruk_payi=float(kuyruk_payi),      # toplam hatanin ne kadari en kotu %5'ten
        n=int(maske.sum()),
    )
    for i, eks in enumerate("xyz"):
        m[f"mae_{eks}"] = float(np.abs(fark[:, i]).mean())
        m[f"rmse_{eks}"] = float(np.sqrt((fark[:, i] ** 2).mean()))

    # saglik gecisinden sonraki birikim
    e_tam = np.linalg.norm(preds - gt, axis=1)
    for w in (50, 100, 300):
        v = []
        for t in gecis_noktalari(h):
            son = min(t + w, len(e_tam))
            if son - t >= w // 2:            # yarim pencere bile yoksa sayma
                v.append(float(e_tam[t:son].mean()))
        m[f"gecis{w}"] = float(np.mean(v)) if v else float("nan")
    m["n_gecis"] = len(gecis_noktalari(h))
    return m


# --------------------------------------------------------------------------- kosu
def yatak_tani(ad: str, cfg, spacing: int, karistir: int) -> dict:
    gt, cons, lad, prof, n = _yukle(ad, spacing, karistir, _geo_birlestir(cfg, {}))

    if ad == "yarisma":
        gercek = np.load(YATAKLAR["yarisma"]["gt_npz"])["saglikli"]
        cikti = {}
        for w, p in PENCERELER.items():
            son = p["son"]
            h = gercek[:son].copy()
            a, b = p["puan"]
            h[a:b + 1] = False
            preds = _kos(cons.iloc[:son - 1], lad.iloc[:son - 1], gt[:son], h,
                         prof, cfg, spacing)
            cikti[f"pencere{w}"] = tani_metrikleri(preds, gt[:son], h, slice(a, b + 1))
        return cikti

    h = yarisma_takvimi(n)
    preds = _kos(cons, lad, gt, h, prof, cfg, spacing)
    return {ad: tani_metrikleri(preds, gt, h)}


def ortala(kayitlar: list[dict]) -> dict:
    """Permutasyonlar arasi ortalama (ve std)."""
    if not kayitlar:
        return {}
    anahtarlar = kayitlar[0].keys()
    out = {}
    for k in anahtarlar:
        v = np.array([r[k] for r in kayitlar], float)
        out[k] = float(np.nanmean(v))
        out[k + "_std"] = float(np.nanstd(v, ddof=1)) if len(v) > 1 else 0.0
    return out


# --------------------------------------------------------------------------- rapor
def rapor(tablo: dict[str, dict]):
    adlar = list(tablo)
    P = "=" * (26 + 13 * len(adlar))

    def satir(etiket, anahtar, fmt="{:.2f}"):
        print(f"  {etiket:<24}" + "".join(
            f"{fmt.format(tablo[a][anahtar]):>13}" if anahtar in tablo[a] else f"{'-':>13}"
            for a in adlar))

    print(f"\n{P}\nTANI METRIKLERI — yalniz KESINTI kareleri\n{P}")
    print(f"  {'':<24}" + "".join(f"{a:>13}" for a in adlar))
    print("  " + "-" * (24 + 13 * len(adlar)))
    satir("kare sayisi", "n", "{:.0f}")
    print()
    satir("ortalama hata [m]", "ort")
    satir("medyan [m]", "medyan")
    print()
    print("  --- KUYRUK (tek cekilis riski) ---")
    satir("p95 [m]", "p95")
    satir("en kotu %5 ort [m]", "kotu5_ort")
    satir("en kotu tek kare [m]", "maks")
    satir("kuyrugun hata payi", "kuyruk_payi", "{:.1%}")
    print()
    print("  --- EKSEN AYRISIMI (MAE) ---")
    for eks in "xyz":
        satir(f"  {eks}", f"mae_{eks}")
    print("  --- EKSEN AYRISIMI (RMSE) ---")
    for eks in "xyz":
        satir(f"  {eks}", f"rmse_{eks}")
    print()
    print("  --- SAGLIK GECISI SONRASI (ilk N kare ortalamasi) ---")
    satir("gecis sayisi", "n_gecis", "{:.0f}")
    for w in (50, 100, 300):
        satir(f"  ilk {w} kare", f"gecis{w}")
    print(f"\n{P}")
    print("  kuyruk payi: toplam hatanin yuzde kaci en kotu %5 kareden geliyor.")
    print("  %5'lik dilim hatanin cok uzerinde bir pay tasiyorsa sorun DRIFT degil SICRAMA.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yatak", nargs="*", default=TUM_YATAKLAR)
    ap.add_argument("--n-perm", type=int, default=3)
    ap.add_argument("--spacing", type=int, default=8)
    args = ap.parse_args()

    cfg = EstimatorConfig(method="homografi")
    ham: dict[str, list[dict]] = {}
    for i in range(args.n_perm):
        for ad in args.yatak:
            try:
                for etiket, m in yatak_tani(ad, cfg, args.spacing, i).items():
                    ham.setdefault(etiket, []).append(m)
            except Exception as e:
                print(f"  {ad}: ATLANDI ({type(e).__name__}: {e})")
        print(f"  permutasyon {i} bitti", flush=True)

    rapor({k: ortala(v) for k, v in ham.items()})


if __name__ == "__main__":
    main()
