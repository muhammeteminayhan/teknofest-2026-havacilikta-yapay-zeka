# -*- coding: utf-8 -*-
"""GOREV 2 — OLCEK CAPASI DENEYI (gercek Gorev 1 kutulariyla).

Kahin (kahin_olcek.py) tavani gosterdi: metre/piksel dogru bilinseydi 12 yatak×takvimin
11'inde hata duser (ort. 16,70 -> 12,01 m). Bu betik ayni duzeltmeyi GERCEK gozlemle
(tasit boyu + 4,5 m ped, src/olcek_capasi.py) yapar ve olcer:

  * GT'li yataklar (video1, video2, 2025 ucuslari) x iki takvim (cevrimici, final):
    tespitler `tespit_onbellek.py` ile uretilmis olmali.
  * Final oturumlari: gercek-GT kontrol noktalari (O1, O3); sozde-GT hazirsa tum oturum E'si.

Kabul olcutu planla ayni: GT'li yataklarda ortalamada kazanmali ve anlamli kotulesme
olmamali; final oturumlari dokunulmamis sinav olarak raporlanir.

    .venv\\Scripts\\python gorev2\\capa_deney.py
    .venv\\Scripts\\python gorev2\\capa_deney.py --final
"""
from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np

GOREV2 = Path(__file__).resolve().parent
sys.path.insert(0, str(GOREV2))

from kabul import _yukle, _geo_birlestir                                 # noqa: E402
from kahin_olcek import TAKVIMLER                                        # noqa: E402
import final_yatak as fy                                                 # noqa: E402
from gorev2_src.evaluation import _meas_from_row, competition_error      # noqa: E402
from gorev2_src.olcek_capasi import CapaAyar, OlcekCapasi, kanca_kur     # noqa: E402
from gorev2_src.vo_pipeline import EstimatorConfig                       # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

YATAKLAR = ["video1", "video2", "v25_rgb_a", "v25_rgb_b", "v25_termal_a", "v25_termal_b"]
VARYANTLAR = {
    "bant0.2": CapaAyar(kazanc=1.0, olu_bant=0.20, ped=False),
    "bant0.3": CapaAyar(kazanc=1.0, olu_bant=0.30, ped=False),
    "bant0.4": CapaAyar(kazanc=1.0, olu_bant=0.40, ped=False),
    "bant0.3 g0.5": CapaAyar(kazanc=0.5, olu_bant=0.30, ped=False),
    "bant0.3 pedli": CapaAyar(kazanc=1.0, olu_bant=0.30, ped=True),
}


def kutular(npz) -> dict[int, list]:
    d = np.load(npz) if not isinstance(npz, dict) else npz
    out: dict[int, list] = {}
    for k, s, x in zip(d["kutu_kare"], d["kutu_sinif"], d["kutu_xyxy"]):
        out.setdefault(int(k), []).append((int(s), *map(float, x)))
    return out


def kos_capali(cr, lr, gt, s, prof, cfg, kut, ayar):
    if ayar is None:
        return fy.kos(cr, lr, gt, s, prof, cfg)
    capa = OlcekCapasi(prof.fx, (prof.width, prof.height), ayar)
    return fy.kos(cr, lr, gt, s, prof, cfg, kanca=kanca_kur(capa, kut, s))


def yataklar(n_perm: int, adlar, cfg):
    sutun = ["temel"] + list(VARYANTLAR)
    print(f"{'yatak':14s} {'takvim':10s} " + " ".join(f"{c:>12s}" for c in sutun))
    toplam = {c: [] for c in sutun}
    for y in adlar:
        tp = fy.CIKTI / f"tespit_{y}.npz"
        if not tp.exists():
            print(f"{y}: tespit onbellegi yok, atlandi")
            continue
        kut = kutular(tp)
        for t in TAKVIMLER:
            r = {c: [] for c in sutun}
            for i in range(n_perm):
                gt, cons, lad, prof, n = _yukle(y, 8, i, _geo_birlestir(cfg, {}))
                cr = [_meas_from_row(x) for x in cons.to_dict("records")]
                lr = [_meas_from_row(x) for x in lad.to_dict("records")]
                s = TAKVIMLER[t](n)
                r["temel"].append(competition_error(kos_capali(cr, lr, gt, s, prof, cfg, kut, None), gt))
                for c, ay in VARYANTLAR.items():
                    r[c].append(competition_error(kos_capali(cr, lr, gt, s, prof, cfg, kut, ay), gt))
            r = {c: float(np.mean(v)) for c, v in r.items()}
            for c in sutun:
                toplam[c].append(r[c])
            print(f"{y:14s} {t:10s} " + " ".join(f"{r[c]:12.2f}" for c in sutun), flush=True)
    print("-" * (26 + 13 * len(sutun)))
    print(f"{'ORTALAMA':25s} " + " ".join(f"{np.mean(toplam[c]):12.2f}" for c in sutun))
    for c in VARYANTLAR:
        kaz = sum(a < b - 1e-9 for a, b in zip(toplam[c], toplam["temel"]))
        kot = max((a / b - 1 for a, b in zip(toplam[c], toplam["temel"])), default=0)
        print(f"  {c:14s}: {kaz}/{len(toplam[c])} iyi | en kotu goreli degisim {100 * kot:+.0f}%")


def finaller(cfg):
    for ad in fy.KONTROLLER:
        d = fy.veri(ad)
        kut = kutular(d)
        gt, cr, lr, prof, n = fy.onbellek_olcumleri(ad)
        print(f"\n{ad} — gercek-GT kontrol noktalari (3B hata, m)")
        for c, ay in [("temel", None)] + list(VARYANTLAR.items()):
            h = fy.kontrol_hatalari(ad, lambda s: kos_capali(cr, lr, gt, s, prof, cfg, kut, ay))
            print(f"  {c:14s} " + " | ".join(f"{e:s}: {e3:7.2f}" for e, e3, _ in h), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=2)
    ap.add_argument("--yatak", nargs="*", default=YATAKLAR)
    ap.add_argument("--final", action="store_true", help="yalniz final kontrol noktalari")
    a = ap.parse_args()
    cfg = EstimatorConfig(method="homografi")
    if a.final:
        finaller(cfg)
    else:
        yataklar(a.n_perm, a.yatak, cfg)


if __name__ == "__main__":
    main()
