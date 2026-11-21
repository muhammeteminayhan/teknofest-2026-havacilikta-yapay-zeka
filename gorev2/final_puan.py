# -*- coding: utf-8 -*-
"""GOREV 2 — FINAL PUAN / SIRA MODELI.

Sartname (Teknik Sartname s.31, Denklem 2): E = karelerin 3B Oklid hatasinin ortalamasi;
"her oturum sonunda TUM TAKIMLARIN hata miktarlari ile takimin puani hesaplanir".
Hatayi puana ceviren formul yayinlanmadi. Bu modul onu RESMI TABLODAN kalibre eder:

  * Her oturumda bizim gonderdigimiz yorungenin E'si (sozde-GT'ye karsi) ve bizim resmi
    Gorev 2 skorumuz bilinir -> olcek parametresi tau.
  * Iki bicim: ustel  s = exp(-E/tau)   ve   Gauss  s = exp(-(E/tau)^2).
    (Tablodaki 1.5e-129 gibi skorlar 1/(1+E/tau) turu hiperbolik bicimleri disliyor:
    o skor icin E/tau ~ 1e128 olmasi gerekirdi.)
  * Diger takimlarin E'si = tau * f^-1(skor). Yeni bir yontemin E'si -> skor -> oturum sirasi,
    Gorev 2 ortalama sirasi, toplam puan (G1/G3/Sunum/FTR sabit) ve genel sira.

Varsayim: bizim hatamiz degisince diger takimlarin skoru degismez (formul bizim E'mize de
bagliysa kucuk sapma olur). Sonuclar iki bicim icin ARALIK olarak verilir.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

GOREV2 = Path(__file__).resolve().parent
sys.path.insert(0, str(GOREV2))
sys.stdout.reconfigure(encoding="utf-8")

CIKTI = GOREV2 / "outputs" / "final"
RESMI = CIKTI / "resmi_sonuclar.json"
TABLO_OTURUMLARI = ["final_o1", "final_o2", "final_o3", "final_o4"]   # resmi tablodaki sira
BIZ = "bugbuster"
BICIMLER = ("ustel", "gauss")


def _f(x, bicim):            # E/tau -> skor
    return np.exp(-x) if bicim == "ustel" else np.exp(-x ** 2)


def _finv(s, bicim):         # skor -> E/tau
    s = np.clip(np.asarray(s, float), 1e-300, 1.0)
    return -np.log(s) if bicim == "ustel" else np.sqrt(-np.log(s))


def resmi():
    return json.loads(RESMI.read_text(encoding="utf-8"))["takimlar"]


def sozde_gt(ad: str) -> np.ndarray:
    return np.load(CIKTI / "colmap" / ad / "sozde_gt.npz")["konum"]


def oturum_E(pred: np.ndarray, ad: str, sozde: np.ndarray | None = None) -> tuple[float, float]:
    """Resmi metrik: saglikli karede referans = sunucunun GT'si, kesintide sozde-GT.

    Doner: (E [m], kesinti karelerinde sozde-GT kapsami)."""
    from final_yatak import veri
    d = veri(ad)
    ref = (sozde_gt(ad) if sozde is None else sozde).copy()
    ref[d["saglikli"]] = d["konum"][d["saglikli"]]
    ok = np.isfinite(ref[:, 0])
    e = np.linalg.norm(pred - ref, axis=1)
    kes = ~d["saglikli"]
    return float(e[ok].mean()), float(ok[kes].mean())


class PuanModeli:
    """Resmi tablo + bizim gonderdigimiz yorungenin E'leri ile kalibre edilmis model."""

    def __init__(self, E_gonderilen: dict[str, float]):
        self.takimlar = resmi()
        self.biz = next(t for t in self.takimlar if t["takim"].startswith(BIZ))
        self.digerleri = [t for t in self.takimlar if t is not self.biz]
        self.E_gonderilen = E_gonderilen
        self.tau = {}
        for j, ad in enumerate(TABLO_OTURUMLARI):
            s = self.biz["oturumlar"][j]["g2"]
            for b in BICIMLER:
                self.tau[(ad, b)] = E_gonderilen[ad] / float(_finv(s, b))

    def diger_E(self, ad: str, bicim: str) -> np.ndarray:
        j = TABLO_OTURUMLARI.index(ad)
        s = np.array([t["oturumlar"][j]["g2"] for t in self.digerleri])
        return self.tau[(ad, bicim)] * _finv(s, bicim)

    def skor(self, ad: str, E: float, bicim: str) -> float:
        return float(_f(E / self.tau[(ad, bicim)], bicim))

    def degerlendir(self, E_yeni: dict[str, float], bicim: str) -> dict:
        g2, oturum_sira = [], []
        for j, ad in enumerate(TABLO_OTURUMLARI):
            s = self.skor(ad, E_yeni[ad], bicim)
            diger = np.array([t["oturumlar"][j]["g2"] for t in self.digerleri])
            g2.append(s)
            oturum_sira.append(int((diger > s).sum()) + 1)
        diger_ort = np.array([np.mean([o["g2"] for o in t["oturumlar"]]) for t in self.digerleri])
        g2_ort = float(np.mean(g2))
        oturumlar = [0.25 * o["g1"] + 0.25 * o["g3"] + 0.40 * s
                     for o, s in zip(self.biz["oturumlar"], g2)]
        toplam = float(np.mean(oturumlar) + 0.05 * self.biz["sunum"] / 100
                       + 0.05 * self.biz["ftr"] / 100)
        diger_top = np.array([t["toplam"] for t in self.digerleri])
        return dict(g2=g2, oturum_sira=oturum_sira, g2_ort=g2_ort,
                    g2_sira=int((diger_ort > g2_ort).sum()) + 1,
                    toplam=toplam, genel_sira=int((diger_top > toplam).sum()) + 1)

    def tablo(self, yontemler: dict[str, dict[str, float]]) -> str:
        """yontemler: ad -> {oturum: E}. Her yontem icin iki bicimde sonuc satiri."""
        sat = []
        bas = (f"{'yontem':30s} {'bicim':6s} " + " ".join(f"{o[-2:].upper():>13s}" for o in TABLO_OTURUMLARI)
               + f" {'G2 ort':>7s} {'G2 sira':>8s} {'toplam':>7s} {'genel':>6s}")
        sat.append(bas)
        sat.append("-" * len(bas))
        for ad, E in yontemler.items():
            for b in BICIMLER:
                r = self.degerlendir(E, b)
                hucre = " ".join(f"{E[o]:6.1f}m {s:.3f}" for o, s in zip(TABLO_OTURUMLARI, r["g2"]))
                sat.append(f"{ad:30s} {b:6s} {hucre} {r['g2_ort']:7.3f} {r['g2_sira']:6d}. "
                           f"{r['toplam']:7.4f} {r['genel_sira']:4d}.")
        sat.append("  (oturum sutunu: E [m] ve tahmini Gorev 2 skoru; oturum sirasi icin "
                   "`degerlendir()` ciktisina bakin)")
        return "\n".join(sat)


def gonderilen_E() -> dict[str, tuple[float, float]]:
    from final_yatak import veri
    return {ad: oturum_E(veri(ad)["konum"], ad) for ad in TABLO_OTURUMLARI}


if __name__ == "__main__":
    Eg = gonderilen_E()
    for ad, (E, kap) in Eg.items():
        print(f"{ad}: gonderilen yorungenin E'si {E:8.2f} m (sozde-GT kapsami %{100 * kap:.0f})")
    m = PuanModeli({ad: e for ad, (e, _) in Eg.items()})
    for b in BICIMLER:
        print(f"tau [{b}]: " + ", ".join(f"{ad[-2:]} {m.tau[(ad, b)]:.1f}" for ad in TABLO_OTURUMLARI))
    print(m.tablo({"gonderilen (kalibrasyon)": {ad: e for ad, (e, _) in Eg.items()}}))
