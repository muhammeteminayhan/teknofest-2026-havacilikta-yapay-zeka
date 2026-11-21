# -*- coding: utf-8 -*-
"""GOREV 2 — FINAL YENIDEN DEGERLENDIRME RAPORU (tablo + grafik).

Girdi: `final_degerlendir.py` ciktisi `outputs/final/final_degerlendirme.json` (5 oturum x
yontemler, sozde-GT E'si ve kontrol noktalari) + resmi tablo.

Ek yontem — MODALITE KURALI (onceden sabitlenmis, oturum basina sonradan secim DEGIL):
  termal (640x512)  -> temel (RAFT)          GT'li 3 termal ucusta ORB-SLAM3 ortalamada kotu
  RGB               -> ORB-SLAM3 + Umeyama   GT'li 3 RGB ucusta ortalama -%42

    .venv\\Scripts\\python gorev2\\final_rapor.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

GOREV2 = Path(__file__).resolve().parent
sys.path.insert(0, str(GOREV2))
import final_puan as fp                                            # noqa: E402
import final_yatak as fy                                           # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
CIKTI = fy.CIKTI
TERMAL = {"final_o3"}
KURALLAR = {"KURAL: RGB->ORB, termal->RAFT": ("orb", "temel"),
            "KURAL: RGB->ORB+z20, termal->RAFT": ("orb+z20", "temel")}


def yukle() -> dict[tuple[str, str], dict]:
    d = json.loads((CIKTI / "final_degerlendirme.json").read_text(encoding="utf-8"))
    return {tuple(k.split("|")): v for k, v in d.items()}


def main():
    S = yukle()
    oturumlar = sorted({o for o, _ in S})
    yontemler = list(dict.fromkeys(y for _, y in S))
    for ad, (rgb, termal) in KURALLAR.items():
        for o in oturumlar:
            kaynak = termal if o in TERMAL else rgb
            if (o, kaynak) in S:
                S[(o, ad)] = S[(o, kaynak)]
        yontemler.append(ad)

    Eg = {o: fp.oturum_E(fy.veri(o)["konum"], o)[0] for o in oturumlar}
    satir = ["## Oturum basina E (sozde-GT'ye gore, m) — kucuk iyi", ""]
    satir.append("| yontem | " + " | ".join(o.replace("final_", "").upper() for o in oturumlar) + " |")
    satir.append("|---|" + "---|" * len(oturumlar))
    satir.append("| gonderilen (yarismada) | " + " | ".join(f"{Eg[o]:.1f}" for o in oturumlar) + " |")
    for y in yontemler:
        satir.append(f"| {y} | " + " | ".join(
            f"{S[(o, y)]['E']:.1f}" if (o, y) in S and S[(o, y)]["E"] is not None else "—"
            for o in oturumlar) + " |")

    model = fp.PuanModeli({o: Eg[o] for o in fp.TABLO_OTURUMLARI})
    satir += ["", "## Tahmini Gorev 2 skoru ve siralar (resmi tablodan kalibre)", ""]
    satir.append("| yontem | bicim | O1 | O2 | O3 | O4 | G2 ort. | G2 sirasi | toplam | genel sira |")
    satir.append("|---|---|---|---|---|---|---|---|---|---|")
    ozet = {}
    adaylar = {"gonderilen (yarismada)": {o: Eg[o] for o in fp.TABLO_OTURUMLARI}}
    for y in yontemler:
        if all((o, y) in S and S[(o, y)]["E"] is not None for o in fp.TABLO_OTURUMLARI):
            adaylar[y] = {o: S[(o, y)]["E"] for o in fp.TABLO_OTURUMLARI}
    for y, E in adaylar.items():
        for b in fp.BICIMLER:
            r = model.degerlendir(E, b)
            ozet[(y, b)] = r
            hucre = " | ".join(f"{s:.3f} ({k}.)" for s, k in zip(r["g2"], r["oturum_sira"]))
            satir.append(f"| {y} | {b} | {hucre} | {r['g2_ort']:.3f} | **{r['g2_sira']}.** | "
                         f"{r['toplam']:.4f} | **{r['genel_sira']}.** |")
    satir += ["", "Resmi: G2 ortalamasi 0,396 (22.), toplam 0,4510 (16.). Kalibrasyon tanim geregi "
              "gonderilen satirini resmi degerlere esitler.",
              f"tau (ustel): " + ", ".join(f"{o[-2:]} {model.tau[(o, 'ustel')]:.1f} m" for o in fp.TABLO_OTURUMLARI),
              f"tau (gauss): " + ", ".join(f"{o[-2:]} {model.tau[(o, 'gauss')]:.1f} m" for o in fp.TABLO_OTURUMLARI)]
    metin = "\n".join(satir)
    print(metin)
    (CIKTI / "final_rapor_tablo.md").write_text(metin, encoding="utf-8")
    json.dump({f"{k[0]}|{k[1]}": v for k, v in ozet.items()},
              open(CIKTI / "final_rapor_ozet.json", "w", encoding="utf-8"), indent=1, default=float)

    # grafik
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    goster = ["gonderilen (yarismada)", "temel", "arac_capa", "hibrit", "orb",
              "KURAL: RGB->ORB, termal->RAFT"]
    goster = [g for g in goster if (g, "ustel") in ozet]
    f, ax = plt.subplots(1, 2, figsize=(15, 5.5))
    x = np.arange(len(fp.TABLO_OTURUMLARI))
    w = 0.8 / len(goster)
    for i, g in enumerate(goster):
        E = adaylar[g]
        ax[0].bar(x + i * w, [E[o] for o in fp.TABLO_OTURUMLARI], w, label=g)
    ax[0].set_xticks(x + 0.4 - w / 2)
    ax[0].set_xticklabels(["O1", "O2", "O3 (termal)", "O4"])
    ax[0].set_yscale("log")
    ax[0].set_ylabel("oturum E [m] (log)")
    ax[0].set_title("Sozde-GT'ye gore Gorev 2 hatasi")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=.3, axis="y")
    for j, b in enumerate(fp.BICIMLER):
        siralar = [ozet[(g, b)]["genel_sira"] for g in goster]
        ax[1].barh(np.arange(len(goster)) + j * 0.4, siralar, 0.4, label=f"{b} bicim")
    ax[1].set_yticks(np.arange(len(goster)) + 0.2)
    ax[1].set_yticklabels(goster, fontsize=9)
    ax[1].set_xlabel("genel sira (35 takim; kisa cubuk = daha iyi)")
    ax[1].set_title("Bu yontemle genel siralama (diger gorevler sabit)")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=.3, axis="x")
    for j, b in enumerate(fp.BICIMLER):
        for i, g in enumerate(goster):
            ax[1].text(ozet[(g, b)]["genel_sira"], i + j * 0.4, f" {ozet[(g, b)]['genel_sira']}.",
                       va="center", fontsize=8)
    f.tight_layout()
    f.savefig(CIKTI / "final_rapor.png", dpi=130)
    print(f"\ngrafik: {CIKTI / 'final_rapor.png'}")


if __name__ == "__main__":
    main()
