# -*- coding: utf-8 -*-
"""GOREV 2 — FINAL OTURUMLARINDA YONTEM KIYASI ve tahmini puan/sira.

Yontemler (hepsi nedensel, yarismada calistirilabilir bicimde):
  temel        : uretim sistemi (RAFT + homografi + merdiven), onbellek yolunda yeniden kosum
  arac_capa    : temel + tasit boyundan olcek capasi (src/olcek_capasi.py, olu bant 0,2)
  orb          : ORB-SLAM3 + Umeyama (yansima serbest), iz kaybinda RAFT'a duser
  hibrit       : RAFT yonu + ORB-SLAM3 olcegi (orbslam3_poz.olcek_kancasi)
  hibrit+capa  : hibrit; ORB izlemiyorken arac capasi

Olcumler:
  * gercek-GT kontrol noktalari (O1, O3) — sunucunun GT'si, kesin;
  * sozde-GT (COLMAP) ile tum oturum E'si -> final_puan.PuanModeli ile tahmini skor ve sira.
RANSAC sira gurultusu yuzunden (O4'te yeniden kosum gonderilenden ort. 24 m sapiyor)
her yontem `--n-perm` permutasyonun ortalamasiyla olculur.

    .venv\\Scripts\\python gorev2\\final_degerlendir.py --n-perm 3
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

GOREV2 = Path(__file__).resolve().parent
sys.path.insert(0, str(GOREV2))

import final_yatak as fy                                             # noqa: E402
import final_puan as fp                                              # noqa: E402
import orbslam3_poz as op                                            # noqa: E402
from capa_deney import kutular                                       # noqa: E402
from gorev2_src.olcek_capasi import CapaAyar, OlcekCapasi, kanca_kur  # noqa: E402
from gorev2_src.vo_pipeline import EstimatorConfig                   # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
OTURUMLAR = ["final_o1", "final_o2", "final_o3", "final_o4", "final_test"]
YONTEMLER = ["temel", "arac_capa", "ped_capa", "orb", "hibrit", "hibrit+capa",
             "temel+z20", "arac_capa+z20", "orb+z20", "hibrit+z20"]
CAPA = CapaAyar(kazanc=0.5, olu_bant=0.30, ped=False)     # capa_deney: 4 yatakta zararsiz, en iyi ort.


def z_limiti(p: np.ndarray, s: np.ndarray, m: float) -> np.ndarray:
    """Birinci takimin yontemi (nedensel): kesintide z, o ana kadar gorulen saglikli z
    araliginin +-m disina cikamaz."""
    q = p.copy()
    lo = hi = None
    for k in range(len(q)):
        if s[k]:
            lo = q[k, 2] if lo is None else min(lo, q[k, 2])
            hi = q[k, 2] if hi is None else max(hi, q[k, 2])
        elif lo is not None:
            q[k, 2] = np.clip(q[k, 2], lo - m, hi + m)
    return q


def iki_kanca(*kancalar):
    def k(kk, est, preds):
        for f in kancalar:
            if f is not None:
                f(kk, est, preds)
    return k


def kosucu(ad, yontem, cr, lr, gt, prof, cfg, kut, orb):
    """saglik maskesi -> tahmin dizisi."""
    if "+z" in yontem:
        taban, m = yontem.rsplit("+z", 1)
        ic = kosucu(ad, taban, cr, lr, gt, prof, cfg, kut, orb)
        return lambda s: (None if (p := ic(s)) is None else z_limiti(p, s, float(m)))

    def kos(s):
        capa = None
        if yontem in ("arac_capa", "hibrit+capa", "ped_capa"):
            ay = CAPA if yontem != "ped_capa" else CapaAyar(kazanc=0.5, olu_bant=0.30, ped=True, arac=False)
            o = OlcekCapasi(prof.fx, (prof.width, prof.height), ay)
            capa = kanca_kur(o, kut, s)
        if yontem in ("temel", "arac_capa", "ped_capa"):
            return fy.kos(cr, lr, gt, s, prof, cfg, kanca=capa)
        if orb is None:
            return None
        C, harita = orb
        if yontem == "orb":
            temel = fy.kos(cr, lr, gt, s, prof, cfg)
            return op.bagla(C, harita, gt, s, temel)
        hib = op.olcek_kancasi(C, harita, gt, s)
        if yontem == "hibrit+capa":
            # arac capasi yalniz ORB izlemiyorken: maskeli kanca
            def capa_orbsuz(k, est, preds, _c=capa):
                if harita[k - 1] < 0:
                    _c(k, est, preds)
                elif s[k - 1]:
                    _c(k, est, preds)          # saglikli karede ogrenmeye devam
            return fy.kos(cr, lr, gt, s, prof, cfg, kanca=iki_kanca(hib, capa_orbsuz))
        return fy.kos(cr, lr, gt, s, prof, cfg, kanca=hib)
    return kos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=3)
    ap.add_argument("--oturum", nargs="*", default=OTURUMLAR)
    ap.add_argument("--yontem", nargs="*", default=YONTEMLER)
    a = ap.parse_args()
    cfg = EstimatorConfig(method="homografi")
    sonuc = {}             # (ad, yontem) -> dict(kontrol=[...], E=float|None, kapsam)
    for ad in a.oturum:
        if not fy.no.yatak_npz(ad, 8).exists():
            print(f"{ad}: RAFT onbellegi yok, atlandi")
            continue
        d = fy.veri(ad)
        kut = kutular(d)
        orb_var = (fy.CIKTI / "colmap" / ad / "orbslam3_poz.csv").exists()
        sozde_var = (fy.CIKTI / "colmap" / ad / "sozde_gt.npz").exists()
        print(f"\n=== {ad}  (ORB-SLAM3 {'var' if orb_var else 'YOK'}, sozde-GT {'var' if sozde_var else 'YOK'})",
              flush=True)
        biriken = {y: ([], []) for y in a.yontem}           # yontem -> (kontrol, E)
        sozde = fp.sozde_gt(ad) if sozde_var else None
        for i in range(a.n_perm):                            # olcum yuklemesi pahali: perm basina bir kez
            gt, cr, lr, prof, n = fy.onbellek_olcumleri(ad, karistir=i)
            orb = op.oku(ad, n) if orb_var else None
            for y in a.yontem:
                ks = kosucu(ad, y, cr, lr, gt, prof, cfg, kut, orb)
                p = ks(d["saglikli"])
                if p is None:
                    continue
                kontrol, E = biriken[y]
                if sozde_var:
                    E.append(fp.oturum_E(p, ad, sozde)[0])
                if ad in fy.KONTROLLER:
                    kontrol.append([e3 for _, e3, _ in fy.kontrol_hatalari(ad, ks)])
        for y in a.yontem:
            kontrol, E = biriken[y]
            if not (E or kontrol):
                continue
            r = dict(E=float(np.mean(E)) if E else None,
                     kontrol=list(np.mean(kontrol, axis=0)) if kontrol else [])
            sonuc[(ad, y)] = r
            ktxt = " | ".join(f"{v:7.2f}" for v in r["kontrol"])
            print(f"  {y:12s} E(sozde-GT) {('%8.2f' % r['E']) if r['E'] is not None else '     yok'} m"
                  f"   kontrol noktalari: {ktxt}", flush=True)
    yol = fy.CIKTI / "final_degerlendirme.json"
    eski = json.loads(yol.read_text(encoding="utf-8")) if yol.exists() else {}
    eski.update({f"{k[0]}|{k[1]}": v for k, v in sonuc.items()})     # oturumlar ayri kosulabilir
    yol.write_text(json.dumps(eski, indent=1, default=float), encoding="utf-8")

    # puan / sira: dort resmi oturumun hepsinde sozde-GT E'si varsa
    if all((o, "temel") in sonuc and sonuc[(o, "temel")]["E"] is not None for o in fp.TABLO_OTURUMLARI):
        Eg = {o: fp.oturum_E(fy.veri(o)["konum"], o)[0] for o in fp.TABLO_OTURUMLARI}
        model = fp.PuanModeli(Eg)
        yontemler = {"gonderilen (resmi)": Eg}
        for y in a.yontem:
            if all((o, y) in sonuc and sonuc[(o, y)]["E"] is not None for o in fp.TABLO_OTURUMLARI):
                yontemler[y] = {o: sonuc[(o, y)]["E"] for o in fp.TABLO_OTURUMLARI}
        print("\n" + model.tablo(yontemler))


if __name__ == "__main__":
    main()
