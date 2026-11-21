# -*- coding: utf-8 -*-
"""GOREV 2 — ORB-SLAM3 (Docker) + Umeyama: kosu ve yarisma mimarisine baglama.

1. KOSU. `orbslam3:yerel` imaji (orbslam3_docker/Dockerfile) icinde `mono_nedensel`,
   `final_yatak.py --colmap-hazirla` ile disa aktarilan JPEG karelerde kosar ve HER KAREDE
   o an dondurulen pozu yazar (nedensel). Kamera yaml'i ayni disa aktarimin parametrelerinden.

2. BAGLAMA (nedensel, yarismadaki gibi):
   * Saglikli kare: GT aynalanir; o anki haritanin ORB merkezleri ile GT'den Umeyama Sim(3)
     (slam_degerlendirme.sim3_fit) — yalniz GECMIS saglikli kareler.
   * Kesinti karesi, harita izliyor ve Sim(3)'u var: konum = Sim(3)(ORB merkezi).
   * Iz kaybi ya da kesintide dogan YENI harita: RAFT temel cizgisinin adimlariyla ilerlenir;
     yeni harita K kare boyunca izlerse, o karelerdeki ORB merkezleri ile bizim (RAFT'la
     ilerleyen) konumlarimiz arasinda Sim(3) kurulup haritaya gecilir.

    .venv\\Scripts\\python gorev2\\orbslam3_poz.py --kos video1
    .venv\\Scripts\\python gorev2\\orbslam3_poz.py --degerlendir video1 video2 ...
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

GOREV2 = Path(__file__).resolve().parent
sys.path.insert(0, str(GOREV2))
from slam_degerlendirme import sim3_fit, sim3_uygula             # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
CIKTI = GOREV2 / "outputs" / "final"
DOCKER = r"C:\Program Files\Docker\Docker\resources\bin\docker.exe"
IZLIYOR = (2, 5)          # OK, OK_KLT


def yaml_yaz(ad: str, n_ozellik: int = 2500) -> Path:
    kok = CIKTI / "colmap" / ad
    g = np.load(kok / "girdi.npz")
    fx, fy, cx, cy, k1, k2, p1, p2 = g["kamera"]
    w, h = g["boyut"]
    metin = f"""%YAML:1.0
File.version: "1.0"
Camera.type: "PinHole"
Camera1.fx: {fx:.6f}
Camera1.fy: {fy:.6f}
Camera1.cx: {cx:.6f}
Camera1.cy: {cy:.6f}
Camera1.k1: {k1:.6f}
Camera1.k2: {k2:.6f}
Camera1.p1: {p1:.6f}
Camera1.p2: {p2:.6f}
Camera.width: {int(w)}
Camera.height: {int(h)}
Camera.fps: 8
Camera.RGB: 0
ORBextractor.nFeatures: {n_ozellik}
ORBextractor.scaleFactor: 1.2
ORBextractor.nLevels: 8
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7
Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1.0
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2.0
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3.0
Viewer.ViewpointX: 0.0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -1.8
Viewer.ViewpointF: 500.0
"""
    yol = kok / "orbslam3.yaml"
    yol.write_text(metin, encoding="utf-8")
    return yol


def kos(ad: str, n_ozellik: int = 2500, etiket: str = "") -> Path:
    yaml_yaz(ad, n_ozellik)
    son = f"_{etiket}" if etiket else ""
    csv = CIKTI / "colmap" / ad / f"orbslam3_poz{son}.csv"
    komut = [DOCKER, "run", "--rm", "-v", f"{CIKTI}:/veri", "orbslam3:yerel", "mono_nedensel",
             "/opt/ORB_SLAM3/Vocabulary/ORBvoc.txt", f"/veri/colmap/{ad}/orbslam3.yaml",
             f"/veri/colmap/{ad}/images", f"/veri/colmap/{ad}/orbslam3_poz{son}.csv", "7.5"]
    print(" ".join(komut), flush=True)
    with open(CIKTI / "colmap" / ad / f"orbslam3_kos{son}.log", "w", encoding="utf-8") as f:
        subprocess.run(komut, stdout=f, stderr=subprocess.STDOUT, check=False)
    return csv


def oku(ad: str, n: int, dosya: str = "orbslam3_poz.csv") -> tuple[np.ndarray, np.ndarray]:
    """(merkezler (n,3) NaN'li, harita kimligi (n,) -1 = izlemiyor)."""
    d = pd.read_csv(CIKTI / "colmap" / ad / dosya)
    C = np.full((n, 3), np.nan)
    harita = np.full(n, -1)
    m, izledi, koptu = 0, False, False
    for r in d.itertuples():
        k, durum = int(r.kare), int(r.durum)
        if k >= n:
            continue
        if durum in (1, 4):                  # baslatilmadi / kayip
            koptu = izledi or koptu          # izlemis bir harita koptuysa sonraki OK yeni harita
            continue
        if durum in IZLIYOR:
            if koptu:
                m, koptu = m + 1, False
            C[k] = [r.cx, r.cy, r.cz]
            harita[k] = m
            izledi = True
    return C, harita


def bagla(C, harita, gt, saglikli, temel, min_ornek=30, K_yeni=30) -> np.ndarray:
    """ORB merkezlerini nedensel olarak yarisma cercevesine bagla (bkz. modul notu)."""
    n = len(gt)
    P = np.zeros((n, 3))
    fit: dict[int, tuple] = {}
    ornek: dict[int, list] = {}       # harita -> saglikli kare indeksleri
    yeni: dict[int, list] = {}        # harita -> kesintide (kare) listesi (bootstrap icin)
    for k in range(n):
        m = int(harita[k])
        if saglikli[k]:
            P[k] = gt[k]
            if m >= 0:
                ornek.setdefault(m, []).append(k)
                if len(ornek[m]) >= min_ornek:
                    f = sim3_fit(C[ornek[m]], gt[ornek[m]], aynalama=True)
                    if f is not None:
                        fit[m] = f
            continue
        if m >= 0 and m in fit:
            P[k] = sim3_uygula(C[k][None], fit[m])[0]
            continue
        # RAFT temel cizgisinin adimiyla ilerle
        P[k] = (P[k - 1] + (temel[k] - temel[k - 1])) if k else gt[0]
        if m >= 0:
            yeni.setdefault(m, []).append(k)
            if len(yeni[m]) >= K_yeni:
                idx = yeni[m][-K_yeni:]
                f = sim3_fit(C[idx], P[idx], aynalama=True)
                if f is not None and f[0] > 0:
                    fit[m] = f
    return P


def olcek_kancasi(C, harita, gt, saglikli, W: int = 40, aralik: int = 8, kazanc: float = 1.0,
                  sinir=(0.8, 1.25), min_ornek: int = 30, bant: float = 0.0):
    """HIBRIT: ORB-SLAM3'u konum degil OLCEK kaynagi olarak kullan.

    `kahin_olcek.py`'deki metre/piksel kahininin GT yerine ORB-SLAM3 ile yapilan hali:
    kesintide son W karede, ORB merkezlerinin (saglikli karelerde kurulmus Sim(3) ile
    metreye cevrilmis) YATAY yol uzunlugu / bizim tahminlerimizin yol uzunlugu oraniyla
    yukseklik durumunu duzeltir. Yon ve konum RAFT'tan gelmeye devam eder; ORB-SLAM3'un
    olcegi kararli (video1: 300 karelik pencerelerde +-%2-3) ama yonu/konumu suruklenir.
    """
    ornek: dict[int, list] = {}
    fit: dict[int, tuple] = {}

    def kanca(k, est, preds):
        j = k - 1
        m = int(harita[j]) if j >= 0 else -1
        if saglikli[j]:
            if m >= 0:
                ornek.setdefault(m, []).append(j)
                if len(ornek[m]) >= min_ornek and len(ornek[m]) % 10 == 0:
                    f = sim3_fit(C[ornek[m]], gt[ornek[m]], aynalama=True)
                    if f is not None:
                        fit[m] = f
            return
        if j % aralik or est.height is None or m < 0 or m not in fit:
            return
        bas = j
        while bas > max(0, j - W) and not saglikli[bas - 1] and harita[bas - 1] == m:
            bas -= 1
        if j - bas < 10:
            return
        Q = sim3_uygula(C[bas:j + 1], fit[m])
        L_orb = np.linalg.norm(np.diff(Q[:, :2], axis=0), axis=1).sum()
        L_biz = np.linalg.norm(np.diff(preds[bas:j + 1, :2], axis=0), axis=1).sum()
        if L_orb < 3.0 or L_biz < 0.5:
            return
        r = float(np.log(L_orb / L_biz))
        if abs(r) <= bant:
            return
        r -= np.sign(r) * bant
        rho = float(np.clip(np.exp(kazanc * r), *sinir))
        est.height = est._clamp_h(est.height * rho)
        a = est._anchor
        if a is not None and a.get("h") is not None:
            a["h"] = est._clamp_h(a["h"] * rho)

    return kanca


def degerlendir(adlar: list[str]):
    from gorev2_src.evaluation import competition_error
    import final_yatak as fy
    from kabul import _yukle, _geo_birlestir
    from kahin_olcek import TAKVIMLER
    from gorev2_src.evaluation import _meas_from_row
    from gorev2_src.vo_pipeline import EstimatorConfig
    cfg = EstimatorConfig(method="homografi")
    for ad in adlar:
        if not (CIKTI / "colmap" / ad / "orbslam3_poz.csv").exists():
            print(f"{ad}: ORB-SLAM3 cikti yok")
            continue
        if ad in fy.OTURUMLAR:
            print(f"{ad}: final oturumlari sozde-GT ile final_puan.py uzerinden degerlendirilir")
            continue
        gt, cons, lad, prof, n = _yukle(ad, 8, 0, _geo_birlestir(cfg, {}))
        cr = [_meas_from_row(x) for x in cons.to_dict("records")]
        lr = [_meas_from_row(x) for x in lad.to_dict("records")]
        C, harita = oku(ad, n)
        iz = np.isfinite(C[:, 0]).mean()
        print(f"\n{ad}: ORB-SLAM3 izleme orani %{100 * iz:.0f}, harita sayisi {harita.max() + 1}")
        for t, fn in TAKVIMLER.items():
            s = fn(n)
            temel = fy.kos(cr, lr, gt, s, prof, cfg)
            P = bagla(C, harita, gt, s, temel)
            VAR = {"b0": dict(kazanc=1.0), "b0.05": dict(kazanc=1.0, bant=0.05),
                   "b0.10": dict(kazanc=1.0, bant=0.10)}
            hib = {v: competition_error(fy.kos(cr, lr, gt, s, prof, cfg,
                                               kanca=olcek_kancasi(C, harita, gt, s, **kw)), gt)
                   for v, kw in VAR.items()}
            e_t, e_o = competition_error(temel, gt), competition_error(P, gt)
            print(f"  {t:10s} RAFT {e_t:7.2f} | ORB+Umeyama {e_o:7.2f} | HIBRIT "
                  + " ".join(f"{v} {e:7.2f}" for v, e in hib.items()), flush=True)
            SONUC.append((ad, t, e_t, e_o, *hib.values()))
    if SONUC:
        a = np.array([r[2:] for r in SONUC])
        print()
        print(f"ORTALAMA ({len(SONUC)} yatak-takvim): RAFT {a[:, 0].mean():.2f} | ORB+Umeyama "
              f"{a[:, 1].mean():.2f} | hibrit b0 {a[:, 2].mean():.2f} | b0.05 {a[:, 3].mean():.2f} | "
              f"b0.10 {a[:, 4].mean():.2f}")
        for i, ad in ((2, "hibrit b0"), (3, "hibrit b0.05"), (4, "hibrit b0.10")):
            kot = np.max(a[:, i] / a[:, 0] - 1)
            print(f"  {ad}: {int((a[:, i] < a[:, 0]).sum())}/{len(a)} yatak-takvimde RAFT'tan iyi | "
                  f"en kotu {100 * kot:+.0f}%")


SONUC: list = []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kos", nargs="*")
    ap.add_argument("--degerlendir", nargs="*")
    ap.add_argument("--n-ozellik", type=int, default=2500)
    ap.add_argument("--etiket", default="", help="tekrar kosulari icin cikti eki (r2, r3 ...)")
    a = ap.parse_args()
    for ad in a.kos or []:
        kos(ad, a.n_ozellik, a.etiket)
    if a.degerlendir:
        degerlendir(a.degerlendir)


if __name__ == "__main__":
    main()
