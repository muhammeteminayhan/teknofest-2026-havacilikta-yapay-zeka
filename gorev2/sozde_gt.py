# -*- coding: utf-8 -*-
"""GOREV 2 — SOZDE-GT: kesinti karelerinin konumunu COLMAP ile cevrimdisi kestir.

NEDEN VAR
---------
Finalde kesinti karelerinin gercek konumu sunucu tarafindan hic verilmedi. Bir yontemin
o oturumlarda kac puan alacagini olcmek icin bu konumlara ihtiyac var. Canli sistemin
yapamadigi seyi cevrimdisi yapabiliriz: butun kareleri birlikte kullanan bir yapi-hareketten
(SfM) rekonstruksiyon + global bundle adjustment, sonra saglikli karelerin GT'sine Sim(3).

Bu bir OLCUM ARACIdir, yarisma sistemi degil: nedensel olmak zorunda degil.

DOGRULUGU NASIL BILIYORUZ
-------------------------
  * video1 / video2 (tam GT): yalniz ilk 450 kareyle oturt -> 1800 karede gercek hata
    (O2/O4/Test'in birebir benzeri).
  * O1 / O3: yalniz ilk saglikli blokla oturt (--hizalama ilk) -> sonraki saglikli bloklarda
    gercek hata.

ORTAM: ayri venv (ana ortamin numpy'ini korumak icin)
    .venv_colmap\\Scripts\\python gorev2\\sozde_gt.py video1
    .venv_colmap\\Scripts\\python gorev2\\sozde_gt.py final_o4 --dongu
Girdi `final_yatak.py --colmap-hazirla <ad>` ile uretilir.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pycolmap

sys.stdout.reconfigure(encoding="utf-8")
KOK = Path(__file__).resolve().parent / "outputs" / "final" / "colmap"


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- Sim(3)
def sim3_fit(A: np.ndarray, B: np.ndarray):
    """Umeyama: B ~= s*R*A + t  (yansima serbest; slam_degerlendirme.sim3_fit(aynalama=True))."""
    mu_a, mu_b = A.mean(0), B.mean(0)
    A0, B0 = A - mu_a, B - mu_b
    U, D, Vt = np.linalg.svd(A0.T @ B0 / len(A))
    # Yansimaya IZIN VERILIR: yarisma GT cercevesi sag-el SfM cercevesine gore aynali
    # (video1'de yasakken dusey hata 17,3 m, serbestken 6,1 m; bkz. slam_degerlendirme.sim3_fit).
    S = np.eye(3)
    R = Vt.T @ S @ U.T
    s = float((D * np.diag(S)).sum() / ((A0 ** 2).sum() / len(A)))
    return s, R, mu_b - s * (R @ mu_a)


def sim3_uygula(P, fit):
    s, R, t = fit
    return s * (R @ P.T).T + t


# --------------------------------------------------------------------------- COLMAP adimlari
def ciftler_yaz(yol: Path, adlar: list[str], komsu: int, ek: tuple[int, ...] = (16, 24, 32, 48),
                ekstra: list[tuple[str, str]] | None = None) -> int:
    n = len(adlar)
    ciftler = set()
    for i in range(n):
        for d in list(range(1, komsu + 1)) + list(ek):
            if i + d < n:
                ciftler.add((adlar[i], adlar[i + d]))
    for a, b in ekstra or []:
        if a != b and (b, a) not in ciftler:
            ciftler.add((a, b))
    yol.write_text("\n".join(f"{a} {b}" for a, b in sorted(ciftler)), encoding="utf-8")
    return len(ciftler)


def ozellik_cikar(db: Path, img: Path, adlar: list[str], kamera: np.ndarray, maks_ozellik: int):
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "OPENCV"
    reader.camera_params = ",".join(f"{v:.6f}" for v in kamera)
    ext = pycolmap.FeatureExtractionOptions()
    ext.max_image_size = 2000
    ext.sift.max_num_features = maks_ozellik
    ext.num_threads = -1
    pycolmap.extract_features(db, img, image_names=adlar, camera_mode=pycolmap.CameraMode.SINGLE,
                              reader_options=reader, extraction_options=ext,
                              device=pycolmap.Device.cpu)


def eslestir(db: Path, ciftler: Path):
    mo = pycolmap.FeatureMatchingOptions()
    mo.num_threads = -1
    po = pycolmap.ImportedPairingOptions()
    po.match_list_path = str(ciftler)
    pycolmap.match_image_pairs(db, matching_options=mo, pairing_options=po,
                               device=pycolmap.Device.cpu)


def haritala(db: Path, img: Path, cikti: Path, global_: bool = False) -> dict:
    cikti.mkdir(parents=True, exist_ok=True)
    if global_:
        # COLMAP 4 global SfM (GLOMAP): donme ortalamasi + global konumlandirma + BA.
        # Artimli haritalamadan cok daha hizli; ic parametreler yine sabit.
        o = pycolmap.GlobalPipelineOptions()
        ba = o.mapper.bundle_adjustment
        ba.refine_focal_length = False
        ba.refine_principal_point = False
        ba.refine_extra_params = False
        o.min_model_size = 30
        return pycolmap.global_mapping(db, img, cikti, options=o)
    o = pycolmap.IncrementalPipelineOptions()
    o.ba_refine_focal_length = False          # kalibrasyon biliniyor
    o.ba_refine_principal_point = False
    o.ba_refine_extra_params = False
    o.multiple_models = True
    o.min_model_size = 30
    o.num_threads = -1
    return pycolmap.incremental_mapping(db, img, cikti, options=o)


def merkezler(rec) -> dict[int, np.ndarray]:
    out = {}
    for im in rec.images.values():
        if hasattr(im, "has_pose") and not im.has_pose:
            continue
        c = im.projection_center() if hasattr(im, "projection_center") \
            else im.cam_from_world().inverse().translation
        out[int(Path(im.name).stem)] = np.asarray(c, float)
    return out


def noktalar(rec, maks: int = 60000) -> np.ndarray:
    X = np.array([p.xyz for p in rec.points3D.values()], float)
    if len(X) > maks:
        X = X[np.random.default_rng(0).choice(len(X), maks, replace=False)]
    return X


def model_oku(yol) -> tuple[dict[int, np.ndarray], np.ndarray]:
    rec = pycolmap.Reconstruction(str(yol))
    return merkezler(rec), noktalar(rec)


def _rot(ax, a):
    ax = ax / np.linalg.norm(ax)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def yatis_duzeltmesi(X_w: np.ndarray, B: np.ndarray):
    """Tek saglikli blokla oturtmanin belirleyemedigi TEK serbestligi zeminden al.

    Saglikli blok neredeyse bir cizgiyse (O1'de uzunluk/genislik 8,8) Sim(3), o cizgi
    etrafindaki yatisi (roll) kotu belirler; kesintide dusey hata mesafeyle buyur. Zemin
    noktalarinin ortalamada yatay oldugu varsayimiyla yalniz ANA EKSEN etrafindaki yatis
    duzeltilir; cizgi boyunca egim Sim(3)'te kalir. Olculdu: O1 gizli blok dusey
    12,4 -> 5,6 m, video1 kesinti dusey 3,0 -> 2,1 m (yatay degismiyor).
    Doner: (Q, merkez) — duzeltilmis = Q·(P - merkez) + merkez.
    """
    m = X_w.mean(0)
    for _ in range(3):                          # kirpilmis duzlem: catilar/agaclar disarida
        _, _, Vt = np.linalg.svd(X_w - m, full_matrices=False)
        d = (X_w - m) @ Vt[2]
        k = np.abs(d) < np.percentile(np.abs(d), 80)
        m = X_w[k].mean(0)
    _, _, Vt = np.linalg.svd(X_w[k] - m, full_matrices=False)
    nrm = Vt[2] if Vt[2][2] > 0 else -Vt[2]
    c0 = B.mean(0)
    _, evec = np.linalg.eigh(np.cov(B[:, :2].T))
    ana = np.r_[evec[:, 1], 0.0]
    nperp = nrm - (nrm @ ana) * ana
    nperp /= np.linalg.norm(nperp)
    z = np.array([0.0, 0.0, 1.0])
    a = np.arctan2(np.cross(nperp, z) @ ana, nperp @ z)
    return _rot(ana, a), c0


# --------------------------------------------------------------------------- hizalama + rapor
def _tek_blok(maske: np.ndarray) -> bool:
    i = np.flatnonzero(maske)
    return len(i) > 0 and (i[-1] - i[0] + 1) == len(i)


def hizala(modeller: list[tuple[dict[int, np.ndarray], np.ndarray]], konum, saglikli,
           kullan: np.ndarray, n: int):
    """Her modeli, icerdigi `kullan` karelerinin GT'sine Sim(3) ile oturt.

    `kullan` tek bir bitisik blok ise ana eksen etrafindaki yatis zeminden duzeltilir
    (`yatis_duzeltmesi`); birden fazla saglikli blokta tam Sim(3) iyi kosulludur.
    Doner: (sozde konum (n,3) NaN'li, model indeksi (n,), ozet listesi)."""
    P = np.full((n, 3), np.nan)
    hangi = np.full(n, -1)
    ozet = []
    for mi, (mer, X) in sorted(enumerate(modeller), key=lambda t: -len(t[1][0])):
        idx = np.array(sorted(mer))
        kul = idx[kullan[idx]]
        if len(kul) < 10:
            ozet.append(dict(model=mi, kare=len(idx), hizalanan=0, not_="GT'li kare yok"))
            continue
        A = np.array([mer[k] for k in kul])
        fit = sim3_fit(A, konum[kul])
        tum = sim3_uygula(np.array([mer[k] for k in idx]), fit)
        Aw = sim3_uygula(A, fit)
        if _tek_blok(kullan) and len(X) > 100:
            Q, c0 = yatis_duzeltmesi(sim3_uygula(X, fit), konum[kul])
            tum = (Q @ (tum - c0).T).T + c0
            Aw = (Q @ (Aw - c0).T).T + c0
        art = np.linalg.norm(Aw - konum[kul], axis=1)
        for k, p in zip(idx, tum):
            if hangi[k] < 0:
                P[k], hangi[k] = p, mi
        ozet.append(dict(model=mi, kare=len(idx), hizalanan=int(len(kul)),
                         kalinti_rms=float(np.sqrt((art ** 2).mean())), olcek=fit[0],
                         aralik=(int(idx.min()), int(idx.max()))))
    return P, hangi, ozet


def aradegerle(P: np.ndarray, maks_bosluk: int = 8) -> np.ndarray:
    P = P.copy()
    ok = np.isfinite(P[:, 0])
    idx = np.flatnonzero(ok)
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= maks_bosluk + 1:
            for k in range(a + 1, b):
                w = (k - a) / (b - a)
                P[k] = (1 - w) * P[a] + w * P[b]
    return P


def hata_raporu(etiket, P, konum, maske):
    m = maske & np.isfinite(P[:, 0])
    if not m.any():
        return dict(etiket=etiket, n=0)
    e = np.linalg.norm(P[m] - konum[m], axis=1)
    son = np.flatnonzero(m)[-50:]
    r = dict(etiket=etiket, n=int(m.sum()), kapsam=float(m.sum() / max(1, maske.sum())),
             ort=float(e.mean()), medyan=float(np.median(e)), p90=float(np.percentile(e, 90)),
             maks=float(e.max()),
             son50=float(np.linalg.norm(P[son] - konum[son], axis=1).mean()))
    log(f"  {etiket:34s} n={r['n']:4d} kapsam %{100 * r['kapsam']:.0f} | ort {r['ort']:7.2f} "
        f"medyan {r['medyan']:7.2f} p90 {r['p90']:7.2f} maks {r['maks']:7.2f} | son 50 kare {r['son50']:7.2f} m")
    return r


def bloklar(s):
    out, b = [], 0
    for i in range(1, len(s) + 1):
        if i == len(s) or s[i] != s[b]:
            out.append((bool(s[b]), b, i - 1))
            b = i
    return out


# --------------------------------------------------------------------------- ana akis
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ad")
    ap.add_argument("--komsu", type=int, default=12, help="ardisik eslestirme penceresi")
    ap.add_argument("--maks-ozellik", type=int, default=2048)
    ap.add_argument("--adim", type=int, default=3,
                    help="SfM her N. karede kurulur, aradakiler aradegerlenir (7,5 fps'te N=3 ~ 0,4 s)")
    ap.add_argument("--dongu", action="store_true",
                    help="1. rekonstruksiyondan uzaysal dongu ciftleri onerip yeniden haritala")
    ap.add_argument("--dongu-yaricap", type=float, default=30.0, help="metre")
    ap.add_argument("--n", type=int, default=None, help="ilk N kare (sure denemesi)")
    ap.add_argument("--global", dest="global_", action="store_true",
                    help="artimli yerine COLMAP 4 global haritalama (GLOMAP)")
    a = ap.parse_args()

    kok = KOK / a.ad
    g = np.load(kok / "girdi.npz")
    konum, saglikli, kareler = g["konum"], g["saglikli"], g["kareler"]
    tam_gt = bool(g["tam_gt"])
    if a.n:
        kareler = kareler[kareler < a.n]
    secim = (kareler % a.adim == 0) | (kareler == kareler.max())
    kareler = kareler[secim]
    n = len(konum)
    adlar = [f"{k:06d}.jpg" for k in kareler]
    calisma = kok / (("deneme" if a.n else "is") + f"_a{a.adim}_f{a.maks_ozellik}")
    calisma.mkdir(parents=True, exist_ok=True)
    db = calisma / "database.db"
    t0 = time.perf_counter()

    # asama isaretleri: kesilen bir kosu yarim veritabaniyla haritalamaya GECMEZ
    if not (calisma / "ozellik.tamam").exists():
        for f in calisma.glob("database.db*"):
            f.unlink()
        log(f"{a.ad}: {len(adlar)} kare, ozellik cikarma...")
        ozellik_cikar(db, kok / "images", adlar, g["kamera"], a.maks_ozellik)
        (calisma / "ozellik.tamam").touch()
        log(f"  ozellik: {time.perf_counter() - t0:.0f} s")
    if not (calisma / "eslestirme.tamam").exists():
        cf = calisma / "ciftler.txt"
        nc = ciftler_yaz(cf, adlar, a.komsu)
        log(f"  {nc} ardisik cift eslestiriliyor...")
        eslestir(db, cf)            # zaten eslesmis ciftler COLMAP tarafindan atlanir
        (calisma / "eslestirme.tamam").touch()
        log(f"  eslestirme: {time.perf_counter() - t0:.0f} s")

    harita = calisma / ("harita1g" if a.global_ else "harita1")
    if not (harita / "haritalama.tamam").exists():
        log(f"haritalama (1. gecis, {'global' if a.global_ else 'artimli'})...")
        haritala(db, kok / "images", harita, a.global_)
        (harita / "haritalama.tamam").touch()
        log(f"  haritalama: {time.perf_counter() - t0:.0f} s")
    modeller = [model_oku(p)
                for p in sorted(harita.iterdir()) if (p / "images.bin").exists()]

    if a.dongu:
        P1, _, _ = hizala(modeller, konum, saglikli, saglikli.copy(), n)
        P1 = aradegerle(P1)
        ok = np.flatnonzero(np.isfinite(P1[:, 0]))
        ekstra = []
        for i in ok[::3]:
            d = np.linalg.norm(P1[ok] - P1[i], axis=1)
            aday = ok[(d < a.dongu_yaricap) & (np.abs(ok - i) > 60)]
            for j in aday[np.argsort(d[np.isin(ok, aday)])][:5]:
                ekstra.append((f"{min(i, j):06d}.jpg", f"{max(i, j):06d}.jpg"))
        ekstra = [(x, y) for x, y in set(ekstra) if x in set(adlar) and y in set(adlar)]
        harita = calisma / ("harita2g" if a.global_ else "harita2")
        if not (harita / "haritalama.tamam").exists():
            cf2 = calisma / "dongu_ciftleri.txt"
            cf2.write_text("\n".join(f"{x} {y}" for x, y in sorted(ekstra)), encoding="utf-8")
            log(f"dongu: {len(ekstra)} uzaysal aday cift eslestiriliyor...")
            if ekstra:
                eslestir(db, cf2)
            log("haritalama (2. gecis, dongulerle)...")
            haritala(db, kok / "images", harita, a.global_)
            (harita / "haritalama.tamam").touch()
            log(f"  2. gecis: {time.perf_counter() - t0:.0f} s")
        modeller = [model_oku(p)
                    for p in sorted(harita.iterdir()) if (p / "images.bin").exists()]

    log(f"modeller: {[len(m[0]) for m in modeller]} kare (toplam {len(adlar)})")
    rapor = dict(ad=a.ad, kare=len(adlar), modeller=[len(m[0]) for m in modeller], dongu=a.dongu)

    # 1) uretim sozde-GT'si: butun saglikli karelerle oturt
    P, hangi, ozet = hizala(modeller, konum, saglikli, saglikli.copy(), n)
    P = aradegerle(P)
    rapor["hizalama_tum"] = ozet
    for o in ozet:
        log(f"  model {o['model']}: {o['kare']} kare, hizalanan {o.get('hizalanan')}, "
            f"kalinti RMS {o.get('kalinti_rms', float('nan')):.2f} m, aralik {o.get('aralik')}")

    # 2) dogrulama: yalniz ILK saglikli blokla oturt, gerisini gercek GT ile olc
    ilk = np.zeros(n, bool)
    b0 = bloklar(saglikli)[0]
    ilk[b0[1]:b0[2] + 1] = True
    Pd, _, _ = hizala(modeller, konum, saglikli, ilk, n)
    Pd = aradegerle(Pd)
    log("DOGRULAMA (yalniz ilk saglikli blokla oturtma):")
    if tam_gt:
        rapor["dogrulama"] = [hata_raporu("kesinti karelerinin tamami", Pd, konum, ~saglikli)]
        rapor["dogrulama"].append(hata_raporu("tum-saglikli oturtmayla (uretim)", P, konum, ~saglikli))
    else:
        rapor["dogrulama"] = []
        for s, x, y in bloklar(saglikli)[1:]:
            if s:
                m = np.zeros(n, bool)
                m[x:y + 1] = True
                rapor["dogrulama"].append(hata_raporu(f"gizli saglikli blok {x}-{y}", Pd, konum, m))
        if not rapor["dogrulama"]:
            log("  (oturumda ikinci saglikli blok yok — dogrulama video1/video2 uzerinden)")

    kap = np.isfinite(P[:, 0])
    rapor["kapsam_kesinti"] = float(kap[~saglikli].mean()) if (~saglikli).any() else 1.0
    log(f"kesinti karelerinde sozde-GT kapsami: %{100 * rapor['kapsam_kesinti']:.1f} | "
        f"toplam sure {time.perf_counter() - t0:.0f} s")
    np.savez(kok / ("sozde_gt_deneme.npz" if a.n else "sozde_gt.npz"), konum=P, kayitli=kap,
             model=hangi, dogrulama_konum=Pd)
    (kok / ("rapor_deneme.json" if a.n else "rapor.json")).write_text(
        json.dumps(rapor, indent=1, ensure_ascii=False, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
