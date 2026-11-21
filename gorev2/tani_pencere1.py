# -*- coding: utf-8 -*-
"""GOREV 2 — TANI: sinav penceresi 1 neden pencere 2'nin 3,5 kati kotu?

SORU
----
Genelleme sinavi ortalamasi 11,08 m ama hata neredeyse tamamen tek pencerede:

    pencere 1 : 17,31 m      pencere 2 : 4,85 m

Iki pencerenin OLU HESAP uzunlugu neredeyse ayni (600 kare vs 600 kare), yani fark
SURE degil. `OLCUM_TURU_2.md` §5'in ilk acik isi bu; denenen hicbir `min_inlier_ratio`
degeri ve yumusak harmanlama orayi duzeltmedi.

Bu betik uretim yolunu (onbellekli, `kabul.py` ile ayni) kosturur ama her karede durum
makinesinin ICINI kaydeder ve iki pencereyi yan yana koyar:

  A) YUKSEKLIK   : kestirilen `height` / gercek yukseklik orani kesinti boyunca
  B) AYRISTIRMA  : hatanin olcek / donukluk / oteleme paylari (Procrustes)
  C) BIRIKIM     : hata nerede hizlaniyor
  D) OLCUM       : n_kept, inlier, merdiven adayinin kapiya takilma orani
  E) KALIBRASYON : son saglikli penceredeki YON CESITLILIGI (Procrustes'in M'yi
                   belirleyebilmesi icin gerekli) — pencere 1 duz ucusta kalibre
                   oluyorsa M'nin donme bileseni kotu kosullanir
  F) UCUS        : GT'den yol, net yer degistirme, yon degisimi, irtifa araligi

NEDEN ONEMLI
------------
Cikti bir sayi degil bir MEKANIZMA. Pencere 1'in cokusu olcek/yukseklik kaynakliysa
bir sonraki faz (mutlak yukseklik gozlemi) dogru hedefe bakiyor demektir; donukluk
kaynakliysa oraya girmeden once durup yeniden dusunmek gerekir.

CALISTIR
--------
    .venv\\Scripts\\python gorev2\\tani_pencere1.py
    .venv\\Scripts\\python gorev2\\tani_pencere1.py --n-perm 5
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

GOREV2 = Path(__file__).resolve().parent
sys.path.insert(0, str(GOREV2))

from kabul import PENCERELER, _yukle, _geo_birlestir              # noqa: E402
from nokta_onbellek import YATAKLAR                               # noqa: E402
from gorev2_src.evaluation import GT_COLS, _meas_from_row         # noqa: E402
from gorev2_src.vo_pipeline import EstimatorConfig, VoLadderEstimator  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------- kosucu
def kos_tanili(cons, lad, gt, healthy, prof, cfg, spacing):
    """`evaluation.run_estimator_ladder` ile AYNI kosu — ustune kare kare ic durum."""
    n = len(gt)
    est = VoLadderEstimator(prof, cfg, spacing=spacing)
    preds = np.zeros((n, 3))
    preds[0] = gt[0]
    est.update_ladder(0, None, None, gt[0], True)

    cons_rows = [_meas_from_row(r) for r in cons.to_dict("records")]
    lad_rows = [_meas_from_row(r) for r in lad.to_dict("records")]

    # merdiven adayinin kapiya takilip takilmadigini kaydet (uretim koduna dokunmadan)
    kutu = {"k": 0, "aday": {}}
    _orij = est._ladder_candidate

    def _sarmal(meas, hop):
        r = _orij(meas, hop)
        kutu["aday"][kutu["k"]] = (r is not None)
        return r

    est._ladder_candidate = _sarmal

    kayit = []
    hizalama_son = None                # son saglikli karedeki hizalama (dondurulur)
    kalib_son = None                   # o andaki kalibrasyon tamponu (E, G)
    for k in range(1, n):
        kutu["k"] = k
        mc = cons_rows[k - 1] if k - 1 < len(cons_rows) else None
        ml = lad_rows[k - 1] if k - 1 < len(lad_rows) else None
        s = bool(healthy[k])
        preds[k] = est.update_ladder(k, mc, ml, gt[k] if s else None, s)

        if s and est.alignment is not None:
            hizalama_son = est.alignment
            kalib_son = (np.array(est._buf_E), np.array(est._buf_G))

        a = est._anchor
        kayit.append(dict(
            k=k, saglikli=s,
            h_durum=est.height, psi=est.psi,
            h_capa=(a or {}).get("h"), capa_ord=(a or {}).get("ord"),
            merdiven_aday=kutu["aday"].get(k),
            c_nkept=(mc.n_kept if mc else 0),
            c_inlier=(mc.hom_inlier_ratio if mc else 0.0),
            l_nkept=(ml.n_kept if ml else 0),
            l_inlier=(ml.hom_inlier_ratio if ml else 0.0),
        ))
    return preds, pd.DataFrame(kayit), hizalama_son, kalib_son


# --------------------------------------------------------------------------- analiz
def procrustes_2b(P, G):
    """G ~ c·R·P en iyi uyumu. Doner: (c, theta_derece)."""
    if len(P) < 3 or (P ** 2).sum() < 1e-12:
        return np.nan, np.nan
    U, S, Vt = np.linalg.svd(G.T @ P)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U = U.copy(); U[:, -1] *= -1
        R = U @ Vt
    c = float(S.sum() / (P ** 2).sum())
    return c, float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))


def ayristir(preds, gt, bas, dil):
    """Puanlanan blokta hatayi olcek / donukluk / kalan diye ayir.

    Olu hesap `bas` karesinde GT'ye capalidir; oradan itibaren yer degistirme
    vektorlerine bakariz. `c` ve `theta` TUM olu hesap uzunlugunda fit edilir
    (puanlanan blogun kendisine degil — aksi halde ayristirma kendini dogrular).
    """
    p = preds[bas:, :2] - gt[bas, :2]
    g = gt[bas:, :2] - gt[bas, :2]
    c, th = procrustes_2b(p, g)
    if not np.isfinite(c):
        return None
    t = np.radians(th)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])

    def _hata(P2):
        tam = np.column_stack([gt[bas, 0] + P2[:, 0], gt[bas, 1] + P2[:, 1],
                               preds[bas:, 2]])
        d = dil.start - bas, dil.stop - bas
        return float(np.linalg.norm(tam[d[0]:d[1]] - gt[dil], axis=1).mean())

    return dict(
        c=c, theta=th,
        ham=_hata(p),
        olcek_duzeltilse=_hata(c * p),
        yon_duzeltilse=_hata((R @ p.T).T),
        ikisi_de=_hata(c * (R @ p.T).T),
        z_hata=float(np.abs(preds[dil, 2] - gt[dil, 2]).mean()),
        xy_hata=float(np.linalg.norm(preds[dil, :2] - gt[dil, :2], axis=1).mean()),
    )


def yon_cesitliligi(EG):
    """Kalibrasyon tamponundaki yon cesitliligi.

    Procrustes M'yi `H = GᵀE` tekil ayrismasindan bulur. Adimlarin hepsi ayni
    yondeyse H neredeyse rank-1 olur ve M'nin donme bileseni kotu kosullanir.
    Olcut: kovaryansin ozdegerleri ve orani.
    """
    if EG is None:
        return None
    E, G = EG
    if len(G) < 3:
        return None
    Gc = G - G.mean(axis=0)
    w = np.linalg.eigvalsh(Gc.T @ Gc / len(Gc))
    w = np.sort(np.abs(w))[::-1]
    acilar = np.degrees(np.arctan2(G[:, 1], G[:, 0]))
    # yon yayilimi: dairesel standart sapma
    c_, s_ = np.cos(np.radians(acilar)), np.sin(np.radians(acilar))
    R_ = np.hypot(c_.mean(), s_.mean())
    dairesel_std = np.degrees(np.sqrt(-2 * np.log(max(R_, 1e-12))))
    return dict(n=len(G), lam1=w[0], lam2=w[1],
                oran=float(w[0] / max(w[1], 1e-15)),
                yon_std=float(dairesel_std),
                adim_med=float(np.median(np.linalg.norm(G, axis=1))))


def ucus_ozeti(gt, bas, son):
    d = np.diff(gt[bas:son + 1], axis=0)
    yol = float(np.linalg.norm(d[:, :2], axis=1).sum())
    net = float(np.linalg.norm(gt[son, :2] - gt[bas, :2]))
    aci = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
    dyaw = np.abs((np.diff(aci) + 180) % 360 - 180)
    return dict(yol=yol, net=net, dogrusallik=net / max(yol, 1e-9),
                hiz_med=float(np.median(np.linalg.norm(d[:, :2], axis=1))),
                yon_degisim=float(dyaw.sum()),
                z_aralik=float(np.ptp(gt[bas:son + 1, 2])),
                z_net=float(gt[son, 2] - gt[bas, 2]))


# --------------------------------------------------------------------------- main
def pencere_tanisi(w, cfg, spacing, karistir):
    p = PENCERELER[w]
    son = p["son"]
    gt_t, cons, lad, prof, n = _yukle("yarisma", spacing, karistir, _geo_birlestir(cfg, {}))
    d = np.load(YATAKLAR["yarisma"]["gt_npz"])
    gercek = d["saglikli"]

    h = gercek[:son].copy()
    a, b = p["puan"]
    h[a:b + 1] = False
    gt = gt_t[:son]

    preds, kyt, hiz, kalib = kos_tanili(
        cons.iloc[:son - 1], lad.iloc[:son - 1], gt, h, prof, cfg, spacing)

    # olu hesabin basi: puanlanan blogun onundeki kesintisiz kesinti dizisinin basi
    bas = a
    while bas > 0 and not h[bas - 1]:
        bas -= 1
    dil = slice(a, b + 1)

    # gercek yukseklik: son saglikli karedeki hizalama dondurulup GT z ile tasinir
    h_ger = np.array([hiz.current_height(float(z)) for z in gt[:, 2]]) if hiz else None

    return dict(
        w=w, bas=bas, puan=(a, b), n_olu=a - bas,
        hata=float(np.linalg.norm(preds[dil] - gt[dil], axis=1).mean()),
        ayr=ayristir(preds, gt, bas, dil),
        kalib=yon_cesitliligi(kalib),
        ucus_olu=ucus_ozeti(gt, bas, b),
        ucus_kalib=ucus_ozeti(gt, max(0, bas - 150), bas - 1),
        hiz=dict(height=hiz.height, resid=hiz.residual_ratio, n=hiz.n_used,
                 z_sign=hiz.z_sign) if hiz else None,
        preds=preds, gt=gt, kyt=kyt, h_ger=h_ger, healthy=h,
    )


def rapor(sonuclar, n_perm):
    P = "=" * 78
    print(f"\n{P}\nA) SONUC VE HATA AYRISTIRMASI\n{P}")
    print(f"  {'':<24}{'pencere 1':>16}{'pencere 2':>16}")
    sat = lambda ad, f: print(f"  {ad:<24}" + "".join(f"{f(s):>16}" for s in sonuclar))
    sat("olu hesap [kare]", lambda s: f"{s['n_olu']}")
    sat("puanlanan blok", lambda s: f"{s['puan'][0]}-{s['puan'][1]}")
    sat("HATA [m]", lambda s: f"{s['hata']:.2f}")
    sat("  yatay (xy)", lambda s: f"{s['ayr']['xy_hata']:.2f}")
    sat("  dusey (z)", lambda s: f"{s['ayr']['z_hata']:.2f}")
    print()
    sat("olcek c", lambda s: f"{s['ayr']['c']:.4f}")
    sat("donukluk theta [deg]", lambda s: f"{s['ayr']['theta']:+.2f}")
    print()
    sat("ham hata [m]", lambda s: f"{s['ayr']['ham']:.2f}")
    sat("olcek duzeltilse", lambda s: f"{s['ayr']['olcek_duzeltilse']:.2f}")
    sat("yon duzeltilse", lambda s: f"{s['ayr']['yon_duzeltilse']:.2f}")
    sat("ikisi de", lambda s: f"{s['ayr']['ikisi_de']:.2f}")

    print(f"\n{P}\nB) YUKSEKLIK DURUMU (kestirilen / gercek)\n{P}")
    for s in sonuclar:
        if s["h_ger"] is None:
            continue
        kyt, bas, b = s["kyt"], s["bas"], s["puan"][1]
        sub = kyt[(kyt.k >= bas) & (kyt.k <= b)]
        oran = sub.h_capa.to_numpy(float) / s["h_ger"][sub.k.to_numpy()]
        print(f"  pencere {s['w']}  capa yuksekligi orani: "
              f"bas {oran[0]:.3f} -> son {oran[-1]:.3f} | "
              f"ort |sapma| %{100*np.abs(oran-1).mean():.1f} | "
              f"en kotu %{100*np.abs(oran-1).max():.1f}")
        adim = max(len(oran) // 6, 1)
        iz = "  ".join(f"{sub.k.iloc[i]}:{oran[i]:.3f}" for i in range(0, len(oran), adim))
        print(f"             {iz}")
        print(f"             gercek yukseklik {s['h_ger'][bas]:.1f} -> "
              f"{s['h_ger'][b]:.1f} m")

    print(f"\n{P}\nC) HATA BIRIKIMI (olu hesap boyunca)\n{P}")
    for s in sonuclar:
        e = np.linalg.norm(s["preds"] - s["gt"], axis=1)
        bas, b = s["bas"], s["puan"][1]
        adim = max((b - bas) // 8, 1)
        iz = "  ".join(f"{k}:{e[k]:.1f}" for k in range(bas, b + 1, adim))
        print(f"  pencere {s['w']}: {iz}")

    print(f"\n{P}\nD) OLCUM KALITESI\n{P}")
    print(f"  {'':<24}{'pencere 1':>16}{'pencere 2':>16}")
    def _ol(s, kol, sadece_olu=True):
        kyt = s["kyt"]
        sub = kyt[(kyt.k >= s["bas"]) & (kyt.k <= s["puan"][1])] if sadece_olu else kyt
        return sub[kol]
    sat("ardisik n_kept med", lambda s: f"{_ol(s,'c_nkept').median():.0f}")
    sat("ardisik inlier med", lambda s: f"{_ol(s,'c_inlier').median():.3f}")
    sat("merdiven n_kept med", lambda s: f"{_ol(s,'l_nkept').median():.0f}")
    sat("merdiven inlier med", lambda s: f"{_ol(s,'l_inlier').median():.3f}")
    sat("merdiven aday RED %", lambda s:
        f"{100*(1-_ol(s,'merdiven_aday').fillna(False).mean()):.1f}")

    print(f"\n{P}\nE) KALIBRASYON PENCERESI (Procrustes M'yi belirleyebiliyor mu?)\n{P}")
    print(f"  {'':<24}{'pencere 1':>16}{'pencere 2':>16}")
    sat("ornek sayisi", lambda s: f"{s['kalib']['n']}" if s['kalib'] else "-")
    sat("adim medyani [m]", lambda s: f"{s['kalib']['adim_med']:.3f}" if s['kalib'] else "-")
    sat("yon std [deg]", lambda s: f"{s['kalib']['yon_std']:.1f}" if s['kalib'] else "-")
    sat("ozdeger orani L1/L2", lambda s: f"{s['kalib']['oran']:.1f}" if s['kalib'] else "-")
    sat("hizalama artik orani", lambda s: f"{s['hiz']['resid']:.4f}" if s['hiz'] else "-")
    sat("hizalama yuksekligi", lambda s: f"{s['hiz']['height']:.2f}" if s['hiz'] else "-")

    print(f"\n{P}\nF) UCUS (GT)\n{P}")
    print(f"  {'':<24}{'pencere 1':>16}{'pencere 2':>16}")
    for ad, anh in (("OLU HESAP", "ucus_olu"), ("KALIB. ONCESI", "ucus_kalib")):
        print(f"  --- {ad} ---")
        for kol, fmt in (("yol", "{:.0f} m"), ("net", "{:.0f} m"),
                         ("dogrusallik", "{:.3f}"), ("hiz_med", "{:.3f} m/kare"),
                         ("yon_degisim", "{:.0f} deg"), ("z_aralik", "{:.1f} m"),
                         ("z_net", "{:+.1f} m")):
            sat(f"  {kol}", lambda s, k=kol, f=fmt: f.format(s[anh][k]))

    print(f"\n{P}")
    print("  Not: sonuclar tek permutasyondur (karistir=0). RANSAC nokta sirasina")
    print(f"  duyarli; ana rakamlar {n_perm} permutasyonda tekrarlandi (yukarida ort).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=3)
    ap.add_argument("--spacing", type=int, default=8)
    args = ap.parse_args()

    cfg = EstimatorConfig(method="homografi")

    # ana rakamlar: permutasyon ortalamasi
    tablo = {w: [] for w in (1, 2)}
    detay = []
    for i in range(args.n_perm):
        for w in (1, 2):
            s = pencere_tanisi(w, cfg, args.spacing, karistir=i)
            tablo[w].append(s)
            if i == 0:
                detay.append(s)
        print(f"  permutasyon {i}: p1 {tablo[1][-1]['hata']:.2f} m · "
              f"p2 {tablo[2][-1]['hata']:.2f} m", flush=True)

    print(f"\n{'=' * 78}\nGENELLEME SINAVI — {args.n_perm} permutasyon\n{'=' * 78}")
    for w in (1, 2):
        v = np.array([s["hata"] for s in tablo[w]])
        c = np.array([s["ayr"]["c"] for s in tablo[w]])
        t = np.array([s["ayr"]["theta"] for s in tablo[w]])
        print(f"  pencere {w}: hata {v.mean():6.2f} ± {v.std(ddof=1) if len(v)>1 else 0:.2f} m"
              f" | olcek {c.mean():.4f} ± {c.std(ddof=1) if len(c)>1 else 0:.4f}"
              f" | yon {t.mean():+.2f}° ± {t.std(ddof=1) if len(t)>1 else 0:.2f}")
    ort = np.mean([np.mean([s["hata"] for s in tablo[w]]) for w in (1, 2)])
    print(f"  ORTALAMA: {ort:.2f} m")

    rapor(detay, args.n_perm)


if __name__ == "__main__":
    main()
