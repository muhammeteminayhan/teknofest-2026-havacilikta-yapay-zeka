# -*- coding: utf-8 -*-
"""final-v1 DENKLIK TESTI: temiz paket, gelistirme deposuyla BIT-BIREBIR mi?

Temiz pakette 14 reddedilen ayar ve ~2000 satir deneme kodu cikarildi. Bunun
davranisi DEGISTIRMEDIGINI kanitlamak zorunlu: her iki uygulama ayni ham nokta
onbelleginden beslenir, ayni kesinti takviminde kosturulur ve tahminler
`np.array_equal` ile karsilastirilir.

Karsilastirilan katmanlar:
  1. OLCUM  — PairMeasurement alan alan (float bit esitligi)
  2. DURUM MAKINESI — kare kare konum tahmini

Calistir (proje kokunden ya da bu klasorden):
    python denklik_testi.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

BURASI = Path(__file__).resolve().parent
sys.path.insert(0, str(BURASI))

# --- gelistirme deposu (referans) ---
DEPO = Path.home() / "Desktop" / "havacilikta-yz-project"
SRC = DEPO / "gorev2" / "src"


def _referansi_yukle():
    if "gorev2_src" not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            "gorev2_src", SRC / "__init__.py", submodule_search_locations=[str(SRC)])
        mod = importlib.util.module_from_spec(spec)
        sys.modules["gorev2_src"] = mod
        spec.loader.exec_module(mod)
    import gorev2_src.vo_pipeline as ref_vp
    import gorev2_src.calibration as ref_cal
    return ref_vp, ref_cal


ORTAK_ALANLAR = [
    "ok", "frozen", "n_kept",
    "sim_ok", "sim_unit_dx", "sim_unit_dy", "sim_dz_unit", "sim_yaw", "sim_inlier_ratio",
    "hom_ok", "hom_ux", "hom_uy", "hom_uz", "hom_nx", "hom_ny", "hom_nz",
    "hom_yaw", "hom_inlier_ratio",
]


def _cift_noktalari(ob, i):
    a, b = int(ob["offsets"][i]), int(ob["offsets"][i + 1])
    return ob["pts0"][a:b].astype(np.float64), ob["pts1"][a:b].astype(np.float64)


def _olcumler(vp, profile, ob, n):
    """Ham noktalardan olcum dizisi uret (ikisi de ayni yoldan)."""
    out, prev_normal = [], None
    for i in range(n):
        if ob["donmus"][i]:
            m = vp.PairMeasurement()
            m.frozen = True
        else:
            p0, p1 = _cift_noktalari(ob, i)
            if len(p0) < 4:
                m = vp.PairMeasurement()
            else:
                m = vp.PairMeasurement.from_points(p0, p1, profile,
                                                   prev_normal=prev_normal)
                if m.hom_ok:
                    prev_normal = m.hom_normal
        out.append(m)
    return out


def _kos(vp, profile, cfg, cons, lad, gt, healthy, spacing=8):
    est = vp.VoLadderEstimator(profile, cfg, spacing=spacing)
    n = len(gt)
    preds = np.zeros((n, 3))
    preds[0] = gt[0]
    est.update_ladder(0, None, None, gt[0], True)
    for k in range(1, n):
        preds[k] = est.update_ladder(
            k, cons[k - 1] if k - 1 < len(cons) else None,
            lad[k - 1] if k - 1 < len(lad) else None,
            gt[k] if healthy[k] else None, bool(healthy[k]))
    return preds


def main():
    import konum
    from konum import vo_pipeline as tmz_vp

    ref_vp, ref_cal = _referansi_yukle()
    cikti = DEPO / "gorev2" / "outputs"

    YATAKLAR = [
        ("video1-RGB", "nokta_raft_cons.npz", "nokta_raft_sp8.npz",
         "THYZ_2026_Ornek_Veri_1_translation.csv", konum.RGB_1080P),
        ("video2-TERMAL", "nokta_raft_v2_cons.npz", "nokta_raft_v2_sp8.npz",
         "THYZ_2026_Ornek_Veri_2_Termal_translation.csv", konum.THERMAL),
    ]
    KESINTI = [(451, 900), (1051, 1550), (1651, 2250)]

    import pandas as pd
    print("=" * 78)
    print("final-v1 DENKLIK TESTI — temiz paket vs gelistirme deposu")
    print("=" * 78)
    tum_gecti = True

    for ad, c_npz, l_npz, gt_csv, prof_t in YATAKLAR:
        pc, pl = cikti / c_npz, cikti / l_npz
        gtp = DEPO / "THYZ_2026_Ornek_Veri_Seti" / gt_csv
        if not (pc.exists() and pl.exists() and gtp.exists()):
            print(f"\n{ad}: veri yok, atlandi")
            continue

        g = pd.read_csv(gtp)
        gt = g[["translation_x", "translation_y", "translation_z"]].to_numpy()[::4]
        ob_c = {k: np.load(pc)[k] for k in ("pts0", "pts1", "offsets", "donmus")}
        ob_l = {k: np.load(pl)[k] for k in ("pts0", "pts1", "offsets", "donmus")}
        n = min(2250, len(gt), len(ob_c["offsets"]), len(ob_l["offsets"]))
        gt = gt[:n]

        prof_r = ref_cal.profile_for_size(prof_t.width, prof_t.height)

        # --- 1) OLCUM katmani
        r_c = _olcumler(ref_vp, prof_r, ob_c, n - 1)
        t_c = _olcumler(tmz_vp, prof_t, ob_c, n - 1)
        r_l = _olcumler(ref_vp, prof_r, ob_l, n - 1)
        t_l = _olcumler(tmz_vp, prof_t, ob_l, n - 1)

        farkli = []
        for alan in ORTAK_ALANLAR:
            for rr, tt, kat in ((r_c, t_c, "ardisik"), (r_l, t_l, "merdiven")):
                a = np.array([getattr(x, alan) for x in rr], dtype=float)
                b = np.array([getattr(x, alan) for x in tt], dtype=float)
                if not np.array_equal(np.nan_to_num(a, nan=-9e9),
                                      np.nan_to_num(b, nan=-9e9)):
                    farkli.append(f"{kat}.{alan}")
        olcum_ok = not farkli

        # --- 2) DURUM MAKINESI
        healthy = np.ones(n, bool)
        for a, b in KESINTI:
            healthy[a:b] = False
        p_ref = _kos(ref_vp, prof_r, ref_vp.EstimatorConfig(method="homografi"),
                     r_c, r_l, gt, healthy)
        p_tmz = _kos(tmz_vp, prof_t, tmz_vp.EstimatorConfig(method="homografi"),
                     t_c, t_l, gt, healthy)
        durum_ok = np.array_equal(p_ref, p_tmz)
        hata_r = float(np.linalg.norm(p_ref - gt, axis=1).mean())
        hata_t = float(np.linalg.norm(p_tmz - gt, axis=1).mean())

        print(f"\n--- {ad} ({n} kare, profil {prof_t.name}) ---")
        print(f"  olcum katmani     : {'BIREBIR' if olcum_ok else 'FARKLI -> ' + ', '.join(farkli[:5])}")
        print(f"  durum makinesi    : {'BIREBIR' if durum_ok else 'FARKLI'}")
        print(f"  resmi metrik      : referans {hata_r:.6f} m · temiz {hata_t:.6f} m")
        tum_gecti &= olcum_ok and durum_ok

    print("\n" + "=" * 78)
    print(f"SONUC: {'TUM KATMANLAR BIT-BIREBIR' if tum_gecti else 'FARK VAR — SEVK ETME'}")
    return 0 if tum_gecti else 1


if __name__ == "__main__":
    sys.exit(main())
