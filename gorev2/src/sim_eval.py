"""Prova sonuçlarının değerlendirilmesi: simülatör kayıtlarından E, tempo, eksik.

Simülatör her tahmini zaman damgasıyla JSONL'e yazar; bu modül kayıtları
GT/manifest ile birleştirip yarışma metriğini, kesinti-içi hatayı, gönderim
temposunu (80/dk limiti) ve eksik kareleri raporlar.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CIKTI_DIR = Path(__file__).resolve().parent.parent / "outputs" / "sim_kayitlari"


def evaluate_run(seed: int) -> dict:
    manifest = json.loads((CIKTI_DIR / f"manifest_seed{seed}.json").read_text(encoding="utf-8"))
    gt = np.array(manifest["gt"])
    healthy = np.array(manifest["healthy"], dtype=bool)
    n = manifest["n_frames"]

    preds = np.full((n, 3), np.nan)
    zamanlar = []
    with open(CIKTI_DIR / f"predictions_seed{seed}.jsonl", encoding="utf-8") as f:
        for satir in f:
            kayit = json.loads(satir)
            p = kayit["payload"]
            i = int(p["frame"].rstrip("/").split("/")[-1])
            t = p["detected_translations"][0]
            preds[i] = [float(t["translation_x"]), float(t["translation_y"]),
                        float(t["translation_z"])]
            zamanlar.append(kayit["zaman"])

    eksik = int(np.isnan(preds[:, 0]).sum())
    dolu = ~np.isnan(preds[:, 0])
    hata = np.linalg.norm(preds[dolu] - gt[dolu], axis=1)

    zamanlar = np.array(sorted(zamanlar))
    # herhangi bir 60 sn'lik pencerede azami gonderim sayisi
    maks_tempo = 0
    for i in range(len(zamanlar)):
        maks_tempo = max(maks_tempo, int(np.searchsorted(
            zamanlar, zamanlar[i] + 60.0) - i))

    m_saglikli = healthy[dolu]
    return dict(
        seed=seed,
        n_frames=n,
        gonderilen=int(dolu.sum()),
        eksik=eksik,
        E=float(hata.mean()),
        E_kesinti=float(hata[~m_saglikli].mean()) if (~m_saglikli).any() else 0.0,
        E_saglikli=float(hata[m_saglikli].mean()) if m_saglikli.any() else 0.0,
        kesinti_orani=float((~healthy).mean()),
        toplam_sure_dk=float((zamanlar[-1] - zamanlar[0]) / 60) if len(zamanlar) > 1 else 0.0,
        maks_tempo_60s=maks_tempo,
    )


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    r = evaluate_run(seed)
    print(f"--- prova raporu (seed {r['seed']}) ---")
    print(f"  gonderilen kare : {r['gonderilen']}/{r['n_frames']}  (eksik: {r['eksik']})")
    print(f"  E (Denklem 2)   : {r['E']:.3f} m")
    print(f"  E kesinti ici   : {r['E_kesinti']:.3f} m  (kayip orani %{100*r['kesinti_orani']:.0f})")
    print(f"  E saglikli      : {r['E_saglikli']:.6f} m  (0 olmali: aynalama)")
    print(f"  toplam sure     : {r['toplam_sure_dk']:.1f} dk")
    print(f"  maks tempo      : {r['maks_tempo_60s']}/60sn  (limit 80)")
