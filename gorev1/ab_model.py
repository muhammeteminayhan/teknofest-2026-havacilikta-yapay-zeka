# -*- coding: utf-8 -*-
"""GOREV 1 — IKI MODEL A/B: mevcut YOLOv8s vs DINOv3-YOLO (arkadas calismasi).

NEDEN AYRI ARAC: `ab_cozunurluk.py` ayni modeli iki COZUNURLUKTE karsilastiriyordu.
Burada iki farkli MODEL var ve bunlar iki farkli ultralytics surumu istiyor:

  * mevcut  -> kurulu ultralytics 8.4.92
  * aday    -> DINOv3-YOLO fork'u (8.4.80); agirlik `ultralytics.nn.modules.dinov3`
               sinifina pickle bagimli, kurulu surumle YUKLENEMEZ

Ayni surecte iki ultralytics olamayacagi icin her model **ayri alt-surecte** kosar ve
tespitlerini JSON'a yazar; karsilastirma ucuncu adimda yapilir. Fork'a erisim yalnizca
`sys.path` ile GECICIDIR — `.venv` degistirilmez (Gorev 3 ve mevcut Gorev 1 risk almasin).

Karsilastirma mantigi (iou/esle/kayip_turu/ciz) ve karar olcutleri `ab_cozunurluk.py`'den
oldugu gibi devralinir; boylece iki tur ayni olcutle degerlendirilmis olur.

Calistir (aday agirlik + gerekiyorsa fork yolu disaridan verilir):
    .venv\\Scripts\\python gorev1\\ab_model.py --aday <yol/best.pt> --fork-yolu <ultralytics-fork>
    .venv\\Scripts\\python gorev1\\ab_model.py --aday <yol/best.pt>          # fork gerekmiyorsa
    .venv\\Scripts\\python gorev1\\ab_model.py --aday <...> --panel 8

ILK KULLANIM (28 Tem 2026): DINOv3-YOLO adayi olculdu ve KULLANICI KARARIYLA UYGULANDI —
tam kayit: `denetim/MODEL_DEGISIMI.md`. Ozet: aday Tasit (+54) ve UAP'de (+12) ustun,
Insan (64->20) ve UAI'de (12->7) geriliyor; 640 (adayin egitim cozunurlugu) 1280'den
belirgin iyi. Olcum NT'de dusus riskine isaret ediyordu, risk bilinerek kabul edildi.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

GOREV1 = Path(__file__).resolve().parent
ROOT = GOREV1.parent
OUT = GOREV1 / "denetim"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(GOREV1))
sys.stdout.reconfigure(encoding="utf-8")

from ab_cozunurluk import (SPEC, KARE_BUTCE_MS, VARSAYILAN_BLOKLAR,  # noqa: E402
                           kutu, esle, kayip_turu, ciz, kareler_bul)

MEVCUT_PT = GOREV1 / "agirliklar" / "best.pt"
MIN_ANLAMLI_ALAN = 40 * 40      # ab_cozunurluk ile ayni gurultu olcutu


# ---------------------------------------------------------------- alt surec
def _alt_surec_kos(model_pt: Path, fork: Path | None, bloklar, imgsz, cikti: Path, conf=0.25, takipsiz=False):
    """Bir modeli AYRI SURECTE kostur (ultralytics surum cakismasini onler)."""
    cmd = [sys.executable, str(Path(__file__).resolve()), "--kos",
           "--model", str(model_pt), "--imgsz", str(imgsz), "--conf", str(conf),
           *(["--takipsiz"] if takipsiz else []),
           "--cikti", str(cikti), "--bloklar-json", json.dumps(bloklar)]
    if fork is not None:
        cmd += ["--fork", str(fork)]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(r.stdout[-2000:])
        print(r.stderr[-3000:])
        sys.exit(f"HATA: alt surec basarisiz ({model_pt.name})")
    return json.loads(cikti.read_text(encoding="utf-8"))


def _kos_modu(args):
    """Alt surec govdesi: modeli yukle, kareleri isle, JSON yaz."""
    if args.fork:                       # fork'u GECICI olarak one al
        sys.path.insert(0, args.fork)
    import ultralytics
    from gorev1.dedektor import Gorev1Detector

    bloklar = [[Path(p) for p in blok] for blok in json.loads(args.bloklar_json)]
    sonuc, sureler = {}, []
    for blok in bloklar:
        d = Gorev1Detector(weights_path=args.model, imgsz=args.imgsz,
                           conf=args.conf)                              # blok basi taze takip
        for p in blok:
            img = cv2.imread(str(p))
            t0 = time.perf_counter()
            # TAKIPSIZ mod: Gorev 1 PUANI mAP ile hesaplanir ve mAP HAM TESPITLER
            # uzerinden olculur; BoT-SORT yalnizca hareket durumu icin gereklidir.
            # Tracker kendi esiklerini (new_track_thresh=0.25) uyguladigi icin dusuk
            # guvenli tespitleri eler ve modellerin saf tespit gucunu gizler.
            if args.takipsiz:
                r = d.model.predict(img, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
                objs = []
                for b, k in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist()):
                    objs.append({"cls": d.spec_cls(int(k)),
                                 "top_left_x": int(b[0]), "top_left_y": int(b[1]),
                                 "bottom_right_x": int(b[2]), "bottom_right_y": int(b[3])})
            else:
                objs = d.detect(img)
            sureler.append((time.perf_counter() - t0) * 1000)
            for o in objs:
                o["cls"] = int(o["cls"])
            sonuc[p.name] = objs
    Path(args.cikti).write_text(json.dumps({
        "tespitler": sonuc,
        "ms": float(np.mean(sureler)) if sureler else 0.0,
        "ultralytics": ultralytics.__version__,
        "model": args.model,
    }), encoding="utf-8")


# ---------------------------------------------------------------- gorsel
def _panel(kare_yolu: Path, objs_a, objs_b, yalniz_a, yalniz_b, cikti: Path):
    """Ayni kare, iki model yan yana. Yesil = yalniz o modelde, mavi = ortak."""
    img = cv2.imread(str(kare_yolu))
    if img is None:
        return False
    renk_a = {i: ((0, 255, 0), "YALNIZ") if i in yalniz_a else ((255, 160, 0), "")
              for i in range(len(objs_a))}
    renk_b = {j: ((0, 255, 0), "YALNIZ") if j in yalniz_b else ((255, 160, 0), "")
              for j in range(len(objs_b))}
    sol, sag = ciz(img, objs_a, renk_a), ciz(img, objs_b, renk_b)
    for im, ad, n in ((sol, "MEVCUT (YOLOv8s)", len(objs_a)),
                      (sag, "ADAY (DINOv3-YOLO)", len(objs_b))):
        cv2.rectangle(im, (0, 0), (im.shape[1], 54), (0, 0, 0), -1)
        cv2.putText(im, f"{ad}  —  {n} tespit", (12, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
    ayrac = np.full((sol.shape[0], 6, 3), 255, np.uint8)
    cv2.imwrite(str(cikti), np.hstack([sol, ayrac, sag]))
    return True


# ---------------------------------------------------------------- ana
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kos", action="store_true", help="(dahili) alt surec modu")
    ap.add_argument("--model", default=None)
    ap.add_argument("--fork", default=None)
    ap.add_argument("--cikti", default=None)
    ap.add_argument("--bloklar-json", dest="bloklar_json", default=None)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--takipsiz", action="store_true", help="BoT-SORT olmadan ham tespit (mAP'e yakin)")
    ap.add_argument("--bloklar", default=VARSAYILAN_BLOKLAR)
    ap.add_argument("--oturum", default="THYZ_2026_Online_Yarisma")
    ap.add_argument("--panel", type=int, default=6)
    ap.add_argument("--aday", default=None, help="karsilastirilacak aday agirlik (.pt)")
    # Iki kol FARKLI cozunurluk isteyebilir: her agirlik KENDI egitim cozunurlugunde
    # kosturulmali (ViT govdeler egitimden uzaklasinca bozulur). Ornek: mevcut
    # best_dinov3.pt 640'ta, trial-8 1280'de egitildi.
    ap.add_argument("--mevcut", default=None,
                    help="referans (mevcut) agirlik; verilmezse agirliklar/best.pt")
    ap.add_argument("--imgsz-aday", dest="imgsz_aday", type=int, default=None,
                    help="aday kolunun imgsz'i (verilmezse --imgsz ile ayni)")
    ap.add_argument("--fork-yolu", dest="fork_yolu", default=None,
                    help="aday ozel bir ultralytics fork'u istiyorsa o dizin (gecici sys.path)")
    args = ap.parse_args()

    if args.kos:
        return _kos_modu(args)
    if not args.aday:
        sys.exit("HATA: --aday <agirlik.pt> verilmeli (karsilastirilacak model).")
    ADAY_PT = Path(args.aday)
    # DIKKAT: main() icinde atandigi icin MEVCUT_PT burada YEREL degiskendir;
    # kosullu atama UnboundLocalError uretirdi. Her durumda ataniyor.
    MEVCUT_PT = Path(args.mevcut) if args.mevcut else GOREV1 / "agirliklar" / "best.pt"
    if not MEVCUT_PT.exists():
        sys.exit(f"HATA: mevcut agirlik yok: {MEVCUT_PT}")
    IMGSZ_ADAY = args.imgsz_aday if args.imgsz_aday is not None else args.imgsz
    FORK = Path(args.fork_yolu) if args.fork_yolu else None
    if not ADAY_PT.exists():
        sys.exit(f"HATA: aday agirlik yok: {ADAY_PT}")

    OUT.mkdir(parents=True, exist_ok=True)
    tum = kareler_bul(args.oturum)
    bloklar = []
    for parca in args.bloklar.split(","):
        bas, _, uz = parca.partition(":")
        bas, uz = int(bas), int(uz or 12)
        blok = [str(p) for p in tum[bas:bas + uz]]
        if blok:
            bloklar.append(blok)
    n_kare = sum(len(b) for b in bloklar)
    print(f"Oturum: {len(tum)} kare | test: {len(bloklar)} ardisik blok, {n_kare} kare")
    print(f"MEVCUT: {MEVCUT_PT.name} @ imgsz={args.imgsz} | "
          f"ADAY: {ADAY_PT.name} @ imgsz={IMGSZ_ADAY}\n")

    print("[1/2] MEVCUT kosuluyor...")
    A = _alt_surec_kos(MEVCUT_PT, None, bloklar, args.imgsz, OUT / "_ab_a.json", 0.25, args.takipsiz)
    print(f"      {A['ms']:.0f} ms/kare | ultralytics {A['ultralytics']}")
    print("[2/2] ADAY kosuluyor...")
    B = _alt_surec_kos(ADAY_PT, FORK, bloklar, IMGSZ_ADAY, OUT / "_ab_b.json", args.conf, args.takipsiz)
    print(f"      {B['ms']:.0f} ms/kare | ultralytics {B['ultralytics']}\n")

    sonuc_a, sonuc_b = A["tespitler"], B["tespitler"]
    yeni_c, kayip_c, ortak_c = Counter(), Counter(), Counter()
    gercek_kayip_c, yer_deg_c = Counter(), Counter()
    a_c, b_c = Counter(), Counter()
    yeni_alan = []
    bos_a = bos_b = 0
    kare_farki = []
    for ad in sonuc_a:
        oa, ob = sonuc_a[ad], sonuc_b.get(ad, [])
        bos_a += not oa
        bos_b += not ob
        for o in oa:
            a_c[o["cls"]] += 1
        for o in ob:
            b_c[o["cls"]] += 1
        ortak, y_a, y_b = esle(oa, ob)
        for i, _ in ortak:
            ortak_c[oa[i]["cls"]] += 1
        for i in y_a:
            kayip_c[oa[i]["cls"]] += 1
            tur = kayip_turu(oa[i], ob, y_b)
            (gercek_kayip_c if tur == "gercek_kayip" else yer_deg_c)[oa[i]["cls"]] += 1
        for j in y_b:
            yeni_c[ob[j]["cls"]] += 1
            x1, y1, x2, y2 = kutu(ob[j])
            yeni_alan.append((x2 - x1) * (y2 - y1))
        kare_farki.append((len(y_b) - len(y_a), ad, y_a, y_b))

    n_yeni, n_gercek_kayip = sum(yeni_c.values()), sum(gercek_kayip_c.values())
    n_a, n_b = sum(a_c.values()), sum(b_c.values())
    kayip_orani = n_gercek_kayip / max(n_a, 1)
    yeni_medyan = int(np.median(yeni_alan)) if yeni_alan else 0
    gurultu_suphesi = bool(yeni_alan) and yeni_medyan < MIN_ANLAMLI_ALAN
    butce_ok = B["ms"] <= KARE_BUTCE_MS

    # ---- gorsel kanit: en cok fark yaratan kareler ----
    kare_farki.sort(key=lambda t: -abs(t[0]))
    yollar = {p.name: p for p in tum}
    cizilen = 0
    for fark, ad, y_a, y_b in kare_farki:
        if cizilen >= args.panel:
            break
        p = yollar.get(ad)
        if p and _panel(p, sonuc_a[ad], sonuc_b.get(ad, []), y_a, y_b,
                        OUT / f"ab_{args.imgsz}_{cizilen:02d}_{ad.replace('.webp','')}.png"):
            cizilen += 1

    # ---- rapor ----
    L = ["=== GOREV 1 — IKI MODEL A/B: mevcut YOLOv8s vs DINOv3-YOLO ===",
         f"{n_kare} gercek yarisma karesi ({len(bloklar)} ardisik blok), imgsz={args.imgsz}",
         f"uretim yolu: Gorev1Detector + BoT-SORT takip (ikisi de ayni)", "",
         f"{'':<22}{'MEVCUT':>10}{'ADAY':>10}",
         f"{'toplam tespit':<22}{n_a:>10}{n_b:>10}",
         f"{'bos kare':<22}{bos_a:>10}{bos_b:>10}",
         f"{'ms/kare':<22}{A['ms']:>10.0f}{B['ms']:>10.0f}",
         f"{'ultralytics':<22}{A['ultralytics']:>10}{B['ultralytics']:>10}", "",
         "--- sinif bazinda tespit ---",
         f"{'sinif':<10}{'MEVCUT':>9}{'ADAY':>9}{'ortak':>9}{'yalniz aday':>13}{'gercek kayip':>14}"]
    for c in sorted(set(a_c) | set(b_c)):
        L.append(f"{SPEC.get(c, '?'):<10}{a_c[c]:>9}{b_c[c]:>9}{ortak_c[c]:>9}"
                 f"{yeni_c[c]:>13}{gercek_kayip_c[c]:>14}")
    L += ["",
          f"yer degistirme (olcum artefakti, kayip DEGIL): {sum(yer_deg_c.values())}",
          "",
          "--- olcutler (ab_cozunurluk.py ile ayni) ---",
          f"[{'OK ' if n_yeni > n_gercek_kayip else 'RED'}] net kazanc: "
          f"+{n_yeni} yalniz-aday / -{n_gercek_kayip} gercek kayip",
          f"[{'OK ' if kayip_orani <= 0.10 else 'RED'}] kayip orani: %{100*kayip_orani:.1f} (esik %10)",
          f"[{'OK ' if not gurultu_suphesi else 'RED'}] yeni tespit boyutu: medyan {yeni_medyan} px2 "
          f"({yeni_medyan**0.5:.0f}x{yeni_medyan**0.5:.0f}) — esik {MIN_ANLAMLI_ALAN} px2",
          f"[{'OK ' if butce_ok else 'RED'}] sure butcesi: {B['ms']:.0f} ms <= {KARE_BUTCE_MS:.0f} ms",
          f"[{'OK ' if bos_b <= bos_a else 'RED'}] bos kare azaldi mi: {bos_a} -> {bos_b}", ""]
    karar = (n_yeni > n_gercek_kayip) and (kayip_orani <= 0.10) and butce_ok \
        and (bos_b <= bos_a) and not gurultu_suphesi
    if karar:
        L += [">>> TAVSIYE: ADAY (DINOv3-YOLO) entegre edilebilir.",
              f"    Gorsel panelde (denetim/ab_model_*.png) YESIL kutularin gercek nesne",
              f"    oldugu GOZLE dogrulanmalidir."]
    else:
        L += [">>> TAVSIYE: MEVCUT model kalsin — aday yukaridaki RED olcutlerinden kaldi."]
    L += ["", f"gorsel kanit: {cizilen} panel -> denetim/ab_model_*.png"]
    metin = "\n".join(L)
    (OUT / f"ab_model_{args.imgsz}_c{args.conf:g}{'_takipsiz' if args.takipsiz else ''}.txt").write_text(metin, encoding="utf-8")
    print(metin)


if __name__ == "__main__":
    main()
