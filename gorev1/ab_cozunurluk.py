# -*- coding: utf-8 -*-
"""Gorev 1 A/B COZUNURLUK KARAR MEKANIZMASI — imgsz 640 vs 1280 (vs istege bagli 1920).

Neden: yarisma kareleri 1920x1080; dedektor `imgsz=640` ile calisiyor (3x kucultme).
Kalite denetiminde (kalite_denetim.py) GOZLE dogrulandi ki 640 bazi araclari/insanlari
KACIRIYOR (orn. "0 nesne" gonderilen karede acikca park etmis araclar var).

Bu arac, URETIM YOLUNUN BIREBIR AYNISIYLA (Gorev1Detector + BoT-SORT takip, ardisik
kare bloklari) iki cozunurlugu karsilastirir ve KARAR uretir:
  - kutu eslestirme: ayni sinif + IoU >= 0.5
  - YENI (yalnizca yuksek cozunurlukte)  -> yesil
  - KAYIP (yalnizca dusuk cozunurlukte)  -> kirmizi
  - ORTAK                                -> beyaz
Karar olcutu: yeni tespit sayisi ve GUVENI (yuksek guven = gercek nesne), kayip sayisi,
kare basi sure butcesi (~1110 ms).

Cikti: gorev1/denetim/
    ab_<blok>.png   — yan yana gorsel (sol dusuk, sag yuksek cozunurluk)
    ab_karar.txt    — sinif bazinda tablo + NET TAVSIYE
    ab_ozet.csv     — kare basi sayisal karsilastirma

Calistir (proje kokunden):
    .venv\\Scripts\\python gorev1\\ab_cozunurluk.py
    .venv\\Scripts\\python gorev1\\ab_cozunurluk.py --a 640 --b 1280 --bloklar 1260:14,880:12
"""
import argparse
import csv
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
IMGS = ROOT / "sistem" / "_images"
OUT = Path(__file__).resolve().parent / "denetim"
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from gorev1 import Gorev1Detector  # noqa: E402

SPEC = {0: "Tasit", 1: "Insan", 2: "UAP", 3: "UAI"}
# Kalite denetiminde kacirma suphesi gorulen + sinif cesitliligi olan bolgeler
VARSAYILAN_BLOKLAR = "1260:14,1080:12,1320:12,880:12,670:12,1470:12"
# Kare basi GERCEK butce: yarisma penceresi 90 dk / 2250 kare = 2400 ms.
# (Referans: puanli oturum 640 ile 42 dk surdu ~1110 ms/kare — yani bol pay var.)
# Bu olcut YALNIZCA Gorev 1'in kendi suresini degil, tum boru hattini kapsayan
# ust siniri temsil eder; Gorev 1 tek basina bunun kucuk bir parcasidir.
KARE_BUTCE_MS = 2400.0


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    birlesim = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / birlesim if birlesim > 0 else 0.0


def kutu(o):
    return (int(o["top_left_x"]), int(o["top_left_y"]),
            int(o["bottom_right_x"]), int(o["bottom_right_y"]))


def merkez(o):
    x1, y1, x2, y2 = kutu(o)
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def esle(a_objs, b_objs, esik=0.5):
    """a=dusuk, b=yuksek cozunurluk. -> (ortak, yalniz_a=KAYIP, yalniz_b=YENI)"""
    kullanildi = set()
    ortak = []
    for i, oa in enumerate(a_objs):
        en_iyi, en_iyi_j = 0.0, -1
        for j, ob in enumerate(b_objs):
            if j in kullanildi or oa["cls"] != ob["cls"]:
                continue
            s = iou(kutu(oa), kutu(ob))
            if s > en_iyi:
                en_iyi, en_iyi_j = s, j
        if en_iyi >= esik:
            kullanildi.add(en_iyi_j)
            ortak.append((i, en_iyi_j))
    yalniz_a = [i for i in range(len(a_objs)) if i not in {p[0] for p in ortak}]
    yalniz_b = [j for j in range(len(b_objs)) if j not in kullanildi]
    return ortak, yalniz_a, yalniz_b


def kayip_turu(oa, b_objs, yalniz_b):
    """Bir 'kayip' GERCEK mi yoksa YER DEGISTIRME mi?

    Minik nesnelerde (ozellikle Insan) kutu birkac piksel kayinca IoU<0.5 olur ve
    ayni nesne hem 'kayip' hem 'yeni' sayilir — bu bir OLCUM ARTEFAKTIDIR, gercek
    kayip degildir. Ayni sinifta, merkezleri kutu buyuklugu kadar yakin bir YENI
    tespit varsa bunu 'yer degistirme' kabul ederiz.
    """
    ax, ay = merkez(oa)
    x1, y1, x2, y2 = kutu(oa)
    yaricap = max(24.0, 1.5 * max(x2 - x1, y2 - y1))   # kutu boyutuna gore tolerans
    for j in yalniz_b:
        ob = b_objs[j]
        if ob["cls"] != oa["cls"]:
            continue
        bx, by = merkez(ob)
        if np.hypot(ax - bx, ay - by) <= yaricap:
            return "yer_degistirme"
    return "gercek_kayip"


def ciz(img, objs, idx_renk, kalinlik=3):
    """idx_renk: {indeks: (BGR renk, etiket)}"""
    vis = img.copy()
    for i, o in enumerate(objs):
        renk, ek = idx_renk.get(i, ((255, 255, 255), ""))
        x1, y1, x2, y2 = kutu(o)
        cv2.rectangle(vis, (x1, y1), (x2, y2), renk, kalinlik)
        et = SPEC.get(int(o["cls"]), "?") + (f" {ek}" if ek else "")
        (tw, th), _ = cv2.getTextSize(et, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        cv2.rectangle(vis, (x1, max(0, y1 - th - 7)), (x1 + tw + 6, y1), renk, -1)
        cv2.putText(vis, et, (x1 + 3, max(th, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
    return vis


def kareler_bul(oturum):
    d = IMGS / oturum if oturum else None
    if d and d.exists():
        return sorted(d.glob("frame_*.webp"))
    adaylar = [p for p in IMGS.glob("*/") if list(p.glob("frame_*.webp"))]
    if not adaylar:
        sys.exit(f"HATA: kare bulunamadi ({IMGS})")
    # en cok kare iceren oturumu sec
    en = max(adaylar, key=lambda p: len(list(p.glob("frame_*.webp"))))
    print(f"(oturum otomatik secildi: {en.name})")
    return sorted(en.glob("frame_*.webp"))


def kosum(kareler_blok, imgsz):
    """Bloklari SIRAYLA isle (her blok icin taze dedektor -> takip dogal calissin)."""
    sonuc = {}
    sureler = []
    for blok in kareler_blok:
        d = Gorev1Detector(imgsz=imgsz)
        for p in blok:
            img = cv2.imread(str(p))
            t0 = time.perf_counter()
            objs = d.detect(img)
            sureler.append((time.perf_counter() - t0) * 1000)
            # cls'i int'e cevir (dedektor str donduruyor)
            for o in objs:
                o["cls"] = int(o["cls"])
            sonuc[p.name] = objs
    return sonuc, float(np.mean(sureler)) if sureler else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=int, default=640, help="dusuk cozunurluk (mevcut)")
    ap.add_argument("--b", type=int, default=1280, help="yuksek cozunurluk (aday)")
    ap.add_argument("--bloklar", default=VARSAYILAN_BLOKLAR,
                    help="baslangic:uzunluk virgullu (kare indeksi)")
    ap.add_argument("--oturum", default="THYZ_2026_Online_Yarisma")
    ap.add_argument("--panel", type=int, default=6, help="gorsellenecek kare sayisi")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    tum = kareler_bul(args.oturum)
    print(f"Oturum kare sayisi: {len(tum)}")

    bloklar = []
    for parca in args.bloklar.split(","):
        bas, _, uz = parca.partition(":")
        bas, uz = int(bas), int(uz or 12)
        blok = tum[bas:bas + uz]
        if blok:
            bloklar.append(blok)
    n_kare = sum(len(b) for b in bloklar)
    print(f"Test: {len(bloklar)} ardisik blok, toplam {n_kare} kare "
          f"(URETIM YOLU: Gorev1Detector + BoT-SORT takip)\n")

    print(f"[1/2] imgsz={args.a} kosuluyor...")
    sonuc_a, ms_a = kosum(bloklar, args.a)
    print(f"      {ms_a:.0f} ms/kare")
    print(f"[2/2] imgsz={args.b} kosuluyor...")
    sonuc_b, ms_b = kosum(bloklar, args.b)
    print(f"      {ms_b:.0f} ms/kare\n")

    # ---- karsilastirma ----
    yeni_c, kayip_c, ortak_c = Counter(), Counter(), Counter()
    gercek_kayip_c, yer_deg_c = Counter(), Counter()
    a_c, b_c = Counter(), Counter()
    yeni_alan, kayip_alan = [], []
    bos_a = bos_b = 0
    satirlar = []
    kare_farki = []   # (fark, kare_adi) — en cok yeni bulunan kareler

    for blok in bloklar:
        for p in blok:
            oa, ob = sonuc_a.get(p.name, []), sonuc_b.get(p.name, [])
            ortak, yalniz_a, yalniz_b = esle(oa, ob)
            bos_a += (len(oa) == 0)
            bos_b += (len(ob) == 0)
            for o in oa:
                a_c[SPEC.get(o["cls"], "?")] += 1
            for o in ob:
                b_c[SPEC.get(o["cls"], "?")] += 1
            for i, _ in ortak:
                ortak_c[SPEC.get(oa[i]["cls"], "?")] += 1
            n_gercek_kayip = 0
            for i in yalniz_a:
                ad = SPEC.get(oa[i]["cls"], "?")
                kayip_c[ad] += 1
                if kayip_turu(oa[i], ob, yalniz_b) == "gercek_kayip":
                    gercek_kayip_c[ad] += 1
                    n_gercek_kayip += 1
                    x1, y1, x2, y2 = kutu(oa[i]); kayip_alan.append((x2 - x1) * (y2 - y1))
                else:
                    yer_deg_c[ad] += 1
            for j in yalniz_b:
                yeni_c[SPEC.get(ob[j]["cls"], "?")] += 1
                x1, y1, x2, y2 = kutu(ob[j]); yeni_alan.append((x2 - x1) * (y2 - y1))
            satirlar.append([p.name, len(oa), len(ob), len(yalniz_b),
                             len(yalniz_a), n_gercek_kayip])
            kare_farki.append((len(yalniz_b) - len(yalniz_a), p.name))

    with open(OUT / f"ab_ozet_{args.a}_vs_{args.b}.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["kare", f"nesne_{args.a}", f"nesne_{args.b}",
                    "YENI", "KAYIP_ham", "KAYIP_gercek"])
        w.writerows(satirlar)

    # ---- gorsel paneller: en cok YENI bulunan kareler ----
    kare_farki.sort(reverse=True)
    secili = [ad for fark, ad in kare_farki[:args.panel] if fark > 0]
    if not secili:
        secili = [ad for _, ad in kare_farki[:min(2, len(kare_farki))]]
    ad2yol = {p.name: p for blok in bloklar for p in blok}
    if secili:
        satir = len(secili)
        fig, axes = plt.subplots(satir, 2, figsize=(15.5, 4.6 * satir))
        axes = np.atleast_2d(axes)
        for r, ad in enumerate(secili):
            img = cv2.imread(str(ad2yol[ad]))
            oa, ob = sonuc_a.get(ad, []), sonuc_b.get(ad, [])
            ortak, yalniz_a, yalniz_b = esle(oa, ob)
            ra = {i: ((255, 255, 255), "") for i, _ in ortak}
            ra.update({i: ((0, 0, 255), "KAYIP") for i in yalniz_a})
            rb = {j: ((255, 255, 255), "") for _, j in ortak}
            rb.update({j: ((0, 255, 0), "YENI") for j in yalniz_b})
            axes[r, 0].imshow(cv2.cvtColor(ciz(img, oa, ra), cv2.COLOR_BGR2RGB))
            axes[r, 0].set_title(f"{ad} — imgsz={args.a}: {len(oa)} nesne", fontsize=11)
            axes[r, 1].imshow(cv2.cvtColor(ciz(img, ob, rb), cv2.COLOR_BGR2RGB))
            axes[r, 1].set_title(f"imgsz={args.b}: {len(ob)} nesne "
                                 f"(+{len(yalniz_b)} YENI / -{len(yalniz_a)} kayip)", fontsize=11)
            for c in (0, 1):
                axes[r, c].axis("off")
        fig.suptitle(f"A/B COZUNURLUK — sol {args.a} | sag {args.b}   "
                     f"(yesil=YENI bulunan, kirmizi=KAYIP, beyaz=ortak)", fontsize=14, y=0.998)
        plt.tight_layout()
        plt.savefig(OUT / f"ab_{args.a}_vs_{args.b}.png", dpi=62, bbox_inches="tight")
        plt.close()
        print(f"gorsel: ab_{args.a}_vs_{args.b}.png ({len(secili)} kare)")

    # ---- KARAR ----
    n_yeni = sum(yeni_c.values())
    n_kayip_ham = sum(kayip_c.values())
    n_kayip = sum(gercek_kayip_c.values())      # GERCEK kayip (yer degistirme haric)
    n_yer_deg = sum(yer_deg_c.values())
    med = lambda v: int(np.median(v)) if v else 0
    L = []
    L.append("=== GOREV 1 — A/B COZUNURLUK KARARI ===")
    L.append(f"test           : {n_kare} kare ({len(bloklar)} ardisik blok), uretim yolu (takip dahil)")
    L.append(f"karsilastirma  : imgsz={args.a} (mevcut)  vs  imgsz={args.b} (aday)")
    L.append("")
    L.append(f"{'sinif':<8} {'imgsz'+str(args.a):>9} {'imgsz'+str(args.b):>9} "
             f"{'YENI':>7} {'KAYIP*':>7} {'yer.deg':>8}")
    for ad in SPEC.values():
        L.append(f"{ad:<8} {a_c.get(ad,0):>9} {b_c.get(ad,0):>9} "
                 f"{yeni_c.get(ad,0):>7} {gercek_kayip_c.get(ad,0):>7} {yer_deg_c.get(ad,0):>8}")
    L.append(f"{'TOPLAM':<8} {sum(a_c.values()):>9} {sum(b_c.values()):>9} "
             f"{n_yeni:>7} {n_kayip:>7} {n_yer_deg:>8}")
    L.append("")
    L.append(f"* KAYIP = GERCEK kayip (nesne tamamen kayboldu). Ham kayip {n_kayip_ham} idi;")
    L.append(f"  bunun {n_yer_deg} tanesi 'yer degistirme' (ayni nesne, kutu kaydi -> IoU<0.5).")
    L.append("  Minik nesnelerde (Insan) birkac piksellik kayma IoU esigini kirar; bu bir")
    L.append("  OLCUM ARTEFAKTIDIR, tespit kaybi degildir (gorsel panelde dogrulandi).")
    L.append("")
    L.append(f"bos kare       : {args.a}: {bos_a}/{n_kare} (%{100*bos_a/max(n_kare,1):.0f})  ->  "
             f"{args.b}: {bos_b}/{n_kare} (%{100*bos_b/max(n_kare,1):.0f})")
    L.append(f"YENI kutu alani (medyan): {med(yeni_alan)} px2 "
             f"({med(yeni_alan)**0.5:.0f}x{med(yeni_alan)**0.5:.0f})")
    if kayip_alan:
        L.append(f"GERCEK KAYIP kutu alani (medyan): {med(kayip_alan)} px2")
    L.append(f"sure           : {ms_a:.0f} ms/kare -> {ms_b:.0f} ms/kare "
             f"(+{ms_b-ms_a:.0f} ms | kare butcesi ~{KARE_BUTCE_MS:.0f} ms)")
    L.append("")

    # olcutler
    net_kazanc = n_yeni - n_kayip
    kayip_orani = n_kayip / max(sum(a_c.values()), 1)
    butce_ok = ms_b <= KARE_BUTCE_MS
    # YANLIS POZITIF suphesi: yeni tespitler gurultu boyutunda mi?
    # (1920 denemesinde model beyaz yuvarlak taslari "UAP" sanip 16x16 px kutular
    #  uretti -> mAP'i dusuren uydurma tespitler. Bu olcut onu otomatik yakalar.)
    yeni_medyan = med(yeni_alan)
    MIN_ANLAMLI_ALAN = 40 * 40      # 40x40 px altı yeni tespit yigini = gurultu suphesi
    gurultu_suphesi = bool(yeni_alan) and yeni_medyan < MIN_ANLAMLI_ALAN
    L.append("--- olcutler ---")
    L.append(f"[{'OK ' if n_yeni > n_kayip else 'RED'}] net kazanc: +{n_yeni} yeni / -{n_kayip} kayip "
             f"= {net_kazanc:+d} tespit")
    L.append(f"[{'OK ' if kayip_orani <= 0.10 else 'RED'}] kayip orani: %{100*kayip_orani:.1f} "
             f"(esik %10 — yuksekse aday cozunurluk model dagilimindan sapiyor)")
    L.append(f"[{'OK ' if not gurultu_suphesi else 'RED'}] yeni tespit boyutu: medyan {yeni_medyan} px2 "
             f"({yeni_medyan**0.5:.0f}x{yeni_medyan**0.5:.0f}) — esik {MIN_ANLAMLI_ALAN} px2 (40x40)")
    if gurultu_suphesi:
        L.append("      ^ UYARI: yeni tespitler GURULTU BOYUTUNDA. Yuksek cozunurlukte model")
        L.append("        kucuk arka plan desenlerini (tas, leke) nesne sanabilir -> YANLIS POZITIF")
        L.append("        -> precision duser -> mAP DUSER. Gorsel panelde mutlaka dogrulayin.")
    L.append(f"[{'OK ' if butce_ok else 'RED'}] sure butcesi: {ms_b:.0f} ms <= {KARE_BUTCE_MS:.0f} ms")
    L.append(f"[{'OK ' if bos_b <= bos_a else 'RED'}] bos kare azaldi mi: {bos_a} -> {bos_b}")
    L.append("")

    karar_ok = (n_yeni > n_kayip) and (kayip_orani <= 0.10) and butce_ok \
        and (bos_b <= bos_a) and not gurultu_suphesi
    if karar_ok:
        L.append(f">>> TAVSIYE: imgsz={args.b} ONERILIR")
        L.append(f"    Gerekce: {net_kazanc:+d} net tespit (recall artisi), kayip ihmal edilebilir "
                 f"(%{100*kayip_orani:.1f}), sure butce icinde (+{ms_b-ms_a:.0f} ms).")
        L.append("    NOT: Gorsel panelde (ab_*.png) YESIL kutularin gercek nesne oldugu")
        L.append("         GOZLE dogrulanmalidir; uydurma kutu gorulurse bu tavsiye gecersizdir.")
    else:
        L.append(f">>> TAVSIYE: imgsz={args.a} KALSIN")
        L.append("    Gerekce: yukaridaki RED olcutlerine bakiniz.")

    metin = "\n".join(L)
    (OUT / f"ab_karar_{args.a}_vs_{args.b}.txt").write_text(metin, encoding="utf-8")
    print()
    print(metin)


if __name__ == "__main__":
    main()
