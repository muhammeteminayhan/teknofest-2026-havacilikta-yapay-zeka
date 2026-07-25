# -*- coding: utf-8 -*-
"""Gorev 1 KALITE DENETIMI — "yarismada gercekten ne gonderdik?"

Puanli oturum logundaki (sistem/_logs/*.log) HER karenin gonderilen `detected_objects`
kutularini, o karenin GERCEK goruntusune (sistem/_images/<oturum>/frame_XXXXXX.webp)
cizer. Boylece gozle denetlenir:
  - kutular dogru nesnenin uzerinde mi (fazladan/uydurma kutu var mi),
  - sinif atamalari (Tasit/Insan/UAP/UAI) dogru mu,
  - bos gelen karelerde gercekten nesne yok mu (kacirma var mi).

Cikti: gorev1/denetim/
    mevcut_<sinif>.png     — her sinif icin temsili kareler (kutular cizili)
    mevcut_bos_kareler.png — 0 nesneli karelerden ornekler (kacirma kontrolu)
    denetim_ozet.csv       — kare basi sayisal ozet
    denetim_ozet.txt       — sinif dagilimi + kutu istatistikleri

Calistir (proje kokunden):
    .venv\\Scripts\\python gorev1\\kalite_denetim.py
    .venv\\Scripts\\python gorev1\\kalite_denetim.py --log <log yolu> --oturum <klasor adi>
"""
import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "sistem" / "_logs"
IMGS = ROOT / "sistem" / "_images"
OUT = Path(__file__).resolve().parent / "denetim"
sys.stdout.reconfigure(encoding="utf-8")

# Sartname sinif kimlikleri (payload'da cls URL'i 1-endeksli: classes/{id+1}/)
SPEC = {0: "Tasit", 1: "Insan", 2: "UAP", 3: "UAI"}
RENK = {0: (0, 200, 255), 1: (0, 255, 0), 2: (255, 80, 80), 3: (255, 0, 255)}  # BGR


def son_log():
    adaylar = sorted(LOGS.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not adaylar:
        sys.exit(f"HATA: log bulunamadi ({LOGS})")
    return adaylar[0]


def log_oku(log_path):
    """Log'u sirayla oku: her `get_current_frame: <image_url>` satirindan sonraki
    `Prediction sent successfully` payload'i O KAREYE aittir. (kare_kaydi listesi)"""
    kayitlar = []
    bekleyen_img = None
    for satir in open(log_path, encoding="utf-8", errors="ignore"):
        m = re.search(r"get_current_frame:\s*(\S+)", satir)
        if m:
            bekleyen_img = m.group(1)
            continue
        i = satir.find('{"frame"')
        if i < 0:
            continue
        try:
            payload = json.loads(satir[i:].strip())
        except json.JSONDecodeError:
            continue
        kayitlar.append({"image_url": bekleyen_img, "payload": payload})
        bekleyen_img = None
    return kayitlar


def cls_id(cls_alan):
    """payload cls -> sartname sinif id (classes/1/ -> 0)."""
    m = re.search(r"/classes/(\d+)/", str(cls_alan))
    return int(m.group(1)) - 1 if m else -1


def kare_yolu(image_url, oturum=None):
    if not image_url:
        return None
    ad = image_url.split("/")[-1]
    klasor = (IMGS / oturum) if oturum else None
    if klasor and (klasor / ad).exists():
        return klasor / ad
    # oturum verilmediyse image_url'in kendi klasorunu dene
    parca = [p for p in image_url.split("/") if p]
    if len(parca) >= 2 and (IMGS / parca[-2] / ad).exists():
        return IMGS / parca[-2] / ad
    for d in IMGS.glob("*/"):
        if (d / ad).exists():
            return d / ad
    return None


def kutu_ciz(img, objs):
    vis = img.copy()
    for o in objs:
        cid = cls_id(o.get("cls"))
        x1, y1 = int(o["top_left_x"]), int(o["top_left_y"])
        x2, y2 = int(o["bottom_right_x"]), int(o["bottom_right_y"])
        renk = RENK.get(cid, (255, 255, 255))
        cv2.rectangle(vis, (x1, y1), (x2, y2), renk, 3)
        etiket = SPEC.get(cid, "?")
        ls, ms = str(o.get("landing_status")), str(o.get("moving_status"))
        if cid == 0:
            etiket += f" hrk={ms}"          # tasit -> hareket durumu
        elif cid in (2, 3):
            etiket += f" inis={ls}"          # UAP/UAI -> inis uygunlugu
        (tw, th), _ = cv2.getTextSize(etiket, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(vis, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), renk, -1)
        cv2.putText(vis, etiket, (x1 + 3, max(th, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    return vis


def panel(kareler, baslik, cikti, oturum, sutun=3):
    """kareler: [(image_url, objs, altyazi)] -> figur"""
    kareler = [k for k in kareler if kare_yolu(k[0], oturum)]
    if not kareler:
        print(f"  (atlandi: {baslik} icin kare yok)")
        return
    n = len(kareler)
    satir = (n + sutun - 1) // sutun
    fig, axes = plt.subplots(satir, sutun, figsize=(7.2 * sutun, 4.2 * satir))
    axes = axes.ravel() if n > 1 else [axes]
    for ax in axes:
        ax.axis("off")
    for ax, (image_url, objs, alt) in zip(axes, kareler):
        img = cv2.imread(str(kare_yolu(image_url, oturum)))
        if img is None:
            continue
        vis = kutu_ciz(img, objs)
        ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        ax.set_title(alt, fontsize=11)
    fig.suptitle(baslik, fontsize=15, y=0.995)
    plt.tight_layout()
    plt.savefig(cikti, dpi=62, bbox_inches="tight")
    plt.close()
    print(f"  kaydedildi: {cikti.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=None, help="log dosyasi (varsayilan: en yeni)")
    ap.add_argument("--oturum", default=None, help="_images altindaki oturum klasoru")
    ap.add_argument("--ornek", type=int, default=6, help="sinif basi ornek kare sayisi")
    args = ap.parse_args()

    log_path = Path(args.log) if args.log else son_log()
    print(f"Log: {log_path.name}")
    kayitlar = log_oku(log_path)
    print(f"Kare kaydi: {len(kayitlar)}")
    if not kayitlar:
        sys.exit("HATA: log'dan kare/payload cikarilamadi.")

    oturum = args.oturum
    if oturum is None:
        ilk = next((k["image_url"] for k in kayitlar if k["image_url"]), None)
        if ilk:
            parca = [p for p in ilk.split("/") if p]
            if len(parca) >= 2:
                oturum = parca[-2]
    print(f"Oturum klasoru: {oturum}")

    OUT.mkdir(parents=True, exist_ok=True)

    # ---- sayisal ozet + sinif basi aday kareler ----
    sinif_c, land_c, mov_c = Counter(), Counter(), Counter()
    alanlar = []
    bos_kareler = []
    sinif_adaylari = {c: [] for c in SPEC}
    satirlar = []

    for i, k in enumerate(kayitlar):
        objs = k["payload"].get("detected_objects", [])
        say = Counter()
        for o in objs:
            cid = cls_id(o.get("cls"))
            sinif_c[SPEC.get(cid, "?")] += 1
            say[cid] += 1
            land_c[str(o.get("landing_status"))] += 1
            mov_c[str(o.get("moving_status"))] += 1
            try:
                alanlar.append((int(o["bottom_right_x"]) - int(o["top_left_x"])) *
                               (int(o["bottom_right_y"]) - int(o["top_left_y"])))
            except (KeyError, ValueError, TypeError):
                pass
        satirlar.append([i, k["image_url"], len(objs)] +
                        [say.get(c, 0) for c in SPEC] +
                        [len(k["payload"].get("reference_predictions", []))])
        if not objs:
            bos_kareler.append((k["image_url"], [], f"kare {i} — 0 nesne"))
        for c in SPEC:
            if say.get(c, 0) > 0:
                sinif_adaylari[c].append((k["image_url"], objs, say[c], i))

    # CSV
    with open(OUT / "denetim_ozet.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["kare", "image_url", "nesne"] + [SPEC[c] for c in SPEC] + ["referans"])
        w.writerows(satirlar)

    n_kare = len(kayitlar)
    n_obj = sum(sinif_c.values())
    n_bos = len(bos_kareler)
    alanlar.sort()
    ozet = [
        "=== GOREV 1 KALITE DENETIMI (sahada gonderilen) ===",
        f"log            : {log_path.name}",
        f"kare           : {n_kare}",
        f"toplam nesne   : {n_obj}  (kare basi {n_obj / max(n_kare,1):.2f})",
        f"bos kare       : {n_bos} (%{100*n_bos/max(n_kare,1):.1f})",
        f"sinif dagilimi : {dict(sinif_c)}",
        f"landing_status : {dict(land_c)}",
        f"moving_status  : {dict(mov_c)}",
    ]
    if alanlar:
        p = lambda q: alanlar[min(len(alanlar) - 1, int(q * len(alanlar)))]
        ozet += [
            f"kutu alani px2 : min {alanlar[0]} | %25 {p(.25)} | medyan {p(.5)} | %75 {p(.75)} | maks {alanlar[-1]}",
            f"kucuk nesne    : {sum(1 for a in alanlar if a < 32*32)} adet <32x32 px "
            f"(%{100*sum(1 for a in alanlar if a < 32*32)/len(alanlar):.1f})",
        ]
    # tutarlilik kontrolu: tasit sayisi == moving 0/1 toplami
    tasit = sinif_c.get("Tasit", 0)
    mov01 = mov_c.get("0", 0) + mov_c.get("1", 0)
    digerleri = n_obj - tasit
    ozet += [
        "",
        "--- sema tutarliligi ---",
        f"Tasit={tasit} vs moving(0/1)={mov01} -> {'TUTARLI' if tasit == mov01 else 'TUTARSIZ!'}",
        f"Tasit-disi={digerleri} vs moving(-1)={mov_c.get('-1',0)} -> "
        f"{'TUTARLI' if digerleri == mov_c.get('-1',0) else 'TUTARSIZ!'}",
    ]
    (OUT / "denetim_ozet.txt").write_text("\n".join(ozet), encoding="utf-8")
    print("\n".join(ozet))

    # ---- gorsel paneller: her sinif icin en cok o siniftan iceren kareler ----
    print("\nPaneller uretiliyor...")
    for c, ad in SPEC.items():
        adaylar = sorted(sinif_adaylari[c], key=lambda t: -t[2])
        # cesitlilik: listeyi tarayarak farkli bolgelerden sec
        secili, adim = [], max(1, len(adaylar) // max(args.ornek, 1))
        for j in range(0, len(adaylar), adim):
            a = adaylar[j]
            secili.append((a[0], a[1], f"kare {a[3]} — {ad}: {a[2]} adet"))
            if len(secili) >= args.ornek:
                break
        panel(secili, f"MEVCUT (imgsz=640) — {ad} tespitleri "
                      f"[toplam {sinif_c.get(ad,0)}]",
              OUT / f"mevcut_{ad}.png", oturum)

    # ---- bos kareler (kacirma kontrolu) ----
    if bos_kareler:
        adim = max(1, len(bos_kareler) // 6)
        panel(bos_kareler[::adim][:6],
              f"MEVCUT (imgsz=640) — 0 NESNELI kareler [{n_bos} adet] "
              f"(gozle: gercekten bos mu, kacirma var mi?)",
              OUT / "mevcut_bos_kareler.png", oturum)

    print(f"\nTAMAM -> {OUT}")


if __name__ == "__main__":
    main()
