# -*- coding: utf-8 -*-
"""FTR raporu icin sonuc grafiklerini uret (video_gorseller ile ayni beyaz tema)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CIKTI = Path(__file__).resolve().parent / "sekiller"
CIKTI.mkdir(exist_ok=True)

# video_gorseller/_stil.py ile ayni palet
ZEMIN, KART = "#ffffff", "#f1f5f9"
KOYU, SOLUK, CIZGI = "#0f172a", "#64748b", "#cbd5e1"
G1, G2, G3 = "#0d9488", "#c2410c", "#7e22ce"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "figure.facecolor": ZEMIN,
    "axes.facecolor": ZEMIN, "axes.edgecolor": CIZGI, "axes.labelcolor": KOYU,
    "text.color": KOYU, "xtick.color": SOLUK, "ytick.color": SOLUK,
    "axes.grid": True, "grid.color": CIZGI, "grid.linewidth": 0.6,
    "grid.alpha": 0.7, "axes.axisbelow": True,
})


def gorev2_kapi():
    """Gorev 2 — kabul kapisi: once/sonra, sinav + yedi bagimsiz yatak.

    YATAY cubuk: rapor iki sutunlu ve sutun genisligi 8,4 cm. Dikey duzende etiketler
    okunmaz kaliyor, ayrica tam genislik gerektirdigi icin Word sekli tek basina bir
    sayfaya atiyordu.
    """
    yatak = ["Sınav · pencere 1", "Sınav · pencere 2", "video1 (RGB)",
             "video2 (termal)", "v25 RGB 4K-A", "v25 RGB 4K-B",
             "v25 termal A", "v25 termal B", "v24"]
    once = [17.26, 5.60, 3.90, 15.05, 3.43, 13.01, 4.18, 11.15, 6.13]
    sonra = [11.46, 5.60, 3.90, 10.35, 3.43, 10.65, 4.18, 8.20, 6.13]

    y = np.arange(len(yatak))[::-1]
    h = 0.36
    fig, ax = plt.subplots(figsize=(5.4, 4.9), dpi=200)
    ax.barh(y + h/2, once, h, label="önce", color=CIZGI, edgecolor=SOLUK, linewidth=0.7)
    ax.barh(y - h/2, sonra, h, label="sonra (z işareti kanıt kapısı)",
            color=G2, edgecolor=G2, linewidth=0.7)
    for yy, a, b in zip(y, once, sonra):
        ax.text(a + 0.3, yy + h/2, f"{a:.2f}".replace(".", ","),
                va="center", fontsize=7, color=SOLUK)
        kazandi = b < a - 0.01
        ax.text(b + 0.3, yy - h/2, f"{b:.2f}".replace(".", ","), va="center",
                fontsize=7, color=KOYU, fontweight="bold" if kazandi else "normal")
    ax.set_yticks(y); ax.set_yticklabels(yatak, fontsize=8)
    ax.set_xlabel("ortalama Öklid hatası [m] — düşük iyi", fontsize=8.5)
    ax.set_xlim(0, 21)
    ax.set_title("Görev 2 — kabul kapısı\n(3 permütasyon, eşleştirilmiş)",
                 fontsize=10.5, fontweight="bold", pad=10)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(CIKTI / "sekil_gorev2_kapi.png", bbox_inches="tight", facecolor=ZEMIN)
    plt.close(fig)


def gorev1_ab():
    """Gorev 1 — sinif bazinda A/B."""
    sinif = ["Taşıt", "İnsan", "UAP", "UAİ"]
    onceki = [286, 20, 24, 7]
    yeni = [289, 54, 12, 19]
    x = np.arange(len(sinif)); w = 0.38
    fig, ax = plt.subplots(figsize=(5.4, 3.5), dpi=200)
    ax.bar(x - w/2, onceki, w, label="önceki ağırlık", color=CIZGI,
           edgecolor=SOLUK, linewidth=0.8)
    ax.bar(x + w/2, yeni, w, label="sevk edilen (trial-8)", color=G1,
           edgecolor=G1, linewidth=0.8)
    for i, (a, b) in enumerate(zip(onceki, yeni)):
        ax.text(i - w/2, a + 4, str(a), ha="center", fontsize=8, color=KOYU)
        ax.text(i + w/2, b + 4, str(b), ha="center", fontsize=8,
                color=KOYU, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(sinif, fontsize=9.5)
    ax.set_ylabel("tespit sayısı (74 gerçek kare)", fontsize=9)
    ax.set_ylim(0, 330)
    ax.set_title("Görev 1 — sınıf bazında A/B", fontsize=11, fontweight="bold", pad=10)
    ax.legend(frameon=False, fontsize=8.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(CIKTI / "sekil_gorev1_ab.png", bbox_inches="tight", facecolor=ZEMIN)
    plt.close(fig)


def yontem_eleme():
    """Klasik on uc adaylari — HEPSI AYNI TURDA, ayni tabloda olculdu.

    DIKKAT: burada yalnizca `degerlendirme/KLASIK_ESLESTIRICI.md` icindeki tek
    tablonun video1 sutunu kullanilir (RAFT tabani 3,77 m). SEA-RAFT ve VGGT
    BASKA turlarda ve VGGT ayrica BASKA bir olcutle (artik/yol yuzdesi) olculdu;
    ayni eksende gostermek birim ve taban karistirmasi oluyordu. Onlar metinde
    ve Tablo 1'de kendi taban/birimleriyle veriliyor.
    """
    ad = ["RAFT + homografi\n(sevk edilen)", "SIFT + homografi", "AKAZE",
          "Lucas-Kanade", "ORB"]
    deger = [3.77, 4.48, 5.46, 6.36, 14.59]
    renk = [G2] + [CIZGI] * 4
    fig, ax = plt.subplots(figsize=(5.4, 3.4), dpi=200)
    y = np.arange(len(ad))[::-1]
    ax.barh(y, deger, 0.6, color=renk, edgecolor=SOLUK, linewidth=0.8)
    for yy, v in zip(y, deger):
        ax.text(v + 0.25, yy, f"{v:.2f}".replace(".", ","), va="center",
                fontsize=8.5, color=KOYU)
    ax.set_yticks(y); ax.set_yticklabels(ad, fontsize=8.5)
    ax.set_xlabel("video1 üzerinde ortalama hata [m] — düşük iyi", fontsize=9)
    ax.set_xlim(0, 17)
    ax.set_title("Aynı turda, aynı boru hattında ölçülen ön uç adayları",
                 fontsize=11, fontweight="bold", pad=10)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(CIKTI / "sekil_yontem_eleme.png", bbox_inches="tight", facecolor=ZEMIN)
    plt.close(fig)


def kapak_seridi():
    """Kapak gorseli — uc gorevin gercek ciktisindan uc kare.

    Kapak sayfasi sablonda bombos kaliyordu; rapor okuyucunun elinde ilk once o
    sayfa aciliyor. Uc panel de MEVCUT sistemin gercek yarisma karelerindeki
    ciktisidir, temsili degildir.
    """
    from PIL import Image

    V = Path(__file__).resolve().parent.parent / "video_gorseller"
    paneller = [
        # (dosya, kirpma kutusu -> (sol, ust, sag, alt) oranlari, baslik)
        (V / "A4_tespit.png", (0.015, 0.100, 0.985, 0.865),
         "Görev 1 — nesne tespiti"),
        # B7'deki yorunge paneli Temmuz kosusunun sapma rakamlarini uzerinde
        # tasidigi icin kapaga konmadi; onun yerine akis eslesmeleri paneli.
        (V / "B3_raft.png", (0.679, 0.158, 0.925, 0.404),
         "Görev 2 — konum kestirimi"),
        (V / "gorev3_paneller" / "gorev3_ref3_kare960.jpg", (0.0, 0.0, 0.62, 0.80),
         "Görev 3 — referans nesne"),
    ]

    fig, eks = plt.subplots(1, 3, figsize=(10.6, 2.9), dpi=200)
    for ax, (yol, kutu, baslik) in zip(eks, paneller):
        with Image.open(yol) as im:
            im = im.convert("RGB")
            g, y = im.size
            sol, ust, sag, alt = kutu
            im = im.crop((int(sol * g), int(ust * y), int(sag * g), int(alt * y)))
            # ortak 4:3 cerceve — panellerin yuksekligi esitlensin
            en, boy = im.size
            hedef = 4 / 3
            if en / boy > hedef:
                yeni_en = int(boy * hedef)
                x0 = (en - yeni_en) // 2
                im = im.crop((x0, 0, x0 + yeni_en, boy))
            else:
                yeni_boy = int(en / hedef)
                y0 = (boy - yeni_boy) // 2
                im = im.crop((0, y0, en, y0 + yeni_boy))
            ax.imshow(im)
        ax.set_title(baslik, fontsize=10, fontweight="bold", pad=6, color=KOYU)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor(CIZGI); s.set_linewidth(1.0)
    fig.tight_layout(pad=0.6, w_pad=1.4)
    fig.savefig(CIKTI / "sekil_kapak.png", bbox_inches="tight", facecolor=ZEMIN)
    plt.close(fig)



def gorev3_yatak():
    """Gorev 3 test yatagi — referans bazinda duyarlilik."""
    ref = [f"ref {i}" for i in range(1, 8)]
    recall = [100, 60, 80, 67, 87, 100, 80]
    ort = 81.9
    x = np.arange(len(ref))
    fig, ax = plt.subplots(figsize=(5.4, 3.2), dpi=200)
    ax.bar(x, recall, 0.62, color=G3, edgecolor=G3, linewidth=0.8)
    for i, v in enumerate(recall):
        ax.text(i, v + 2.5, f"%{v}", ha="center", fontsize=8.5, color=KOYU)
    ax.axhline(ort, color=SOLUK, lw=1.2, ls="--")

    ax.set_xticks(x); ax.set_xticklabels(ref, fontsize=8.5)
    ax.set_ylabel("duyarlılık [%]", fontsize=9)
    ax.set_ylim(0, 118)
    ax.set_title("Görev 3 — referans bazında duyarlılık\n(105 aktif kare, resmî arama pencereleri)",
                 fontsize=10.5, fontweight="bold", pad=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(CIKTI / "sekil_gorev3_yatak.png", bbox_inches="tight", facecolor=ZEMIN)
    plt.close(fig)


def karar_sureci():
    """Bir fikrin uretim koduna girene kadar gectigi uc kapi."""
    fig, ax = plt.subplots(figsize=(5.4, 4.4), dpi=200)
    ax.axis("off")

    def kutu(y, baslik, alt, renk):
        h = 24
        ax.add_patch(plt.Rectangle((4, y), 60, h, facecolor=KART,
                                   edgecolor=renk, lw=1.6))
        ax.text(34, y + h - 7.0, baslik, ha="center", va="center",
                fontsize=9.0, fontweight="bold", color=KOYU)
        ax.text(34, y + 8.0, alt, ha="center", va="center", fontsize=7.3,
                color=SOLUK, linespacing=1.6)

    def ok(y0, y1):
        ax.annotate("", xy=(34, y1), xytext=(34, y0),
                    arrowprops=dict(arrowstyle="-|>", color=SOLUK, lw=1.5))

    def ele(y, metin):
        ax.annotate("", xy=(84, y), xytext=(64, y),
                    arrowprops=dict(arrowstyle="-|>", color=G2, lw=1.4))
        ax.text(85, y, metin, ha="left", va="center", fontsize=7.2,
                color=G2, fontweight="bold", linespacing=1.4)

    ax.text(34, 105, "bir fikir", ha="center", va="center", fontsize=9.5,
            color=KOYU, fontweight="bold")
    ok(101, 95)
    kutu(71, "1 · TAVAN (KÂHİN) ÖLÇÜMÜ",
         "kusursuz bilgiyle beslenirse ne kazandırır?\nüretim kodu yazılmadan · 10–30 dk", G2)
    ele(83, "kazanç\nyoksa")
    ok(71, 65)
    kutu(41, "2 · ÇOK YATAKLI KABUL KAPISI",
         "2024 · 2025 · 2026 — sekiz bağımsız uçuş\neşleştirilmiş permütasyonla ölçüm", G1)
    ele(53, "bir yatak\nkötüleşirse")
    ok(41, 35)
    kutu(11, "3 · GÖRSEL DENETİM",
         "sayı iyi — kutular DOĞRU YERDE mi?\nher aday için panel üretilir, gözle bakılır", G3)
    ele(23, "yanlış\nnesnede")
    ok(11, 5)
    ax.text(34, 0, "SEVK", ha="center", va="center", fontsize=10,
            fontweight="bold", color=KOYU,
            bbox=dict(boxstyle="round,pad=0.45", fc="#dcfce7", ec=G1, lw=1.4))
    ax.set_xlim(0, 112); ax.set_ylim(-6, 110)
    fig.tight_layout(pad=0.2)
    fig.savefig(CIKTI / "sekil_karar_sureci.png", bbox_inches="tight", facecolor=ZEMIN)
    plt.close(fig)


if __name__ == "__main__":
    gorev2_kapi()
    gorev1_ab()
    yontem_eleme()
    kapak_seridi()
    gorev3_yatak()
    karar_sureci()
    for p in sorted(CIKTI.glob("*.png")):
        print(f"  {p.name}")
