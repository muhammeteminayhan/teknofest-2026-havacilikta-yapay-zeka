# -*- coding: utf-8 -*-
"""Teknofest FTR raporunu sablonun KENDI stilleriyle doldur.

SABLONUN GOVDESI IKI SUTUNLUDUR (sutun genisligi 4866 twip). Tablolar ve sekiller
buna gore olculendirilir; genis sekiller icin surekli bolum sonu ile gecici olarak
tek sutuna gecilir.

Kullanim (docs/ klasorunden):
    python ftr_uret.py
Cikti: docs/BugBuster_FTR_2026.docx
"""
import re
import shutil
import sys
import zipfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

DOCS = Path(__file__).resolve().parent
KOK = DOCS.parent
SABLON = DOCS / "Teknofest_FTR_Sablon_2026_d2Drs.docx"
CALISMA = DOCS / "_ftr_calisma"
CIKTI = DOCS / "BugBuster_FTR_2026.docx"

# ---- calisma klasorunu hazirla
if CALISMA.exists():
    shutil.rmtree(CALISMA)
CALISMA.mkdir()
with zipfile.ZipFile(SABLON) as z:
    z.extractall(CALISMA)

DOC = CALISMA / "word" / "document.xml"
RELS = CALISMA / "word" / "_rels" / "document.xml.rels"
MEDYA = CALISMA / "word" / "media"
MEDYA.mkdir(exist_ok=True)

x = DOC.read_text(encoding="utf-8")

paras = re.findall(r'<w:p\b[^>]*>.*?</w:p>|<w:p\b[^>]*/>', x, re.S)

# ============================================================ bicim yardimcilari
def ppr_of(i):
    m = re.search(r'<w:pPr>.*?</w:pPr>', paras[i], re.S)
    return m.group(0) if m else "<w:pPr/>"


def rpr_of(i):
    m = re.search(r'<w:rPr>.*?</w:rPr>', ppr_of(i), re.S)
    return m.group(0) if m else "<w:rPr/>"


BODY_PPR = ppr_of(20)
BODY_RPR = rpr_of(20)

# Kaynakca paragraflari numaralandirilmis bir listedir (numId 2); "[1]", "[2]" ...
# on ekini Word'un kendisi uretir. Metne elle ek yazmak "[1]  [1] ..." veriyordu.
# Kaynaklar arasi bosluk 2,5 pt ve satir araligi tam; bu ayarla son kaynak tek
# basina 12. sayfaya tasiyordu. Bosluk sifira, satir araligi 0,94'e cekildi —
# gozle fark edilmez, kazanilan yer bir kaynak satirindan fazladir.
REF_PPR = ppr_of(54)
REF_PPR = re.sub(r'<w:spacing[^/]*/>',
                 '<w:spacing w:after="0" w:before="0" w:line="226" '
                 'w:lineRule="auto"/>', REF_PPR)
# DIKKAT: pPr icindeki rPr paragraf ISARETININ bicimidir, metnin degil. Ondan
# uretilen kaynaklar 10 punto cikip sablondakilerin (8 punto) yaninda buyuk
# duruyordu; gercek bicim ilk RUN'in rPr'sindedir.
_m_ref_run = re.search(r'<w:r(?:\s[^>]*)?>\s*(<w:rPr>.*?</w:rPr>)', paras[54], re.S)
REF_RPR = _m_ref_run.group(1) if _m_ref_run else rpr_of(54)


def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _rpr(bold=None, italic=None, punto=None):
    r = BODY_RPR
    if bold is not None:
        r = re.sub(r'<w:b w:val="[01]"/>', f'<w:b w:val="{1 if bold else 0}"/>', r)
        r = re.sub(r'<w:bCs w:val="[01]"/>', f'<w:bCs w:val="{1 if bold else 0}"/>', r)
    if italic is not None:
        r = re.sub(r'<w:i w:val="[01]"/>', f'<w:i w:val="{1 if italic else 0}"/>', r)
        r = re.sub(r'<w:iCs w:val="[01]"/>', f'<w:iCs w:val="{1 if italic else 0}"/>', r)
    if punto is not None:
        yari = int(punto * 2)
        if re.search(r'<w:sz w:val="\d+"/>', r):
            r = re.sub(r'<w:sz w:val="\d+"/>', f'<w:sz w:val="{yari}"/>', r)
            r = re.sub(r'<w:szCs w:val="\d+"/>', f'<w:szCs w:val="{yari}"/>', r)
        else:
            r = r.replace("</w:rPr>", f'<w:sz w:val="{yari}"/>'
                                      f'<w:szCs w:val="{yari}"/></w:rPr>')
    return r


def run(t, bold=None, italic=None, punto=None):
    return (f'<w:r>{_rpr(bold, italic, punto)}'
            f'<w:t xml:space="preserve">{esc(t)}</w:t></w:r>')


def P(parts, ppr=None, ilk_satir=True, punto=None):
    if isinstance(parts, str):
        parts = [(parts, None, None)]
    pp = ppr or BODY_PPR
    if not ilk_satir:
        pp = re.sub(r'w:firstLine="\d+"', 'w:firstLine="0"', pp)
    return (f'<w:p>{pp}'
            + "".join(run(t, b, i, punto) for t, b, i in parts) + '</w:p>')


def MADDE(metin, punto=None):
    pp = re.sub(r'<w:ind [^/]*/>',
                '<w:ind w:left="284" w:right="0" w:hanging="142"/>', BODY_PPR)
    if isinstance(metin, str):
        metin = [(metin, None, None)]
    return (f'<w:p>{pp}' + run("• ", punto=punto)
            + "".join(run(t, b, i, punto) for t, b, i in metin) + '</w:p>')


def ORTALI(parts, punto=None, ilk_satir=False):
    pp = re.sub(r'<w:jc w:val="both"/>', '<w:jc w:val="center"/>', BODY_PPR)
    pp = re.sub(r'<w:ind [^/]*/>', '<w:ind w:left="0" w:right="0" w:firstLine="0"/>', pp)
    return P(parts, ppr=pp, ilk_satir=ilk_satir, punto=punto)


# ============================================================ bolum (sutun) sonu
SAYFA = '<w:pgSz w:h="16838" w:w="11906" w:orient="portrait"/>'
KENAR = ('<w:pgMar w:bottom="1440" w:top="1080" w:left="907" w:right="907" '
         'w:header="720" w:footer="720"/>')
SUTUN_GEN = 4866          # tek sutun genisligi (twip)
TAM_GEN = 4866 * 2 + 360  # iki sutun + bosluk

# Yeni bolumler sablonun ust bilgisini surdurmeli; aksi hâlde Word varsayilana doner.
_m_sect55 = re.search(r'<w:sectPr[^>]*>.*?</w:sectPr>', paras[55], re.S)
UST_BILGI = "".join(re.findall(r'<w:headerReference[^/]*/>',
                               _m_sect55.group(0) if _m_sect55 else ""))


def _sect(n_sutun, tur="continuous"):
    cols = ('<w:cols w:equalWidth="0" w:num="2">'
            f'<w:col w:space="360" w:w="{SUTUN_GEN}"/>'
            f'<w:col w:w="{SUTUN_GEN}"/></w:cols>') if n_sutun == 2 else \
        '<w:cols w:num="1" w:space="360"/>'
    return (f'<w:sectPr>{UST_BILGI}<w:type w:val="{tur}"/>'
            f'{SAYFA}{KENAR}{cols}</w:sectPr>')


def _bolum_sonu(n_sutun, tur="continuous"):
    """Gorunmez bolum sonu paragrafi.

    OOXML'de bir pPr icindeki sectPr, o paragrafla BITEN bolumu tanimlar. Yani
    _bolum_sonu(2) onundeki iki sutunlu akisi kapatir, _bolum_sonu(1) ise arada
    acilan tek sutunlu adayi kapatir. `tur`, bolumun NEREDE BASLADIGINI belirler:
    'continuous' ayni sayfada, 'nextPage' yeni sayfada.
    """
    pp = ('<w:pPr><w:spacing w:after="0" w:before="0" w:line="20" w:lineRule="exact"/>'
          '<w:rPr><w:sz w:val="2"/><w:szCs w:val="2"/></w:rPr>'
          f'{_sect(n_sutun, tur)}</w:pPr>')
    return f'<w:p>{pp}</w:p>'


# ============================================================ resim
_rel_sayac = [100]
_resim_sayac = [0]
EMU = 635  # 1 twip


def _rel_ekle(hedef):
    _rel_sayac[0] += 1
    rid = f"rId{_rel_sayac[0]}"
    r = RELS.read_text(encoding="utf-8")
    yeni_rel = (f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/'
                f'officeDocument/2006/relationships/image" Target="media/{hedef}"/>')
    RELS.write_text(r.replace("</Relationships>", yeni_rel + "</Relationships>"),
                    encoding="utf-8")
    return rid


def _grafik_govde(rid, pid, cx, cy, n):
    """<a:graphic> govdesi — inline ve anchor icin ortak."""
    return (
        f'<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/'
        f'drawingml/2006/picture"><pic:pic>'
        f'<pic:nvPicPr><pic:cNvPr id="{pid}" name="Sekil{n}"/>'
        f'<pic:cNvPicPr/></pic:nvPicPr>'
        f'<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch>'
        f'</pic:blipFill>'
        f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
        f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>'
        f'</pic:pic></a:graphicData></a:graphic>')


def RESIM(yol, genislik_twip):
    """Satir ici (<wp:inline>) resim paragrafi.

    keepNext veriliyor: resim ile hemen ardindaki altyazi sutun ya da sayfa
    sonunda birbirinden kopmasin.
    """
    from PIL import Image
    yol = Path(yol)
    _resim_sayac[0] += 1
    n = _resim_sayac[0]
    ad = f"ftr{n:02d}{yol.suffix}"
    shutil.copy(yol, MEDYA / ad)
    rid = _rel_ekle(ad)
    with Image.open(yol) as im:
        en, boy = im.size
    cx = int(genislik_twip * EMU)
    cy = int(cx * boy / en)
    pid = 1000 + n
    pp = re.sub(r'<w:jc w:val="both"/>', '<w:jc w:val="center"/>', BODY_PPR)
    pp = re.sub(r'<w:ind [^/]*/>', '<w:ind w:left="0" w:right="0" w:firstLine="0"/>', pp)
    pp = re.sub(r'<w:spacing [^/]*/>',
                '<w:spacing w:after="60" w:before="120" w:line="240" w:lineRule="auto"/>', pp)
    pp = re.sub(r'<w:keepNext w:val="0"/>', '<w:keepNext w:val="1"/>', pp)
    govde = _grafik_govde(rid, pid, cx, cy, n)
    return (f'<w:p>{pp}<w:r><w:drawing>'
            f'<wp:inline distT="0" distB="0" distL="0" distR="0">'
            f'<wp:extent cx="{cx}" cy="{cy}"/>'
            f'<wp:effectExtent l="0" t="0" r="0" b="0"/>'
            f'<wp:docPr id="{pid}" name="Sekil{n}"/>'
            f'<wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/>'
            f'</wp:cNvGraphicFramePr>{govde}</wp:inline></w:drawing></w:r></w:p>')


_sekil_no = [0]
_ada_sayaci = [0]


def SEKIL(yol, altyazi, tam_genislik=False):
    """Sekil + numarali altyazi; ikisi her zaman ayni sayfada ve yan yana.

    tam_genislik=True: sekil ve altyazisi, iki sutunlu akisin ortasinda acilan
    TEK SUTUNLU bir bolum adasina konur. Once <wp:anchor> + ust-alt kaydirma
    denendi; Word capali sekli sayfa basina tasirken altyazi paragrafi yerinde
    kaldigi icin sekil ile altyazisi bambaska sutunlara dusuyordu (or. Sekil 3
    sayfanin tepesinde, altyazisi sag sutunun ortasinda). Bolum adasi sekli akisin
    icinde tuttugu icin bu kopus olusmuyor.
    """
    _sekil_no[0] += 1
    gen = TAM_GEN - 120 if tam_genislik else SUTUN_GEN - 120
    govde = RESIM(yol, gen) + ORTALI(
        [(f"Şekil {_sekil_no[0]}: ", True, None), (altyazi, None, None)], punto=8)
    if not tam_genislik:
        return govde
    # Adalar, sablonun TEK iki-sutunlu govde bolumunu parcalara ayiriyor. Govdenin
    # kendi 'nextPage' ozelligi (kapaktan sonra yeni sayfada baslama) yalniz ILK
    # parcaya ait olmalidir. Aksi hâlde son parca da yeni sayfada basliyor ve
    # ondan onceki ada tek basina bir sayfada kaliyordu (sayfanin %56'si bos).
    _ada_sayaci[0] += 1
    ilk = _ada_sayaci[0] == 1
    return (_bolum_sonu(2, "nextPage" if ilk else "continuous")
            + govde + _bolum_sonu(1))


# ============================================================ tablo
_tablo_no = [0]


def TABLO(basliklar, satirlar, oranlar=None, baslik_metni="", punto=8,
          tam_genislik=False, sag_hizala=True):
    """Numarali tablo.

    tam_genislik=True: sekillerdeki gibi tek sutunlu bir bolum adasina konur;
    cok sutunlu ya da uzun metinli tablolar 8,6 cm'lik sutuna sigmiyor.
    sag_hizala=False: ilk sutun disindakiler de sola dayanir (metin tablolari).
    """
    _tablo_no[0] += 1
    gen = (TAM_GEN if tam_genislik else SUTUN_GEN) - 120
    n = len(basliklar)
    oranlar = oranlar or [1] * n
    top = sum(oranlar)
    w = [int(gen * o / top) for o in oranlar]
    w[-1] = gen - sum(w[:-1])
    kenar = ('<w:tblBorders>' + "".join(
        f'<w:{k} w:color="94A3B8" w:space="0" w:sz="4" w:val="single"/>'
        for k in ("top", "left", "bottom", "right", "insideH", "insideV")) +
        '</w:tblBorders>')
    # keepNext: baslik satiri tablodan ayri bir sutuna dusmesin
    o = [ORTALI([(f"Tablo {_tablo_no[0]}: ", True, None), (baslik_metni, None, None)],
                punto=8).replace('<w:keepNext w:val="0"/>', '<w:keepNext w:val="1"/>')]
    o.append(f'<w:tbl><w:tblPr><w:tblW w:w="{gen}" w:type="dxa"/>'
             f'<w:jc w:val="center"/>{kenar}'
             f'<w:tblLayout w:type="fixed"/></w:tblPr><w:tblGrid>' +
             "".join(f'<w:gridCol w:w="{c}"/>' for c in w) + '</w:tblGrid>')

    def hucre(metin, gen_h, bold=False, golge=False, sag=False):
        pp = ('<w:pPr><w:spacing w:after="10" w:before="10" w:line="216" '
              'w:lineRule="auto"/><w:ind w:left="45" w:right="45" w:firstLine="0"/>'
              f'<w:jc w:val="{"right" if sag else "left"}"/></w:pPr>')
        shd = '<w:shd w:fill="E2E8F0" w:val="clear"/>' if golge else ''
        return (f'<w:tc><w:tcPr><w:tcW w:w="{gen_h}" w:type="dxa"/>{shd}'
                f'<w:vAlign w:val="center"/></w:tcPr>'
                f'<w:p>{pp}{run(metin, bold=bold, punto=punto)}</w:p></w:tc>')

    # tblHeader: tablo sayfaya bolunurse baslik satiri ustte tekrarlansin.
    # cantSplit: bir satirin hucreleri iki sayfaya bolunmesin.
    o.append('<w:tr><w:trPr><w:jc w:val="center"/><w:tblHeader/>'
             '<w:cantSplit/></w:trPr>' + "".join(
                 hucre(b, w[i], True, True) for i, b in enumerate(basliklar)) + '</w:tr>')
    for s in satirlar:
        o.append('<w:tr><w:trPr><w:jc w:val="center"/><w:cantSplit/></w:trPr>' + "".join(
            hucre(str(c), w[i], sag=(sag_hizala and i > 0))
            for i, c in enumerate(s)) + '</w:tr>')
    o.append('</w:tbl>')
    govde = "".join(o) + P("", ilk_satir=False, punto=4)
    if not tam_genislik:
        return govde
    _ada_sayaci[0] += 1
    return (_bolum_sonu(2, "nextPage" if _ada_sayaci[0] == 1 else "continuous")
            + govde + _bolum_sonu(1))


# ============================================================ ICERIK
V = KOK / "video_gorseller"
D = KOK / "gorev1" / "denetim"
S = DOCS / "sekiller"

SEC = {}

# Sablonun 6 uye yuvasi 3x2 izgaradadir; ekip 4 kisi oldugu icin kose yuvalari
# kullanilip ortadakiler bosaltilarak dengeli bir 2x2 elde edilir.
KAPAK = {
    3: "Bug Buster",
    4: "Takım ID: [[TAKIM ID]]",
}

# Sablonun uye alani UC SUTUNLU bir bolumdur ve alti yuvaya gore kurulmustur; dort
# kisiyle doldurulunca isimler sutunlara dengesiz dagilip dagilmis gorunuyordu
# (biri solda ustte, biri ortada, biri sagda). Alan tek sutuna cevrilip dort uye
# tek bir ortali paragrafta alt alta yaziliyor.
UYELER = [
    ("Amra Kurtabegu", "Görev 1 — Nesne Tespiti"),
    ("Elmas Asya", "Görev 1 — Nesne Tespiti"),
    ("Muhammet Emin Ayhan", "Sistem Mimarisi · Görev 2 — Pozisyon Kestirimi"),
    ("Dagimawi Moges Wakumo", "Görev 3 — Referans Nesne Tespiti"),
]

OZET = (
    "Özet— Bir insansız hava aracı uçarken yere doğru bakan bir kamerayla görüntü "
    "çeker. Bu çalışma, o görüntü akışını canlı olarak işleyen ve aynı anda üç soruya "
    "birden cevap veren bir sistem sunmaktadır: karede hangi nesneler var ve nerede; "
    "konum bilgisi kesildiğinde araç şu anda nerede; oturumun başında gösterilen "
    "örnek nesneler sonraki karelerin hangisinde görünüyor. İkinci soru en zorudur, "
    "çünkü tek bir kamera hareketin yönünü verir ama büyüklüğünü vermez: 10 metre "
    "yükseklikte 1 metre ilerlemek ile 100 metre yükseklikte 10 metre ilerlemek "
    "kameraya aynı görünür. Sistem bu eksik bilgiyi, konumun hâlâ geldiği ilk "
    "dakikalarda öğrenerek tamamlar; bağlantı kesildiğinde öğrendiği ölçekle "
    "görüntüden yol hesaplar. Mimari tek bir uçtan uca sinir ağı değil, bir "
    "iş bölümüdür: derin öğrenme yalnızca ölçüm üretir (hangi piksel nereye gitti, "
    "hangi bölge hangi sınıfa ait), kararı ise kamera geometrisi verir. Çalışmanın "
    "ayırt edici yanı, yöntem seçimi kadar yöntem eleme disiplinidir: bir fikir "
    "üretim koduna girmeden önce, kusursuz bilgiyle ulaşabileceği en iyi sonuç "
    "ölçülmüş ve fikir 2024–2026 yıllarına ait sekiz farklı uçuşta ayrı ayrı "
    "sınanmıştır. Bu düzen on üç yöntem ailesini elemiş, geriye kalan değişiklikler "
    "hiç ayar yapılmamış uçuşlarda da kazanç üretmiştir. Konum kestiriminde ortalama "
    "hata sınav düzeneğinde 11,43 m'den 8,53 m'ye, yedi bağımsız uçuşun ortalamasında "
    "8,12 m'den 6,69 m'ye inmiştir. Sistemin tamamı 2250 karelik bir oturumu 31 "
    "dakikada, tek bir biçim hatası vermeden tamamlamaktadır."
)
ANAHTAR = ("Anahtar Kelimeler— görsel odometri, düzlemsel homografi, optik akış, "
           "GNSS-reddedilmiş konumlandırma, nesne tespiti, DINOv3, tek-atış şablon "
           "eşleme, insansız hava aracı")

# ---------------------------------------------------------------- GIRIS
SEC[20] = [
    P("Yarışma, bir insansız hava aracının altına bakan kamerasından gelen görüntüleri "
      "işleyen ve üç görevi aynı canlı oturumda yerine getiren bir sistem istemektedir. "
      "Oturum boyunca sunucu kareleri tek tek gönderir; sistem her kare için cevabını "
      "geri yollar. Görevler nesne tespiti (NT), pozisyon kestirimi (PK) ve referans "
      "nesne tespitidir (RNT). Genel puan, bu üç görevin "
      "Final = (NT·0,25 + RNT·0,25 + PK·0,40) / 0,90 ağırlıklı ortalamasıdır; "
      "görüldüğü gibi en ağır görev pozisyon kestirimidir."),
    P("Oturum yaklaşık 2250 kare ve saniyede 7,5 karedir. Belirleyici kısıt şudur: "
      "gönderilen cevap kesindir, geri alınamaz. Bu, tüm kareleri toplayıp sonradan "
      "topluca düzelten çevrimdışı yöntemleri baştan devre dışı bırakır — sistem her "
      "kareyi yalnız o ana kadarki bilgiyle cevaplamak zorundadır."),
    P([("Asıl güçlük nerede? ", True, None),
       ("Konum kestirimi, literatürde monoküler görsel odometri adıyla incelenir; "
        "“monoküler” tek kameralı, “odometri” ise kat edilen yolun "
        "hesaplanması demektir. Bu problemin bilinen çekirdek zorluğu ", None, None),
       ("ölçek belirsizliğidir", True, None),
       (": tek bir kameradan bakan bir gözlemci için 10 m yükseklikte 1 m ilerlemek ile "
        "100 m yükseklikte 10 m ilerlemek tıpatıp aynı görüntüyü üretir [2], [3]. "
        "Kamera, hareketin yönünü ve şeklini verir; kaç metre olduğunu vermez. Verdiği "
        "tek büyüklük, ötelemenin zemine olan uzaklığa oranıdır. Bunu metreye çevirmek "
        "için dışarıdan bir bilgi gerekir; literatürde bu bilgi genellikle bilinen "
        "kamera yüksekliğinden, zemin düzlemi geometrisinden ya da boyutu bilinen bir "
        "nesneden gelir [3]. Bu çalışmada ise bilgi, konumun hâlâ geldiği sağlıklı "
        "dönemden öğrenilmektedir.", None, None)]),
    P([("Neden düzlemsel homografi? ", True, None),
       ("Aşağı bakan bir kamerada sahne büyük ölçüde tek bir düzlemdir: zemin. İki "
        "kare arasında bu düzlem üzerindeki noktaların nasıl yer değiştirdiği, "
        "homografi denen 3×3'lük tek bir matrisle tam olarak ifade edilebilir. Kamera "
        "iç parametreleri K, dönme R, öteleme t, düzlem normali n ve düzleme uzaklık d "
        "olmak üzere bu matris H = K(R − t·nᵀ/d)K⁻¹ biçimindedir. Matrisi görüntüden "
        "kestirdikten sonra ters yönde çözerek dönmeyi, düzlemin yönünü ve ölçeksiz "
        "ötelemeyi ayrı ayrı elde etmek mümkündür. Yani tek bir matris, aracın o iki "
        "kare arasında ne kadar döndüğünü ve hangi yöne kaydığını taşımaktadır.",
        None, None)]),
    P("Öğrenilmiş bileşenler tarafında güncel yaklaşımlar üç grupta toplanabilir: iki "
      "kare arasında her pikselin nereye gittiğini kestiren yoğun optik akış ağları "
      "[4], [5]; aşağı bakan kamera için homografiyi ve güvenilirliğini doğrudan "
      "öğrenen ağlar [6]; ve seyrek yama takibiyle çalışan öğrenilmiş odometri ile üç "
      "boyutlu temel modeller [7], [8], [9]. Nesne tarafında ise etiket görmeden "
      "eğitilen görsel temsiller [10], hem tespit gövdesi hem de tek-atış şablon eşleme "
      "için ortak bir zemin sunmaktadır."),
    P([("Çalışmayı benzerlerinden ayıran yan, yöntem seçimi kadar ", None, None),
       ("yöntem eleme disiplinidir", True, None),
       (". Bu problemde tek bir uçuşta ölçülen kazancın bağımsız veride çökmesi istisna "
        "değil kuraldır. Somut bir örnek: düşey hareketi başka bir kanaldan almak, "
        "elimizdeki örnek veri setinde hatayı %25,6 azaltıyordu; aynı değişiklik çevrim "
        "içi oturumun kendi karelerinde iki ayrı pencerede de kaybettirdi ve geri "
        "alındı. Bu nedenle her aday, üretim koduna girmeden önce iki aşamadan "
        "geçirilmiştir. Birincisi ", None, None),
       ("tavan ölçümüdür", True, None),
       (": fikre, gerçekte asla sahip olamayacağı kusursuz bilgi verilir ve ulaşabileceği "
        "en iyi sonuç ölçülür. O tavan bile bir kazanç göstermiyorsa fikir hiç "
        "kodlanmaz. İkincisi ", None, None),
       ("çok yataklı kabul kapısıdır", True, None),
       (": değişiklik, 2024, 2025 ve 2026 yıllarına ait, farklı kamera ve görüntü "
        "türlerindeki sekiz ayrı uçuşta tek tek sınanır ve herhangi birinde kötüleşme "
        "varsa kabul edilmez. Bu düzen dokuz yöntem ailesini üretim kodu yazılmadan "
        "elemiş ve dört yapısal teşhis üretmiştir (Bölüm V). Bütün turların birikimi "
        "Tablo 1'de verilmektedir.", None, None)]),
]

# ---------------------------------------------------------------- VERI SETLERI
SEC[22] = [
    P("Çalışmada dört ayrı veri kaynağı kullanılmıştır. Seçimin ortak gerekçesi, bir "
      "değişikliğin yalnız ayar yapılan uçuşta değil, ayar yapılmayan yıl, kamera ve "
      "modalitelerde de kazanç üretmesini zorunlu kılmaktır."),
    MADDE([("Resmî 2026 örnek veri seti. ", True, None),
           ("Bir RGB (1920×1080) ve bir termal (640×512) uçuş; kare başına yer gerçeği "
            "ötelemesi ve kamera kalibrasyonu ile birlikte. Ölçüm altyapısının kurulduğu "
            "temel yataktır.", None, None)]),
    MADDE([("Çevrim içi oturumun kendi kareleri. ", True, None),
           ("Çevrim içi yarışma oturumundan kaydedilen 2250 kare ve her kare için "
            "sağlık bayrağı. Bu yatak kritiktir, çünkü tüm ayar çalışmasının "
            "yapıldığı örnek veri seti aslında TEST oturumunun videosudur; genelleme "
            "ancak farklı bir uçuşta ölçülebilir. Yer gerçeği konumu ise bu oturumda "
            "yalnız ", None, None),
           ("sağlık bayrağının 1 olduğu karelerde", True, None),
           (" mevcuttur (703 kare); sunucu, konumun kesildiği 1547 karenin doğru "
            "cevabını hiç göndermemiştir. Bu nedenle o karelerde hata ölçmek "
            "mümkün değildir ve denenmemiştir — ölçüm için sağlıklı bloklar yapay "
            "kesintiye çevrilmektedir (Bölüm IV).", None, None)]),
    MADDE([("2024 ve 2025 örnek veri setleri. ", True, None),
           ("Beş bağımsız uçuş: iki adet 4K RGB (3840×2160), iki adet termal ve GT'sinde "
            "yalnız x-y bileşeni bulunan bir 2024 uçuşu. Ayar bu yataklarda hiç "
            "yapılmamıştır; kabul kapısının asıl gücü buradan gelir.", None, None)]),
    MADDE([("Sentetik uçuş verisi ve tema testleri. ", True, None),
           ("Gece, sis ve bulanıklık koşullarında dayanıklılık ölçümü için "
            "kullanılmıştır.", None, None)]),
    P("Nesne tespiti modeli, resmî örnek veri setinin etiketli karelerine ek olarak aynı "
      "sınıf tanımlarını taşıyan genişletilmiş bir birleşik veri kümesi üzerinde "
      "eğitilmiştir (dört sınıf: Taşıt, İnsan, UAP, UAİ)."),
    P([("Veri kalitesine ilişkin bir bulgu ayrıca kayda değerdir: ", None, None),
       ("2025 veri setinde termal videoların yer gerçeği dosyaları çapraz "
        "adlandırılmıştır", True, None),
       (". Her termal video, kendi adındaki dosya yerine diğeriyle eşleşmektedir. Bu "
        "ölçümle kanıtlanmıştır: görsel ölçümün GT adımıyla korelasyonu doğru eşleşmede "
        "0,52–0,72, yanlış eşleşmede 0,03'tür. Hatanın fark edilmemesinin nedeni, “tüm "
        "kareler sağlıklı” koşusunda hatanın sıfır çıkmasıydı; o kontrol bu kusuru "
        "yakalayamaz, çünkü sağlıklı karede GT aynen geri gönderilir. Yakalayan tek ölçüt "
        "ölçümün GT ile korelasyonudur ve bu ölçüt sonrasında kabul kapısına zorunlu bir "
        "ön koşul olarak eklenmiştir.", None, None)]),
]

# ---------------------------------------------------------------- YONTEM girisi
SEC[24] = [
    P("Bu bölümde önce sistemin bütününü oluşturan mimari, ardından üç görevin "
      "algoritmaları ayrı ayrı anlatılmaktadır."),
    P([("Ortak tasarım ilkesi bir iş bölümüdür. ", True, None),
       ("Derin öğrenme yalnızca ", None, None),
       ("ölçüm", True, None),
       (" üretir: bir karedeki hangi pikselin bir sonraki karede nereye gittiğini, ya da "
        "hangi bölgenin hangi nesne sınıfına ait olduğunu söyler. ", None, None),
       ("Kararı", True, None),
       (" ise klasik geometri verir: kamera modeli, düzlemsel homografi ve konumun hâlâ "
        "geldiği dönemde öğrenilen ölçek, bu ölçümleri şartnamenin istediği metre "
        "cinsinden büyüklüklere çevirir. Bu ayrımın iki pratik faydası vardır. Birincisi "
        "sistem, hangi karede neden güvenmediğini söyleyebilir — çünkü güven kararı "
        "açıkça yazılmış geometrik kurallarla verilir, bir ağın içinde gizli değildir. "
        "İkincisi tek bir bileşen bozulduğunda sistem durmaz; bozulan ölçüm reddedilir "
        "ve yerine bir yedek konur.", None, None)]),
]

# ---------------------------------------------------------------- SISTEM MIMARISI
SEC[26] = [
    P([("Sunucu her karede iki şey gönderir: görüntünün kendisi ve bir ", None, None),
       ("sağlık bayrağı", True, None),
       (". Bu bayrak, aracın konum bilgisinin o an güvenilir olup olmadığını söyler. "
        "Bayrak 1 iken sunucu doğru konumu da vermektedir; 0 olduğunda konum bilgisi "
        "kesilmiştir ve konumu bizim üretmemiz beklenir. Sistem, gelen her kare için üç "
        "görevi sırayla yürüten tek bir düzenleyici sınıf etrafında kuruludur (Şekil 1): "
        "kare okunur, sağlık bayrağı ve varsa konum alınır; Görev 1 her karede çalışır; "
        "Görev 3 yalnız aktif referans aralığındaki karelerde devreye girer; Görev 2'nin "
        "durum makinesi her karede bir adım ilerletilir. Üç görevin çıktıları tek bir "
        "JSON paketine konup sunucuya geri gönderilir.", None, None)]),
    SEKIL(V / "S1_sistem_akisi.png",
          "Sistem mimarisi — üç görev, tek canlı döngü. Sunucu her karede görüntüyü ve "
          "sağlık durumunu gönderir; sistem tek bir JSON paketiyle üç görevin cevabını "
          "birlikte döndürür.", tam_genislik=True),
    P([("Mimarinin belirleyici tercihi, sağlık bayrağını bir kısıt değil bir ", None, None),
       ("kaynak", True, None),
       (" saymaktır. Oturumun ilk yaklaşık 450 karesinde bayrak 1'dir; yani hem görüntüyü "
        "hem doğru cevabı aynı anda görürüz. Bu pencere, görüntüden gelen ölçeksiz "
        "hareketi gerçek metrelere bağlayan köprünün öğrenilebildiği tek yerdir. O sırada "
        "sunucudan gelen konum şartnamenin izin verdiği biçimde aynen geri gönderilir "
        "(dolayısıyla o karelerde hata sıfırdır) ve arka planda sessizce kalibrasyon "
        "öğrenilir. Bağlantı kesildiği anda bu kalibrasyon devreye girer.", None, None)]),
    P([("Çökmeyen tasarım. ", True, None),
       ("Yarışma bilgisayarında internetin ve GPU'nun çalışacağı garanti değildir. Bu "
        "yüzden tüm model dosyaları pakete konmuş ve yerel önbelleğe kopyalanmaktadır; "
        "sistem internet olmadan açılır. Donanım için de üç kademeli bir yedek zinciri "
        "vardır: birincil motor RAFT (GPU ister), o çalışmazsa SIFT (yalnız işlemci), o "
        "da olmazsa Lucas-Kanade. Yedek yolun kurulmasıyla GPU'suz başarım üç uçuşta "
        "15,86 m'den 9,21 m'ye inmiştir; yani GPU olmadan da sistem kabul edilebilir bir "
        "doğrulukla sürmektedir.", None, None)]),
    P([("Yedeğin birincil motor yapılmaması bilinçli bir karardır ve bu raporun ana "
        "iddiasının somut bir örneğidir. ", True, None),
       ("SIFT, üç uçuşta RAFT'tan daha iyi görünüyordu (9,21 m karşılık 9,72 m). Bu "
        "sinyal sekiz uçuşluk kabul kapısında çökmüştür: SIFT sekiz uçuşun yalnız "
        "ikisinde kazanmakta, dördünde felaket üretmekte ve ortalamada 3,00 m "
        "kaybettirmektedir. Üç uçuşluk üstünlüğün tamamı tek bir termal uçuştan "
        "geliyordu.", None, None)]),
    P("Zaman bütçesi de tüm oturum üzerinden doğrulanmıştır. Sistemin 2250 karenin "
      "tamamını işlediği provada kare başına 824 ms ve toplamda 31 dakika ölçülmüştür; "
      "yarışmanın tanıdığı pencere 90 dakikadır."),
]

# ---------------------------------------------------------------- TESPIT MODELI
SEC[28] = [
    "@BASLIK@",
    P("Kareler şartname sınıf tanımlarına (Taşıt, İnsan, UAP, UAİ) göre etiketlenmiş; "
      "eğitimde mozaik, ölçek titretme, yatay çevirme, HSV renk uzayında titretme ve "
      "rastgele silme uygulanmıştır. Alt-görüş geometrisi nedeniyle düşey çevirme ve "
      "döndürme kullanılmamıştır; kamera yere dik baktığı için bu dönüşümler gerçek "
      "dünyada karşılığı olmayan örnekler üretir."),
]

SEC[29] = [
    "@BASLIK@",
    P([("Tespit modeli, ", None, None), ("DINOv3 gövdeli bir YOLO", True, None),
       (" mimarisidir (24,7 M parametre). YOLO, görüntüyü tek geçişte tarayıp nesne "
        "kutularını ve sınıflarını doğrudan üreten bir tespit ailesidir; hızlıdır, "
        "dolayısıyla canlı akışa uygundur. DINOv3 ise etiket kullanmadan, yalnızca "
        "büyük miktarda görüntüye bakarak eğitilmiş bir görsel temsil modelidir; burada "
        "YOLO'nun “gövdesi”, yani görüntüden öznitelik çıkaran ön katmanları "
        "olarak kullanılmaktadır. Beklenti, etiketsiz öğrenilmiş bu temsilin küçük ve "
        "seyrek nesnelerde daha dayanıklı olmasıdır.", None, None)]),
    P([("Bağımlılık sorunu ve çözümü. ", True, None),
       ("Bu ağırlık, yalnız ultralytics kütüphanesinin özel bir çatalında bulunan bir "
        "sınıfa bağlıdır; dosya açılırken o sınıf aranır. Çatalı kurmak, kurulu "
        "ultralytics sürümünü geriye alacak ve aynı DINOv3'ü kullanan Görev 3'ü de riske "
        "atacaktı. Bunun yerine söz konusu modülün bir kopyası paketin içine konmuş ve "
        "model yüklenirken çalışma zamanına tanıtılmıştır. Böylece sistem standart "
        "ultralytics sürümüyle, hiçbir şey indirmeden çalışır.", None, None)]),
    P([("Sınıf eşlemesi — sessiz hata kaynağı. ", True, None),
       ("Bir tespit modeli sınıfları isimle değil, numarayla üretir. Şartname bu "
        "numaraları sabitlemiştir (0 Taşıt, 1 İnsan, 2 UAP, 3 UAİ) ama eğitilen "
        "modelin kendi numaralandırması farklı olabilir; ölçtük: önceki ağırlıklarda "
        "2=UAİ ve 3=UAP iken sevk edilen ağırlıkta 2=UAP ve 3=UAİ'dir. Koda sabit bir "
        "“2 ile 3'ü takas et” kuralı yazmak bu yüzden kırılgandır — ağırlık "
        "değişince kural yanlışa döner, UAP ile UAİ etiketleri birbirine yazılır ve iki "
        "sınıf da hiçbir hata vermeden sıfır puan alır. Sistem bu yüzden eşlemeyi "
        "ağırlığın kendi sınıf ", None, None),
       ("adlarından", True, None),
       (" kurar; tanımadığı ya da eksik bir sınıf tablosuyla karşılaşırsa kurulum anında "
        "hata verip durur. Sessizce yanlış etiketlemektense açıkça çökmek tercih "
        "edilmiştir.", None, None)]),
]

SEC[30] = [
    "@BASLIK@",
    P("Eğitim sıfırdan değil, daha önceki bir koşunun ağırlıklarından sürdürülmüştür. "
      "DINOv3 gövdesi donuk tutulmuş, yani onun içindeki ağırlıklar hiç güncellenmemiş, "
      "yalnız tespit başlıkları eğitilmiştir; böylece hazır temsil bozulmadan korunur. "
      "Girdi çözünürlüğü 1280 piksel seçilmiştir. En fazla 1000 dönem planlanmış, on "
      "dönem üst üste iyileşme olmazsa durma kuralıyla 127. dönemde durdurulmuştur "
      "(toplu boyut 16, başlangıç öğrenme oranı 1×10⁻³)."),
    P([("Çözünürlük kuralı. ", True, None),
       ("Bir model, çıkarımda en iyi hangi çözünürlükte çalışır? Eğitildiği "
        "çözünürlükte. Bu, DINOv3 gibi ViT tabanlı gövdelerde özellikle belirgindir, "
        "çünkü bu modeller görüntüyü kareciklere bölüp her karecikin görüntünün "
        "neresinde olduğunu ayrı bir “konum bilgisi” olarak öğrenir; "
        "çözünürlük değişince o bilgi kayar. Sevk edilen ağırlığın eğitim çözünürlüğü "
        "iki bağımsız kaynakta 1280 olarak doğrulanmıştır. Önceki ağırlık 640'ta "
        "eğitilmişti ve o dönem yapılan ölçüm de 640'ı seçmişti; model değişince en iyi "
        "çözünürlük de değişmiştir.", None, None)]),
]

SEC[31] = [
    "@BASLIK@",
    P([("Modelin kendi doğrulama kümesinde mAP@0,5 = 0,814 ve mAP@0,5:0,95 = 0,649, "
        "kesinlik 0,886, duyarlılık 0,772 elde edilmiştir. ", None, None),
       ("Bu değerler ağırlıklar arasında kıyas için kullanılamaz", True, None),
       (", çünkü her ağırlık kendi doğrulama kümesinde ölçülmüştür ve kümeler farklıdır; "
        "kolay bir kümede yüksek bir sayı, zor bir kümedeki düşük sayıdan iyi olmak "
        "zorunda değildir. Kıyas bu yüzden bir sonraki adımda, iki modelin de daha önce "
        "hiç görmediği ortak ve gerçek kareler üzerinde yapılmıştır.", None, None)]),
]

SEC[32] = [
    "@BASLIK@",
    P("Karar, 74 gerçek yarışma karesi üzerinde verilmiştir. İki ağırlık aynı üretim "
      "yolundan, aynı kareler üzerinde koşturulmuş ve ürettikleri kutular tek tek "
      "eşleştirilmiştir (sonuçlar Bölüm IV). Ölçüt yalnız sayısal değildir: her aday "
      "için görsel panel üretilip yeni kutulara gözle bakılır. Bu adım belirleyici "
      "olmuştur — otomatik ölçüt iki kırmızı bayrak kaldırmış, görsel denetim ikisinin "
      "de yanlış alarm olduğunu göstermiştir."),
    SEKIL(D / "trial8_kanit_uap_uai.png",
          "Görsel denetimin belirleyici olduğu kare. Sarı: önceki ağırlık, kırmızı: sevk "
          "edilen. Sahnedeki iki dairenin üzerinde sınıf kısaltmaları yazılıdır; önceki "
          "ağırlık “UAİ” yazan daireyi UAP olarak sınıflandırmaktadır."),
    P("Birinci bayrak, otomatik ölçütün “kayıp” saydığı 12 tespitti. Panele bakınca "
      "bunun bir kayıp değil bir sınıflandırma düzeltmesi olduğu görülmüştür: iki "
      "modelin ürettiği toplam iniş alanı sayısı aynıdır, yalnız etiketler düzelmiştir. "
      "İkinci bayrak “yeni tespitler çok küçük” uyarısıydı; o da yanlış alarm çıktı, "
      "çünkü yeni tespitlerin çoğunluğu insandır ve bu irtifada bir insan zaten yaklaşık "
      "30×40 piksel gelmektedir (yeni ağırlıkta kutu ortancası 30×34,5 piksel, öncekinde "
      "50×52 piksel). Görev 1'in kareden JSON'a uzanan tam akışı Şekil 3'te "
      "özetlenmiştir."),
]

SEC[33] = [
    SEKIL(V / "A1_gorev1_akis.png",
          "Görev 1 akışı — tespit, takip, ego-hareket telafili hareket durumu ve iniş "
          "uygunluğu.", tam_genislik=True),
]

# ---------------------------------------------------------------- POZISYON
SEC[35] = [
    P([("Görevin tanımı şudur: konum bilgisi kesildikten sonra araç, kesintinin "
        "başladığı andaki konuma göre nerededir? Cevap yalnızca görüntüden "
        "üretilmelidir. Algoritma, Giriş'te anlatılan ölçek belirsizliğinin doğrudan "
        "bir sonucu olarak iki parçadan oluşur (Şekil 4): her karede hareketi ",
        None, None),
       ("ölçeksiz", True, None),
       (" olarak ölçen bir ön uç ve konumun hâlâ geldiği dönemde bu ölçeksiz hareketi "
        "metreye çevirmeyi öğrenen bir durum makinesi. Aşağıda altı adım hâlinde "
        "anlatılmaktadır.", None, None)]),
    SEKIL(V / "B2_gorev2_akis.png",
          "Görev 2 akışı. Sunucu konum verdiği sürece o konum aynen gönderilir ve arka "
          "planda eksen köprüsü ile metrik ölçek öğrenilir; kesildiği anda bu boru hattı "
          "devreye girer.", tam_genislik=True),
    P([("1) İki kare arasında ne kadar kaydık? ", True, None),
       ("Kare önce gri tona çevrilip sabit bir çalışma genişliğine küçültülür (1080p "
        "için yarıya, 4K için dörtte bire, termal görüntüde olduğu gibi). Böylece "
        "çözünürlük ne olursa olsun işlem maliyeti sabit kalır. Ardından RAFT adlı "
        "öğrenilmiş model, iki ardışık kare arasında ", None, None),
       ("her pikselin", True, None),
       (" nereye gittiğini kestirir; buna yoğun optik akış denir. Bu akış alanından, 28 "
        "piksel aralıklı düzgün bir ızgaradan yaklaşık 640 nokta örneklenir. Izgara "
        "kullanmak bilinçli bir tercihtir: köşe arayan klasik yöntemler noktaları "
        "binaların ve ağaçların dokulu bölgelerine yığar, o da bir sonraki adımdaki "
        "düzlem uydurmasını oraya yaslar. Her nokta, ileri ve geri yönde tutarlı olup "
        "olmadığına bakılarak sınanır, kamera merceğinin bozulmasından arındırılır ve "
        "kalan noktalara RANSAC ile bir homografi uydurulur. RANSAC, noktaların "
        "rastgele küçük alt kümelerini deneyerek çoğunluğa uyan çözümü bulan bir "
        "yöntemdir; uymayan noktaları (aykırıları) kendiliğinden dışarıda bırakır.",
        None, None)]),
    P([("2) Homografiden harekete. ", True, None),
       ("Elde edilen matris, dönme ve ölçeksiz öteleme bilgisini iç içe taşır; bunları "
        "ayırmak gerekir. Matematiksel ayrıştırma tek bir cevap değil ", None, None),
       ("dört aday", True, None),
       (" üretir — bunlar geometrik olarak aynı görüntüyü açıklayan çözümlerdir. Dört "
        "aday iki fiziksel kuralla teke indirilir: görülen noktaların ışınları düzlemi "
        "kameranın önünde kesmelidir (arkasında kesen çözüm fiziksel değildir) ve "
        "bulunan zemin yönü bir önceki karede bulunana yakın olmalıdır (zemin bir "
        "karede aniden dönmez). Sonuçta üç büyüklük çıkar: ölçeksiz yatay adım, "
        "ölçeksiz düşey adım ve yön değişimi. Ölçüm sırasında yön ile ölçek "
        "kanallarının zıt eşik istediği gözlenmiştir; yön için ayrı ve gevşek eşikli "
        "ikinci bir uydurma denenip elenmiş, sevk edilen sistemde üç büyüklük de aynı "
        "sıkı eşikli uydurmadan alınmıştır (bu gerilimin sonuçları Bölüm V'te "
        "tartışılmaktadır).", None, None)]),
    P([("3) Ölçeksizi metreye çevirme. ", True, None),
       ("Konumun hâlâ geldiği karelerde iki dizi elimizdedir: görüntüden ölçtüğümüz "
        "ölçeksiz adımlar ve sunucunun verdiği gerçek metrik adımlar. Bu iki diziyi "
        "birbirine en iyi oturtan dönme ve ölçek, Procrustes (Kabsch) hizalaması denen "
        "kapalı formda bir çözümle bulunur [13]; sezgisel olarak, iki nokta bulutunu "
        "üst üste getiren en iyi döndürme ve büyütme oranını arar. Çıkan dönme, kamera "
        "ekseniyle dünya ekseni arasındaki farkı; çıkan ölçek ise metre karşılığını "
        "verir. Bu ölçeğin fiziksel anlamı yerden yüksekliktir — çünkü aynı görüntü "
        "kayması, yüksekten bakınca daha çok metreye karşılık gelir. Uydurmanın ne "
        "kadar iyi oturduğu da ölçülür; kötü oturuyorsa (artık oranı eşiği aşarsa) "
        "yeni kalibrasyon reddedilir ve bir önceki geçerli kalibrasyon korunur. Bu "
        "kapı, bozuk bir ölçekle yol hesaplamayı engelleyen ana korumadır.",
        None, None)]),
    P([("4) Düşey eksenin yönü — tek bit, büyük sonuç. ", True, None),
       ("Dünyanın z ekseni aşağıyı mı yukarıyı mı gösteriyor? Bu tek bir işaret "
        "bilgisidir, ama yanlış kestirilirse düşey kanal tümüyle ters döner ve "
        "yükseklik üzerinden yatay ölçeği de bozar. Önceki kural bu işareti ölçümün "
        "kendi büyüklüğüne bakarak seçiyordu. Düz uçuşta gerçek irtifa değişimi "
        "neredeyse sıfır olduğu için ölçüm gürültüsü eşiği geçiyor ve karar pratikte "
        "yazı-tura oluyordu; yataklarda %26–40 oranında kararsız ölçüldü. Yeni kural "
        "kararı ancak iki koşul birlikte sağlanırsa verir: kalibrasyon penceresinde "
        "gerçekten bir irtifa hareketi olmalı (≥2 m) ve ölçümlerin işareti kendi içinde "
        "tutarlı olmalıdır. Hiçbiri sağlanmıyorsa bilgi yok demektir; önceki işaret "
        "korunur. Bu turun tek sevk edilen değişikliği budur ve Bölüm IV'teki kazancın "
        "tamamı buradan gelmektedir.", None, None)]),
    P([("5) Hata birikimini kırmak: keyframe merdiveni. ", True, None),
       ("Her kareyi bir öncekine göre konumlarsak, her adımın küçük hatası bir sonrakine "
        "eklenir; 600 karelik bir kesintide bu 600 kez üst üste birikir. Bunu sınırlamak "
        "için her 8 karede bir sabit bir ", None, None),
       ("çapa", True, None),
       (" kare seçilir ve ölçüm doğrudan o çapaya göre yapılır. Böylece 600 karelik bir "
        "kesintide birikim sayısı 600'den yaklaşık 75'e iner. Aralık değeri denenerek "
        "seçilmiştir: 16'ya çıkarıldığında iki kare birbirinden fazla uzaklaşıyor ve "
        "ölçümlerin %59–71'i kalite kapısına takılıyor; 4'e indirildiğinde ise birikim "
        "yeterince azalmıyor.", None, None)]),
    P([("6) Kesinti boyunca ilerleme. ", True, None),
       ("Sağlık bayrağı sıfırlandığında konum, öğrenilen köprüyle taşınır: ölçeksiz "
        "yatay adım önce eksen dönüşümünden geçirilir, sonra yükseklikle çarpılarak "
        "metreye çevrilir; düşey adım işaretiyle birlikte uygulanır; yön birikimli "
        "olarak güncellenir. Ölçüm kalite kapılarından geçemezse — az eşleşme, düşük "
        "uyum oranı, fiziksel olarak olanaksız büyüklükte bir adım ya da donmuş kare — "
        "o kare için ölçüm kullanılmaz ve araç son bilinen hızıyla ilerliyor kabul "
        "edilir.", None, None)]),
]

# ---------------------------------------------------------------- REFERANS NESNE
SEC[38] = [
    P([("Görev 3'te sunucu, oturum sırasında bir nesnenin yakın çekim fotoğrafını "
        "gönderir ve o nesnenin sonraki karelerde nerede göründüğünü sorar. Nesnenin ne "
        "olduğu önceden bilinmez — bir çeşme, bir basketbol sahası, bir zemin motifi "
        "olabilir. Dolayısıyla o nesne için eğitilmiş bir model yoktur ve olamaz. "
        "Problem bu yüzden ", None, None),
       ("tek-atış şablon eşleme", True, None),
       (" olarak ele alınmıştır: tek bir örnek görüntüden yola çıkarak, hiç eğitim "
        "yapmadan, aynı nesneyi başka bir görüntüde bulmak (Şekil 5).", None, None)]),
    P([("1) Nesneyi bir sayı dizisine çevirmek. ", True, None),
       ("Hem referans fotoğrafı hem de gelen kare, ağırlıkları hiç değiştirilmeyen bir "
        "DINOv3 gövdesinden geçirilir. Bu model her görüntüyü sayı vektörlerine çevirir. "
        "Referans için tüm görüntüyü özetleyen tek bir vektör saklanır (CLS vektörü); "
        "bu vektör, “bu nesne neye benziyor” sorusunun sayısal karşılığıdır.",
        None, None)]),
    P([("2) Kareyi tarayıp benzerlik haritası çıkarmak. ", True, None),
       ("Aynı model kareyi de küçük parçalara (yamalara) böler ve her yama için bir "
        "vektör üretir. Her yamanın vektörü ile referansın vektörü arasındaki açı "
        "ölçülür — kosinüs benzerliği. İki vektör aynı yöne bakıyorsa benzerlik 1'e, "
        "ilgisizse 0'a yaklaşır. Sonuç, karenin üzerine serilmiş bir benzerlik "
        "haritasıdır. Bu işlem üç ayrı çözünürlükte (672, 448 ve 224 piksel) yapılıp "
        "haritalar birleştirilir, çünkü aynı nesne farklı irtifalarda farklı "
        "büyüklükte görünür.", None, None)]),
    P([("3) Küçük nesneler için karolama. ", True, None),
       ("1920×1080'lik bir karede küçük bir nesne, modele verilen küçültülmüş görüntüde "
        "birkaç piksele iner ve kaybolur. Bunu önlemek için kare %20 örtüşmeli 640 "
        "piksellik karolara bölünür ve her karoda ayrıca arama yapılır [12]. Karonun "
        "içinde nesne göreli olarak büyüdüğü için benzerlik tepesi belirginleşir.",
        None, None)]),
    P([("4) Eşiği sahneye göre ayarlamak. ", True, None),
       ("“Benzerlik 0,4'ün üstündeyse tespit var” gibi sabit bir eşik, "
        "sahnenin kontrastına göre bazen çok gevşek bazen çok sıkı kalır. Bunun yerine "
        "tepe değerinin ", None, None),
       ("kendi arka planından", True, None),
       (" ne kadar ayrıştığına bakılır. Haritanın ortalaması ve standart sapması (σ, "
        "değerlerin ortalamadan tipik uzaklığı) hesaplanır; tespit ancak tepe, "
        "ortalamadan 3σ yukarıdaysa geçerli sayılır, kutu sınırı ise ortalama + 2σ "
        "düzeyinden çizilir. Böylece eşik her karede kendini ayarlar. Ayrıca kutu "
        "karenin alanının %55'inden büyükse tespit reddedilir; bu, benzerliğin tüm "
        "kareye yayıldığı, yani modelin aslında bir şey bulamadığı durumları eler.",
        None, None)]),
    P([("5) Haritadan kutuya. ", True, None),
       ("Eşiği geçen bölge önce morfolojik kapama ile boşluklarından arındırılır, sonra "
        "bağlantılı bileşenlerden kutu çıkarılır ve üst üste binen kutular bastırılır. "
        "Her referans için yalnız en güçlü tek kutu gönderilir. Bunun nedeni çıktı "
        "biçimidir: şartnamenin istediği pakette bir güven skoru alanı yoktur, "
        "dolayısıyla gönderilen her fazladan kutu doğrudan yanlış pozitif sayılır ve "
        "puanı düşürür.", None, None)]),
    SEKIL(V / "C1_gorev3_akis.png",
          "Görev 3 akışı — dondurulmuş DINOv3 gövdesiyle çok ölçekli şablon eşleme, "
          "karolama ve istatistiksel eşik.", tam_genislik=True),
    P([("6) Arama penceresi. ", True, None),
       ("Sunucu her referans için bir kare aralığı bildirir; referans yalnız kendi "
        "aralığında aranır. Bu, nesnenin sahnede hiç bulunmadığı karelerde üretilebilecek "
        "yanlış pozitifleri yapısal olarak ortadan kaldırır — eşik tartışması yalnızca "
        "aralık içindeki davranışı ilgilendirir.", None, None)]),
]

# ---------------------------------------------------------------- SONUCLAR
SEC[41] = [
    P("Bu bölüm dört soruya cevap vermektedir: üç görevin her biri ne kadar iyi "
      "çalışıyor, hangi fikirler denenip elendi ve sevk edilecek paketin tamamı "
      "gerçek bir oturumu tamamlayabiliyor mu. Bütün sayılar aynı boru hattından ve "
      "gerçek uçuş kareleri üzerinden gelmektedir; farklı turlarda ölçülen değerler "
      "kendi taban değerleriyle birlikte verilmiştir."),
    P("Bir değişikliğin üretim koduna girebilmek için geçmesi gereken üç kapı Şekil "
      "6'da özetlenmektedir. Bu bölümdeki her sonuç o düzenin çıktısıdır: önce "
      "fikrin kusursuz bilgiyle ulaşabileceği tavan ölçülür, sonra fikir sekiz ayrı "
      "uçuşta tek tek sınanır, en sonunda üretilen kutulara gözle bakılır."),
    SEKIL(S / "sekil_karar_sureci.png",
          "Bir fikrin sevk edilene kadar geçtiği üç kapı. Kapılardan biri bile "
          "geçilemezse fikir kapatılır ve gerekçesi kayda geçirilir."),

    P([("Pozisyon kestirimi. ", True, None),
       ("Bu turun tek sevk edilen değişikliği, Bölüm III'te anlatılan düşey eksen "
        "işareti kuralıdır. Kazancı vermeden önce bir kısıtı açıkça belirtmek gerekir: "
        "çevrim içi oturumda sunucu, ", None, None),
       ("konum bilgisinin kesildiği karelerin doğru cevabını hiç göndermemiştir", True, None),
       (". Bu yüzden gerçek kesinti karelerindeki hata ölçülemez ve bu raporda hiçbir "
        "sayıya temel yapılmamıştır. Ölçüm, birbirinden bağımsız iki kapıdan "
        "gelmektedir.", None, None)]),
    P([("Birinci kapı — genelleme sınavı. ", True, None),
       ("Çevrim içi oturumda konum bilgisinin geldiği üç sağlıklı blok vardır: 0–450, "
        "900–1050 ve 1550–1650. kareler. Sunucu bu karelerin doğru konumunu vermiştir. "
        "Sınav, bu bloklardan birini ", None, None),
       ("yapay olarak", True, None),
       (" kesintiye çevirir — konum bilgisi sisteme verilmez, sistem o bloğu yalnız "
        "görüntüden çözmek zorunda kalır — ve üretilen konum, elimizdeki gerçek konumla "
        "karşılaştırılır. Böylece gerçek yarışma görüntüleri üzerinde, gerçekçi kesinti "
        "uzunluğunda, her kare için gerçek hata ölçülebilir. Üstelik sistem o bloğa, "
        "kendisinden önceki ", None, None),
       ("gerçek", True, None),
       (" kesintinin sürüklenmesini de taşıyarak gelir; yani sınav gerçek koşuldan kolay "
        "değil, ondan zordur.", None, None)]),
    P([("İkinci kapı — yedi bağımsız uçuş. ", True, None),
       ("Bunların tamamı örnek veri setlerinden gelir ve her karesinde yer gerçeği "
        "vardır; beşinde (2024 ve 2025 uçuşları) hiç ayar yapılmamıştır. Ölçüm tek "
        "koşuda gürültülüdür, çünkü RANSAC noktaları rastgele seçer; bu yüzden her uçuş "
        "üç farklı nokta sırasıyla tekrarlanmış, eski ve yeni sürüm her seferinde ",
        None, None),
       ("aynı sırayla", True, None),
       (" karşılaştırılmıştır (Şekil 7).", None, None)]),
    SEKIL(S / "sekil_gorev2_kapi.png",
          "Görev 2 kabul kapısı — dokuz ölçüm: yedi bağımsız yatak ve genelleme "
          "sınavının iki penceresi. Hiçbirinde kötüleşme yoktur; kazanç dört "
          "ölçümde toplanmaktadır (sınav penceresi 1, video2-termal, v25 RGB 4K-B, "
          "v25 termal B) — bunlar düşey eksen işaretinin kararsız ölçüldüğü "
          "yatakların ta kendisidir."),
    P("Genelleme sınavı ortalaması 11,43 m'den 8,53 m'ye, yedi bağımsız yatağın "
      "ortalaması 8,12 m'den 6,69 m'ye inmiştir. Eşik değerleri 0,5–5,0 m aralığında "
      "taranmış ve sonuç değişmemiştir; yani değer tek bir yatağa uydurulmuş değildir."),

    P([("Nesne tespiti. ", True, None),
       ("Ağırlık seçimi, 74 gerçek yarışma karesi üzerinde aynı üretim yolundan "
        "koşturulan bir A/B ölçümüyle yapılmıştır (Şekil 8). Modelin kendi doğrulama "
        "kümesindeki değerleri (mAP@0,5 = 0,814, mAP@0,5:0,95 = 0,649) ağırlıklar arası "
        "kıyas için kullanılamaz, çünkü kümeler farklıdır.", None, None)]),
    SEKIL(S / "sekil_gorev1_ab.png",
          "Görev 1 — sınıf bazında A/B, 74 gerçek yarışma karesi, ham tespit."),
    P("Sevk edilen ağırlık, önceki model değişiminde gerileyen tam iki sınıfı "
      "onarmaktadır: İnsan 20'den 54'e, UAİ 7'den 19'a çıkmıştır. UAP sütunundaki azalma "
      "bir gerileme değil, Şekil 2'de gösterilen sınıflandırma düzeltmesidir."),

    P([("Referans nesne tespiti. ", True, None),
       ("Bu görevin puan ölçütü mAP'tir ancak elde yer gerçeği kutusu yoktur; bu nedenle "
        "sunucunun her referans için verdiği kare aralığı zayıf ama gerçek bir etiket "
        "olarak kullanılmıştır: o aralıkta nesne sahnededir. Ölçülen büyüklük, aktif "
        "karelerin kaçında tespit üretildiğidir (Şekil 9).", None, None)]),
    SEKIL(S / "sekil_gorev3_yatak.png",
          "Görev 3 test yatağı — yedi referansın her biri için duyarlılık; kesikli çizgi "
          "ortalamadır (%81,9). Ölçüt, tespit "
          "üretilip üretilmediğini yanıtlar; kutunun doğru nesnede olup olmadığı ancak "
          "görsel denetimle görülebilir (bkz. Şekil 11)."),
    P("Ortalama duyarlılık %81,9'dur (86/105 kare). Ardışık örneklenen karelerde kutu "
      "merkezi yedi referansın altısında yer değiştirmemekte, birinde 55,5 piksel "
      "oynamaktadır; bu, konumlandırma kararlılığının dolaylı bir göstergesidir."),

    P([("Elenen yöntem aileleri. ", True, None),
       ("Ön uç seçimi için klasik adayların tamamı tek bir turda, aynı boru hattında ve "
        "aynı kare kümesinde ölçülmüştür (Şekil 10). Öğrenilmiş adaylar ayrı turlarda "
        "ölçüldüğü için aynı eksende gösterilmemiş, kendi taban değerleriyle birlikte "
        "verilmiştir: SEA-RAFT eşleştirilmiş ölçümde RAFT ile eşittir (3,91 m karşılık "
        "3,90 m, fark −0,00 ± 0,25). Kanal ayrıştırması bunun nedenini açıklamaktadır: "
        "yöntem yön kanalında daha iyidir (1,34° karşılık 1,71°) ancak ölçek kanalında "
        "daha kötüdür (%3,93 karşılık %2,99) ve kesintide hata ölçek baskın olduğu için "
        "kazanç sönmektedir.", None, None)]),
    SEKIL(S / "sekil_yontem_eleme.png",
          "Klasik ön uç adaylarının aynı turda ölçülen sonuçları (video1, kesintideki "
          "ortalama Öklid hatası). Tek koşuda RANSAC gürültüsü bu yatakta ±0,12 m'dir."),
    P([("VGGT farklı bir ölçütle sınanmıştır", True, None),
       (", çünkü yörünge üretemediği için uçtan uca hata ile kıyaslanamaz. Sekiz "
        "karelik parçalar tek tek yer gerçeğine kâhin Sim(3) ile oturtulmuş, yani "
        "herhangi bir birleştirme yönteminin ulaşabileceği en iyi sonuç ölçülmüştür: "
        "artık hatanın kat edilen yola oranı %8,1'dir. Aynı ölçüt aynı pencerelerde "
        "mevcut ön uçta — üstelik kâhin olmadan — %1,0 çıkmaktadır.", None, None)]),
    P("Tablo 1, üretim koduna girmeden ya da girip geri alınarak elenen ailelerin "
      "tamamını ve eleme gerekçelerini bir arada vermektedir. Her satır ölçülmüş bir "
      "sonuçtur; hiçbiri okuma ya da sezgiyle kapatılmamıştır."),
    TABLO(["Yöntem ailesi", "Nasıl sınandı", "Ölçülen sonuç", "Karar"],
          [["Düşey kanalı benzerlik ölçeğinden almak",
            "örnek veri + oturumun kendi kareleri",
            "örnek veride −%25,6; oturumun iki penceresinde de kayıp",
            "geri alındı"],
           ["SEA-RAFT (optik akış)", "eşleştirilmiş ölçüm, üç yatak",
            "3,91 m — RAFT tabanı 3,90 m; fark −0,00 ± 0,25 (eşit)", "elendi"],
           ["VGGT (3B temel model)", "parça içi geometri, kâhin Sim(3)",
            "artık/yol %8,1 — aynı ölçütte mevcut ön uç %1,0", "elendi"],
           ["SIFT / AKAZE / ORB", "aynı tur, video1 (RAFT tabanı 3,77 m)",
            "4,48 / 5,46 / 14,59 m", "SIFT CPU yedeği oldu"],
           ["DPVO (öğrenilmiş yama odometrisi)", "kurulum denemesi",
            "çalışma zamanı bağımlılığı çözülemedi", "koşturulamadı"],
           ["Döngü kapama / yer tanıma", "18 eşik bileşimi, iki senaryo ailesi",
            "en iyi bileşimde 4,343 → 4,344 m", "kapalı"],
           ["Kusursuz yükseklik gözlemi (tavan)", "kâhin ölçümü, yedi yatak",
            "yedi yatağın beşinde KÖTÜLEŞME", "aile kapatıldı"],
           ["Yüksek frekans bastırma (filtre/yumuşatma)", "hata enerjisi ayrıştırması",
            "yüksek frekans tümüyle silinse kazanç 0,06 m", "aile kapatıldı"],
           ["Hareketli nesne maskeleme", "aykırı eşleşme dağılımı",
            "aykırıların %1,4'ü araç kutularında (alan oranı %1,0)", "aile kapatıldı"],
           ["Merdivene ayrı hizalama (4 ve 1 serbestlik)", "sınav + üç yatak",
            "video1 −0,231 m ama sınavın 2. penceresi 17,19 → 25,97 m", "elendi"],
           ["Düzlem içi ortonormal taban", "altı permütasyon, üç yatak",
            "sınav −0,136 m, video2-termal +0,326 m", "elendi"],
           ["Görev 3 — yama kanalı", "görsel panel denetimi",
            "sahnede karşılığı olmayan kutular (hayalet)", "geri alındı"],
           ["Görev 3 — mutlak eşiği gevşetmek", "test yatağı + görsel panel",
            "duyarlılık %81,9 → %85,7, ama eklenen dört tespit de yanlış nesnede",
            "reddedildi"]],
          [7, 6, 9, 4],
          "Denenip elenen yöntem aileleri ve eleme gerekçeleri. Aksi belirtilmedikçe "
          "değerler kesintideki ortalama Öklid hatasıdır [m].",
          tam_genislik=True, sag_hizala=False),
    P("Tablo 1'in ilk ve son satırları aynı dersi iki ayrı görevde vermektedir: ölçüm "
      "iyi görünen bir değişikliğin, farklı bir yatakta ya da görsel denetimde "
      "elenebilmesi kuraldır. Şekil 11 son satırın somut kanıtıdır."),
    SEKIL(S / "sekil_gorsel_denetim.png",
          "Görsel denetimin bir değişikliği elediği durum. Solda sunucunun verdiği "
          "referans nesne (basketbol sahası), sağda eşiğin gevşetilmesiyle gelen tespit: "
          "yol kenarı. Diğer üç kare de benzerdir — siyah-beyaz bir zemin motifi kiremit "
          "çatıya, yaya geçidine ve kapı önü paspasına eşleşmiştir. Sayısal ölçüt bu "
          "dört kareyi de “başarı” saymaktadır."),

    P([("Tam sistem provası. ", True, None),
       ("Son olarak sevk edilecek paketin bütünü, çevrim içi oturumun kendi 2250 karesi "
        "ve gerçek kesinti takvimiyle uçtan uca koşturulmuştur (Tablo 2). Bu prova "
        "sistemin ", None, None),
       ("çalışabilirliğini", True, None),
       (" ölçer: süre, çıktı biçimi ve kararlılık. Tabloda bir Görev 2 hata değeri "
        "bilerek yoktur — gerçek kesinti karelerinin doğru cevabı elimizde olmadığı için "
        "oradan bir hata sayısı üretilemez; o görevin doğruluğu yukarıdaki iki kapıyla "
        "ölçülmüştür.", None, None)]),
    TABLO(["Ölçüt", "Değer"],
          [["Kare başı süre", "824 ms"],
           ["2250 karelik oturum", "31 dk (pencere 90 dk)"],
           ["Görev 1 toplam tespit", "5540"],
           ["Çıktı şeması hatası", "0"],
           ["Görev 3 duyarlılığı", "%81,9"]],
          [4, 4],
          "Tam sistem provası — 2250 gerçek kare, gerçek kesinti takvimi."),
]

# ---------------------------------------------------------------- DEGERLENDIRME
SEC[43] = [
    P("Elde edilen sonuçlar, geometri merkezli ve sağlık-durumlu hibrit mimarinin bu "
      "problem için uygun olduğunu göstermektedir. Konum kestiriminde genelleme sınavı "
      "hatası 11,43 m'den 8,53 m'ye, yedi bağımsız yatağın ortalaması 8,12 m'den 6,69 m'ye "
      "inmiş; nesne tespitinde önceki model değişiminde gerileyen iki sınıf onarılmış; "
      "referans nesne tespitinde eğitim gerektirmeyen tek-atış eşleme, yedi referansın "
      "tamamında çalışmış ve aktif karelerin %81,9'unda tespit üretmiştir."),
    P([("Çalışmanın literatüre asıl katkısı yeni bir yöntem değil, bir ", None, None),
       ("değerlendirme yöntemidir", True, None),
       (". Monoküler görsel odometri literatüründe sonuçlar çoğunlukla tek bir veri "
        "kümesi ailesinde raporlanır; oysa bu problemde tek bir uçuşta ölçülen kazancın "
        "başka bir uçuşta yok olması kuraldır. Bu çalışmada her aday, kod yazılmadan "
        "önce bir tavan deneyinden geçirilmiştir: fikre gerçekte asla sahip olamayacağı "
        "kusursuz bilgi verilir ve ulaşabileceği en iyi sonuç ölçülür. Tavan bile bir "
        "kazanç göstermiyorsa fikir hiç kodlanmaz. Bu deney tipik olarak 10–30 dakika "
        "sürmekte ve haftalarca sürecek bir alt sistemi tek oturumda elemektedir. Yalnız "
        "son turda dokuz yöntem ailesi bu biçimde, üretim kodu hiç yazılmadan "
        "kapatılmıştır; Tablo 1 bütün turların birikimini, yazılıp sonradan geri "
        "alınanlar dâhil vermektedir.", None, None)]),
    P("Kapatılan ailelerden dört yapısal teşhis çıkmıştır; bunlar tek tek yöntemlerden daha "
      "değerlidir, çünkü gelecekteki fikirleri önceden elemektedirler:"),
    MADDE([("Yükseklik bir hata kaynağı değil, bir telafidir. ", True, None),
           ("Kesinti adımında yalnız sonuç gözlenmekte; eksen dönüşümü, birim adım ve "
            "yükseklik ayrı ayrı gözlenmemektedir. Yükseklik serbest kalan tek parametredir "
            "ve diğerlerindeki yanlılığı soğurmaktadır. Kusursuz bir yükseklik gözlemi "
            "verildiğinde sistem yedi yatağın beşinde kötüleşmiştir. Bu bulgu, daha önce "
            "ayrı gerekçelerle kaydedilmiş beş başarısızlığı tek bir açıklamaya "
            "bağlamaktadır.", None, None)]),
    MADDE([("Kesinti hatası saf sürüklenmedir. ", True, None),
           ("Hata enerjisinin yüksek frekanslı bileşeni sekiz yatakta da ölçülebilir "
            "düzeyin altındadır; yüksek frekans kusursuz biçimde silinseydi kazanç 0,06 "
            "m'de kalırdı. Dolayısıyla filtreleme, yumuşatma ve sıçrama bastırma ailesinin "
            "tamamı bu problemde etkisizdir.", None, None)]),
    MADDE([("Aykırı eşleşmelerin kaynağı üç boyutlu yapıdır, hareketli nesneler değil. ",
            True, None),
           ("Aykırıların yalnızca %1,4'ü hareketli araç kutularına düşmektedir ki bu, "
            "kutuların kapladığı alan oranıyla (%1,0) aynıdır. Ayrıca inlier kümesi uzaysal "
            "olarak kümelenmemektedir. Bina ve ağaç paralaksı sahnenin tamamına "
            "dağıldığından, daha iyi nokta seçimine dayanan yaklaşımların düzeltebileceği "
            "bir şey bulunmamaktadır.", None, None)]),
    MADDE([("Yön ve ölçek kanalları zıt yönde çekmektedir. ", True, None),
           ("Aynı gerilim üç bağımsız ön uçta gözlenmiştir. Kesintide hata ölçek baskın "
            "olduğundan, bir ön uç adayının kaderini ölçek kanalındaki başarımı "
            "belirlemektedir.", None, None)]),
    P("Çalışmanın bilinen sınırları da açıkça kaydedilmiştir. Konum kestiriminde kalan hata "
      "ölçektedir ve yüksekliğin kesinti boyunca mutlak bir gözlemi yoktur. Termal "
      "modalitede hata dağılımının kuyruğu ağırdır: toplam hatanın dörtte biri en kötü %5 "
      "kareden gelmektedir; bu risk, genelleme sınavının pencerelerinde kuyruk "
      "bulunmadığından o sınavla yakalanamamakta ve sonraki çalışmanın en anlamlı hedefi "
      "olarak belirlenmektedir. Referans nesne tespitinde ise kutu isabeti, yer gerçeği "
      "kutusu bulunmadığı için doğrudan ölçülememekte; mevcut ölçüt “tespit var mı” "
      "sorusunu yanıtlamakta, “doğru yerde mi” sorusunu yanıtlayamamaktadır."),
]

# DIKKAT: metinlerde "[n] " on eki YOK — Word bu paragraflari numaralandirilmis
# liste (numId 2) olarak bicimlendirdigi icin koseli parantezli sayiyi kendi basar.
REFS = [
    "TEKNOFEST 2026 Havacılıkta Yapay Zekâ Yarışması Teknik Şartnamesi, 2026.",
    "R. Neyestani, F. Picariello, I. Ahmed, P. Daponte and L. De Vito, “From pixels to "
    "precision: A survey of monocular visual odometry in digital twin applications,” "
    "Sensors, vol. 24, no. 4, 2024.",
    "T. Phan and A. Kim, “Toward specialized learning-based approaches for visual "
    "odometry: A comprehensive survey,” J. Intell. Robot. Syst., 2025.",
    "Z. Teed and J. Deng, “RAFT: Recurrent all-pairs field transforms for optical "
    "flow,” in Proc. ECCV, 2020, pp. 402–419.",
    "Y. Wang, L. Lipson and J. Deng, “SEA-RAFT: Simple, efficient, accurate RAFT for "
    "optical flow,” in Proc. ECCV, 2024.",
    "Y. Xu and G. C. H. E. de Croon, “CUAHN-VIO: Content-and-uncertainty-aware "
    "homography network for visual-inertial odometry,” Robot. Auton. Syst., 2025.",
    "L. Lipson, Z. Teed and J. Deng, “Deep patch visual SLAM,” in Proc. ECCV, 2024.",
    "J. Wang et al., “VGGT: Visual geometry grounded transformer,” in Proc. CVPR, 2025.",
    "R. Murai, E. Dexheimer and A. J. Davison, “MASt3R-SLAM: Real-time dense SLAM with "
    "3D reconstruction priors,” in Proc. CVPR, 2025.",
    "M. Oquab et al., “DINOv2: Learning robust visual features without supervision,” "
    "Trans. Mach. Learn. Res., 2024.",
    "N. Aharon, R. Orfaig and B.-Z. Bobrovsky, “BoT-SORT: Robust associations "
    "multi-pedestrian tracking,” arXiv:2206.14651, 2022.",
    "F. C. Akyon, S. O. Altinuc and A. Temizel, “Slicing aided hyper inference and "
    "fine-tuning for small object detection,” in Proc. IEEE ICIP, 2022, pp. 966–970.",
    "S. Umeyama, “Least-squares estimation of transformation parameters between two "
    "point patterns,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 13, no. 4, "
    "pp. 376–380, 1991.",
]

# ================================================================= UYGULA
yeni = list(paras)


def metin_degistir(i, yeni_metin, punto=None):
    """Paragrafin metnini degistir, bicimini koru.

    Sablonun kapak run'lari icinde <w:br/> ile ayrilmis birden fazla <w:t> bulunur;
    bu yuzden run'in govdesi rPr korunarak bastan kurulur.
    """
    p = yeni[i]
    # DIKKAT: run'lar nitelikli olabilir (<w:r w:rsidR="...">); niteliksiz desen
    # sablonun kapak run'larini hic yakalamiyordu ve degistirme sessizce atlaniyordu.
    runs = re.findall(r'<w:r(?:\s[^>]*)?>.*?</w:r>', p, re.S)
    if not runs:
        return
    ilk = runs[0]
    m = re.search(r'<w:rPr>.*?</w:rPr>', ilk, re.S)
    rpr = m.group(0) if m else ""
    if punto:
        yari = int(punto * 2)
        if re.search(r'<w:sz w:val="\d+"/>', rpr):
            rpr = re.sub(r'<w:sz w:val="\d+"/>', f'<w:sz w:val="{yari}"/>', rpr)
        elif rpr:
            rpr = rpr.replace("</w:rPr>", f'<w:sz w:val="{yari}"/></w:rPr>')
    ilk_yeni = f'<w:r>{rpr}<w:t xml:space="preserve">{esc(yeni_metin)}</w:t></w:r>'
    p = p.replace(ilk, "\x00", 1)
    for r in runs[1:]:
        p = p.replace(r, "", 1)
    yeni[i] = p.replace("\x00", ilk_yeni)


for i, t in KAPAK.items():
    metin_degistir(i, t)

# ---- kapak: baslik satir araligi. Sablonda w:line="120" (yarim satir) yazili;
# 11 puntoluk basliklar bu yuzden ust uste biniyordu.
for i in range(0, 5):
    yeni[i] = re.sub(r'(<w:spacing[^/]*?)w:line="120"', r'\1w:line="240"', yeni[i])

# Takim adi kapagin kimlik satiridir; sablonda govde metniyle ayni agirlikta
# kaliyor ve basliklarin arasinda kayboluyordu. Koyu ve bir punto buyuk.
yeni[3] = re.sub(r'<w:b w:val="[01]"/>', '<w:b w:val="1"/>', yeni[3])
yeni[3] = re.sub(r'<w:bCs w:val="[01]"/>', '<w:bCs w:val="1"/>', yeni[3])
yeni[3] = re.sub(r'<w:sz w:val="\d+"/>', '<w:sz w:val="26"/>', yeni[3])
yeni[3] = re.sub(r'<w:szCs w:val="\d+"/>', '<w:szCs w:val="26"/>', yeni[3])
# Takim ID satirinin altina nefes payi: uye blogu basliklara yapisik duruyordu.
yeni[4] = re.sub(r'(<w:spacing[^/]*?)w:after="\d+"', r'\1w:after="260"', yeni[4])

# ---- kapak: uye alani (bkz. UYELER yorumu)
# Ad koyu, gorev normal: dort satirin hangi kismi ad hangisi gorev, tek bakista
# ayrilsin. Tek paragraf icinde <w:br/> ile alt alta -> satir araligi esit kalir.
_uye_ppr = re.sub(r'w:line="\d+"', 'w:line="276"', ppr_of(6))
_uye_rpr = rpr_of(6)


def _uye_run(metin, kalin):
    r = _uye_rpr
    if kalin:
        r = re.sub(r'<w:b w:val="[01]"/>', '<w:b w:val="1"/>', r)
        r = re.sub(r'<w:bCs w:val="[01]"/>', '<w:bCs w:val="1"/>', r)
    return f'<w:r>{r}<w:t xml:space="preserve">{esc(metin)}</w:t></w:r>'


_satirlar = []
for _k, (_ad, _gorev) in enumerate(UYELER):
    _on = '<w:r>' + _uye_rpr + '<w:br/></w:r>' if _k else ""
    _satirlar.append(_on + _uye_run(_ad, True) + _uye_run(f"  ·  {_gorev}", False))
yeni[6] = f'<w:p>{_uye_ppr}' + "".join(_satirlar) + '</w:p>'
for i in (7, 8, 9, 10, 11):
    yeni[i] = ""
# 12: kapak gorseli · 13: gorsel alti tek satir · 14/15/16: sectPr tasidiklari
# icin bosaltilir, silinmez
_kapak_ppr = re.sub(r'<w:spacing [^/]*/>',
                    '<w:spacing w:after="120" w:before="360" w:line="240" '
                    'w:lineRule="auto"/>', ppr_of(6))
_kapak_ppr = re.sub(r'<w:jc w:val="\w+"/>', '<w:jc w:val="center"/>', _kapak_ppr)
yeni[12] = RESIM(S / "sekil_kapak.png", 10000).replace(
    '<w:spacing w:after="60" w:before="120" w:line="240" w:lineRule="auto"/>',
    '<w:spacing w:after="120" w:before="480" w:line="240" w:lineRule="auto"/>')
yeni[13] = (f'<w:p>{_kapak_ppr}<w:r>{rpr_of(6)}<w:t xml:space="preserve">'
            + esc("Üç görev, tek canlı oturum: nesne tespiti · "
                  "GNSS-reddedilmiş konum kestirimi · referans nesne eşleme")
            + '</w:t></w:r></w:p>')
metin_degistir(14, "")
for i in (14, 15, 16):
    yeni[i] = re.sub(r'<w:cols w:equalWidth="0" w:num="3">.*?</w:cols>',
                     '<w:cols w:num="1" w:space="360"/>', yeni[i], flags=re.S)

metin_degistir(17, OZET)
metin_degistir(18, ANAHTAR)

for idx, blok in SEC.items():
    yeni[idx] = "".join(paras[idx] if b == "@BASLIK@" else b for b in blok)

metin_degistir(46, "Rapor içerisinde alıntı yapılan kaynaklar aşağıda listelenmiştir.")
yeni[47] = ""
yeni[48] = ""
# Sablonda 49-55 arasi YEDI numarali kaynak yuvasi var; ilk yedi kaynak bunlara
# yazilir, kalanlar ayni liste bicimiyle (numId 2) 55'in ardina eklenir. Boylece
# numaralandirma [1]'den [13]'e kesintisiz akar ve bicim tek tiptir.
for k in range(7):
    metin_degistir(49 + k, REFS[k])
    yeni[49 + k] = re.sub(r'<w:spacing[^/]*/>',
                          '<w:spacing w:after="0" w:before="0" w:line="226" '
                          'w:lineRule="auto"/>', yeni[49 + k], count=1)

# DIKKAT: 55. paragraf GOVDENIN IKI SUTUNLU sectPr'sini tasiyor; bolum daima SON
# paragrafta bitmeli, bu yuzden sectPr eklenen son kaynaga tasiniyor.
_m = re.search(r'<w:sectPr[^>]*>.*?</w:sectPr>', yeni[55], re.S)
_govde_sect = _m.group(0) if _m else ""
_p55 = yeni[55].replace(_govde_sect, "", 1) if _govde_sect else yeni[55]
_ekler = [f'<w:p>{REF_PPR}<w:r>{REF_RPR}'
          f'<w:t xml:space="preserve">{esc(r)}</w:t></w:r></w:p>' for r in REFS[7:]]
# 'nextPage' artik ilk govde parcasinda (bkz. SEKIL); burada kalirsa kaynakcadan
# once gereksiz bir sayfa sonu olusur.
_govde_sect = _govde_sect.replace('<w:type w:val="nextPage"/>',
                                  '<w:type w:val="continuous"/>')
if _govde_sect and _ekler:
    # sema sirasi: sectPr, pPr'nin SONUNDA (rPr'den sonra) yer alir.
    _ekler[-1] = _ekler[-1].replace("</w:pPr>", f"{_govde_sect}</w:pPr>", 1)
yeni[55] = _p55 + "".join(_ekler)
# Sablonun govde sonundaki bos paragrafi + 'nextPage' bolumu, kaynakcadan sonra
# bombos bir sayfa aciyordu.
yeni[56] = ""

govde = x
for eski, yenisi in zip(paras, yeni):
    if eski != yenisi:
        govde = govde.replace(eski, yenisi, 1)


# Belgenin en sonundaki (govde duzeyindeki) sectPr 'nextPage' idi; kaynakcadan
# sonra tamamen bos bir sayfa aciyordu. O bolumde artik hic paragraf kalmadigi
# icin surekli yapiliyor.
_kuyruk = govde[govde.rfind("</w:p>"):]
_m_son = re.search(r'<w:sectPr[^>]*>.*?</w:sectPr>', _kuyruk, re.S)
if _m_son:
    govde = govde.replace(
        _m_son.group(0),
        _m_son.group(0).replace('<w:type w:val="nextPage"/>',
                                '<w:type w:val="continuous"/>'), 1)
DOC.write_text(govde, encoding="utf-8")

# ---- paketle
if CIKTI.exists():
    CIKTI.unlink()
dosyalar = [p for p in CALISMA.rglob("*") if p.is_file()]
dosyalar.sort(key=lambda p: (p.name != "[Content_Types].xml", str(p)))
with zipfile.ZipFile(CIKTI, "w", zipfile.ZIP_DEFLATED) as z:
    for p in dosyalar:
        z.write(p, p.relative_to(CALISMA).as_posix())

print(f"{CIKTI.name}  ({CIKTI.stat().st_size / 1024:.0f} KB)")
print(f"  sekil: {_sekil_no[0]} · tablo: {_tablo_no[0]} · gomulu resim: {_resim_sayac[0]}")
shutil.rmtree(CALISMA)
