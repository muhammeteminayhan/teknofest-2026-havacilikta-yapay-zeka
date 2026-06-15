"""
DroneVehicle indirme (RGB + TERMAL eşli hava araç tespiti — termal modalite için KİLİT).

İçerik: 28,439 RGB-kızılötesi eşli görüntü (5 araç sınıfı). Yarışmada termal kameradan
da kare geleceğinden RGB+termal füzyon ve termal-sağlam tespit için en değerli kaynak.

Erişim: veri seti BaiduYun (Çin) üzerinden dağıtılıyor; Google Drive yansıları zaman
zaman değişiyor. Bu yüzden indirme YARI-OTOMATİK: aşağıdaki URL/landing sayfasından
indirip `datasets/DroneVehicle/` altına açın, sonra YOLO formatına dönüştürün.

Resmi depo: https://github.com/VisDrone/DroneVehicle
Makale: https://arxiv.org/abs/2003.02437

Kullanım:
    # 1) Veriyi manuel indirip datasets/DroneVehicle/ altına açın (raw/)
    # 2) Etiketleri YOLO formatına çevirin (oriented->axis-aligned):
    python gorev1_nesne_tespiti/veri/indir_dronevehicle.py --donustur
"""
from __future__ import annotations

import argparse
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
from gorev1_nesne_tespiti.configs import config as C  # noqa: E402

KAYNAK = "https://github.com/VisDrone/DroneVehicle"
HEDEF = os.path.join(C.DATASETS_DIR, "DroneVehicle")


def donustur():
    """DroneVehicle oriented-bbox (xml) → YOLO axis-aligned txt. (İskelet — raw/ bekler.)"""
    raw = os.path.join(HEDEF, "raw")
    if not os.path.isdir(raw):
        print(f"HATA: {raw} yok. Önce veriyi manuel indirip buraya açın.")
        print(f"Kaynak: {KAYNAK}")
        return
    # TODO(sonraki tur): xml oriented kutuları oku → eksen-hizalı YOLO txt yaz,
    # sınıfları Taşıt(0)'a indir. RGB ve termal alt-klasörleri ayrı işle.
    print("Dönüştürme iskeleti: raw/ bulundu. YOLO dönüştürücü sonraki turda eklenecek.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--donustur", action="store_true", help="raw/ → YOLO formatı")
    args = ap.parse_args()
    os.makedirs(HEDEF, exist_ok=True)
    if args.donustur:
        donustur()
    else:
        print("DroneVehicle yarı-otomatik indirme.")
        print(f"  1) İndir: {KAYNAK}")
        print(f"  2) Aç:    {os.path.join(HEDEF, 'raw')}")
        print(f"  3) Çevir: python {__file__} --donustur")


if __name__ == "__main__":
    main()
