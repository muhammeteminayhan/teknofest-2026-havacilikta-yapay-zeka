"""
HIT-UAV indirme (yüksek-irtifa TERMAL insan/araç — termal sağlamlık).

İçerik: ~2,898 termal hava görüntüsü; person/bicycle/car/other vehicle. 60-130m irtifa,
30-90° açı. Yarışmadaki termal modalite için DroneVehicle'ı tamamlar (özellikle insan).

Resmi depo: https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset
Nature Sci Data: https://www.nature.com/articles/s41597-023-02066-6

Kullanım:
    python gorev1_nesne_tespiti/veri/indir_hituav.py
(YOLO formatı depoda hazır geliyor; datasets/HIT-UAV altına klonlanır/açılır.)
"""
from __future__ import annotations

import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
from gorev1_nesne_tespiti.configs import config as C  # noqa: E402

KAYNAK = "https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset"
HEDEF = os.path.join(C.DATASETS_DIR, "HIT-UAV")


def main():
    os.makedirs(HEDEF, exist_ok=True)
    print("HIT-UAV (termal) — depo YOLO formatı içerir.")
    print(f"  git clone {KAYNAK} {HEDEF}")
    print("  (veya zip indirip açın). Sınıflar: person/bicycle/car/other-vehicle")
    print("  Eşleme: person→İnsan(1), car/other-vehicle→Taşıt(0), bicycle→Taşıt(0)")


if __name__ == "__main__":
    main()
