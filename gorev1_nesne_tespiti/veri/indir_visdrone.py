"""
VisDrone-DET indirme (RGB hava araç/insan tespiti — birincil dış veri).

İçerik: ~10k statik görüntü + 2.6M kutu; sınıflar pedestrian/people/car/van/truck/
bus/motor... (siniflar.py'de Taşıt/İnsan'a eşlenir). Lisans: akademik/araştırma.
Ultralytics `VisDrone.yaml` ile otomatik indirilir ve YOLO formatına dönüştürülür.

Kullanım (internetli makinede, yarışmadan ÖNCE):
    python gorev1_nesne_tespiti/veri/indir_visdrone.py
Ardından datasets/VisDrone altında train/val/test (YOLO) hazır olur.
"""
from __future__ import annotations

import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
from gorev1_nesne_tespiti.configs import config as C  # noqa: E402


def main():
    os.makedirs(C.DATASETS_DIR, exist_ok=True)
    # Ultralytics settings ile veri kök dizinini datasets/ yap.
    from ultralytics import settings
    settings.update({"datasets_dir": C.DATASETS_DIR})
    # check_det_dataset VisDrone.yaml'ı çözer ve eksikse otomatik indirir/dönüştürür.
    from ultralytics.data.utils import check_det_dataset
    print("VisDrone indiriliyor/çözülüyor (datasets_dir =", C.DATASETS_DIR, ")...")
    info = check_det_dataset("VisDrone.yaml", autodownload=True)
    print("Hazır:")
    for k in ("train", "val", "test", "nc", "names"):
        if k in info:
            print(f"  {k}: {info[k]}")
    print("\nNot: sınıf eşlemesi gorev1_nesne_tespiti/veri/siniflar.py:visdrone_eslemesi")


if __name__ == "__main__":
    main()
