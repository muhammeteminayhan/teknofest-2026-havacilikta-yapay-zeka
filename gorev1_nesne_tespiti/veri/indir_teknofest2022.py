"""
TEKNOFEST 2022 Ulaşımda YZ veri seti (Türkçe yarışma kuşbakışı verisi — en yakın dağılım).

İçerik: ~25,000 kuşbakışı görüntü, 300k+ etiket; sınıflar Taşıt & İnsan/Yaya (YOLO formatı).
Bu, TEKNOFEST kamera açısı/irtifa dağılımına en yakın açık kaynaktır; Taşıt/İnsan için
güçlü ön-eğitim sağlar (UAP/UAİ yok).

Kaynak (topluluk): https://github.com/sezer-muhammed/Teknofest-Ulasimda-Yapay-Zeka-Veri-Seti

Kullanım:
    python gorev1_nesne_tespiti/veri/indir_teknofest2022.py
"""
from __future__ import annotations

import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
from gorev1_nesne_tespiti.configs import config as C  # noqa: E402

KAYNAK = "https://github.com/sezer-muhammed/Teknofest-Ulasimda-Yapay-Zeka-Veri-Seti"
HEDEF = os.path.join(C.DATASETS_DIR, "Teknofest2022")


def main():
    os.makedirs(HEDEF, exist_ok=True)
    print("TEKNOFEST 2022 Ulaşımda YZ — YOLO formatı, Taşıt+Yaya.")
    print(f"  git clone {KAYNAK} {HEDEF}")
    print("  Sınıf eşleme: vehicle→Taşıt(0), pedestrian/human→İnsan(1)")
    print("  Not: kamera açısı/irtifa TEKNOFEST'e en yakın açık veri; ön-eğitim için ideal.")


if __name__ == "__main__":
    main()
