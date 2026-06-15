"""
Görev 1 fine-tune scripti (iskelet).

Strateji (sonraki turlarda doldurulacak):
1. Ön-eğitim/fine-tune: VisDrone + TEKNOFEST2022 (Taşıt/İnsan) → YOLOv11.
2. Termal: DroneVehicle + HIT-UAV ile fine-tune (ayrı ağırlık ya da füzyon).
3. UAP/UAİ: örnek veri + sentetik artırma ile ek sınıflar.
4. Birleşik 4-sınıf (Taşıt/İnsan/UAP/UAİ) veri yaml'ı ile son fine-tune.

Çıktı ağırlığı `agirliklar/g1_finetuned.pt` olarak kaydedilir (config.YOLO_FINETUNED).

Kullanım (internetli makinede):
    python gorev1_nesne_tespiti/egitim/train.py --data datasets/g1_birlesik.yaml --epochs 100 --imgsz 1280
"""
from __future__ import annotations

import argparse
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
from gorev1_nesne_tespiti.configs import config as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="YOLO veri yaml yolu")
    ap.add_argument("--model", default=C.YOLO_MODEL, help="başlangıç ağırlığı")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=1280, help="küçük nesne için yüksek")
    ap.add_argument("--batch", type=int, default=8, help="8GB VRAM için ölçekle")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.model)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, device=C.DEVICE, half=C.HALF,
                project=C.WEIGHTS_DIR, name="g1_train")
    print(f"Eğitim bitti. En iyi ağırlık: {C.WEIGHTS_DIR}/g1_train/weights/best.pt")
    print(f"Yarışma için {C.YOLO_FINETUNED} olarak kopyalayın.")


if __name__ == "__main__":
    main()
