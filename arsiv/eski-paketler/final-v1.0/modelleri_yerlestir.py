# -*- coding: utf-8 -*-
"""Paketteki model onbelleklerini kullanicinin ~/.cache klasorune kopyalar.

Yarisma bilgisayarinda RAFT ve DINOv3 agirliklari yoksa sistem internete cikmaya
calisir; yarisma ortaminda internet garanti degildir. Bu betik ikisini de yerine
koyar. Zaten varsa dokunmaz.

    python modelleri_yerlestir.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

BURASI = Path(__file__).resolve().parent
EV = Path.home()

ISLER = [
    (BURASI / "modeller" / "torch_hub_checkpoints",
     EV / ".cache" / "torch" / "hub" / "checkpoints",
     "RAFT optik akis agirligi (Gorev 2)"),
    (BURASI / "modeller" / "huggingface_hub",
     EV / ".cache" / "huggingface" / "hub",
     "DINOv3 omurgasi (Gorev 1 ve 3)"),
]


def kopyala(kaynak: Path, hedef: Path, ad: str) -> bool:
    if not kaynak.exists():
        print(f"  [ATLANDI] {ad}: pakette yok ({kaynak})")
        return True
    hedef.mkdir(parents=True, exist_ok=True)
    yeni = 0
    for oge in kaynak.iterdir():
        varis = hedef / oge.name
        if varis.exists():
            print(f"  [ZATEN VAR] {varis.name}")
            continue
        if oge.is_dir():
            shutil.copytree(oge, varis)
        else:
            shutil.copy2(oge, varis)
        print(f"  [KOPYALANDI] {varis}")
        yeni += 1
    if yeni == 0:
        print(f"  {ad}: hepsi zaten yerindeydi.")
    return True


def main() -> int:
    print("Model onbellekleri yerlestiriliyor\n")
    for kaynak, hedef, ad in ISLER:
        print(f"{ad}")
        kopyala(kaynak, hedef, ad)
        print()

    # dogrulama
    raft = list((EV / ".cache/torch/hub/checkpoints").glob("raft_large_*.pth"))
    dino = list((EV / ".cache/huggingface/hub").glob("models--facebook--dinov3*"))
    print("=" * 56)
    print(f"  RAFT   : {'VAR — ' + raft[0].name if raft else 'YOK'}")
    print(f"  DINOv3 : {'VAR — ' + dino[0].name if dino else 'YOK'}")
    print("=" * 56)
    if raft and dino:
        print("\nHer sey yerinde. Simdi:  python dogrula.py")
        return 0
    print("\nEKSIK VAR — YARISMA_KULLANIM.md > Sorun 3'e bakin.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
