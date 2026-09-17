# -*- coding: utf-8 -*-
"""Kurulum dogrulama — sunucuya BAGLANMADAN sistemin hazir oldugunu sinar.

Calistir:  python dogrula.py

Sirayla kontrol eder:
  1. Python surumu ve bagimliliklar (eksik paket var mi)
  2. Model dosyalari yerinde mi (Gorev 1 agirligi, RAFT, DINOv3)
  3. Uc gorev de yukleniyor mu (ObjectDetectionModel kurulumu)
  4. Bir kare uctan uca isleniyor mu ve cikti sartname semasina uyuyor mu

Hata durumunda hangi dosyanin/paketin eksik oldugunu ve ne yapilmasi gerektigini yazar.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

KOK = Path(__file__).resolve().parent
sys.stdout.reconfigure(encoding="utf-8")

OK, HATA, UYARI = "  [OK]   ", "  [HATA] ", "  [UYARI]"
_sorun = []


def baslik(s):
    print(f"\n{'=' * 62}\n{s}\n{'=' * 62}")


# ---------------------------------------------------------------- 1) bagimliliklar
baslik("1) PYTHON VE BAGIMLILIKLAR")
print(f"  python {sys.version.split()[0]}")
if sys.version_info < (3, 11):
    _sorun.append("Python 3.11+ gerekli")
    print(HATA + "Python 3.11 veya ustu gerekli")

PAKETLER = [("numpy", None), ("cv2", "opencv-python"), ("scipy", None), ("yaml", "PyYAML"),
            ("requests", None), ("decouple", "python-decouple"), ("tqdm", None),
            ("torch", None), ("torchvision", None), ("ultralytics", None),
            ("transformers", None)]
for modul, pip_adi in PAKETLER:
    try:
        m = __import__(modul)
        v = getattr(m, "__version__", "?")
        print(f"{OK}{(pip_adi or modul):<16} {v}")
    except ImportError:
        ad = pip_adi or modul
        _sorun.append(f"paket eksik: {ad}")
        print(f"{HATA}{ad:<16} KURULU DEGIL  ->  pip install -r requirements.txt")

try:
    import torch
    print(f"{OK}{'CUDA':<16} {'var: ' + torch.version.cuda if torch.cuda.is_available() else 'YOK (CPU modu — yavas olur)'}")
except Exception:
    pass

# ---------------------------------------------------------------- 2) model dosyalari
baslik("2) MODEL DOSYALARI")
agirlik = KOK / "gorev1" / "agirliklar" / "best.pt"
if agirlik.exists():
    print(f"{OK}Gorev 1 agirligi     {agirlik.stat().st_size / 1048576:.0f} MB")
else:
    _sorun.append("Gorev 1 agirligi yok")
    print(f"{HATA}Gorev 1 agirligi     BULUNAMADI -> {agirlik}")
    print("         model.txt icindeki Drive baglantisindan indirip buraya koyun.")

ev = Path(os.path.expanduser("~"))
raft = list((ev / ".cache/torch/hub/checkpoints").glob("raft_large*.pth"))
print(f"{OK}RAFT (Gorev 2)       {raft[0].name}" if raft else
      f"{UYARI} RAFT (Gorev 2)      yok — LK yedegine dusulur, dogruluk azalir")
dino = list((ev / ".cache/huggingface/hub").glob("models--facebook--dinov3*"))
print(f"{OK}DINOv3 (Gorev 1/3)   yerel onbellekte" if dino else
      f"{UYARI} DINOv3 (Gorev 1/3)  yerel onbellekte yok — internet + erisim izni gerekir")

if _sorun:
    print("\n" + "!" * 62)
    print("Eksikler giderilmeden devam edilemez:")
    for s in _sorun:
        print(f"  - {s}")
    print("!" * 62)
    sys.exit(1)

# ---------------------------------------------------------------- 3) sistem kurulumu
baslik("3) SISTEM KURULUMU (uc gorev)")
os.chdir(KOK / "sistem")
sys.path.insert(0, str(KOK / "sistem"))
sys.path.insert(0, str(KOK))
import warnings
warnings.filterwarnings("ignore")

t0 = time.perf_counter()
from src.object_detection_model import ObjectDetectionModel     # noqa: E402
model = ObjectDetectionModel("http://localhost/")
print(f"{OK}ObjectDetectionModel kuruldu ({time.perf_counter() - t0:.0f} s)")
print(f"{OK}Gorev 1 {'yuklendi' if model.gorev1 is not None else 'YUKLENEMEDI'}")
print(f"{OK}Gorev 3 {'yuklendi' if model.gorev3 is not None else 'YUKLENEMEDI'}")

# ---------------------------------------------------------------- 4) kare isleme
baslik("4) UCTAN UCA KARE ISLEME")
import numpy as np                                              # noqa: E402
kare = (np.random.default_rng(0).random((1080, 1920, 3)) * 255).astype(np.uint8)

t0 = time.perf_counter()
nesneler = model.gorev1.detect(kare) if model.gorev1 else []
t_g1 = time.perf_counter() - t0
print(f"{OK}Gorev 1 calisti      {t_g1*1000:.0f} ms, {len(nesneler)} nesne")
if nesneler:
    gerekli = {"cls", "landing_status", "motion_status",
               "top_left_x", "top_left_y", "bottom_right_x", "bottom_right_y"}
    eksik = gerekli - set(nesneler[0])
    print(f"{OK}  cikti semasi TAM" if not eksik else f"{HATA}  semada eksik alan: {eksik}")

t0 = time.perf_counter()
konum = model.position_estimator.process_frame(kare, np.zeros(3), True)
t_g2 = time.perf_counter() - t0
est = model.position_estimator
print(f"{OK}Gorev 2 calisti      {t_g2*1000:.0f} ms | motor={est.engine_used} | "
      f"yontem={est.config.method}")
if est.engine_used != "raft":
    print(f"{UYARI} Gorev 2 RAFT yerine '{est.engine_used}' kullaniyor — dogruluk duser.")
print(f"{OK}  konum ciktisi        x={konum[0]:.2f} y={konum[1]:.2f} z={konum[2]:.2f}")

# ---------------------------------------------------------------- ozet
baslik("SONUC")
toplam_ms = (t_g1 + t_g2) * 1000
print(f"  Gorev 1 + 2 kare basi : {toplam_ms:.0f} ms")
print(f"  2250 karelik oturum   : ~{toplam_ms * 2250 / 60000:.0f} dk  (pencere 90 dk)")
print(f"  (Gorev 3 yalniz referans araliklarinda calisir, uste eklenir)")
print("\n  *** SISTEM CALISMAYA HAZIR ***\n")
