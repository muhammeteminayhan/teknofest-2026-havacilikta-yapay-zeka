"""
Görev 1 birim testleri — ağır model (torch/ultralytics) GEREKMEZ; yalnızca cv2+numpy.

Kapsam: sınıf eşleme, ego-hareket homografisi (sentetik kaydırma geri-kurtarımı),
artık-hareket hesabı, UAP/UAİ iniş-uygunluk mantığı. `python tests/test_g1.py` ile çalışır.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)


def test_siniflar():
    from gorev1_nesne_tespiti.veri.siniflar import (
        visdrone_eslemesi, coco_eslemesi, SINIF_ID)
    assert SINIF_ID == {"Tasit": 0, "Insan": 1, "UAP": 2, "UAI": 3}
    assert visdrone_eslemesi(4) == "Tasit"      # car
    assert visdrone_eslemesi(1) == "Insan"      # pedestrian
    assert visdrone_eslemesi(0) is None         # ignored
    assert coco_eslemesi(0) == "Insan"          # person
    assert coco_eslemesi(2) == "Tasit"          # car
    print("test_siniflar PASS")


def test_ego_homografi():
    """Sentetik dokuyu bilinen ofsetle kaydır → homografi o ötelemeyi kurtarmalı."""
    import cv2
    from ortak.ego_hareket import kestir_homografi, sabit_konum_tahmini
    rng = np.random.default_rng(0)
    base = (rng.uniform(0, 255, size=(480, 640))).astype(np.uint8)
    base = cv2.GaussianBlur(base, (3, 3), 0)
    dx, dy = 12, 7
    # prev = sol-üst kırpma; cur = (dx,dy) kaydırılmış kırpma (sahne sağa-aşağı kayar)
    prev = base[20:420, 20:580]
    cur = base[20 + dy:420 + dy, 20 + dx:580 + dx]
    prev = cv2.cvtColor(prev, cv2.COLOR_GRAY2BGR)
    cur = cv2.cvtColor(cur, cv2.COLOR_GRAY2BGR)
    ego = kestir_homografi(prev, cur)
    assert ego.ok and ego.H is not None, "homografi kestirilemedi"
    # prev'deki bir nokta → cur'da yaklaşık (x-dx, y-dy) olmalı (sahne kaydığı için)
    p = np.array([[300.0, 200.0]])
    q = sabit_konum_tahmini(ego.H, p)[0]
    assert abs(q[0] - (300 - dx)) < 2.0 and abs(q[1] - (200 - dy)) < 2.0, q
    print("test_ego_homografi PASS")


def test_artik_hareket():
    """Saf öteleme H altında: H ile birlikte kayan nokta ~0 artık; ekstra kayan >0."""
    from ortak.ego_hareket import artik_hareket_px
    H = np.array([[1, 0, -10.0], [0, 1, -5.0], [0, 0, 1.0]])  # x-10, y-5
    prev = np.array([200.0, 150.0])
    # Sabit nesne: cur tam H kadar kaymış → artık ~0
    cur_sabit = np.array([190.0, 145.0])
    assert artik_hareket_px(H, prev, cur_sabit) < 1e-3
    # Hareketli nesne: H üstüne +20px
    cur_hareketli = np.array([210.0, 145.0])
    assert artik_hareket_px(H, prev, cur_hareketli) > 15.0
    print("test_artik_hareket PASS")


def test_inis_doluluk():
    from gorev1_nesne_tespiti.inis import inis_durumu_belirle
    W, H = 1920, 1080
    alan = (900.0, 500.0, 1000.0, 600.0)        # tam karede, kenara değmiyor
    # Engel yok → uygun (1)
    assert inis_durumu_belirle(alan, [], W, H) == 1
    # Daire içine düşen engel → uygun değil (0)
    engel = (940.0, 540.0, 960.0, 560.0)        # merkez (950,550) alan merkezine yakın
    assert inis_durumu_belirle(alan, [engel], W, H) == 0
    # Kenara değen alan → uygun değil (0)
    alan_kenar = (0.0, 500.0, 100.0, 600.0)
    assert inis_durumu_belirle(alan_kenar, [], W, H) == 0
    print("test_inis_doluluk PASS")


def test_g1_model_import():
    """g1_model torch olmadan import edilebilmeli (ağır bağımlılık tembel)."""
    import importlib
    m = importlib.import_module("gorev1_nesne_tespiti.entegrasyon.g1_model")
    assert hasattr(m, "Gorev1Modeli") and hasattr(m, "to_detected_objects")
    print("test_g1_model_import PASS")


def run_all():
    tests = [test_siniflar, test_ego_homografi, test_artik_hareket,
             test_inis_doluluk, test_g1_model_import]
    failed = 0
    for t in tests:
        try:
            t()
        except Exception as e:
            failed += 1
            print(f"{t.__name__} FAIL: {e}")
    print(f"\n{len(tests)-failed}/{len(tests)} test geçti.")
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if run_all() else 1)
