"""Görev 1 — Nesne Tespiti paketi.

Vanilla YOLOv8s modeli (`agirliklar/best.pt`) ile taşıt / insan / iniş alanı
(UAP-UAİ) tespiti. Standart `ultralytics` kullanır — özel fork YOKTUR, çevrimdışı
indirme tuzağı YOKTUR (best.pt tek başına yüklenir).

Kullanım:
    from gorev1 import Gorev1Detector
    det = Gorev1Detector()
    nesneler = det.detect(frame_bgr)   # list[dict], şartname şeması
"""
from pathlib import Path

BEST_PT = Path(__file__).resolve().parent / "agirliklar" / "best.pt"

from .dedektor import Gorev1Detector  # noqa: E402

__all__ = ["Gorev1Detector", "BEST_PT"]
