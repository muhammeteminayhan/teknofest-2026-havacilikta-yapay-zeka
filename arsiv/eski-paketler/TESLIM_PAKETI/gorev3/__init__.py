"""Görev 3 — Görüntü Eşleme paketi.

Referans nesneleri video karelerinde bulan zero-shot şablon eşleme. DINO patch
özellikleri + FPN çok-ölçek kosinüs benzerliği. Fork'suz: `template_matcher.py`
(bağımsız), `dino_backbone.py` (minimal DINOv3/DINOv2), `eslestirici.py` (sarmalayıcı).

Kullanım:
    from gorev3 import Gorev3Matcher
    m = Gorev3Matcher(kind="dinov3")
    m.register_reference(ref_bgr, object_id=0)
    eslesmeler = m.match(frame_bgr)   # list[dict]
"""
from .eslestirici import Gorev3Matcher

__all__ = ["Gorev3Matcher"]
