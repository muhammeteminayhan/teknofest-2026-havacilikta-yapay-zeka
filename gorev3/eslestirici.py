"""Görev 3 — Görüntü Eşleme: durumlu sarmalayıcı.

Oturum başında referans nesneler kaydedilir (`register_reference` /
`register_from_server`), sonra her karede `match(frame_bgr)` çağrılır. DINO patch
özellikleri üzerinden şablon eşleme (kosinüs benzerliği + FPN çok-ölçek) yapan
`TemplateMatcher` (fork'tan kopya, değiştirilmedi) kullanılır.

Omurga tercihi:
  - VARSAYILAN "dinov3" (HuggingFace, önbellekten çevrimdışı).
  - Görev 1 vanilla YOLOv8s olduğu için paylaşılacak DINO omurgası YOK; Görev 3
    kendi omurgasını yükler (~0,1 GB VRAM). İstenirse "dinov2"ye tek satırla geçilir.

Çıktı: her eşleşme için {object_id, top_left_x/y, bottom_right_x/y} sözlüğü;
`sistem` tarafında `DetectedUndefinedObject`e çevrilir.
"""
from __future__ import annotations

import contextlib
import io
import logging
import os
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import yaml

from .dino_backbone import build_backbone
from .template_matcher import TemplateMatcher

_CONFIG = Path(__file__).resolve().parent / "template_config.yaml"


@contextlib.contextmanager
def _sessiz():
    """TemplateMatcher'ın print gürültüsünü yut (yarışma logunu temiz tutar)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield


class Gorev3Matcher:
    def __init__(self, kind: str = "dinov3", backbone=None, config_path: Optional[str] = None,
                 remove_bg: bool = True):
        with open(config_path or _CONFIG, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        if backbone is None:
            backbone = build_backbone(kind)
        with _sessiz():
            self.matcher = TemplateMatcher(backbone, cfg)
        self.n_ref = 0
        # Referans arka planini rembg ile temizle (nesneyi izole eder -> temiz CLS
        # token -> daha iyi esleme). Kapatmak icin remove_bg=False.
        self.remove_bg = remove_bg
        logging.info(f"Gorev3Matcher hazir: omurga={kind}, arka_plan_temizle={remove_bg}")

    @staticmethod
    def _arka_plan_temizle(rgb: np.ndarray) -> np.ndarray:
        """Referans gorselinin arka planini rembg (u2net) ile sil. Basarisiz olursa
        (rembg yok / u2net inememis) orijinali dondur — sistem cokmez."""
        try:
            from rembg import remove
            out = remove(rgb)                       # RGBA (arka plan seffaf/siyah)
            if out.ndim == 3 and out.shape[2] == 4:
                out = cv2.cvtColor(out, cv2.COLOR_RGBA2RGB)
            return out
        except Exception as e:
            logging.warning(f"Gorev3: arka plan temizleme atlandi (rembg/u2net): {e}")
            return rgb

    # ── Referans kaydı ──────────────────────────────────────────────────────
    def register_reference(self, image_bgr: np.ndarray, object_id: int) -> None:
        """Tek bir referans nesneyi kaydet (BGR girer, RGB'ye çevrilir; istenirse
        arka planı rembg ile temizlenir)."""
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        if self.remove_bg:
            rgb = self._arka_plan_temizle(rgb)
        with _sessiz():
            self.matcher.add_reference([rgb], object_id=object_id)
        self.n_ref += 1
        logging.info(f"Gorev3: referans {object_id} kaydedildi (toplam {self.n_ref}).")

    def register_from_server(self, references: list, download: Optional[Callable] = None) -> int:
        """Sunucudan gelen frames-JSON `references` listesinden kayıt yap.

        Şema kesin bilinmediği için savunmacı okuma: her öğede görsel URL'i
        (url/image_url/reference_url) ve id (id/object_id) aranır. `download(url)`
        verilirse görseli indirir (auth gerekebilir); vermezse yerel yol dener.
        Ham yapı loglanır. Hata olursa Görev 3 pasif kalır (0 döner)."""
        if not references:
            logging.info("Gorev3: sunucuda referans yok (references=[]), pasif.")
            return 0
        logging.info(f"Gorev3: {len(references)} referans geldi, ham ornek: {references[0]}")
        n = 0
        for i, ref in enumerate(references):
            try:
                if isinstance(ref, dict):
                    url = ref.get("image_url") or ref.get("url") or ref.get("reference_url")
                    oid = ref.get("object_id", ref.get("id", i))
                else:
                    url, oid = ref, i
                if download is not None and url:
                    img = download(url)
                else:
                    img = cv2.imread(str(url)) if url else None
                if img is None:
                    logging.error(f"Gorev3: referans {i} görseli okunamadı ({url}).")
                    continue
                self.register_reference(img, object_id=int(oid))
                n += 1
            except Exception as e:
                logging.error(f"Gorev3: referans {i} kaydı hatası: {e}")
        return n

    # ── Eşleme ──────────────────────────────────────────────────────────────
    def match(self, frame_bgr: np.ndarray) -> list[dict]:
        """Kareyi referanslarla eşle; şema sözlükleri döndür. Referans yoksa []."""
        if self.n_ref == 0:
            return []
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        with _sessiz():
            dets = self.matcher.match(rgb)
        out = []
        for d in dets:
            x1, y1, x2, y2 = d.to_xyxy()
            out.append({"object_id": int(d.object_id),
                        "top_left_x": int(x1), "top_left_y": int(y1),
                        "bottom_right_x": int(x2), "bottom_right_y": int(y2)})
        return out
