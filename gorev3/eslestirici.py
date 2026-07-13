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
`sistem` tarafında `ReferencePrediction`a (referansın kendi url'i + kare url'i +
bbox) çevrilir.
"""
from __future__ import annotations

import contextlib
import io
import logging
import os
import re
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import yaml

from .dino_backbone import build_backbone
from .template_matcher import TemplateMatcher

_CONFIG = Path(__file__).resolve().parent / "template_config.yaml"


def _frame_id_cek(url):
    """'.../frames/165/' -> 165 (int). Bulunamazsa None."""
    if not url:
        return None
    try:
        return int(str(url).rstrip("/").split("/")[-1])
    except (ValueError, AttributeError):
        return None


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
        # Referans arama kare araliklari: object_id -> (frame_start_id, frame_end_id).
        # Referans SADECE bu aralikta aranir (aralik disi yanlis pozitif olmaz).
        self.ref_araliklar = {}
        # Ayni referansi (order) iki kez kaydetmeyi onle (idempotent).
        self.kayitli_orderlar = set()
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
    def register_reference(self, image_bgr: np.ndarray, object_id: int,
                           frame_range: Optional[tuple] = None) -> None:
        """Tek bir referans nesneyi kaydet (BGR girer, RGB'ye çevrilir; istenirse
        arka planı rembg ile temizlenir). `frame_range=(start_id, end_id)` verilirse
        bu referans yalnızca o kare aralığında aranır."""
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        if self.remove_bg:
            rgb = self._arka_plan_temizle(rgb)
        with _sessiz():
            self.matcher.add_reference([rgb], object_id=object_id)
        if frame_range is not None:
            self.ref_araliklar[object_id] = frame_range
        self.n_ref += 1
        logging.info(f"Gorev3: referans {object_id} kaydedildi (aralik={frame_range}, "
                     f"toplam {self.n_ref}).")

    def register_from_server(self, references: list, download: Optional[Callable] = None,
                             ref_image_paths: Optional[dict] = None) -> int:
        """Sunucudan (`GET /reference/`) gelen referans girdilerini kaydet.

        Gerçek sunucu formatı: her referans bir sözlük —
          `url`        : referansın API url'i (submission'da `reference` alanı),
          `image_url`  : referans görselinin media yolu,
          `order`      : referans sırası/kimliği (1, 2, ...),
          `frame_start`/`frame_end` : referansın aranacağı kare aralığı.
        Görsel önceliği: `ref_image_paths[ref['url']]` (main.py oturum başında toptan
        indirir) → yoksa `download(image_url)` → yoksa yerel `image_url`. Aynı `order`
        ikinci kez gelirse ATLANIR (idempotent). Hata olursa o referans atlanır (Görev 3
        çökmez). Kaydedilen YENİ referans sayısı döner."""
        if not references:
            return 0
        n = 0
        for i, ref in enumerate(references):
            try:
                if not isinstance(ref, dict):
                    continue
                oid = int(ref.get("order", ref.get("object_id", ref.get("id", i))))
                if oid in self.kayitli_orderlar:
                    continue  # bu referans zaten kayitli
                # Referans API url'i (submission) ile görsel media yolu AYRIDIR.
                ref_url = ref.get("url") or ref.get("reference_url")
                img_url = ref.get("image_url") or ref_url
                fr = (_frame_id_cek(ref.get("frame_start")), _frame_id_cek(ref.get("frame_end")))
                yerel = ref_image_paths.get(ref_url) if (ref_image_paths and ref_url) else None
                if yerel:
                    img = cv2.imread(str(yerel))          # main'in önceden indirdiği görsel
                elif download and img_url:
                    img = download(img_url)               # auth'lu indir
                elif img_url:
                    img = cv2.imread(str(img_url))        # yerel yol
                else:
                    img = None
                if img is None:
                    logging.error(f"Gorev3: referans order={oid} görseli okunamadı ({img_url}).")
                    continue
                self.register_reference(img, object_id=oid,
                                        frame_range=fr if fr[0] is not None else None)
                self.kayitli_orderlar.add(oid)
                n += 1
            except Exception as e:
                logging.error(f"Gorev3: referans {i} kaydı hatası: {e}")
        if n:
            logging.info(f"Gorev3: {n} yeni referans kaydedildi (toplam {self.n_ref}).")
        return n

    # ── Eşleme ──────────────────────────────────────────────────────────────
    def match(self, frame_bgr: np.ndarray, frame_id: Optional[int] = None,
              active_ids: Optional[set] = None) -> list[dict]:
        """Kareyi referanslarla eşle; şema sözlükleri döndür. Referans yoksa [].

        `active_ids` verilirse (main.py'nin bu kare için pencere-içi seçtiği
        referansların `order`'ları) SADECE o referanslar aranır — en güvenilir kapı,
        çünkü pencere kontrolü main'de resmî `frame_start/end_image_url` ile yapılır.
        Verilmezse `frame_id` aralığına, o da yoksa tüm referanslara düşülür."""
        if self.n_ref == 0:
            return []
        tum = self.matcher.references
        secili = tum
        if active_ids is not None:
            secili = [r for r in tum if r.object_id in active_ids]
        elif frame_id is not None and self.ref_araliklar:
            aktif = {oid for oid, (a, b) in self.ref_araliklar.items()
                     if a is None or b is None or a <= frame_id <= b}
            # aralığı olmayan referanslar her zaman aktif
            aralikli = set(self.ref_araliklar)
            secili = [r for r in tum if r.object_id in aktif or r.object_id not in aralikli]
        if not secili:
            return []
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self.matcher.references = secili
        try:
            with _sessiz():
                dets = self.matcher.match(rgb)
        finally:
            self.matcher.references = tum   # her durumda eski hâle döndür
        out = []
        for d in dets:
            x1, y1, x2, y2 = d.to_xyxy()
            out.append({"object_id": int(d.object_id),
                        "top_left_x": int(x1), "top_left_y": int(y1),
                        "bottom_right_x": int(x2), "bottom_right_y": int(y2)})
        return out
