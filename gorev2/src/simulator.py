"""Yarışma değerlendirme sunucusunun (THYZDS) yerel taklidi — tam prova için.

sistem/ (canlı bağlantı arayüzü) katmanının kullandığı uçları birebir sunar:

* ``POST /auth/``          → ``{"token": ...}``
* ``GET  /frames/``        → kare listesi (url, image_url, video_name, session)
* ``GET  /translation/``   → konum listesi (translation_x/y/z, health_status)
* ``GET  /media/...jpg``   → kare görseli (videodan anlık JPEG kodlanır, disk şişmez)
* ``POST /prediction/``    → tahmini zaman damgasıyla JSONL dosyasına kaydeder

Kesinti çizelgesi ``evaluation.make_outage_schedule`` ile üretilir (yarışmayla
aynı istatistik); sağlıksız karelerde konumlar ``"NaN"`` ve ``health_status=0``
gönderilir — gerçek sunucu davranışı (teknik şartname Şekil 16).

Kullanım (gorev2 klasöründen):
    python -m src.simulator --video ..\\THYZ_2026_Ornek_Veri_Seti\\THYZ_2026_Ornek_Veri_1.MP4 ^
        --csv ..\\THYZ_2026_Ornek_Veri_Seti\\THYZ_2026_Ornek_Veri_1_translation.csv ^
        --seed 42 --port 5000
"""
from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import cv2
import numpy as np

from . import data_io
from .evaluation import make_outage_schedule

CIKTI_DIR = Path(__file__).resolve().parent.parent / "outputs" / "sim_kayitlari"


class SimState:
    """Sunucunun servis edeceği veri + kayıt dosyaları."""

    def __init__(self, video, csv, seed, port, limit=None, healthy_prefix=450):
        self.video_path = Path(video)
        self.video_name = f"sim_{self.video_path.stem}"
        self.port = port

        gt = data_io.load_translations(csv)
        sub = data_io.subsample_competition(gt).reset_index(drop=True)
        if limit:
            sub = sub.iloc[:limit].reset_index(drop=True)
        self.kept_idx = sub["frame_idx"].to_numpy()
        self.gt = sub[["translation_x", "translation_y", "translation_z"]].to_numpy()
        self.healthy = make_outage_schedule(
            n_frames=len(sub), healthy_prefix=healthy_prefix, rng=seed)

        base = f"http://127.0.0.1:{port}"
        self.frames_json = [
            dict(url=f"{base}/frames/{i}/",
                 image_url=f"/{self.video_name}/frame_{int(idx):06d}.jpg",
                 video_name=self.video_name,
                 session=f"{base}/session/1/")
            for i, idx in enumerate(self.kept_idx)
        ]
        self.translations_json = []
        for i in range(len(sub)):
            if self.healthy[i]:
                t = dict(translation_x=float(self.gt[i, 0]),
                         translation_y=float(self.gt[i, 1]),
                         translation_z=float(self.gt[i, 2]),
                         health_status=1)
            else:  # gerçek sunucu: NaN + 0 (Şekil 16)
                t = dict(translation_x="NaN", translation_y="NaN",
                         translation_z="NaN", health_status=0)
            self.translations_json.append(t)

        # kayıt dosyaları
        CIKTI_DIR.mkdir(parents=True, exist_ok=True)
        self.pred_path = CIKTI_DIR / f"predictions_seed{seed}.jsonl"
        self.pred_path.write_text("", encoding="utf-8")
        manifest = dict(video=str(self.video_path), video_name=self.video_name,
                        seed=seed, n_frames=len(sub),
                        kept_idx=self.kept_idx.tolist(),
                        healthy=self.healthy.astype(int).tolist(),
                        gt=self.gt.tolist())
        (CIKTI_DIR / f"manifest_seed{seed}.json").write_text(
            json.dumps(manifest), encoding="utf-8")

        # görüntü sunumu için video okuyucu (istekler sıralı gelir)
        self._cap = cv2.VideoCapture(str(self.video_path))
        self._son_idx = -1

    def frame_jpg(self, idx: int) -> bytes:
        """İstenen native kare indeksini anlık JPEG olarak kodla."""
        if idx <= self._son_idx:                       # geri sarma: seek
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            self._son_idx = idx - 1
        while self._son_idx < idx - 1:                 # ileri: ucuz grab
            self._cap.grab()
            self._son_idx += 1
        ok, frame = self._cap.read()
        self._son_idx = idx
        if not ok:
            raise IOError(f"kare okunamadi: {idx}")
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return buf.tobytes()


def make_handler(state: SimState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):                  # sessiz
            pass

        def _json(self, obj, status=200):
            veri = json.dumps(obj).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(veri)))
            self.end_headers()
            self.wfile.write(veri)

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            govde = self.rfile.read(n).decode("utf-8") if n else ""
            if self.path.startswith("/auth"):
                self._json({"token": "sim-token"})
            elif self.path.startswith("/prediction"):
                kayit = dict(zaman=time.time(), payload=json.loads(govde))
                with open(state.pred_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(kayit) + "\n")
                self._json({"detail": "kaydedildi"}, status=201)
            else:
                self._json({"detail": "bilinmeyen uc"}, status=404)

        def do_GET(self):
            if self.path.startswith("/frames"):
                self._json(state.frames_json)
            elif self.path.startswith("/translation"):
                self._json(state.translations_json)
            elif self.path.startswith("/media/"):
                # /media/<video_name>/frame_XXXXXX.jpg
                try:
                    idx = int(Path(self.path).stem.split("_")[-1])
                    veri = state.frame_jpg(idx)
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(veri)))
                    self.end_headers()
                    self.wfile.write(veri)
                except Exception as e:
                    self._json({"detail": str(e)}, status=500)
            else:
                self._json({"detail": "bilinmeyen uc"}, status=404)

    return Handler


def main():
    ap = argparse.ArgumentParser(description="Yerel yarisma sunucusu taklidi")
    ap.add_argument("--video", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--limit", type=int, default=None, help="kare sayisi siniri (kisa test)")
    ap.add_argument("--prefix", type=int, default=450, help="garantili saglikli on ek")
    args = ap.parse_args()

    state = SimState(args.video, args.csv, args.seed, args.port,
                     limit=args.limit, healthy_prefix=args.prefix)
    sunucu = HTTPServer(("127.0.0.1", args.port), make_handler(state))
    kayip = 100 * (1 - state.healthy.mean())
    print(f"simulator hazir: http://127.0.0.1:{args.port}/  "
          f"({len(state.frames_json)} kare, seed {args.seed}, kayip %{kayip:.0f})")
    print(f"tahmin kaydi: {state.pred_path}")
    sunucu.serve_forever()


if __name__ == "__main__":
    main()
