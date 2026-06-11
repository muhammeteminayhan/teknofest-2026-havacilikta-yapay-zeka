"""
Kamera kalibrasyonu — MATLAB cameraIntrinsics formatını ayrıştırıp OpenCV
kamera matrisi (K) ve distorsiyon katsayılarına dönüştürür.

Kalibrasyon dosyası formatı (Kamera_Kalibrasyon_Parametreleri_2026.txt):

    Termal Camera Intrinsics
                     FocalLength: [731.7965 732.0172]
                  PrincipalPoint: [319.2367 251.2424]
                RadialDistortion: [-0.3507 0.1137]
            TangentialDistortion: [0 0]
                       ImageSize: [512 640]

Notlar (MATLAB -> OpenCV):
- FocalLength = [fx fy] (piksel).
- PrincipalPoint = [cx cy] (piksel, 1-tabanlı). OpenCV 0-tabanlı olduğundan
  -1 ofset uygulanır (config.MATLAB_TO_OPENCV_PP_OFFSET).
- ImageSize = [satır sütun] = [H W]  (DİKKAT: yükseklik önce gelir).
- RadialDistortion = [k1 k2 (k3)], TangentialDistortion = [p1 p2].
- OpenCV dist = [k1, k2, p1, p2, k3].
- K matrisini fx/fy/cx/cy'den doğrudan kurarız; MATLAB IntrinsicMatrix'in
  transpoze sorunu bu yüzden bizi etkilemez.
"""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs"))
import config  # noqa: E402


@dataclass
class CameraIntrinsics:
    """Tek bir kameranın intrinsics + distorsiyon bilgisi (OpenCV uyumlu)."""

    name: str
    fx: float
    fy: float
    cx: float
    cy: float
    dist: np.ndarray          # [k1, k2, p1, p2, k3]
    width: int                # kalibrasyon görüntü genişliği (px)
    height: int               # kalibrasyon görüntü yüksekliği (px)

    @property
    def K(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx],
             [0.0, self.fy, self.cy],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def scaled_to(self, width: int, height: int) -> "CameraIntrinsics":
        """Intrinsics'i hedef çözünürlüğe ölçekle (yeniden boyutlandırılmış kare için)."""
        sx = width / float(self.width)
        sy = height / float(self.height)
        return CameraIntrinsics(
            name=f"{self.name}@{width}x{height}",
            fx=self.fx * sx,
            fy=self.fy * sy,
            cx=self.cx * sx,
            cy=self.cy * sy,
            dist=self.dist.copy(),
            width=width,
            height=height,
        )

    def __repr__(self) -> str:
        return (f"CameraIntrinsics({self.name}: fx={self.fx:.2f} fy={self.fy:.2f} "
                f"cx={self.cx:.2f} cy={self.cy:.2f} size={self.width}x{self.height} "
                f"dist={np.round(self.dist, 4).tolist()})")


def _grab_vec(block: str, key: str) -> list[float]:
    """`Key: [a b c]` desenini blok içinden ayıkla."""
    m = re.search(rf"{key}\s*:\s*\[([^\]]*)\]", block)
    if not m:
        return []
    raw = m.group(1).replace(",", " ")
    return [float(x) for x in raw.split()]


def parse_calibration_file(path: str | None = None) -> dict[str, CameraIntrinsics]:
    """
    Kalibrasyon txt dosyasını ayrıştır → {anahtar: CameraIntrinsics}.

    Anahtarlar: 'thermal', 'rgb_4k', 'rgb_1080p' (başlık metninden türetilir).
    """
    if path is None:
        path = config.CALIB_FILE
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    # "<Başlık> Intrinsics" satırlarına göre blokları böl
    headers = list(re.finditer(r"(?im)^[ \t]*([A-Za-z0-9 .]+?)\s+Intrinsics\s*$", text))
    cams: dict[str, CameraIntrinsics] = {}
    for i, h in enumerate(headers):
        title = h.group(1).strip()
        start = h.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        block = text[start:end]

        focal = _grab_vec(block, "FocalLength")
        pp = _grab_vec(block, "PrincipalPoint")
        radial = _grab_vec(block, "RadialDistortion")
        tangential = _grab_vec(block, "TangentialDistortion")
        imsize = _grab_vec(block, "ImageSize")  # [H, W]
        if not (focal and pp and imsize):
            continue

        fx, fy = focal[0], focal[1] if len(focal) > 1 else focal[0]
        cx, cy = pp[0], pp[1]
        if config.MATLAB_TO_OPENCV_PP_OFFSET:
            cx -= 1.0
            cy -= 1.0
        k1 = radial[0] if len(radial) > 0 else 0.0
        k2 = radial[1] if len(radial) > 1 else 0.0
        k3 = radial[2] if len(radial) > 2 else 0.0
        p1 = tangential[0] if len(tangential) > 0 else 0.0
        p2 = tangential[1] if len(tangential) > 1 else 0.0
        dist = np.array([k1, k2, p1, p2, k3], dtype=np.float64)

        height, width = int(round(imsize[0])), int(round(imsize[1]))

        key = _title_to_key(title)
        cams[key] = CameraIntrinsics(key, fx, fy, cx, cy, dist, width, height)
    return cams


def _title_to_key(title: str) -> str:
    t = title.lower()
    if "termal" in t or "thermal" in t:
        return "thermal"
    if "4k" in t:
        return "rgb_4k"
    if "1080" in t:
        return "rgb_1080p"
    return re.sub(r"[^a-z0-9]+", "_", t).strip("_")


def intrinsics_for_resolution(
    cams: dict[str, CameraIntrinsics],
    width: int,
    height: int,
    modality: str = "rgb",
) -> CameraIntrinsics:
    """
    Gerçek kare çözünürlüğüne en uygun kalibrasyonu seç ve gerekiyorsa ölçekle.

    En-boy oranı en yakın olan kalibrasyon seçilir; piksel boyutu farklıysa
    intrinsics doğrusal ölçeklenir. (4K video bazen 3840x2160 olarak gelebilir;
    kalibrasyon 4000x3000 olduğundan ölçekleme/oran kontrolü önemlidir.)
    """
    if modality == "thermal":
        candidates = {k: v for k, v in cams.items() if k == "thermal"} or cams
    else:
        candidates = {k: v for k, v in cams.items() if k.startswith("rgb")} or cams

    target_ar = width / float(height)
    best_key = min(
        candidates,
        key=lambda k: abs((candidates[k].width / candidates[k].height) - target_ar),
    )
    base = candidates[best_key]
    if (base.width, base.height) == (width, height):
        return base
    return base.scaled_to(width, height)


class CameraModel:
    """
    Bir kameranın intrinsics + distorsiyonunu tutar; kare/nokta undistort eder.

    undistort haritası (remap) ilk kullanımda hesaplanıp önbelleğe alınır.
    """

    def __init__(self, intr: CameraIntrinsics):
        self.intr = intr
        self.K = intr.K
        self.dist = intr.dist
        # Undistort sonrası optimal yeni kamera matrisi (alpha=0: siyah kenar yok)
        import cv2
        self.new_K, self._roi = cv2.getOptimalNewCameraMatrix(
            self.K, self.dist, (intr.width, intr.height), 0, (intr.width, intr.height)
        )
        self._map1 = None
        self._map2 = None

    def _ensure_maps(self):
        if self._map1 is None:
            import cv2
            self._map1, self._map2 = cv2.initUndistortRectifyMap(
                self.K, self.dist, None, self.new_K,
                (self.intr.width, self.intr.height), cv2.CV_16SC2,
            )

    def undistort_image(self, img: np.ndarray) -> np.ndarray:
        """Kareyi undistort et (kenarlar kırpılmaz; new_K geçerli olur)."""
        import cv2
        self._ensure_maps()
        return cv2.remap(img, self._map1, self._map2, cv2.INTER_LINEAR)

    def undistort_points(self, pts: np.ndarray) -> np.ndarray:
        """
        Nx2 piksel noktalarını undistort et → new_K çerçevesinde Nx2 piksel.
        (Özellik takibi distorsiyonlu görüntüde yapıldıysa noktaları düzeltmek için.)
        """
        import cv2
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
        out = cv2.undistortPoints(pts, self.K, self.dist, P=self.new_K)
        return out.reshape(-1, 2)


def load_camera(modality: str = "rgb", width: int | None = None, height: int | None = None) -> CameraModel:
    """Kalibrasyonu yükle ve (varsa) hedef çözünürlüğe göre CameraModel döndür."""
    cams = parse_calibration_file()
    if width and height:
        intr = intrinsics_for_resolution(cams, width, height, modality)
    else:
        key = "thermal" if modality == "thermal" else "rgb_1080p"
        intr = cams.get(key) or next(iter(cams.values()))
    return CameraModel(intr)


if __name__ == "__main__":
    # Hızlı kontrol: kalibrasyonu ayrıştır ve yazdır
    cams = parse_calibration_file()
    print(f"Ayrıştırılan kameralar: {list(cams.keys())}\n")
    for k, v in cams.items():
        print(v)
        print("  K =\n", np.round(v.K, 2))
