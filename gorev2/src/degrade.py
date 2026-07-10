"""Kontrollü görüntü bozulmaları: tema dayanıklılığı testleri için.

Amaç: elimizdeki GT'li videoları bilinen şiddette bozup ("gece", "sis",
"hareket bulanıklığı") yarışma metriğini yeniden ölçmek. GT değişmediği için
sonuç, **bozulma şiddeti ↔ E** eğrisidir — zor temaların nicel provası.

Bozulma modelleri (fiziksel gerekçeleriyle):

* **gece:** gamma sıkıştırma + parlaklık düşürme + sensör gürültüsü.
  Karanlıkta kamera kazancı artar → gürültü belirginleşir; doku kontrastı düşer.
* **sis:** atmosferik saçılma modeli ``I' = I·(1-t) + A·t`` (A: hava ışığı) +
  hafif bulanıklık. Kontrast ve uzak doku kaybolur.
* **blur:** doğrusal hareket bulanıklığı çekirdeği — titreşim/uzun poz benzeri.

Şiddetler: "orta" ve "agir". Üretim ``cv2.VideoWriter`` ile videonun native
FPS'inde yapılır; alt akış (7,5 FPS altörnekleme, ölçüm, değerlendirme)
hiçbir değişiklik gerektirmez.
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

# tema -> siddet -> parametreler
PROFILLER = {
    "gece": {
        "orta": dict(gamma=1.8, parlaklik=0.35, gurultu=6.0),
        "agir": dict(gamma=2.2, parlaklik=0.18, gurultu=12.0),
    },
    "sis": {
        "orta": dict(t=0.45, hava=235.0, blur=3),
        "agir": dict(t=0.68, hava=235.0, blur=5),
    },
    "blur": {
        "orta": dict(uzunluk=9, aci_derece=15.0),
        "agir": dict(uzunluk=17, aci_derece=15.0),
    },
}


# Hiz notu: kare basina tam cozunurluk float islemleri (np.power, rng.normal)
# ~180 ms surer ve uretimi saatlere cikarir. Iki hizlandirma kullanilir:
# 1) Deger-bazli donusumler (gamma, sis karisimi) 256 girisli LUT ile yapilir.
# 2) Sensor gurultusu, onceden uretilmis kucuk bir "gurultu bankasi"ndan
#    dondurulerek eklenir (gorsel olarak ayirt edilemez, ~10x hizli).
_gurultu_bankasi: dict = {}


def _gurultu_al(shape, sigma: float, rng: np.random.Generator, i: int) -> np.ndarray:
    anahtar = (shape, sigma)
    if anahtar not in _gurultu_bankasi:
        _gurultu_bankasi[anahtar] = [
            rng.normal(0.0, sigma, shape).astype(np.int16) for _ in range(16)
        ]
    return _gurultu_bankasi[anahtar][i % 16]


_kare_sayaci = 0


def _gece(frame: np.ndarray, gamma: float, parlaklik: float, gurultu: float,
          rng: np.random.Generator) -> np.ndarray:
    global _kare_sayaci
    x = np.arange(256, dtype=np.float32) / 255.0
    lut = np.clip(np.power(x, gamma) * parlaklik * 255.0, 0, 255).astype(np.uint8)
    f = cv2.LUT(frame, lut)
    if gurultu > 0:
        _kare_sayaci += 1
        g = _gurultu_al(frame.shape, gurultu, rng, _kare_sayaci)
        f = np.clip(f.astype(np.int16) + g, 0, 255).astype(np.uint8)
    return f


def _sis(frame: np.ndarray, t: float, hava: float, blur: int,
         rng: np.random.Generator) -> np.ndarray:
    x = np.arange(256, dtype=np.float32)
    lut = np.clip(x * (1.0 - t) + hava * t, 0, 255).astype(np.uint8)
    f = cv2.LUT(frame, lut)
    if blur > 1:
        f = cv2.GaussianBlur(f, (blur | 1, blur | 1), 0)
    return f


def _blur(frame: np.ndarray, uzunluk: int, aci_derece: float,
          rng: np.random.Generator) -> np.ndarray:
    cekirdek = np.zeros((uzunluk, uzunluk), np.float32)
    cekirdek[uzunluk // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((uzunluk / 2 - 0.5, uzunluk / 2 - 0.5), aci_derece, 1.0)
    cekirdek = cv2.warpAffine(cekirdek, M, (uzunluk, uzunluk))
    cekirdek /= cekirdek.sum()
    return cv2.filter2D(frame, -1, cekirdek)


_FONKSIYONLAR = {"gece": _gece, "sis": _sis, "blur": _blur}


def degrade_frame(frame: np.ndarray, tema: str, siddet: str,
                  rng: np.random.Generator) -> np.ndarray:
    """Tek kareye bozulma uygula."""
    return _FONKSIYONLAR[tema](frame, rng=rng, **PROFILLER[tema][siddet])


def make_degraded_video(
    src_video: str | Path,
    dst_video: str | Path,
    tema: str,
    siddet: str,
    seed: int = 0,
    progress: bool = True,
) -> Path:
    """Videonun bozulmuş kopyasını üret (native FPS korunur, önbellekli).

    Önbellek kontrolü bütünlük doğrular: var olan dosyanın kare sayısı
    kaynağınkiyle eşleşmiyorsa (yarım kalmış/bozuk üretim) yeniden üretilir.
    """
    dst_video = Path(dst_video)
    src_cap = cv2.VideoCapture(str(src_video))
    n_kaynak = int(src_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_cap.release()
    if dst_video.exists():
        c = cv2.VideoCapture(str(dst_video))
        n_var = int(c.get(cv2.CAP_PROP_FRAME_COUNT))
        c.release()
        if n_var == n_kaynak:
            print(f"zaten var ve tam ({n_var} kare): {dst_video.name}")
            return dst_video
        print(f"yarim kalmis ({n_var}/{n_kaynak}), yeniden uretiliyor: {dst_video.name}")
        dst_video.unlink()
    dst_video.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    cap = cv2.VideoCapture(str(src_video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    yazici = cv2.VideoWriter(str(dst_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    pbar = None
    if progress:
        from tqdm import tqdm

        pbar = tqdm(total=n, desc=f"{tema}-{siddet}", unit="kare", mininterval=5)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yazici.write(degrade_frame(frame, tema, siddet, rng))
            if pbar is not None:
                pbar.update(1)
    finally:
        cap.release()
        yazici.release()
        if pbar is not None:
            pbar.close()
    print(f"{dst_video.name}: {time.perf_counter() - t0:.0f} s")
    return dst_video
