"""İndirme yardımcıları (ilerleme çubuklu HTTP indirme + zip açma)."""
from __future__ import annotations

import os
import zipfile


def indir(url: str, hedef: str, parca: int = 1 << 20) -> str:
    """url'yi hedef dosyaya indir (varsa atla). Döndürür: hedef yol."""
    import requests
    if os.path.exists(hedef) and os.path.getsize(hedef) > 0:
        print(f"  zaten var, atlanıyor: {hedef}")
        return hedef
    os.makedirs(os.path.dirname(hedef), exist_ok=True)
    print(f"  indiriliyor: {url}")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        toplam = int(r.headers.get("content-length", 0))
        inen = 0
        with open(hedef + ".part", "wb") as f:
            for chunk in r.iter_content(parca):
                f.write(chunk)
                inen += len(chunk)
                if toplam:
                    print(f"\r  {inen/1e6:.0f}/{toplam/1e6:.0f} MB", end="", flush=True)
        print()
    os.replace(hedef + ".part", hedef)
    return hedef


def zip_ac(zip_yolu: str, hedef_dir: str) -> None:
    print(f"  açılıyor: {zip_yolu} -> {hedef_dir}")
    with zipfile.ZipFile(zip_yolu) as z:
        z.extractall(hedef_dir)
