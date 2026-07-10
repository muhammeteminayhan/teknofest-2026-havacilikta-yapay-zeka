"""Rapor çıktıları: figürlerin ``gorev2/reports/`` altına kaydedilmesi.

Her aşamanın defteri, ürettiği figürleri ``save_fig`` ile kaydeder;
dosya adları ``<defter-no>_<açıklayıcı-ad>.png`` düzenindedir.
"""
from __future__ import annotations

from pathlib import Path

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


def save_fig(fig, name: str, dpi: int = 150) -> Path:
    """Figürü reports/ altına PNG olarak kaydet ve yolu yazdır."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"figur kaydedildi: {path.name}")
    return path
