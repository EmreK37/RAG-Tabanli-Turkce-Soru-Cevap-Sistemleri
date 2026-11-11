"""Pytest çalışma yoluna proje kök dizinini ekler."""
from __future__ import annotations

import sys
from pathlib import Path

PROJE_KOKU = str(Path(__file__).resolve().parent.parent)
if PROJE_KOKU not in sys.path:
    sys.path.insert(0, PROJE_KOKU)


