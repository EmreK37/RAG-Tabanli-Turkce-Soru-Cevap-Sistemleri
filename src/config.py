"""Deney konfigürasyonu yükleyici.

Tek doğruluk kaynağı: configs/deney.yaml. Kod içinde sabit deney
değeri tutulmaz; her şey buradan okunur.
"""
from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "deney.yaml"


def load_config(path: str | Path | None = None) -> dict:
    """YAML konfigürasyonunu sözlük olarak döndürür."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with open(cfg_path, encoding="utf-8-sig") as f:
        return yaml.safe_load(f)


def model_configs(cfg: dict) -> list[dict]:
    """Konfigürasyondaki model tanımlarını döndürür."""
    return cfg.get("modeller", [])
