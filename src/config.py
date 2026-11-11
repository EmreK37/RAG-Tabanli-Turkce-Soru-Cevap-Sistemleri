"""Deney konfigürasyonu yükleyici.

Tek doğruluk kaynağı: configs/deney.yaml. Kod içinde sabit deney
değeri tutulmaz; her şey buradan okunur.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG: Path = REPO_ROOT / "configs" / "deney.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """YAML konfigürasyonunu doğrular ve sözlük olarak döndürür."""
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG
    if not cfg_path.exists():
        raise FileNotFoundError(f"Konfigürasyon dosyası bulunamadı: {cfg_path}")

    with open(cfg_path, encoding="utf-8-sig") as f:
        icerik = yaml.safe_load(f)

    if not isinstance(icerik, dict):
        raise ValueError(f"Geçersiz veya boş YAML konfigürasyonu: {cfg_path}")
    return icerik


def model_configs(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Konfigürasyondaki model tanımlarını döndürür."""
    modeller = cfg.get("modeller", [])
    if not isinstance(modeller, list):
        raise ValueError("Konfigürasyondaki 'modeller' alanı bir liste olmalıdır.")
    return modeller

