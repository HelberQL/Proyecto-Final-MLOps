"""Utilidades compartidas: carga de configuracion y logging."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import yaml


def load_config(path: str | Path = "config.yaml") -> Dict:
    """Carga el archivo YAML de configuracion."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontro el archivo de configuracion: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging(level: int = logging.INFO) -> None:
    """Configura el logger raiz con un formato uniforme."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
