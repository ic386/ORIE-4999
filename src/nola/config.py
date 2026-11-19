"""Paths and YAML config loading. Everything resolves relative to the repo root."""

from __future__ import annotations

from functools import cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"

DATA = ROOT / "data"
RAW = DATA / "raw"              # straight from Socrata (parquet)
METADATA = RAW / "metadata"     # Socrata view JSON = data dictionaries
INTERIM = DATA / "interim"      # cleaned, one file per source
PROCESSED = DATA / "processed"  # joined + H3-indexed + analysis tables
EXTERNAL = DATA / "external"    # hand-collected lists (second lines, permits), ACS/TIGER

OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"
MAPS = OUTPUTS / "maps"

ALL_DIRS = [RAW, METADATA, INTERIM, PROCESSED, EXTERNAL, FIGURES, TABLES, MAPS]


def ensure_dirs() -> None:
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


@cache
def load(name: str) -> dict:
    """Load config/<name>.yaml."""
    with open(CONFIG_DIR / "{}.yaml".format(name)) as f:
        return yaml.safe_load(f)


def datasets() -> dict:
    return load("datasets")


def params() -> dict:
    return load("params")


def years() -> list[int]:
    return list(params()["years"])
