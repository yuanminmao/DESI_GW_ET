#!/usr/bin/env python3
"""Replace machine-specific paths in committed JSON and CSV metadata."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "results" / "tables"
REPOSITORY_MARKERS = (
    "/results/",
    "/data/",
    "/catalog/",
    "/cosmology/",
    "/mcmc/",
    "/plot/",
    "/paper/",
)


def normalize_string(value: str) -> str:
    """Return a repository-relative form of a machine-specific path."""
    if not value.startswith(("/", "\\")):
        return value
    for marker in REPOSITORY_MARKERS:
        if marker in value:
            relative = value.split(marker, 1)[1]
            normalized = f"{marker.strip('/')}/{relative}"
            return normalized.replace(
                "data/gw170817/early_background_table.npz",
                "data/early_background_table.npz",
            )
    return value


def normalize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: normalize_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize_json(item) for item in value]
    if isinstance(value, str):
        return normalize_string(value)
    return value


def rewrite_json(path: Path) -> bool:
    original = json.loads(path.read_text(encoding="utf-8"))
    normalized = normalize_json(original)
    if normalized == original:
        return False
    path.write_text(
        json.dumps(normalized, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return True


def rewrite_csv(path: Path) -> bool:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    normalized = [[normalize_string(cell) for cell in row] for row in rows]
    if normalized == rows:
        return False
    with path.open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream, lineterminator="\n").writerows(normalized)
    return True


def main() -> None:
    changed: list[Path] = []
    for path in sorted(TABLES.glob("*.json")):
        if rewrite_json(path):
            changed.append(path)
    for path in sorted(TABLES.glob("*.csv")):
        if rewrite_csv(path):
            changed.append(path)
    print(f"Normalized {len(changed)} metadata files.")


if __name__ == "__main__":
    main()
