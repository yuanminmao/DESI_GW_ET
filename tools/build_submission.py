#!/usr/bin/env python3
"""Build a deterministic, self-contained journal-source ZIP archive."""

from __future__ import annotations

import re
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"
OUTPUT = ROOT / "dist" / "paper-source.zip"
FIXED_TIMESTAMP = (2026, 1, 1, 0, 0, 0)


def referenced_figures() -> list[Path]:
    source = (PAPER / "main.tex").read_text(encoding="utf-8")
    names = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", source)
    return [PAPER / "figures" / name for name in names]


def referenced_inputs() -> list[Path]:
    source = (PAPER / "main.tex").read_text(encoding="utf-8")
    names = re.findall(r"\\input\{([^}]+)\}", source)
    paths = [(PAPER / name).resolve() for name in names]
    if any(PAPER.resolve() not in path.parents for path in paths):
        raise ValueError("Manuscript input points outside paper/")
    return paths


def add_file(archive: zipfile.ZipFile, source: Path, name: str) -> None:
    info = zipfile.ZipInfo(name, date_time=FIXED_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    archive.writestr(info, source.read_bytes())


def main() -> None:
    files = [PAPER / "main.tex", *referenced_inputs()]
    figures = referenced_figures()
    missing = [path for path in files + figures if not path.is_file()]
    if missing:
        formatted = "\n".join(f"  {path}" for path in missing)
        raise FileNotFoundError(f"Submission inputs are missing:\n{formatted}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUTPUT, "w") as archive:
        for path in files:
            add_file(archive, path, str(path.relative_to(PAPER)))
        for path in figures:
            add_file(archive, path, f"figures/{path.name}")
    print(f"Created: {OUTPUT}")


if __name__ == "__main__":
    main()
