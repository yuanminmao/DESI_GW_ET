"""Shared project paths (import from any module)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
CATALOGS = RESULTS / "catalogs"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"
CHAINS = ROOT / "results" / "chains"


def chain_root(name: str) -> Path:
    """MCMC chain prefix, e.g. chain_root('GW_ET') -> .../chains/GW_ET/GW_ET."""
    return CHAINS / name / name


def ensure_result_dirs() -> None:
    for d in (CATALOGS, FIGURES, TABLES, CHAINS):
        d.mkdir(parents=True, exist_ok=True)


def setup_sys_path() -> Path:
    """Add project packages to sys.path for flat imports (gw_mcmc, plot_gw_de_fom, ...)."""
    for sub in (ROOT, ROOT / "mcmc", ROOT / "cosmology", ROOT / "catalog", ROOT / "plot"):
        s = str(sub)
        if s not in sys.path:
            sys.path.insert(0, s)
    return ROOT
