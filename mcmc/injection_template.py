"""Utilities for loading the fixed injected w(z) history."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paths import DATA  # noqa: E402

N_W = 29
DEFAULT_INJECTION_TEMPLATE = DATA / "gw_fiducial_w_bins.txt"


def load_injected_template(
    path: Path = DEFAULT_INJECTION_TEMPLATE,
    *,
    n_bins: int = N_W,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return injected w_i, z_min and z_max for the requested bins."""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"injected w(z) template not found: {path}")
    rows = np.loadtxt(path, comments="#", ndmin=2)
    if rows.shape[0] < n_bins or rows.shape[1] < 5:
        raise ValueError(
            f"injected w(z) template must have at least {n_bins} rows and 5 columns: {path}"
        )
    rows = rows[:n_bins]
    z_min = rows[:, 1].astype(float)
    z_max = rows[:, 2].astype(float)
    w_injected = rows[:, 4].astype(float)
    if not all(np.all(np.isfinite(values)) for values in (z_min, z_max, w_injected)):
        raise ValueError(f"injected w(z) template contains non-finite values: {path}")
    return w_injected, z_min, z_max
