"""Flat LambdaCDM cosmology: luminosity distance for GW inference."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paths import setup_sys_path

setup_sys_path()

from gw_cosmology import C_KM_S


def e_squared_lcdm(z: np.ndarray, omega_m: float) -> np.ndarray:
    """E(z)^2 for flat LambdaCDM."""
    z = np.asarray(z, dtype=float)
    omega_de = 1.0 - float(omega_m)
    return omega_m * (1.0 + z) ** 3 + omega_de


class LCDMCosmology:
    """Pre-tabulated luminosity distance d_L(z) [Mpc] for LCDM likelihood calls."""

    def __init__(
        self,
        *,
        omega_m: float = 0.315,
        z_max: float = 3.0,
        n_grid: int = 4000,
    ) -> None:
        self.omega_m = float(omega_m)
        self.z_max = float(z_max)
        self.z_grid = np.linspace(0.0, z_max, n_grid)

    def luminosity_distance(
        self,
        z: np.ndarray,
        h0: float,
        *,
        omega_m: float | None = None,
    ) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        om = self.omega_m if omega_m is None else float(omega_m)
        ez = np.sqrt(np.maximum(e_squared_lcdm(self.z_grid, om), 1e-30))
        chi = (C_KM_S / h0) * cumulative_trapezoid(1.0 / ez, self.z_grid, initial=0.0)
        d_l_grid = (1.0 + self.z_grid) * chi
        interp = interp1d(
            self.z_grid,
            d_l_grid,
            kind="cubic",
            bounds_error=False,
            fill_value=(0.0, d_l_grid[-1]),
        )
        return interp(z)
