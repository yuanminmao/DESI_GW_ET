"""Flat CPL dark-energy cosmology: luminosity distance for GW inference."""

from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d

C_KM_S = 299792.458  # km/s


def de_integral_cpl(z: np.ndarray, w0: float, wa: float) -> np.ndarray:
    """
    int_0^z (1+w)/(1+z) dz  for  w(z) = w0 + wa * z/(1+z).
    """
    zp1 = 1.0 + np.asarray(z, dtype=float)
    return (1.0 + w0) * np.log(zp1) + wa * (np.log(zp1) + 1.0 / zp1 - 1.0)


def e_squared_cpl(z: np.ndarray, omega_m: float, w0: float, wa: float) -> np.ndarray:
    """E(z)^2 for flat universe with CPL dark energy."""
    z = np.asarray(z, dtype=float)
    omega_de = 1.0 - omega_m
    rho_ratio = np.exp(3.0 * de_integral_cpl(z, w0, wa))
    return omega_m * zp1_cubed(z) + omega_de * rho_ratio


def zp1_cubed(z: np.ndarray) -> np.ndarray:
    return (1.0 + np.asarray(z, dtype=float)) ** 3


def w_cpl(z: np.ndarray, w0: float, wa: float) -> np.ndarray:
    """CPL equation of state: w(z) = w0 + wa * z/(1+z)."""
    z = np.asarray(z, dtype=float)
    return w0 + wa * z / (1.0 + z)


class CPLCosmology:
    """Pre-tabulated luminosity distance d_L(z) [Mpc] for repeated likelihood calls."""

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
        w0: float,
        wa: float,
        *,
        omega_m: float | None = None,
    ) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        om = self.omega_m if omega_m is None else float(omega_m)
        ez = np.sqrt(np.maximum(e_squared_cpl(self.z_grid, om, w0, wa), 1e-30))
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


def distance_modulus(
    z: np.ndarray,
    h0: float,
    w0: float,
    wa: float,
    *,
    omega_m: float = 0.315,
    z_max: float = 3.0,
) -> np.ndarray:
    """Distance modulus mu = 5 log10(d_L/Mpc) + 25."""
    cosmo = CPLCosmology(omega_m=omega_m, z_max=max(float(np.max(z)) * 1.05, z_max))
    d_l = cosmo.luminosity_distance(z, h0, w0, wa)
    d_l = np.maximum(d_l, 1e-30)
    return 5.0 * np.log10(d_l) + 25.0
