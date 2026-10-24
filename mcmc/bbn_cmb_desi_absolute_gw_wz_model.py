"""Physical likelihood for DESI DR2 + BBN + CMB + absolute ET GW distances.

Following Gu et al., Nature Astronomy (2025), DOI 10.1038/s41550-025-02669-6:
29 piecewise-constant w bins are uniform in scale factor below z=2.5; w=-1
above z=2.5. The five-neighbour Horndeski correlation prior is included.
GW luminosity distances retain the SAME sampled H0 and Omega_m as BAO/CMB.
No GW scale profiling, H0 averaging, or GW-only posterior summary enters here.

CAMB supplies the baseline LCDM sound horizons and recombination distance
at every sampled point. The late-time w-bin distance correction is integrated
explicitly; no low-precision CMB emulator is used. Points whose altered DE
density is large enough to change recombination are rejected because the
LCDM RECFAST sound horizon would then be inconsistent with their background.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import camb
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import cho_factor, cho_solve

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paths import DATA, RESULTS  # noqa: E402

C_KM_S = 299792.458
N_W = 29
Z_W_MAX = 2.5
N_EFF = 3.046
MNU_EV = 0.06
OMEGA_NU_H2 = MNU_EV / 93.14

BBN_MEAN, BBN_SIGMA = 0.02196, 0.00063
CMB_THETA100_MEAN, CMB_THETA100_SIGMA = 1.04098, 0.00042
PARAMETER_NAMES = ("H0", "Omega_m", "omega_b") + tuple(f"w{i}" for i in range(1, N_W + 1))
LOWER = np.array((20.0, 0.01, 0.005) + (-6.0,) * N_W)
UPPER = np.array((100.0, 0.99, 0.040) + (4.0,) * N_W)

DESI_MEAN = DATA / "desi_gaussian_bao_ALL_GCcomb_mean.txt"
DESI_COV = DATA / "desi_gaussian_bao_ALL_GCcomb_cov.txt"


def catalog_path(n_events: int) -> Path:
    return RESULTS / "catalogs" / f"N{n_events:04d}" / "realization_0042.txt"


def w_geometry() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    a_edges = np.linspace(1.0, 1.0 / (1.0 + Z_W_MAX), N_W + 1)
    a_centres = 0.5 * (a_edges[:-1] + a_edges[1:])
    z_lower = 1.0 / a_edges[:-1] - 1.0
    z_upper = 1.0 / a_edges[1:] - 1.0
    z_centres = 1.0 / a_centres - 1.0
    return a_centres, z_lower, z_upper, z_centres


def de_kernel(z: np.ndarray) -> np.ndarray:
    """log[rho_DE(z)/rho_DE(0)] = K(z) @ (1+w)."""
    z = np.atleast_1d(np.asarray(z, dtype=float))
    _, lower, upper, _ = w_geometry()
    cap = np.minimum(z[:, None], upper[None, :])
    intervals = np.where(
        z[:, None] > lower[None, :],
        np.log1p(np.maximum(cap, lower[None, :])) - np.log1p(lower[None, :]),
        0.0,
    )
    return 3.0 * intervals


def horndeski_covariance() -> np.ndarray:
    """Return the unmodified Horndeski correlation covariance C_Pi."""
    a, _, _, _ = w_geometry()
    variance = 0.05 + 0.8 * a**2
    correlation = np.exp(-np.power(np.abs(np.log(a[:, None] / a[None, :])) / 0.3, 1.2))
    return np.sqrt(variance[:, None] * variance[None, :]) * correlation


def horndeski_raw_precision() -> np.ndarray:
    """Return the unmodified inverse correlation covariance C_Pi^-1."""
    covariance = horndeski_covariance()
    precision = np.linalg.solve(covariance, np.eye(N_W))
    return 0.5 * (precision + precision.T)


def horndeski_precision() -> np.ndarray:
    """Equations (6)-(8): (I-S)^T C_Pi^-1 (I-S), S = five-bin mean."""
    covariance = horndeski_covariance()
    smoothing = np.zeros((N_W, N_W))
    for i in range(N_W):
        neighbours = np.arange(max(0, i - 2), min(N_W, i + 3))
        smoothing[i, neighbours] = 1.0 / len(neighbours)
    difference = np.eye(N_W) - smoothing
    solved = np.linalg.solve(covariance, difference)
    precision = np.einsum("ki,kj->ij", difference, solved, optimize=True)
    return 0.5 * (precision + precision.T)


@dataclass(frozen=True)
class Observations:
    gw_z: np.ndarray
    gw_distance: np.ndarray
    gw_sigma: np.ndarray
    bao_z: np.ndarray
    bao_value: np.ndarray
    bao_quantity: tuple[str, ...]
    bao_cov_factor: tuple[np.ndarray, bool]
    gw_distance_loglike: Optional[Callable[[np.ndarray, np.ndarray], float]] = None


def load_observations(n_events: int) -> Observations:
    path = catalog_path(n_events)
    if not path.is_file():
        raise FileNotFoundError(f"GW catalogue is missing: {path}")
    gw = np.loadtxt(path, skiprows=1, ndmin=2)
    if gw.shape != (n_events, 3) or not np.all(np.isfinite(gw)):
        raise ValueError(f"Expected {n_events} finite (z, dL, sigma) GW rows in {path}")
    if np.any(gw[:, 0] < 0) or np.any(gw[:, 1:] <= 0):
        raise ValueError(f"Invalid redshift, distance, or uncertainty in {path}")
    bao_z, bao_value, bao_quantity, bao_cov_factor = load_desi_bao()
    return Observations(
        gw[:, 0], gw[:, 1], gw[:, 2], bao_z, bao_value,
        bao_quantity, bao_cov_factor,
    )


def load_desi_bao() -> tuple[np.ndarray, np.ndarray, tuple[str, ...], tuple[np.ndarray, bool]]:
    """Load the same DESI DR2 measurements for simulated or observed sirens."""
    if not DESI_MEAN.is_file() or not DESI_COV.is_file():
        raise FileNotFoundError("DESI DR2 BAO mean/covariance files are missing")
    bao_rows: list[tuple[float, float, str]] = []
    for line in DESI_MEAN.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        z, value, quantity = line.split()[:3]
        if quantity not in {"DM_over_rs", "DH_over_rs", "DV_over_rs"}:
            raise ValueError(f"Unsupported BAO observable: {quantity}")
        bao_rows.append((float(z), float(value), quantity))
    if not bao_rows:
        raise ValueError(f"No DESI DR2 BAO rows in {DESI_MEAN}")
    bao_z = np.asarray([r[0] for r in bao_rows])
    bao_value = np.asarray([r[1] for r in bao_rows])
    covariance = np.loadtxt(DESI_COV)
    if covariance.shape != (len(bao_rows), len(bao_rows)) or not np.all(np.isfinite(covariance)):
        raise ValueError(f"Invalid DESI DR2 covariance in {DESI_COV}")
    return bao_z, bao_value, tuple(r[2] for r in bao_rows), cho_factor(covariance, lower=True)


def camb_lcdm_background(
    h0: float, omega_m: float, omega_b: float, z_grid: np.ndarray,
) -> tuple[float, float, float, float, float, np.ndarray]:
    """Exact CAMB LCDM baseline: DM(z*), r*, rd, z*, Omega_DE and H(z)."""
    h = h0 / 100.0
    omega_c_h2 = omega_m * h * h - omega_b - OMEGA_NU_H2
    if omega_c_h2 <= 0:
        raise ValueError("Physical cold-dark-matter density must be positive")
    pars = camb.CAMBparams()
    pars.set_cosmology(
        H0=h0, ombh2=omega_b, omch2=omega_c_h2, mnu=MNU_EV,
        nnu=N_EFF, num_massive_neutrinos=1, omk=0.0, tau=0.054,
    )
    pars.set_dark_energy(w=-1.0)
    background = camb.get_background(pars)
    derived = background.get_derived_params()
    rstar, rd, zstar = (float(derived[name]) for name in ("rstar", "rdrag", "zstar"))
    dmstar = rstar / (float(derived["thetastar"]) / 100.0)
    omega_de = float(background.get_Omega("de", 0.0))
    hubble_grid = np.asarray(background.hubble_parameter(z_grid), dtype=float)
    return dmstar, rstar, rd, zstar, omega_de, hubble_grid


class EarlyBackground:
    """Exact CAMB LCDM early/background quantities at each sampled point."""

    def evaluate(
        self, h0: float, omega_m: float, omega_b: float, z_grid: np.ndarray,
    ) -> tuple[float, float, float, float, float, np.ndarray]:
        return camb_lcdm_background(h0, omega_m, omega_b, z_grid)


class JointLikelihood:
    def __init__(self, data: Observations, early: EarlyBackground, *, grid_size: int = 3200) -> None:
        if grid_size < 1000:
            raise ValueError("grid_size must be >= 1000 for the CMB distance integral")
        self.data = data
        self.early = early
        self.u_grid = np.linspace(0.0, np.log1p(1200.0), grid_size)
        self.z_grid = np.expm1(self.u_grid)
        self.one_plus_z = np.exp(self.u_grid)
        self.kernel = de_kernel(self.z_grid)
        self.prior_precision = horndeski_precision()

    def components(self, point: np.ndarray) -> dict[str, float]:
        values = np.asarray(point, dtype=float)
        if values.shape != (len(PARAMETER_NAMES),) or not np.all(np.isfinite(values)):
            raise ValueError("Point must be finite and ordered H0, Omega_m, omega_b, w1..w29")
        if np.any(values < LOWER) or np.any(values > UPPER):
            raise ValueError("Point is outside the broad flat parameter bounds")
        h0, omega_m, omega_b = values[:3]
        if omega_m * (h0 / 100.0)**2 <= omega_b + OMEGA_NU_H2:
            raise ValueError("Point has non-positive cold-dark-matter density")
        dmstar_lcdm, rstar, rd, zstar, omega_de, hubble_grid = self.early.evaluate(
            h0, omega_m, omega_b, self.z_grid
        )
        if omega_de <= 0:
            raise ValueError("Point has non-positive dark-energy density")
        if zstar > self.z_grid[-1] or rd <= 0 or rstar <= 0:
            raise ValueError("Invalid CAMB recombination/sound-horizon result")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            rho_de = np.exp(self.kernel @ (1.0 + values[3:]))
            e2_lcdm = (hubble_grid / h0)**2
            # Only the DE density changes; CAMB supplies radiation and the
            # massive-neutrino transition exactly in the common background.
            e2 = e2_lcdm + omega_de * (rho_de - 1.0)
        if not np.all(np.isfinite(e2) & (e2 > 0)):
            raise ValueError("Invalid H(z) for the proposed w bins")
        early_de_fraction = abs(omega_de * (rho_de[-1] - 1.0)) / float(
            np.interp(zstar, self.z_grid, e2_lcdm)
        )
        if early_de_fraction > 1e-4:
            raise ValueError("Early DE would alter the CAMB RECFAST sound horizon")
        # Integrate in u=ln(1+z); dz/E = exp(u) du/E.
        dm_grid = (C_KM_S / h0) * cumulative_trapezoid(
            self.one_plus_z / np.sqrt(e2), self.u_grid, initial=0.0
        )
        dm_gw = np.interp(self.data.gw_z, self.z_grid, dm_grid)
        gw_model = (1.0 + self.data.gw_z) * dm_gw
        if self.data.gw_distance_loglike is None:
            gw_residual = (self.data.gw_distance - gw_model) / self.data.gw_sigma
            gw_chi2 = float(np.einsum("i,i->", gw_residual, gw_residual))
        else:
            gw_chi2 = -2.0 * float(self.data.gw_distance_loglike(self.data.gw_z, gw_model))

        dm_bao = np.interp(self.data.bao_z, self.z_grid, dm_grid)
        e2_bao = np.interp(self.data.bao_z, self.z_grid, e2)
        dh_bao = C_KM_S / (h0 * np.sqrt(e2_bao))

        bao_model = np.empty(len(self.data.bao_z))
        for i, quantity in enumerate(self.data.bao_quantity):
            if quantity == "DM_over_rs":
                bao_model[i] = dm_bao[i] / rd
            elif quantity == "DH_over_rs":
                bao_model[i] = dh_bao[i] / rd
            else:
                bao_model[i] = np.cbrt(
                    self.data.bao_z[i] * dm_bao[i]**2 * dh_bao[i]
                ) / rd
        bao_diff = bao_model - self.data.bao_value
        bao_chi2 = float(bao_diff @ cho_solve(self.data.bao_cov_factor, bao_diff))

        # Replace only the low-z distance with the 29-bin history. CAMB retains
        # RECFAST's early-universe r*, rd, z* and the neutrino background.
        delta_integrand = self.one_plus_z * (
            1.0 / np.sqrt(e2) - 1.0 / np.sqrt(e2_lcdm)
        )
        delta_curve = (C_KM_S / h0) * cumulative_trapezoid(
            delta_integrand, self.u_grid, initial=0.0
        )
        dmstar = dmstar_lcdm + float(np.interp(zstar, self.z_grid, delta_curve))
        theta100 = 100.0 * rstar / dmstar
        cmb_chi2 = ((theta100 - CMB_THETA100_MEAN) / CMB_THETA100_SIGMA)**2
        bbn_chi2 = ((omega_b - BBN_MEAN) / BBN_SIGMA)**2
        centred_w = values[3:] + 1.0  # (I-S) removes any constant shift.
        prior_chi2 = float(np.einsum("i,ij,j->", centred_w, self.prior_precision, centred_w))
        answer = {
            "GW_absolute": gw_chi2,
            "DESI_DR2_BAO": bao_chi2,
            "CMB_theta_star": float(cmb_chi2),
            "BBN_omega_b_h2": float(bbn_chi2),
            "Horndeski_prior": max(prior_chi2, 0.0),
            "theta100_model": float(theta100),
            "rdrag_Mpc": float(rd),
            "early_de_fraction_at_zstar": float(early_de_fraction),
        }
        if not all(math.isfinite(value) for value in answer.values()):
            raise ValueError("Non-finite likelihood component")
        return answer

    def loglike(self, point: np.ndarray) -> float:
        try:
            chi2 = self.components(point)
        except (ValueError, OverflowError, camb.baseconfig.CAMBError):
            return -math.inf
        return -0.5 * sum(chi2[key] for key in (
            "GW_absolute", "DESI_DR2_BAO", "CMB_theta_star",
            "BBN_omega_b_h2", "Horndeski_prior",
        ))
