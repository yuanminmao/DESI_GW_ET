"""Bayesian evidences for CPL and flat LambdaCDM GW distance models.

The likelihood and priors match ``gw_mcmc.py``.  The CPL model samples
``(H0, omegam, w0, wa)``; the nested LambdaCDM model samples only
``(H0, omegam)`` with ``w0=-1`` and ``wa=0`` fixed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Callable

import numpy as np
from scipy.optimize import differential_evolution

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paths import CATALOGS, setup_sys_path

setup_sys_path()

from gw_cosmology import CPLCosmology


H0_BOUNDS = (20.0, 100.0)
OMEGAM_BOUNDS = (0.01, 0.99)
W0_BOUNDS = (-3.0, 3.0)
WA_BOUNDS = (-30.0, 3.0)
DEFAULT_CATALOG_SIZES = (100, 500, 1000, 3000)
DEFAULT_REALIZATION = 42
DEFAULT_NLIVE = 300
DEFAULT_DLOGZ = 0.1


@dataclass
class EvidenceResult:
    """Evidence comparison result for one catalogue."""

    ln_b: float
    ln_z_cpl: float
    ln_z_lcdm: float
    n_events: int
    method: str
    ln_zerr_cpl: float | None = None
    ln_zerr_lcdm: float | None = None
    ln_b_err: float | None = None
    chi2_cpl: float | None = None
    chi2_lcdm: float | None = None
    delta_chi2: float | None = None


def catalog_path_for(
    n_events: int,
    realization: int = DEFAULT_REALIZATION,
    *,
    root: Path = CATALOGS,
) -> Path:
    """Return the standard path for a GW mock catalogue."""
    if int(n_events) <= 0:
        raise ValueError("n_events must be positive")
    if int(realization) < 0:
        raise ValueError("realization must be non-negative")
    return Path(root) / f"N{int(n_events):04d}" / f"realization_{int(realization):04d}.txt"


def load_gw_catalog(
    path: Path,
    *,
    expected_n_events: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and validate z, observed luminosity distance, and uncertainty."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"GW catalogue not found: {path}")
    rows: list[list[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields or fields[0].startswith("#") or len(fields) < 3:
            continue
        try:
            rows.append([float(fields[0]), float(fields[1]), float(fields[2])])
        except ValueError:
            continue
    if not rows:
        raise ValueError(f"no numeric catalog rows found in {path}")
    data = np.asarray(rows, dtype=float)
    if not np.all(np.isfinite(data)):
        raise ValueError(f"GW catalogue contains non-finite values: {path}")
    z, distance, sigma = data[:, 0], data[:, 1], data[:, 2]
    if np.any(z < 0.0) or np.any(distance <= 0.0) or np.any(sigma <= 0.0):
        raise ValueError(
            f"GW catalogue requires z >= 0, dL > 0, and sigma_dL > 0: {path}"
        )
    if np.any(np.diff(z) < 0.0):
        raise ValueError(f"GW catalogue redshifts must be sorted: {path}")
    if expected_n_events is not None and data.shape[0] != int(expected_n_events):
        raise ValueError(
            f"GW catalogue event-count mismatch: expected {int(expected_n_events)}, "
            f"found {data.shape[0]} in {path}"
        )
    return z, distance, sigma


def _log_likelihood_factory(
    z: np.ndarray,
    d_obs: np.ndarray,
    sigma: np.ndarray,
) -> tuple[
    CPLCosmology,
    Callable[[float, float, float, float], float],
    float,
]:
    cosmo = CPLCosmology(z_max=max(float(np.max(z)) * 1.05, 2.5))
    inv_var = 1.0 / np.maximum(sigma**2, 1e-30)
    log_norm = -0.5 * float(np.sum(np.log(2.0 * np.pi * sigma**2)))

    def log_likelihood(h0: float, omegam: float, w0: float, wa: float) -> float:
        try:
            d_model = cosmo.luminosity_distance(z, h0, w0, wa, omega_m=omegam)
        except Exception:
            return -np.inf
        if not np.all(np.isfinite(d_model)):
            return -np.inf
        chi2 = float(np.sum((d_obs - d_model) ** 2 * inv_var))
        return -0.5 * chi2 + log_norm

    return cosmo, log_likelihood, log_norm


def _run_dynesty(
    loglike: Callable[[np.ndarray], float],
    bounds: tuple[tuple[float, float], ...],
    *,
    nlive: int,
    dlogz: float,
    seed: int,
) -> tuple[float, float]:
    try:
        import dynesty
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "dynesty is required for method='dynesty'; install it in the analysis environment"
        ) from exc

    lo = np.array([pair[0] for pair in bounds], dtype=float)
    width = np.array([pair[1] - pair[0] for pair in bounds], dtype=float)

    def prior_transform(unit_cube: np.ndarray) -> np.ndarray:
        return lo + width * np.asarray(unit_cube, dtype=float)

    sampler = dynesty.NestedSampler(
        loglike,
        prior_transform,
        ndim=len(bounds),
        nlive=int(nlive),
        rstate=np.random.default_rng(int(seed)),
    )
    sampler.run_nested(dlogz=float(dlogz), print_progress=False)
    return float(sampler.results.logz[-1]), float(sampler.results.logzerr[-1])


def _bic_evidence(
    loglike: Callable[[np.ndarray], float],
    bounds: tuple[tuple[float, float], ...],
    n_data: int,
    log_normalization: float,
) -> tuple[float, float]:
    result = differential_evolution(
        lambda theta: -float(loglike(np.asarray(theta))),
        bounds=bounds,
        polish=True,
        seed=0,
        workers=1,
    )
    # result.fun = -log(L_max) = chi2_min / 2 - log_normalization.
    chi2_min = 2.0 * (float(result.fun) + float(log_normalization))
    bic = chi2_min + len(bounds) * np.log(float(n_data))
    return float(bic), float(chi2_min)


def compute_ln_b_catalog(
    catalog: Path,
    *,
    expected_n_events: int | None = None,
    method: str = "dynesty",
    nlive: int = DEFAULT_NLIVE,
    dlogz: float = DEFAULT_DLOGZ,
    seed: int = 12345,
) -> EvidenceResult:
    """Compute ln(Z_CPL/Z_LCDM) for one GW catalogue."""
    catalog = catalog.resolve()
    z, d_obs, sigma = load_gw_catalog(
        catalog,
        expected_n_events=expected_n_events,
    )
    _, log_likelihood, log_normalization = _log_likelihood_factory(z, d_obs, sigma)

    cpl_bounds = (H0_BOUNDS, OMEGAM_BOUNDS, W0_BOUNDS, WA_BOUNDS)
    lcdm_bounds = (H0_BOUNDS, OMEGAM_BOUNDS)

    def cpl_loglike(theta: np.ndarray) -> float:
        return log_likelihood(*np.asarray(theta, dtype=float))

    def lcdm_loglike(theta: np.ndarray) -> float:
        h0, omegam = np.asarray(theta, dtype=float)
        return log_likelihood(h0, omegam, -1.0, 0.0)

    if method == "dynesty":
        ln_z_cpl, ln_zerr_cpl = _run_dynesty(
            cpl_loglike, cpl_bounds, nlive=nlive, dlogz=dlogz, seed=seed,
        )
        ln_z_lcdm, ln_zerr_lcdm = _run_dynesty(
            lcdm_loglike, lcdm_bounds, nlive=nlive, dlogz=dlogz, seed=seed + 1,
        )
        return EvidenceResult(
            ln_b=ln_z_cpl - ln_z_lcdm,
            ln_z_cpl=ln_z_cpl,
            ln_z_lcdm=ln_z_lcdm,
            n_events=int(z.size),
            method="dynesty",
            ln_zerr_cpl=ln_zerr_cpl,
            ln_zerr_lcdm=ln_zerr_lcdm,
            ln_b_err=float(np.hypot(ln_zerr_cpl, ln_zerr_lcdm)),
        )

    if method == "bic":
        bic_cpl, chi2_cpl = _bic_evidence(
            cpl_loglike,
            cpl_bounds,
            int(z.size),
            log_normalization,
        )
        bic_lcdm, chi2_lcdm = _bic_evidence(
            lcdm_loglike,
            lcdm_bounds,
            int(z.size),
            log_normalization,
        )
        # ln B_CPL,LCDM ≈ -0.5 (BIC_CPL - BIC_LCDM).
        ln_b = -0.5 * (bic_cpl - bic_lcdm)
        return EvidenceResult(
            ln_b=float(ln_b),
            ln_z_cpl=np.nan,
            ln_z_lcdm=np.nan,
            n_events=int(z.size),
            method="bic",
            chi2_cpl=float(chi2_cpl),
            chi2_lcdm=float(chi2_lcdm),
            delta_chi2=float(chi2_lcdm - chi2_cpl),
        )

    raise ValueError("method must be 'dynesty' or 'bic'")
