#!/usr/bin/env python3
"""Preliminary 29-bin w(z) reconstruction from ET GW distances + prior only.

The likelihood contains only the absolute luminosity distances in one mock ET
catalogue.  It samples/fits H0, Omega_m and 29 piecewise-constant w bins; no
DESI, CMB, BBN or omega_b term is present.  The w bins and the five-neighbour
modified-Horndeski correlation prior are identical to the joint reconstruction
in ``bbn_cmb_desi_absolute_gw_wz_model.py``.  Above z=2.5, w is fixed to -1.

This script is intentionally a fast first-look analysis.  It finds the bounded
MAP with nonlinear least squares, builds a local Gauss--Newton (Laplace)
covariance, and applies the same broad top-hat bounds by rejection sampling.
The JSON output labels the approximation explicitly; publication results
should be replaced or validated with independent converged MCMC chains.

Examples
--------
Validate the current GW3000 catalogue and likelihood without writing results::

    python3 mcmc/reconstruct_wz_gw_prior.py --check --n-events 3000

Run the preliminary GW3000 reconstruction::

    python3 mcmc/reconstruct_wz_gw_prior.py --laplace --n-events 3000
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import least_squares

# NumPy 2 linked against Apple's Accelerate can emit spurious matmul warnings
# for finite arrays.  Every matrix used below is checked explicitly.
warnings.filterwarnings(
    "ignore",
    message=r"(divide by zero|overflow|invalid value) encountered in matmul",
    category=RuntimeWarning,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from bbn_cmb_desi_absolute_gw_wz_model import (  # noqa: E402
    C_KM_S,
    N_W,
    RESULTS,
    Z_W_MAX,
    catalog_path,
    de_kernel,
    horndeski_precision,
    w_geometry,
)

PARAMETER_NAMES = ("H0", "Omega_m") + tuple(f"w{i}" for i in range(1, N_W + 1))
LOWER = np.asarray((20.0, 0.01) + (-6.0,) * N_W)
UPPER = np.asarray((100.0, 0.99) + (4.0,) * N_W)
FIDUCIAL_H0 = 57.365
FIDUCIAL_OMEGA_M = 0.423
INJECTION_PATH = ROOT / "data" / "gw_fiducial_w_bins.txt"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
CHAINS = RESULTS / "chains"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_injection() -> np.ndarray:
    rows = np.loadtxt(INJECTION_PATH, comments="#", ndmin=2)
    if rows.shape[0] < N_W or rows.shape[1] < 5:
        raise ValueError(f"Malformed injected w-bin file: {INJECTION_PATH}")
    values = np.asarray(rows[:N_W, 4], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Non-finite injected w bins: {INJECTION_PATH}")
    return values


def prior_residual_factor(precision: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return R with ||R d||^2=d^T P d and P's ordered eigenvalues."""
    eigenvalues, eigenvectors = np.linalg.eigh(precision)
    tolerance = max(float(eigenvalues[-1]) * 1e-11, 1e-12)
    positive = eigenvalues > tolerance
    if np.count_nonzero(positive) != N_W - 1:
        raise ValueError(
            "The modified correlation prior should have exactly one constant-w null mode"
        )
    factor = np.sqrt(eigenvalues[positive])[:, None] * eigenvectors[:, positive].T
    if not np.allclose(factor.T @ factor, precision, rtol=1e-8, atol=1e-9):
        raise ValueError("Prior square-root factor does not reproduce its precision")
    return factor, eigenvalues[::-1]


@dataclass(frozen=True)
class GWData:
    z: np.ndarray
    distance: np.ndarray
    sigma: np.ndarray
    path: Path


def load_catalog(n_events: int) -> GWData:
    path = catalog_path(n_events)
    if not path.is_file():
        raise FileNotFoundError(f"Missing GW catalogue: {path}")
    rows = np.loadtxt(path, skiprows=1, ndmin=2)
    if rows.shape != (n_events, 3) or not np.all(np.isfinite(rows)):
        raise ValueError(f"Expected {n_events} finite (z,dL,sigma) rows in {path}")
    if np.any(rows[:, 0] < 0.0) or np.any(rows[:, 1:] <= 0.0):
        raise ValueError(f"Invalid redshift, distance or uncertainty in {path}")
    return GWData(rows[:, 0], rows[:, 1], rows[:, 2], path.resolve())


class GWOnlyLikelihood:
    """Absolute-distance likelihood in a flat late-time binned-w background."""

    def __init__(
        self,
        data: GWData,
        *,
        grid_size: int = 3200,
        prior_amplitude: float = 1.0,
    ) -> None:
        if grid_size < 1000:
            raise ValueError("grid_size must be at least 1000")
        if not math.isfinite(prior_amplitude) or prior_amplitude <= 0.0:
            raise ValueError("prior_amplitude must be finite and positive")
        self.data = data
        self.grid_size = int(grid_size)
        self.prior_amplitude = float(prior_amplitude)
        self.z_grid = np.linspace(0.0, max(3.5, 1.01 * float(np.max(data.z))), grid_size)
        self.kernel = de_kernel(self.z_grid)
        self.prior_precision = horndeski_precision()
        self.prior_factor, self.prior_eigenvalues = prior_residual_factor(
            self.prior_precision
        )

    def distance_model(self, point: np.ndarray) -> np.ndarray:
        values = np.asarray(point, dtype=float)
        if values.shape != (len(PARAMETER_NAMES),) or not np.all(np.isfinite(values)):
            raise ValueError("Expected finite H0, Omega_m, w1..w29")
        h0, omega_m = values[:2]
        w = values[2:]
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            rho_de = np.exp(self.kernel @ (1.0 + w))
            e2 = omega_m * (1.0 + self.z_grid) ** 3 + (1.0 - omega_m) * rho_de
        if not np.all(np.isfinite(e2) & (e2 > 0.0)):
            raise ValueError("Proposed w bins give an invalid expansion rate")
        comoving = (C_KM_S / h0) * cumulative_trapezoid(
            1.0 / np.sqrt(e2), self.z_grid, initial=0.0
        )
        return (1.0 + self.data.z) * np.interp(self.data.z, self.z_grid, comoving)

    def residuals(self, point: np.ndarray) -> np.ndarray:
        model = self.distance_model(point)
        gw = (self.data.distance - model) / self.data.sigma
        prior = (
            math.sqrt(self.prior_amplitude)
            * self.prior_factor
            @ (np.asarray(point, dtype=float)[2:] + 1.0)
        )
        return np.concatenate((gw, prior))

    def components(self, point: np.ndarray) -> dict[str, float]:
        model = self.distance_model(point)
        gw_residual = (self.data.distance - model) / self.data.sigma
        delta = np.asarray(point, dtype=float)[2:] + 1.0
        return {
            "GW": float(gw_residual @ gw_residual),
            "Horndeski_prior": float(
                self.prior_amplitude * (delta @ self.prior_precision @ delta)
            ),
        }


def initial_points(injected_w: np.ndarray) -> list[tuple[str, np.ndarray]]:
    fiducial = np.asarray([FIDUCIAL_H0, FIDUCIAL_OMEGA_M, *injected_w])
    lcdm = np.asarray([63.7, 0.44, *([-1.0] * N_W)])
    return [("injected_history", fiducial), ("lcdm", lcdm)]


def fit_map(
    model: GWOnlyLikelihood, injected_w: np.ndarray
) -> tuple[object, list[dict[str, object]]]:
    scale = np.asarray((2.0, 0.04) + (0.4,) * N_W)
    fits = []
    records: list[dict[str, object]] = []
    for label, start in initial_points(injected_w):
        result = least_squares(
            model.residuals,
            start,
            bounds=(LOWER, UPPER),
            method="trf",
            x_scale=scale,
            ftol=2e-12,
            xtol=2e-12,
            gtol=2e-12,
            max_nfev=2500,
            verbose=0,
        )
        fits.append(result)
        records.append(
            {
                "start": label,
                "success": bool(result.success),
                "status": int(result.status),
                "message": str(result.message),
                "n_function_evaluations": int(result.nfev),
                "chi2_total": float(2.0 * result.cost),
                "optimality": float(result.optimality),
            }
        )
    best = min(fits, key=lambda result: result.cost)
    if not best.success or not np.all(np.isfinite(best.x)):
        raise RuntimeError(f"MAP optimization failed: {best.message}")
    return best, records


def fit_lcdm(model: GWOnlyLikelihood) -> object:
    def residuals(values: np.ndarray) -> np.ndarray:
        point = np.asarray([values[0], values[1], *([-1.0] * N_W)])
        prediction = model.distance_model(point)
        return (model.data.distance - prediction) / model.data.sigma

    result = least_squares(
        residuals,
        np.asarray([63.7, 0.44]),
        bounds=(LOWER[:2], UPPER[:2]),
        x_scale=np.asarray([2.0, 0.04]),
        ftol=2e-12,
        xtol=2e-12,
        gtol=2e-12,
        max_nfev=500,
    )
    if not result.success:
        raise RuntimeError(f"LCDM optimization failed: {result.message}")
    return result


def laplace_covariance(jacobian: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fisher = np.asarray(jacobian, dtype=float).T @ np.asarray(jacobian, dtype=float)
    fisher = 0.5 * (fisher + fisher.T)
    eigenvalues, eigenvectors = np.linalg.eigh(fisher)
    if eigenvalues[0] <= 0.0 or not np.all(np.isfinite(eigenvalues)):
        raise ValueError("Local posterior Fisher matrix is not positive definite")
    covariance = (eigenvectors / eigenvalues) @ eigenvectors.T
    covariance = 0.5 * (covariance + covariance.T)
    return covariance, eigenvalues


def bounded_laplace_draws(
    mean: np.ndarray,
    covariance: np.ndarray,
    *,
    target: int,
    seed: int,
) -> tuple[np.ndarray, int]:
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if eigenvalues[0] <= 0.0:
        raise ValueError("Laplace covariance is not positive definite")
    transform = eigenvectors * np.sqrt(eigenvalues)
    rng = np.random.default_rng(seed)
    accepted: list[np.ndarray] = []
    n_accepted = 0
    n_proposed = 0
    maximum = max(2_000_000, 200 * target)
    while n_accepted < target and n_proposed < maximum:
        chunk = min(max(20_000, 2 * (target - n_accepted)), maximum - n_proposed)
        candidates = mean + rng.standard_normal((chunk, mean.size)) @ transform.T
        valid = np.all((candidates >= LOWER) & (candidates <= UPPER), axis=1)
        selected = candidates[valid]
        if selected.size:
            accepted.append(selected)
            n_accepted += selected.shape[0]
        n_proposed += chunk
    if n_accepted < target:
        raise RuntimeError(
            f"Only {n_accepted} bounded Laplace draws accepted from {n_proposed} proposals"
        )
    return np.concatenate(accepted, axis=0)[:target], n_proposed


def mahalanobis(deviation: np.ndarray, covariance: np.ndarray) -> float:
    return math.sqrt(
        max(float(deviation @ np.linalg.solve(covariance, deviation)), 0.0)
    )


def write_figure(
    path: Path,
    *,
    lower: np.ndarray,
    upper: np.ndarray,
    mean: np.ndarray,
    q16: np.ndarray,
    q84: np.ndarray,
    map_w: np.ndarray,
    injected_w: np.ndarray,
    n_events: int,
    interval_label: str = "68% bounded-Laplace interval",
    title_suffix: str = "(preliminary)",
    center_label: str = "posterior mean",
    fit_label: str = "MAP",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    edges = np.r_[lower, upper[-1]]
    extend = lambda values: np.r_[values, values[-1]]
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    ax.fill_between(
        edges,
        extend(q16),
        extend(q84),
        step="post",
        color="#4C78A8",
        alpha=0.28,
        label=interval_label,
    )
    ax.step(edges, extend(mean), where="post", color="#1F5A99", lw=2.1,
            label=center_label)
    ax.step(edges, extend(map_w), where="post", color="#1F5A99", lw=1.2,
            ls=":", label=fit_label)
    ax.step(edges, extend(injected_w), where="post", color="#D95F02", lw=2.0,
            label="injected w(z)")
    ax.axhline(-1.0, color="black", ls="--", lw=1.1, label="flat LCDM")
    ax.set(
        xlim=(0.0, Z_W_MAX),
        xlabel="Redshift z",
        ylabel="w(z)",
        title=f"GW{n_events} + modified Horndeski correlation prior {title_suffix}",
    )
    ax.grid(alpha=0.15)
    ax.legend(frameon=False, ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def check(n_events: int, grid_size: int) -> None:
    data = load_catalog(n_events)
    injection = load_injection()
    model = GWOnlyLikelihood(data, grid_size=grid_size)
    point = np.asarray([FIDUCIAL_H0, FIDUCIAL_OMEGA_M, *injection])
    components = model.components(point)
    print(f"GW{n_events}: {data.path}")
    print(f"catalog SHA256: {sha256(data.path)}")
    print(f"z range: {data.z.min():.6f} .. {data.z.max():.6f}")
    print(f"median fractional distance error: {np.median(data.sigma / data.distance):.6f}")
    print(f"injected-point GW chi2 / event: {components['GW'] / n_events:.6f}")
    print(f"injected-history prior chi2: {components['Horndeski_prior']:.6f}")
    print("Included terms: GW_absolute + Horndeski_prior")
    print("Excluded terms: DESI, CMB, BBN, omega_b")


def run_laplace(args: argparse.Namespace) -> dict[str, object]:
    data = load_catalog(args.n_events)
    injected_w = load_injection()
    model = GWOnlyLikelihood(data, grid_size=args.grid_size)
    map_fit, optimization_records = fit_map(model, injected_w)
    map_point = np.asarray(map_fit.x)
    map_components = model.components(map_point)
    covariance_unbounded, fisher_eigenvalues = laplace_covariance(map_fit.jac)
    draws, proposed = bounded_laplace_draws(
        map_point, covariance_unbounded, target=args.draws, seed=args.seed
    )
    posterior_mean = np.mean(draws, axis=0)
    posterior_median = np.median(draws, axis=0)
    q16, q84 = np.quantile(draws, [0.16, 0.84], axis=0)
    posterior_covariance = np.cov(draws, rowvar=False, ddof=0)
    posterior_covariance = 0.5 * (posterior_covariance + posterior_covariance.T)
    w_covariance = posterior_covariance[2:, 2:]
    np.linalg.cholesky(w_covariance)

    lcdm_fit = fit_lcdm(model)
    lcdm_point = np.asarray([lcdm_fit.x[0], lcdm_fit.x[1], *([-1.0] * N_W)])
    lcdm_components = model.components(lcdm_point)
    injection_point = np.asarray([FIDUCIAL_H0, FIDUCIAL_OMEGA_M, *injected_w])
    injection_components = model.components(injection_point)

    # Verify numerical integration at twice the production grid density.
    fine_model = GWOnlyLikelihood(data, grid_size=2 * args.grid_size)
    coarse_distance = model.distance_model(map_point)
    fine_distance = fine_model.distance_model(map_point)
    grid_max_relative_distance_change = float(
        np.max(np.abs(fine_distance - coarse_distance) / fine_distance)
    )
    fine_components = fine_model.components(map_point)

    mean_w = posterior_mean[2:]
    map_w = map_point[2:]
    snr_mean = mahalanobis(mean_w + 1.0, w_covariance)
    snr_map = mahalanobis(map_w + 1.0, w_covariance)
    injection_distance = mahalanobis(mean_w - injected_w, w_covariance)
    effective_w_dof = float(N_W - np.trace(model.prior_precision @ w_covariance))
    covered = (injected_w >= q16[2:]) & (injected_w <= q84[2:])
    at_boundary = np.minimum(map_point - LOWER, UPPER - map_point)
    map_at_boundary = bool(np.min(at_boundary) < 1e-5)

    _, z_lower, z_upper, z_centres = w_geometry()
    stem = f"gw{args.n_events:04d}_prior_wz"
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    sample_dir = CHAINS / f"GW{args.n_events:04d}_PRIOR_wz_laplace"
    sample_dir.mkdir(parents=True, exist_ok=True)
    table_path = TABLES / f"{stem}_reconstruction.csv"
    covariance_path = TABLES / f"{stem}_covariance.txt"
    full_covariance_path = TABLES / f"{stem}_full_laplace_covariance.txt"
    summary_path = TABLES / f"{stem}_summary.json"
    figure_path = FIGURES / f"{stem}.png"
    samples_path = sample_dir / "bounded_laplace_samples.npz"

    with table_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "bin", "z_min", "z_max", "z_center", "w_mean", "w_median",
                "w_q16", "w_q84", "w_sigma", "w_map", "w_injected",
                "injected_in_68pct",
            )
        )
        for i in range(N_W):
            writer.writerow(
                (
                    i + 1,
                    z_lower[i],
                    z_upper[i],
                    z_centres[i],
                    mean_w[i],
                    posterior_median[i + 2],
                    q16[i + 2],
                    q84[i + 2],
                    math.sqrt(w_covariance[i, i]),
                    map_w[i],
                    injected_w[i],
                    bool(covered[i]),
                )
            )
    np.savetxt(
        covariance_path,
        w_covariance,
        fmt="%.12e",
        header="29x29 w-bin covariance from bounded local Laplace draws",
    )
    np.savetxt(
        full_covariance_path,
        covariance_unbounded,
        fmt="%.12e",
        header="31x31 unbounded local Gauss-Newton covariance: H0, Omega_m, w1..w29",
    )
    np.savez_compressed(
        samples_path,
        samples=draws,
        parameter_names=np.asarray(PARAMETER_NAMES),
        map=map_point,
    )
    write_figure(
        figure_path,
        lower=z_lower,
        upper=z_upper,
        mean=mean_w,
        q16=q16[2:],
        q84=q84[2:],
        map_w=map_w,
        injected_w=injected_w,
        n_events=args.n_events,
    )

    summary: dict[str, object] = {
        "analysis": f"{args.n_events} simulated ET standard sirens + modified Horndeski correlation prior only",
        "status": (
            "diagnostic only: MAP is on a flat-prior boundary, so the local Laplace posterior is invalid"
            if map_at_boundary
            else "preliminary bounded Laplace approximation; validate with converged MCMC before publication"
        ),
        "posterior_valid_for_inference": not map_at_boundary,
        "posterior_method": "bounded_laplace_gauss_newton",
        "included_likelihood_terms": ["GW_absolute", "Horndeski_prior"],
        "explicitly_excluded": ["DESI_DR2_BAO", "CMB_theta_star", "BBN_omega_b_h2", "omega_b"],
        "sampled_parameters": list(PARAMETER_NAMES),
        "n_w_bins": N_W,
        "w_bin_spacing": "uniform in scale factor for 0 <= z < 2.5",
        "w_above_z_2p5": -1.0,
        "late_time_background": "flat; E2=Omega_m(1+z)^3+(1-Omega_m)rho_DE(z)/rho_DE(0); radiation omitted",
        "horndeski_prior": {
            "amplitude_A": 1.0,
            "precision": "(I-S)^T C_Pi^-1 (I-S), S=five-bin running mean",
            "rank": N_W - 1,
            "constant_w_null_mode": True,
            "largest_to_smallest_eigenvalues": model.prior_eigenvalues.tolist(),
        },
        "flat_prior_bounds": {
            "H0": [float(LOWER[0]), float(UPPER[0])],
            "Omega_m": [float(LOWER[1]), float(UPPER[1])],
            "w_i": [float(LOWER[2]), float(UPPER[2])],
        },
        "map": {name: float(value) for name, value in zip(PARAMETER_NAMES, map_point)},
        "posterior_mean": {
            name: float(value) for name, value in zip(PARAMETER_NAMES, posterior_mean)
        },
        "posterior_median": {
            name: float(value) for name, value in zip(PARAMETER_NAMES, posterior_median)
        },
        "chi2_components_at_map": map_components,
        "chi2_total_at_map": float(sum(map_components.values())),
        "chi2_components_at_injected_point": injection_components,
        "chi2_total_at_injected_point": float(sum(injection_components.values())),
        "lcdm_best_fit": {
            "H0": float(lcdm_fit.x[0]),
            "Omega_m": float(lcdm_fit.x[1]),
            "GW_chi2": float(lcdm_components["GW"]),
            "delta_chi2_lcdm_gw_minus_map_total": float(
                lcdm_components["GW"] - sum(map_components.values())
            ),
            "note": "diagnostic only; not a Wilks significance or Bayes factor",
        },
        "reconstruction_diagnostics": {
            "snr_posterior_mean_vs_w_minus_one": snr_mean,
            "snr_map_vs_w_minus_one": snr_map,
            "snr_definition": "sqrt[(w+1)^T Cov(w)^-1 (w+1)] using bounded-Laplace marginalized w covariance",
            "posterior_mean_distance_from_injected_w": injection_distance,
            "injected_bins_inside_marginal_68pct": int(np.count_nonzero(covered)),
            "effective_w_degrees_of_freedom": effective_w_dof,
            "effective_w_dof_definition": "29 - Tr(P_Horndeski Cov(w))",
            "minimum_map_distance_to_any_flat_bound": float(np.min(at_boundary)),
            "map_at_flat_prior_boundary": map_at_boundary,
        },
        "optimization": optimization_records,
        "laplace": {
            "bounded_retained_draws": int(draws.shape[0]),
            "gaussian_proposals": int(proposed),
            "acceptance_fraction": float(draws.shape[0] / proposed),
            "fisher_eigenvalue_min": float(fisher_eigenvalues[0]),
            "fisher_eigenvalue_max": float(fisher_eigenvalues[-1]),
        },
        "numerics": {
            "distance_grid_size": args.grid_size,
            "fine_grid_size": 2 * args.grid_size,
            "max_relative_distance_change_on_grid_doubling_at_map": grid_max_relative_distance_change,
            "map_total_chi2_on_fine_grid": float(sum(fine_components.values())),
            "map_total_chi2_coarse_minus_fine": float(
                sum(map_components.values()) - sum(fine_components.values())
            ),
            "seed": args.seed,
        },
        "catalog": {
            "path": str(data.path),
            "sha256": sha256(data.path),
            "n_events": args.n_events,
            "z_min": float(np.min(data.z)),
            "z_median": float(np.median(data.z)),
            "z_max": float(np.max(data.z)),
            "fraction_z_le_2p5": float(np.mean(data.z <= Z_W_MAX)),
            "median_fractional_distance_error": float(np.median(data.sigma / data.distance)),
        },
        "injection": {
            "H0": FIDUCIAL_H0,
            "Omega_m": FIDUCIAL_OMEGA_M,
            "w_path": str(INJECTION_PATH.resolve()),
            "w_sha256": sha256(INJECTION_PATH),
        },
        "outputs": {
            "reconstruction_csv": str(table_path.resolve()),
            "w_covariance": str(covariance_path.resolve()),
            "full_unbounded_laplace_covariance": str(full_covariance_path.resolve()),
            "bounded_laplace_samples": str(samples_path.resolve()),
            "figure": str(figure_path.resolve()),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    summary["outputs"]["summary"] = str(summary_path.resolve())  # type: ignore[index]
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Validate data/model only")
    mode.add_argument("--laplace", action="store_true", help="Run the preliminary bounded-Laplace reconstruction")
    parser.add_argument("--n-events", type=int, default=3000)
    parser.add_argument("--grid-size", type=int, default=3200)
    parser.add_argument("--draws", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260930)
    args = parser.parse_args()
    if not args.check and not args.laplace:
        args.check = True
    if args.n_events < 1 or args.grid_size < 1000 or args.draws < 1000:
        parser.error("--n-events must be positive, --grid-size >=1000 and --draws >=1000")
    return args


def main() -> None:
    args = parse_args()
    if args.check:
        check(args.n_events, args.grid_size)
        return
    result = run_laplace(args)
    diagnostics = result["reconstruction_diagnostics"]
    outputs = result["outputs"]
    print(f"GW{args.n_events} + prior preliminary reconstruction complete")
    print(f"SNR(mean vs w=-1): {diagnostics['snr_posterior_mean_vs_w_minus_one']:.4f}")
    print(f"Effective w degrees of freedom: {diagnostics['effective_w_degrees_of_freedom']:.4f}")
    print(f"Summary: {outputs['summary']}")
    print(f"Figure: {outputs['figure']}")


if __name__ == "__main__":
    main()
