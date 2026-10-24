#!/usr/bin/env python3
"""Formal Cobaya MCMC for the GW-only + correlation-prior w(z) model.

This self-contained script varies H0, Omega_m and all 29 w bins under the
broad flat bounds, and its external log likelihood is exactly

    -2 log L = chi2_GW + chi2_Horndeski-prior.

DESI, CMB, BBN and omega_b are absent.  Four or more MPI processes are needed
for the formal run.  Posterior products are written only after every sampled
parameter passes rank-normalized and folded split Rhat - 1 < 0.01.

    mpiexec -n 4 python3 mcmc/reconstruct_wz_gw.py --run --n-events 3000
    mpiexec -n 4 python3 mcmc/reconstruct_wz_gw.py --run --n-events 3000 --resume
    python3 mcmc/reconstruct_wz_gw.py --postprocess --n-events 3000

Run or resume the four production A=1 catalogues in sequence::

    python3 mcmc/reconstruct_wz_gw.py --run-four
    python3 mcmc/reconstruct_wz_gw.py --status-only

The separate ``gw_pca_001.py`` script owns the weak-prior A=0.01 chains used
for the GW-only PCA data-Fisher subtraction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from cobaya import run
from scipy.integrate import cumulative_trapezoid
from scipy.stats import norm, rankdata

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from mcmc.reconstruct_wz_desi_bbn_cmb_gw import (  # noqa: E402
    C_KM_S,
    N_W,
    RESULTS,
    Z_W_MAX,
    catalog_path,
    de_kernel,
    horndeski_precision,
    w_geometry,
)
from paths import RECONSTRUCT_WZ_GW_TABLES, GW_WZ_CHAINS, project_path  # noqa: E402

PARAMETER_NAMES = ("H0", "Omega_m") + tuple(f"w{i}" for i in range(1, N_W + 1))
LOWER = np.asarray((20.0, 0.01) + (-6.0,) * N_W)
UPPER = np.asarray((100.0, 0.99) + (4.0,) * N_W)
FIDUCIAL_H0 = 57.365
FIDUCIAL_OMEGA_M = 0.423
INJECTION_PATH = ROOT / "data" / "gw_fiducial_w_bins.txt"
DEFAULT_INJECTION_TEMPLATE = INJECTION_PATH
TABLES = RECONSTRUCT_WZ_GW_TABLES
FIGURES = RESULTS / "figures"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_injected_template(
    path: Path = DEFAULT_INJECTION_TEMPLATE,
    *,
    n_bins: int = N_W,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return injected w_i, z_min and z_max for the requested bins."""
    path = Path(path).expanduser().resolve()
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


def load_injection() -> np.ndarray:
    """Return the default 29 injected w bins used by this reconstruction."""
    return load_injected_template()[0]


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
    interval_label: str = "68% MCMC credible interval",
    title_suffix: str = "(MCMC)",
    center_label: str = "posterior mean",
    fit_label: str = "best retained sample",
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

A1_ROOT = GW_WZ_CHAINS
DEFAULT_EVENTS = (100, 500, 1000, 3000)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true")
    mode.add_argument(
        "--run-four",
        action="store_true",
        help="run or resume the four production A=1 catalogues in sequence",
    )
    mode.add_argument(
        "--status-only",
        action="store_true",
        help="report the four production A=1 states without launching anything",
    )
    mode.add_argument("--postprocess", action="store_true")
    mode.add_argument("--check", action="store_true")
    parser.add_argument("--n-events", type=int, default=3000)
    parser.add_argument(
        "--events",
        type=int,
        nargs="+",
        default=list(DEFAULT_EVENTS),
        help="catalogue sizes for --run-four/--status-only (default: 100 500 1000 3000)",
    )
    parser.add_argument(
        "--mpi-processes",
        type=int,
        default=4,
        help="MPI ranks per catalogue in --run-four mode (minimum: 4)",
    )
    parser.add_argument("--grid-size", type=int, default=3200)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="accepted-step ceiling per rank (default: 250000 single, 1000000 batch)",
    )
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="with --run-four, print pending commands without launching them",
    )
    args = parser.parse_args()
    if args.resume and not args.run:
        parser.error("--resume requires --run")
    if args.dry_run and not args.run_four:
        parser.error("--dry-run requires --run-four")
    if (args.run_four or args.status_only) and args.output_root is not None:
        parser.error("--output-root is available only for a single-catalogue mode")
    args.events = list(dict.fromkeys(args.events))
    if not args.events or any(value not in DEFAULT_EVENTS for value in args.events):
        parser.error("--events may contain only 100, 500, 1000 and 3000")
    if args.mpi_processes < 4:
        parser.error("--mpi-processes must be at least 4")
    if args.max_samples is None:
        args.max_samples = 1_000_000 if args.run_four else 250_000
    if args.n_events < 1 or args.grid_size < 1000 or args.max_samples < 1:
        parser.error("invalid event count, grid size or sample limit")
    return args


def prefix_for(
    args: argparse.Namespace,
    *,
    default_root: Path = A1_ROOT,
) -> Path:
    root = default_root if args.output_root is None else args.output_root
    return root.expanduser().resolve() / f"GW{args.n_events:04d}" / "chain"


def initial_proposal() -> tuple[np.ndarray, np.ndarray, str]:
    """Return a self-contained diagonal initializer around the injection."""
    injection = load_injection()
    mean = np.asarray([FIDUCIAL_H0, FIDUCIAL_OMEGA_M, *injection])
    scales = np.asarray((2.0, 0.08) + (0.45,) * N_W)
    covariance = np.diag(scales**2)
    return mean, covariance, "diagonal proposal around the injected history"


def production_summary_path(n_events: int) -> Path:
    """Return the accepted A=1 summary for one production catalogue."""
    return TABLES / f"gw{n_events:04d}_prior_wz_mcmc_summary.json"


def production_chain_files(n_events: int) -> list[Path]:
    """Return native A=1 Cobaya chains for one production catalogue."""
    directory = A1_ROOT / f"GW{n_events:04d}"
    return sorted(directory.glob("chain.[0-9]*.txt"))


def accepted_production_summary(n_events: int) -> tuple[bool, str]:
    """Report whether the saved A=1 posterior passed the independent Rhat test."""
    path = production_summary_path(n_events)
    if not path.is_file():
        return False, "no accepted summary"
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
        amplitude = float(summary.get("prior_amplitude", 1.0))
        diagnostic = summary["rank_folded_split_rhat"]
        maximum = float(diagnostic["max_rhat_minus_one"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return False, f"invalid summary: {exc}"
    if amplitude != 1.0:
        return False, f"summary records prior amplitude A={amplitude:g}"
    if not diagnostic.get("all_pass") or maximum >= 0.01:
        return False, f"independent max Rhat-1={maximum:.6f} has not passed"
    draws = int(summary.get("total_retained_weighted_draws", 0))
    return True, f"accepted: max Rhat-1={maximum:.6f}, retained draws={draws}"


def production_sampling_command(
    n_events: int, args: argparse.Namespace
) -> list[str]:
    """Build the MPI command for one production A=1 catalogue."""
    command = [
        "mpiexec",
        "-n",
        str(args.mpi_processes),
        sys.executable,
        str(Path(__file__).resolve()),
        "--run",
        "--n-events",
        str(n_events),
        "--grid-size",
        str(args.grid_size),
        "--max-samples",
        str(args.max_samples),
        "--seed",
        str(args.seed),
    ]
    if production_chain_files(n_events):
        command.append("--resume")
    return command


def run_four_catalogues(args: argparse.Namespace) -> None:
    """Run/resume the selected production A=1 catalogues sequentially."""
    states = {
        n_events: accepted_production_summary(n_events)
        for n_events in args.events
    }
    for n_events, (complete, detail) in states.items():
        label = "COMPLETE" if complete else "PENDING"
        print(f"GW{n_events:04d} A=1: {label} -- {detail}")
    if args.status_only:
        return

    if not args.dry_run and shutil.which("mpiexec") is None:
        raise SystemExit("mpiexec is not available on PATH")
    for n_events in args.events:
        complete, _ = accepted_production_summary(n_events)
        if complete:
            print(f"GW{n_events:04d} A=1: skip accepted posterior")
            continue
        command = production_sampling_command(n_events, args)
        print(f"GW{n_events:04d} A=1 command:\n  {shlex.join(command)}", flush=True)
        if args.dry_run:
            continue
        completed = subprocess.run(command, cwd=ROOT, check=False)
        if completed.returncode:
            raise SystemExit(
                f"GW{n_events:04d} A=1 stopped with exit code "
                f"{completed.returncode}; later catalogues were not launched"
            )
        accepted, detail = accepted_production_summary(n_events)
        if not accepted:
            raise SystemExit(
                f"GW{n_events:04d} A=1 ended without an accepted posterior: {detail}"
            )
        print(f"GW{n_events:04d} A=1: {detail}", flush=True)


def cobaya_info(
    model: GWOnlyLikelihood,
    prefix: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, object], str]:
    reference, covariance, proposal_source = initial_proposal()
    standard_deviation = np.sqrt(np.diag(covariance))
    params: dict[str, object] = {}
    for i, name in enumerate(PARAMETER_NAMES):
        params[name] = {
            "prior": {"min": float(LOWER[i]), "max": float(UPPER[i])},
            "ref": {
                "dist": "norm",
                "loc": float(np.clip(reference[i], LOWER[i] + 1e-6, UPPER[i] - 1e-6)),
                "scale": float(max(0.05 * standard_deviation[i], 1e-5)),
            },
            "proposal": float(max(standard_deviation[i], 1e-5)),
        }

    def loglike(**values: float) -> float:
        point = np.asarray([values[name] for name in PARAMETER_NAMES])
        try:
            components = model.components(point)
        except (ValueError, OverflowError):
            return -math.inf
        return -0.5 * float(sum(components.values()))

    info: dict[str, object] = {
        "params": params,
        "likelihood": {
            "GW_absolute_plus_Horndeski_prior": {
                "external": loglike,
                "input_params": list(PARAMETER_NAMES),
            }
        },
        "sampler": {
            "mcmc": {
                "covmat": covariance,
                "covmat_params": list(PARAMETER_NAMES),
                "burn_in": "2d",
                "max_samples": args.max_samples,
                "Rminus1_stop": 0.01,
                "Rminus1_cl_stop": 0.05,
                "learn_proposal": True,
                "learn_every": "40d",
                "learn_proposal_Rminus1_max": 100.0,
                "max_tries": 10000,
                "seed": args.seed + args.n_events,
            }
        },
        "output": str(prefix),
        "stop_at_error": True,
    }
    return info, proposal_source


def chain_files(prefix: Path) -> list[Path]:
    pattern = re.compile(re.escape(prefix.name) + r"\.\d+\.txt$")
    return sorted(
        path
        for path in prefix.parent.glob(prefix.name + ".*.txt")
        if pattern.fullmatch(path.name)
    )


def retained_chain(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    columns = tuple(range(2 + len(PARAMETER_NAMES)))
    rows = np.loadtxt(path, comments="#", usecols=columns, ndmin=2)
    if rows.shape[1] != len(columns) or not np.all(np.isfinite(rows)):
        raise ValueError(f"Malformed or non-finite Cobaya chain: {path}")
    raw_weights = rows[:, 0]
    weights = raw_weights.astype(np.int64)
    if np.any(weights <= 0) or not np.array_equal(raw_weights, weights):
        raise ValueError(f"Non-integer repeat weights in {path}")
    total = int(np.sum(weights))
    if total < 2000:
        raise ValueError(f"Too few weighted samples in {path}")
    cumulative = np.cumsum(weights)
    retained = np.minimum(weights, np.maximum(0, cumulative - total // 2))
    selected = retained > 0
    return rows[selected, 2:], retained[selected], rows[selected, 1], total


def split_rhat(chains: np.ndarray) -> float:
    n_chains, length = chains.shape
    half = length // 2
    if n_chains < 2 or half < 2:
        return math.inf
    halves = np.concatenate((chains[:, :half], chains[:, -half:]), axis=0)
    within = float(np.mean(np.var(halves, axis=1, ddof=1)))
    if not math.isfinite(within) or within <= 0.0:
        return math.inf
    between = float(half * np.var(np.mean(halves, axis=1), ddof=1))
    variance = (half - 1.0) * within / half + between / half
    return float(np.sqrt(max(variance / within, 0.0)))


def rank_folded_rhat(
    groups: list[tuple[np.ndarray, np.ndarray, np.ndarray, int]],
) -> dict[str, object]:
    common = min(int(np.sum(group[1])) for group in groups)
    if common < 1000:
        raise ValueError("Too few retained weighted draws for split Rhat")
    diagnostics: dict[str, object] = {}
    for i, name in enumerate(PARAMETER_NAMES):
        repeated = [np.repeat(group[0][:, i], group[1])[-common:] for group in groups]
        values = np.stack(repeated)
        flat = values.reshape(-1)
        ranks = rankdata(flat, method="average")
        normalized = norm.ppf((ranks - 0.375) / (len(flat) + 0.25)).reshape(values.shape)
        rank = split_rhat(normalized)
        folded = np.abs(flat - np.median(flat))
        folded_ranks = rankdata(folded, method="average")
        folded_normalized = norm.ppf(
            (folded_ranks - 0.375) / (len(flat) + 0.25)
        ).reshape(values.shape)
        folded_rhat = split_rhat(folded_normalized)
        diagnostics[name] = {
            "rank_rhat": rank,
            "folded_rhat": folded_rhat,
            "max_rhat": max(rank, folded_rhat),
        }
    maximum = max(float(item["max_rhat"]) for item in diagnostics.values())  # type: ignore[union-attr]
    return {
        "common_retained_draws_per_chain": common,
        "max_rhat_minus_one": maximum - 1.0,
        "all_pass": bool(maximum - 1.0 < 0.01),
        "per_parameter": diagnostics,
    }


def weighted_quantile(
    values: np.ndarray, weights: np.ndarray, probabilities: tuple[float, ...]
) -> np.ndarray:
    order = np.argsort(values)
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    targets = np.asarray(probabilities) * cumulative[-1]
    indices = np.searchsorted(cumulative, targets, side="left")
    return sorted_values[np.clip(indices, 0, len(sorted_values) - 1)]


def postprocess(
    prefix: Path,
    n_events: int,
    grid_size: int,
    prior_amplitude: float,
    *,
    table_root: Path = TABLES,
) -> dict[str, object]:
    files = chain_files(prefix)
    if len(files) < 4:
        raise ValueError(f"At least four native Cobaya chains are required at {prefix}")
    groups = [retained_chain(path) for path in files]
    rhat = rank_folded_rhat(groups)
    if not rhat["all_pass"]:
        raise ValueError(
            f"GW{n_events}: max rank/folded split Rhat-1="
            f"{rhat['max_rhat_minus_one']:.5f}; resume the chains"
        )
    samples = np.concatenate([group[0] for group in groups])
    weights = np.concatenate([group[1] for group in groups])
    minuslogpost = np.concatenate([group[2] for group in groups])
    mean = np.average(samples, axis=0, weights=weights)
    centered = samples - mean
    covariance = np.einsum(
        "ni,nj,n->ij", centered, centered, weights, optimize=True
    ) / np.sum(weights)
    covariance = 0.5 * (covariance + covariance.T)
    w_covariance = covariance[2:, 2:]
    np.linalg.cholesky(w_covariance)
    quantiles = np.stack(
        [weighted_quantile(samples[:, i], weights, (0.16, 0.5, 0.84))
         for i in range(samples.shape[1])]
    )
    best = samples[np.argmin(minuslogpost)]
    injection = load_injection()
    model = GWOnlyLikelihood(
        load_catalog(n_events),
        grid_size=grid_size,
        prior_amplitude=prior_amplitude,
    )
    mean_components = model.components(mean)
    best_components = model.components(best)
    mean_w = mean[2:]
    covered = (injection >= quantiles[2:, 0]) & (injection <= quantiles[2:, 2])
    snr = mahalanobis(mean_w + 1.0, w_covariance)
    injection_distance = mahalanobis(mean_w - injection, w_covariance)
    # Neff is defined for the production A=1 posterior.  A=0.01 chains are
    # auxiliary chains used only to recover the prior-subtracted data Fisher.
    neff = (
        float(N_W - np.trace(model.prior_precision @ w_covariance))
        if prior_amplitude == 1.0
        else None
    )

    _, z_lower, z_upper, z_centres = w_geometry()
    # Preserve the established A=1 product names while keeping the weak-prior
    # auxiliary products visibly distinct.
    amplitude_tag = "_prior" if prior_amplitude == 1.0 else "_prior_A001"
    stem = f"gw{n_events:04d}{amplitude_tag}_wz_mcmc"
    table_root.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    table_path = table_root / f"{stem}_reconstruction.csv"
    covariance_path = table_root / f"{stem}_covariance.csv"
    summary_path = table_root / f"{stem}_summary.json"
    figure_path = FIGURES / f"{stem}.png"
    with table_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "bin", "z_min", "z_max", "z_center", "w_mean", "w_q16",
                "w_median", "w_q84", "w_sigma", "w_best_chain_sample",
                "w_injected", "injected_in_68pct",
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
                    quantiles[i + 2, 0],
                    quantiles[i + 2, 1],
                    quantiles[i + 2, 2],
                    math.sqrt(w_covariance[i, i]),
                    best[i + 2],
                    injection[i],
                    bool(covered[i]),
                )
            )
    np.savetxt(
        covariance_path,
        w_covariance,
        fmt="%.12e",
        delimiter=",",
        header=",".join(f"w{i}" for i in range(1, N_W + 1)),
        comments="",
    )
    write_figure(
        figure_path,
        lower=z_lower,
        upper=z_upper,
        mean=mean_w,
        q16=quantiles[2:, 0],
        q84=quantiles[2:, 2],
        map_w=best[2:],
        injected_w=injection,
        n_events=n_events,
        interval_label="68% MCMC credible interval",
        title_suffix=f"(A={prior_amplitude:g} MCMC)",
        center_label="posterior mean",
        fit_label="best retained sample",
    )
    catalog = load_catalog(n_events)
    summary: dict[str, object] = {
        "analysis": f"{n_events} simulated ET standard sirens + modified Horndeski correlation prior only",
        "posterior_method": "Cobaya Metropolis-Hastings MCMC",
        "prior_amplitude": prior_amplitude,
        "sampling_acceptance": (
            "Posterior products require the independent second-half rank-normalized "
            "and folded split-Rhat test below to pass for every parameter; the Cobaya "
            "checkpoint flag is not used as the final acceptance criterion."
        ),
        "included_likelihood_terms": ["GW_absolute", "Horndeski_prior"],
        "explicitly_excluded": ["DESI_DR2_BAO", "CMB_theta_star", "BBN_omega_b_h2", "omega_b"],
        "sampled_parameters": list(PARAMETER_NAMES),
        "catalog": {
            "path": project_path(catalog.path),
            "sha256": sha256(catalog.path),
            "n_events": n_events,
        },
        "n_independent_chains": len(files),
        "chain_files": [project_path(path) for path in files],
        "total_retained_weighted_draws": int(np.sum(weights)),
        "rank_folded_split_rhat": rhat,
        "posterior_mean": {name: float(value) for name, value in zip(PARAMETER_NAMES, mean)},
        "posterior_quantiles_16_50_84": {
            name: [float(value) for value in quantiles[i]]
            for i, name in enumerate(PARAMETER_NAMES)
        },
        "best_retained_sample": {
            name: float(value) for name, value in zip(PARAMETER_NAMES, best)
        },
        "chi2_components_at_posterior_mean": mean_components,
        "chi2_components_at_best_retained_sample": best_components,
        "diagnostics": {
            "snr_posterior_mean_vs_w_minus_one": snr,
            "snr_definition": "sqrt[(w+1)^T Cov(w)^-1 (w+1)]",
            "posterior_mean_distance_from_injected_w": injection_distance,
            "injected_bins_inside_marginal_68pct": int(np.count_nonzero(covered)),
            "effective_w_degrees_of_freedom": neff,
            "effective_w_dof_definition": (
                "29 - Tr(P_Horndeski Cov(w)); defined only for A=1"
            ),
            "posterior_probability_Omega_m_lt_0p05": float(
                np.sum(weights[samples[:, 1] < 0.05]) / np.sum(weights)
            ),
            "posterior_probability_any_w_at_box_edge_0p05": float(
                np.sum(weights[np.any(
                    (samples[:, 2:] < LOWER[2] + 0.05)
                    | (samples[:, 2:] > UPPER[2] - 0.05), axis=1
                )]) / np.sum(weights)
            ),
        },
        "outputs": {
            "reconstruction_csv": project_path(table_path),
            "w_covariance": project_path(covariance_path),
            "summary": project_path(summary_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def run_chains(
    args: argparse.Namespace,
    *,
    prior_amplitude: float = 1.0,
    default_root: Path = A1_ROOT,
    table_root: Path = TABLES,
) -> None:
    from mpi4py import MPI

    communicator = MPI.COMM_WORLD
    if communicator.size < 4:
        raise SystemExit("Formal sampling requires at least four MPI processes")
    prefix = prefix_for(args, default_root=default_root)
    if communicator.rank == 0:
        existing = chain_files(prefix) or list(prefix.parent.glob(prefix.name + ".*.yaml"))
        if existing and not args.resume:
            error = f"Existing output at {prefix}; use --resume or another --output-root"
        elif args.resume and not existing:
            error = f"No output to resume at {prefix}"
        else:
            prefix.parent.mkdir(parents=True, exist_ok=True)
            error = None
    else:
        error = None
    error = communicator.bcast(error, root=0)
    if error:
        raise FileExistsError(error)

    model = GWOnlyLikelihood(
        load_catalog(args.n_events),
        grid_size=args.grid_size,
        prior_amplitude=prior_amplitude,
    )
    info, proposal_source = cobaya_info(model, prefix, args)
    if communicator.rank == 0:
        print(
            f"GW{args.n_events}, prior A={prior_amplitude:g}: sampling "
            f"{len(PARAMETER_NAMES)} parameters on {communicator.size} MPI chains"
        )
        print(f"Initial proposal source: {proposal_source}")
    _, sampler = run(info, resume=args.resume)
    communicator.Barrier()
    if communicator.rank == 0:
        metadata = {
            "catalog": project_path(model.data.path),
            "catalog_sha256": sha256(model.data.path),
            "included_terms": ["GW_absolute", "Horndeski_prior"],
            "prior_amplitude": prior_amplitude,
            "excluded_terms": ["DESI", "CMB", "BBN", "omega_b"],
            "mpi_processes": communicator.size,
            "cobaya_converged": bool(getattr(sampler, "converged", False)),
            "proposal_source": proposal_source,
        }
        prefix.with_suffix(".metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        try:
            result = postprocess(
                prefix,
                args.n_events,
                args.grid_size,
                prior_amplitude,
                table_root=table_root,
            )
        except ValueError as exc:
            print(f"GW{args.n_events}: posterior products pending: {exc}")
        else:
            print(
                f"GW{args.n_events}: converged; SNR="
                f"{result['diagnostics']['snr_posterior_mean_vs_w_minus_one']:.4f}"
            )
    communicator.Barrier()


def main() -> None:
    args = parse_args()
    if args.run_four or args.status_only:
        run_four_catalogues(args)
    elif args.check:
        model = GWOnlyLikelihood(
            load_catalog(args.n_events),
            grid_size=args.grid_size,
            prior_amplitude=1.0,
        )
        point = np.asarray([FIDUCIAL_H0, FIDUCIAL_OMEGA_M, *load_injection()])
        print(json.dumps(model.components(point), indent=2))
        print(f"catalog SHA256: {sha256(model.data.path)}")
    elif args.postprocess:
        result = postprocess(
            prefix_for(args),
            args.n_events,
            args.grid_size,
            1.0,
        )
        print(json.dumps(result["diagnostics"], indent=2))
    else:
        run_chains(args, prior_amplitude=1.0, default_root=A1_ROOT)


if __name__ == "__main__":
    main()
