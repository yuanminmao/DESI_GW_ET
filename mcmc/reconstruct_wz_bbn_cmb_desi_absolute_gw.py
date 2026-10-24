#!/usr/bin/env python3
"""Fit and reconstruct non-parametric w(z) from BBN+CMB+DESI DR2+ET GW.

The GW100, GW500, GW1000 and GW3000 realization_0042 catalogues can replace
the article's SN Ia luminosity-distance data, one catalogue per independent fit.
All 32 parameters (H0, Omega_m, omega_b h^2, w1..w29) are sampled jointly.
The GW term uses every event's absolute dL and sigma, with the same H0 used
by the other probes. The paper's SNR is computed from the converged 29-bin
posterior covariance: sqrt[(w_mean+1)^T C_w^-1 (w_mean+1)].

Run the file directly (for example with an editor Run button) to verify all
inputs and likelihood terms without starting a long MCMC. This is equivalent
to:
    python3 mcmc/reconstruct_wz_bbn_cmb_desi_absolute_gw.py --check

Formal run with four independent Cobaya MPI chains per catalogue:
    mpiexec -n 4 python3 mcmc/reconstruct_wz_bbn_cmb_desi_absolute_gw.py --run

Run or resume GW100 automatically, then validate and plot its results:
    python3 mcmc/reconstruct_wz_bbn_cmb_desi_absolute_gw.py --gw100

Rebuild tables, covariance and SNR from completed chains:
    python3 mcmc/reconstruct_wz_bbn_cmb_desi_absolute_gw.py --postprocess-only

Refine the best retained chain sample and draw the optimized best-fit curves:
    python3 mcmc/compute_best_fit_w_four_gw.py

Use --output-root NEW_DIRECTORY for a fresh run or --resume for compatible
existing chains. A reconstruction/SNR is emitted only when every sampled
parameter passes rank/folded split Rhat-1 < 0.01 on weighted retained draws.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

# Apple's Accelerate backend in NumPy 2 can warn spuriously during SciPy's
# finite, positive-definite multivariate-normal proposal calculations. The
# proposal and posterior covariance matrices are checked explicitly below.
warnings.filterwarnings(
    "ignore", message=r"(divide by zero|overflow|invalid value) encountered in matmul",
    category=RuntimeWarning,
)

import numpy as np
from cobaya import run
from scipy.stats import norm, rankdata

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from bbn_cmb_desi_absolute_gw_wz_model import (  # noqa: E402
    LOWER,
    N_W,
    PARAMETER_NAMES,
    RESULTS,
    UPPER,
    EarlyBackground,
    JointLikelihood,
    catalog_path,
    load_observations,
    w_geometry,
)
DEFAULT_EVENTS = (500, 1000, 3000)
SUPPORTED_EVENTS = (100,) + DEFAULT_EVENTS
DEFAULT_ROOT = RESULTS / "chains" / "COBAYA_DESI_DR2_BBN_CMB_ABSOLUTE_GW_wz"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
GW100_STEM = "desi_dr2_bbn_cmb_absolute_gw0100_wz"
GW100_FIGURE = FIGURES / "wz_posterior_cov_bbn_cmb_desi_absolute_gw0100.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Validate inputs and evaluate one point without MCMC")
    mode.add_argument("--run", action="store_true", help="Run four-chain Cobaya MCMC on the selected catalogues")
    mode.add_argument("--postprocess-only", action="store_true", help="Calculate reconstruction and SNR from chains")
    mode.add_argument("--gw100", action="store_true", help="Auto-resume, validate and plot the GW100 reconstruction")
    parser.add_argument("--n-events", type=int, nargs="+", default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--grid-size", type=int, default=3200)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum accepted samples per MPI chain and catalogue (GW100: 1000000; others: 300000)")
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--resume", action="store_true", help="Resume compatible existing Cobaya outputs")
    args = parser.parse_args()
    if not (args.check or args.run or args.postprocess_only or args.gw100):
        # A plain editor/IDE Run should do something useful and finish. Formal
        # sampling still requires an explicit four-process MPI command.
        args.check = True
    if args.n_events is None:
        args.n_events = [100] if args.gw100 else list(DEFAULT_EVENTS)
    if args.max_samples is None:
        args.max_samples = 1_000_000 if args.gw100 else 300_000
    if len(args.n_events) != len(set(args.n_events)) or set(args.n_events) - set(SUPPORTED_EVENTS):
        parser.error("--n-events must be distinct values chosen from 100 500 1000 3000")
    if args.gw100 and args.n_events != [100]:
        parser.error("--gw100 only supports --n-events 100")
    if args.gw100 and args.output_root.expanduser().resolve() != DEFAULT_ROOT.resolve():
        parser.error("--gw100 uses the existing default chains output path")
    if args.grid_size < 1000 or args.max_samples < 1:
        parser.error("--grid-size must be >= 1000 and --max-samples must be positive")
    if args.resume and not args.run:
        parser.error("--resume requires --run")
    return args


def initial_point() -> np.ndarray:
    # Fixed, documented reference for initializing the sampler. It is not a
    # posterior mean and never substitutes for a sampled likelihood parameter.
    return np.asarray([57.365, 0.423, 0.02196] + [-1.0] * N_W)


def proposal_covariance() -> np.ndarray:
    sigmas = np.asarray([1.5, 0.02, 0.0004] + [0.35] * N_W)
    covariance = np.diag(sigmas**2)
    indices = np.arange(N_W)
    covariance[3:, 3:] = 0.35**2 * np.exp(-np.abs(indices[:, None] - indices[None, :]) / 2.5)
    return covariance


def cobaya_info(model: JointLikelihood, prefix: Path, args: argparse.Namespace, n_events: int) -> dict:
    reference = initial_point()
    sigmas = np.sqrt(np.diag(proposal_covariance()))
    params = {}
    for i, name in enumerate(PARAMETER_NAMES):
        params[name] = {
            "prior": {"min": float(LOWER[i]), "max": float(UPPER[i])},
            "ref": {"dist": "norm", "loc": float(reference[i]),
                    "scale": float(max(0.1 * sigmas[i], 1e-6))},
            "proposal": float(sigmas[i]),
        }

    def loglike(**values: float) -> float:
        point = np.asarray([values[name] for name in PARAMETER_NAMES])
        return model.loglike(point)

    return {
        "params": params,
        "likelihood": {"DESI_DR2_BBN_CMB_absolute_GW_Horndeski": {
            "external": loglike, "input_params": list(PARAMETER_NAMES),
        }},
        "sampler": {"mcmc": {
            "covmat": proposal_covariance(),
            "covmat_params": list(PARAMETER_NAMES),
            "burn_in": "2d",
            "max_samples": args.max_samples,
            "Rminus1_stop": 0.01,
            "Rminus1_cl_stop": 0.05,
            "learn_proposal": True,
            "learn_every": "40d",
            # The 32-dimensional H0/Omega_m/w degeneracy can leave early
            # independent chains far apart. Cobaya's default threshold 2
            # prevents any covariance learning in that regime; let it learn
            # from within-chain covariance at the first valid check.
            "learn_proposal_Rminus1_max": 100.0,
            "seed": args.seed + n_events,
        }},
        "output": str(prefix),
        "stop_at_error": True,
    }


def chain_files(prefix: Path) -> list[Path]:
    pattern = re.compile(re.escape(prefix.name) + r"\.\d+\.txt$")
    return sorted(path for path in prefix.parent.glob(prefix.name + ".*.txt")
                  if pattern.fullmatch(path.name))


def retained_chain(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    # Native Cobaya layout: repeat weight, minuslogpost, then sampled params.
    columns = tuple(range(2 + len(PARAMETER_NAMES)))
    rows = np.loadtxt(path, comments="#", usecols=columns, ndmin=2)
    if rows.shape[1] != len(columns) or not np.all(np.isfinite(rows)):
        raise ValueError(f"Malformed or non-finite Cobaya chain: {path}")
    raw = rows[:, 0]
    weights = raw.astype(np.int64)
    if np.any(weights <= 0) or not np.array_equal(raw, weights):
        raise ValueError(f"Cobaya repeat weights must be positive integers: {path}")
    total = int(np.sum(weights))
    if total < 2000:
        raise ValueError(f"Chain has too few weighted draws: {path}")
    # Exact overlap of each compressed row with the chronological second half.
    cumulative = np.cumsum(weights)
    retained = np.minimum(weights, np.maximum(0, cumulative - total // 2))
    selected = retained > 0
    return rows[selected, 2:], retained[selected], rows[selected, 1], total


def split_rhat(chains: np.ndarray) -> float:
    m, n = chains.shape
    half = n // 2
    if m < 2 or half < 2:
        return math.inf
    halves = np.concatenate((chains[:, :half], chains[:, -half:]), axis=0)
    within = float(np.mean(np.var(halves, axis=1, ddof=1)))
    if not math.isfinite(within) or within <= 0:
        return math.inf
    between = float(half * np.var(np.mean(halves, axis=1), ddof=1))
    variance = (half - 1.0) * within / half + between / half
    return float(np.sqrt(max(variance / within, 0.0)))


def rank_folded_rhat(groups: list[tuple[np.ndarray, np.ndarray, np.ndarray, int]]) -> dict:
    """Exact repeated-weight rank/folded split Rhat, one parameter at a time."""
    n_common = min(int(np.sum(group[1])) for group in groups)
    if n_common < 1000:
        raise ValueError("Too few retained weighted draws for Rhat")
    diagnostics = {}
    for i, name in enumerate(PARAMETER_NAMES):
        repeated = [np.repeat(group[0][:, i], group[1])[-n_common:] for group in groups]
        values = np.stack(repeated)
        flat = values.reshape(-1)
        ranks = rankdata(flat, method="average")
        z = norm.ppf((ranks - 0.375) / (len(flat) + 0.25)).reshape(values.shape)
        rank = split_rhat(z)
        folded = np.abs(flat - np.median(flat))
        folded_ranks = rankdata(folded, method="average")
        folded_z = norm.ppf((folded_ranks - 0.375) / (len(flat) + 0.25)).reshape(values.shape)
        folded_rhat = split_rhat(folded_z)
        diagnostics[name] = {
            "rank_rhat": rank, "folded_rhat": folded_rhat,
            "max_rhat": max(rank, folded_rhat),
        }
    maximum = max(value["max_rhat"] for value in diagnostics.values())
    return {"common_retained_draws_per_chain": n_common,
            "max_rhat_minus_one": maximum - 1.0,
            "all_pass": bool(maximum - 1.0 < 0.01),
            "per_parameter": diagnostics}


def weighted_quantile(values: np.ndarray, weights: np.ndarray, probabilities: tuple[float, ...]) -> np.ndarray:
    order = np.argsort(values)
    sorted_values, sorted_weights = values[order], weights[order]
    cumulative = np.cumsum(sorted_weights)
    targets = np.asarray(probabilities) * cumulative[-1]
    indices = np.searchsorted(cumulative, targets, side="left")
    return sorted_values[np.clip(indices, 0, len(sorted_values) - 1)]


def write_plot(path: Path, z_lower: np.ndarray, z_upper: np.ndarray,
               q16: np.ndarray, best_fit: np.ndarray, q84: np.ndarray, n_events: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    edges = np.r_[z_lower, z_upper[-1]]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.fill_between(edges, np.r_[q16, q16[-1]], np.r_[q84, q84[-1]],
                    step="post", color="#1f77b4", alpha=0.3, label="68% credible")
    ax.step(edges, np.r_[best_fit, best_fit[-1]], where="post", color="#1f77b4",
            linewidth=2, label="best fit (locally optimized)")
    ax.axhline(-1.0, color="black", linestyle="--", linewidth=1.2, label="LCDM")
    ax.set(xlabel="Redshift z", ylabel="w(z)", xlim=(0, 2.5),
           title=f"DESI DR2 + BBN + CMB + GW{n_events}")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def optimized_fit_paths(n_events: int) -> tuple[Path, Path]:
    if n_events == 100:
        stem = "desi_dr2_bbn_cmb_absolute_gw0100_bestfit_horndeski"
    else:
        stem = "desi_dr2_bbn_cmb_absolute_gw_bestfit_horndeski"
    return TABLES / f"{stem}.csv", TABLES / f"{stem}_summary.json"


def render_optimized_reconstruction(n_events: int) -> bool:
    """Plot the converged local chi2 minimum once its fit products exist."""
    stem = f"desi_dr2_bbn_cmb_absolute_gw{n_events:04d}_wz"
    summary_path = TABLES / f"{stem}_summary.json"
    table_path = TABLES / f"{stem}.csv"
    best_table, best_summary = optimized_fit_paths(n_events)
    if not (best_table.is_file() and best_summary.is_file()):
        return False

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    fit = json.loads(best_summary.read_text(encoding="utf-8"))
    cases = [case for case in fit.get("cases", []) if case.get("n_gw") == n_events]
    if fit.get("A") != 1.0 or len(cases) != 1:
        raise ValueError(f"GW{n_events}: missing unique A=1 optimized fit")
    case = cases[0]
    if not (summary.get("rank_folded_split_rhat", {}).get("all_pass")
            and case.get("optimization", {}).get("converged")
            and Path(case["start_summary"]).resolve() == summary_path.resolve()):
        raise ValueError(f"GW{n_events}: optimized fit is not tied to a converged reconstruction")

    model = JointLikelihood(load_observations(n_events), EarlyBackground())
    chain_point = np.asarray([summary["map_sample"][name] for name in PARAMETER_NAMES])
    fit_point = np.asarray([case["parameters"][name] for name in PARAMETER_NAMES])
    chain_chi2 = -2.0 * model.loglike(chain_point)
    fit_chi2 = -2.0 * model.loglike(fit_point)
    if not np.isfinite(chain_chi2):
        raise ValueError(f"GW{n_events}: non-finite best retained chain sample")
    if not np.isclose(chain_chi2, case["chain_start_chi2_total"], rtol=1e-8, atol=1e-6):
        return False  # Chains changed after this local fit; rerun the optimizer.
    if not (np.isfinite(fit_chi2)
            and np.isclose(fit_chi2, case["chi2_total"], rtol=1e-8, atol=1e-6)
            and fit_chi2 <= chain_chi2 + 1e-8):
        raise ValueError(f"GW{n_events}: optimized fit has inconsistent chi2")

    with table_path.open(newline="", encoding="utf-8") as stream:
        reconstruction = list(csv.DictReader(stream))
    with best_table.open(newline="", encoding="utf-8") as stream:
        best_rows = [row for row in csv.DictReader(stream) if int(row["n_gw"]) == n_events]
    if len(reconstruction) != N_W or len(best_rows) != N_W:
        raise ValueError(f"GW{n_events}: optimized fit must have {N_W} matching bins")
    z_lower = np.asarray([float(row["z_min"]) for row in reconstruction])
    z_upper = np.asarray([float(row["z_max"]) for row in reconstruction])
    z_centres = np.asarray([float(row["z_center"]) for row in reconstruction])
    q16 = np.asarray([float(row["w_q16"]) for row in reconstruction])
    q84 = np.asarray([float(row["w_q84"]) for row in reconstruction])
    best_fit = np.asarray([float(row["w_bestfit"]) for row in best_rows])
    fit_centres = np.asarray([float(row["z_center"]) for row in best_rows])
    if ([int(row["bin"]) for row in best_rows] != list(range(1, N_W + 1))
            or not np.allclose(fit_centres, z_centres, rtol=0, atol=1e-12)
            or not np.allclose(best_fit, fit_point[3:], rtol=0, atol=1e-12)):
        raise ValueError(f"GW{n_events}: optimized fit CSV and JSON disagree")

    figure = FIGURES / f"{stem}.png"
    FIGURES.mkdir(parents=True, exist_ok=True)
    write_plot(figure, z_lower, z_upper, q16, best_fit, q84, n_events)
    summary["best_fit_curve"] = {
        "definition": "local minimum of data chi2 plus A=1 Horndeski prior, started from the best retained chain sample or a better previous fit",
        "chi2_total": fit_chi2,
        "source": str(best_summary),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return True


def postprocess(prefix: Path, n_events: int) -> dict:
    files = chain_files(prefix)
    if len(files) < 4:
        raise ValueError(f"Four independent native Cobaya chains are required: {prefix}")
    groups = [retained_chain(file) for file in files]
    diagnostic = rank_folded_rhat(groups)
    if not diagnostic["all_pass"]:
        raise ValueError(
            f"GW{n_events}: chains have max rank/folded split Rhat-1 = "
            f"{diagnostic['max_rhat_minus_one']:.5f}; resume sampling before SNR"
        )
    samples = np.concatenate([group[0] for group in groups])
    weights = np.concatenate([group[1] for group in groups])
    minuslogpost = np.concatenate([group[2] for group in groups])
    mean = np.average(samples, axis=0, weights=weights)
    centred = samples - mean
    covariance = np.einsum("ni,nj,n->ij", centred, centred, weights,
                           optimize=True) / np.sum(weights)
    covariance = 0.5 * (covariance + covariance.T)
    w_covariance = covariance[3:, 3:]
    np.linalg.cholesky(w_covariance)  # Fail if covariance is singular/invalid.
    deviation = mean[3:] + 1.0
    snr2 = float(deviation @ np.linalg.solve(w_covariance, deviation))
    snr = math.sqrt(max(snr2, 0.0))
    best_sample = samples[np.argmin(minuslogpost)]
    best_deviation = best_sample[3:] + 1.0
    best_snr2 = float(best_deviation @ np.linalg.solve(w_covariance, best_deviation))
    best_snr = math.sqrt(max(best_snr2, 0.0))
    quantiles = np.stack([
        weighted_quantile(samples[:, i], weights, (0.16, 0.5, 0.84))
        for i in range(samples.shape[1])
    ])
    _, z_lower, z_upper, z_centres = w_geometry()
    stem = f"desi_dr2_bbn_cmb_absolute_gw{n_events:04d}_wz"
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    table = TABLES / f"{stem}.csv"
    cov_path = TABLES / f"{stem}_covariance.txt"
    summary_path = TABLES / f"{stem}_summary.json"
    figure = FIGURES / f"{stem}.png"
    with table.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("bin", "z_min", "z_max", "z_center", "w_mean", "w_q16", "w_median", "w_q84", "w_sigma"))
        for i in range(N_W):
            writer.writerow((i + 1, z_lower[i], z_upper[i], z_centres[i],
                             mean[i + 3], quantiles[i + 3, 0], quantiles[i + 3, 1],
                             quantiles[i + 3, 2], math.sqrt(w_covariance[i, i])))
    np.savetxt(cov_path, w_covariance, fmt="%.12e",
               header="29x29 posterior covariance of w bins; exact Cobaya repeat weights")
    summary = {
        "catalog": str(catalog_path(n_events).resolve()),
        "n_gw": n_events,
        "prior_amplitude": 1.0,
        "prior_definition": "A=1 modified Horndeski correlation prior",
        "probe_combination": "DESI DR2 BAO + BBN omega_b h2 + Planck PR4 theta_* + absolute ET GW",
        "sampled_parameters": list(PARAMETER_NAMES),
        "gw_distance_treatment": "all events, absolute dL; sampled H0 and Omega_m shared with DESI/CMB",
        "n_independent_chains": len(files),
        "total_retained_weighted_draws": int(np.sum(weights)),
        "chain_weighted_draws_before_second_half": [group[3] for group in groups],
        "rank_folded_split_rhat": diagnostic,
        "snr_vs_lcdm": snr,
        "snr_posterior_mean_vs_lcdm": snr,
        "snr_best_chain_sample_vs_lcdm": best_snr,
        "snr_definition": "sqrt[(w_estimate+1)^T C_w^-1 (w_estimate+1)]; both posterior mean and best sampled point reported",
        "joint_posterior_mean": {name: float(value) for name, value in zip(PARAMETER_NAMES, mean)},
        "map_sample": {name: float(value) for name, value in zip(
            PARAMETER_NAMES, best_sample)},
        "outputs": {"reconstruction_csv": str(table), "w_covariance": str(cov_path),
                    "figure": str(figure)},
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if not render_optimized_reconstruction(n_events):
        figure.unlink(missing_ok=True)
        counts = "100" if n_events == 100 else "500 1000 3000"
        print(f"GW{n_events}: best-fit figure pending; run "
              f"mcmc/compute_best_fit_w_four_gw.py --n-events {counts}")
    return summary


def check(args: argparse.Namespace) -> None:
    early = EarlyBackground()
    point = initial_point()
    injected_path = HERE.parent / "data" / "gw_fiducial_w_bins.txt"
    if injected_path.is_file():
        injected_rows = np.loadtxt(injected_path, comments="#", ndmin=2)
        if injected_rows.shape[0] < N_W or injected_rows.shape[1] < 5:
            raise ValueError(f"Malformed simulation-input w bins: {injected_path}")
        injection = point.copy()
        injection[3:] = injected_rows[:N_W, 4]
    else:
        injection = None
    for n_events in args.n_events:
        model = JointLikelihood(load_observations(n_events), early, grid_size=args.grid_size)
        components = model.components(point)
        print(f"GW{n_events}: catalogue rows={len(model.data.gw_z)}; "
              f"GW chi2={components['GW_absolute']:.3f}; "
              f"DESI chi2={components['DESI_DR2_BAO']:.3f}; "
              f"CMB 100theta*={components['theta100_model']:.6f}; "
              f"loglike={model.loglike(point):.3f}")
        if injection is not None:
            injected_components = model.components(injection)
            print(f"GW{n_events}: chi2 per simulated event at injected w(z) = "
                  f"{injected_components['GW_absolute'] / n_events:.3f}")
    print("\nInput and likelihood checks passed.")
    print("Formal four-chain command:")
    counts = " ".join(map(str, args.n_events))
    print(f"  mpiexec -n 4 python3 mcmc/reconstruct_wz_bbn_cmb_desi_absolute_gw.py "
          f"--run --n-events {counts}")


def run_chains(args: argparse.Namespace) -> None:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    if comm.size < 4:
        if comm.rank == 0:
            print("Formal reconstruction needs at least four MPI processes: mpiexec -n 4 python3 ... --run",
                  file=sys.stderr)
        raise SystemExit(2)
    for n_events in args.n_events:
        prefix = args.output_root.expanduser().resolve() / f"GW{n_events:04d}" / "chain"
        if comm.rank == 0:
            existing = chain_files(prefix) or list(prefix.parent.glob(prefix.name + ".*.yaml"))
            if existing and not args.resume:
                state_error = f"Existing Cobaya output at {prefix}; use --resume or --output-root NEW_DIRECTORY"
            elif args.resume and not existing:
                state_error = f"No Cobaya output to resume at {prefix}"
            else:
                prefix.parent.mkdir(parents=True, exist_ok=True)
                state_error = None
        else:
            state_error = None
        state_error = comm.bcast(state_error, root=0)
        if state_error:
            raise FileExistsError(state_error)
        model = JointLikelihood(load_observations(n_events),
                                EarlyBackground(), grid_size=args.grid_size)
        reference_loglike = model.loglike(initial_point())
        if not math.isfinite(reference_loglike):
            raise ValueError(f"Non-finite initial reference likelihood for GW{n_events}")
        if comm.rank == 0:
            print(f"GW{n_events}: sampling all 32 parameters, initial loglike={reference_loglike:.3f}", flush=True)
        updated, sampler = run(cobaya_info(model, prefix, args, n_events), resume=args.resume)
        comm.Barrier()
        if comm.rank == 0:
            metadata = {
                "catalog": str(catalog_path(n_events).resolve()),
                "model_source": str((HERE / "bbn_cmb_desi_absolute_gw_wz_model.py").resolve()),
                "early_universe_background": "exact CAMB at each sampled point; w-bin distance correction integrated",
                "mpi_processes": comm.size,
                "cobaya_converged": bool(getattr(sampler, "converged", False)),
                "chain_files": [str(path) for path in chain_files(prefix)],
            }
            prefix.with_suffix(".metadata.json").write_text(
                json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
            )
            try:
                result = postprocess(prefix, n_events)
            except ValueError as exc:
                print(f"GW{n_events}: posterior/SNR pending: {exc}", flush=True)
            else:
                print(f"GW{n_events}: SNR versus w=-1 = {result['snr_vs_lcdm']:.3f}; "
                      f"table={result['outputs']['reconstruction_csv']}", flush=True)
        comm.Barrier()


def run_command(command: list[str], *, env: dict[str, str] | None = None) -> int:
    print("Running:", " ".join(command), flush=True)
    return subprocess.run(command, cwd=HERE.parent, env=env, check=False).returncode


def run_gw100(args: argparse.Namespace) -> None:
    """Complete the former GW100 convenience workflow in this script."""
    prefix = DEFAULT_ROOT / "GW0100" / "chain"
    table = TABLES / f"{GW100_STEM}.csv"
    covariance = TABLES / f"{GW100_STEM}_covariance.txt"
    summary_path = TABLES / f"{GW100_STEM}_summary.json"

    if not summary_path.is_file():
        mpiexec = shutil.which("mpiexec")
        if mpiexec is None:
            raise SystemExit("mpiexec is required for the four-chain GW100 reconstruction")
        existing = bool(chain_files(prefix)) or prefix.with_suffix(".input.yaml").is_file()
        command = [
            mpiexec, "-n", "4", sys.executable, str(Path(__file__).resolve()),
            "--run", "--n-events", "100", "--max-samples", str(args.max_samples),
        ]
        if existing:
            command.append("--resume")
        env = os.environ.copy()
        env.update({
            "VECLIB_MAXIMUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        })
        status = run_command(command, env=env)
        checkpoint = prefix.with_suffix(".checkpoint")
        if not summary_path.is_file() and checkpoint.is_file() and "converged: true" in checkpoint.read_text(encoding="utf-8"):
            # The launcher can warn after convergence but before postprocessing.
            run_command([sys.executable, str(Path(__file__).resolve()),
                         "--postprocess-only", "--n-events", "100"])
        if not summary_path.is_file():
            raise SystemExit(
                f"GW100 products are pending (MPI exit status {status}). "
                f"Rerun this script with --gw100 to resume the saved chains at {prefix.parent}."
            )
        if status:
            print(f"MPI exited with status {status}; validating completed products.", flush=True)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    diagnostic = summary.get("rank_folded_split_rhat", {})
    if summary.get("n_gw") != 100 or not diagnostic.get("all_pass"):
        raise SystemExit("GW100 summary is not a converged 100-event reconstruction")
    if not table.is_file() or not covariance.is_file():
        raise SystemExit("GW100 summary exists but its posterior table or covariance is missing")

    snr = float(summary["snr_posterior_mean_vs_lcdm"])
    plotter = HERE.parent / "plot" / "plot_wz_posterior_cov.py"
    plot_status = run_command([
        sys.executable, str(plotter), "--table", str(table), "--cov", str(covariance),
        "--out", str(GW100_FIGURE), "--label",
        f"BBN + CMB + DESI DR2 + GW100   SNR={snr:.2f}", "--no-show",
    ])
    if plot_status or not GW100_FIGURE.is_file():
        raise SystemExit("GW100 posterior is ready, but plotting failed")
    print(f"GW100 SNR versus w=-1: {snr:.6f}")
    print(f"Maximum rank/folded Rhat-1: {diagnostic['max_rhat_minus_one']:.6f}")
    print(f"Reconstruction: {table}")
    print(f"Full covariance: {covariance}")
    print(f"Summary: {summary_path}")
    print(f"Figure: {GW100_FIGURE}")


def main() -> None:
    args = parse_args()
    if args.check:
        check(args)
    elif args.gw100:
        run_gw100(args)
    elif args.postprocess_only:
        for n_events in args.n_events:
            prefix = args.output_root.expanduser().resolve() / f"GW{n_events:04d}" / "chain"
            result = postprocess(prefix, n_events)
            print(f"GW{n_events}: SNR versus w=-1 = {result['snr_vs_lcdm']:.3f}; "
                  f"table={result['outputs']['reconstruction_csv']}")
    else:
        run_chains(args)


if __name__ == "__main__":
    main()
