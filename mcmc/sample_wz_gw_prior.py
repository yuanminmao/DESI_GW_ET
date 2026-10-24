#!/usr/bin/env python3
"""Formal Cobaya MCMC for the GW-only + correlation-prior w(z) model.

This is the sampling companion to ``reconstruct_wz_gw_prior.py``.  It varies
H0, Omega_m and all 29 w bins under the broad flat bounds, and its external
log likelihood is exactly

    -2 log L = chi2_GW + chi2_Horndeski-prior.

DESI, CMB, BBN and omega_b are absent.  Four or more MPI processes are needed
for the formal run.  Posterior products are written only after every sampled
parameter passes rank-normalized and folded split Rhat - 1 < 0.01.

    mpiexec -n 4 python3 mcmc/sample_wz_gw_prior.py --run --n-events 3000
    mpiexec -n 4 python3 mcmc/sample_wz_gw_prior.py --run --n-events 3000 --resume
    python3 mcmc/sample_wz_gw_prior.py --postprocess --n-events 3000

For the weak-prior chains required by the PCA data-Fisher subtraction::

    mpiexec -n 4 python3 mcmc/sample_wz_gw_prior.py --run \
        --n-events 3000 --prior-amplitude 0.01
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
from cobaya import run
from scipy.stats import norm, rankdata

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from reconstruct_wz_gw_prior import (  # noqa: E402
    CHAINS,
    FIGURES,
    LOWER,
    N_W,
    PARAMETER_NAMES,
    TABLES,
    UPPER,
    FIDUCIAL_H0,
    FIDUCIAL_OMEGA_M,
    GWOnlyLikelihood,
    load_catalog,
    load_injection,
    mahalanobis,
    sha256,
    w_geometry,
    write_figure,
)

A1_ROOT = CHAINS / "COBAYA_GW_PRIOR_wz"
A001_ROOT = CHAINS / "COBAYA_GW_PRIOR_A001_wz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--postprocess", action="store_true")
    mode.add_argument("--check", action="store_true")
    parser.add_argument("--n-events", type=int, default=3000)
    parser.add_argument("--grid-size", type=int, default=3200)
    parser.add_argument("--max-samples", type=int, default=250_000)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument(
        "--prior-amplitude", type=float, choices=(1.0, 0.01), default=1.0,
        help="Correlation-prior amplitude: 1 for reconstruction/Neff, 0.01 for PCA Fisher extraction",
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.resume and not args.run:
        parser.error("--resume requires --run")
    if args.n_events < 1 or args.grid_size < 1000 or args.max_samples < 1:
        parser.error("invalid event count, grid size or sample limit")
    return args


def prefix_for(args: argparse.Namespace) -> Path:
    if args.output_root is None:
        root = A1_ROOT if args.prior_amplitude == 1.0 else A001_ROOT
    else:
        root = args.output_root
    return root.expanduser().resolve() / f"GW{args.n_events:04d}" / "chain"


def laplace_proposal(
    n_events: int, prior_amplitude: float
) -> tuple[np.ndarray, np.ndarray, str]:
    """Use the first-look bounded draws only as an adaptive-MCMC initializer."""
    summary_path = TABLES / f"gw{n_events:04d}_prior_wz_summary.json"
    sample_path = CHAINS / f"GW{n_events:04d}_PRIOR_wz_laplace" / "bounded_laplace_samples.npz"
    catalog = load_catalog(n_events)
    if summary_path.is_file() and sample_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("catalog", {}).get("sha256") == sha256(catalog.path):
            archive = np.load(sample_path)
            samples = np.asarray(archive["samples"], dtype=float)
            if samples.ndim == 2 and samples.shape[1] == len(PARAMETER_NAMES):
                mean = np.mean(samples, axis=0)
                covariance = np.cov(samples, rowvar=False, ddof=0)
                covariance = 0.5 * (covariance + covariance.T)
                np.linalg.cholesky(covariance)
                if prior_amplitude == 1.0:
                    return mean, covariance, str(sample_path.resolve())
                # A weak-prior posterior is much broader.  Inflate the A=1
                # initializer without changing its correlations; Cobaya then
                # adapts it using the actual A=0.01 target.
                scales = np.asarray((1.5, 2.0) + (4.0,) * N_W)
                weak_covariance = covariance * scales[:, None] * scales[None, :]
                weak_covariance += np.diag(
                    np.asarray((0.05, 0.002) + (0.02,) * N_W) ** 2
                )
                np.linalg.cholesky(weak_covariance)
                return (
                    mean,
                    weak_covariance,
                    f"inflated A=1 bounded-Laplace initializer: {sample_path.resolve()}",
                )

    injection = load_injection()
    mean = np.asarray([FIDUCIAL_H0, FIDUCIAL_OMEGA_M, *injection])
    scales = np.asarray((2.0, 0.08) + (0.45,) * N_W)
    covariance = np.diag(scales**2)
    return mean, covariance, "diagonal fallback around the injected history"


def cobaya_info(
    model: GWOnlyLikelihood,
    prefix: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, object], str]:
    reference, covariance, proposal_source = laplace_proposal(
        args.n_events, args.prior_amplitude
    )
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
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    table_path = TABLES / f"{stem}_reconstruction.csv"
    covariance_path = TABLES / f"{stem}_covariance.txt"
    summary_path = TABLES / f"{stem}_summary.json"
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
        header="29x29 marginalized w-bin covariance from converged GW+prior Cobaya chains",
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
            "path": str(catalog.path),
            "sha256": sha256(catalog.path),
            "n_events": n_events,
        },
        "n_independent_chains": len(files),
        "chain_files": [str(path.resolve()) for path in files],
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
            "reconstruction_csv": str(table_path.resolve()),
            "w_covariance": str(covariance_path.resolve()),
            "figure": str(figure_path.resolve()),
            "summary": str(summary_path.resolve()),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def run_chains(args: argparse.Namespace) -> None:
    from mpi4py import MPI

    communicator = MPI.COMM_WORLD
    if communicator.size < 4:
        raise SystemExit("Formal sampling requires at least four MPI processes")
    prefix = prefix_for(args)
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
        prior_amplitude=args.prior_amplitude,
    )
    info, proposal_source = cobaya_info(model, prefix, args)
    if communicator.rank == 0:
        print(
            f"GW{args.n_events}, prior A={args.prior_amplitude:g}: sampling "
            f"{len(PARAMETER_NAMES)} parameters on {communicator.size} MPI chains"
        )
        print(f"Initial proposal source: {proposal_source}")
    _, sampler = run(info, resume=args.resume)
    communicator.Barrier()
    if communicator.rank == 0:
        metadata = {
            "catalog": str(model.data.path),
            "catalog_sha256": sha256(model.data.path),
            "included_terms": ["GW_absolute", "Horndeski_prior"],
            "prior_amplitude": args.prior_amplitude,
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
                prefix, args.n_events, args.grid_size, args.prior_amplitude
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
    if args.check:
        model = GWOnlyLikelihood(
            load_catalog(args.n_events),
            grid_size=args.grid_size,
            prior_amplitude=args.prior_amplitude,
        )
        point = np.asarray([FIDUCIAL_H0, FIDUCIAL_OMEGA_M, *load_injection()])
        print(json.dumps(model.components(point), indent=2))
        print(f"catalog SHA256: {sha256(model.data.path)}")
    elif args.postprocess:
        result = postprocess(
            prefix_for(args),
            args.n_events,
            args.grid_size,
            args.prior_amplitude,
        )
        print(json.dumps(result["diagnostics"], indent=2))
    else:
        run_chains(args)


if __name__ == "__main__":
    main()
