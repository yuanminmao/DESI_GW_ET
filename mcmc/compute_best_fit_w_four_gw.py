#!/usr/bin/env python3
"""Refine the joint best-fit 29-bin w(z) histories for four ET GW catalogues.

The four cases contain 100, 500, 1000, and 3000 simulated ET GW events. For
each catalogue, the objective is chi2_data + chi2_prior with unit Horndeski
prior strength. The fit varies H0, Omega_m, omega_b, and all 29 w bins
together, starting a bounded local minimization from the best retained MCMC
sample. This is an auxiliary product for the 29-bin reconstruction and PCA;
it is not the direct one-parameter injected-amplitude posterior P(A|data).

Run all four cases:
    python3 mcmc/compute_best_fit_w_four_gw.py

Run selected cases:
    python3 mcmc/compute_best_fit_w_four_gw.py --n-events 100 1000

Outputs are written to results/tables/*_bestfit_horndeski.csv and
*_bestfit_horndeski_summary.json.
This is a local refinement from the best chain sample; global optimality is
not established by a local optimizer.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from bbn_cmb_desi_absolute_gw_wz_model import (  # noqa: E402
    LOWER, N_W, PARAMETER_NAMES, RESULTS, UPPER, EarlyBackground,
    JointLikelihood, catalog_path, horndeski_precision, load_observations,
    w_geometry,
)

EVENT_COUNTS = (100, 500, 1000, 3000)
DATA_CHI2_KEYS = ("GW_absolute", "DESI_DR2_BAO", "CMB_theta_star", "BBN_omega_b_h2")
OUTPUT_STEM = "desi_dr2_bbn_cmb_absolute_gw_bestfit_horndeski"
PARAMETER_SCALE = np.array([10.0, 0.1, 0.001] + [1.0] * N_W)


def prior_matrices() -> tuple[np.ndarray, np.ndarray]:
    """Return five-bin smoothing S and the Gu et al. covariance C_Pi."""
    a, _, _, _ = w_geometry()
    variance = 0.05 + 0.8 * a**2
    separation = np.abs(np.log(a[:, None] / a[None, :]))
    correlation = np.exp(-((separation / 0.3) ** 1.2))
    covariance = np.sqrt(variance[:, None] * variance[None, :]) * correlation
    smoothing = np.zeros((N_W, N_W))
    for i in range(N_W):
        neighbours = slice(max(0, i - 2), min(N_W, i + 3))
        smoothing[i, neighbours] = 1.0 / (neighbours.stop - neighbours.start)
    return smoothing, covariance


def prior_terms(w: np.ndarray, smoothing: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    w_fid = smoothing @ w
    delta = w - w_fid
    chi2 = float(delta @ np.linalg.solve(covariance, delta))
    return w_fid, delta, chi2


def load_chain_start(n_events: int, *, preview: bool = False) -> tuple[np.ndarray, Path]:
    suffix = "_preview" if preview else ""
    path = RESULTS / "tables" / f"desi_dr2_bbn_cmb_absolute_gw{n_events:04d}_wz{suffix}_summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"Converged MCMC summary is missing: {path}")
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("n_gw") != n_events or (not preview and not summary.get("rank_folded_split_rhat", {}).get("all_pass")):
        raise ValueError(f"GW{n_events} summary is not a converged reconstruction: {path}")
    try:
        key = "best_chain_sample" if preview else "map_sample"
        point = np.array([summary[key][name] for name in PARAMETER_NAMES], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"GW{n_events} summary has no complete joint MAP sample: {path}") from exc
    if point.shape != LOWER.shape or not np.all(np.isfinite(point)):
        raise ValueError(f"GW{n_events} MAP sample has invalid parameters: {path}")
    return point, path


def total_chi2(components: dict[str, float]) -> float:
    return sum(components[key] for key in DATA_CHI2_KEYS) + components["Horndeski_prior"]


def fit_case(n_events: int, *, maxiter: int, refine: bool,
             smoothing: np.ndarray, covariance: np.ndarray,
             resume_point: np.ndarray | None = None,
             preview: bool = False) -> tuple[list[dict], dict]:
    chain_start, source_summary = load_chain_start(n_events, preview=preview)
    model = JointLikelihood(load_observations(n_events), EarlyBackground())
    chain_chi2 = total_chi2(model.components(chain_start))
    start = chain_start
    start_source = "best retained MCMC sample"
    if resume_point is not None:
        resumed_chi2 = total_chi2(model.components(resume_point))
        if resumed_chi2 < chain_chi2:
            start = resume_point
            start_source = "previous local optimization output"
    start_components = model.components(start)
    start_chi2 = total_chi2(start_components)
    best_point = start.copy()
    best_chi2 = start_chi2
    evaluations = 0
    optimizer_result = None

    if refine:
        scaled_bounds = list(zip((LOWER - start) / PARAMETER_SCALE,
                                 (UPPER - start) / PARAMETER_SCALE))

        def objective(scaled: np.ndarray) -> float:
            nonlocal best_point, best_chi2, evaluations
            evaluations += 1
            point = start + PARAMETER_SCALE * scaled
            chi2 = -2.0 * model.loglike(point)
            if not math.isfinite(chi2):
                return 1e12
            if chi2 < best_chi2:
                best_point, best_chi2 = point.copy(), chi2
            return chi2

        optimizer_result = minimize(
            objective, np.zeros(len(start)), method="L-BFGS-B", jac="2-point",
            bounds=scaled_bounds,
            options={"maxiter": maxiter, "ftol": 1e-10, "gtol": 1e-5,
                     "finite_diff_rel_step": 1e-5, "maxls": 30},
        )

    components = model.components(best_point)
    chi2_data = sum(components[key] for key in DATA_CHI2_KEYS)
    w = best_point[3:]
    w_fid, delta, chi2_prior = prior_terms(w, smoothing, covariance)
    model_prior = components["Horndeski_prior"]
    if not np.isclose(chi2_prior, model_prior, rtol=1e-7, atol=1e-8):
        raise ValueError(f"GW{n_events}: explicit Horndeski prior differs from the likelihood")
    if not np.isclose(chi2_data + chi2_prior, best_chi2, rtol=1e-7, atol=1e-6):
        raise ValueError(f"GW{n_events}: optimized objective differs from recomputed chi2")
    _, z_lo, z_hi, z_center = w_geometry()
    rows = [
        {"n_gw": n_events, "bin": i + 1, "z_min": float(z_lo[i]),
         "z_max": float(z_hi[i]), "z_center": float(z_center[i]),
         "w_bestfit": float(w[i]), "w_fid_5bin": float(w_fid[i]),
         "delta_w": float(delta[i])}
        for i in range(N_W)
    ]
    report = {
        "n_gw": n_events,
        "catalog": str(catalog_path(n_events)),
        "start_summary": str(source_summary),
        "source_mcmc_converged": not preview,
        "start_source": start_source,
        "chain_start_chi2_total": chain_chi2,
        "start_chi2_total": start_chi2,
        "chi2_data": chi2_data,
        "chi2_prior": chi2_prior,
        "chi2_total": chi2_data + chi2_prior,
        "chi2_components": {key: components[key] for key in (*DATA_CHI2_KEYS, "Horndeski_prior")},
        "parameters": {name: float(value) for name, value in zip(PARAMETER_NAMES, best_point)},
        "optimization": {
            "method": "L-BFGS-B local refinement of the joint chi2 objective" if refine else start_source,
            "converged": bool(optimizer_result.success) if optimizer_result is not None else None,
            "message": str(optimizer_result.message) if optimizer_result is not None else "refinement disabled",
            "iterations": int(optimizer_result.nit) if optimizer_result is not None else 0,
            "objective_evaluations": evaluations,
        },
    }
    return rows, report


def load_resume_points(path: Path) -> dict[int, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"No prior optimization output to resume: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("A") != 1.0:
        raise ValueError(f"Prior optimization output does not use A=1: {path}")
    points = {}
    for case in payload["cases"]:
        point = np.array([case["parameters"][name] for name in PARAMETER_NAMES], dtype=float)
        if point.shape != LOWER.shape or not np.all(np.isfinite(point)):
            raise ValueError(f"Invalid resume point for GW{case['n_gw']}: {path}")
        points[int(case["n_gw"])] = point
    return points


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-events", type=int, nargs="+", default=list(EVENT_COUNTS),
                        choices=EVENT_COUNTS, help="GW catalogues to process (default: all four)")
    parser.add_argument("--maxiter", type=int, default=500, help="Maximum local optimizer iterations per case")
    parser.add_argument("--chain-only", action="store_true", help="Use the best MCMC sample without local refinement")
    parser.add_argument("--resume", action="store_true", help="Continue from a previous output when it improved on the MCMC start")
    parser.add_argument("--start-from", type=Path, help="Use a compatible prior A=1 optimization summary as an alternate start")
    parser.add_argument("--preview", action="store_true", help="Use an unconverged GW100 preview summary and separate preview outputs")
    parser.add_argument("--out-root", type=Path, default=RESULTS / "tables")
    args = parser.parse_args()
    if args.maxiter < 1 or len(set(args.n_events)) != len(args.n_events):
        parser.error("--maxiter must be positive and --n-events values must be distinct")
    if args.preview and args.n_events != [100]:
        parser.error("--preview requires --n-events 100")
    smoothing, covariance = prior_matrices()
    difference = np.eye(N_W) - smoothing
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        derived_precision = difference.T @ np.linalg.solve(covariance, difference)
    if not np.allclose(derived_precision, horndeski_precision(), rtol=1e-8, atol=1e-9):
        raise ValueError("Explicit Gu et al. prior does not match the MCMC likelihood")

    out_root = args.out_root.expanduser().resolve()
    output_stem = ("desi_dr2_bbn_cmb_absolute_gw0100_preview_bestfit_horndeski"
                   if args.preview else
                   "desi_dr2_bbn_cmb_absolute_gw0100_bestfit_horndeski"
                   if args.n_events == [100] else OUTPUT_STEM)
    json_path = out_root / f"{output_stem}_summary.json"
    resume_points = load_resume_points(args.start_from) if args.start_from else (
        load_resume_points(json_path) if args.resume else {})
    all_rows, cases = [], []
    for n_events in args.n_events:
        rows, case = fit_case(n_events, maxiter=args.maxiter, refine=not args.chain_only,
                              smoothing=smoothing, covariance=covariance,
                              resume_point=resume_points.get(n_events), preview=args.preview)
        all_rows.extend(rows)
        cases.append(case)
        print(f"GW{n_events}: chi2_data={case['chi2_data']:.6f}, "
              f"chi2_prior={case['chi2_prior']:.6f}, chi2_total={case['chi2_total']:.6f}; "
              f"iterations={case['optimization']['iterations']}, "
              f"converged={case['optimization']['converged']}", flush=True)

    out_root.mkdir(parents=True, exist_ok=True)
    csv_path = out_root / f"{output_stem}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    payload = {
        "method": "joint minimum of chi2_data + A*chi2_prior over H0, Omega_m, omega_b, w1..w29",
        "A": 1.0,
        "prior": "D_w=(I-S)w; S=five-bin running average; chi2_prior=D_w^T C_Pi^-1 D_w",
        "prior_source": "Gu et al., Nature Astronomy 2025, DOI 10.1038/s41550-025-02669-6, Methods eqs. (6)-(8)",
        "global_optimum_proven": False,
        "source_mcmc_converged": not args.preview,
        "cases": cases,
        "csv": str(csv_path),
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved: {csv_path}\nSaved: {json_path}")
    if not args.preview and out_root == (RESULTS / "tables").resolve():
        from reconstruct_wz_bbn_cmb_desi_absolute_gw import render_optimized_reconstruction
        for n_events in args.n_events:
            if render_optimized_reconstruction(n_events):
                print(f"GW{n_events}: optimized best-fit reconstruction figure updated")


if __name__ == "__main__":
    main()
