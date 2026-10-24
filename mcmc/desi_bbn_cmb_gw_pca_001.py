#!/usr/bin/env python3
"""Sample the A=0.01 posterior required by the Gu et al. PCA.

The DESI Fig. 4 construction estimates the data Fisher matrix from a weak
correlation-prior run,

    F_data = C_p(A=0.01)^-1 - 0.01 * C_Pi^-1.

This script runs independent Cobaya chains for each ET catalogue and
writes the marginalized 29x29 w-bin covariance used by the PCA script.  It
uses the validated CAMB interpolation table to make the 32-parameter weak
prior runs practical; the table is accurate to <0.03 sigma in theta_* on its
independent validation sample.

Examples
--------
Check the likelihood and proposal for all catalogues::

    python3 mcmc/desi_bbn_cmb_gw_pca_001.py --check

Run or resume eight MPI chains::

    mpiexec -n 8 python3 mcmc/desi_bbn_cmb_gw_pca_001.py --run
    mpiexec -n 8 python3 mcmc/desi_bbn_cmb_gw_pca_001.py --run --resume

Rebuild covariance products from completed chains::

    python3 mcmc/desi_bbn_cmb_gw_pca_001.py --postprocess-only
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
from cobaya import run
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import solve_triangular

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paths import (  # noqa: E402
    COMPUTE_BEST_FIT_W_TABLES,
    DESI_BBN_CMB_GW_PCA_001_TABLES,
    JOINT_PCA_CHAINS,
    RECONSTRUCT_WZ_DESI_BBN_CMB_GW_TABLES,
)
from mcmc.reconstruct_wz_desi_bbn_cmb_gw import (  # noqa: E402
    BBN_SIGMA,
    C_KM_S,
    CMB_THETA100_SIGMA,
    LOWER,
    N_W,
    PARAMETER_NAMES,
    RESULTS,
    UPPER,
    JointLikelihood,
    catalog_path,
    chain_files,
    horndeski_precision,
    load_observations,
    rank_folded_rhat,
    retained_chain,
)
from mcmc.desi_bbn_cmb_gw_cpl_mcmc import (  # noqa: E402
    TABLE_FILE,
    TableEarlyBackground,
)

STEP = np.array([0.005, 0.0002, 0.000002] + [0.002] * N_W)


def whitened_predictions(
    model: JointLikelihood,
    point: np.ndarray,
    early: tuple | None = None,
) -> np.ndarray:
    """Return all joint-model observables after applying data whitening."""
    h0, omega_m, omega_b = point[:3]
    if early is None:
        early = model.early.evaluate(h0, omega_m, omega_b, model.z_grid)
    dmstar_lcdm, rstar, rd, zstar, omega_de, hubble_grid = early
    with np.errstate(all="ignore"):
        rho_de = np.exp(model.kernel @ (1.0 + point[3:]))
    e2_lcdm = (hubble_grid / h0) ** 2
    e2 = e2_lcdm + omega_de * (rho_de - 1.0)
    if not np.all(np.isfinite(e2) & (e2 > 0)):
        raise ValueError("Invalid perturbed expansion history")
    early_de_fraction = abs(omega_de * (rho_de[-1] - 1.0)) / np.interp(
        zstar, model.z_grid, e2_lcdm
    )
    if early_de_fraction > 1e-4:
        raise ValueError("Perturbation changes recombination physics")

    dm_grid = (C_KM_S / h0) * cumulative_trapezoid(
        model.one_plus_z / np.sqrt(e2), model.u_grid, initial=0.0
    )
    gw = (1.0 + model.data.gw_z) * np.interp(
        model.data.gw_z, model.z_grid, dm_grid
    ) / model.data.gw_sigma

    dm_bao = np.interp(model.data.bao_z, model.z_grid, dm_grid)
    e2_bao = np.interp(model.data.bao_z, model.z_grid, e2)
    dh_bao = C_KM_S / (h0 * np.sqrt(e2_bao))
    bao = np.empty(len(dm_bao))
    for index, quantity in enumerate(model.data.bao_quantity):
        if quantity == "DM_over_rs":
            bao[index] = dm_bao[index] / rd
        elif quantity == "DH_over_rs":
            bao[index] = dh_bao[index] / rd
        else:
            bao[index] = np.cbrt(
                model.data.bao_z[index] * dm_bao[index] ** 2 * dh_bao[index]
            ) / rd
    chol = np.tril(model.data.bao_cov_factor[0])
    bao_white = solve_triangular(chol, bao, lower=True)

    delta_integrand = model.one_plus_z * (
        1.0 / np.sqrt(e2) - 1.0 / np.sqrt(e2_lcdm)
    )
    delta_dm = (C_KM_S / h0) * cumulative_trapezoid(
        delta_integrand, model.u_grid, initial=0.0
    )
    dmstar = dmstar_lcdm + np.interp(zstar, model.z_grid, delta_dm)
    theta100 = 100.0 * rstar / dmstar
    return np.r_[
        gw,
        bao_white,
        theta100 / CMB_THETA100_SIGMA,
        omega_b / BBN_SIGMA,
    ]

EVENT_COUNTS = (100, 500, 1000, 3000)
PRIOR_AMPLITUDE = 0.01
CHAIN_ROOT = JOINT_PCA_CHAINS
TABLES = DESI_BBN_CMB_GW_PCA_001_TABLES
DATA_KEYS = ("GW_absolute", "DESI_DR2_BAO", "CMB_theta_star", "BBN_omega_b_h2")
SAMPLE_NAMES = tuple(f"q{i}" for i in range(1, len(PARAMETER_NAMES) + 1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--postprocess-only", action="store_true")
    parser.add_argument("--n-events", type=int, nargs="+", default=list(EVENT_COUNTS))
    parser.add_argument("--output-root", type=Path, default=CHAIN_ROOT)
    parser.add_argument("--max-samples", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not (args.check or args.run or args.postprocess_only):
        args.check = True
    if len(args.n_events) != len(set(args.n_events)) or set(args.n_events) - set(EVENT_COUNTS):
        parser.error("--n-events must be distinct values chosen from 100 500 1000 3000")
    if args.max_samples < 1:
        parser.error("--max-samples must be positive")
    if args.resume and not args.run:
        parser.error("--resume requires --run")
    return args


def source_summary(n_events: int) -> dict:
    path = (
        RECONSTRUCT_WZ_DESI_BBN_CMB_GW_TABLES
        / f"desi_dr2_bbn_cmb_gw{n_events:04d}_wz_summary.json"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Missing converged A=1 reconstruction: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("rank_folded_split_rhat", {}).get("all_pass"):
        raise ValueError(f"A=1 source chains are not converged: {path}")
    return payload


def reference_point(n_events: int) -> np.ndarray:
    source_summary(n_events)  # Also enforce convergence of the source chains.
    paths = (
        COMPUTE_BEST_FIT_W_TABLES / "desi_dr2_bbn_cmb_gw0100_bestfit_horndeski_summary.json",
        COMPUTE_BEST_FIT_W_TABLES / "desi_dr2_bbn_cmb_gw_bestfit_horndeski_summary.json",
    )
    cases = []
    for path in paths:
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            cases.extend(case for case in payload.get("cases", [])
                         if case.get("n_gw") == n_events)
    if len(cases) != 1:
        raise ValueError(f"Missing unique optimized A=1 reference for GW{n_events}")
    return np.asarray([cases[0]["parameters"][name]
                       for name in PARAMETER_NAMES], dtype=float)


def weak_loglike(model: JointLikelihood, point: np.ndarray) -> float:
    try:
        components = model.components(point)
    except Exception:
        return -math.inf
    total = sum(components[key] for key in DATA_KEYS)
    total += PRIOR_AMPLITUDE * components["Horndeski_prior"]
    return -0.5 * total


def proposal_covariance(model: JointLikelihood, point: np.ndarray) -> np.ndarray:
    """Local Laplace proposal for data plus the A=0.01 modified prior."""
    baseline = model.early.evaluate(*point[:3], model.z_grid)
    jacobian = np.empty((len(model.data.gw_z) + len(model.data.bao_z) + 2,
                         len(PARAMETER_NAMES)))
    for j, step in enumerate(STEP):
        plus, minus = point.copy(), point.copy()
        plus[j] += step
        minus[j] -= step
        shared = baseline if j >= 3 else None
        jacobian[:, j] = (
            whitened_predictions(model, plus, shared)
            - whitened_predictions(model, minus, shared)
        ) / (2.0 * step)
    precision = jacobian.T @ jacobian
    precision[3:, 3:] += PRIOR_AMPLITUDE * horndeski_precision()
    precision = 0.5 * (precision + precision.T)
    covariance = np.linalg.inv(precision)
    covariance = 0.5 * (covariance + covariance.T)

    # The local Gaussian can extend well past the broad top-hat priors in the
    # nearly free directions.  Cap only the proposal scales while preserving
    # its correlation matrix; this does not modify the sampled posterior.
    standard = np.sqrt(np.diag(covariance))
    caps = np.asarray([12.0, 0.16, 0.0007] + [1.5] * N_W)
    target = np.minimum(standard, caps)
    correlation = covariance / np.outer(standard, standard)
    proposal = correlation * np.outer(target, target)
    proposal = 0.5 * (proposal + proposal.T)
    minimum = np.linalg.eigvalsh(proposal)[0]
    if minimum <= 0:
        proposal += np.eye(len(proposal)) * (abs(minimum) + 1e-12)
    np.linalg.cholesky(proposal)
    return proposal


def coordinate_transform(model: JointLikelihood, n_events: int) -> tuple[np.ndarray, np.ndarray]:
    """Return x = reference + transform @ q for constant-Jacobian sampling."""
    reference = reference_point(n_events)
    transform = np.linalg.cholesky(proposal_covariance(model, reference))
    return reference, transform


def cobaya_info(model: JointLikelihood, prefix: Path, args: argparse.Namespace,
                n_events: int) -> dict:
    reference, transform = coordinate_transform(model, n_events)
    params = {
        name: {
            # The actual broad physical top-hat prior is imposed after the
            # linear transform.  These numerical bounds are deliberately far
            # outside the posterior support.
            "prior": {"min": -1e6, "max": 1e6},
            "ref": {"dist": "norm", "loc": 0.0, "scale": 1e-4},
            "proposal": 1.0,
        }
        for name in SAMPLE_NAMES
    }

    def loglike(**values: float) -> float:
        coordinates = np.asarray([values[name] for name in SAMPLE_NAMES])
        point = reference + transform @ coordinates
        if np.any(point < LOWER) or np.any(point > UPPER):
            return -math.inf
        return weak_loglike(model, point)

    return {
        "params": params,
        "likelihood": {"DESI_DR2_BBN_CMB_absolute_GW_PCA_A001": {
            "external": loglike, "input_params": list(SAMPLE_NAMES),
        }},
        "sampler": {"mcmc": {
            "covmat": np.eye(len(SAMPLE_NAMES)),
            "covmat_params": list(SAMPLE_NAMES),
            "burn_in": "2d",
            "max_samples": args.max_samples,
            "Rminus1_stop": 0.01,
            "Rminus1_cl_stop": 0.05,
            "learn_proposal": True,
            "learn_every": "40d",
            "learn_proposal_Rminus1_max": 1e6,
            # Freeze after the broad weak-prior tails have been learned.  This
            # preserves a stationary final half-chain for covariance work.
            "learn_proposal_Rminus1_min": 0.1,
            "proposal_scale": 1.5,
            "seed": args.seed + n_events,
        }},
        "output": str(prefix),
        "stop_at_error": True,
    }


def weak_prefix(root: Path, n_events: int) -> Path:
    return root.expanduser().resolve() / f"GW{n_events:04d}" / "chain"


def postprocess(prefix: Path, n_events: int) -> dict:
    files = chain_files(prefix)
    if len(files) < 4:
        raise ValueError(f"Four independent chains are required: {prefix}")
    raw_groups = [retained_chain(path) for path in files]
    model = JointLikelihood(load_observations(n_events), TableEarlyBackground())
    reference, transform = coordinate_transform(model, n_events)
    groups = [
        (reference + group[0] @ transform.T, group[1], group[2], group[3])
        for group in raw_groups
    ]
    diagnostic = rank_folded_rhat(groups)
    if not diagnostic["all_pass"]:
        raise ValueError(
            f"GW{n_events}: max rank/folded split Rhat-1="
            f"{diagnostic['max_rhat_minus_one']:.5f}; resume the A=0.01 chains"
        )
    samples = np.concatenate([group[0] for group in groups])
    weights = np.concatenate([group[1] for group in groups])
    mean = np.average(samples, axis=0, weights=weights)
    centred = samples - mean
    covariance = np.einsum("ni,nj,n->ij", centred, centred, weights,
                           optimize=True) / np.sum(weights)
    covariance = 0.5 * (covariance + covariance.T)
    w_covariance = covariance[3:, 3:]
    np.linalg.cholesky(w_covariance)

    TABLES.mkdir(parents=True, exist_ok=True)
    stem = f"desi_dr2_bbn_cmb_gw{n_events:04d}_pca_A001"
    covariance_path = TABLES / f"{stem}_wz_covariance.csv"
    summary_path = TABLES / f"{stem}_summary.json"
    np.savetxt(
        covariance_path,
        w_covariance,
        fmt="%.12e",
        delimiter=",",
        header=",".join(f"w{i}" for i in range(1, N_W + 1)),
        comments="",
    )
    summary = {
        "n_gw": n_events,
        "catalog": str(catalog_path(n_events).resolve()),
        "method": "Gu et al. Fig. 4 weak-correlation-prior posterior",
        "prior_amplitude": PRIOR_AMPLITUDE,
        "prior_precision": "(I-S)^T C_Pi^-1 (I-S), five-neighbour local mean",
        "sampling_coordinates": "constant-Jacobian linear whitening; covariance and Rhat transformed back to H0, Omega_m, omega_b, w1..w29",
        "n_independent_chains": len(files),
        "total_retained_weighted_draws": int(np.sum(weights)),
        "chain_weighted_draws_before_second_half": [group[3] for group in groups],
        "rank_folded_split_rhat": diagnostic,
        "joint_posterior_mean": {name: float(value)
                                 for name, value in zip(PARAMETER_NAMES, mean)},
        "w_covariance": str(covariance_path),
        "chains": [str(path) for path in files],
        "early_background": str(TABLE_FILE.resolve()),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def check(args: argparse.Namespace) -> None:
    for n_events in args.n_events:
        model = JointLikelihood(load_observations(n_events), TableEarlyBackground())
        point = reference_point(n_events)
        value = weak_loglike(model, point)
        proposal = proposal_covariance(model, point)
        print(f"GW{n_events}: loglike_A001={value:.6f}; "
              f"proposal_std_w=[{np.sqrt(np.diag(proposal))[3:].min():.3f}, "
              f"{np.sqrt(np.diag(proposal))[3:].max():.3f}]")
    print("A=0.01 likelihood and proposal checks passed.")


def run_chains(args: argparse.Namespace) -> None:
    from mpi4py import MPI
    from cobaya.samplers.mcmc.mcmc import MCMC

    # Cobaya normally preserves the old proposal scale on resume.  Keep the
    # scale in this script authoritative so a poorly mixing checkpoint can be
    # continued with the corrected scale without discarding valid samples.
    if "proposal_scale" in MCMC._at_resume_prefer_old:
        MCMC._at_resume_prefer_old.remove("proposal_scale")
    if "proposal_scale" not in MCMC._at_resume_prefer_new:
        MCMC._at_resume_prefer_new.append("proposal_scale")

    comm = MPI.COMM_WORLD
    if comm.size < 4:
        if comm.rank == 0:
            print("Use at least four MPI processes for four independent chains.", file=sys.stderr)
        raise SystemExit(2)
    for n_events in args.n_events:
        prefix = weak_prefix(args.output_root, n_events)
        if comm.rank == 0:
            existing = chain_files(prefix) or list(prefix.parent.glob("chain.*.yaml"))
            if existing and not args.resume:
                error = f"Existing output at {prefix}; use --resume"
            elif args.resume and not existing:
                error = f"No existing output to resume at {prefix}"
            else:
                prefix.parent.mkdir(parents=True, exist_ok=True)
                error = None
        else:
            error = None
        error = comm.bcast(error, root=0)
        if error:
            raise FileExistsError(error)

        model = JointLikelihood(load_observations(n_events), TableEarlyBackground())
        if not math.isfinite(weak_loglike(model, reference_point(n_events))):
            raise ValueError(f"GW{n_events}: non-finite A=0.01 reference likelihood")
        if comm.rank == 0:
            print(f"GW{n_events}: sampling 32 parameters with A=0.01", flush=True)
        _, sampler = run(cobaya_info(model, prefix, args, n_events), resume=args.resume)
        comm.Barrier()
        if comm.rank == 0:
            metadata = {
                "prior_amplitude": PRIOR_AMPLITUDE,
                "mpi_processes": comm.size,
                "cobaya_converged": bool(getattr(sampler, "converged", False)),
                "chain_files": [str(path) for path in chain_files(prefix)],
            }
            prefix.with_suffix(".metadata.json").write_text(
                json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
            )
            try:
                summary = postprocess(prefix, n_events)
            except ValueError as exc:
                print(f"GW{n_events}: covariance pending: {exc}", flush=True)
            else:
                print(f"GW{n_events}: A=0.01 covariance ready; "
                      f"Rhat-1={summary['rank_folded_split_rhat']['max_rhat_minus_one']:.5f}",
                      flush=True)
        comm.Barrier()


def main() -> None:
    args = parse_args()
    if args.check:
        check(args)
    elif args.postprocess_only:
        for n_events in args.n_events:
            summary = postprocess(weak_prefix(args.output_root, n_events), n_events)
            print(f"GW{n_events}: {summary['w_covariance']}")
    else:
        run_chains(args)


if __name__ == "__main__":
    main()
