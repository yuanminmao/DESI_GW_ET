#!/usr/bin/env python3
"""Compute joint-data CPL versus LambdaCDM Wilks statistics by N_GW.

Both nested models use the same DESI DR2 + BBN + compressed CMB + absolute
ET-GW likelihood.  LambdaCDM varies ``(H0, omegam, omega_b)`` with
``(w0, wa)=(-1, 0)`` fixed; CPL additionally varies ``(w0, wa)``.  For each
catalogue this script reports

    Delta chi2 = chi2_min(LambdaCDM) - chi2_min(CPL)

and converts it with a chi-square distribution with two degrees of freedom.
The quoted Gaussian significance is two-sided, matching
``gw_cpl_wilks.py``.

Run all four catalogues::

    python3 analysis/cpl_lcdm_comparison/desi_bbn_cmb_gw_cpl_wilks.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2, norm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paths import DESI_BBN_CMB_GW_CPL_WILKS_TABLES, project_path, setup_sys_path

setup_sys_path()

from mcmc.reconstruct_wz_desi_bbn_cmb_gw import BBN_MEAN, catalog_path
from mcmc.desi_bbn_cmb_gw_cpl_mcmc import (
    LOWER,
    UPPER,
    CPLJointLikelihood,
    TableEarlyBackground,
    load_observations,
)

DEFAULT_N_EVENTS = (100, 500, 1000, 3000)
CHI2_KEYS = ("GW_absolute", "DESI_DR2_BAO", "BBN_omega_b_h2", "CMB_theta_star")

CPL_REFERENCE = np.asarray([67.4, 0.315, BBN_MEAN, -1.0, 0.0])
CPL_SCALE = np.asarray([10.0, 0.10, 0.001, 0.5, 2.0])
CPL_STARTS = np.asarray([
    CPL_REFERENCE,
    [57.4, 0.42, BBN_MEAN, -0.7, -1.0],
    [60.0, 0.40, BBN_MEAN, 0.0, -3.0],
    [64.0, 0.35, 0.0220, -0.5, -1.5],
])

LCDM_REFERENCE = CPL_REFERENCE[:3]
LCDM_SCALE = CPL_SCALE[:3]
LCDM_STARTS = np.asarray([
    LCDM_REFERENCE,
    [60.0, 0.40, BBN_MEAN],
    [70.0, 0.30, 0.0225],
    [64.0, 0.35, 0.0215],
])


@dataclass(frozen=True)
class FitResult:
    point: np.ndarray
    chi2: float
    n_successful_starts: int
    max_near_best_delta_chi2: float


def total_chi2(model: CPLJointLikelihood, point: np.ndarray) -> float:
    value = model.loglike(point)
    return -2.0 * value if np.isfinite(value) else 1.0e100


def _best_fit(
    model: CPLJointLikelihood,
    *,
    reference: np.ndarray,
    scale: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    starts: np.ndarray,
    expand: Callable[[np.ndarray], np.ndarray],
) -> FitResult:
    """Run bounded Powell fits from several starts, then L-BFGS-B polish."""
    bounds = list(zip((lower - reference) / scale, (upper - reference) / scale))

    def objective(offset: np.ndarray) -> float:
        return total_chi2(model, expand(reference + scale * offset))

    fits: list[tuple[float, np.ndarray]] = []
    for start in starts:
        offset = (np.asarray(start, dtype=float) - reference) / scale
        powell = minimize(
            objective,
            offset,
            method="Powell",
            bounds=bounds,
            options={"maxiter": 4000, "xtol": 1.0e-9, "ftol": 1.0e-11},
        )
        polished = minimize(
            objective,
            powell.x,
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": 4000,
                "ftol": 1.0e-15,
                "gtol": 1.0e-8,
                "maxls": 50,
                "finite_diff_rel_step": 1.0e-7,
            },
        )
        candidate = polished if polished.fun <= powell.fun else powell
        point = expand(reference + scale * candidate.x)
        value = float(candidate.fun)
        if value < 1.0e99 and np.isfinite(value) and np.all(np.isfinite(point)):
            fits.append((value, point))
    if not fits:
        raise RuntimeError("all deterministic optimization starts failed")

    fits.sort(key=lambda item: item[0])
    best_chi2, best_point = fits[0]
    # Starts that find the same basin should agree much more closely than the
    # displayed precision. Ignore clearly inferior local basins in this check.
    near_best = [value for value, _ in fits if value - best_chi2 < 1.0]
    spread = max(near_best) - best_chi2
    return FitResult(best_point, best_chi2, len(fits), float(spread))


def fit_cpl(model: CPLJointLikelihood) -> FitResult:
    return _best_fit(
        model,
        reference=CPL_REFERENCE,
        scale=CPL_SCALE,
        lower=LOWER,
        upper=UPPER,
        starts=CPL_STARTS,
        expand=lambda point: point,
    )


def fit_lcdm(model: CPLJointLikelihood) -> FitResult:
    return _best_fit(
        model,
        reference=LCDM_REFERENCE,
        scale=LCDM_SCALE,
        lower=LOWER[:3],
        upper=UPPER[:3],
        starts=LCDM_STARTS,
        expand=lambda point: np.r_[point, -1.0, 0.0],
    )


def at_boundary(point: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> bool:
    return bool(
        np.any(np.isclose(point, lower, atol=1.0e-6, rtol=0.0))
        or np.any(np.isclose(point, upper, atol=1.0e-6, rtol=0.0))
    )


def fit_one(n_events: int) -> dict[str, float | int | bool | str]:
    model = CPLJointLikelihood(load_observations(n_events), TableEarlyBackground())
    cpl = fit_cpl(model)
    lcdm = fit_lcdm(model)

    # Recompute the totals from named likelihood components as an independent
    # check that the optimizer objective contains all four requested probes.
    cpl_components = model.components(cpl.point)
    lcdm_components = model.components(lcdm.point)
    cpl_total = float(sum(cpl_components[key] for key in CHI2_KEYS))
    lcdm_total = float(sum(lcdm_components[key] for key in CHI2_KEYS))
    if not np.isclose(cpl_total, cpl.chi2, rtol=1.0e-10, atol=1.0e-8):
        raise ValueError("CPL component chi2 does not match the optimized objective")
    if not np.isclose(lcdm_total, lcdm.chi2, rtol=1.0e-10, atol=1.0e-8):
        raise ValueError("LambdaCDM component chi2 does not match the optimized objective")

    delta = max(0.0, lcdm_total - cpl_total)
    p_value = float(chi2.sf(delta, df=2))
    n_sigma = float(norm.isf(p_value / 2.0))
    cpl_boundary = at_boundary(cpl.point, LOWER, UPPER)
    lcdm_boundary = at_boundary(lcdm.point[:3], LOWER[:3], UPPER[:3])
    catalog = catalog_path(n_events).resolve()

    row: dict[str, float | int | bool | str] = {
        "n_gw": int(n_events),
        "catalog": project_path(catalog),
        "catalog_sha256": hashlib.sha256(catalog.read_bytes()).hexdigest(),
        "cpl_H0_map": float(cpl.point[0]),
        "cpl_omegam_map": float(cpl.point[1]),
        "cpl_omega_b_map": float(cpl.point[2]),
        "cpl_w0_map": float(cpl.point[3]),
        "cpl_wa_map": float(cpl.point[4]),
        "lcdm_H0_map": float(lcdm.point[0]),
        "lcdm_omegam_map": float(lcdm.point[1]),
        "lcdm_omega_b_map": float(lcdm.point[2]),
        "chi2_map_cpl": cpl_total,
        "chi2_map_lcdm": lcdm_total,
        "delta_chi2_map": delta,
        "p_value_chi2_2dof": p_value,
        "n_sigma_1d_two_sided": n_sigma,
        "wilks_chi2_2dof_valid": not (cpl_boundary or lcdm_boundary),
        "cpl_map_at_prior_boundary": cpl_boundary,
        "lcdm_map_at_prior_boundary": lcdm_boundary,
        "n_cpl_successful_starts": cpl.n_successful_starts,
        "n_lcdm_successful_starts": lcdm.n_successful_starts,
        "cpl_near_best_delta_chi2_spread": cpl.max_near_best_delta_chi2,
        "lcdm_near_best_delta_chi2_spread": lcdm.max_near_best_delta_chi2,
    }
    for key in CHI2_KEYS:
        row[f"cpl_chi2_{key}"] = float(cpl_components[key])
        row[f"lcdm_chi2_{key}"] = float(lcdm_components[key])
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--n-events", type=int, nargs="+", choices=DEFAULT_N_EVENTS,
        default=list(DEFAULT_N_EVENTS), help="ET catalogue sizes (default: all four)",
    )
    parser.add_argument(
        "--out-json", type=Path,
        default=DESI_BBN_CMB_GW_CPL_WILKS_TABLES / "desi_bbn_cmb_gw_cpl_wilks.json",
    )
    parser.add_argument(
        "--out-csv", type=Path,
        default=DESI_BBN_CMB_GW_CPL_WILKS_TABLES / "desi_bbn_cmb_gw_cpl_wilks.csv",
    )
    args = parser.parse_args()
    counts = list(dict.fromkeys(args.n_events))
    if len(counts) != len(args.n_events):
        parser.error("--n-events values must be distinct")

    rows = []
    for count in counts:
        print(f"Optimizing joint LambdaCDM and CPL fits for N_GW={count} ...", flush=True)
        row = fit_one(count)
        rows.append(row)
        print(
            f"  Delta chi2={row['delta_chi2_map']:.6f}, "
            f"p={row['p_value_chi2_2dof']:.6g}, "
            f"N_sigma={row['n_sigma_1d_two_sided']:.4f}",
            flush=True,
        )

    args.out_json = args.out_json.expanduser().resolve()
    args.out_csv = args.out_csv.expanduser().resolve()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    with args.out_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 77)
    print(" N_GW      chi2_CPL     chi2_LCDM   Delta_chi2       p-value    N_sigma")
    print("=" * 77)
    for row in rows:
        print(
            f"{row['n_gw']:5d}  {row['chi2_map_cpl']:13.4f} "
            f"{row['chi2_map_lcdm']:13.4f} {row['delta_chi2_map']:12.4f} "
            f"{row['p_value_chi2_2dof']:13.5g} {row['n_sigma_1d_two_sided']:10.3f}"
        )
    print("=" * 77)
    if any(not row["wilks_chi2_2dof_valid"] for row in rows):
        print("WARNING: at least one optimum is on a prior boundary; its Wilks conversion is nominal.")
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
