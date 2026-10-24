"""Compute GW-only CPL MAP fits and Wilks significances by catalogue size.

Each selected GW catalogue is fitted independently with flat CPL and flat
LambdaCDM. The nested-model statistic is

    Delta chi2_MAP = chi2_MAP(LambdaCDM) - chi2_MAP(CPL),

with a two-degree-of-freedom Wilks conversion. DIC is intentionally handled by
``compute_gw_dic_by_n.py`` because it requires retained posterior chains;
MAP/Wilks needs only the GW catalogues.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2, norm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cosmology.gw_cosmology import CPLCosmology
from cosmology.gw_lcdm_cosmology import LCDMCosmology
from mcmc.gw_evidence import (
    DEFAULT_CATALOG_SIZES,
    DEFAULT_REALIZATION,
    H0_BOUNDS,
    OMEGAM_BOUNDS,
    W0_BOUNDS,
    WA_BOUNDS,
    catalog_path_for,
    load_gw_catalog,
)
from paths import TABLES

CPL_LOWER = np.array(
    [H0_BOUNDS[0], OMEGAM_BOUNDS[0], W0_BOUNDS[0], WA_BOUNDS[0]],
    dtype=float,
)
CPL_UPPER = np.array(
    [H0_BOUNDS[1], OMEGAM_BOUNDS[1], W0_BOUNDS[1], WA_BOUNDS[1]],
    dtype=float,
)
LCDM_LOWER = np.array([H0_BOUNDS[0], OMEGAM_BOUNDS[0]], dtype=float)
LCDM_UPPER = np.array([H0_BOUNDS[1], OMEGAM_BOUNDS[1]], dtype=float)

# Multiple deterministic starts make the non-linear CPL fit independent of an
# MCMC chain while covering the broad prior volume used by the GW analysis.
CPL_STARTS = np.array(
    [
        [57.1577, 0.42279, -1.0, 0.0],
        [67.4, 0.315, -1.0, 0.0],
        [70.0, 0.30, 0.0, -4.0],
        [70.0, 0.40, 0.5, -6.0],
        [60.0, 0.50, 2.0, -15.0],
        [80.0, 0.20, -1.5, 1.0],
    ],
    dtype=float,
)
LCDM_STARTS = np.array(
    [[57.1577, 0.42279], [67.4, 0.315], [60.0, 0.50], [80.0, 0.20]],
    dtype=float,
)


def _best_least_squares(
    residuals: Callable[[np.ndarray], np.ndarray],
    starts: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
):
    """Return the lowest-chi-square converged fit across deterministic starts."""
    fits = []
    for start in np.asarray(starts, dtype=float):
        fit = least_squares(
            residuals,
            np.clip(start, lower, upper),
            bounds=(lower, upper),
            xtol=1e-13,
            ftol=1e-13,
            gtol=1e-13,
            max_nfev=4000,
        )
        if fit.success and np.all(np.isfinite(fit.x)) and np.all(np.isfinite(fit.fun)):
            fits.append(fit)
    if not fits:
        raise RuntimeError("all least-squares starts failed")
    return min(fits, key=lambda fit: float(np.dot(fit.fun, fit.fun)))


def fit_one(
    n_events: int,
    realization: int = DEFAULT_REALIZATION,
) -> dict[str, float | int | bool | str]:
    """Fit CPL and LambdaCDM to one validated GW catalogue."""
    catalog = catalog_path_for(n_events, realization).resolve()
    z, d_obs, sigma = load_gw_catalog(
        catalog,
        expected_n_events=n_events,
    )
    z_max = max(float(z.max()) * 1.05, 2.5)
    cpl = CPLCosmology(z_max=z_max)
    lcdm = LCDMCosmology(z_max=z_max)
    inv_sigma = 1.0 / sigma

    def cpl_residuals(theta: np.ndarray) -> np.ndarray:
        h0, omega_m, w0, wa = theta
        model = cpl.luminosity_distance(
            z,
            h0,
            w0,
            wa,
            omega_m=omega_m,
        )
        return (d_obs - model) * inv_sigma

    def lcdm_residuals(theta: np.ndarray) -> np.ndarray:
        h0, omega_m = theta
        model = lcdm.luminosity_distance(z, h0, omega_m=omega_m)
        return (d_obs - model) * inv_sigma

    cpl_fit = _best_least_squares(
        cpl_residuals,
        CPL_STARTS,
        CPL_LOWER,
        CPL_UPPER,
    )
    lcdm_fit = _best_least_squares(
        lcdm_residuals,
        LCDM_STARTS,
        LCDM_LOWER,
        LCDM_UPPER,
    )
    chi2_cpl = float(np.dot(cpl_fit.fun, cpl_fit.fun))
    chi2_lcdm = float(np.dot(lcdm_fit.fun, lcdm_fit.fun))
    delta_chi2 = max(0.0, chi2_lcdm - chi2_cpl)
    at_boundary = bool(
        np.any(np.isclose(cpl_fit.x, CPL_LOWER, atol=1e-7, rtol=0.0))
        or np.any(np.isclose(cpl_fit.x, CPL_UPPER, atol=1e-7, rtol=0.0))
        or np.any(np.isclose(lcdm_fit.x, LCDM_LOWER, atol=1e-7, rtol=0.0))
        or np.any(np.isclose(lcdm_fit.x, LCDM_UPPER, atol=1e-7, rtol=0.0))
    )
    # Report the numerical chi-square conversion for every catalogue. A fit
    # at a prior boundary remains flagged because the usual Wilks calibration
    # does not apply there; its displayed sigma is only a nominal value.
    p_value = float(chi2.sf(delta_chi2, df=2))
    n_sigma = float(norm.isf(p_value / 2.0))

    return {
        "n_gw": int(n_events),
        "realization": int(realization),
        "catalog": str(catalog),
        "catalog_sha256": hashlib.sha256(catalog.read_bytes()).hexdigest(),
        "H0_map": float(cpl_fit.x[0]),
        "omegam_map": float(cpl_fit.x[1]),
        "w0_map": float(cpl_fit.x[2]),
        "wa_map": float(cpl_fit.x[3]),
        "chi2_map_w0waCDM": chi2_cpl,
        "chi2_map_lcdm": chi2_lcdm,
        "delta_chi2_map": delta_chi2,
        "p_value_chi2_2dof": p_value,
        "n_sigma_1d_two_sided": n_sigma,
        "wilks_chi2_2dof_valid": not at_boundary,
        "cpl_map_at_prior_boundary": at_boundary,
        "n_cpl_starts": int(CPL_STARTS.shape[0]),
        "n_lcdm_starts": int(LCDM_STARTS.shape[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute GW-only CPL/LambdaCDM MAP and Wilks diagnostics."
    )
    parser.add_argument(
        "--n-events",
        type=int,
        nargs="+",
        default=list(DEFAULT_CATALOG_SIZES),
        help="GW catalogue sizes (default: 100 500 1000 3000).",
    )
    parser.add_argument(
        "--realization",
        type=int,
        default=DEFAULT_REALIZATION,
        help="Mock realization index (default: %(default)s).",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=TABLES / "gw_map_wilks_by_n.json",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=TABLES / "gw_map_wilks_by_n.csv",
    )
    args = parser.parse_args()

    n_values = list(dict.fromkeys(int(value) for value in args.n_events))
    if not n_values or any(value <= 0 for value in n_values):
        parser.error("--n-events values must be positive")
    if args.realization < 0:
        parser.error("--realization must be non-negative")

    rows = [fit_one(n, int(args.realization)) for n in n_values]
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    with args.out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 76)
    print(" N_GW       w0_MAP       wa_MAP    Delta_chi2    N_sigma")
    print("=" * 76)
    for row in rows:
        sigma_text = f"{row['n_sigma_1d_two_sided']:8.3f}"
        print(
            f"{row['n_gw']:5d}  "
            f"{row['w0_map']:11.4f}  {row['wa_map']:11.4f}  "
            f"{row['delta_chi2_map']:12.4f}  "
            f"{sigma_text}"
        )
    print("=" * 76)
    if any(not row["wilks_chi2_2dof_valid"] for row in rows):
        print("Boundary fits have nominal chi-square sigma values; see validity flags in JSON/CSV.")
    print(f"Wrote {args.out_json.resolve()}")
    print(f"Wrote {args.out_csv.resolve()}")


if __name__ == "__main__":
    main()
