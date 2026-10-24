#!/usr/bin/env python3
"""Compute numerically stable local-Fisher PCA matrices for the ET catalogs.

This is a deterministic alternative to inverting the marginalized covariance
of a weak-prior MCMC chain.  At the common injected cosmology, it computes the
Jacobian of the whitened data predictions, forms the Gauss--Newton data Fisher
matrix J.T @ J, and marginalizes the nuisance cosmological parameters with a
Schur complement.  A second calculation with half-sized finite-difference
steps quantifies the numerical stability of the first four eigenmodes.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcmc.analyze_pca_bbn_cmb_desi_gw import (  # noqa: E402
    STEP as JOINT_STEP,
    whitened_predictions,
)
from mcmc.bbn_cmb_desi_absolute_gw_wz_model import (  # noqa: E402
    BBN_MEAN,
    N_W,
    JointLikelihood,
    load_observations,
    w_geometry,
)
from mcmc.early_background_table import (  # noqa: E402
    TABLE_FILE,
    TableEarlyBackground,
)
from mcmc.reconstruct_wz_gw_prior import (  # noqa: E402
    FIDUCIAL_H0,
    FIDUCIAL_OMEGA_M,
    GWOnlyLikelihood,
    load_catalog,
    load_injection,
)


EVENT_COUNTS = (100, 500, 1000, 3000)
TABLES = ROOT / "results" / "tables"
GW_ONLY_STEM = "gw_only_stable_local_pca_data_fisher"
JOINT_STEM = "desi_dr2_bbn_cmb_absolute_gw_stable_local_pca_data_fisher"
SUMMARY_PATH = TABLES / "stable_local_pca_summary.json"
MODES_PATH = TABLES / "stable_local_pca_modes_pc1-4.csv"
GW_ONLY_STEP = np.asarray([0.005, 0.0002] + [0.002] * N_W)
N_STABLE_MODES = 4
MIN_RELATIVE_EIGENGAP = 0.10
MIN_STEP_OVERLAP = 0.995
MAX_RELATIVE_EIGENVALUE_CHANGE = 0.005


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-events",
        type=int,
        nargs="+",
        default=list(EVENT_COUNTS),
        help="Catalog sizes chosen from 100, 500, 1000, and 3000",
    )
    args = parser.parse_args()
    args.n_events = list(dict.fromkeys(args.n_events))
    if not args.n_events or set(args.n_events) - set(EVENT_COUNTS):
        parser.error("--n-events must be chosen from 100 500 1000 3000")
    return args


def symmetric(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    return 0.5 * (matrix + matrix.T)


def marginalized_fisher(jacobian: np.ndarray, n_nuisance: int) -> np.ndarray:
    """Return the w-bin Fisher after Gaussian nuisance marginalization."""
    full = symmetric(np.asarray(jacobian, dtype=float).T @ jacobian)
    nuisance = full[:n_nuisance, :n_nuisance]
    cross = full[:n_nuisance, n_nuisance:]
    w_block = full[n_nuisance:, n_nuisance:]
    np.linalg.cholesky(nuisance)
    answer = symmetric(w_block - cross.T @ np.linalg.solve(nuisance, cross))
    if answer.shape != (N_W, N_W) or not np.all(np.isfinite(answer)):
        raise ValueError("Invalid marginalized 29x29 data Fisher matrix")
    return answer


def ordered_eigensystem(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values, vectors = np.linalg.eigh(symmetric(matrix))
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    for column in range(vectors.shape[1]):
        vector = vectors[:, column]
        if vector[np.argmax(np.abs(vector))] < 0.0:
            vectors[:, column] *= -1.0
    return values, vectors


def gw_only_fisher(
    model: GWOnlyLikelihood,
    injected_w: np.ndarray,
    *,
    step_scale: float,
) -> np.ndarray:
    point = np.r_[FIDUCIAL_H0, FIDUCIAL_OMEGA_M, injected_w]
    steps = step_scale * GW_ONLY_STEP
    jacobian = np.empty((model.data.z.size, point.size))
    for column, step in enumerate(steps):
        plus, minus = point.copy(), point.copy()
        plus[column] += step
        minus[column] -= step
        jacobian[:, column] = (
            model.distance_model(plus) / model.data.sigma
            - model.distance_model(minus) / model.data.sigma
        ) / (2.0 * step)
    return marginalized_fisher(jacobian, 2)


def joint_fisher(
    model: JointLikelihood,
    injected_w: np.ndarray,
    *,
    step_scale: float,
) -> np.ndarray:
    point = np.r_[FIDUCIAL_H0, FIDUCIAL_OMEGA_M, BBN_MEAN, injected_w]
    steps = step_scale * JOINT_STEP
    baseline = model.early.evaluate(*point[:3], model.z_grid)
    n_observables = model.data.gw_z.size + model.data.bao_z.size + 2
    jacobian = np.empty((n_observables, point.size))
    for column, step in enumerate(steps):
        plus, minus = point.copy(), point.copy()
        plus[column] += step
        minus[column] -= step
        shared_early = baseline if column >= 3 else None
        jacobian[:, column] = (
            whitened_predictions(model, plus, shared_early)
            - whitened_predictions(model, minus, shared_early)
        ) / (2.0 * step)
    return marginalized_fisher(jacobian, 3)


def stability_record(
    probe: str,
    n_events: int,
    production: np.ndarray,
    half_step: np.ndarray,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    values, vectors = ordered_eigensystem(production)
    check_values, check_vectors = ordered_eigensystem(half_step)
    gaps = (values[:N_STABLE_MODES] - values[1 : N_STABLE_MODES + 1]) / values[
        :N_STABLE_MODES
    ]
    overlaps = np.abs(
        np.sum(
            vectors[:, :N_STABLE_MODES] * check_vectors[:, :N_STABLE_MODES],
            axis=0,
        )
    )
    relative_changes = np.abs(
        check_values[:N_STABLE_MODES] - values[:N_STABLE_MODES]
    ) / values[:N_STABLE_MODES]
    stable = bool(
        np.min(gaps) >= MIN_RELATIVE_EIGENGAP
        and np.min(overlaps) >= MIN_STEP_OVERLAP
        and np.max(relative_changes) <= MAX_RELATIVE_EIGENVALUE_CHANGE
    )
    record: dict[str, object] = {
        "probe": probe,
        "n_gw": n_events,
        "lambda_pc1_pc5": values[:5].tolist(),
        "relative_eigengap_pc1_pc4": gaps.tolist(),
        "half_step_abs_mode_overlap_pc1_pc4": overlaps.tolist(),
        "half_step_relative_eigenvalue_change_pc1_pc4": relative_changes.tolist(),
        "minimum_relative_eigengap_pc1_pc4": float(np.min(gaps)),
        "minimum_half_step_abs_mode_overlap_pc1_pc4": float(np.min(overlaps)),
        "maximum_half_step_relative_eigenvalue_change_pc1_pc4": float(
            np.max(relative_changes)
        ),
        "passes_stability_thresholds": stable,
    }
    return record, values, vectors


def write_modes(rows: list[dict[str, object]]) -> None:
    with MODES_PATH.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    TABLES.mkdir(parents=True, exist_ok=True)
    injected_w = load_injection()
    _, z_min, z_max, z_center = w_geometry()
    diagnostics: list[dict[str, object]] = []
    mode_rows: list[dict[str, object]] = []

    for n_events in args.n_events:
        gw_model = GWOnlyLikelihood(load_catalog(n_events), prior_amplitude=1.0)
        joint_model = JointLikelihood(
            load_observations(n_events),
            TableEarlyBackground(),
        )
        cases = (
            (
                "GW-only",
                GW_ONLY_STEM,
                gw_only_fisher(gw_model, injected_w, step_scale=1.0),
                gw_only_fisher(gw_model, injected_w, step_scale=0.5),
            ),
            (
                "DESI+BBN+CMB+GW",
                JOINT_STEM,
                joint_fisher(joint_model, injected_w, step_scale=1.0),
                joint_fisher(joint_model, injected_w, step_scale=0.5),
            ),
        )
        for probe, stem, fisher, half_step_fisher in cases:
            record, values, vectors = stability_record(
                probe,
                n_events,
                fisher,
                half_step_fisher,
            )
            diagnostics.append(record)
            fisher_path = TABLES / f"{stem}_gw{n_events:04d}.txt"
            eigenvalue_path = TABLES / f"{stem}_eigenvalues_gw{n_events:04d}.txt"
            np.savetxt(
                fisher_path,
                fisher,
                fmt="%.12e",
                header=(
                    "Stable local Gauss-Newton data Fisher, nuisance-marginalized "
                    "by Schur complement at the injected cosmology"
                ),
            )
            np.savetxt(
                eigenvalue_path,
                values,
                fmt="%.12e",
                header="Descending eigenvalues of the stable local data Fisher",
            )
            for pc in range(N_STABLE_MODES):
                for bin_index in range(N_W):
                    mode_rows.append(
                        {
                            "probe": probe,
                            "n_gw": n_events,
                            "pc": pc + 1,
                            "bin": bin_index + 1,
                            "z_min": z_min[bin_index],
                            "z_max": z_max[bin_index],
                            "z_center": z_center[bin_index],
                            "e_i": vectors[bin_index, pc],
                        }
                    )
            status = "PASS" if record["passes_stability_thresholds"] else "FAIL"
            print(
                f"{probe:20s} GW{n_events:04d}: {status}; "
                f"min gap={record['minimum_relative_eigengap_pc1_pc4']:.3f}; "
                f"min half-step overlap="
                f"{record['minimum_half_step_abs_mode_overlap_pc1_pc4']:.8f}"
            )

    write_modes(mode_rows)
    all_pass = all(bool(case["passes_stability_thresholds"]) for case in diagnostics)
    summary = {
        "method": (
            "Local Gauss-Newton data Fisher J^T J at one common injected "
            "cosmology; nuisance parameters marginalized by Schur complement"
        ),
        "reason": (
            "Avoid inversion noise from the weak-prior MCMC covariance and "
            "require resolved eigengaps for individually interpretable PCs"
        ),
        "reference_cosmology": {
            "H0": FIDUCIAL_H0,
            "Omega_m": FIDUCIAL_OMEGA_M,
            "omega_b_for_joint": BBN_MEAN,
            "w_bins": str((ROOT / "data" / "gw_fiducial_w_bins.txt").resolve()),
        },
        "nuisance_marginalization": {
            "GW-only": ["H0", "Omega_m"],
            "DESI+BBN+CMB+GW": ["H0", "Omega_m", "omega_b"],
        },
        "joint_early_background_table": str(TABLE_FILE.resolve()),
        "stability_test": {
            "repeat_with_finite_difference_step_scale": 0.5,
            "minimum_relative_eigengap": MIN_RELATIVE_EIGENGAP,
            "minimum_absolute_mode_overlap": MIN_STEP_OVERLAP,
            "maximum_relative_eigenvalue_change": MAX_RELATIVE_EIGENVALUE_CHANGE,
            "all_cases_pass": all_pass,
        },
        "cases": diagnostics,
        "outputs": {
            "modes_csv": str(MODES_PATH.resolve()),
            "gw_only_fisher_stem": GW_ONLY_STEM,
            "joint_fisher_stem": JOINT_STEM,
        },
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved: {MODES_PATH.resolve()}")
    print(f"Saved: {SUMMARY_PATH.resolve()}")
    if not all_pass:
        raise RuntimeError("At least one PC1--PC4 stability check failed")


if __name__ == "__main__":
    main()
