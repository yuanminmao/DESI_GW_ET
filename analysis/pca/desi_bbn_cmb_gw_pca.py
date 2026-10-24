#!/usr/bin/env python3
"""Build DESI Fig. 4 PCA table products for BBN+CMB+DESI DR2+absolute ET GW.

Following Gu et al. (2025), DOI 10.1038/s41550-025-02669-6,
Supplement section 2, the data Fisher matrix is extracted from a converged
weak-correlation-prior chain,

    F_data = C_p(A=0.01)^-1 - 0.01 * Ctilde_Pi^-1,

following the supplied Figure 4 source code.  The modified Horndeski
precision used both in the subtraction and as the prior reference is

    Ctilde_Pi^-1 = (I-S)^T C_Pi^-1 (I-S).

The A=1 chains remain the source of posterior mode amplitudes, Neff, and the
Gaussian distance from w=-1.  Raw, including non-positive, data eigenvalues
are retained in the output tables.  Non-positive tail modes are placed at an
explicitly recorded display floor in the logarithmic spectrum.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcmc.reconstruct_wz_desi_bbn_cmb_gw import (  # noqa: E402
    N_W, horndeski_precision, w_geometry,
)
from paths import (  # noqa: E402
    DESI_BBN_CMB_GW_PCA_001_TABLES,
    DESI_BBN_CMB_GW_PCA_TABLES,
    RECONSTRUCT_WZ_DESI_BBN_CMB_GW_TABLES,
    project_path,
)

EVENT_COUNTS = (100, 500, 1000, 3000)
TABLES = DESI_BBN_CMB_GW_PCA_TABLES
STEM = "desi_bbn_cmb_gw_pca"
SPECTRUM_FLOOR = 1.2e-7


def ordered_eigensystem(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values, vectors = np.linalg.eigh(matrix)
    values, vectors = values[::-1], vectors[:, ::-1]
    for j in range(vectors.shape[1]):
        if vectors[np.argmax(abs(vectors[:, j])), j] < 0:
            vectors[:, j] *= -1
    return values, vectors


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    prior = horndeski_precision()
    prior_lambda, _ = ordered_eigensystem(prior)
    # The smoothing prior has one free constant-w direction.  Following the
    # DESI Fig. 4 convention, discard it and order the remaining prior
    # eigenvalues from weak to strong.
    prior_positive_ascending = prior_lambda[::-1][1:]
    _, z_min, z_max, z_center = w_geometry()

    spectrum_rows: list[dict] = []
    mode_rows: list[dict] = []
    amplitude_rows: list[dict] = []
    summaries: list[dict] = []
    for n in EVENT_COUNTS:
        source_a1 = (
            RECONSTRUCT_WZ_DESI_BBN_CMB_GW_TABLES
            / f"desi_dr2_bbn_cmb_gw{n:04d}_wz"
        )
        posterior_a1 = json.loads(
            (source_a1.parent / f"{source_a1.name}_summary.json").read_text())
        if not posterior_a1["rank_folded_split_rhat"]["all_pass"]:
            raise ValueError(f"GW{n} A=1 chains are not converged")
        covariance_a1 = np.loadtxt(
            source_a1.parent / f"{source_a1.name}_covariance.csv",
            delimiter=",",
            skiprows=1,
        )
        mean = np.array([posterior_a1["joint_posterior_mean"][f"w{i}"]
                         for i in range(1, N_W + 1)])
        if covariance_a1.shape != (N_W, N_W):
            raise ValueError(f"GW{n} A=1 posterior covariance has wrong shape")
        np.linalg.cholesky(covariance_a1)

        source_a001 = (
            DESI_BBN_CMB_GW_PCA_001_TABLES
            / f"desi_dr2_bbn_cmb_gw{n:04d}_pca_A001"
        )
        posterior_a001 = json.loads(
            (source_a001.parent / f"{source_a001.name}_summary.json").read_text())
        if posterior_a001.get("prior_amplitude") != 0.01:
            raise ValueError(f"GW{n} weak-prior chain does not use A=0.01")
        if not posterior_a001["rank_folded_split_rhat"]["all_pass"]:
            raise ValueError(f"GW{n} A=0.01 chains are not converged")
        covariance_a001 = np.loadtxt(
            source_a001.parent / f"{source_a001.name}_wz_covariance.csv",
            delimiter=",",
            skiprows=1,
        )
        if covariance_a001.shape != (N_W, N_W):
            raise ValueError(f"GW{n} A=0.01 posterior covariance has wrong shape")
        np.linalg.cholesky(covariance_a001)

        # Match the supplied Figure 4 notebook: subtract 0.01 times the
        # modified w-bin prior precision from the inverse weak-prior posterior
        # covariance.  The same precision is used for the separate prior
        # eigendecomposition and N_eff.
        posterior_precision_a001 = np.linalg.solve(
            covariance_a001, np.eye(N_W))
        fisher = posterior_precision_a001 - 0.01 * prior
        fisher = 0.5 * (fisher + fisher.T)
        fisher_lambda, eigenvectors = ordered_eigensystem(fisher)
        posterior_lambda, _ = ordered_eigensystem(
            np.linalg.solve(covariance_a1, np.eye(N_W)))
        negative_count = int(np.count_nonzero(fisher_lambda <= 0.0))

        np.savetxt(
            TABLES / f"{STEM}_data_fisher_gw{n:04d}.csv", fisher,
            fmt="%.12e",
            delimiter=",",
            header=",".join(f"w{i}" for i in range(1, N_W + 1)),
            comments="",
        )
        np.savetxt(
            TABLES / f"{STEM}_data_eigenvalues_raw_gw{n:04d}.csv",
            fisher_lambda,
            fmt="%.12e",
            delimiter=",",
            header="eigenvalue",
            comments="",
        )
        for j in range(N_W):
            spectrum_rows.append({
                "n_gw": n, "mode": j + 1,
                "lambda_data_A001_subtracted_raw": fisher_lambda[j],
                "positive_raw_eigenvalue": bool(fisher_lambda[j] > 0.0),
                "lambda_data_display": max(fisher_lambda[j], SPECTRUM_FLOOR),
                "lambda_prior_descending": max(prior_lambda[j], 0.0),
                "lambda_prior_ascending_nonzero": (prior_positive_ascending[j]
                                                    if j < N_W - 1 else ""),
                "lambda_posterior_A1": posterior_lambda[j],
            })
        for j in range(5):
            vector = eigenvectors[:, j]
            gamma = float(vector @ (mean + 1.0))
            sigma_posterior = math.sqrt(
                float(vector @ covariance_a1 @ vector))
            prior_precision_along_mode = float(vector @ prior @ vector)
            amplitude_rows.append({
                "n_gw": n, "pc": j + 1,
                "gamma_mean": gamma, "sigma_projection_A1": sigma_posterior,
                "gamma_over_sigma_A1": gamma / sigma_posterior,
                "lambda_data_A001_subtracted": fisher_lambda[j],
                "lambda_prior_along_same_mode": prior_precision_along_mode,
                "data_to_prior_precision_ratio": (fisher_lambda[j] / prior_precision_along_mode
                                                  if prior_precision_along_mode > 0 else math.inf),
            })
            for i in range(N_W):
                mode_rows.append({
                    "n_gw": n, "pc": j + 1, "bin": i + 1,
                    "z_min": z_min[i], "z_max": z_max[i],
                    "z_center": z_center[i], "e_i": vector[i],
                })
        projected_cov = (eigenvectors[:, :5].T @ covariance_a1
                         @ eigenvectors[:, :5])
        np.savetxt(
            TABLES / f"{STEM}_projected_cov_gw{n:04d}.csv",
            projected_cov,
            fmt="%.12e",
            delimiter=",",
            header=",".join(f"pc{i}" for i in range(1, 6)),
            comments="",
        )
        deviation = mean + 1.0
        snr = math.sqrt(float(
            deviation @ np.linalg.solve(covariance_a1, deviation)))
        if not np.isclose(snr, posterior_a1["snr_vs_lcdm"], rtol=1e-7):
            raise AssertionError(f"GW{n}: reconstructed posterior SNR differs from source")
        # Tr(P C) = sum_ij P_ij C_ji.  Evaluate it element by element to
        # avoid spurious Accelerate BLAS floating-point warnings from a full
        # matrix product when only the trace is needed.
        trace_prior_posterior = float(np.sum(prior * covariance_a1.T))
        neff = float(N_W - trace_prior_posterior)
        summaries.append({
            "n_gw": n,
            "neff_A1": neff,
            "trace_prior_precision_times_Cw": trace_prior_posterior,
            "posterior_gaussian_distance_from_w_minus_1": snr,
            "max_rank_folded_split_rhat_A1": 1.0 + posterior_a1["rank_folded_split_rhat"]["max_rhat_minus_one"],
            "max_rank_folded_split_rhat_A001": 1.0 + posterior_a001["rank_folded_split_rhat"]["max_rhat_minus_one"],
            "data_fisher_first_five": fisher_lambda[:5].tolist(),
            "data_fisher_nonpositive_count": negative_count,
            "data_fisher_minimum_raw_eigenvalue": float(fisher_lambda[-1]),
            "source_A1_chain_summary": project_path(
                source_a1.parent / f"{source_a1.name}_summary.json"
            ),
            "source_A001_chain_summary": project_path(
                source_a001.parent / f"{source_a001.name}_summary.json"
            ),
        })
        print(f"GW{n}: Neff={neff:.3f}; SNR={snr:.3f}; "
              f"DESI Fdata lambda1..5={np.array2string(fisher_lambda[:5], precision=3)}; "
              f"non-positive={negative_count}, min={fisher_lambda[-1]:.4g}")

    write_csv(TABLES / f"{STEM}_spectrum.csv", spectrum_rows)
    write_csv(TABLES / f"{STEM}_modes.csv", mode_rows)
    write_csv(TABLES / f"{STEM}_amplitudes.csv", amplitude_rows)
    summary_csv_rows = [{
        "n_gw": case["n_gw"],
        "neff_A1": case["neff_A1"],
        "trace_prior_precision_times_Cw": case["trace_prior_precision_times_Cw"],
        "posterior_gaussian_distance_from_w_minus_1": case["posterior_gaussian_distance_from_w_minus_1"],
        "max_rank_folded_split_rhat_A1": case["max_rank_folded_split_rhat_A1"],
        "max_rank_folded_split_rhat_A001": case["max_rank_folded_split_rhat_A001"],
        "data_fisher_nonpositive_count": case["data_fisher_nonpositive_count"],
        "data_fisher_minimum_raw_eigenvalue": case["data_fisher_minimum_raw_eigenvalue"],
        **{f"lambda_data_pc{j+1}": value
           for j, value in enumerate(case["data_fisher_first_five"])},
    } for case in summaries]
    write_csv(TABLES / f"{STEM}_summary.csv", summary_csv_rows)
    write_csv(TABLES / "desi_bbn_cmb_gw_neff.csv", [{
        "n_gw": case["n_gw"],
        "trace_prior_precision_times_Cw": case["trace_prior_precision_times_Cw"],
        "N_eff": case["neff_A1"],
    } for case in summaries])
    (TABLES / f"{STEM}_summary.json").write_text(
        json.dumps({"method": "Supplied Figure 4 source: F_data = inv(C_p(A=0.01)) - 0.01*Ctilde_Pi^-1; data and prior diagonalized separately",
                    "fisher_subtraction_precision": "Ctilde_Pi^-1 = (I-S)^T C_Pi^-1 (I-S), matching INV_D_w_horn_prior.txt",
                    "modified_prior_precision": "Ctilde_Pi^-1 = (I-S)^T C_Pi^-1 (I-S), S is the five-neighbour local mean",
                    "data_fisher_source": "converged A=0.01 Cobaya chains",
                    "posterior_amplitude_source": "converged A=1 Cobaya chains",
                    "spectrum_definition": "descending eigenvalues of weak-prior-subtracted F_data and ascending nonzero eigenvalues of the separately diagonalized prior precision",
                    "log_plot_rule": f"retain raw eigenvalues in tables; display non-positive F_data tail modes at the plotting floor {SPECTRUM_FLOOR:.1e}",
                    "neff_formula": "29 - trace(P_prior @ C_post_A1)",
                    "cases": summaries}, indent=2) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
