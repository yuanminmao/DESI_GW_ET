#!/usr/bin/env python3
"""Compute cumulative PCA SNR of the best retained MCMC sample.

The PCA basis is the eigensystem of the marginalized 29-bin posterior
precision C_w^{-1}.  Modes are ordered from largest to smallest eigenvalue,
so sigma(gamma_i) = lambda_i**(-1/2) and

    SNR_m = sqrt(sum_{i=1}^m gamma_i**2 * lambda_i).

At m=29 this is identically the covariance-weighted distance
sqrt[(w_best + 1)^T C_w^{-1} (w_best + 1)].
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
EVENT_COUNTS = (100, 500, 1000, 3000)
N_W = 29
STEM = "desi_dr2_bbn_cmb_absolute_gw_pca_cumulative_snr_best"


def ordered_posterior_modes(covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    precision = np.linalg.solve(covariance, np.eye(N_W))
    precision = 0.5 * (precision + precision.T)
    eigenvalues, eigenvectors = np.linalg.eigh(precision)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    if np.any(eigenvalues <= 0.0):
        raise ValueError("Posterior precision is not positive definite")
    # Fix the otherwise arbitrary signs to make signed s_i reproducible.
    for i in range(N_W):
        pivot = np.argmax(np.abs(eigenvectors[:, i]))
        if eigenvectors[pivot, i] < 0.0:
            eigenvectors[:, i] *= -1.0
    return eigenvalues, eigenvectors


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int]] = []
    checks: list[dict[str, float | int]] = []

    for n_gw in EVENT_COUNTS:
        source = f"desi_dr2_bbn_cmb_absolute_gw{n_gw:04d}_wz"
        summary = json.loads((TABLES / f"{source}_summary.json").read_text())
        if not summary["rank_folded_split_rhat"]["all_pass"]:
            raise ValueError(f"GW{n_gw} chain is not converged")
        covariance = np.loadtxt(TABLES / f"{source}_covariance.txt")
        if covariance.shape != (N_W, N_W):
            raise ValueError(f"GW{n_gw} covariance has shape {covariance.shape}")

        eigenvalues, eigenvectors = ordered_posterior_modes(covariance)
        w_best = np.array([summary["map_sample"][f"w{i}"]
                           for i in range(1, N_W + 1)])
        departure = w_best + 1.0
        gamma = eigenvectors.T @ departure
        sigma = 1.0 / np.sqrt(eigenvalues)
        signed_snr = gamma / sigma
        cumulative_snr = np.sqrt(np.cumsum(signed_snr**2))
        direct_snr = float(np.sqrt(
            departure @ np.linalg.solve(covariance, departure)))
        if not np.isclose(cumulative_snr[-1], direct_snr, rtol=2e-12):
            raise AssertionError(f"GW{n_gw}: PCA and direct SNR disagree")
        reported = float(summary["snr_best_chain_sample_vs_lcdm"])
        if not np.isclose(direct_snr, reported, rtol=2e-7):
            raise AssertionError(f"GW{n_gw}: result disagrees with chain summary")

        for i in range(N_W):
            rows.append({
                "n_gw": n_gw,
                "mode": i + 1,
                "lambda_posterior": eigenvalues[i],
                "sigma_gamma": sigma[i],
                "gamma_best": gamma[i],
                "signed_mode_snr": signed_snr[i],
                "mode_snr_squared": signed_snr[i] ** 2,
                "cumulative_snr": cumulative_snr[i],
            })
        checks.append({
            "n_gw": n_gw,
            "snr_mode_1": cumulative_snr[0],
            "snr_mode_3": cumulative_snr[2],
            "snr_mode_5": cumulative_snr[4],
            "snr_mode_10": cumulative_snr[9],
            "snr_mode_29": cumulative_snr[-1],
            "snr_direct_quadratic_form": direct_snr,
            "snr_reported_in_source_summary": reported,
        })

    with (TABLES / f"{STEM}.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (TABLES / f"{STEM}_summary.json").write_text(
        json.dumps({
            "basis": "eigenvectors of the marginalized posterior precision inv(C_w)",
            "ordering": "descending lambda (best constrained first)",
            "reference": "w_Lambda=(-1,...,-1)",
            "estimate": "minimum-minuslogpost retained MCMC sample (map_sample)",
            "formula": "SNR_m=sqrt(sum_{i<=m} gamma_i^2 lambda_i)",
            "checks": checks,
        }, indent=2) + "\n", encoding="utf-8")

    colors = ("#9ecae1", "#4292c6", "#2171b5", "#08306b")
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    for color, n_gw in zip(colors, EVENT_COUNTS):
        selected = [row for row in rows if row["n_gw"] == n_gw]
        ax.plot([row["mode"] for row in selected],
                [row["cumulative_snr"] for row in selected],
                marker="o", ms=3.2, lw=2.0, color=color, label=f"GW {n_gw}")
    ax.set(xlabel="Number of retained PCA modes, m",
           ylabel=r"Cumulative $\mathrm{SNR}_m$", xlim=(1, N_W))
    ax.set_xticks([1, 5, 10, 15, 20, 25, 29])
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{STEM}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    for check in checks:
        print("GW{n_gw}: SNR_1={snr_mode_1:.4f}, SNR_3={snr_mode_3:.4f}, "
              "SNR_5={snr_mode_5:.4f}, SNR_10={snr_mode_10:.4f}, "
              "SNR_29={snr_mode_29:.4f}".format(**check))


if __name__ == "__main__":
    main()
