#!/usr/bin/env python3
"""DESI Fig. 4 PCA for BBN+CMB+DESI DR2+absolute ET GW catalogues.

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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import PchipInterpolator
from scipy.linalg import solve_triangular

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcmc.bbn_cmb_desi_absolute_gw_wz_model import (  # noqa: E402
    BBN_MEAN, BBN_SIGMA, CMB_THETA100_MEAN, CMB_THETA100_SIGMA,
    C_KM_S, N_W, PARAMETER_NAMES, EarlyBackground, JointLikelihood,
    horndeski_precision, load_observations, w_geometry,
)

EVENT_COUNTS = (100, 500, 1000, 3000)
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
STEM = "desi_dr2_bbn_cmb_absolute_gw_pca"
STEP = np.array([0.005, 0.0002, 0.000002] + [0.002] * N_W)
SPECTRUM_FLOOR = 1.2e-7


def whitened_predictions(model: JointLikelihood, point: np.ndarray,
                         early: tuple | None = None) -> np.ndarray:
    """Return all model observables after applying their data whitening."""
    h0, omega_m, omega_b = point[:3]
    if early is None:
        early = model.early.evaluate(h0, omega_m, omega_b, model.z_grid)
    dmstar_lcdm, rstar, rd, zstar, omega_de, hubble_grid = early
    # Some Accelerate BLAS builds report spurious FP flags for finite matmuls.
    # Explicit finite/positivity checks below still reject invalid histories.
    with np.errstate(all="ignore"):
        rho_de = np.exp(model.kernel @ (1.0 + point[3:]))
    e2_lcdm = (hubble_grid / h0)**2
    e2 = e2_lcdm + omega_de * (rho_de - 1.0)
    if not np.all(np.isfinite(e2) & (e2 > 0)):
        raise ValueError("Invalid perturbed expansion history")
    early_de_fraction = abs(omega_de * (rho_de[-1] - 1.0)) / np.interp(
        zstar, model.z_grid, e2_lcdm)
    if early_de_fraction > 1e-4:
        raise ValueError("Perturbation changes recombination physics")

    dm_grid = (C_KM_S / h0) * cumulative_trapezoid(
        model.one_plus_z / np.sqrt(e2), model.u_grid, initial=0.0)
    gw = (1.0 + model.data.gw_z) * np.interp(
        model.data.gw_z, model.z_grid, dm_grid) / model.data.gw_sigma

    dm_bao = np.interp(model.data.bao_z, model.z_grid, dm_grid)
    e2_bao = np.interp(model.data.bao_z, model.z_grid, e2)
    dh_bao = C_KM_S / (h0 * np.sqrt(e2_bao))
    bao = np.empty(len(dm_bao))
    for i, quantity in enumerate(model.data.bao_quantity):
        if quantity == "DM_over_rs":
            bao[i] = dm_bao[i] / rd
        elif quantity == "DH_over_rs":
            bao[i] = dh_bao[i] / rd
        else:
            bao[i] = np.cbrt(model.data.bao_z[i] * dm_bao[i]**2 * dh_bao[i]) / rd
    chol = np.tril(model.data.bao_cov_factor[0])
    bao_white = solve_triangular(chol, bao, lower=True)

    delta_integrand = model.one_plus_z * (
        1.0 / np.sqrt(e2) - 1.0 / np.sqrt(e2_lcdm))
    delta_dm = (C_KM_S / h0) * cumulative_trapezoid(
        delta_integrand, model.u_grid, initial=0.0)
    dmstar = dmstar_lcdm + np.interp(zstar, model.z_grid, delta_dm)
    theta100 = 100.0 * rstar / dmstar
    return np.r_[gw, bao_white, theta100 / CMB_THETA100_SIGMA,
                 omega_b / BBN_SIGMA]


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


def style_axes(ax: plt.Axes) -> None:
    """Apply one restrained publication frame to every PCA panel."""
    for spine in ax.spines.values():
        spine.set_color("#333333")
        spine.set_linewidth(1.05)
    ax.tick_params(
        axis="both", which="major", direction="in", top=True, right=True,
        length=5.0, width=1.0, color="#333333",
    )
    ax.tick_params(
        axis="both", which="minor", direction="in", top=True, right=True,
        length=3.0, width=0.8, color="#333333",
    )


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
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
    modes_by_n: dict[int, np.ndarray] = {}
    spectra_by_n: dict[int, np.ndarray] = {}
    for n in EVENT_COUNTS:
        source_a1 = TABLES / f"desi_dr2_bbn_cmb_absolute_gw{n:04d}_wz"
        posterior_a1 = json.loads(
            (TABLES / f"{source_a1.name}_summary.json").read_text())
        if not posterior_a1["rank_folded_split_rhat"]["all_pass"]:
            raise ValueError(f"GW{n} A=1 chains are not converged")
        covariance_a1 = np.loadtxt(TABLES / f"{source_a1.name}_covariance.txt")
        mean = np.array([posterior_a1["joint_posterior_mean"][f"w{i}"]
                         for i in range(1, N_W + 1)])
        if covariance_a1.shape != (N_W, N_W):
            raise ValueError(f"GW{n} A=1 posterior covariance has wrong shape")
        np.linalg.cholesky(covariance_a1)

        source_a001 = TABLES / f"desi_dr2_bbn_cmb_absolute_gw{n:04d}_pca_A001"
        posterior_a001 = json.loads(
            (TABLES / f"{source_a001.name}_summary.json").read_text())
        if posterior_a001.get("prior_amplitude") != 0.01:
            raise ValueError(f"GW{n} weak-prior chain does not use A=0.01")
        if not posterior_a001["rank_folded_split_rhat"]["all_pass"]:
            raise ValueError(f"GW{n} A=0.01 chains are not converged")
        covariance_a001 = np.loadtxt(TABLES / f"{source_a001.name}_wz_covariance.txt")
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
            TABLES / f"{STEM}_data_fisher_gw{n:04d}.txt", fisher,
            fmt="%.12e",
            header="F_data = inv(C_p(A=0.01)) - 0.01*inv(Ctilde_Pi)",
        )
        np.savetxt(
            TABLES / f"{STEM}_data_eigenvalues_raw_gw{n:04d}.txt",
            fisher_lambda, fmt="%.12e",
            header="Descending raw eigenvalues of F_data; non-positive values are retained",
        )
        modes_by_n[n] = eigenvectors[:, :3]
        spectra_by_n[n] = fisher_lambda
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
        np.savetxt(TABLES / f"{STEM}_projected_cov_gw{n:04d}.txt",
                   projected_cov, fmt="%.12e",
                   header="Posterior A=1 covariance of first five DESI-method data-Fisher PCs; off-diagonal terms are nonzero")
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
            "source_A1_chain_summary": str(TABLES / f"{source_a1.name}_summary.json"),
            "source_A001_chain_summary": str(TABLES / f"{source_a001.name}_summary.json"),
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
    write_csv(TABLES / "desi_dr2_bbn_cmb_absolute_gw_neff.csv", [{
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

    # Reuse the project's established palette. GW samples follow the four
    # light-to-dark blues used by the earlier event-count comparison; PC1 uses
    # the orange from reconstruction points, PC2 uses deep lemon,
    # and PC3 uses the third blue from the event-count palette.
    # The saved tables retain the original discrete-bin values; interpolation
    # is display-only.
    colors = ("#9ecae1", "#4292c6", "#2171b5", "#08306b")
    mode_colors = ("#D95319", "#FFCC00", "#2171b5")
    z_plot = np.linspace(0.0, 1.5, 700)
    z_nodes = np.r_[0.0, z_center, 2.5]
    scaled_font = lambda key: FontProperties(  # noqa: E731
        size=plt.rcParams[key]).get_size_in_points() * 1.12
    with plt.rc_context({
        "font.size": scaled_font("font.size"),
        "axes.labelsize": scaled_font("axes.labelsize"),
        "axes.titlesize": scaled_font("axes.titlesize"),
        "xtick.labelsize": scaled_font("xtick.labelsize"),
        "ytick.labelsize": scaled_font("ytick.labelsize"),
    }):
        fig, axs = plt.subplots(2, 2, figsize=(11.5, 7.0), sharex=True, sharey=True)
        for ax, n in zip(axs.flat, EVENT_COUNTS):
            for j, color in enumerate(mode_colors):
                values = modes_by_n[n][:, j]
                smooth = PchipInterpolator(
                    z_nodes, np.r_[values[0], values, values[-1]])(z_plot)
                ax.plot(z_plot, smooth, color=color, lw=2.0, label=f"PC{j+1}")
            ax.axhline(0, color="0.45", lw=0.8)
            ax.set(xlim=(0, 1.5), xlabel="Redshift z", title=f"GW {n}")
            style_axes(ax)
        for ax in axs[:, 0]:
            ax.set_ylabel(r"$e_i(z)$")
        axs[0, 1].legend(loc="upper right", frameon=False, fontsize=9 * 1.12)
        fig.tight_layout()
        fig.savefig(FIGURES / f"{STEM}_modes.png", dpi=220,
                    bbox_inches="tight", pad_inches=0.08, facecolor="white")
        plt.close(fig)

    indices = np.arange(1, N_W + 1)
    prior_indices = indices[:-1]
    fig, ax = plt.subplots(figsize=(13.2, 7.8))
    for color, n in zip(colors, EVENT_COUNTS):
        # Keep all 29 modes in the plotted x-range.  The weak-prior
        # subtraction produces noisy/non-positive tail eigenvalues; retain
        # those modes at the explicit plotting floor so the tail is drawn and
        # then clipped below the visible y-range instead of being omitted.
        data_indices = indices
        display_values = np.maximum(spectra_by_n[n], SPECTRUM_FLOOR)
        data_plot = np.linspace(1.0, float(N_W), 900)
        data_smooth = 10.0 ** PchipInterpolator(
            data_indices, np.log10(display_values))(data_plot)
        ax.semilogy(data_plot, data_smooth, lw=4.2, color=color,
                    label=f"GW {n}")
    prior_plot = np.linspace(1.0, float(N_W - 1), 700)
    prior_smooth = 10.0 ** PchipInterpolator(
        prior_indices, np.log10(prior_positive_ascending))(prior_plot)
    ax.semilogy(prior_plot, prior_smooth, color="#D95319", lw=4.2,
                label="Horndeski prior")
    ax.set(xlim=(1, 29), ylim=(1e-3, 3e2),
           xticks=[5, 10, 15, 20, 25, 29])
    ax.set_xlabel("Mode number", fontsize=30)
    ax.set_ylabel(r"$\lambda_i$", fontsize=30)
    ax.legend(loc="center right", frameon=False, fontsize=26, ncol=1)
    style_axes(ax)
    for spine in ax.spines.values():
        spine.set_color("#111111")
        spine.set_linewidth(2.10)
    ax.tick_params(axis="both", which="major", labelsize=30,
                   color="#111111", width=2.0)
    ax.tick_params(axis="both", which="minor", labelsize=30,
                   color="#111111", width=1.6)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{STEM}_spectrum.png", dpi=220,
                bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    x = np.arange(1, 6)
    for offset, color, n in zip((-0.27, -0.09, 0.09, 0.27), colors, EVENT_COUNTS):
        ratios = [row["data_to_prior_precision_ratio"] for row in amplitude_rows
                  if row["n_gw"] == n]
        ax.bar(x + offset, ratios, width=0.17, color=color,
               edgecolor="white", linewidth=0.65, label=f"GW {n}")
    ax.set(xlabel="Mode number", ylabel="Data",
           xticks=x, yscale="log", ylim=(0.1, 2000))
    ax.legend(frameon=False, ncol=4, fontsize=9)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{STEM}_prior_ratio.png", dpi=220,
                bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
