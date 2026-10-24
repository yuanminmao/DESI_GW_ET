#!/usr/bin/env python3
"""Neff and covariant PCA for the GW-only + correlation-prior reconstructions.

Definitions
-----------
For each catalogue, the production A=1 posterior supplies the marginalized
29x29 w-bin covariance C_A1 and posterior mean.  Its effective number of
data-constrained dark-energy degrees of freedom is

    Neff = 29 - Tr(P_Horndeski C_A1).

The data Fisher used for PC shapes and the spectrum is recovered from a
separate, converged A=0.01 weak-prior posterior,

    F_data = inv(C_A001) - 0.01 P_Horndeski.

This matches the convention used by ``analyze_pca_bbn_cmb_desi_gw.py`` while
using only GW absolute distances and the correlation prior.  Raw negative
tail eigenvalues are retained in tables and placed at a documented positive
floor only for the logarithmic plot.

Examples
--------
Compute Neff from the currently available A=1 products::

    python3 mcmc/analyze_pca_gw_prior_only.py --neff-only --n-events 3000

Create the four-catalogue PC1--4 and spectrum figures after both A=1 and
A=0.01 chains exist for all catalogues::

    python3 mcmc/analyze_pca_gw_prior_only.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcmc.bbn_cmb_desi_absolute_gw_wz_model import (  # noqa: E402
    N_W,
    horndeski_precision,
    w_geometry,
)
from mcmc.reconstruct_wz_gw_prior import load_catalog, sha256  # noqa: E402

DEFAULT_EVENTS = (100, 500, 1000, 3000)
WEAK_AMPLITUDE = 0.01
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
STEM = "gw_prior_only_pca"
DEFAULT_SPECTRUM_FLOOR = 1.0e-7
CATALOG_COLORS = ("#9ecae1", "#4292c6", "#2171b5", "#08306b")
MODE_COLORS = (
    "#D88200",  # PC1: orange
    "#FFCC00",  # PC2: yellow
    "#7EC200",  # PC3: green
    "#0072BD",  # PC4: blue
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--n-events", type=int, nargs="+", default=list(DEFAULT_EVENTS),
        help="Distinct catalogue sizes (default: 100 500 1000 3000)",
    )
    parser.add_argument(
        "--neff-only", action="store_true",
        help="Compute Neff from A=1 products without requiring A=0.01 chains",
    )
    parser.add_argument(
        "--spectrum-floor", type=float, default=DEFAULT_SPECTRUM_FLOOR,
        help="Positive display floor for non-positive/noisy Fisher tail modes",
    )
    args = parser.parse_args()
    args.n_events = list(dict.fromkeys(args.n_events))
    if not args.n_events or any(value <= 0 for value in args.n_events):
        parser.error("--n-events must contain positive distinct values")
    if not math.isfinite(args.spectrum_floor) or args.spectrum_floor <= 0.0:
        parser.error("--spectrum-floor must be finite and positive")
    return args


def product_stem(n_events: int, amplitude: float) -> str:
    if amplitude == 1.0:
        return f"gw{n_events:04d}_prior_wz_mcmc"
    if amplitude == WEAK_AMPLITUDE:
        return f"gw{n_events:04d}_prior_A001_wz_mcmc"
    raise ValueError(f"Unsupported prior amplitude: {amplitude}")


def load_posterior(
    n_events: int, amplitude: float
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    stem = product_stem(n_events, amplitude)
    summary_path = TABLES / f"{stem}_summary.json"
    covariance_path = TABLES / f"{stem}_covariance.txt"
    missing = [str(path) for path in (summary_path, covariance_path) if not path.is_file()]
    if missing:
        command = (
            "mpiexec -n 4 python3 mcmc/sample_wz_gw_prior.py --run "
            f"--n-events {n_events} --prior-amplitude {amplitude:g}"
        )
        raise FileNotFoundError(
            "Missing GW-only posterior products:\n  "
            + "\n  ".join(missing)
            + f"\nGenerate them with:\n  {command}"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    recorded_amplitude = float(summary.get("prior_amplitude", 1.0))
    if not math.isclose(recorded_amplitude, amplitude, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"{summary_path} records prior A={recorded_amplitude}, expected {amplitude}"
        )
    diagnostic = summary.get("rank_folded_split_rhat", {})
    if not isinstance(diagnostic, dict) or not diagnostic.get("all_pass"):
        raise ValueError(f"Unconverged posterior: {summary_path}")
    catalog = load_catalog(n_events)
    recorded_hash = summary.get("catalog", {}).get("sha256")  # type: ignore[union-attr]
    current_hash = sha256(catalog.path)
    if recorded_hash != current_hash:
        raise ValueError(
            f"Catalogue hash mismatch for GW{n_events}: posterior={recorded_hash}, current={current_hash}"
        )
    covariance = np.loadtxt(covariance_path)
    if covariance.shape != (N_W, N_W) or not np.all(np.isfinite(covariance)):
        raise ValueError(f"Invalid covariance shape/content: {covariance_path}")
    covariance = 0.5 * (covariance + covariance.T)
    np.linalg.cholesky(covariance)
    posterior_mean = summary.get("posterior_mean", {})
    mean = np.asarray(
        [posterior_mean[f"w{i}"] for i in range(1, N_W + 1)],  # type: ignore[index]
        dtype=float,
    )
    if mean.shape != (N_W,) or not np.all(np.isfinite(mean)):
        raise ValueError(f"Invalid posterior w mean in {summary_path}")
    return summary, covariance, mean


def ordered_eigensystem(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    for column in range(vectors.shape[1]):
        vector = vectors[:, column]
        if vector[np.argmax(np.abs(vector))] < 0.0:
            vectors[:, column] *= -1.0
    return values, vectors


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def style_axes(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_color("#222222")
        spine.set_linewidth(1.25)
    ax.tick_params(
        axis="both", which="major", direction="in", top=True, right=True,
        length=5.0, width=1.1, color="#222222",
    )
    ax.tick_params(
        axis="both", which="minor", direction="in", top=True, right=True,
        length=3.0, width=0.8, color="#222222",
    )


def event_tag(event_counts: list[int]) -> str:
    return "_".join(f"{value:04d}" for value in event_counts)


def plot_modes(
    event_counts: list[int], modes_by_n: dict[int, np.ndarray], output: Path
) -> None:
    _, _, _, z_centres = w_geometry()
    order = np.argsort(z_centres)
    centres = z_centres[order]
    z_plot = np.linspace(0.0, 2.5, 800)
    ncols = 2 if len(event_counts) > 1 else 1
    nrows = math.ceil(len(event_counts) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(11.8, 4.0 * nrows), sharex=True, sharey=True,
        squeeze=False, facecolor="white",
    )
    flat_axes = axes.ravel()
    for ax, n_events in zip(flat_axes, event_counts):
        modes = modes_by_n[n_events][order]
        for pc, color in enumerate(MODE_COLORS):
            values = modes[:, pc]
            nodes = np.r_[0.0, centres, 2.5]
            smooth = PchipInterpolator(
                nodes, np.r_[values[0], values, values[-1]]
            )(z_plot)
            ax.plot(z_plot, smooth, color=color, lw=2.5, label=f"PC{pc + 1}")
        ax.axhline(0.0, color="0.45", lw=0.9)
        ax.set(
            xlim=(0.0, 2.5), ylim=(-0.6, 0.6),
            xticks=[0.5, 1.0, 1.5, 2.0, 2.5],
            yticks=[-0.6, -0.3, 0.0, 0.3, 0.6],
            xlabel=r"$z$", ylabel=r"$e_i(z)$", title=f"GW{n_events}",
        )
        style_axes(ax)
    for ax in flat_axes[len(event_counts):]:
        ax.set_visible(False)
    handles, labels = flat_axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995),
        ncol=4, frameon=False, fontsize=12,
    )
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.09, top=0.90,
                        wspace=0.16, hspace=0.24)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_spectrum(
    event_counts: list[int],
    spectra_by_n: dict[int, np.ndarray],
    prior_positive_ascending: np.ndarray,
    *,
    floor: float,
    output: Path,
) -> None:
    indices = np.arange(1, N_W + 1)
    dense = np.linspace(1.0, float(N_W), 1000)
    fig, ax = plt.subplots(figsize=(13.2, 7.8), facecolor="white")
    colors = CATALOG_COLORS[-len(event_counts):]
    positive_values: list[float] = []
    for color, n_events in zip(colors, event_counts):
        raw = spectra_by_n[n_events]
        positive_values.extend(raw[raw > 0.0].tolist())
        displayed = np.maximum(raw, floor)
        smooth = 10.0 ** PchipInterpolator(indices, np.log10(displayed))(dense)
        ax.semilogy(dense, smooth, lw=4.0, color=color, label=f"GW {n_events}")
    prior_indices = np.arange(1, N_W)
    prior_dense = np.linspace(1.0, float(N_W - 1), 900)
    prior_smooth = 10.0 ** PchipInterpolator(
        prior_indices, np.log10(prior_positive_ascending)
    )(prior_dense)
    ax.semilogy(
        prior_dense, prior_smooth, color="#D95319", lw=4.0,
        label="Horndeski prior",
    )
    positive_values.extend(prior_positive_ascending.tolist())
    visible_min = max(floor * 10.0, min(positive_values) / 3.0)
    visible_max = max(positive_values) * 2.5
    ax.set(
        xlim=(1, N_W), ylim=(visible_min, visible_max),
        xticks=[5, 10, 15, 20, 25, 29],
        xlabel="Mode number", ylabel=r"$\lambda_i$",
    )
    ax.legend(loc="center right", frameon=False, fontsize=22)
    ax.xaxis.label.set_size(28)
    ax.yaxis.label.set_size(28)
    ax.tick_params(axis="both", which="major", labelsize=25)
    style_axes(ax)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    prior = horndeski_precision()
    if prior.shape != (N_W, N_W):
        raise ValueError("Unexpected prior precision shape")
    prior_eigenvalues, _ = ordered_eigensystem(prior)
    # One exactly free constant-w mode is removed; the remaining prior
    # eigenvalues are displayed weak-to-strong, as in the reference figure.
    prior_positive_ascending = prior_eigenvalues[::-1][1:]
    if np.any(prior_positive_ascending <= 0.0):
        raise ValueError("Expected 28 positive non-constant prior modes")
    _, z_min, z_max, z_centres = w_geometry()

    neff_rows: list[dict[str, object]] = []
    spectrum_rows: list[dict[str, object]] = []
    mode_rows: list[dict[str, object]] = []
    cases: list[dict[str, object]] = []
    modes_by_n: dict[int, np.ndarray] = {}
    spectra_by_n: dict[int, np.ndarray] = {}

    for n_events in args.n_events:
        a1_summary, covariance_a1, mean_a1 = load_posterior(n_events, 1.0)
        trace = float(np.sum(prior * covariance_a1.T))
        neff = float(N_W - trace)
        snr = math.sqrt(
            max(float((mean_a1 + 1.0) @ np.linalg.solve(covariance_a1, mean_a1 + 1.0)), 0.0)
        )
        neff_rows.append(
            {
                "n_gw": n_events,
                "trace_prior_precision_times_Cw_A1": trace,
                "N_eff_A1": neff,
                "snr_mean_vs_w_minus_one_A1": snr,
                "max_rank_folded_split_rhat_minus_one_A1": a1_summary[
                    "rank_folded_split_rhat"
                ]["max_rhat_minus_one"],
                "catalog_sha256": a1_summary["catalog"]["sha256"],
            }
        )
        print(f"GW{n_events:04d}: Neff={neff:.6f}, SNR={snr:.6f}")
        if args.neff_only:
            continue

        weak_summary, covariance_weak, _ = load_posterior(
            n_events, WEAK_AMPLITUDE
        )
        if weak_summary["catalog"]["sha256"] != a1_summary["catalog"]["sha256"]:
            raise ValueError(f"GW{n_events}: A=1 and A=0.01 catalogue hashes differ")
        posterior_precision_weak = np.linalg.solve(
            covariance_weak, np.eye(N_W)
        )
        fisher = posterior_precision_weak - WEAK_AMPLITUDE * prior
        fisher = 0.5 * (fisher + fisher.T)
        fisher_values, fisher_vectors = ordered_eigensystem(fisher)
        modes_by_n[n_events] = fisher_vectors[:, :4]
        spectra_by_n[n_events] = fisher_values
        nonpositive = int(np.count_nonzero(fisher_values <= 0.0))
        condition_weak = float(np.linalg.cond(covariance_weak))

        np.savetxt(
            TABLES / f"{STEM}_data_fisher_gw{n_events:04d}.txt",
            fisher,
            fmt="%.12e",
            header="GW-only F_data = inv(C_w,A=0.01) - 0.01 P_Horndeski",
        )
        np.savetxt(
            TABLES / f"{STEM}_data_eigenvalues_raw_gw{n_events:04d}.txt",
            fisher_values,
            fmt="%.12e",
            header="Descending raw GW-only data-Fisher eigenvalues; non-positive tail retained",
        )
        for mode in range(N_W):
            spectrum_rows.append(
                {
                    "n_gw": n_events,
                    "mode": mode + 1,
                    "lambda_data_raw": fisher_values[mode],
                    "positive_raw_eigenvalue": bool(fisher_values[mode] > 0.0),
                    "lambda_data_display": max(fisher_values[mode], args.spectrum_floor),
                    "lambda_prior_ascending_nonzero": (
                        prior_positive_ascending[mode] if mode < N_W - 1 else ""
                    ),
                }
            )
        for pc in range(4):
            vector = fisher_vectors[:, pc]
            for i in range(N_W):
                mode_rows.append(
                    {
                        "n_gw": n_events,
                        "pc": pc + 1,
                        "bin": i + 1,
                        "z_min": z_min[i],
                        "z_max": z_max[i],
                        "z_center": z_centres[i],
                        "e_i": vector[i],
                    }
                )
        cases.append(
            {
                "n_gw": n_events,
                "N_eff_A1": neff,
                "trace_prior_precision_times_Cw_A1": trace,
                "snr_mean_vs_w_minus_one_A1": snr,
                "first_four_data_fisher_eigenvalues": fisher_values[:4].tolist(),
                "data_fisher_nonpositive_count": nonpositive,
                "data_fisher_minimum_raw_eigenvalue": float(fisher_values[-1]),
                "weak_posterior_covariance_condition_number": condition_weak,
                "catalog_sha256": a1_summary["catalog"]["sha256"],
                "source_A1_summary": str(
                    (TABLES / f"{product_stem(n_events, 1.0)}_summary.json").resolve()
                ),
                "source_A001_summary": str(
                    (TABLES / f"{product_stem(n_events, WEAK_AMPLITUDE)}_summary.json").resolve()
                ),
            }
        )
        print(
            f"  lambda_data PC1..4={np.array2string(fisher_values[:4], precision=5)}; "
            f"non-positive tail modes={nonpositive}"
        )

    neff_path = TABLES / "gw_prior_only_neff.csv"
    write_csv(neff_path, neff_rows)
    print(f"Saved: {neff_path.resolve()}")
    if args.neff_only:
        return

    write_csv(TABLES / f"{STEM}_spectrum.csv", spectrum_rows)
    write_csv(TABLES / f"{STEM}_modes_pc1-4.csv", mode_rows)
    tag = event_tag(args.n_events)
    modes_figure = FIGURES / f"gw_prior_only_wz_pca_modes_pc1-4_gw{tag}_preview.png"
    spectrum_figure = FIGURES / f"{STEM}_spectrum.png"
    plot_modes(args.n_events, modes_by_n, modes_figure)
    plot_spectrum(
        args.n_events,
        spectra_by_n,
        prior_positive_ascending,
        floor=args.spectrum_floor,
        output=spectrum_figure,
    )
    summary_path = TABLES / f"{STEM}_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "probe_combination": "absolute ET GW distances + modified Horndeski correlation prior only",
                "explicitly_excluded": ["DESI", "CMB", "BBN", "omega_b"],
                "neff_formula": "29 - Tr(P_Horndeski C_w,A=1)",
                "data_fisher_formula": "inv(C_w,A=0.01) - 0.01 P_Horndeski",
                "mode_order": "descending eigenvalues of the prior-subtracted data Fisher",
                "mode_sign": "largest-absolute component is positive",
                "prior_spectrum": "28 non-constant prior eigenvalues in ascending order",
                "spectrum_display_floor": args.spectrum_floor,
                "cases": cases,
                "outputs": {
                    "neff_csv": str(neff_path.resolve()),
                    "modes_csv": str((TABLES / f"{STEM}_modes_pc1-4.csv").resolve()),
                    "spectrum_csv": str((TABLES / f"{STEM}_spectrum.csv").resolve()),
                    "modes_figure": str(modes_figure.resolve()),
                    "spectrum_figure": str(spectrum_figure.resolve()),
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved: {modes_figure.resolve()}")
    print(f"Saved: {spectrum_figure.resolve()}")
    print(f"Saved: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
