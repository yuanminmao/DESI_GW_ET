#!/usr/bin/env python3
"""Plot GW-only and joint DESI+BBN+CMB+GW PCA spectra side by side."""

from __future__ import annotations

import argparse
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
)


EVENT_COUNTS = (100, 500, 1000, 3000)
COLORS = ("#9ecae1", "#4292c6", "#2171b5", "#08306b")
PRIOR_COLOR = "#D95319"
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
DEFAULT_OUTPUT = (
    FIGURES
    / "gw_prior_only_and_desi_dr2_bbn_cmb_absolute_gw_pca_spectrum_side_by_side.png"
)
DISPLAY_FLOOR = 1.0e-7
TEXT_SCALE = 1.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    if args.dpi < 50:
        parser.error("--dpi must be at least 50")
    return args


def ordered_eigenvalues(matrix: np.ndarray) -> np.ndarray:
    matrix = 0.5 * (matrix + matrix.T)
    return np.linalg.eigvalsh(matrix)[::-1]


def load_spectra(stem: str) -> dict[int, np.ndarray]:
    spectra: dict[int, np.ndarray] = {}
    for n_events in EVENT_COUNTS:
        path = TABLES / f"{stem}_data_eigenvalues_raw_gw{n_events:04d}.txt"
        values = np.asarray(np.loadtxt(path), dtype=float)
        if values.shape != (N_W,) or not np.all(np.isfinite(values)):
            raise ValueError(f"Invalid PCA spectrum: {path}")
        if np.any(np.diff(values) > 1.0e-10):
            raise ValueError(f"PCA eigenvalues are not descending: {path}")
        spectra[n_events] = values
    return spectra


def draw_panel(
    ax: plt.Axes,
    spectra: dict[int, np.ndarray],
    prior_positive_ascending: np.ndarray,
    *,
    title: str,
) -> None:
    indices = np.arange(1, N_W + 1)
    dense = np.linspace(1.0, float(N_W), 1000)
    for color, n_events in zip(COLORS, EVENT_COUNTS):
        displayed = np.maximum(spectra[n_events], DISPLAY_FLOOR)
        smooth = 10.0 ** PchipInterpolator(
            indices, np.log10(displayed)
        )(dense)
        ax.semilogy(
            dense,
            smooth,
            color=color,
            lw=3.2,
            label=f"GW {n_events}",
        )

    prior_indices = np.arange(1, N_W)
    prior_dense = np.linspace(1.0, float(N_W - 1), 900)
    prior_smooth = 10.0 ** PchipInterpolator(
        prior_indices, np.log10(prior_positive_ascending)
    )(prior_dense)
    ax.semilogy(
        prior_dense,
        prior_smooth,
        color=PRIOR_COLOR,
        lw=3.2,
        label="Horndeski prior",
    )
    ax.set(
        xlim=(1, N_W),
        ylim=(1.0e-3, 3.0e2),
        xticks=[5, 10, 15, 20, 25, 29],
        xlabel="Mode number",
    )
    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        top=True,
        right=True,
        length=6.0,
        width=1.35,
        labelsize=18 * TEXT_SCALE,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        direction="in",
        top=True,
        right=True,
        length=3.5,
        width=1.0,
    )
    ax.xaxis.label.set_size(22 * TEXT_SCALE * 0.8)
    ax.text(
        0.965,
        0.95,
        title,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=22 * 1.2,
        fontweight="bold",
        zorder=10,
    )
    for spine in ax.spines.values():
        spine.set_color("#222222")
        spine.set_linewidth(2.1)


def main() -> None:
    args = parse_args()
    gw_only = load_spectra("gw_prior_only_pca")
    joint = load_spectra("desi_dr2_bbn_cmb_absolute_gw_pca")

    prior_eigenvalues = ordered_eigenvalues(horndeski_precision())
    prior_positive_ascending = prior_eigenvalues[::-1][1:]
    if (
        prior_positive_ascending.shape != (N_W - 1,)
        or np.any(prior_positive_ascending <= 0.0)
    ):
        raise ValueError("Expected 28 positive non-constant Horndeski-prior modes")

    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "xtick.direction": "in",
        "ytick.direction": "in",
    })
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(18.0, 7.2),
        sharex=True,
        sharey=True,
        facecolor="white",
    )
    draw_panel(axes[0], gw_only, prior_positive_ascending, title="GW-only")
    draw_panel(
        axes[1],
        joint,
        prior_positive_ascending,
        title="DESI+BBN+CMB+GW",
    )
    axes[0].set_ylabel(r"$\lambda_i$", fontsize=24 * TEXT_SCALE)
    axes[1].tick_params(axis="y", which="both", labelleft=False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=len(labels),
        frameon=False,
        fontsize=17 * TEXT_SCALE,
        columnspacing=1.35,
        handlelength=2.7,
        handletextpad=0.55,
    )
    fig.subplots_adjust(
        left=0.095,
        right=0.975,
        bottom=0.17,
        top=0.82,
        wspace=0.08,
    )

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=args.dpi, facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    print(f"Saved: {output}")
    print(f"Saved: {output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
