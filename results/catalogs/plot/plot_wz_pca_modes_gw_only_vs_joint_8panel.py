#!/usr/bin/env python3
"""Plot GW-only and joint PC1--PC4 eigenmodes in a 2x4 comparison.

The top row contains modes from absolute ET GW distances only, while the
bottom row contains modes from DESI DR2 BAO + BBN + CMB + absolute ET GW.
Columns correspond to 100, 500, 1000, and 3000 standard sirens.

The Fisher-matrix PCA follows the supplied ``mode.ipynb`` algorithm: take
the real eigensystem, rank modes by descending absolute eigenvalue, normalize
every eigenvector to unit Euclidean norm, and plot it at the 29 redshifts
defined by the left edges of bins that are uniform in scale factor.  A
shape-preserving PCHIP curve is drawn through the unchanged discrete modes,
with the final-bin value held to z=2.5.  Eigenvector signs are fixed
deterministically because an eigenvector and its negative represent the same
principal component.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paths import (  # noqa: E402
    FIGURES,
    DESI_BBN_CMB_GW_PCA_TABLES,
    GW_PCA_TABLES,
    PCA_TABLES,
)

EVENT_COUNTS = (100, 500, 1000, 3000)
N_W = 29
N_MODES = 4
Z_MAX = 2.5
Y_LIM = (-0.6, 0.6)
FIGSIZE = (24.0, 9.0)
TEXT_SCALE = 1.5

POSTERIOR_GW_ONLY_STEM = "gw_pca_data_fisher"
POSTERIOR_JOINT_STEM = "desi_bbn_cmb_gw_pca_data_fisher"
GW_POSTERIOR_SUMMARY = GW_PCA_TABLES / "gw_pca_summary.json"
JOINT_POSTERIOR_SUMMARY = (
    DESI_BBN_CMB_GW_PCA_TABLES / "desi_bbn_cmb_gw_pca_summary.json"
)
A1_MODES_PATH = (
    PCA_TABLES / "analyze_pca_a1_posterior" / "a1_posterior_pca_modes_pc1-4.csv"
)
DEFAULT_OUTPUT = (
    FIGURES
    / "wz_pca_modes_pc1-4_gw_vs_desi_bbn_cmb_gw0100_0500_1000_3000.png"
)
MODE_NOTEBOOK_OUTPUT = (
    FIGURES
    / "wz_pca_modes_pc1-4_weak_prior_fisher_gw_vs_desi_bbn_cmb_gw0100_0500_1000_3000.png"
)
A1_OUTPUT = (
    FIGURES
    / "wz_pca_modes_pc1-4_A1_posterior_gw_vs_desi_bbn_cmb_gw0100_0500_1000_3000.png"
)

MODE_COLORS = (
    "#D88200",  # PC1: orange
    "#FFCC00",  # PC2: yellow
    "#7EC200",  # PC3: green
    "#0072BD",  # PC4: blue
)
LINE_WIDTH = 3.0
ZERO_LINE_WIDTH = 1.1
SPINE_WIDTH = 1.8
REFERENCE_BOX_ASPECT = 5.0 / 8.8
TITLE_SIZE = 21
LABEL_SIZE = 21 * TEXT_SCALE
TICK_SIZE = 16 * TEXT_SCALE
LEGEND_SIZE = 17 * TEXT_SCALE
ROW_LABEL_SIZE = 19


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--mode-ipynb",
        action="store_true",
        help=(
            "Use the supplied notebook's full weak-prior subtraction, "
            "F_data=inv(C_A001)-0.01*P_prior, for the requested target figure"
        ),
    )
    source.add_argument(
        "--a1",
        action="store_true",
        help="Use PC1--PC4 from the production A=1 posterior covariances",
    )
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    if args.dpi < 50:
        parser.error("--dpi must be at least 50")
    return args


def validate_notebook_inputs() -> None:
    """Require both rows to come from the notebook's weak-prior subtraction."""
    required_phrases = {
        GW_POSTERIOR_SUMMARY: ("A=0.01", "0.01"),
        JOINT_POSTERIOR_SUMMARY: ("A=0.01", "0.01"),
    }
    for path, phrases in required_phrases.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing weak-prior PCA summary: {path}")
        summary = json.loads(path.read_text(encoding="utf-8"))
        description = json.dumps(summary, sort_keys=True)
        if not all(phrase in description for phrase in phrases):
            raise ValueError(
                f"PCA summary does not document the A=0.01 prior subtraction: {path}"
            )
    for directory, stem in (
        (GW_PCA_TABLES, POSTERIOR_GW_ONLY_STEM),
        (DESI_BBN_CMB_GW_PCA_TABLES, POSTERIOR_JOINT_STEM),
    ):
        for n_events in EVENT_COUNTS:
            path = directory / f"{stem}_gw{n_events:04d}.csv"
            fisher = np.asarray(
                np.loadtxt(path, delimiter=",", skiprows=1), dtype=float
            )
            if fisher.shape != (N_W, N_W) or not np.all(np.isfinite(fisher)):
                raise ValueError(f"Invalid weak-prior-subtracted Fisher matrix: {path}")


def load_a1_modes() -> dict[tuple[str, int], np.ndarray]:
    if not A1_MODES_PATH.is_file():
        raise FileNotFoundError(
            f"Missing A=1 posterior PCA modes: {A1_MODES_PATH}; run "
            "mcmc/analyze_pca_a1_posterior.py"
        )
    import csv

    modes = {
        (probe, n_events): np.full((N_W, N_MODES), np.nan)
        for probe in ("GW-only", "DESI+BBN+CMB+GW")
        for n_events in EVENT_COUNTS
    }
    with A1_MODES_PATH.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if float(row["prior_amplitude"]) != 1.0:
                raise ValueError(f"Non-A=1 row in {A1_MODES_PATH}")
            key = (row["probe"], int(row["n_gw"]))
            if key not in modes:
                raise ValueError(f"Unexpected A=1 PCA case in {A1_MODES_PATH}: {key}")
            modes[key][int(row["bin"]) - 1, int(row["pc"]) - 1] = float(row["e_i"])
    for key, values in modes.items():
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Incomplete A=1 PCA modes for {key}: {A1_MODES_PATH}")
    return modes


def geometry() -> np.ndarray:
    """Return the 29 PCA redshifts used by the supplied notebook.

    ``mode.ipynb`` defines ``z_CPCA`` from ``a_para_point[:-1]`` rather than
    from bin centres.  Keeping that convention is important when comparing
    the plotted eigenvectors point by point with the source calculation.
    """
    a_edges = np.linspace(1.0, 1.0 / (1.0 + Z_MAX), N_W + 1)
    return 1.0 / a_edges[:-1] - 1.0


def ordered_modes(directory: Path, stem: str, n_events: int) -> np.ndarray:
    """Return PC1--PC4 using the eigensystem convention in ``mode.ipynb``."""
    path = directory / f"{stem}_gw{n_events:04d}.csv"
    fisher = np.asarray(
        np.loadtxt(path, delimiter=",", skiprows=1), dtype=float
    )
    if fisher.shape != (N_W, N_W) or not np.all(np.isfinite(fisher)):
        raise ValueError(f"Invalid 29x29 Fisher matrix: {path}")
    fisher = 0.5 * (fisher + fisher.T)
    values, vectors = np.linalg.eig(fisher)
    imaginary_scale = max(
        float(np.max(np.abs(values.imag))),
        float(np.max(np.abs(vectors.imag))),
    )
    if imaginary_scale > 1.0e-10:
        raise ValueError(
            f"Fisher eigensystem has a non-negligible imaginary part "
            f"({imaginary_scale:.3e}): {path}"
        )
    values = values.real
    vectors = vectors.real
    order = np.argsort(np.abs(values))[::-1]
    vectors = vectors[:, order[:N_MODES]]
    norms = np.sqrt(np.sum(vectors**2, axis=0))
    if np.any(~np.isfinite(norms)) or np.any(norms == 0.0):
        raise ValueError(f"Invalid Fisher eigenvector norm: {path}")
    vectors /= norms
    for column in range(N_MODES):
        vector = vectors[:, column]
        if vector[np.argmax(np.abs(vector))] < 0.0:
            vectors[:, column] *= -1.0
    return vectors


def draw_modes(
    ax: plt.Axes,
    modes: np.ndarray,
    *,
    row: int,
    column: int,
) -> None:
    z_cpca = geometry()
    order = np.argsort(z_cpca)
    z_cpca = z_cpca[order]
    modes = np.asarray(modes, dtype=float)[order]
    z_plot = np.linspace(0.0, Z_MAX, 700)
    z_nodes = np.r_[z_cpca, Z_MAX]

    for pc, color in enumerate(MODE_COLORS):
        values = modes[:, pc]
        norm = float(np.sqrt(np.sum(values**2)))
        if not np.isfinite(norm) or norm == 0.0:
            raise ValueError(f"PC{pc + 1} has an invalid norm")
        values = values / norm
        # Display-only, shape-preserving interpolation.  The PCA values at
        # all 29 notebook nodes are unchanged; the final value is held to the
        # upper edge z=2.5 so that the plotted domain covers the final bin.
        smooth = PchipInterpolator(
            z_nodes,
            np.r_[values, values[-1]],
        )(z_plot)
        ax.plot(
            z_plot,
            smooth,
            color=color,
            lw=LINE_WIDTH,
            label=f"PC{pc + 1}",
        )

    ax.axhline(0.0, color="0.55", lw=ZERO_LINE_WIDTH, zorder=0)
    ax.set_xlim(0.0, Z_MAX)
    ax.set_ylim(*Y_LIM)
    ax.set_box_aspect(REFERENCE_BOX_ASPECT)
    ax.set_xticks([0.5, 1.0, 1.5, 2.0, 2.5])
    ax.set_yticks([-0.6, -0.3, 0.0, 0.3, 0.6])
    ax.set_xlabel(r"$z$" if row == 1 else "", fontsize=LABEL_SIZE)
    ax.set_ylabel(r"$e_i(z)$" if column == 0 else "", fontsize=LABEL_SIZE)
    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        top=True,
        right=True,
        labelsize=TICK_SIZE,
        labelbottom=row == 1,
        labelleft=column == 0,
        width=SPINE_WIDTH,
        length=5.0,
    )
    for spine in ax.spines.values():
        spine.set_color("#222222")
        spine.set_linewidth(SPINE_WIDTH)


def main() -> None:
    args = parse_args()
    a1_modes = None
    if args.a1:
        a1_modes = load_a1_modes()
        default_output = A1_OUTPUT
        row_labels = ("GW-only, A=1", "DESI+BBN+CMB+GW, A=1")
    else:
        validate_notebook_inputs()
        gw_only_directory = GW_PCA_TABLES
        joint_directory = DESI_BBN_CMB_GW_PCA_TABLES
        gw_only_stem = POSTERIOR_GW_ONLY_STEM
        joint_stem = POSTERIOR_JOINT_STEM
        default_output = MODE_NOTEBOOK_OUTPUT if args.mode_ipynb else DEFAULT_OUTPUT
        row_labels = ("GW-only", "DESI+BBN+CMB+GW")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
        }
    )
    fig, axes = plt.subplots(
        2,
        4,
        figsize=FIGSIZE,
        sharex=True,
        sharey=True,
        facecolor="white",
    )

    for column, n_events in enumerate(EVENT_COUNTS):
        axes[0, column].set_title(f"GW{n_events}", fontsize=TITLE_SIZE)
        gw_modes = (
            a1_modes[("GW-only", n_events)]
            if a1_modes is not None
            else ordered_modes(gw_only_directory, gw_only_stem, n_events)
        )
        joint_modes = (
            a1_modes[("DESI+BBN+CMB+GW", n_events)]
            if a1_modes is not None
            else ordered_modes(joint_directory, joint_stem, n_events)
        )
        draw_modes(
            axes[0, column],
            gw_modes,
            row=0,
            column=column,
        )
        draw_modes(
            axes[1, column],
            joint_modes,
            row=1,
            column=column,
        )

    for ax, row_label in zip((axes[0, 0], axes[1, 0]), row_labels):
        ax.text(
            0.965,
            0.93,
            row_label,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=ROW_LABEL_SIZE,
            fontweight="bold",
            zorder=10,
        )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.54, 0.985),
        ncol=N_MODES,
        frameon=False,
        fontsize=LEGEND_SIZE,
        columnspacing=1.4,
        handlelength=2.6,
        handletextpad=0.55,
    )
    fig.subplots_adjust(
        left=0.055,
        right=0.985,
        bottom=0.09,
        top=0.87,
        wspace=0.08,
        hspace=0.16,
    )

    output = (args.output or default_output).expanduser().with_suffix(".png").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=args.dpi, facecolor="white")
    plt.show(block=True)
    plt.close(fig)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
