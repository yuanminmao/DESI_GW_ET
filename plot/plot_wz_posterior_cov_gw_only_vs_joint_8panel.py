#!/usr/bin/env python3
"""Combine the GW-only and joint w(z) posterior reconstructions in 2x4 panels.

The top row contains GW absolute distances plus the A=1 Horndeski correlation
prior only.  The bottom row contains BBN+CMB(theta*)+DESI DR2+absolute GW with
the same correlation prior.  Columns correspond to 100, 500, 1000, and 3000
standard sirens.  All panels use identical axes and the same injected w bins.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from plot_gw_prior_wz_posterior_cov_4 import (  # noqa: E402
    product_paths as gw_only_paths,
    table_curves as gw_only_table_curves,
    validate_summary as validate_gw_only_summary,
)
from plot_wz_pca_combined import (  # noqa: E402
    BEST_FIT_COLOR,
    INJECTED_COLOR,
    WZ_LINE_WIDTH,
    WZ_MARKER_SIZE,
    WZ_REFERENCE_BOX_ASPECT,
    WZ_REFERENCE_LINE_WIDTH,
    WZ_SPINE_WIDTH,
)
from plot_wz_posterior_cov_4 import (  # noqa: E402
    DEFAULT_N_EVENTS,
    FIGURES,
    MPL_RC,
    best_fit_bins,
    draw_wz_posterior_axis,
    load_summary,
    result_paths,
    sample_w_posterior,
)


DEFAULT_OUTPUT = (
    FIGURES
    / "wz_posterior_cov_gw_only_vs_desi_bbn_cmb_gw0100_0500_1000_3000_with_injected_w.png"
)
INJECTION_TABLE = ROOT / "data" / "gw_fiducial_w_bins.txt"
Z_MAX = 2.5
Y_LIM = (-4.0, 4.0)
FIGSIZE = (24.0, 9.0)
TEXT_SCALE = 1.5
TITLE_SIZE = 21
LABEL_SIZE = 21 * TEXT_SCALE
TICK_SIZE = 16 * TEXT_SCALE
LEGEND_SIZE = 17 * TEXT_SCALE
ROW_LABEL_SIZE = 19


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples", type=int, default=8000)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    if args.samples < 2:
        parser.error("--samples must be at least 2")
    if args.dpi < 50:
        parser.error("--dpi must be at least 50")
    return args


def injection_bins() -> tuple[np.ndarray, np.ndarray]:
    rows = np.loadtxt(INJECTION_TABLE, comments="#", ndmin=2)[:29]
    if rows.shape[0] != 29 or rows.shape[1] < 5:
        raise ValueError(f"Invalid injected-w table: {INJECTION_TABLE}")
    return rows[:, 3].astype(float), rows[:, 4].astype(float)


def restyle_axis(
    ax: plt.Axes,
    *,
    z_bins: np.ndarray,
    injected_w: np.ndarray,
    row: int,
    column: int,
) -> None:
    """Apply one consistent publication style after drawing a panel."""
    ax.set_box_aspect(WZ_REFERENCE_BOX_ASPECT)
    ax.plot(
        z_bins,
        injected_w,
        "o",
        color=INJECTED_COLOR,
        ms=0.58 * WZ_MARKER_SIZE,
        mec="white",
        mew=0.55,
        label=r"injected $w_i$",
        zorder=5,
    )
    for line in ax.lines:
        label = line.get_label()
        if label == r"best-fit $w_i$":
            line.set_visible(False)
        elif label == r"$\Lambda$CDM ($w=-1$)":
            line.set_linewidth(0.4 * WZ_REFERENCE_LINE_WIDTH)
        else:
            line.set_linewidth(0.8 * WZ_LINE_WIDTH)
        if label == r"best-fit $w(z)$":
            line.set_color(BEST_FIT_COLOR)

    ax.set_xlim(0.0, Z_MAX)
    ax.set_ylim(*Y_LIM)
    ax.set_xticks([0.5, 1.0, 1.5, 2.0, 2.5])
    ax.set_yticks([-4, -2, 0, 2, 4])
    ax.set_xlabel(r"$z$" if row == 1 else "", fontsize=LABEL_SIZE)
    ax.set_ylabel(r"$w(z)$" if column == 0 else "", fontsize=LABEL_SIZE)
    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        top=True,
        right=True,
        labelsize=TICK_SIZE,
        labelbottom=row == 1,
        labelleft=column == 0,
        width=0.75 * WZ_SPINE_WIDTH,
        length=5.0,
    )
    for spine in ax.spines.values():
        spine.set_linewidth(0.75 * WZ_SPINE_WIDTH)


def main() -> None:
    args = parse_args()
    plt.rcParams.update(MPL_RC)
    injected_z, injected_w = injection_bins()
    fig, axes = plt.subplots(
        2,
        4,
        figsize=FIGSIZE,
        sharex=True,
        sharey=True,
        facecolor="white",
    )

    for column, n_events in enumerate(DEFAULT_N_EVENTS):
        # Top row: GW-only + A=1 Horndeski prior.
        table, covariance, summary = gw_only_paths(n_events)
        validate_gw_only_summary(n_events, summary)
        z_bins, w_mean, _, _, _, samples = sample_w_posterior(
            table_path=table,
            covariance_path=covariance,
            n_samples=args.samples,
            rng=np.random.default_rng(42000 + n_events),
        )
        best_fit, table_injected = gw_only_table_curves(table, z_bins)
        if not (
            z_bins.shape == injected_z.shape
            and np.allclose(table_injected, injected_w, rtol=0.0, atol=1.0e-12)
        ):
            raise ValueError(f"GW{n_events}: GW-only injected bins do not match")
        top = axes[0, column]
        draw_wz_posterior_axis(
            top,
            z_bins,
            w_mean,
            samples,
            title=f"GW{n_events}",
            best_fit=best_fit,
            show_ylabel=column == 0,
            show_legend=False,
            z_plot_max=Z_MAX,
        )
        top.set_title(f"GW{n_events}", fontsize=TITLE_SIZE)
        restyle_axis(
            top,
            z_bins=z_bins,
            injected_w=injected_w,
            row=0,
            column=column,
        )

        # Bottom row: BBN+CMB+DESI DR2+GW + A=1 Horndeski prior.
        table, covariance, _ = result_paths(n_events)
        load_summary(n_events)
        joint_z, joint_mean, _, _, _, joint_samples = sample_w_posterior(
            table_path=table,
            covariance_path=covariance,
            n_samples=args.samples,
            rng=np.random.default_rng(52000 + n_events),
        )
        if joint_z.shape != injected_z.shape:
            raise ValueError(f"GW{n_events}: joint and injected redshift bins differ")
        bottom = axes[1, column]
        draw_wz_posterior_axis(
            bottom,
            joint_z,
            joint_mean,
            joint_samples,
            title="",
            best_fit=best_fit_bins(n_events, joint_z),
            show_ylabel=column == 0,
            show_legend=False,
            z_plot_max=Z_MAX,
        )
        bottom.set_title("")
        restyle_axis(
            bottom,
            z_bins=joint_z,
            injected_w=injected_w,
            row=1,
            column=column,
        )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    legend_items = [
        (handle, label.replace("C.L.", "credible interval"))
        for handle, label in zip(handles, labels)
        if label != r"best-fit $w_i$"
    ]
    handles, labels = map(list, zip(*legend_items))
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.54, 0.985),
        ncol=len(labels),
        fontsize=LEGEND_SIZE,
        frameon=False,
        columnspacing=1.4,
        handlelength=2.6,
        handletextpad=0.55,
    )
    for ax, row_label in (
        (axes[0, 0], "GW-only"),
        (axes[1, 0], "DESI+BBN+CMB+GW"),
    ):
        ax.text(
            0.965,
            0.93,
            row_label,
            transform=ax.transAxes,
            va="top",
            ha="right",
            fontsize=ROW_LABEL_SIZE,
            fontweight="bold",
            zorder=10,
        )
    fig.subplots_adjust(
        left=0.055,
        right=0.995,
        bottom=0.09,
        top=0.87,
        wspace=0.08,
        hspace=0.16,
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
