#!/usr/bin/env python3
"""Plot the four w(z) posteriors and the first three data PCA modes."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator

from plot_wz_posterior_cov_4 import (
    DEFAULT_N_EVENTS,
    FIGURES,
    MPL_RC,
    TABLES,
    best_fit_bins,
    draw_wz_posterior_axis,
    load_summary,
    result_paths,
    sample_w_posterior,
)

Z_MAX = 2.5
MODE_COLORS = ("#D88200", "#FFCC00", "#7EC200")
BEST_FIT_COLOR = "#0072BD"
INJECTED_COLOR = "#D95319"

PCA_TABLE = TABLES / "desi_dr2_bbn_cmb_absolute_gw_pca_modes.csv"
INJECTION_TABLE = Path(__file__).resolve().parents[1] / "data" / "gw_fiducial_w_bins.txt"
WZ_OUTPUT = FIGURES / "wz_posterior_cov_gw0100_0500_1000_3000_with_injected_w.png"
PCA_OUTPUT = FIGURES / "wz_pca_modes_gw0100_0500_1000_3000.png"

# The posterior figure is sized at the final PRD two-column width.  Its
# panel geometry follows the w(z) row in the earlier combined w(z)+PCA
# figure, but the four panels are arranged as a compact 2x2 grid so that
# labels remain legible after journal typesetting.
WZ_FIGSIZE = (17.6 * 1.7, 10.0 * 1.7)
PCA_FIGSIZE = WZ_FIGSIZE
STYLE_SCALE = 1.2
WZ_TITLE_SIZE = 25 * STYLE_SCALE
WZ_LABEL_SIZE = 30 * STYLE_SCALE
WZ_TICK_SIZE = 25 * STYLE_SCALE
WZ_LEGEND_SIZE = 25 * STYLE_SCALE
WZ_LINE_WIDTH = 3 * STYLE_SCALE
WZ_REFERENCE_LINE_WIDTH = 4 * STYLE_SCALE
WZ_MARKER_SIZE = 10 * STYLE_SCALE
WZ_SPINE_WIDTH = 2 * STYLE_SCALE
# Measured from the w(z) panels in
# wz_posterior_cov_and_pca_modes_*_with_injected_w.png.
WZ_REFERENCE_BOX_ASPECT = 5.0 / 8.8
WZ_DPI = 600
PCA_TITLE_SIZE = WZ_TITLE_SIZE
PCA_LABEL_SIZE = WZ_LABEL_SIZE
PCA_TICK_SIZE = WZ_TICK_SIZE
PCA_LEGEND_SIZE = WZ_LEGEND_SIZE
PCA_LINE_WIDTH = WZ_LINE_WIDTH
PCA_ZERO_LINE_WIDTH = 0.8 * 1.2 * STYLE_SCALE
PCA_SPINE_WIDTH = WZ_SPINE_WIDTH

WZ_PANEL_LAYOUT = {
    "left": 0.045,
    "right": 0.995,
    "bottom": 0.075,
    "top": 0.89,
    "wspace": -0.3,
    "hspace": 0.5,
}
PCA_PANEL_LAYOUT = WZ_PANEL_LAYOUT.copy()


def load_pca_modes(path: Path) -> dict[int, dict[int, tuple[np.ndarray, np.ndarray]]]:
    """Load the first three PCA modes for each GW catalog."""
    modes: dict[int, dict[int, list[tuple[float, float]]]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            n_gw = int(row["n_gw"])
            pc = int(row["pc"])
            if pc <= 3:
                modes.setdefault(n_gw, {}).setdefault(pc, []).append(
                    (float(row["z_center"]), float(row["e_i"]))
                )

    return {
        n_gw: {
            pc: (
                np.asarray([z for z, _ in values]),
                np.asarray([value for _, value in values]),
            )
            for pc, values in pcs.items()
        }
        for n_gw, pcs in modes.items()
    }


def scale_axes(
    axes: np.ndarray,
    *,
    font_scale: float,
    outer_only: bool = True,
) -> None:
    """Scale panel text while optionally hiding interior shared labels."""
    for ax in axes.flat:
        if outer_only:
            ax.label_outer()
        ax.title.set_fontsize(ax.title.get_fontsize() * font_scale)
        ax.xaxis.label.set_fontsize(ax.xaxis.label.get_fontsize() * font_scale)
        ax.yaxis.label.set_fontsize(ax.yaxis.label.get_fontsize() * font_scale)
        for label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
            label.set_fontsize(label.get_fontsize() * font_scale)


def draw_pca_panel(
    ax,
    modes: dict[int, tuple[np.ndarray, np.ndarray]],
    n_events: int,
) -> None:
    """Draw one PCA panel using the preview-figure style."""
    z_plot = np.linspace(0.0, Z_MAX, 700)
    for pc, color in zip((1, 2, 3), MODE_COLORS):
        z_center, values = modes[pc]
        z_nodes = np.r_[0.0, z_center, Z_MAX]
        smooth = PchipInterpolator(
            z_nodes, np.r_[values[0], values, values[-1]],
        )(z_plot)
        ax.plot(z_plot, smooth, color=color, lw=PCA_LINE_WIDTH, label=f"PC{pc}")

    ax.axhline(0.0, color="0.45", lw=PCA_ZERO_LINE_WIDTH)
    ax.set_xlim(0.0, Z_MAX)
    ax.set_ylim(-0.6, 0.6)
    ax.set_yticks([-0.6, -0.3, 0.0, 0.3, 0.6])
    ax.set_xticks([0.5, 1.0, 1.5, 2.0, 2.5])
    ax.set_xlabel("")
    ax.set_title(f"GW{n_events}", fontsize=PCA_TITLE_SIZE)
    ax.set_ylabel(r"$e_i(z)$", fontsize=PCA_LABEL_SIZE)
    ax.tick_params(
        axis="both", which="major", direction="in", top=True, right=True,
        labelsize=PCA_TICK_SIZE, labelbottom=True, labelleft=True,
    )
    for spine in ax.spines.values():
        spine.set_linewidth(PCA_SPINE_WIDTH)


def main() -> None:
    plt.rcParams.update(MPL_RC)
    modes = load_pca_modes(PCA_TABLE)
    injected_rows = np.loadtxt(INJECTION_TABLE, comments="#", ndmin=2)[:29]
    injected_z = injected_rows[:, 3]
    injected_w = injected_rows[:, 4]

    wz_fig, wz_axes = plt.subplots(
        2, 2, figsize=WZ_FIGSIZE, sharex=True, sharey=True, facecolor="white",
    )
    pca_fig, pca_axes = plt.subplots(
        2, 2, figsize=PCA_FIGSIZE, sharex=True, sharey=True, facecolor="white",
    )
    wz_axes = wz_axes.ravel()
    pca_axes = pca_axes.ravel()
    # The reference aspect ratio leaves unused horizontal room inside each
    # subplot cell.  Anchor the two columns toward the center to remove that
    # visual gap while preserving the reference panel geometry.
    wz_axes[0].set_anchor((0.5, 0.5))
    wz_axes[2].set_anchor((0.5, 0.5))
    wz_axes[1].set_anchor((0.5, 0.5))
    wz_axes[3].set_anchor((0.5, 0.5))
    pca_axes[0].set_anchor((0.5, 0.5))
    pca_axes[2].set_anchor((0.5, 0.5))
    pca_axes[1].set_anchor((0.5, 0.5))
    pca_axes[3].set_anchor((0.5, 0.5))

    for column, n_events in enumerate(DEFAULT_N_EVENTS):
        table, covariance, _ = result_paths(n_events)
        load_summary(n_events)
        z_bins, w_mean, _, _, _, samples = sample_w_posterior(
            table_path=table,
            covariance_path=covariance,
        )

        wz_ax = wz_axes[column]
        draw_wz_posterior_axis(
            wz_ax,
            z_bins,
            w_mean,
            samples,
            title=f"GW{n_events}",
            best_fit=best_fit_bins(n_events, z_bins),
            show_ylabel=True,
            show_legend=False,
            z_plot_max=Z_MAX,
        )
        wz_ax.set_box_aspect(WZ_REFERENCE_BOX_ASPECT)
        wz_ax.plot(
            injected_z,
            injected_w,
            "o",
            color=INJECTED_COLOR,
            ms=WZ_MARKER_SIZE,
            mec="white",
            mew=0.5 * 1.2 * STYLE_SCALE,
            label=r"injected $w_i$",
            zorder=5,
        )
        for line in wz_ax.lines:
            label = line.get_label()
            if label == r"best-fit $w_i$":
                line.set_visible(False)
            elif label == r"$\Lambda$CDM ($w=-1$)":
                line.set_linewidth(WZ_REFERENCE_LINE_WIDTH)
            else:
                line.set_linewidth(WZ_LINE_WIDTH)
            if label == r"best-fit $w(z)$":
                line.set_color(BEST_FIT_COLOR)
        wz_ax.set_xticks([0.5, 1.0, 1.5, 2.0, 2.5])
        wz_ax.set_xlabel(r"$z$", fontsize=WZ_LABEL_SIZE)
        wz_ax.set_ylabel(r"$w(z)$", fontsize=WZ_LABEL_SIZE)
        wz_ax.set_title(f"GW{n_events}", fontsize=WZ_TITLE_SIZE)
        wz_ax.tick_params(
            axis="both",
            which="major",
            direction="in",
            top=True,
            right=True,
            labelsize=WZ_TICK_SIZE,
            labelbottom=True,
            labelleft=True,
            width=WZ_SPINE_WIDTH,
            length=4.0 * 1.2 * STYLE_SCALE,
        )
        for spine in wz_ax.spines.values():
            spine.set_linewidth(WZ_SPINE_WIDTH)

        draw_pca_panel(pca_axes[column], modes[n_events], n_events)

    for pca_ax in pca_axes:
        # Match the posterior figure's per-panel labeling convention and use
        # the PCA quantity itself rather than a generic "Redshift" label.
        pca_ax.set_box_aspect(WZ_REFERENCE_BOX_ASPECT)
        pca_ax.set_xlabel(r"$z$", fontsize=PCA_LABEL_SIZE)
        pca_ax.set_ylabel(r"$e_i(z)$", fontsize=PCA_LABEL_SIZE)
        pca_ax.tick_params(
            axis="both",
            which="major",
            direction="in",
            top=True,
            right=True,
            labelsize=PCA_TICK_SIZE,
            labelbottom=True,
            labelleft=True,
            width=PCA_SPINE_WIDTH,
            length=4.0 * 1.2 * STYLE_SCALE,
        )
        for spine in pca_ax.spines.values():
            spine.set_linewidth(PCA_SPINE_WIDTH)

    wz_handles, wz_labels = wz_axes[0].get_legend_handles_labels()
    wz_legend = [
        (handle, label.replace("C.L.", "credible interval"))
        for handle, label in zip(wz_handles, wz_labels)
        if label != r"best-fit $w_i$"
    ]
    wz_handles, wz_labels = map(list, zip(*wz_legend))
    wz_fig.legend(
        wz_handles,
        wz_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        ncol=len(wz_labels),
        fontsize=WZ_LEGEND_SIZE,
        frameon=False,
        columnspacing=1.25,
        handlelength=2.6,
        handletextpad=0.55,
        borderaxespad=0.0,
    )
    wz_fig.subplots_adjust(**WZ_PANEL_LAYOUT)

    pca_handles, pca_labels = pca_axes[0].get_legend_handles_labels()
    pca_fig.legend(
        pca_handles,
        pca_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=3,
        fontsize=PCA_LEGEND_SIZE,
        frameon=False,
        columnspacing=2.2,
    )
    pca_fig.subplots_adjust(**PCA_PANEL_LAYOUT)

    WZ_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    wz_fig.savefig(WZ_OUTPUT, dpi=WZ_DPI, facecolor="white")
    wz_fig.savefig(WZ_OUTPUT.with_suffix(".pdf"), facecolor="white")
    pca_fig.savefig(PCA_OUTPUT, dpi=WZ_DPI, facecolor="white")
    pca_fig.savefig(PCA_OUTPUT.with_suffix(".pdf"), facecolor="white")
    plt.close(wz_fig)
    plt.close(pca_fig)
    print(f"Saved: {WZ_OUTPUT.resolve()}")
    print(f"Saved: {WZ_OUTPUT.with_suffix('.pdf').resolve()}")
    print(f"Saved: {PCA_OUTPUT.resolve()}")
    print(f"Saved: {PCA_OUTPUT.with_suffix('.pdf').resolve()}")


if __name__ == "__main__":
    main()
