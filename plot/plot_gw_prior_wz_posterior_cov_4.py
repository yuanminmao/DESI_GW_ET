#!/usr/bin/env python3
"""Plot the four GW-only + prior A=1 w(z) posteriors with injected w bins.

The layout intentionally matches
``wz_posterior_cov_gw0100_0500_1000_3000_with_injected_w``.  Its inputs are
the converged GW-only + modified-Horndeski-prior MCMC products, not the
DESI/BBN/CMB joint products.
"""

from __future__ import annotations

import argparse
import json
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

from plot_wz_pca_combined import (  # noqa: E402
    BEST_FIT_COLOR,
    INJECTED_COLOR,
    WZ_DPI,
    WZ_FIGSIZE,
    WZ_LABEL_SIZE,
    WZ_LEGEND_SIZE,
    WZ_LINE_WIDTH,
    WZ_MARKER_SIZE,
    WZ_PANEL_LAYOUT,
    WZ_REFERENCE_BOX_ASPECT,
    WZ_REFERENCE_LINE_WIDTH,
    WZ_SPINE_WIDTH,
    WZ_TICK_SIZE,
    WZ_TITLE_SIZE,
)
from plot_wz_posterior_cov_4 import (  # noqa: E402
    FIGURES,
    MPL_RC,
    TABLES,
    draw_wz_posterior_axis,
    sample_w_posterior,
)


EVENT_COUNTS = (100, 500, 1000, 3000)
DEFAULT_OUTPUT = (
    FIGURES
    / "gw_prior_only_wz_posterior_cov_gw0100_0500_1000_3000_with_injected_w.png"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples", type=int, default=8000)
    parser.add_argument("--dpi", type=int, default=WZ_DPI)
    args = parser.parse_args()
    if args.samples < 2:
        parser.error("--samples must be at least 2")
    if args.dpi < 50:
        parser.error("--dpi must be at least 50")
    return args


def product_paths(n_events: int) -> tuple[Path, Path, Path]:
    stem = f"gw{n_events:04d}_prior_wz_mcmc"
    return (
        TABLES / f"{stem}_reconstruction.csv",
        TABLES / f"{stem}_covariance.txt",
        TABLES / f"{stem}_summary.json",
    )


def validate_summary(n_events: int, path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing GW{n_events} A=1 summary: {path}")
    summary = json.loads(path.read_text(encoding="utf-8"))
    amplitude = float(summary.get("prior_amplitude", 1.0))
    diagnostic = summary.get("rank_folded_split_rhat", {})
    if amplitude != 1.0:
        raise ValueError(f"GW{n_events} summary uses prior amplitude A={amplitude:g}")
    if not isinstance(diagnostic, dict) or not diagnostic.get("all_pass"):
        raise ValueError(f"GW{n_events} A=1 chains did not pass rank/folded Rhat")
    if int(summary.get("catalog", {}).get("n_events", -1)) != n_events:  # type: ignore[union-attr]
        raise ValueError(f"GW{n_events} summary has inconsistent catalogue metadata")
    return summary


def table_curves(
    table_path: Path, z_bins: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(table_path, delimiter=",", names=True, encoding="utf-8")
    data = np.atleast_1d(data)
    names = set(data.dtype.names or ())
    required = {"z_center", "w_best_chain_sample", "w_injected"}
    if not required.issubset(names):
        raise ValueError(
            f"{table_path} must contain {sorted(required)}; found {sorted(names)}"
        )
    table_z = np.asarray(data["z_center"], dtype=float)
    best = np.asarray(data["w_best_chain_sample"], dtype=float)
    injected = np.asarray(data["w_injected"], dtype=float)
    if (
        table_z.shape != z_bins.shape
        or not np.allclose(table_z, z_bins, rtol=0.0, atol=1e-12)
        or not np.all(np.isfinite(best))
        or not np.all(np.isfinite(injected))
    ):
        raise ValueError(f"Invalid or misaligned reconstruction rows: {table_path}")
    return best, injected


def main() -> None:
    args = parse_args()
    plt.rcParams.update(MPL_RC)
    fig, axes = plt.subplots(
        2, 2, figsize=WZ_FIGSIZE, sharex=True, sharey=True, facecolor="white"
    )
    axes = axes.ravel()
    reference_z: np.ndarray | None = None
    reference_injected: np.ndarray | None = None

    for panel, (ax, n_events) in enumerate(zip(axes, EVENT_COUNTS)):
        table, covariance, summary = product_paths(n_events)
        validate_summary(n_events, summary)
        z_bins, w_mean, _, _, _, samples = sample_w_posterior(
            table_path=table,
            covariance_path=covariance,
            n_samples=args.samples,
            rng=np.random.default_rng(42000 + n_events),
        )
        best, injected = table_curves(table, z_bins)
        if reference_z is None:
            reference_z = z_bins.copy()
            reference_injected = injected.copy()
        elif not (
            np.allclose(z_bins, reference_z, rtol=0.0, atol=1e-12)
            and np.allclose(injected, reference_injected, rtol=0.0, atol=1e-12)
        ):
            raise ValueError("The four catalogues do not use identical injected w bins")

        ax.set_anchor((0.5, 0.5))
        draw_wz_posterior_axis(
            ax,
            z_bins,
            w_mean,
            samples,
            title=f"GW{n_events}",
            best_fit=best,
            show_ylabel=True,
            show_legend=False,
            z_plot_max=2.5,
        )
        ax.set_box_aspect(WZ_REFERENCE_BOX_ASPECT)
        ax.plot(
            z_bins,
            injected,
            "o",
            color=INJECTED_COLOR,
            ms=WZ_MARKER_SIZE,
            mec="white",
            mew=0.5 * 1.2 * 1.2,
            label=r"injected $w_i$",
            zorder=5,
        )
        for line in ax.lines:
            label = line.get_label()
            if label == r"best-fit $w_i$":
                line.set_visible(False)
            elif label == r"$\Lambda$CDM ($w=-1$)":
                line.set_linewidth(WZ_REFERENCE_LINE_WIDTH)
            else:
                line.set_linewidth(WZ_LINE_WIDTH)
            if label == r"best-fit $w(z)$":
                line.set_color(BEST_FIT_COLOR)
        ax.set_xticks([0.5, 1.0, 1.5, 2.0, 2.5])
        ax.set_xlabel(r"$z$", fontsize=WZ_LABEL_SIZE)
        ax.set_ylabel(r"$w(z)$", fontsize=WZ_LABEL_SIZE)
        ax.set_title(f"GW{n_events}", fontsize=WZ_TITLE_SIZE)
        ax.tick_params(
            axis="both",
            which="major",
            direction="in",
            top=True,
            right=True,
            labelsize=WZ_TICK_SIZE,
            labelbottom=True,
            labelleft=True,
            width=WZ_SPINE_WIDTH,
            length=4.0 * 1.2 * 1.2,
        )
        for spine in ax.spines.values():
            spine.set_linewidth(WZ_SPINE_WIDTH)

    handles, labels = axes[0].get_legend_handles_labels()
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
        bbox_to_anchor=(0.5, 0.97),
        ncol=len(labels),
        fontsize=WZ_LEGEND_SIZE,
        frameon=False,
        columnspacing=1.25,
        handlelength=2.6,
        handletextpad=0.55,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(**WZ_PANEL_LAYOUT)

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=args.dpi, facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    print(f"Saved: {output}")
    print(f"Saved: {output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
