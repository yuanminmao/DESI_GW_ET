#!/usr/bin/env python3
"""Combine the GW-only and joint w(z) posterior reconstructions in 2x4 panels.

The top row contains GW absolute distances plus the A=1 Horndeski correlation
prior only.  The bottom row contains BBN+CMB(theta*)+DESI DR2+absolute GW with
the same correlation prior.  Columns correspond to 100, 500, 1000, and 3000
standard sirens.  All panels use identical axes and the same injected w bins.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from paths import (  # noqa: E402
    COMPUTE_BEST_FIT_W_TABLES,
    FIGURES,
    RECONSTRUCT_WZ_DESI_BBN_CMB_GW_TABLES,
    RECONSTRUCT_WZ_GW_TABLES,
)


warnings.filterwarnings(
    "ignore",
    message=r"(divide by zero|overflow|invalid value) encountered in matmul",
    category=RuntimeWarning,
)

DEFAULT_N_EVENTS = (100, 500, 1000, 3000)
N_Z = 800
N_SAMPLES = 8000
RNG_SEED = 42

MPL_RC = {
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
}

DATA_ZW = RECONSTRUCT_WZ_DESI_BBN_CMB_GW_TABLES / "desi_dr2_bbn_cmb_gw1000_wz.csv"
DATA_COV = RECONSTRUCT_WZ_DESI_BBN_CMB_GW_TABLES / "desi_dr2_bbn_cmb_gw1000_wz_covariance.csv"
BEST_FIT_TABLE = COMPUTE_BEST_FIT_W_TABLES / "desi_dr2_bbn_cmb_gw_bestfit_horndeski.csv"
BEST_FIT_SUMMARY = COMPUTE_BEST_FIT_W_TABLES / "desi_dr2_bbn_cmb_gw_bestfit_horndeski_summary.json"
GW100_BEST_FIT_TABLE = COMPUTE_BEST_FIT_W_TABLES / "desi_dr2_bbn_cmb_gw0100_bestfit_horndeski.csv"
GW100_BEST_FIT_SUMMARY = COMPUTE_BEST_FIT_W_TABLES / "desi_dr2_bbn_cmb_gw0100_bestfit_horndeski_summary.json"

BEST_FIT_COLOR = "#0072BD"
INJECTED_COLOR = "#D95319"
STYLE_SCALE = 1.2
WZ_LINE_WIDTH = 3 * STYLE_SCALE
WZ_REFERENCE_LINE_WIDTH = 4 * STYLE_SCALE
WZ_MARKER_SIZE = 10 * STYLE_SCALE
WZ_SPINE_WIDTH = 2 * STYLE_SCALE
WZ_REFERENCE_BOX_ASPECT = 5.0 / 8.8


def _load_cov(
    path: Path,
    n_bins: int,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Load, validate, and symmetrize a posterior covariance matrix."""
    if not path.is_file():
        raise FileNotFoundError(f"posterior covariance not found: {path}")
    cov = np.loadtxt(path, delimiter=",", skiprows=1).astype(float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(f"posterior covariance must be square; got {cov.shape} from {path}")
    if mask is not None:
        if cov.shape[0] == len(mask):
            cov = cov[np.ix_(mask, mask)]
        elif cov.shape[0] == int(mask.sum()) + 1:
            cov = cov[:-1, :-1]
    elif cov.shape[0] == n_bins + 1:
        cov = cov[:-1, :-1]
    if cov.shape != (n_bins, n_bins):
        raise ValueError(
            f"bin count mismatch: {n_bins} w bins vs {cov.shape[0]}x{cov.shape[1]} cov"
        )
    if not np.all(np.isfinite(cov)):
        raise ValueError(f"posterior covariance contains non-finite values: {path}")
    cov = 0.5 * (cov + cov.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    tolerance = max(float(eigenvalues[-1]) * 1e-10, 1e-12)
    if float(eigenvalues[0]) < -tolerance:
        raise ValueError(
            "posterior covariance is not positive semidefinite: "
            f"min eigenvalue={eigenvalues[0]:.6g}"
        )
    if eigenvalues[0] < 0.0:
        eigenvalues = np.maximum(eigenvalues, 0.0)
        cov = (eigenvectors * eigenvalues[None, :]) @ eigenvectors.T
        cov = 0.5 * (cov + cov.T)
    return cov


def load_z_w_bin_table(
    table_path: Path = DATA_ZW,
    covariance_path: Path = DATA_COV,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return z centers, posterior means, covariance, and bin edges."""
    table_path = Path(table_path).expanduser().resolve()
    covariance_path = Path(covariance_path).expanduser().resolve()
    if not table_path.is_file():
        raise FileNotFoundError(f"w(z) reconstruction table not found: {table_path}")

    if table_path.suffix.lower() == ".csv":
        data = np.genfromtxt(table_path, delimiter=",", names=True, encoding="utf-8")
        data = np.atleast_1d(data)
        names = set(data.dtype.names or ())
        required = {"z_min", "z_max", "z_center"}
        if not required.issubset(names):
            raise ValueError(
                f"CSV must contain {sorted(required)}; found {sorted(names)} in {table_path}"
            )
        mean_name = "w_mean" if "w_mean" in names else "w"
        if mean_name not in names:
            raise ValueError(f"CSV must contain w_mean or w: {table_path}")
        zp_lo = np.asarray(data["z_min"], dtype=float)
        zp_hi = np.asarray(data["z_max"], dtype=float)
        z = np.asarray(data["z_center"], dtype=float)
        w = np.asarray(data[mean_name], dtype=float)
        sigma = (
            np.asarray(data["w_sigma"], dtype=float)
            if "w_sigma" in names
            else np.ones_like(z)
        )
    else:
        data = np.loadtxt(table_path, comments="#")
        data = np.atleast_2d(data)
        if data.shape[1] < 5:
            raise ValueError(
                f"legacy w(z) table needs at least five columns; got {data.shape[1]}"
            )
        zp_lo = data[:, 1].astype(float)
        zp_hi = data[:, 2].astype(float)
        z = data[:, 3].astype(float)
        w = data[:, 4].astype(float)
        sigma = data[:, 5].astype(float) if data.shape[1] >= 6 else np.ones_like(z)

    if not all(np.all(np.isfinite(value)) for value in (zp_lo, zp_hi, z, w, sigma)):
        raise ValueError(f"w(z) table contains non-finite values: {table_path}")
    mask = sigma > 0
    if not np.any(mask):
        raise ValueError(f"w(z) table contains no bins with positive sigma: {table_path}")

    z, w = z[mask], w[mask]
    zp_lo, zp_hi = zp_lo[mask], zp_hi[mask]
    cov = _load_cov(covariance_path, len(z), mask)
    return z, w, cov, zp_lo, zp_hi


def sample_w_posterior(
    *,
    table_path: Path = DATA_ZW,
    covariance_path: Path = DATA_COV,
    n_samples: int = N_SAMPLES,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Draw w-bin samples from the posterior Gaussian approximation."""
    if int(n_samples) < 2:
        raise ValueError("n_samples must be at least two")
    z, w, cov, zp_lo, zp_hi = load_z_w_bin_table(table_path, covariance_path)
    rng = np.random.default_rng(RNG_SEED) if rng is None else rng
    samples = rng.multivariate_normal(
        mean=w, cov=cov, size=int(n_samples), check_valid="raise"
    )
    return z, w, cov, zp_lo, zp_hi, samples


def _percentile_curves(
    z_bins: np.ndarray,
    samples: np.ndarray,
    z_fine: np.ndarray,
) -> tuple[np.ndarray, ...]:
    quantiles = np.percentile(samples, [2.5, 16.0, 50.0, 84.0, 97.5], axis=0)
    return tuple(np.interp(z_fine, z_bins, values) for values in quantiles)


def draw_wz_posterior_axis(
    ax,
    z_bins: np.ndarray,
    w_mean: np.ndarray,
    samples: np.ndarray,
    *,
    title: str,
    best_fit: np.ndarray | None = None,
    show_ylabel: bool = True,
    show_legend: bool = True,
    z_plot_max: float = 2.5,
) -> None:
    """Draw one w(z) posterior panel."""
    z_plot = np.linspace(0.0, z_plot_max, N_Z)
    q2, q16, q50, q84, q97 = _percentile_curves(z_bins, samples, z_plot)
    if best_fit is not None:
        best_fit = np.asarray(best_fit, dtype=float)
        if best_fit.shape != w_mean.shape or not np.all(np.isfinite(best_fit)):
            raise ValueError("best_fit must contain one finite value for each w bin")
        center_line = np.interp(z_plot, z_bins, best_fit)
        center_points = best_fit
        line_label = r"best-fit $w(z)$"
        point_label = r"best-fit $w_i$"
    else:
        center_line = q50
        center_points = w_mean
        line_label = r"posterior median $w(z)$"
        point_label = r"posterior mean $w_i$"

    ax.set_facecolor("white")
    ax.fill_between(
        z_plot, q2, q97, color="#0072BD", alpha=0.25, lw=0, label=r"95% C.L."
    )
    ax.fill_between(
        z_plot, q16, q84, color="#0072BD", alpha=0.45, lw=0, label=r"68% C.L."
    )
    ax.plot(z_plot, center_line, color="#0072BD", lw=2, label=line_label)
    ax.plot(
        z_bins,
        center_points,
        "o",
        ms=6,
        mfc="#D95319",
        mec="white",
        mew=0.5,
        label=point_label,
        zorder=4,
    )
    ax.axhline(-1, color="black", ls="--", lw=2.2, label=r"$\Lambda$CDM ($w=-1$)")
    ax.set_xlabel(r"$z$", fontsize=22)
    if show_ylabel:
        ax.set_ylabel(r"$w(z)$", fontsize=22)
    ax.set_title(title, fontsize=18)
    ax.set_xlim(0.0, z_plot_max)
    ax.set_ylim(-4.0, 4.0)
    ax.set_yticks([-2, 0, 2, 4])
    ax.tick_params(labelsize=18, which="both")
    if show_legend:
        legend = ax.legend(
            fontsize=16,
            loc="upper right",
            frameon=True,
            facecolor="white",
            edgecolor="black",
            fancybox=False,
        )
        legend.get_frame().set_linewidth(1.0)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)


def best_fit_paths(n_events: int) -> tuple[Path, Path]:
    if n_events == 100:
        return GW100_BEST_FIT_TABLE, GW100_BEST_FIT_SUMMARY
    return BEST_FIT_TABLE, BEST_FIT_SUMMARY


def result_paths(n_events: int) -> tuple[Path, Path, Path]:
    stem = f"desi_dr2_bbn_cmb_gw{n_events:04d}_wz"
    return (
        RECONSTRUCT_WZ_DESI_BBN_CMB_GW_TABLES / f"{stem}.csv",
        RECONSTRUCT_WZ_DESI_BBN_CMB_GW_TABLES / f"{stem}_covariance.csv",
        RECONSTRUCT_WZ_DESI_BBN_CMB_GW_TABLES / f"{stem}_summary.json",
    )


def load_summary(n_events: int) -> dict:
    summary_path = result_paths(n_events)[2]
    if not summary_path.is_file():
        raise FileNotFoundError(f"Converged GW{n_events} summary is missing: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    diagnostic = summary.get("rank_folded_split_rhat", {})
    if not diagnostic.get("all_pass"):
        raise ValueError(f"GW{n_events} chains did not pass rank/folded Rhat < 1.01")
    return summary


def best_fit_bins(n_events: int, z_bins: np.ndarray) -> np.ndarray:
    """Read the locally optimized A=1 Horndeski fit."""
    best_fit_table, best_fit_summary = best_fit_paths(n_events)
    if not best_fit_table.is_file() or not best_fit_summary.is_file():
        raise FileNotFoundError(
            f"GW{n_events} optimized best-fit output is missing; "
            f"run analysis/posterior/compute_best_fit_w.py --n-events {n_events}"
        )
    payload = json.loads(best_fit_summary.read_text(encoding="utf-8"))
    if payload.get("A") != 1.0:
        raise ValueError("Optimized best-fit summary must use Horndeski prior amplitude A=1")
    cases = [case for case in payload.get("cases", []) if case.get("n_gw") == n_events]
    if len(cases) != 1 or not cases[0].get("optimization", {}).get("converged"):
        raise ValueError(f"GW{n_events} has no converged optimized best-fit result")
    with best_fit_table.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if int(row["n_gw"]) == n_events]
    expected_bins = list(range(1, len(z_bins) + 1))
    if len(rows) != len(z_bins) or [int(row["bin"]) for row in rows] != expected_bins:
        raise ValueError(f"GW{n_events} optimized best-fit table does not match the w bins")
    try:
        values = np.asarray([float(row["w_bestfit"]) for row in rows])
        z_centers = np.asarray([float(row["z_center"]) for row in rows])
        summary_values = np.asarray(
            [cases[0]["parameters"][f"w{i}"] for i in expected_bins], dtype=float
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"GW{n_events} optimized best-fit output is incomplete") from exc
    if not (
        np.all(np.isfinite(values))
        and np.allclose(z_centers, z_bins, atol=1e-12)
        and np.allclose(values, summary_values, rtol=1e-12, atol=1e-12)
    ):
        raise ValueError(f"GW{n_events} optimized best-fit CSV and JSON disagree")
    return values


def gw_only_paths(n_events: int) -> tuple[Path, Path, Path]:
    stem = f"gw{n_events:04d}_prior_wz_mcmc"
    return (
        RECONSTRUCT_WZ_GW_TABLES / f"{stem}_reconstruction.csv",
        RECONSTRUCT_WZ_GW_TABLES / f"{stem}_covariance.csv",
        RECONSTRUCT_WZ_GW_TABLES / f"{stem}_summary.json",
    )


def validate_gw_only_summary(n_events: int, path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing GW{n_events} A=1 summary: {path}")
    summary = json.loads(path.read_text(encoding="utf-8"))
    amplitude = float(summary.get("prior_amplitude", 1.0))
    diagnostic = summary.get("rank_folded_split_rhat", {})
    if amplitude != 1.0:
        raise ValueError(f"GW{n_events} summary uses prior amplitude A={amplitude:g}")
    if not isinstance(diagnostic, dict) or not diagnostic.get("all_pass"):
        raise ValueError(f"GW{n_events} A=1 chains did not pass rank/folded Rhat")
    catalog = summary.get("catalog", {})
    if not isinstance(catalog, dict) or int(catalog.get("n_events", -1)) != n_events:
        raise ValueError(f"GW{n_events} summary has inconsistent catalogue metadata")
    return summary


def gw_only_table_curves(
    table_path: Path,
    z_bins: np.ndarray,
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


DEFAULT_OUTPUT = (
    FIGURES
    / "wz_posterior_cov_gw_vs_desi_bbn_cmb_gw0100_0500_1000_3000_with_injected_w.png"
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

    output = args.output.expanduser().with_suffix(".png").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=args.dpi, facecolor="white")
    plt.show(block=True)
    plt.close(fig)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
