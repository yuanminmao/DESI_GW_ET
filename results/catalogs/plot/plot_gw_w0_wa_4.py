"""Compare GW-only and DESI+BBN+CMB+GW CPL constraints in four panels.

DETF-style FoM for CPL (w0, wa):
  FoM = 1 / (sigma_w0 * sigma_wa * sqrt(1 - rho^2))

Also reports the Mahalanobis separation of each posterior from LambdaCDM
(w0=-1, wa=0).

  python3 mcmc/gw_cpl_mcmc.py
  python3 mcmc/desi_bbn_cmb_gw_cpl_mcmc.py
  python3 plot/plot_gw_w0_wa_4.py
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import getdist
from getdist.mcsamples import Density1D
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from getdist import loadMCSamples
from scipy.ndimage import gaussian_filter, label

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paths import (
    FIGURES,
    PLOT_GW_W0_WA_4_TABLES,
    gw_cpl_chain_root,
    joint_cpl_chain_root,
    setup_sys_path,
)

setup_sys_path()

# Keep all artifacts named after this dedicated four-chain posterior plot.
OUT_JSON = PLOT_GW_W0_WA_4_TABLES / "gw_w0_wa_4.json"
OUT_FIG = FIGURES / "gw_w0_wa_4.png"
DEFAULT_N_EVENTS = (100, 500, 1000, 3000)
BY_N_COLORS = {
    100: "#9ecae1", 500: "#4292c6", 1000: "#0072BD", 3000: "#08306b",
}
SMOOTH_SIGMA_2D = 3
FINE_BINS_2D = 420
MIN_ISLAND_MASS = 0.005

W0_LCDM = -1.0
WA_LCDM = 0.0
DEFAULT_H0_OMEGA_M_YLIM = (0.01, 0.99)

# Default display limits (the sampling priors are slightly wider).
DEFAULT_XLIM = (-2.0, 3.0)
DEFAULT_YLIM = (-20.0, 3.0)

# Reference-style axis limits (LambdaCDM neighborhood)
PAPER_XLIM = (-1.45, 0.25)
PAPER_YLIM = (-2.85, 1.85)

# Background region colors (Quintessence / Phantom / Quintom)
C_QUINTESSENCE = "#8a8a8a"
C_PHANTOM = "#d8d8d8"
C_QUINTOM = "#B8C0E6"  # blue-purple Quintom regions
C_LCDM = "red"  # LambdaCDM marker

# Contour colors
C_ET = "#0072BD"
C_JOINT = "#D55E00"

# chi^2 thresholds for 2D Gaussian (w0, wa)
CHI2_LEVELS = {
    "1sigma": 2.30,
    "2sigma": 6.17,
    "3sigma": 11.62,
}

C_POSTERIOR = C_ET
REGION_LABEL_SIZE = 15  # legend (ET mock GW, LambdaCDM)
CORNER_LABEL_SIZE = 11  # Quintom / Quintessence / Phantom corner labels

MPL_RC = {
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "font.size": 13,
    "axes.labelsize": 15,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
}


@dataclass
class FoMResult:
    sigma_w0: float
    sigma_wa: float
    rho_w0_wa: float
    fom: float
    w0_mean: float
    wa_mean: float
    delta_w0_from_lcdm: float
    delta_wa_from_lcdm: float
    chi2_lcdm: float
    sigma_lcdm_equiv: float
    lcdm_inside_1sigma: bool
    lcdm_inside_2sigma: bool
    lcdm_inside_3sigma: bool
    n_samples: int
    gelman_rminus1: float


def load_samples(chain_root: Path):
    chain_root = chain_root.resolve()
    if not sorted(chain_root.parent.glob(f"{chain_root.name}_*.txt")):
        return None
    # Always read the current text chains. This avoids stale GetDist pickles
    # after an MCMC resume and keeps the script portable to restricted hosts.
    previous_cache = getdist.cache_dir
    getdist.cache_dir = str(Path(tempfile.gettempdir()) / "desi_gw_getdist_cache")
    try:
        samples = loadMCSamples(
            str(chain_root), no_cache=True, settings={"feedback": False}
        )
    finally:
        getdist.cache_dir = previous_cache
    for p in samples.getParamNames().names:
        if p.name == "w0":
            p.label = r"w_0"
        elif p.name == "wa":
            p.label = r"w_a"
    return samples


def load_w0_wa(chain_root: Path):
    samples = load_samples(chain_root)
    if samples is None:
        raise FileNotFoundError(f"no chains under {chain_root.parent}")
    names = [p.name for p in samples.getParamNames().names]
    idx = {n: i for i, n in enumerate(names)}
    for key in ("w0", "wa"):
        if key not in idx:
            raise ValueError(f"parameter {key!r} not in chains: {names}")
    w0 = samples.samples[:, idx["w0"]].astype(float)
    wa = samples.samples[:, idx["wa"]].astype(float)
    rminus1 = float(samples.getGelmanRubin())
    return samples, w0, wa, rminus1


def _limits_from_samples(sample_list: list, *, pad_frac: float = 0.08) -> tuple[tuple[float, float], tuple[float, float]]:
    w0_all = []
    wa_all = []
    for s in sample_list:
        names = [p.name for p in s.getParamNames().names]
        idx = {n: i for i, n in enumerate(names)}
        w0_all.append(s.samples[:, idx["w0"]])
        wa_all.append(s.samples[:, idx["wa"]])
    w0 = np.concatenate(w0_all)
    wa = np.concatenate(wa_all)
    # Use central 99.7% range so wide priors do not blow up the axes.
    w0_lo, w0_hi = np.percentile(w0, [0.15, 99.85])
    wa_lo, wa_hi = np.percentile(wa, [0.15, 99.85])
    w0_lo = min(w0_lo, W0_LCDM)
    w0_hi = max(w0_hi, W0_LCDM)
    wa_lo = min(wa_lo, WA_LCDM)
    wa_hi = max(wa_hi, WA_LCDM)
    dw = (w0_hi - w0_lo) * pad_frac
    dh = (wa_hi - wa_lo) * pad_frac
    return (w0_lo - dw, w0_hi + dw), (wa_lo - dh, wa_hi + dh)


def draw_de_region_background(
    ax,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    *,
    divider_lw: float = 0.9,
) -> None:
    """Shade Quintessence / Phantom / Quintom regions (w_a = -1 - w_0 boundary)."""
    w0_min, w0_max = xlim
    wa_min, wa_max = ylim
    w0 = np.linspace(w0_min, w0_max, 600)

    right = w0 > W0_LCDM
    left = w0 < W0_LCDM
    diag = -1.0 - w0

    ax.fill_between(
        w0[right], np.maximum(diag[right], wa_min), wa_max,
        color=C_QUINTESSENCE, alpha=0.90, linewidth=0, zorder=0,
    )
    ax.fill_between(
        w0[right], wa_min, np.minimum(diag[right], wa_max),
        color=C_QUINTOM, alpha=0.45, linewidth=0, zorder=0,
    )
    ax.fill_between(
        w0[left], wa_min, np.minimum(diag[left], wa_max),
        color=C_PHANTOM, alpha=0.78, linewidth=0, zorder=0,
    )
    ax.fill_between(
        w0[left], np.maximum(diag[left], wa_min), wa_max,
        color=C_QUINTOM, alpha=0.45, linewidth=0, zorder=0,
    )

    ax.axvline(W0_LCDM, ls="--", color="black", lw=1, zorder=1)
    ax.plot(w0, diag, ls="--", color="black", lw=1, zorder=1)


def draw_corner_region_labels(ax) -> None:
    """Region names at plot corners (axes coordinates); drawn last so they stay on top."""
    corner_labels = [
        ("Quintom A", (0.008, 0.94), "left", "top"),
        ("Quintessence", (0.992, 0.94), "right", "top"),
        ("Phantom", (0.008, 0.06), "left", "bottom"),
        ("Quintom B", (0.992, 0.06), "right", "bottom"),
    ]
    for text, (x, y), ha, va in corner_labels:
        ax.text(
            x, y, text,
            transform=ax.transAxes,
            color="black",
            fontsize=CORNER_LABEL_SIZE,
            fontweight="bold",
            ha=ha,
            va=va,
            zorder=110,
            clip_on=False,
        )


def _is_white_color(color: str) -> bool:
    return str(color).lower() in {"#ffffff", "#fff", "white"}


def _is_joint_color(color: str) -> bool:
    return str(color).upper() == C_JOINT.upper()


def _legend_label(name: str, res: FoMResult, *, legend_stat: str = "fom") -> str:
    if legend_stat == "name":
        return name
    if legend_stat == "lcdm_sigma":
        return rf"{name} ({res.sigma_lcdm_equiv:.1f}$\sigma$)"
    if legend_stat == "fom":
        return rf"{name} (FoM={res.fom:.1f})"
    raise ValueError(f"unknown legend_stat {legend_stat!r}; use 'name', 'fom' or 'lcdm_sigma'")


def _legend_line_handle(color: str, label: str) -> plt.Line2D:
    if _is_white_color(color):
        return plt.Line2D(
            [0], [0],
            marker="s",
            color="w",
            markerfacecolor="white",
            markeredgecolor="#666666",
            markeredgewidth=1.0,
            markersize=11,
            linestyle="None",
            label=label,
        )
    return plt.Line2D([0], [0], color=color, lw=3, label=label)


def _plot_sample_contours(
    ax,
    samples,
    *,
    color: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    zorder: int = 3,
    alpha_scale: float = 1.0,
) -> bool:
    density = samples.get2DDensity(
        "w0",
        "wa",
        lims=[list(xlim), list(ylim)],
        smooth_scale_1D=0.3,
    )
    l68, l95 = density.getContourLevels([0.68, 0.95])
    lo, hi = sorted([float(l68), float(l95)])
    pmax = float(density.P.max())
    if not np.isfinite(pmax) or pmax <= 0 or hi <= lo:
        return False

    inner_alpha = (0.70 if _is_joint_color(color) else 0.45) * alpha_scale
    outer_alpha = min(1.0, 1.0 * alpha_scale)
    line_alpha = min(1.0, 0.95 * alpha_scale)
    line_widths = (
        [1.5, 1.8]
        if _is_joint_color(color)
        else ([1.2, 1.4] if _is_white_color(color) else [1.0, 1.2])
    )

    ax.contourf(
        density.x,
        density.y,
        density.P,
        levels=[lo, pmax * 1.01],
        colors=[color],
        alpha=inner_alpha,
        zorder=zorder,
    )
    ax.contourf(
        density.x,
        density.y,
        density.P,
        levels=[hi, pmax * 1.01],
        colors=[color],
        alpha=outer_alpha,
        zorder=zorder + 1,
    )
    ax.contour(
        density.x,
        density.y,
        density.P,
        levels=[lo, hi],
        colors=[color],
        linewidths=line_widths,
        alpha=line_alpha,
        zorder=zorder + 2,
    )
    return True


def plot_w0_wa_fom(
    sample_sets: list[tuple[str, object, FoMResult, str]],
    *,
    out_fig: Path,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    lcdm_color: str = C_LCDM,
    suptitle: str | None = None,
    fill_alpha_scale: float = 1.0,
    legend_fontsize: float | None = None,
    divider_lw: float = 0.9,
    legend_stat: str = "fom",
    dpi: int = 200,
    show: bool = True,
) -> None:
    """Publication-style w0-wa plane with DE regions and nested contours."""
    plt.rcParams.update(MPL_RC)
    samples_only = [s for _, s, _, _ in sample_sets]
    if xlim is None or ylim is None:
        auto_x, auto_y = _limits_from_samples(samples_only)
        xlim = xlim or auto_x
        ylim = ylim or auto_y

    fig, ax = plt.subplots(figsize=(6.5, 6.5), facecolor="white")
    ax.set_facecolor("white")
    ax.set_box_aspect(1)
    draw_de_region_background(ax, xlim, ylim, divider_lw=divider_lw)

    legend_handles = []
    for i, (name, samples, res, color) in enumerate(sample_sets):
        z_base = 3 + 3 * i
        if _plot_sample_contours(
            ax, samples, color=color, xlim=xlim, ylim=ylim, zorder=z_base,
            alpha_scale=fill_alpha_scale,
        ):
            legend_handles.append(
                _legend_line_handle(color, _legend_label(name, res, legend_stat=legend_stat)),
            )

    if xlim[0] <= W0_LCDM <= xlim[1] and ylim[0] <= WA_LCDM <= ylim[1]:
        ax.scatter(
            [W0_LCDM],
            [WA_LCDM],
            marker="*",
            s=380,
            c=lcdm_color,
            edgecolors="white",
            linewidths=0.8,
            zorder=12,
        )
        legend_handles.append(
            plt.Line2D(
                [0], [0],
                marker="*",
                color="w",
                markerfacecolor=lcdm_color,
                markersize=20,
                linestyle="None",
                label=r"$\Lambda$CDM",
            )
        )

    ax.set_xlabel(r"$w_0$", fontsize=16)
    ax.set_ylabel(r"$w_a$", fontsize=16)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.tick_params(axis="both", which="both", labelsize=13)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    draw_corner_region_labels(ax)

    n_items = len(legend_handles)
    ncol = 2 if n_items <= 4 else 3
    leg_fs = REGION_LABEL_SIZE if legend_fontsize is None else legend_fontsize
    leg = fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=ncol,
        frameon=False,
        fontsize=leg_fs,
        handlelength=2.2,
        handletextpad=0.6,
        columnspacing=1.4,
        borderaxespad=0.0,
    )
    if leg is not None:
        leg.set_zorder(100)
        if leg.get_frame() is not None:
            leg.get_frame().set_zorder(100)

    if suptitle:
        fig.suptitle(suptitle, fontsize=12, y=1.02)

    fig.tight_layout()
    out_fig = out_fig.expanduser().with_suffix(".png").resolve()
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out_fig}")
    if show:
        plt.show(block=True)
    plt.close(fig)


def plot_h0_omegam_contours(
    sample_sets: list[tuple[str, object, FoMResult, str]],
    *,
    out_fig: Path,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    fiducial_h0: float = 67.4,
    fiducial_omegam: float = 0.315,
    dpi: int = 200,
    show: bool = True,
) -> None:
    """Overlay H0-Omega_m contours for multiple MCMC chains."""
    plt.rcParams.update(MPL_RC)
    samples_only = [s for _, s, _, _ in sample_sets]

    if xlim is None or ylim is None:
        h0_parts, om_parts = [], []
        for s in samples_only:
            names = [p.name for p in s.getParamNames().names]
            idx = {n: i for i, n in enumerate(names)}
            h0_parts.append(s.samples[:, idx["H0"]].astype(float))
            om_parts.append(s.samples[:, idx["omegam"]].astype(float))
        h0_vals = np.concatenate(h0_parts)
        pad_h = 0.08 * (float(h0_vals.max()) - float(h0_vals.min()) or 1.0)
        auto_x = (float(h0_vals.min()) - pad_h, float(h0_vals.max()) + pad_h)
        xlim = xlim or auto_x
        ylim = ylim or DEFAULT_H0_OMEGA_M_YLIM

    fig, ax = plt.subplots(figsize=(6.5, 6.5), facecolor="white")
    ax.set_facecolor("white")
    ax.set_box_aspect(1)

    legend_handles = []
    for i, (name, samples, res, color) in enumerate(sample_sets):
        density = samples.get2DDensity(
            "H0",
            "omegam",
            lims=[list(xlim), list(ylim)],
            smooth_scale_1D=0.3,
        )
        l68, l95 = density.getContourLevels([0.68, 0.95])
        lo, hi = sorted([float(l68), float(l95)])
        pmax = float(density.P.max())
        if not np.isfinite(pmax) or pmax <= 0 or hi <= lo:
            continue
        z_base = 3 + 3 * i
        ax.contourf(
            density.x, density.y, density.P,
            levels=[lo, pmax * 1.01], colors=[color], alpha=0.45, zorder=z_base,
        )
        ax.contourf(
            density.x, density.y, density.P,
            levels=[hi, pmax * 1.01], colors=[color], alpha=1.0, zorder=z_base + 1,
        )
        ax.contour(
            density.x, density.y, density.P,
            levels=[lo, hi], colors=[color],
            linewidths=[1.0, 1.2] if not _is_white_color(color) else [1.2, 1.4],
            alpha=0.95, zorder=z_base + 2,
        )
        legend_handles.append(_legend_line_handle(color, name))

    if xlim[0] <= fiducial_h0 <= xlim[1] and ylim[0] <= fiducial_omegam <= ylim[1]:
        ax.scatter(
            [fiducial_h0], [fiducial_omegam], marker="*", s=380, c=C_LCDM,
            edgecolors="white", linewidths=0.8, zorder=12,
        )
        legend_handles.append(
            plt.Line2D(
                [0], [0], marker="*", color="w", markerfacecolor=C_LCDM,
                markersize=20, linestyle="None", label=r"Fiducial $\Lambda$CDM",
            )
        )

    ax.set_xlabel(r"$H_0$ [km s$^{-1}$ Mpc$^{-1}$]", fontsize=16)
    ax.set_ylabel(r"$\Omega_m$", fontsize=16)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.tick_params(axis="both", which="both", labelsize=13)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    n_items = len(legend_handles)
    ncol = 2 if n_items <= 4 else 3
    fig.legend(
        handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 0.995),
        ncol=ncol, frameon=False, fontsize=REGION_LABEL_SIZE,
        handlelength=2.2, handletextpad=0.6, columnspacing=1.4, borderaxespad=0.0,
    )
    fig.tight_layout()
    out_fig = out_fig.expanduser().with_suffix(".png").resolve()
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out_fig}")
    if show:
        plt.show(block=True)
    plt.close(fig)


def plot_fom_improvement(
    cases: list[tuple[str, FoMResult, str]],
    *,
    baseline_label: str,
    out_fig: Path,
    dpi: int = 200,
    show: bool = True,
) -> None:
    """Bar chart of FoM improvement relative to a baseline case."""
    baseline_fom = None
    for label, res, _ in cases:
        if label == baseline_label:
            baseline_fom = res.fom
            break
    if baseline_fom is None or baseline_fom <= 0:
        raise ValueError(f"baseline {baseline_label!r} not found or FoM <= 0")

    labels = [label for label, _, _ in cases]
    ratios = [res.fom / baseline_fom for _, res, _ in cases]
    colors = [color for _, _, color in cases]

    plt.rcParams.update(MPL_RC)
    fig, ax = plt.subplots(figsize=(7.5, 5.0), facecolor="white")
    x = np.arange(len(labels))
    bars = ax.bar(x, ratios, color=colors, edgecolor="black", linewidth=0.8, alpha=0.9)
    ax.axhline(1.0, color="black", ls="--", lw=1.0, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=12)
    ax.set_ylabel(r"FoM / FoM$_{\rm baseline}$", fontsize=15)
    ax.tick_params(axis="y", labelsize=13)
    for bar, ratio in zip(bars, ratios):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
            f"{ratio:.2f}×", ha="center", va="bottom", fontsize=11,
        )
    fig.tight_layout()
    out_fig = out_fig.expanduser().with_suffix(".png").resolve()
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out_fig}")
    if show:
        plt.show(block=True)
    plt.close(fig)


def compute_fom(
    w0: np.ndarray,
    wa: np.ndarray,
    weights: np.ndarray | None = None,
) -> FoMResult:
    """Return weighted CPL summary statistics for a compressed MCMC chain."""
    if weights is None:
        weights = np.ones_like(w0, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if w0.shape != wa.shape or weights.shape != w0.shape:
        raise ValueError("w0, wa and weights must have the same shape")
    if np.any(~np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("MCMC weights must be finite and positive")
    w0_mean = float(np.average(w0, weights=weights))
    wa_mean = float(np.average(wa, weights=weights))
    cov = np.cov(np.vstack((w0, wa)), aweights=weights, ddof=1)
    sigma_w0 = float(np.sqrt(cov[0, 0]))
    sigma_wa = float(np.sqrt(cov[1, 1]))
    rho = float(cov[0, 1] / max(sigma_w0 * sigma_wa, 1e-30))
    denom = sigma_w0 * sigma_wa * np.sqrt(max(1.0 - rho**2, 1e-30))
    fom = float(1.0 / denom)

    delta = np.array([w0_mean - W0_LCDM, wa_mean - WA_LCDM], dtype=float)
    chi2_lcdm = float(delta @ np.linalg.inv(cov) @ delta)
    sigma_lcdm_equiv = float(np.sqrt(chi2_lcdm))

    return FoMResult(
        sigma_w0=sigma_w0,
        sigma_wa=sigma_wa,
        rho_w0_wa=rho,
        fom=fom,
        w0_mean=w0_mean,
        wa_mean=wa_mean,
        delta_w0_from_lcdm=float(delta[0]),
        delta_wa_from_lcdm=float(delta[1]),
        chi2_lcdm=chi2_lcdm,
        sigma_lcdm_equiv=sigma_lcdm_equiv,
        lcdm_inside_1sigma=chi2_lcdm <= CHI2_LEVELS["1sigma"],
        lcdm_inside_2sigma=chi2_lcdm <= CHI2_LEVELS["2sigma"],
        lcdm_inside_3sigma=chi2_lcdm <= CHI2_LEVELS["3sigma"],
        n_samples=int(round(float(np.sum(weights)))),
        gelman_rminus1=float("nan"),
    )


def interpret_lcdm_exclusion(res: FoMResult) -> str:
    if res.lcdm_inside_1sigma:
        return "LambdaCDM lies within the 68% CL contour: the data do not exclude a cosmological constant."
    if res.lcdm_inside_2sigma:
        return "LambdaCDM is outside 68% CL but inside 95% CL: weak evidence against LambdaCDM."
    if res.lcdm_inside_3sigma:
        return "LambdaCDM is excluded at ~95% CL (~2 sigma): moderate evidence for dynamical dark energy."
    return (
        f"LambdaCDM is excluded at >99.7% CL (~{res.sigma_lcdm_equiv:.1f} sigma): "
        "the posterior strongly favors dynamical dark energy over LambdaCDM."
    )


def format_report(res: FoMResult, *, title: str = "ET mock GW") -> str:
    lines = [
        f"{title} — dark-energy Figure of Merit (CPL w0, wa)",
        "=" * 62,
        f"Samples: {res.n_samples}",
        "",
        "Marginalized constraints (Gaussian approx. from MCMC):",
        f"  w0 = {res.w0_mean:+.4f} +/- {res.sigma_w0:.4f}",
        f"  wa = {res.wa_mean:+.4f} +/- {res.sigma_wa:.4f}",
        f"  corr(w0, wa) = {res.rho_w0_wa:+.4f}",
        "",
        f"DETF FoM = 1 / (sigma_w0 sigma_wa sqrt(1-rho^2)) = {res.fom:.2f}",
        "  (larger FoM => tighter joint w0-wa constraint)",
        "",
        f"LambdaCDM reference: w0 = {W0_LCDM:.1f}, wa = {WA_LCDM:.1f}",
        f"  offset from mean: Delta w0 = {res.delta_w0_from_lcdm:+.4f}, Delta wa = {res.delta_wa_from_lcdm:+.4f}",
        f"  chi^2(LambdaCDM | posterior) = {res.chi2_lcdm:.2f}  (~{res.sigma_lcdm_equiv:.2f} sigma)",
        f"  inside 68% CL: {res.lcdm_inside_1sigma}",
        f"  inside 95% CL: {res.lcdm_inside_2sigma}",
        f"  inside 99.7% CL: {res.lcdm_inside_3sigma}",
        "",
        "Interpretation:",
        f"  {interpret_lcdm_exclusion(res)}",
    ]
    if np.isfinite(res.gelman_rminus1):
        lines.extend(["", f"Gelman-Rubin R-1 (max): {res.gelman_rminus1:.4f}"])
    return "\n".join(lines)


def _sample_fom(samples) -> FoMResult:
    names = [p.name for p in samples.getParamNames().names]
    idx = {n: i for i, n in enumerate(names)}
    return compute_fom(
        samples.samples[:, idx["w0"]].astype(float),
        samples.samples[:, idx["wa"]].astype(float),
        samples.weights,
    )


def _chain_root_for_n(n_events: int) -> Path:
    return gw_cpl_chain_root(n_events)


def _joint_chain_root_for_n(n_events: int) -> Path:
    return joint_cpl_chain_root(n_events)


def _format_by_n_table(cases: list[tuple[int, str, FoMResult]]) -> str:
    lines = [
        "CPL constraints by data combination and N_GW — summary",
        "=" * 101,
        f"{'N_GW':>8} {'Data':<25} {'sigma(w0)':>12} {'sigma(wa)':>12} "
        f"{'rho':>10} {'FoM':>10} {'R-1':>10} {'LCDM(sigma)':>14}",
        "-" * 101,
    ]
    for n_events, dataset, res in cases:
        lines.append(
            f"{n_events:8d} {dataset:<25} {res.sigma_w0:12.4f} {res.sigma_wa:12.4f} "
            f"{res.rho_w0_wa:10.4f} {res.fom:10.2f} {res.gelman_rminus1:10.4f} "
            f"{res.sigma_lcdm_equiv:14.3f}"
        )
    return "\n".join(lines)


def _remove_low_mass_islands(density: np.ndarray, contour_level: float) -> tuple[np.ndarray, int]:
    """Hide negligible disconnected contour islands in the visualization only."""
    components, n_components = label(density >= contour_level)
    total_mass = float(np.sum(density))
    filtered = density.copy()
    removed = 0
    if total_mass <= 0.0:
        return filtered, removed
    for component in range(1, n_components + 1):
        region = components == component
        if float(np.sum(density[region])) / total_mass < MIN_ISLAND_MASS:
            filtered[region] = 0.0
            removed += 1
    return filtered, removed


def _smoothed_w0_wa_density(
    samples,
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    smooth_sigma_2d: float,
    fine_bins_2d: int,
) -> tuple[object, np.ndarray, float, float, int] | None:
    density = samples.get2DDensity(
        "w0", "wa", lims=[list(xlim), list(ylim)], smooth_scale_1D=0.75,
        fine_bins_2D=int(fine_bins_2d),
    )
    probability = np.maximum(
        gaussian_filter(density.P, sigma=float(smooth_sigma_2d)), 0.0
    )
    peak = float(probability.max())
    if not np.isfinite(peak) or peak <= 0.0:
        return None
    probability /= peak
    original = density.P
    density.P = probability
    level68, level95 = density.getContourLevels([0.68, 0.95])
    density.P = original
    low, high = sorted((float(level68), float(level95)))
    probability, removed = _remove_low_mass_islands(probability, low)
    if removed:
        density.P = probability
        level68, level95 = density.getContourLevels([0.68, 0.95])
        density.P = original
        low, high = sorted((float(level68), float(level95)))
    if not (np.isfinite(high) and high > low):
        return None
    return density, probability, low, high, removed


def plot_w0_wa_fom_grid_by_n(
    sample_sets: list[tuple[int, object, object]], *, out_fig: Path,
    xlim: tuple[float, float], ylim: tuple[float, float], dpi: int, show: bool,
    smooth_sigma_2d: float, fine_bins_2d: int,
) -> None:
    """Overlay GW-only and joint 68%/95% w0-wa contours for each N_GW."""
    plt.rcParams.update(MPL_RC)
    ncols = 2
    nrows = int(np.ceil(len(sample_sets) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10.4, 4.6 * nrows), facecolor="white")
    axes_flat = np.atleast_1d(axes).ravel()
    contour_specs = (
        ("GW", C_ET, 0.25, 0.62, 3),
        ("DESI + BBN + CMB + GW", C_JOINT, 0.28, 0.72, 6),
    )
    for ax, (n_events, gw_samples, joint_samples) in zip(axes_flat, sample_sets):
        ax.set_facecolor("white")
        draw_de_region_background(ax, xlim, ylim, divider_lw=0.9)
        for samples, (dataset, color, outer_alpha, inner_alpha, zorder) in zip(
            (gw_samples, joint_samples), contour_specs
        ):
            smoothed = _smoothed_w0_wa_density(
                samples,
                xlim=xlim,
                ylim=ylim,
                smooth_sigma_2d=smooth_sigma_2d,
                fine_bins_2d=fine_bins_2d,
            )
            if smoothed is None:
                print(f"N_GW={n_events}, {dataset}: no finite contour density in plot limits.")
                continue
            density, probability, low, high, removed = smoothed
            if removed:
                print(
                    f"N_GW={n_events}, {dataset}: suppressed {removed} "
                    "low-mass isolated contour island(s)."
                )
            peak = float(probability.max())
            ax.contourf(
                density.x, density.y, probability,
                levels=[low, peak * 1.01], colors=[color], alpha=outer_alpha,
                zorder=zorder,
            )
            ax.contourf(
                density.x, density.y, probability,
                levels=[high, peak * 1.01], colors=[color], alpha=inner_alpha,
                zorder=zorder + 1,
            )
            ax.contour(
                density.x, density.y, probability,
                levels=[low, high], colors=[color], linewidths=[1.25, 1.55],
                zorder=zorder + 2,
            )
        ax.scatter(
            [-1.0], [0.0], marker="*", s=170, c=C_LCDM,
            edgecolors="white", linewidths=0.7, zorder=10,
        )
        ax.set(
            xlim=xlim, ylim=ylim, xlabel=r"$w_0$", ylabel=r"$w_a$",
            title=rf"$N_{{\rm GW}}={n_events}$",
        )
        ax.tick_params(axis="both", which="both", labelsize=11)
    for ax in axes_flat[len(sample_sets):]:
        ax.axis("off")

    legend_handles = [
        plt.Line2D([0], [0], color=C_ET, lw=3, label="GW"),
        plt.Line2D([0], [0], color=C_JOINT, lw=3,
                   label="DESI + BBN + CMB + GW"),
        plt.Line2D(
            [0], [0], marker="*", color="w", markerfacecolor=C_LCDM,
            markersize=15, linestyle="None", label=r"$\Lambda$CDM",
        ),
    ]
    fig.legend(
        handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.995),
        ncol=3, frameon=False, fontsize=13, handlelength=2.2,
        columnspacing=1.5,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.955))
    out_fig = out_fig.expanduser().with_suffix(".png").resolve()
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=int(dpi), bbox_inches="tight", facecolor="white")
    print(f"Saved: {out_fig}")
    if show:
        plt.show(block=True)
    plt.close(fig)


def main_by_n(args: argparse.Namespace) -> None:
    n_list = sorted(set(int(n) for n in args.n_events if int(n) > 0))
    if not n_list:
        raise ValueError("--n-events must contain at least one positive integer")
    grid_sets: list[tuple[int, object, object]] = []
    rows: list[tuple[int, str, FoMResult]] = []
    reports: list[str] = []
    for n_events in n_list:
        gw_root = _chain_root_for_n(n_events).resolve()
        joint_root = _joint_chain_root_for_n(n_events).resolve()
        if not sorted(gw_root.parent.glob(f"{gw_root.name}_*.txt")):
            raise FileNotFoundError(
                f"no GW-only chains for N_GW={n_events}: "
                f"{gw_root.parent / (gw_root.name + '_*.txt')}"
            )
        if not sorted(joint_root.parent.glob(f"{joint_root.name}_*.txt")):
            raise FileNotFoundError(
                f"no joint chains for N_GW={n_events}: "
                f"{joint_root.parent / (joint_root.name + '_*.txt')}\n"
                "Run: python3 mcmc/desi_bbn_cmb_gw_cpl_mcmc.py"
            )

        gw_samples, gw_w0, gw_wa, gw_rminus1 = load_w0_wa(gw_root)
        joint_samples, joint_w0, joint_wa, joint_rminus1 = load_w0_wa(joint_root)
        gw_result = compute_fom(gw_w0, gw_wa, gw_samples.weights)
        joint_result = compute_fom(joint_w0, joint_wa, joint_samples.weights)
        gw_result.gelman_rminus1 = gw_rminus1
        joint_result.gelman_rminus1 = joint_rminus1

        grid_sets.append((n_events, gw_samples, joint_samples))
        rows.extend((
            (n_events, "GW", gw_result),
            (n_events, "DESI+BBN+CMB+GW", joint_result),
        ))
        reports.extend((
            f"N_GW={n_events}; GW only\n"
            + format_report(gw_result, title="ET mock GW"),
            f"N_GW={n_events}; DESI+BBN+CMB+GW\n"
            + format_report(joint_result, title="DESI + BBN + CMB + ET mock GW"),
        ))
        print(
            f"N_GW={n_events}: GW R-1={gw_rminus1:.4f}, "
            f"joint R-1={joint_rminus1:.4f}; "
            f"FoM {gw_result.fom:.2f} -> {joint_result.fom:.2f}"
        )
    report = _format_by_n_table(rows) + "\n\n" + "\n\n".join(reports) + "\n"
    print("\n" + report)
    args.out_json.resolve().parent.mkdir(parents=True, exist_ok=True)
    json_cases = []
    for n_events in n_list:
        by_dataset = {
            dataset: asdict(result)
            for count, dataset, result in rows
            if count == n_events
        }
        json_cases.append({"N_GW": n_events, "constraints": by_dataset})
    args.out_json.write_text(
        json.dumps({"cases": json_cases}, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote: {args.out_json.resolve()}")
    xlim, ylim = (PAPER_XLIM, PAPER_YLIM) if args.paper_limits else (DEFAULT_XLIM, DEFAULT_YLIM)
    plot_w0_wa_fom_grid_by_n(grid_sets, out_fig=args.out_fig.resolve(), xlim=xlim, ylim=ylim,
                             dpi=args.dpi, show=True, smooth_sigma_2d=args.smooth_sigma_2d,
                             fine_bins_2d=args.fine_bins_2d)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare GW-only and DESI+BBN+CMB+GW CPL posteriors.",
    )
    ap.add_argument("--n-events", type=int, nargs="+", default=list(DEFAULT_N_EVENTS),
                    help="N_GW values to plot (default: 100 500 1000 3000).")
    ap.add_argument("--out-json", type=Path, default=OUT_JSON)
    ap.add_argument(
        "--paper-limits",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reference-style axis limits near LambdaCDM.",
    )
    ap.add_argument("--out-fig", type=Path, default=OUT_FIG)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--smooth-sigma-2d", type=float, default=SMOOTH_SIGMA_2D,
                    help="Extra 2D smoothing used for both contour sets.")
    ap.add_argument("--fine-bins-2d", type=int, default=FINE_BINS_2D,
                    help="2D density-grid resolution used for the contour grid.")
    args = ap.parse_args()

    main_by_n(args)


if __name__ == "__main__":
    main()
