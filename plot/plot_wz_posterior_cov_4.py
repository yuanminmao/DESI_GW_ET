"""Plot a reconstructed w(z) posterior from its mean and covariance.

By default this plots the BBN + CMB + DESI DR2 + GW100/500/1000/3000
best-fit single figures and four-panel comparison. Use --n-events for the
older joint absolute-GW chain products. Legacy whitespace tables with columns
``bin, z_min, z_max, z_center, w, sigma`` are also accepted.  Rows with
non-positive sigma are excluded (for example a fixed anchor bin at z > 2.5).

Shared helpers (H0, d_L from binned w) are used by plot_cc_bao_data.py,
plot_dl_desi_lcdm_frac.py, and catalog/dmu_desi_lcdm.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d

# NumPy 2.0.2 with Apple's Accelerate backend can emit false-positive matmul
# warnings for finite 29-dimensional Gaussian draws.  The covariance is
# explicitly checked for finiteness and positive semidefiniteness below.
warnings.filterwarnings(
    "ignore",
    message=r"(divide by zero|overflow|invalid value) encountered in matmul",
    category=RuntimeWarning,
)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paths import setup_sys_path, FIGURES, TABLES

setup_sys_path()

from gw_cosmology import C_KM_S

DEFAULT_N_EVENTS = (100, 500, 1000, 3000)
ABSOLUTE_GW_CHAIN_DIRS = {
    100: ROOT / "results/chains/COBAYA_DESI_DR2_CMB_ABSOLUTE_GW_N0100_0300_1000_3000_SEED0042_20260916/GW0100",
    500: ROOT / "results/chains/COBAYA_DESI_DR2_CMB_ABSOLUTE_GW_NEW_FID_20260915/GW0500",
    1000: ROOT / "results/chains/COBAYA_DESI_DR2_CMB_ABSOLUTE_GW_NEW_FID_20260915/GW1000",
    3000: ROOT / "results/chains/COBAYA_DESI_DR2_CMB_ABSOLUTE_GW_N3000_SEED0042_20260916/GW3000",
}


def absolute_gw_paths(n_events: int) -> tuple[Path, Path, Path]:
    """Return the joint absolute-GW reconstruction, covariance, and figure paths."""
    n_events = int(n_events)
    if n_events not in ABSOLUTE_GW_CHAIN_DIRS:
        raise ValueError(f"unsupported GW event count: {n_events}; choose {DEFAULT_N_EVENTS}")
    stem = f"cobaya_joint_absolute_gw{n_events:04d}_wz"
    chain_dir = ABSOLUTE_GW_CHAIN_DIRS[n_events]
    return (
        chain_dir / f"{stem}_reconstruction.csv",
        chain_dir / f"{stem}_covariance.txt",
        FIGURES / f"wz_posterior_cov_absolute_gw{n_events:04d}.png",
    )


DATA_ZW = TABLES / "desi_dr2_bbn_cmb_absolute_gw1000_wz.csv"
DATA_COV = TABLES / "desi_dr2_bbn_cmb_absolute_gw1000_wz_covariance.txt"
OUT_FIG = FIGURES / "wz_posterior_cov_bbn_cmb_desi_absolute_gw1000.png"

H0_PLANCK = 67.4  # km/s/Mpc, Planck 2018
OMEGA_M_PLANCK = 0.315

Z_PLOT_MAX = 2.5
COMPARISON_Z_PLOT_MAX = 1.5
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


FOUR_GW_OUTPUT = (
    FIGURES / "wz_posterior_cov_bbn_cmb_desi_absolute_gw0100_0500_1000_3000_comparison.png"
)
BBN_GW_CHAIN_ROOT = ROOT / "results/chains/COBAYA_DESI_DR2_BBN_CMB_ABSOLUTE_GW_wz"
BEST_FIT_TABLE = TABLES / "desi_dr2_bbn_cmb_absolute_gw_bestfit_horndeski.csv"
BEST_FIT_SUMMARY = TABLES / "desi_dr2_bbn_cmb_absolute_gw_bestfit_horndeski_summary.json"
GW100_BEST_FIT_TABLE = TABLES / "desi_dr2_bbn_cmb_absolute_gw0100_bestfit_horndeski.csv"
GW100_BEST_FIT_SUMMARY = TABLES / "desi_dr2_bbn_cmb_absolute_gw0100_bestfit_horndeski_summary.json"


def relative_gw_paths(n_events: int) -> tuple[Path, Path, Path]:
    """Return the standard reconstruction, covariance, and figure paths."""
    if int(n_events) < 1:
        raise ValueError("the GW event count must be positive")
    tag = f"{int(n_events):04d}"
    stem = f"desi_dr2_cmb_relative_gw{tag}_wz"
    return (
        TABLES / f"{stem}_reconstruction.csv",
        TABLES / f"{stem}_covariance.txt",
        FIGURES / f"wz_posterior_cov_relative_gw{tag}.png",
    )


def _load_cov(
    path: Path,
    n_bins: int,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"posterior covariance not found: {path}")
    cov = np.loadtxt(path, comments="#").astype(float)
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
            f"posterior covariance is not positive semidefinite: min eigenvalue={eigenvalues[0]:.6g}"
        )
    # Remove tiny negative eigenvalues caused solely by text-rounding.
    if eigenvalues[0] < 0.0:
        eigenvalues = np.maximum(eigenvalues, 0.0)
        cov = (eigenvectors * eigenvalues[None, :]) @ eigenvectors.T
        cov = 0.5 * (cov + cov.T)
    return cov


def load_z_w_bin_table(
    table_path: Path = DATA_ZW,
    covariance_path: Path = DATA_COV,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (z_center, w_mean, cov, z_bin_lo, z_bin_hi) for bins with sigma > 0."""
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


def rho_de_ratio_binned(
    z: np.ndarray,
    zp_lo: np.ndarray,
    zp_hi: np.ndarray,
    w_bins: np.ndarray,
) -> np.ndarray:
    """rho_de(z)/rho_de,0 for piecewise-constant w(z) bins."""
    seg = (1.0 + w_bins) * np.log((1 + zp_hi) / (1 + zp_lo))
    cum = np.cumsum(seg)

    z = np.atleast_1d(z).astype(float)
    rho = np.zeros_like(z)
    for k, zz in enumerate(z):
        if zz < zp_lo[0]:
            integral = (1.0 + w_bins[0]) * np.log(1 + zz)
        else:
            i = min(int(np.searchsorted(zp_hi, zz)), len(w_bins) - 1)
            integral = cum[i - 1] if i > 0 else 0.0
            integral += (1.0 + w_bins[i]) * np.log((1 + zz) / (1 + zp_lo[i]))
        rho[k] = np.exp(3.0 * integral)
    return rho


def h_z_lcdm(
    z: np.ndarray,
    *,
    h0: float = H0_PLANCK,
    omega_m: float = OMEGA_M_PLANCK,
) -> np.ndarray:
    """Flat LambdaCDM H(z) [km/s/Mpc] with Planck background."""
    omega_de = 1.0 - omega_m
    z = np.asarray(z, dtype=float)
    return h0 * np.sqrt(np.maximum(omega_m * (1.0 + z) ** 3 + omega_de, 1e-30))


def dl_mpc_from_w_bins(
    z: np.ndarray,
    zp_lo: np.ndarray,
    zp_hi: np.ndarray,
    w_bins: np.ndarray,
    *,
    h0: float = H0_PLANCK,
    omega_m: float = OMEGA_M_PLANCK,
    n_grid: int = 4000,
) -> np.ndarray:
    """Luminosity distance d_L(z) [Mpc] for piecewise-constant w(z) (Planck H0)."""
    z = np.asarray(z, dtype=float)
    omega_de = 1.0 - omega_m
    z_hi = max(float(z.max()), float(np.max(zp_hi))) * 1.05
    z_grid = np.linspace(0.0, z_hi, int(n_grid))
    ez = np.sqrt(
        np.maximum(
            omega_m * (1.0 + z_grid) ** 3
            + omega_de * rho_de_ratio_binned(z_grid, zp_lo, zp_hi, w_bins),
            1e-30,
        )
    )
    chi = cumulative_trapezoid(1.0 / ez, z_grid, initial=0.0)
    d_l = (1.0 + z_grid) * (C_KM_S / h0) * chi
    return interp1d(
        z_grid,
        d_l,
        kind="cubic",
        bounds_error=False,
        fill_value=(0.0, float(d_l[-1])),
    )(z)


def _percentile_curves(
    z_bins: np.ndarray,
    samples: np.ndarray,
    z_fine: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Percentiles per bin, linearly interpolated onto z_fine."""
    qs = np.percentile(samples, [2.5, 16.0, 50.0, 84.0, 97.5], axis=0)
    return tuple(np.interp(z_fine, z_bins, q) for q in qs)


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
    z_plot_max: float = Z_PLOT_MAX,
) -> None:
    """Draw one panel with the same layout, colors, and labels as the single figure."""
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
    ax.fill_between(z_plot, q2, q97, color="#0072BD", alpha=0.25, lw=0, label=r"95% C.L.")
    ax.fill_between(z_plot, q16, q84, color="#0072BD", alpha=0.45, lw=0, label=r"68% C.L.")
    ax.plot(z_plot, center_line, color="#0072BD", lw=2, label=line_label)
    ax.plot(z_bins, center_points, "o", ms=6, mfc="#D95319", mec="white", mew=0.5,
            label=point_label, zorder=4)
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
        leg = ax.legend(fontsize=16, loc="upper right", frameon=True,
                        facecolor="white", edgecolor="black", fancybox=False)
        leg.get_frame().set_linewidth(1.0)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)


def plot_wz_posterior(
    *,
    table_path: Path = DATA_ZW,
    covariance_path: Path = DATA_COV,
    out_fig: Path = OUT_FIG,
    dpi: int = 200,
    n_samples: int = N_SAMPLES,
    dataset_label: str = "DESI DR2 + CMB + 1000 absolute GW",
    best_fit: np.ndarray | None = None,
    show: bool = True,
) -> None:
    z, w, _, _, _, samples = sample_w_posterior(
        table_path=table_path,
        covariance_path=covariance_path,
        n_samples=n_samples,
    )

    plt.rcParams.update(MPL_RC)
    fig, ax = plt.subplots(figsize=(8.8, 5.0), facecolor="white")
    draw_wz_posterior_axis(ax, z, w, samples, title=dataset_label, best_fit=best_fit)

    fig.tight_layout()
    out_fig = out_fig.resolve()
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"saved to: {out_fig}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def best_fit_paths(n_events: int) -> tuple[Path, Path]:
    if n_events == 100:
        return GW100_BEST_FIT_TABLE, GW100_BEST_FIT_SUMMARY
    return BEST_FIT_TABLE, BEST_FIT_SUMMARY


def result_paths(n_events: int) -> tuple[Path, Path, Path]:
    stem = f"desi_dr2_bbn_cmb_absolute_gw{n_events:04d}_wz"
    return (TABLES / f"{stem}.csv", TABLES / f"{stem}_covariance.txt",
            TABLES / f"{stem}_summary.json")


def individual_figure_path(n_events: int) -> Path:
    return FIGURES / f"wz_posterior_cov_bbn_cmb_desi_absolute_gw{n_events:04d}.png"


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
    """Read the locally optimized A=1 Horndeski fit, not the chain MAP sample."""
    best_fit_table, best_fit_summary = best_fit_paths(n_events)
    if not best_fit_table.is_file() or not best_fit_summary.is_file():
        raise FileNotFoundError(
            f"GW{n_events} optimized best-fit output is missing; "
            f"run mcmc/compute_best_fit_w_four_gw.py --n-events {n_events}"
        )
    payload = json.loads(best_fit_summary.read_text(encoding="utf-8"))
    if payload.get("A") != 1.0:
        raise ValueError("Optimized best-fit summary must use Horndeski prior amplitude A=1")
    cases = [case for case in payload.get("cases", []) if case.get("n_gw") == n_events]
    if len(cases) != 1 or not cases[0].get("optimization", {}).get("converged"):
        raise ValueError(f"GW{n_events} has no converged optimized best-fit result")
    with best_fit_table.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if int(row["n_gw"]) == n_events]
    if len(rows) != len(z_bins) or [int(row["bin"]) for row in rows] != list(range(1, len(z_bins) + 1)):
        raise ValueError(f"GW{n_events} optimized best-fit table does not match the w bins")
    try:
        values = np.asarray([float(row["w_bestfit"]) for row in rows])
        z_centers = np.asarray([float(row["z_center"]) for row in rows])
        summary_values = np.asarray([cases[0]["parameters"][f"w{i}"]
                                     for i in range(1, len(z_bins) + 1)], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"GW{n_events} optimized best-fit output is incomplete") from exc
    if not (np.all(np.isfinite(values)) and np.allclose(z_centers, z_bins, atol=1e-12)
            and np.allclose(values, summary_values, rtol=1e-12, atol=1e-12)):
        raise ValueError(f"GW{n_events} optimized best-fit CSV and JSON disagree")
    return values


def plot_individual(n_events: int, *, n_samples: int = N_SAMPLES, dpi: int = 200) -> Path:
    table, covariance, _ = result_paths(n_events)
    load_summary(n_events)
    z_bins, _, _, _, _ = load_z_w_bin_table(table, covariance)
    best_fit = best_fit_bins(n_events, z_bins)
    output = individual_figure_path(n_events)
    plot_wz_posterior(
        table_path=table, covariance_path=covariance, out_fig=output,
        dpi=dpi, n_samples=n_samples,
        dataset_label=f"BBN + CMB + DESI DR2 + GW{n_events}",
        best_fit=best_fit, show=False,
    )
    return output


def reconstruction_status(n_events: int) -> str:
    """Human-readable status for editor/IDE runs before products exist."""
    table, covariance, summary = result_paths(n_events)
    if table.is_file() and covariance.is_file() and summary.is_file():
        return "ready"
    chain_dir = BBN_GW_CHAIN_ROOT / f"GW{n_events:04d}"
    chains = sorted(chain_dir.glob("chain.[0-9]*.txt"))
    progress = chain_dir / "chain.progress"
    if progress.is_file():
        try:
            rows = [line.split() for line in progress.read_text(encoding="utf-8").splitlines()
                    if line.strip() and not line.lstrip().startswith("#")]
            if rows and len(rows[-1]) >= 4:
                return (f"sampling ({len(chains)} chains; Cobaya R-1="
                        f"{float(rows[-1][3]):.4f}, target 0.01)")
        except (OSError, ValueError):
            pass
    if chains:
        return f"sampling ({len(chains)} chain files; no convergence record yet)"
    return "not started"


def products_ready() -> bool:
    return (all(all(path.is_file() for path in result_paths(n)) for n in DEFAULT_N_EVENTS)
            and all(all(path.is_file() for path in best_fit_paths(n)) for n in DEFAULT_N_EVENTS))


def print_pending_status() -> None:
    print("The four-panel figure is waiting for converged reconstruction products:")
    for n_events in DEFAULT_N_EVENTS:
        print(f"  GW{n_events}: {reconstruction_status(n_events)}")
    print("\nAfter all four-chain runs finish, build/verify products with:")
    print("  python3 mcmc/reconstruct_wz_bbn_cmb_desi_absolute_gw.py --postprocess-only")
    print("  python3 mcmc/compute_best_fit_w_four_gw.py")
    print("  python3 mcmc/compute_best_fit_w_four_gw.py --n-events 100")
    print("Then run this plotting file again.")


def plot_four(*, out: Path = FOUR_GW_OUTPUT, n_samples: int = N_SAMPLES, dpi: int = 200) -> Path:
    if n_samples < 2 or dpi < 50:
        raise ValueError("n_samples must be >= 2 and dpi must be >= 50")
    plt.rcParams.update(MPL_RC)
    fig, axes = plt.subplots(2, 2, figsize=(17.6, 10.0), sharex=True, sharey=True,
                             facecolor="white")
    for panel, (ax, n_events) in enumerate(zip(axes.flat, DEFAULT_N_EVENTS)):
        table, covariance, _ = result_paths(n_events)
        load_summary(n_events)
        z_bins, w_mean, _, _, _, samples = sample_w_posterior(
            table_path=table, covariance_path=covariance,
            n_samples=n_samples,
        )
        best_fit = best_fit_bins(n_events, z_bins)
        draw_wz_posterior_axis(
            ax, z_bins, w_mean, samples,
            title=f"GW{n_events}",
            best_fit=best_fit, show_ylabel=panel % 2 == 0, show_legend=False,
            z_plot_max=COMPARISON_Z_PLOT_MAX,
        )
        if panel < 2:
            ax.set_xlabel("")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=len(labels), fontsize=15, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=1.0, h_pad=1.0)
    out = out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--n-events",
        type=int,
        nargs="+",
        help="older joint absolute-GW chain cases (100, 500, 1000, 3000)",
    )
    ap.add_argument("--table", type=Path, help="custom w(z) reconstruction table")
    ap.add_argument("--cov", type=Path, help="custom posterior covariance matrix")
    ap.add_argument("--out", type=Path, help="custom output figure")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--samples", type=int, default=N_SAMPLES)
    ap.add_argument("--label", help="custom plot title")
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument(
        "--compare-four-gw", action="store_true",
        help="plot BBN + CMB + DESI DR2 + GW100/500/1000/3000 singles and comparison",
    )
    args = ap.parse_args()

    comparison_mode = args.compare_four_gw or not any(
        value is not None for value in (args.n_events, args.table, args.cov, args.label)
    )
    if comparison_mode:
        if args.n_events or args.table or args.cov or args.label:
            ap.error("--compare-four-gw cannot be combined with --n-events/--table/--cov/--label")
        if args.samples < 2 or args.dpi < 50:
            ap.error("--compare-four-gw requires --samples >= 2 and --dpi >= 50")
        if not products_ready():
            print_pending_status()
            return
        for n_events in DEFAULT_N_EVENTS:
            plot_individual(n_events, n_samples=args.samples, dpi=args.dpi)
        figure = plot_four(out=args.out or FOUR_GW_OUTPUT, n_samples=args.samples, dpi=args.dpi)
        print(f"Saved: {figure}\nSaved: {figure.with_suffix('.txt')}")
        return

    if (args.table is None) != (args.cov is None):
        ap.error("--table and --cov must be provided together")
    custom_paths = any(value is not None for value in (args.table, args.cov, args.out, args.label))
    event_counts = args.n_events or ()
    if event_counts:
        if len(event_counts) > 1 and custom_paths:
            ap.error("custom --table/--cov/--out/--label can only be used with one event count")
        for n_events in event_counts:
            table, covariance, output = absolute_gw_paths(n_events)
            plot_wz_posterior(
                table_path=args.table or table,
                covariance_path=args.cov or covariance,
                out_fig=args.out or output,
                dpi=int(args.dpi),
                n_samples=int(args.samples),
                dataset_label=args.label or f"DESI DR2 + CMB + {n_events} absolute GW",
                show=not args.no_show,
            )
    else:
        plot_wz_posterior(
            table_path=args.table or DATA_ZW,
            covariance_path=args.cov or DATA_COV,
            out_fig=args.out or OUT_FIG,
            dpi=int(args.dpi),
            n_samples=int(args.samples),
            dataset_label=args.label or "DESI DR2 + CMB + 1000 absolute GW",
            show=not args.no_show,
        )


if __name__ == "__main__":
    main()
