"""
Posterior w(z): sample from cov, smooth percentiles, plot vs z.

Reads ``data/z_w.txt`` and ``data/cov_w_posterior.txt``.
Rows with sigma <= 0 are excluded (anchor bin at z > 2.5).

Shared helpers (H0, d_L from binned w) are used by plot_cc_bao_data.py,
plot_dl_desi_lcdm_frac.py, and catalog/dmu_desi_lcdm.py.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paths import setup_sys_path, DATA, FIGURES

setup_sys_path()

from gw_cosmology import C_KM_S

DATA_ZW = DATA / "z_w.txt"
DATA_COV = DATA / "cov_w_posterior.txt"
OUT_FIG = FIGURES / "wz_posterior_and_cov.png"

H0_PLANCK = 67.4  # km/s/Mpc, Planck 2018
OMEGA_M_PLANCK = 0.315

Z_PLOT_MAX = 2.5
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


def draw_multivariate_normal(
    mean: np.ndarray,
    covariance: np.ndarray,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw samples while hiding known spurious Apple Accelerate warnings."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"(divide by zero|overflow|invalid value) encountered in matmul",
            category=RuntimeWarning,
        )
        samples = rng.multivariate_normal(mean=mean, cov=covariance, size=int(size))
    if not np.all(np.isfinite(samples)):
        raise FloatingPointError("Non-finite values in Gaussian posterior draws")
    return samples


def _load_cov(n_bins: int, mask: np.ndarray | None = None) -> np.ndarray:
    cov = np.loadtxt(DATA_COV, comments="#").astype(float)
    if mask is not None:
        if cov.shape[0] == len(mask):
            cov = cov[np.ix_(mask, mask)]
        elif cov.shape[0] == int(mask.sum()) + 1:
            cov = cov[:-1, :-1]
    elif cov.shape[0] == n_bins + 1:
        cov = cov[:-1, :-1]
    if cov.shape[0] != n_bins:
        raise ValueError(
            f"bin count mismatch: {n_bins} w bins vs {cov.shape[0]}x{cov.shape[1]} cov"
        )
    return cov


def load_z_w_bin_table() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (z_center, w_mean, cov, z_bin_lo, z_bin_hi) for bins with sigma > 0."""
    data = np.loadtxt(DATA_ZW, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)

    zp_lo = data[:, 1].astype(float)
    zp_hi = data[:, 2].astype(float)
    z = data[:, 3].astype(float)
    w = data[:, 4].astype(float)
    sigma = data[:, 5].astype(float) if data.shape[1] >= 6 else np.ones_like(z)
    mask = sigma > 0

    z, w = z[mask], w[mask]
    zp_lo, zp_hi = zp_lo[mask], zp_hi[mask]
    cov = _load_cov(len(z), mask)
    return z, w, cov, zp_lo, zp_hi


def sample_w_posterior(
    *,
    n_samples: int = N_SAMPLES,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Draw w-bin samples from the DESI posterior Gaussian approximation."""
    z, w, cov, zp_lo, zp_hi = load_z_w_bin_table()
    rng = np.random.default_rng(RNG_SEED) if rng is None else rng
    samples = draw_multivariate_normal(w, cov, int(n_samples), rng)
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


def plot_wz_posterior(
    *,
    out_fig: Path = OUT_FIG,
    dpi: int = 200,
    show: bool = True,
) -> None:
    z, w, cov, _, _ = load_z_w_bin_table()
    rng = np.random.default_rng(RNG_SEED)
    samples = draw_multivariate_normal(w, cov, N_SAMPLES, rng)

    z_plot = np.linspace(0.0, Z_PLOT_MAX, N_Z)
    q2, q16, q50, q84, q97 = _percentile_curves(z, samples, z_plot)

    plt.rcParams.update(MPL_RC)
    fig, ax = plt.subplots(figsize=(8.8, 5.0), facecolor="white")
    ax.set_facecolor("white")

    ax.fill_between(
        z_plot, q2, q97, color="#0072BD", alpha=0.25, lw=0,
        label="95% credible interval",
    )
    ax.fill_between(
        z_plot, q16, q84, color="#0072BD", alpha=0.45, lw=0,
        label="68% credible interval",
    )
    ax.plot(z_plot, q50, color="#0072BD", lw=2, label=r"best-fit $w(z)$")
    ax.plot(z, w, "o", ms=6, mfc="#D95319", mec="white", mew=0.5,
            label=r"best-fit $w_i$", zorder=4)
    ax.axhline(-1, color="black", ls="--", lw=2.2, label=r"$\Lambda$CDM ($w=-1$)")

    ax.set_xlabel(r"$z$", fontsize=22)
    ax.set_ylabel(r"$w(z)$", fontsize=22)
    ax.set_xlim(0.0, Z_PLOT_MAX)
    ax.set_ylim(-4.0, 4.0)
    ax.set_yticks([-2, 0, 2, 4])
    ax.tick_params(labelsize=18, which="both")

    leg = ax.legend(fontsize=16, loc="upper right", frameon=True,
                    facecolor="white", edgecolor="black", fancybox=False)
    leg.get_frame().set_linewidth(1.0)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    fig.tight_layout()
    out_fig = out_fig.resolve()
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"saved to: {out_fig}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot DESI w(z) posterior with confidence bands.")
    ap.add_argument("--out", type=Path, default=OUT_FIG)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()
    plot_wz_posterior(out_fig=args.out, dpi=int(args.dpi), show=not args.no_show)


if __name__ == "__main__":
    main()
