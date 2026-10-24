from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paths import setup_sys_path, CATALOGS, FIGURES

setup_sys_path()

OUT_FIG_DEFAULT = FIGURES / "mock_GW_catalog_ET.png"
DEFAULT_N_EVENTS = (100, 500, 1000, 3000)
DEFAULT_REALIZATION = 42
OUT_BY_N_DEFAULT = FIGURES / "mock_GW_catalog_ET_by_4.png"
SINGLE_Z_MAX = 5
BY_N_Z_MAX = 4
LABEL_FONTSIZE = 22
TICK_FONTSIZE = 17
PANEL_LABEL_FONTSIZE = 15
PANEL_TICK_FONTSIZE = 12
PANEL_MARKER_SIZE = 1.8

# Data points/errorbars keep color
C_DATA = "#0072BD"  # ET color from plot_sensitivity.py
PANEL_COLORS = ("#9ecae1", "#4292c6", "#2171b5", "#08306b")

# Reference-figure style: serif/TeX-like fonts, plain black frame
MPL_RC: dict[str, Any] = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "font.family": "serif",
    "mathtext.fontset": "stix",
}


def load_catalog_txt(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load mock catalog: z, d_L_obs, sigma_dL (Mpc). Skips # lines and any row whose
    first field is not a float (e.g. savetxt text header).
    """
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"catalog not found: {path}")

    rows: list[list[float]] = []
    with path.open() as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 3:
                continue
            try:
                rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
            except ValueError:
                continue

    if not rows:
        raise ValueError(f"no numeric data rows (≥3 columns) in {path}")

    d = np.asarray(rows, dtype=np.float64)
    return d[:, 0], d[:, 1], d[:, 2]


def catalog_path(root: Path, n_events: int, realization: int) -> Path:
    """Return the standard path for one generated GW realization."""
    return root / f"N{n_events:04d}" / f"realization_{realization:04d}.txt"


def _masked_catalog(
    z: np.ndarray,
    d_l: np.ndarray,
    sigma: np.ndarray,
    *,
    z_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = (z >= 0.0) & (z <= z_max)
    return z[mask], d_l[mask], sigma[mask]


def plot_catalog(
    *,
    catalog_path: Path,
    out_fig: Path,
    dpi: int = 300,
    show: bool = True,
) -> Path:
    import matplotlib.pyplot as plt

    z_s, dL_obs, sigma_dL = load_catalog_txt(catalog_path)
    z_s, dL_obs, sigma_dL = _masked_catalog(
        z_s, dL_obs, sigma_dL, z_max=SINGLE_Z_MAX
    )
    n_pts = int(z_s.size)

    plt.rcParams.update(MPL_RC)

    fig, ax = plt.subplots(figsize=(8, 6), facecolor="white")
    ax.set_facecolor("white")

    dL_obs_gpc = dL_obs / 1e3
    sigma_gpc = sigma_dL / 1e3

    ax.errorbar(
        z_s,
        dL_obs_gpc,
        yerr=sigma_gpc,
        fmt="o",
        ms=2.0,
        color=C_DATA,
        ecolor=C_DATA,
        alpha=0.85,
        elinewidth=0.8,
        capsize=1.8,
        capthick=0.8,
        zorder=2,
    )

    ax.set_xlabel(r"$z$", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(r"$d_L(z)$ [Gpc]", fontsize=LABEL_FONTSIZE)
    ax.set_xlim(0.0, SINGLE_Z_MAX)
    ax.set_ylim(0.0, None)
    ax.tick_params(axis="both", which="both", labelsize=TICK_FONTSIZE)
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: "" if abs(y) < 1e-12 else f"{y:g}"))

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    out_fig = out_fig.resolve()
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_fig, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(
        f"Read {n_pts} events with z in [0, {SINGLE_Z_MAX:g}] from {catalog_path.resolve()}\n"
        f"Wrote {out_fig}"
    )
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out_fig


def plot_catalogs_by_n(
    *,
    root: Path,
    n_events_list: list[int],
    realization: int,
    out_fig: Path,
    dpi: int = 300,
    show: bool = True,
) -> Path:
    """Plot one realization for each requested catalog size in a panel grid."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    datasets: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
    y_max = 0.0
    for n_events in sorted(n_events_list):
        path = catalog_path(root, n_events, realization)
        z_s, d_l_obs, sigma_d_l = _masked_catalog(
            *load_catalog_txt(path), z_max=BY_N_Z_MAX
        )
        d_l_gpc = d_l_obs / 1e3
        sigma_gpc = sigma_d_l / 1e3
        if d_l_gpc.size:
            y_max = max(y_max, float(np.max(d_l_gpc + sigma_gpc)))
        datasets.append((n_events, z_s, d_l_gpc, sigma_gpc))

    n_cols = 2
    n_rows = int(np.ceil(len(datasets) / n_cols))
    plt.rcParams.update(MPL_RC)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(10.4, 4.4 * n_rows), facecolor="white", squeeze=False
    )

    for idx, (n_events, z_s, d_l_gpc, sigma_gpc) in enumerate(datasets):
        ax = axes.flat[idx]
        color = PANEL_COLORS[idx % len(PANEL_COLORS)]
        ax.errorbar(
            z_s, d_l_gpc, yerr=sigma_gpc, fmt="o", ms=PANEL_MARKER_SIZE,
            color=color, ecolor=color, alpha=0.85, elinewidth=0.8,
            capsize=1.6, capthick=0.8, zorder=2,
        )
        ax.set_xlim(0.0, BY_N_Z_MAX)
        ax.set_ylim(0.0, y_max * 1.05 if y_max > 0 else 1.0)
        ax.set_xlabel(r"$z$", fontsize=PANEL_LABEL_FONTSIZE)
        ax.set_ylabel(r"$d_L(z)$ [Gpc]", fontsize=PANEL_LABEL_FONTSIZE)
        ax.tick_params(axis="both", which="both", labelsize=PANEL_TICK_FONTSIZE)
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda y, _: "" if abs(y) < 1e-12 else f"{y:g}")
        )
        ax.set_title(rf"$N_{{\rm GW}}={n_events}$", fontsize=PANEL_LABEL_FONTSIZE)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)

    for ax in axes.flat[len(datasets):]:
        ax.set_visible(False)

    out_fig = out_fig.resolve()
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_fig, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"Wrote {out_fig}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out_fig


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot ET mock GW catalogs.")
    ap.add_argument(
        "--mode", choices=("single", "by-n"), default="by-n",
        help="Plot one catalog, or panel plots for catalog sizes (default: by-n).",
    )
    ap.add_argument(
        "--catalog",
        type=Path,
        default=None,
        help="Input catalog for --mode single (required in that mode).",
    )
    ap.add_argument(
        "--root", type=Path, default=CATALOGS,
        help=f"Catalog realization root for --mode by-n (default: {CATALOGS})",
    )
    ap.add_argument(
        "--realization", type=int, default=DEFAULT_REALIZATION,
        help="Realization index for --mode by-n (default: %(default)s).",
    )
    ap.add_argument(
        "--n-events", type=int, nargs="+", default=list(DEFAULT_N_EVENTS),
        help="Catalog sizes for --mode by-n (default: 100 500 1000 3000).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG (defaults depend on --mode).",
    )
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--no-show", action="store_true", help="Do not open figure window after save.")
    args = ap.parse_args()

    if args.mode == "single":
        if args.catalog is None:
            ap.error("--catalog is required with --mode single")
        cat = args.catalog.expanduser()
        if not cat.is_file():
            print(f"error: catalog file not found:\n  {cat.resolve()}", file=sys.stderr)
            print("hint: run  python catalog/gw_mock_catalog_ET.py", file=sys.stderr)
            sys.exit(1)
        plot_catalog(
            catalog_path=cat,
            out_fig=(args.out or OUT_FIG_DEFAULT).expanduser(),
            dpi=int(args.dpi),
            show=not args.no_show,
        )
        return

    root = args.root.expanduser().resolve()
    n_list = sorted(set(int(n) for n in args.n_events))
    if not n_list or any(n <= 0 for n in n_list):
        ap.error("--n-events must contain one or more positive integers")
    for n_events in n_list:
        path = catalog_path(root, n_events, int(args.realization))
        if not path.is_file():
            print(f"error: catalog file not found:\n  {path}", file=sys.stderr)
            print("hint: run  python catalog/generate_gw_mock_realizations.py", file=sys.stderr)
            sys.exit(1)
    plot_catalogs_by_n(
        root=root,
        n_events_list=n_list,
        realization=int(args.realization),
        out_fig=(args.out or OUT_BY_N_DEFAULT).expanduser(),
        dpi=int(args.dpi),
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
