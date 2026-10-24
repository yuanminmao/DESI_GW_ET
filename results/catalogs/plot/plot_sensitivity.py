"""
Noise characteristic strain h_n(f) = sqrt(f S_h(f)) for ET-D.

  python plot/plot_sensitivity.py   ->  results/figures/sensitivity_curves.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paths import setup_sys_path, DATA, FIGURES

setup_sys_path()

FILE_ET = DATA / "ET-D.txt"
OUT_FIG = FIGURES / "sensitivity_curves.png"

matplotlib.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'stix',
    'font.size': 16,
    'axes.labelsize': 16,
    'legend.fontsize': 14,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'figure.dpi': 150,
})


def load_hn(filename: Path):
    """Last column = total ASD sqrt(S_h); h_n(f) = sqrt(f S_h) = sqrt(f)*ASD."""
    data = np.loadtxt(filename, comments='#')
    freq = data[:, 0]
    asd = data[:, -1]
    return freq, np.sqrt(freq) * asd


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot ET-D noise characteristic strain curve.")
    ap.add_argument("--input", type=Path, default=FILE_ET, help=f"ET-D sensitivity file (default: {FILE_ET})")
    ap.add_argument("--out", type=Path, default=OUT_FIG, help=f"Output PNG (default: {OUT_FIG})")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    f_et, hn_et = load_hn(args.input.resolve())

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.loglog(f_et, hn_et, color='#0072BD', lw=2.0, label='ET (ET-D 10 km)')

    ax.set_xlim(1, 5000)
    ax.set_ylim(1e-24, 1e-16)
    ax.set_xlabel(r'$f$ [Hz]')
    ax.set_ylabel(r'$h_n(f)$')

    leg = ax.legend(loc='upper right', frameon=True, facecolor='white',
                    edgecolor='black', fancybox=False)
    leg.get_frame().set_linewidth(1.0)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    plt.tight_layout()
    out = args.out.expanduser().with_suffix(".png").resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=int(args.dpi), bbox_inches='tight')
    print(f"saved: {out}")
    plt.show(block=True)
    plt.close(fig)


if __name__ == "__main__":
    main()
