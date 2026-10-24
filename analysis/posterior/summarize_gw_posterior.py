"""Summarize four-parameter CPL MCMC posteriors for four GW catalog sizes.

The input chains are produced by ``gw_cpl_mcmc.py``. Each row has the
CosmoMC-style layout ``weight, log_likelihood, H0, omegam, w0, wa``.
Cobaya writes repeat counts in the first column, so posterior moments must
use those weights.

Run with:
    python analysis/posterior/summarize_gw_posterior.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paths import SUMMARIZE_GW_POSTERIOR_TABLES, gw_cpl_chain_root


DEFAULT_N_EVENTS = (100, 500, 1000, 3000)
PARAMETERS = ("H0", "omegam", "w0", "wa")
DEFAULT_CSV = SUMMARIZE_GW_POSTERIOR_TABLES / "gw_four_catalogs_posterior_mean_variance.csv"


def load_posterior(n_events: int) -> tuple[np.ndarray, np.ndarray]:
    """Load weighted H0, omegam, w0, wa samples for one catalog size."""
    prefix = gw_cpl_chain_root(n_events)
    files = sorted(prefix.parent.glob(f"{prefix.name}_*.txt"))
    if not files:
        raise FileNotFoundError(f"no MCMC chain files found for N_GW={n_events}: {prefix.parent}")

    chains: list[np.ndarray] = []
    for file in files:
        values = np.loadtxt(file, ndmin=2)
        if values.shape[1] != 6:
            raise ValueError(f"expected six columns in {file}, found {values.shape[1]}")
        if not np.all(np.isfinite(values)) or np.any(values[:, 0] <= 0):
            raise ValueError(f"invalid weights or samples in {file}")
        chains.append(values)
    data = np.vstack(chains)
    return data[:, 0], data[:, 2:6]


def summarize(n_events: int) -> dict[str, int | float]:
    weights, samples = load_posterior(n_events)
    means = np.average(samples, axis=0, weights=weights)
    variances = np.average((samples - means)**2, axis=0, weights=weights)
    row: dict[str, int | float] = {
        "n_gw": n_events,
        "n_samples": int(samples.shape[0]),
        "weight_sum": float(weights.sum()),
    }
    for name, value in zip(PARAMETERS, means):
        row[f"{name}_mean"] = float(value)
    for name, value in zip(PARAMETERS, variances):
        row[f"{name}_variance"] = float(value)
    return row


def print_table(rows: list[dict[str, int | float]]) -> str:
    lines = [
        "Four-parameter CPL posterior mean and variance",
        "=" * 125,
        "N_GW    samples    mean(H0)     var(H0)  mean(omegam) var(omegam)     mean(w0)     var(w0)     mean(wa)      var(wa)",
        "-" * 125,
    ]
    for row in rows:
        lines.append(
            f"{row['n_gw']:>4}  {row['n_samples']:>9}  "
            f"{row['H0_mean']:>10.5f}  {row['H0_variance']:>10.5f}  "
            f"{row['omegam_mean']:>12.6f}  {row['omegam_variance']:>11.6f}  "
            f"{row['w0_mean']:>11.6f}  {row['w0_variance']:>10.6f}  "
            f"{row['wa_mean']:>11.6f}  {row['wa_variance']:>11.6f}",
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize four-parameter GW MCMC posterior samples.")
    ap.add_argument("--n-events", type=int, nargs="+", default=list(DEFAULT_N_EVENTS))
    ap.add_argument("--out-csv", type=Path, default=DEFAULT_CSV)
    args = ap.parse_args()

    n_events = sorted(set(n for n in args.n_events if n > 0))
    if not n_events:
        raise ValueError("--n-events must contain at least one positive integer")
    rows = [summarize(n) for n in n_events]
    report = print_table(rows)
    print(report)

    csv_path = args.out_csv.resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote: {csv_path}")


if __name__ == "__main__":
    main()
