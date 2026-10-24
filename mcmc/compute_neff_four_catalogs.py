#!/usr/bin/env python3
"""Compute covariant-PCA N_eff for the four ET GW catalogues.

The effective number of data-constrained dark-energy degrees of freedom is

    N_eff = N_w - Tr(Ctilde_Pi^{-1} C_w),

where C_w is the marginalized posterior covariance of the 29 w(z) bins and
Ctilde_Pi^{-1} is the modified Horndeski prior precision used by the chains.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcmc.bbn_cmb_desi_absolute_gw_wz_model import (  # noqa: E402
    N_W,
    horndeski_precision,
)


EVENT_COUNTS = (100, 500, 1000, 3000)
TABLE_DIR = ROOT / "results" / "tables"
OUTPUT = TABLE_DIR / "desi_dr2_bbn_cmb_absolute_gw_neff.csv"


def main() -> None:
    prior_precision = horndeski_precision()
    if prior_precision.shape != (N_W, N_W):
        raise ValueError("Prior precision has an unexpected shape")

    rows = []
    for n_gw in EVENT_COUNTS:
        covariance_path = (
            TABLE_DIR
            / f"desi_dr2_bbn_cmb_absolute_gw{n_gw:04d}_wz_covariance.txt"
        )
        covariance = np.loadtxt(covariance_path)
        if covariance.shape != (N_W, N_W):
            raise ValueError(
                f"{covariance_path} has shape {covariance.shape}, "
                f"expected {(N_W, N_W)}"
            )
        np.linalg.cholesky(covariance)

        # Tr(P C) = sum_ij P_ij C_ji.
        trace_term = float(np.einsum("ij,ji->", prior_precision, covariance))
        n_eff = float(N_W - trace_term)
        rows.append(
            {
                "n_gw": n_gw,
                "trace_prior_precision_times_Cw": trace_term,
                "N_eff": n_eff,
            }
        )
        print(f"GW{n_gw:04d}: trace={trace_term:.12f}, N_eff={n_eff:.12f}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
