#!/usr/bin/env python3
"""Compute GW-only truncated-PCA SNRs and Gaussian Bayes factors.

For each ET catalogue, recover the marginalized GW data Fisher matrix and
score from the converged A=0.01 GW+Horndeski-prior reconstruction,

    F_data = C_0.01^-1 - 0.01 P,
    j      = C_0.01^-1 (<w>_0.01 + 1).

The PCA basis consists of the descending eigenvectors of ``F_data``.  For the
first ``m`` modes, define

    F_m = E_m^T F_data E_m,  P_m = E_m^T P E_m,
    Q_m = F_m + P_m,         j_m = E_m^T j,
    mu_m = Q_m^-1 j_m.

The reported cumulative significance and analytic Gaussian evidence ratio are

    SNR_m^2       = mu_m^T Q_m mu_m,
    ln B_m,Lambda = SNR_m^2/2 + [ln det(P_m)-ln det(Q_m)]/2.

Positive ``ln B_m,Lambda`` favors the m-mode GW reconstruction over flat
LambdaCDM.  This is the same compressed Gaussian calculation used by
``compute_pca_snr_bayes_by_m.py``, but with GW absolute distances plus the
Horndeski correlation prior only.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcmc.analyze_pca_gw_prior_only import (  # noqa: E402
    load_posterior,
    ordered_eigensystem,
    product_stem,
)
from mcmc.bbn_cmb_desi_absolute_gw_wz_model import (  # noqa: E402
    N_W,
    horndeski_precision,
)


EVENT_COUNTS = (100, 500, 1000, 3000)
MAX_MODE = 7
WEAK_PRIOR_AMPLITUDE = 0.01
TABLES = ROOT / "results" / "tables"
STEM = "gw_prior_only_pca_snr_bayes_by_m"


def symmetric(matrix: np.ndarray) -> np.ndarray:
    """Remove insignificant antisymmetric numerical roundoff."""
    return 0.5 * (matrix + matrix.T)


def compute_catalog(
    n_events: int, prior_precision: np.ndarray
) -> list[dict[str, object]]:
    """Calculate the cumulative first-m-mode statistics for one catalogue."""
    weak_summary, weak_covariance, weak_mean = load_posterior(
        n_events, WEAK_PRIOR_AMPLITUDE
    )
    weak_precision = symmetric(
        np.linalg.solve(weak_covariance, np.eye(N_W))
    )
    data_fisher = symmetric(
        weak_precision - WEAK_PRIOR_AMPLITUDE * prior_precision
    )
    eigenvalues, eigenvectors = ordered_eigensystem(data_fisher)
    if np.any(eigenvalues[:MAX_MODE] <= 0.0):
        raise ValueError(
            f"GW{n_events}: one of the first {MAX_MODE} data modes is non-positive"
        )

    # Wiener-filter normal equation for departure d=w-w_LCDM=w+1:
    # (F_data + 0.01 P) <d>_0.01 = j.
    weak_departure = weak_mean + 1.0
    score = weak_precision @ weak_departure

    rows: list[dict[str, object]] = []
    previous_snr = -math.inf
    for m in range(1, MAX_MODE + 1):
        modes = eigenvectors[:, :m]
        fisher_m = symmetric(modes.T @ data_fisher @ modes)
        prior_m = symmetric(modes.T @ prior_precision @ modes)
        posterior_precision_m = symmetric(fisher_m + prior_m)
        score_m = modes.T @ score

        # These checks also reject a retained constant-prior null direction.
        np.linalg.cholesky(prior_m)
        np.linalg.cholesky(posterior_precision_m)
        posterior_mean_m = np.linalg.solve(posterior_precision_m, score_m)
        snr_squared = float(
            posterior_mean_m @ posterior_precision_m @ posterior_mean_m
        )
        if snr_squared < -1.0e-12:
            raise ValueError(f"GW{n_events}, m={m}: negative SNR squared")
        snr = math.sqrt(max(snr_squared, 0.0))
        if snr + 1.0e-12 < previous_snr:
            raise AssertionError(f"GW{n_events}: cumulative SNR decreased at m={m}")
        previous_snr = snr

        sign_prior, logdet_prior = np.linalg.slogdet(prior_m)
        sign_post, logdet_post = np.linalg.slogdet(posterior_precision_m)
        if sign_prior <= 0.0 or sign_post <= 0.0:
            raise ValueError(f"GW{n_events}, m={m}: invalid determinant sign")
        delta_ln_l = 0.5 * snr_squared
        delta_ln_v = 0.5 * (logdet_prior - logdet_post)
        ln_b = delta_ln_l + delta_ln_v
        rows.append(
            {
                "n_gw": n_events,
                "m": m,
                "SNR_m": snr,
                "delta_ln_L": delta_ln_l,
                "delta_ln_V": delta_ln_v,
                "ln_B_m_over_LCDM": ln_b,
                "B_m_over_LCDM": math.exp(ln_b),
                "lambda_data_m": float(eigenvalues[m - 1]),
                "catalog_sha256": weak_summary["catalog"]["sha256"],
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_wide_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    wide_rows: list[dict[str, object]] = []
    for m in range(1, MAX_MODE + 1):
        wide: dict[str, object] = {"m": m}
        for n_events in EVENT_COUNTS:
            row = next(
                item
                for item in rows
                if int(item["n_gw"]) == n_events and int(item["m"]) == m
            )
            wide[f"SNR_N{n_events:04d}"] = row["SNR_m"]
            wide[f"lnB_N{n_events:04d}"] = row["ln_B_m_over_LCDM"]
        wide_rows.append(wide)
    return wide_rows


def write_latex(path: Path, wide_rows: list[dict[str, object]]) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\caption{",
        r"Covariance-weighted reconstruction significance and Bayesian evidence",
        r"ratios for the GW-only plus Horndeski-prior analysis and different",
        r"numbers of included principal components. The quantities SNR and",
        r"$\ln B_{m,\Lambda}$ are calculated using the first $m$ principal components.}",
        r"\label{tab:gw_prior_only_pca_snr_bayes}",
        r"\centering",
        r"\begin{ruledtabular}",
        r"\begin{tabular}{c|cccc|cccc}",
        r"& \multicolumn{4}{c|}{SNR}",
        r"& \multicolumn{4}{c}{$\ln B_{m,\Lambda}$}",
        r"\\",
        r"$m$",
        r"& GW100 & GW500 & GW1000 & GW3000",
        r"& GW100 & GW500 & GW1000 & GW3000",
        r"\\",
        r"\hline",
    ]
    for wide in wide_rows:
        snr_fields = [
            f"{float(wide[f'SNR_N{n_events:04d}']):.3f}"
            for n_events in EVENT_COUNTS
        ]
        lnb_fields = [
            f"{float(wide[f'lnB_N{n_events:04d}']):.3f}"
            for n_events in EVENT_COUNTS
        ]
        lines.append(
            f"{int(wide['m'])} & "
            + " & ".join(snr_fields)
            + " & "
            + " & ".join(lnb_fields)
            + r" \\"
        )
    lines.extend(
        [
            r"\end{tabular}",
            r"\end{ruledtabular}",
            r"\end{table*}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    prior_precision = horndeski_precision()
    rows = [
        row
        for n_events in EVENT_COUNTS
        for row in compute_catalog(n_events, prior_precision)
    ]
    wide_rows = make_wide_rows(rows)

    long_path = TABLES / f"{STEM}.csv"
    wide_path = TABLES / f"{STEM}_wide.csv"
    latex_path = TABLES / f"{STEM}.tex"
    json_path = TABLES / f"{STEM}.json"
    write_csv(long_path, rows)
    write_csv(wide_path, wide_rows)
    write_latex(latex_path, wide_rows)
    json_path.write_text(
        json.dumps(
            {
                "description": (
                    "GW-only SNR and analytic Gaussian Bayes factor for the "
                    "first m data-defined PCA modes"
                ),
                "data_combination": "absolute ET GW + Horndeski correlation prior",
                "reference_model": "flat LCDM, w_i=-1",
                "mode_basis": (
                    "descending eigenvectors of "
                    "F_data=inv(C_A001)-0.01 P_Horndeski"
                ),
                "method": {
                    "posterior_precision_m": (
                        "Q_m=E_m^T(F_data+P_Horndeski)E_m"
                    ),
                    "score_m": "j_m=E_m^T inv(C_A001) <w+1>_A001",
                    "posterior_mean_m": "mu_m=inv(Q_m)j_m",
                    "snr": "sqrt(mu_m^T Q_m mu_m)",
                    "delta_ln_L": "SNR_m^2/2",
                    "delta_ln_V": "[ln det(P_m)-ln det(Q_m)]/2",
                    "ln_bayes_factor": "delta_ln_L+delta_ln_V",
                },
                "sign_convention": (
                    "positive ln_B_m_over_LCDM favors the m-mode w(z) model"
                ),
                "approximation": (
                    "Analytic Gaussian evidence compressed from converged "
                    "GW-only A=0.01 MCMC chains"
                ),
                "source_files": {
                    str(n_events): [
                        str(
                            (
                                TABLES
                                / f"{product_stem(n_events, WEAK_PRIOR_AMPLITUDE)}_summary.json"
                            ).relative_to(ROOT)
                        ),
                        str(
                            (
                                TABLES
                                / f"{product_stem(n_events, WEAK_PRIOR_AMPLITUDE)}_covariance.txt"
                            ).relative_to(ROOT)
                        ),
                    ]
                    for n_events in EVENT_COUNTS
                },
                "results": rows,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print("m  " + "  ".join(f"GW{n}: SNR / lnB" for n in EVENT_COUNTS))
    for wide in wide_rows:
        fields = [f"{int(wide['m']):d}"]
        for n_events in EVENT_COUNTS:
            fields.append(
                f"{float(wide[f'SNR_N{n_events:04d}']):.3f} / "
                f"{float(wide[f'lnB_N{n_events:04d}']):+.3f}"
            )
        print("  ".join(fields))
    for path in (long_path, wide_path, latex_path, json_path):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
