#!/usr/bin/env python3
"""Compute truncated-PCA SNRs and Gaussian Bayes factors for m=1,...,6.

For each BBN+CMB+DESI DR2+ET catalog, the data Fisher matrix and score are
recovered from the converged A=0.01 weak-prior reconstruction,

    F_data = C_0.01^-1 - 0.01 P,
    j      = C_0.01^-1 d_0.01,

where ``P`` is the modified Horndeski prior precision and ``d=w+1``.  The
PCA modes are the eigenvectors of ``F_data``, ordered from largest to
smallest eigenvalue.  After retaining the first ``m`` modes (columns of
``E_m``), the compressed matrices are

    F_m = E_m^T F_data E_m,  P_m = E_m^T P E_m,
    Q_m = F_m + P_m,         j_m = E_m^T j.

The Gaussian posterior mean is ``mu_m=Q_m^-1 j_m``.  The reported quantities
are

    SNR(m)^2 = mu_m^T Q_m mu_m,
    ln B_m,Lambda = SNR(m)^2/2 + [ln det(P_m)-ln det(Q_m)]/2.

Thus positive ln B favors the m-mode w(z) model over flat LambdaCDM.  This is
an analytic Gaussian evidence calculation based on the converged chains; it
is not a new nonlinear dynesty run.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.special import ndtri


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcmc.bbn_cmb_desi_absolute_gw_wz_model import (  # noqa: E402
    N_W,
    horndeski_precision,
)


EVENT_COUNTS = (100, 500, 1000, 3000)
MAX_MODE = 7
WEAK_PRIOR_AMPLITUDE = 0.01
EVIDENCE_NLIVE = 300
EVIDENCE_DLOGZ = 0.1
EVIDENCE_SEED = 20260925
TABLES = ROOT / "results" / "tables"
STEM = "desi_dr2_bbn_cmb_absolute_gw_pca_snr_bayes_by_m"


def symmetric(matrix: np.ndarray) -> np.ndarray:
    """Remove insignificant antisymmetric roundoff from a square matrix."""
    return 0.5 * (matrix + matrix.T)


def ordered_modes(fisher: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return descending eigenvalues and sign-fixed orthonormal eigenvectors."""
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric(fisher))
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    for index in range(eigenvectors.shape[1]):
        pivot = int(np.argmax(np.abs(eigenvectors[:, index])))
        if eigenvectors[pivot, index] < 0.0:
            eigenvectors[:, index] *= -1.0
    return eigenvalues, eigenvectors


def source_paths(n_gw: int) -> tuple[Path, Path]:
    stem = f"desi_dr2_bbn_cmb_absolute_gw{n_gw:04d}_pca_A001"
    return TABLES / f"{stem}_summary.json", TABLES / f"{stem}_wz_covariance.txt"


def estimate_ln_b_sigma(
    fisher_m: np.ndarray,
    prior_m: np.ndarray,
    score_m: np.ndarray,
    *,
    seed: int,
) -> float:
    """Estimate the 1-sigma numerical uncertainty of ln B with dynesty.

    The sampler integrates the already-compressed Gaussian m-mode model.  The
    LCDM reference evidence is fixed to one in this relative normalization,
    so the reported log-evidence error is the log-Bayes-factor error.  This is
    a nested-sampling integration error, not an additional data/model error.
    """
    import dynesty

    prior_covariance = np.linalg.solve(prior_m, np.eye(prior_m.shape[0]))
    prior_cholesky = np.linalg.cholesky(
        0.5 * (prior_covariance + prior_covariance.T)
    )

    def prior_transform(unit_cube: np.ndarray) -> np.ndarray:
        normal = ndtri(np.clip(np.asarray(unit_cube, dtype=float), 1e-12, 1.0 - 1e-12))
        return prior_cholesky @ normal

    def loglike(amplitude: np.ndarray) -> float:
        amplitude = np.asarray(amplitude, dtype=float)
        return float(score_m @ amplitude - 0.5 * amplitude @ fisher_m @ amplitude)

    sampler = dynesty.NestedSampler(
        loglike,
        prior_transform,
        ndim=prior_m.shape[0],
        nlive=EVIDENCE_NLIVE,
        rstate=np.random.default_rng(seed),
    )
    sampler.run_nested(dlogz=EVIDENCE_DLOGZ, print_progress=False)
    return float(sampler.results.logzerr[-1])


def compute_catalog(n_gw: int, prior_precision: np.ndarray) -> list[dict[str, object]]:
    summary_path, covariance_path = source_paths(n_gw)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("prior_amplitude") != WEAK_PRIOR_AMPLITUDE:
        raise ValueError(f"GW{n_gw}: expected an A=0.01 source chain")
    if not summary["rank_folded_split_rhat"]["all_pass"]:
        raise ValueError(f"GW{n_gw}: A=0.01 source chains are not converged")

    weak_covariance = np.loadtxt(covariance_path)
    if weak_covariance.shape != (N_W, N_W):
        raise ValueError(
            f"GW{n_gw}: covariance has shape {weak_covariance.shape}, "
            f"expected {(N_W, N_W)}"
        )
    np.linalg.cholesky(weak_covariance)
    weak_precision = symmetric(
        np.linalg.solve(weak_covariance, np.eye(N_W))
    )
    data_fisher = symmetric(
        weak_precision - WEAK_PRIOR_AMPLITUDE * prior_precision
    )
    eigenvalues, eigenvectors = ordered_modes(data_fisher)
    if np.any(eigenvalues[:MAX_MODE] <= 0.0):
        raise ValueError(f"GW{n_gw}: one of the first {MAX_MODE} modes is non-positive")

    weak_mean_departure = np.array([
        summary["joint_posterior_mean"][f"w{i}"] + 1.0
        for i in range(1, N_W + 1)
    ])
    # Wiener-filter normal equation: (F + 0.01 P) d_0.01 = j.
    score = weak_precision @ weak_mean_departure

    rows: list[dict[str, object]] = []
    previous_snr = -math.inf
    for m in range(1, MAX_MODE + 1):
        modes = eigenvectors[:, :m]
        fisher_m = symmetric(modes.T @ data_fisher @ modes)
        prior_m = symmetric(modes.T @ prior_precision @ modes)
        posterior_precision_m = symmetric(fisher_m + prior_m)
        score_m = modes.T @ score

        # Cholesky calls also provide explicit positive-definiteness checks.
        np.linalg.cholesky(prior_m)
        np.linalg.cholesky(posterior_precision_m)
        posterior_mean_m = np.linalg.solve(posterior_precision_m, score_m)
        snr_squared = float(
            posterior_mean_m @ posterior_precision_m @ posterior_mean_m
        )
        if snr_squared < -1e-12:
            raise ValueError(f"GW{n_gw}, m={m}: negative SNR squared")
        snr_squared = max(snr_squared, 0.0)
        snr = math.sqrt(snr_squared)
        if snr + 1e-12 < previous_snr:
            raise AssertionError(f"GW{n_gw}: cumulative SNR decreased at m={m}")
        previous_snr = snr

        sign_prior, logdet_prior = np.linalg.slogdet(prior_m)
        sign_post, logdet_post = np.linalg.slogdet(posterior_precision_m)
        if sign_prior <= 0.0 or sign_post <= 0.0:
            raise ValueError(f"GW{n_gw}, m={m}: invalid determinant sign")
        delta_ln_l = 0.5 * snr_squared
        delta_ln_v = 0.5 * (logdet_prior - logdet_post)
        ln_b = delta_ln_l + delta_ln_v
        ln_b_sigma = estimate_ln_b_sigma(
            fisher_m,
            prior_m,
            score_m,
            seed=EVIDENCE_SEED + 100 * n_gw + m,
        )
        rows.append({
            "n_gw": n_gw,
            "m": m,
            "SNR_m": snr,
            "delta_ln_L": delta_ln_l,
            "delta_ln_V": delta_ln_v,
            "ln_B_m_over_LCDM": ln_b,
            "ln_B_sigma": ln_b_sigma,
            "B_m_over_LCDM": math.exp(ln_b),
            "B_lower_1sigma": math.exp(ln_b - ln_b_sigma),
            "B_upper_1sigma": math.exp(ln_b + ln_b_sigma),
            "lambda_data_m": float(eigenvalues[m - 1]),
        })
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    prior_precision = horndeski_precision()
    rows = [
        row
        for n_gw in EVENT_COUNTS
        for row in compute_catalog(n_gw, prior_precision)
    ]

    long_path = TABLES / f"{STEM}.csv"
    write_csv(long_path, rows)

    # A compact publication-style layout: one row per m and paired columns
    # for each catalog.
    wide_rows: list[dict[str, object]] = []
    for m in range(1, MAX_MODE + 1):
        wide: dict[str, object] = {"m": m}
        for n_gw in EVENT_COUNTS:
            row = next(
                item for item in rows
                if int(item["n_gw"]) == n_gw and int(item["m"]) == m
            )
            wide[f"SNR_N{n_gw:04d}"] = row["SNR_m"]
            wide[f"lnB_N{n_gw:04d}"] = row["ln_B_m_over_LCDM"]
            wide[f"lnB_sigma_N{n_gw:04d}"] = row["ln_B_sigma"]
            wide[f"B_N{n_gw:04d}"] = row["B_m_over_LCDM"]
            wide[f"B_lower_1sigma_N{n_gw:04d}"] = row["B_lower_1sigma"]
            wide[f"B_upper_1sigma_N{n_gw:04d}"] = row["B_upper_1sigma"]
        wide_rows.append(wide)
    wide_path = TABLES / f"{STEM}_wide.csv"
    write_csv(wide_path, wide_rows)

    json_path = TABLES / f"{STEM}.json"
    json_path.write_text(
        json.dumps({
            "description": (
                "SNR and analytic Gaussian Bayes factor for the first m "
                "data-defined PCA modes"
            ),
            "data_combination": "BBN + CMB(theta*) + DESI DR2 + absolute ET GW",
            "reference_model": "flat LCDM, w_i=-1",
            "mode_basis": (
                "descending eigenvectors of F_data = inv(C_A001) - 0.01 P_Horndeski"
            ),
            "method": {
                "posterior_precision_m": "Q_m=E_m^T(F_data+P_Horndeski)E_m",
                "score_m": "j_m=E_m^T inv(C_A001) <w+1>_A001",
                "posterior_mean_m": "mu_m=inv(Q_m)j_m",
                "snr": "sqrt(mu_m^T Q_m mu_m)",
                "delta_ln_L": "SNR_m^2/2",
                "delta_ln_V": "[ln det(P_m)-ln det(Q_m)]/2",
                "ln_bayes_factor": "delta_ln_L+delta_ln_V",
                "ln_bayes_factor_sigma": (
                    "dynesty log-evidence integration error for the compressed "
                    f"Gaussian model (nlive={EVIDENCE_NLIVE}, dlogz={EVIDENCE_DLOGZ})"
                ),
            },
            "sign_convention": "positive ln_B_m_over_LCDM favors the m-mode w(z) model",
            "approximation": (
                "Analytic Gaussian evidence compressed from converged A=0.01 MCMC "
                "chains; sigma is the nested-sampling integration error of this "
                "compressed model, not a data/model uncertainty"
            ),
            "source_files": {
                str(n_gw): [
                    str(path.relative_to(ROOT)) for path in source_paths(n_gw)
                ]
                for n_gw in EVENT_COUNTS
            },
            "results": rows,
        }, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("m  " + "  ".join(f"GW{n}: SNR / lnB" for n in EVENT_COUNTS))
    for wide in wide_rows:
        fields = [f"{int(wide['m']):d}"]
        for n_gw in EVENT_COUNTS:
            fields.append(
                f"{float(wide[f'SNR_N{n_gw:04d}']):.4f} / "
                f"{float(wide[f'lnB_N{n_gw:04d}']):+.4f} "
                f"± {float(wide[f'lnB_sigma_N{n_gw:04d}']):.4f}"
            )
        print("  ".join(fields))
    print(f"Wrote {long_path}")
    print(f"Wrote {wide_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
