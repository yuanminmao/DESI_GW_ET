#!/usr/bin/env python3
"""Build the computed LaTeX tables included by ``paper/main.tex``.

The manuscript has five tables.  Three describe fixed inputs and priors and
are maintained directly under ``paper/tables``.  This script owns the other
two tables, whose displayed values come from machine-readable analysis
products under ``results/tables``:

* ``gw_and_desi_bbn_cmb_gw_pca_snr_bayes.tex`` from the GW-only and joint PCA wide CSVs;
* ``gw_and_desi_bbn_cmb_gw_cpl_wilks.tex`` from the GW-only and joint Wilks CSVs.

Run without arguments to refresh the committed TeX fragments.  Use
``--check`` in tests to verify that the manuscript tables are up to date
without changing files.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paths import (  # noqa: E402
    DESI_BBN_CMB_GW_CPL_WILKS_TABLES,
    DESI_BBN_CMB_GW_SNR_BAYES_TABLES,
    GW_CPL_WILKS_TABLES,
    GW_SNR_BAYES_TABLES,
    PAPER_TABLES,
)


EVENT_COUNTS = (100, 500, 1000, 3000)
MODES = tuple(range(1, 8))

GW_PCA = GW_SNR_BAYES_TABLES / "gw_pca_snr_bayes_by_m_wide.csv"
JOINT_PCA = (
    DESI_BBN_CMB_GW_SNR_BAYES_TABLES
    / "desi_bbn_cmb_gw_pca_snr_bayes_by_m_wide.csv"
)
GW_WILKS = GW_CPL_WILKS_TABLES / "gw_cpl_wilks.csv"
JOINT_WILKS = (
    DESI_BBN_CMB_GW_CPL_WILKS_TABLES / "desi_bbn_cmb_gw_cpl_wilks.csv"
)


def read_indexed_csv(path: Path, key: str, expected: tuple[int, ...]) -> dict[int, dict[str, str]]:
    """Read a CSV and require exactly one row for every expected key."""
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    indexed = {int(row[key]): row for row in rows}
    if len(indexed) != len(rows) or tuple(sorted(indexed)) != expected:
        raise ValueError(
            f"{path}: expected unique {key} values {expected}, got {tuple(sorted(indexed))}"
        )
    return indexed


def render_pca_table() -> str:
    gw = read_indexed_csv(GW_PCA, "m", MODES)
    joint = read_indexed_csv(JOINT_PCA, "m", MODES)
    lines = [
        r"\begin{table*}[t]",
        r"\caption{",
        r"Comparison of covariance-weighted reconstruction significance and Bayesian",
        r"evidence ratios for two analyses.",
        r"The quantities SNR and $\ln B_{m,\Lambda}$ are calculated using the first",
        r"$m$ principal components.",
        r"}",
        r"\label{tab:gw_and_desi_bbn_cmb_gw_pca_snr_bayes}",
        "",
        r"\centering",
        "",
    ]
    for section, rows in (("GW-only", gw), ("DESI+BBN+CMB+GW", joint)):
        lines.extend(
            [
                rf"\textbf{{{section}}}",
                "",
                r"\vspace{0.2cm}",
                "",
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
        )
        for mode in MODES:
            row = rows[mode]
            snr = [f"{float(row[f'SNR_N{count:04d}']):.3f}" for count in EVENT_COUNTS]
            lnb = [f"{float(row[f'lnB_N{count:04d}']):.3f}" for count in EVENT_COUNTS]
            lines.append(
                f"{mode} & " + " & ".join(snr) + " & " + " & ".join(lnb) + r"\\"
            )
        lines.extend([r"\end{tabular}", r"\end{ruledtabular}"])
        if section == "GW-only":
            lines.extend(["", "", r"\vspace{0.5cm}", "", ""])
    lines.extend([r"\end{table*}", ""])
    return "\n".join(lines)


def render_wilks_table() -> str:
    gw = read_indexed_csv(GW_WILKS, "n_gw", EVENT_COUNTS)
    joint = read_indexed_csv(JOINT_WILKS, "n_gw", EVENT_COUNTS)
    lines = [
        r"\begin{table*}[t!]",
        r"\caption{",
        r"Comparison of the Wilks-theorem model-comparison statistics for the",
        r"GW-only and DESI+BBN+CMB+GW analyses.",
        r"The significance $N_\sigma$ is obtained from $\Delta\chi^2$",
        r"assuming two additional degrees of freedom ($\Delta k=2$).",
        r"For the GW-only case, the CPL maximum-likelihood solutions for",
        r"100 and 500 events reach the prior boundaries, and therefore the",
        r"corresponding significances are not reported.",
        r"}",
        r"\label{tab:gw_and_desi_bbn_cmb_gw_cpl_wilks}",
        r"\centering",
        r"\small",
        r"\begin{ruledtabular}",
        r"\begin{tabular}{c|cc|cc}",
        r"& \multicolumn{2}{c|}{GW-only}",
        r"& \multicolumn{2}{c}{DESI+BBN+CMB+GW}",
        r"\\",
        r"$N_{\rm GW}$",
        r"& $\Delta\chi^2$",
        r"& $N_\sigma$",
        r"& $\Delta\chi^2$",
        r"& $N_\sigma$",
        r"\\",
        r"\hline",
    ]
    for count in EVENT_COUNTS:
        gw_row = gw[count]
        joint_row = joint[count]
        valid = gw_row["wilks_chi2_2dof_valid"].strip().lower() == "true"
        gw_sigma = f"{float(gw_row['n_sigma_1d_two_sided']):.3f}" if valid else "--"
        lines.extend(
            [
                str(count),
                rf"& {float(gw_row['delta_chi2_map']):.3f}",
                rf"& {gw_sigma}",
                rf"& {float(joint_row['delta_chi2_map']):.3f}",
                rf"& {float(joint_row['n_sigma_1d_two_sided']):.3f}",
                r"\\",
            ]
        )
    lines.extend(
        [r"\end{tabular}", r"\end{ruledtabular}", r"\end{table*}", ""]
    )
    return "\n".join(lines)


def products() -> dict[Path, str]:
    return {
        PAPER_TABLES / "gw_and_desi_bbn_cmb_gw_pca_snr_bayes.tex": render_pca_table(),
        PAPER_TABLES / "gw_and_desi_bbn_cmb_gw_cpl_wilks.tex": render_wilks_table(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if committed paper tables differ from the CSV products",
    )
    args = parser.parse_args()

    stale: list[Path] = []
    for path, content in products().items():
        if args.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != content:
                stale.append(path)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            print(f"Wrote {path.relative_to(ROOT)}")

    if stale:
        names = "\n".join(f"  {path.relative_to(ROOT)}" for path in stale)
        raise SystemExit(
            "Paper tables are missing or stale:\n"
            f"{names}\n"
            "Run: python tools/build_paper_tables.py"
        )


if __name__ == "__main__":
    main()
