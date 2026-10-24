"""Shared project paths (import from any module)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
PAPER = ROOT / "paper"
PAPER_TABLES = PAPER / "tables"
RESULTS = ROOT / "results"
CATALOGS = RESULTS / "catalogs"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"
CHAINS = ROOT / "results" / "chains"

# Raw sampling products are grouped by the script that owns them. Event-count
# variants live below these roots; distinct prior choices have distinct scripts.
GW_CPL_CHAINS = CHAINS / "gw_cpl_mcmc"
JOINT_CPL_CHAINS = CHAINS / "desi_bbn_cmb_gw_cpl_mcmc"
JOINT_WZ_CHAINS = CHAINS / "reconstruct_wz_desi_bbn_cmb_gw"
GW_WZ_CHAINS = CHAINS / "reconstruct_wz_gw"
GW_PCA_001_CHAINS = CHAINS / "gw_pca_001"
JOINT_PCA_CHAINS = CHAINS / "desi_bbn_cmb_gw_pca_001"

# Keep the three analysis categories, then name every leaf directory after the
# Python file that generates it. This gives each product one unambiguous owner.
CPL_LCDM_COMPARISON_TABLES = TABLES / "cpl_lcdm_comparison"
PCA_TABLES = TABLES / "pca"
POSTERIOR_TABLES = TABLES / "posterior"

RECONSTRUCT_WZ_GW_TABLES = POSTERIOR_TABLES / "reconstruct_wz_gw"
RECONSTRUCT_WZ_DESI_BBN_CMB_GW_TABLES = (
    POSTERIOR_TABLES / "reconstruct_wz_desi_bbn_cmb_gw"
)
SUMMARIZE_GW_POSTERIOR_TABLES = POSTERIOR_TABLES / "summarize_gw_posterior"
COMPUTE_BEST_FIT_W_TABLES = POSTERIOR_TABLES / "compute_best_fit_w"

GW_PCA_001_TABLES = PCA_TABLES / "gw_pca_001"
DESI_BBN_CMB_GW_PCA_001_TABLES = PCA_TABLES / "desi_bbn_cmb_gw_pca_001"
GW_PCA_TABLES = PCA_TABLES / "gw_pca"
DESI_BBN_CMB_GW_PCA_TABLES = PCA_TABLES / "desi_bbn_cmb_gw_pca"
GW_SNR_BAYES_TABLES = PCA_TABLES / "gw_snr_bayes"
DESI_BBN_CMB_GW_SNR_BAYES_TABLES = PCA_TABLES / "desi_bbn_cmb_gw_snr_bayes"

PLOT_GW_W0_WA_4_TABLES = CPL_LCDM_COMPARISON_TABLES / "plot_gw_w0_wa_4"
GW_CPL_WILKS_TABLES = CPL_LCDM_COMPARISON_TABLES / "gw_cpl_wilks"
DESI_BBN_CMB_GW_CPL_WILKS_TABLES = (
    CPL_LCDM_COMPARISON_TABLES / "desi_bbn_cmb_gw_cpl_wilks"
)
DESI_BBN_CMB_GW_CPL_MCMC_TABLES = (
    CPL_LCDM_COMPARISON_TABLES / "desi_bbn_cmb_gw_cpl_mcmc"
)


def gw_cpl_chain_root(n_events: int) -> Path:
    """GetDist/CosmoMC prefix for one GW-only CPL catalogue."""
    name = f"GW_ET_N{int(n_events):04d}"
    return GW_CPL_CHAINS / f"GW{int(n_events):04d}" / name


def joint_cpl_chain_root(n_events: int) -> Path:
    """GetDist/CosmoMC prefix for one joint-data CPL catalogue."""
    name = f"DESI_BBN_CMB_GW_ET_CPL_N{int(n_events):04d}"
    return JOINT_CPL_CHAINS / f"GW{int(n_events):04d}" / name


def project_path(path: Path) -> str:
    """Return a repository-relative path when possible, else an absolute path."""
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def ensure_result_dirs() -> None:
    for d in (
        CATALOGS,
        FIGURES,
        CHAINS,
        GW_CPL_CHAINS,
        JOINT_CPL_CHAINS,
        JOINT_WZ_CHAINS,
        GW_WZ_CHAINS,
        GW_PCA_001_CHAINS,
        JOINT_PCA_CHAINS,
        CPL_LCDM_COMPARISON_TABLES,
        PCA_TABLES,
        POSTERIOR_TABLES,
        RECONSTRUCT_WZ_GW_TABLES,
        GW_PCA_001_TABLES,
        RECONSTRUCT_WZ_DESI_BBN_CMB_GW_TABLES,
        DESI_BBN_CMB_GW_PCA_001_TABLES,
        SUMMARIZE_GW_POSTERIOR_TABLES,
        COMPUTE_BEST_FIT_W_TABLES,
        PLOT_GW_W0_WA_4_TABLES,
        GW_CPL_WILKS_TABLES,
        DESI_BBN_CMB_GW_CPL_WILKS_TABLES,
        GW_PCA_TABLES,
        DESI_BBN_CMB_GW_PCA_TABLES,
        GW_SNR_BAYES_TABLES,
        DESI_BBN_CMB_GW_SNR_BAYES_TABLES,
        DESI_BBN_CMB_GW_CPL_MCMC_TABLES,
        PAPER_TABLES,
    ):
        d.mkdir(parents=True, exist_ok=True)


def setup_sys_path() -> Path:
    """Add project packages to sys.path for flat imports."""
    for sub in (ROOT, ROOT / "mcmc", ROOT / "cosmology", ROOT / "catalog", ROOT / "plot"):
        s = str(sub)
        if s not in sys.path:
            sys.path.insert(0, s)
    return ROOT
