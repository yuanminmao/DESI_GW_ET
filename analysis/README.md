# Analysis workflow

The scripts in this package convert reconstruction products and MCMC chains
into compact numerical tables.  Plotting is deliberately kept in `plot/`.

## Tables used by the manuscript

`tools/build_paper_tables.py` is the single publication-table exporter. It
reads the PCA and Wilks CSV products below and writes the two computed LaTeX
fragments included by `paper/main.tex` to `paper/tables/`. Run it with
`--check` to detect stale manuscript values without changing files.

The other three manuscript tables describe fixed inputs and priors, so they
live directly in `paper/tables/` and are not analysis results.

## Posterior summaries

- `posterior/summarize_gw_posterior.py` reads the four GW-only CPL chains and
  writes weighted posterior moments to
  `results/tables/posterior/summarize_gw_posterior/`.
- `posterior/compute_best_fit_w.py` refines the joint 29-bin reconstruction
  from the best retained chain samples and writes its CSV/JSON products to
  `results/tables/posterior/compute_best_fit_w/`.

## Model comparison

- `cpl_lcdm_comparison/gw_cpl_wilks.py` compares GW-only CPL and LCDM.
- `cpl_lcdm_comparison/desi_bbn_cmb_gw_cpl_wilks.py` performs the same comparison
  for DESI DR2 + BBN + CMB + GW.

Each script writes to its matching directory:
`results/tables/cpl_lcdm_comparison/gw_cpl_wilks/` and
`results/tables/cpl_lcdm_comparison/desi_bbn_cmb_gw_cpl_wilks/`.

## Principal-component analysis

- `pca/desi_bbn_cmb_gw_pca.py` builds joint-data Fisher, eigenvalue, mode,
  amplitude, and effective-degree-of-freedom tables.
- `pca/gw_pca.py` builds the corresponding GW-only tables.
- `pca/desi_bbn_cmb_gw_snr_bayes.py` and
  `pca/gw_snr_bayes.py` calculate truncated-mode SNR and Gaussian
  Bayes factors. Their wide CSV products feed the combined manuscript table.

Each PCA script writes to a same-named directory under `results/tables/pca/`,
for example `gw_pca/`, `desi_bbn_cmb_gw_pca/`, `gw_snr_bayes/`, and
`desi_bbn_cmb_gw_snr_bayes/`.

## Upstream reconstruction products

The table builders do not own their MCMC reconstruction inputs:

- joint A=1 products live in
  `posterior/reconstruct_wz_desi_bbn_cmb_gw/`, while joint A=0.01 products
  live in `pca/desi_bbn_cmb_gw_pca_001/`;
- GW-only A=1 products live in `posterior/reconstruct_wz_gw/`, while GW-only
  A=0.01 products live in `pca/gw_pca_001/`.

The three top-level categories are `cpl_lcdm_comparison`, `pca`, and
`posterior`. Every leaf results directory is named exactly after its generating
Python file. Shared locations are defined in `paths.py`; new scripts should
import those constants instead of constructing `results/tables` paths directly.

`results/tables/` contains machine-readable analysis products only. Final
LaTeX fragments belong in `paper/tables/`.

File formats follow one project-wide rule:

- JSON stores structured analysis summaries, metadata, fit statistics,
  Bayes factors, PCA statistics, and confidence intervals;
- CSV stores rectangular numerical tables and plotting sequences, including
  covariance/Fisher matrices and eigenvalue series;
- TXT is reserved for sampler or third-party formats such as Cobaya/GetDist
  chains and is not used for analysis tables.
