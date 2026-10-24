# Results

`catalogs/` contains the four mock ET standard-siren catalogues used by the
paper. `tables/` contains compact posterior, PCA, evidence, and Wilks summary
products needed to reproduce the published numerical results and six figures.

Raw MCMC chains are intentionally not committed. They are written under
`results/chains/`, ignored by Git, and should be deposited in a dedicated
research-data archive if public chain-level reproducibility is required.

Chain directories mirror the generating file in `mcmc/`:

```text
chains/
├── gw_cpl_mcmc/GWxxxx/
├── desi_bbn_cmb_gw_cpl_mcmc/GWxxxx/
├── reconstruct_wz_desi_bbn_cmb_gw/GWxxxx/
├── desi_bbn_cmb_gw_pca_001/GWxxxx/
├── reconstruct_wz_gw/GWxxxx/
└── gw_pca_001/GWxxxx/
```

`GWxxxx` identifies the catalogue size. `reconstruct_wz_gw` uses the production
correlation prior A=1; `gw_pca_001` is the weak-prior A=0.01 auxiliary run used
for PCA. Both use a self-contained diagonal proposal around the injected
history; no auxiliary Laplace-chain directory is required.

Result formats are standardized: JSON contains structured analysis summaries,
CSV contains rectangular numerical tables and plotting sequences, and TXT is
reserved for sampler/third-party formats such as the chains above. Consequently,
`results/tables/` contains no TXT products.

Compact numerical products under `results/tables/` retain three analysis
categories. Inside each category, every leaf directory is named after its
generating Python file without `.py`:

```text
tables/
├── cpl_lcdm_comparison/
│   ├── gw_cpl_wilks/
│   ├── desi_bbn_cmb_gw_cpl_wilks/
│   └── plot_gw_w0_wa_4/
├── pca/
│   ├── gw_pca/
│   ├── desi_bbn_cmb_gw_pca/
│   ├── gw_pca_001/
│   ├── desi_bbn_cmb_gw_pca_001/
│   ├── gw_snr_bayes/
│   └── desi_bbn_cmb_gw_snr_bayes/
└── posterior/
    ├── reconstruct_wz_gw/
    ├── reconstruct_wz_desi_bbn_cmb_gw/
    ├── summarize_gw_posterior/
    └── compute_best_fit_w/
```

The LaTeX tables actually included by the manuscript are kept separately in
`paper/tables/`. The two computed fragments there are exported from the PCA
and model-comparison CSVs by `make tables`; `results/tables/` deliberately
contains no publication-layout TeX.

Regenerated plots are written to `results/figures/`. Run `make sync-figures`
to copy publication outputs into the self-contained `paper/figures/` tree.
