# Manuscript tables

These are the five table fragments included directly by `paper/main.tex`.

- `wz_bins.tex`, `desi_dr2_bao.tex`, and `parameter_priors.tex` document fixed
  analysis inputs and are edited as manuscript source.
- `gw_and_desi_bbn_cmb_gw_pca_snr_bayes.tex` and `gw_and_desi_bbn_cmb_gw_cpl_wilks.tex` are generated from
  the machine-readable CSV products in `results/tables/`.

Refresh the two computed tables with:

```bash
make tables
```

Do not edit the generated files by hand. `make check` verifies that their
values still agree with the analysis CSVs.
