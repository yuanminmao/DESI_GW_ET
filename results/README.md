# Results

`catalogs/` contains the four mock ET standard-siren catalogues used by the
paper. `tables/` contains compact posterior, PCA, evidence, and Wilks summary
products needed to reproduce the published numerical results and six figures.

Raw MCMC chains are intentionally not committed. They are written under
`results/chains/`, ignored by Git, and should be deposited in a dedicated
research-data archive if public chain-level reproducibility is required.

Regenerated plots are written to `results/figures/`. Run `make sync-figures`
to copy publication outputs into the self-contained `paper/figures/` tree.
