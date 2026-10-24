# Reproducibility workflow

Run every command from the repository root. The random catalogue realization
used by the paper is already committed, so reproducing the plots from compact
products does not rerun catalogue generation or MCMC.

## 1. Validate the repository

```bash
make check
```

The check compiles all Python sources, validates JSON metadata, verifies the
catalogue sizes, confirms that every manuscript figure exists, and rejects
personal absolute paths in tracked text files.

Raw chains are grouped under `results/chains/<mcmc-script-name>/`, with
`GWxxxx` subdirectories for catalogue size and, where needed, `A1`/`A001`
subdirectories for the correlation-prior amplitude.

## 2. Regenerate mock catalogues (optional)

The generator writes beneath `results/catalogs/` and uses deterministic seeds.
Its default mode extends existing 1000-event catalogues to 3000 events.

```bash
python catalog/gw_mock_catalog_ET.py --help
```

Do not replace the committed realization unless you intend to change the
paper's numerical results.

## 3. CPL inference and model comparison

GW-only and joint-data chains:

```bash
python mcmc/gw_cpl_mcmc.py
python mcmc/desi_bbn_cmb_gw_cpl_mcmc.py
```

Fast deterministic model-comparison summaries:

```bash
python analysis/cpl_lcdm_comparison/gw_cpl_wilks.py
python analysis/cpl_lcdm_comparison/desi_bbn_cmb_gw_cpl_wilks.py
```

After the CPL chains exist, regenerate the contour figure:

```bash
python plot/plot_gw_w0_wa_4.py
```

## 4. Nonparametric reconstruction

Production A=1 joint reconstructions:

```bash
mpiexec -n 4 python mcmc/reconstruct_wz_desi_bbn_cmb_gw.py --run
python mcmc/reconstruct_wz_desi_bbn_cmb_gw.py --postprocess-only
python analysis/posterior/compute_best_fit_w.py
```

GW-only A=1 and weak-prior runs:

```bash
python mcmc/reconstruct_wz_gw.py --run-four --help
mpiexec -n 4 python mcmc/reconstruct_wz_gw.py --run --n-events 3000
mpiexec -n 4 python mcmc/gw_pca_001.py --run --n-events 3000
```

Repeat the last two commands for 100, 500, and 1000 events. Production runs
must satisfy the convergence checks built into the post-processing scripts.

## 5. PCA products and evidence

Joint weak-prior chains and PCA:

```bash
mpiexec -n 8 python mcmc/desi_bbn_cmb_gw_pca_001.py --run
python mcmc/desi_bbn_cmb_gw_pca_001.py --postprocess-only
python analysis/pca/desi_bbn_cmb_gw_pca.py
python analysis/pca/desi_bbn_cmb_gw_snr_bayes.py
```

GW-only PCA:

```bash
python analysis/pca/gw_pca.py
python analysis/pca/gw_snr_bayes.py
```

## 6. Figures and manuscript

```bash
make tables
make figures
make sync-figures
make paper
make submission
```

`make tables` exports the combined PCA/evidence and Wilks LaTeX fragments from
the machine-readable CSV products. The manuscript includes all five table
fragments from `paper/tables/`; `make check` detects stale computed fragments.

`make figures` deliberately excludes the CPL contour plot because its chains
are too large for GitHub. The committed final contour plot keeps the manuscript
self-contained.
