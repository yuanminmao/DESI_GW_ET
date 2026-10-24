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
python mcmc/gw_mcmc.py
python mcmc/desi_bbn_cmb_gw_cpl_mcmc.py
```

Fast deterministic model-comparison summaries:

```bash
python mcmc/compute_gw_map_wilks_by_n.py
python mcmc/compute_joint_map_wilks_by_n.py
```

After the CPL chains exist, regenerate the contour figure:

```bash
python plot/plot_gw_w0_wa_4.py --no-show
```

## 4. Nonparametric reconstruction

Production A=1 joint reconstructions:

```bash
mpiexec -n 4 python mcmc/reconstruct_wz_bbn_cmb_desi_absolute_gw.py --run
python mcmc/reconstruct_wz_bbn_cmb_desi_absolute_gw.py --postprocess-only
python mcmc/compute_best_fit_w_four_gw.py
```

GW-only A=1 and weak-prior runs:

```bash
python mcmc/run_four_gw_prior_mcmc.py --help
mpiexec -n 4 python mcmc/sample_wz_gw_prior.py --run --n-events 3000
mpiexec -n 4 python mcmc/sample_wz_gw_prior.py --run --n-events 3000 --prior-amplitude 0.01
```

Repeat the last two commands for 100, 500, and 1000 events. Production runs
must satisfy the convergence checks built into the post-processing scripts.

## 5. PCA products and evidence

Joint weak-prior chains and PCA:

```bash
mpiexec -n 8 python mcmc/sample_pca_weak_prior.py --run
python mcmc/sample_pca_weak_prior.py --postprocess-only
python mcmc/analyze_pca_bbn_cmb_desi_gw.py
python mcmc/compute_pca_snr_bayes_by_m.py
```

GW-only PCA:

```bash
python mcmc/analyze_pca_gw_prior_only.py
python mcmc/compute_gw_prior_only_pca_snr_bayes_by_m.py
```

The stable local-Fisher calculation is an independent numerical diagnostic:

```bash
python mcmc/compute_stable_local_pca_fishers.py
```

## 6. Figures and manuscript

```bash
make figures
make sync-figures
make paper
make submission
```

`make figures` deliberately excludes the CPL contour plot because its chains
are too large for GitHub. The committed final contour plot keeps the manuscript
self-contained.
