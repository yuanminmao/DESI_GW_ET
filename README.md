# Testing Dynamical Dark-Energy Model with Standard Sirens

This repository contains the cleaned manuscript source, mock Einstein Telescope
(ET) bright-standard-siren catalogues, compact posterior products, and the
Python analysis used for the article **“Testing Dynamical Dark-Energy Model
with Standard Sirens.”**

The repository is intentionally compact. The final manuscript, its seven
figures, all four mock catalogues, and the numerical summary tables are
included. Multi-gigabyte MCMC chains and local build artefacts are excluded.

## Repository layout

```text
catalog/          Mock ET catalogue generation
cosmology/        Flat LCDM and CPL background models
mcmc/             MCMC models, sampling, and posterior reconstruction
analysis/    Numerical-product builders and manuscript-table exporter
plot/             Publication-figure scripts
data/             Small numerical inputs used by the analysis
results/catalogs/ Published mock catalogues (realization 0042)
results/tables/   Compact products grouped by analysis role and generating script
paper/            Final LaTeX source, five table fragments, figures, and PDF
tools/            Repository-maintenance and submission helpers
tests/            Fast integrity tests
```

## Quick start

Python 3.9 or newer is required. An MPI implementation is also required for
the formal multi-chain Cobaya runs.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[inference]'
make check
```

Compile the paper with a TeX distribution containing REVTeX 4.2:

```bash
make paper
```

`make paper` first refreshes the two computed table fragments from the CSV
products in `results/tables/`. Use `make tables` when only those fragments
need updating.

The compiled file is written to `build/paper/main.pdf`. The exact submitted
PDF is retained as `paper/manuscript.pdf`.

## Reproduce the compact-product figures

```bash
make figures
make sync-figures
make paper
```

This regenerates six figures directly from the included data, catalogues, and
summary tables. The CPL contour figure `gw_w0_wa_4.png` requires the excluded
CosmoMC-style chains; regenerate those first with the CPL sampling scripts.
See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the full workflow.

## Build a journal-source archive

```bash
make submission
```

This creates `dist/paper-source.zip` with `main.tex`, its five table fragments,
the inline bibliography, and only the figures referenced by the manuscript.
macOS metadata, caches, old drafts, presentation material, and analysis chains
are not included.

## Data and large products

The committed catalogues use random realization 42 and contain 100, 500,
1000, and 3000 detected events. Summary JSON files retain references to the
chain locations for provenance, but `results/chains/` is deliberately ignored
because the original chains total roughly 13 GB. For archival publication,
store those chains in a DOI-backed data repository rather than GitHub.

## Citation and license

Citation metadata is provided in `CITATION.cff`. No software or data license
has been selected in this cleaned copy; choose and add an appropriate license
before inviting third-party reuse.
