PYTHON ?= python3
MPLCONFIGDIR ?= $(CURDIR)/.cache/matplotlib
PYTHONPYCACHEPREFIX ?= $(CURDIR)/.cache/pycache
export MPLCONFIGDIR
export PYTHONPYCACHEPREFIX

.PHONY: help install check tables figures sync-figures paper submission

help:
	@echo "install       Install the full Python environment"
	@echo "check         Run syntax and repository-integrity checks"
	@echo "tables        Refresh computed LaTeX tables used by paper/main.tex"
	@echo "figures       Regenerate the six figures that use compact products"
	@echo "sync-figures  Copy regenerated figures into paper/figures"
	@echo "paper         Compile paper/main.tex into build/paper/main.pdf"
	@echo "submission    Build dist/paper-source.zip"

install:
	$(PYTHON) -m pip install -e '.[inference]'

check:
	$(PYTHON) -m compileall -q catalog cosmology mcmc plot analysis paths.py tools tests
	$(PYTHON) tools/build_paper_tables.py --check
	$(PYTHON) -m unittest discover -s tests -v

tables:
	$(PYTHON) tools/build_paper_tables.py

figures:
	$(PYTHON) plot/plot_wz_posterior_cov.py
	$(PYTHON) plot/plot_sensitivity.py
	$(PYTHON) plot/plot_gw_mock_catalog_ET.py
	$(PYTHON) plot/plot_wz_posterior_cov_gw_only_vs_joint_8panel.py
	$(PYTHON) plot/plot_pca_spectrum_gw_only_vs_joint.py
	$(PYTHON) plot/plot_wz_pca_modes_gw_only_vs_joint_8panel.py

sync-figures:
	$(PYTHON) tools/sync_paper_figures.py

paper: tables
	mkdir -p build/paper
	cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=../build/paper main.tex

submission: tables
	$(PYTHON) tools/build_submission.py
