PYTHON ?= python3
MPLCONFIGDIR ?= $(CURDIR)/.cache/matplotlib
PYTHONPYCACHEPREFIX ?= $(CURDIR)/.cache/pycache
export MPLCONFIGDIR
export PYTHONPYCACHEPREFIX

.PHONY: help install check figures sync-figures paper submission

help:
	@echo "install       Install the full Python environment"
	@echo "check         Run syntax and repository-integrity checks"
	@echo "figures       Regenerate the six figures that use compact products"
	@echo "sync-figures  Copy regenerated figures into paper/figures"
	@echo "paper         Compile paper/main.tex into build/paper/main.pdf"
	@echo "submission    Build dist/paper-source.zip"

install:
	$(PYTHON) -m pip install -e '.[inference]'

check:
	$(PYTHON) -m compileall -q catalog cosmology mcmc plot paths.py tools tests
	$(PYTHON) -m unittest discover -s tests -v

figures:
	$(PYTHON) plot/plot_wz_posterior_cov.py --no-show
	$(PYTHON) plot/plot_sensitivity.py --no-show
	$(PYTHON) plot/plot_gw_mock_catalog_ET.py --no-show
	$(PYTHON) plot/plot_wz_posterior_cov_gw_only_vs_joint_8panel.py
	$(PYTHON) plot/plot_pca_spectrum_gw_only_vs_joint.py
	$(PYTHON) plot/plot_wz_pca_modes_gw_only_vs_joint_8panel.py

sync-figures:
	$(PYTHON) tools/sync_paper_figures.py

paper:
	mkdir -p build/paper
	cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=../build/paper main.tex

submission:
	$(PYTHON) tools/build_submission.py
