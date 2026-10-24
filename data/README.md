# Analysis inputs

- `ET-D.txt`: ET-D sensitivity curve used by the mock-catalogue generator.
- `desi_gaussian_bao_ALL_GCcomb_{mean,cov}.txt`: DESI DR2 Gaussian BAO data
  vector and covariance used by the joint likelihood.
- `gw_fiducial_w_bins.txt`: 29-bin fiducial dark-energy history injected into
  the mock standard-siren catalogues.
- `z_w.csv` and `cov_w_posterior.csv`: best-fitting binned history and
  covariance used for the reference reconstruction figure.
- `early_background_table.npz`: validated interpolation table for the early
  background/CMB acoustic-scale calculation used by expensive sampling runs.

CSV is used for project-owned rectangular tables and plotting sequences. TXT
is retained for sampler inputs and externally supplied formats, including the
ET-D curve and DESI Gaussian data files. These files are numerical inputs, not
executable instructions.
