"""
Bayesian GW standard-siren inference (CPL: H0, Omega_m, w0, wa) with flat
priors, sampled with Cobaya's Metropolis--Hastings MCMC sampler.

  H0       ~ U(20, 100)
  omegam   ~ U(0.01, 0.99)
  w0       ~ U(-3, 3)
  wa       ~ U(-30, 3)

By default, this script analyzes the four complete catalogs available under
``results/catalogs``: 100, 500, 1000, and 3000.  It writes the retained
posterior samples as CosmoMC-style chains for separate downstream analysis.

  python mcmc/gw_mcmc.py
  python mcmc/gw_mcmc.py --n-events 1000 3000
  python mcmc/gw_mcmc.py --catalog path/to/catalog.txt --chains path/to/output
  python mcmc/gw_mcmc.py --quick
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import Callable

import numpy as np
from cobaya.log import LoggedError
from cobaya.run import run as cobaya_run
from scipy.integrate import cumulative_trapezoid

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paths import setup_sys_path, CATALOGS, chain_root

setup_sys_path()

from gw_cosmology import C_KM_S, CPLCosmology, e_squared_cpl

DEFAULT_N_EVENTS = (100, 500, 1000, 3000)
DEFAULT_REALIZATION = 42
CATALOG_ROOT = CATALOGS

PRIOR_BOUNDS = {
    "H0": (20.0, 100.0),
    "omegam": (0.01, 0.99),
    "w0": (-3.0, 3.0),
    "wa": (-30.0, 3.0),
}
PARAM_NAMES = list(PRIOR_BOUNDS.keys())

N_CHAINS = 6            # 或 8
N_SAMPLES = 12000       # accepted samples per Cobaya chain
BURN_IN_FRAC = 0.3
RMINUS1_MAX = 0.01
RNG_SEED = 42


def flat_log_prior(theta: np.ndarray) -> float:
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (len(PRIOR_BOUNDS),) or not np.all(np.isfinite(theta)):
        return -np.inf
    for val, (lo, hi) in zip(theta, PRIOR_BOUNDS.values()):
        if not (lo <= val <= hi):
            return -np.inf
    return 0.0


def gelman_rubin_rminus1(
    chains: list[np.ndarray],
    weights: list[np.ndarray] | None = None,
) -> tuple[float, np.ndarray]:
    """Return the conventional between-chain Gelman--Rubin R-1 diagnostic.

    ``weights`` are optional positive frequency weights per chain (e.g. Cobaya
    Metropolis--Hastings repeat counts); omit them for unweighted samples.
    Chains of unequal length are truncated to their common minimum length.
    """
    if len(chains) < 2:
        raise ValueError("Gelman--Rubin requires at least two chains")
    if any(chain.ndim != 2 for chain in chains):
        raise ValueError("each chain must be a two-dimensional array")
    if len({chain.shape[1] for chain in chains}) != 1:
        raise ValueError("chains must have matching parameter counts")
    n_min = min(chain.shape[0] for chain in chains)
    if n_min < 2:
        raise ValueError("chains must have at least two samples")
    chains = [chain[-n_min:] for chain in chains]
    if weights is None:
        weight_arrays = [np.ones(n_min) for _ in chains]
    else:
        if len(weights) != len(chains):
            raise ValueError("weights must have one array per chain")
        weight_arrays = [np.asarray(w, dtype=float)[-n_min:] for w in weights]
    n_chains = len(chains)
    means = np.array(
        [np.average(c, axis=0, weights=w) for c, w in zip(chains, weight_arrays)]
    )
    eff_n = np.array([w.sum() for w in weight_arrays])
    vars_within = np.array(
        [
            np.average((c - m) ** 2, axis=0, weights=w) * (n / (n - 1.0))
            for c, w, m, n in zip(chains, weight_arrays, means, eff_n)
        ]
    )
    n_avg = float(eff_n.mean())
    mean_overall = np.average(means, axis=0, weights=eff_n)
    b = n_avg * ((means - mean_overall) ** 2).sum(axis=0) / (n_chains - 1)
    w = np.average(vars_within, axis=0, weights=eff_n)
    var_hat = ((n_avg - 1.0) / n_avg) * w + b / n_avg
    r = np.sqrt(var_hat / np.maximum(w, 1e-30))
    return float(np.max(r - 1.0)), r - 1.0


def run_cobaya_chain(
    log_likelihood: Callable[[np.ndarray], float],
    *,
    param_names: list[str],
    prior_bounds: dict[str, tuple[float, float]],
    output_path: Path,
    max_samples: int,
    seed: int,
    progress: bool = True,
    ref_point: np.ndarray | None = None,
    ref_scatter: float = 1.0,
    proposal_scale: float = 1.0 / 20.0,
    resume: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run one Cobaya MCMC chain; return (samples, loglikes, weights).

    ``ref_point``/``ref_scatter`` optionally narrow the initial-point
    distribution to a fraction of the prior width around an anchor; by
    default the initial point is drawn from the full prior box.
    ``proposal_scale`` sets the initial per-parameter proposal width as a
    fraction of the prior width; for very informative likelihoods this should
    be tightened to roughly the expected conditional posterior width.
    When ``resume`` is false, any existing products at ``output_path`` are
    replaced and the sampler always starts from a new initial point.
    """
    if max_samples < 1:
        raise ValueError("max_samples must be positive")

    def external_loglike(**kwargs: float) -> float:
        theta = np.array([kwargs[name] for name in param_names], dtype=float)
        value = float(log_likelihood(theta))
        return value if np.isfinite(value) else -1.0e30

    params = {}
    for index, name in enumerate(param_names):
        lo, hi = prior_bounds[name]
        if ref_point is None:
            ref_lo, ref_hi = lo, hi
        else:
            center = float(ref_point[index])
            half_width = 0.5 * ref_scatter * (hi - lo)
            ref_lo, ref_hi = center - half_width, center + half_width
            ref_lo, ref_hi = max(lo, ref_lo), min(hi, ref_hi)
            if ref_hi <= ref_lo:
                ref_lo, ref_hi = lo, hi
        params[name] = {
            "prior": {"min": lo, "max": hi},
            "ref": {"min": ref_lo, "max": ref_hi},
            "proposal": proposal_scale * (hi - lo),
        }
    info = {
        "params": params,
        "likelihood": {
            "loglike": {"external": external_loglike, "input_params": list(param_names)}
        },
        "sampler": {
            "mcmc": {
                "max_samples": int(max_samples),
                "seed": int(seed),
                "learn_proposal": True,
                # A generous budget of failed proposals before Cobaya gives up;
                # a mismatched initial proposal width should not be fatal.
                "max_tries": 10000,
                # Near-zero so the single-chain split convergence check never
                # stops a chain early; independent chains stay equal length.
                "Rminus1_stop": 0.01,
            }
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    debug = None if progress else logging.WARNING
    if not resume:
        _, sampler = cobaya_run(
            info, output=str(output_path), resume=False, force=True, debug=debug
        )
    else:
        try:
            _, sampler = cobaya_run(
                info, output=str(output_path), resume=True, force=False, debug=debug
            )
        except LoggedError as error:
            # A saved checkpoint from an incompatible (e.g. older) run
            # configuration cannot be resumed; discard it and restart.
            if "not compatible" not in str(error).lower():
                raise
            print(f"  {output_path}: stale checkpoint incompatible, restarting fresh.")
            _, sampler = cobaya_run(
                info, output=str(output_path), resume=False, force=True, debug=debug
            )
    data = sampler.products()["sample"].data
    samples = data[list(param_names)].to_numpy(dtype=float)
    loglikes = -0.5 * data["chi2"].to_numpy(dtype=float)
    weights = data["weight"].to_numpy(dtype=float)
    return samples, loglikes, weights


def write_cosmomc_chains(
    chain_samples: list[np.ndarray],
    log_probs: list[np.ndarray],
    chain_root: Path,
    *,
    param_names: list[str] | None = None,
    prior_bounds: dict[str, tuple[float, float]] | None = None,
    ranges_header: list[str] | None = None,
    weights: list[np.ndarray] | None = None,
) -> None:
    names = param_names or PARAM_NAMES
    bounds = prior_bounds or PRIOR_BOUNDS
    if len(chain_samples) != len(log_probs) or not chain_samples:
        raise ValueError("chain_samples and log_probs must be non-empty lists of equal length")
    if len(names) != len(bounds):
        raise ValueError("param_names and prior_bounds must have the same length")
    if weights is not None and len(weights) != len(chain_samples):
        raise ValueError("weights must have one array per chain")
    for index, (samples, lnp) in enumerate(zip(chain_samples, log_probs)):
        if samples.ndim != 2 or samples.shape[1] != len(names):
            raise ValueError("each sample array must have one column per parameter")
        if lnp.ndim != 1 or lnp.shape[0] != samples.shape[0]:
            raise ValueError("each log-probability array must match its sample array")
        if weights is not None and weights[index].shape != lnp.shape:
            raise ValueError("each weight array must match its sample array")
    chain_root.parent.mkdir(parents=True, exist_ok=True)
    for old in chain_root.parent.glob(f"{chain_root.name}_*.txt"):
        old.unlink()
    (chain_root.parent / f"{chain_root.name}.paramnames").write_text(
        "\n".join(names) + "\n"
    )
    if ranges_header is None:
        fixed_lines = [
            "# Flat priors: lower  upper",
            *(f"{name:<6} {lo:<7g} {hi:g}" for name, (lo, hi) in bounds.items()),
            "",
        ]
    else:
        fixed_lines = list(ranges_header) + [""]
    (chain_root.parent / f"{chain_root.name}.ranges").write_text("\n".join(fixed_lines))

    for i, (samples, lnp) in enumerate(zip(chain_samples, log_probs), start=1):
        wt = np.ones(samples.shape[0]) if weights is None else weights[i - 1]
        data = np.column_stack([wt, lnp, samples])
        np.savetxt(chain_root.parent / f"{chain_root.name}_{i}.txt", data, fmt="%.10e")


def load_gw_catalog(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load (z, dL_obs, sigma_dL) [Mpc] from mock GW catalog."""
    rows: list[list[float]] = []
    with path.open() as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 3:
                continue
            try:
                rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
            except ValueError:
                continue
    if not rows:
        raise ValueError(f"no numeric rows in {path}")
    data = np.asarray(rows, dtype=float)
    if not np.all(np.isfinite(data)):
        raise ValueError(f"catalog contains non-finite values: {path}")
    if np.any(data[:, 0] < 0.0):
        raise ValueError(f"catalog contains negative redshifts: {path}")
    if np.any(data[:, 2] <= 0.0):
        raise ValueError(f"catalog contains non-positive distance errors: {path}")
    return data[:, 0], data[:, 1], data[:, 2]


def make_log_probability(
    z: np.ndarray,
    d_obs: np.ndarray,
    sigma: np.ndarray,
    cosmo: CPLCosmology,
):
    """Build a numerically stable, fast CPL log likelihood.

    The background integral is evaluated on ``cosmo.z_grid`` and then linearly
    interpolated at the fixed catalog redshifts.  This avoids constructing a
    cubic spline at every MCMC proposal; with the 4000-point grid, the lookup
    error is negligible compared with the catalog distance errors.
    """
    z = np.asarray(z, dtype=float)
    d_obs = np.asarray(d_obs, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if z.ndim != 1 or d_obs.shape != z.shape or sigma.shape != z.shape:
        raise ValueError("z, d_obs, and sigma must be one-dimensional arrays of equal length")
    if z.size == 0 or np.any(~np.isfinite(np.concatenate((z, d_obs, sigma)))):
        raise ValueError("catalog arrays must be non-empty and finite")
    if np.any(z < 0.0) or np.any(sigma <= 0.0):
        raise ValueError("redshifts must be non-negative and errors must be positive")
    if float(np.max(z)) > cosmo.z_grid[-1]:
        raise ValueError("cosmology integration grid does not cover all catalog redshifts")

    inv_var = 1.0 / np.maximum(sigma**2, 1e-30)
    z_grid = cosmo.z_grid
    one_plus_z = 1.0 + z

    def log_prob(theta: np.ndarray) -> float:
        theta = np.asarray(theta, dtype=float)
        lp = flat_log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        h0, omega_m, w0, wa = theta
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            e_squared = e_squared_cpl(z_grid, omega_m, w0, wa)
            if not np.all(np.isfinite(e_squared)) or np.any(e_squared <= 0.0):
                return -np.inf
            chi_grid = (C_KM_S / h0) * cumulative_trapezoid(
                1.0 / np.sqrt(e_squared), z_grid, initial=0.0
            )
            d_mod = one_plus_z * np.interp(z, z_grid, chi_grid)
        if not np.all(np.isfinite(d_mod)):
            return -np.inf
        residual = d_obs - d_mod
        chi2 = float(np.dot(residual * inv_var, residual))
        if not np.isfinite(chi2):
            return -np.inf
        return lp - 0.5 * chi2

    return log_prob


def catalog_path_for(n_events: int, realization: int) -> Path:
    """Return the standard location of one complete GW mock catalog."""
    return CATALOG_ROOT / f"N{n_events:04d}" / f"realization_{realization:04d}.txt"


def run_catalog(
    *,
    catalog: Path,
    chains: Path,
    n_chains: int,
    max_samples: int,
    burn_in: float,
    rmax: float,
    seed: int,
    plot_hint: str,
    progress: bool,
    fresh: bool,
) -> None:
    """Run the CPL sampler for one catalog and write its CosmoMC-style chains."""
    if not catalog.is_file():
        raise FileNotFoundError(f"Catalog not found: {catalog}")

    z, d_obs, sigma = load_gw_catalog(catalog)
    if not 0.0 <= burn_in < 1.0:
        raise ValueError("burn_in must satisfy 0 <= burn_in < 1")
    if n_chains < 2:
        raise ValueError("n_chains must be at least two for Gelman--Rubin")
    z_max = float(np.max(z)) * 1.05
    cosmo = CPLCosmology(z_max=max(z_max, 2.5))
    log_likelihood = make_log_probability(z, d_obs, sigma, cosmo)

    print(f"GW catalog: {catalog}  ({z.size} events, z in [{z.min():.3f}, {z.max():.3f}])")
    print(f"Priors: {PRIOR_BOUNDS}")
    print(f"Running {n_chains} Cobaya MCMC chains x {max_samples} accepted samples ...")

    cobaya_root = chains.parent / f".{chains.name}_cobaya"
    if fresh and cobaya_root.is_dir():
        shutil.rmtree(cobaya_root)

    posterior: list[np.ndarray] = []
    posterior_loglikes: list[np.ndarray] = []
    posterior_weights: list[np.ndarray] = []
    for i in range(n_chains):
        chain_seed = seed + 17 * i
        print(f"  chain {i + 1}/{n_chains} (seed={chain_seed})")
        samples, loglikes, weights = run_cobaya_chain(
            log_likelihood,
            param_names=PARAM_NAMES,
            prior_bounds=PRIOR_BOUNDS,
            output_path=cobaya_root / f"chain_{i + 1}",
            max_samples=max_samples,
            seed=chain_seed,
            progress=progress,
        )
        burn = int(burn_in * samples.shape[0])
        posterior.append(samples[burn:])
        posterior_loglikes.append(loglikes[burn:])
        posterior_weights.append(weights[burn:])

    rmax_val, rvals = gelman_rubin_rminus1(posterior, posterior_weights)
    print(f"Gelman--Rubin R-1 (max): {rmax_val:.4f}  per param {dict(zip(PARAM_NAMES, rvals))}")
    if rmax_val > rmax:
        print(f"WARNING: R-1 > {rmax}; consider longer chains.")

    write_cosmomc_chains(posterior, posterior_loglikes, chains, weights=posterior_weights)
    print(f"Wrote {n_chains} chains under {chains.parent}")

    print(f"Run  {plot_hint}  for w0-wa FoM.")


def main(
    *,
    default_catalog: Path | None = None,
    default_chains: Path | None = None,
    plot_hint: str = "python plot/plot_gw_de_fom.py",
) -> None:
    ap = argparse.ArgumentParser(description="GW CPL MCMC (H0, Omega_m, w0, wa).")
    ap.add_argument(
        "--n-events", type=int, nargs="+", default=list(DEFAULT_N_EVENTS),
        help="Catalog sizes to analyze (default: 100 500 1000 3000). Ignored with --catalog.",
    )
    ap.add_argument(
        "--realization", type=int, default=DEFAULT_REALIZATION,
        help="Mock realization index for --n-events catalogs (default: %(default)s).",
    )
    ap.add_argument(
        "--catalog", type=Path, default=default_catalog,
        help="Run one explicitly specified catalog instead of the default four catalogs.",
    )
    ap.add_argument(
        "--chains", type=Path, default=default_chains,
        help="Output chain prefix for --catalog. Defaults to results/chains/GW_ET/GW_ET.",
    )
    ap.add_argument("--n-chains", type=int, default=N_CHAINS)
    ap.add_argument(
        "--nsteps", type=int, default=N_SAMPLES,
        help="Accepted samples per Cobaya MCMC chain (default: %(default)s).",
    )
    ap.add_argument("--burn-in", type=float, default=BURN_IN_FRAC)
    ap.add_argument("--rmax", type=float, default=RMINUS1_MAX)
    ap.add_argument("--seed", type=int, default=RNG_SEED)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument(
        "--fresh",
        action="store_true",
        help="Discard any resumable Cobaya chain state and restart from scratch.",
    )
    ap.add_argument(
        "--no-progress",
        action="store_true",
        help="Reduce Cobaya's per-chain console logging.",
    )
    args = ap.parse_args()

    n_chains = 2 if args.quick else int(args.n_chains)
    nsteps = 800 if args.quick else int(args.nsteps)

    if args.catalog is not None:
        catalog = args.catalog.resolve()
        chains = (args.chains or chain_root("GW_ET")).resolve()
        targets = [(catalog, chains, int(args.seed))]
    else:
        n_events_list = list(dict.fromkeys(int(n) for n in args.n_events))
        if any(n <= 0 for n in n_events_list):
            raise ValueError("--n-events values must be positive")
        if args.chains is not None:
            raise ValueError("--chains can only be used together with --catalog")
        targets = [
            (
                catalog_path_for(n, int(args.realization)).resolve(),
                chain_root(f"GW_ET_N{n:04d}").resolve(),
                int(args.seed) + 101 * i,
            )
            for i, n in enumerate(n_events_list)
        ]

    for index, (catalog, chains, seed) in enumerate(targets, start=1):
        if len(targets) > 1:
            print("=" * 78)
            print(f"Catalog {index}/{len(targets)}")
        run_catalog(
            catalog=catalog,
            chains=chains,
            n_chains=n_chains,
            max_samples=nsteps,
            burn_in=float(args.burn_in),
            rmax=float(args.rmax),
            seed=seed,
            plot_hint=plot_hint,
            progress=not args.no_progress,
            fresh=args.fresh,
        )


if __name__ == "__main__":
    main()
