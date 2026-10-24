#!/usr/bin/env python3
"""CPL MCMC for the joint DESI DR2 + BBN + CMB + ET-GW likelihood.

This is the joint-data counterpart of ``mcmc/gw_mcmc.py``.  For each ET
catalogue it samples

    (H0, Omega_m, omega_b, w0, wa)

with the absolute GW distances, the DESI DR2 BAO covariance, the BBN prior
on ``omega_b = Omega_b h^2``, and the compressed CMB acoustic scale.  The
same H0, Omega_m, w0 and wa are used by every likelihood term.

Run the four catalogue sizes and then make the comparison figure with::

    python3 mcmc/desi_bbn_cmb_gw_cpl_mcmc.py
    python3 plot/plot_gw_w0_wa_4.py --no-show

Useful shorter runs::

    python3 mcmc/desi_bbn_cmb_gw_cpl_mcmc.py --n-events 1000
    python3 mcmc/desi_bbn_cmb_gw_cpl_mcmc.py --quick --n-events 100
    python3 mcmc/desi_bbn_cmb_gw_cpl_mcmc.py --check

The quick mode is only a pipeline check; use the default chains for science.
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
import warnings
from pathlib import Path

# Apple's Accelerate backend can emit spurious matmul warnings even for a
# finite, explicitly validated residual vector. Non-finite likelihood values
# are checked below, so suppress only these backend-specific warnings.
warnings.filterwarnings(
    "ignore",
    message=r"(divide by zero|overflow|invalid value) encountered in matmul",
    category=RuntimeWarning,
)

import camb
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import cho_solve
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paths import chain_root, setup_sys_path

setup_sys_path()

from bbn_cmb_desi_absolute_gw_wz_model import (
    BBN_MEAN,
    BBN_SIGMA,
    C_KM_S,
    CMB_THETA100_MEAN,
    CMB_THETA100_SIGMA,
    OMEGA_NU_H2,
    Observations,
    load_observations,
)
from early_background_table import TableEarlyBackground
from gw_cosmology import de_integral_cpl
from gw_mcmc import (
    gelman_rubin_rminus1,
    run_cobaya_chain,
    write_cosmomc_chains,
)

DEFAULT_N_EVENTS = (100, 500, 1000, 3000)
PARAMETER_NAMES = ("H0", "omegam", "omega_b", "w0", "wa")
PRIOR_BOUNDS = {
    "H0": (20.0, 100.0),
    "omegam": (0.01, 0.99),
    "omega_b": (0.005, 0.040),
    "w0": (-3.0, 3.0),
    "wa": (-30.0, 3.0),
}
LOWER = np.asarray([PRIOR_BOUNDS[name][0] for name in PARAMETER_NAMES])
UPPER = np.asarray([PRIOR_BOUNDS[name][1] for name in PARAMETER_NAMES])

N_CHAINS = 6
N_SAMPLES = 12_000
BURN_IN_FRAC = 0.30
RMINUS1_MAX = 0.01
RNG_SEED = 20260929
EARLY_DE_FRACTION_MAX = 1.0e-4

# Scales used only by the deterministic initialization fit.
REFERENCE_POINT = np.asarray([67.4, 0.315, BBN_MEAN, -1.0, 0.0])
OPTIMIZER_SCALE = np.asarray([10.0, 0.10, 0.001, 0.5, 2.0])


def output_chain_root(n_events: int) -> Path:
    """CosmoMC chain prefix consumed by plot/plot_gw_w0_wa_4.py."""
    return chain_root(f"DESI_BBN_CMB_GW_ET_CPL_N{n_events:04d}")


class CPLJointLikelihood:
    """DESI DR2 + BBN + compressed CMB + absolute ET-GW CPL likelihood."""

    def __init__(self, data: Observations, early: TableEarlyBackground) -> None:
        self.data = data
        self.early = early
        self.u_grid = early.u_grid
        self.z_grid = early.z_grid
        self.one_plus_z = np.exp(self.u_grid)

    def components(self, point: np.ndarray) -> dict[str, float]:
        values = np.asarray(point, dtype=float)
        if values.shape != LOWER.shape or not np.all(np.isfinite(values)):
            raise ValueError("point must be finite and ordered H0, omegam, omega_b, w0, wa")
        if np.any(values < LOWER) or np.any(values > UPPER):
            raise ValueError("point is outside the flat parameter bounds")

        h0, omega_m, omega_b, w0, wa = values
        h = h0 / 100.0
        if omega_m * h * h <= omega_b + OMEGA_NU_H2:
            raise ValueError("point has non-positive cold-dark-matter density")

        dmstar_lcdm, rstar, rdrag, zstar, omega_de, hubble_lcdm = self.early.evaluate(
            h0, omega_m, omega_b, self.z_grid
        )
        if omega_de <= 0.0 or rstar <= 0.0 or rdrag <= 0.0:
            raise ValueError("invalid early-universe background")

        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            rho_de = np.exp(3.0 * de_integral_cpl(self.z_grid, w0, wa))
            e2_lcdm = np.square(hubble_lcdm / h0)
            e2 = e2_lcdm + omega_de * (rho_de - 1.0)
        if not np.all(np.isfinite(e2) & (e2 > 0.0)):
            raise ValueError("invalid CPL expansion history")

        e2_lcdm_star = float(np.interp(zstar, self.z_grid, e2_lcdm))
        rho_de_star = float(np.interp(zstar, self.z_grid, rho_de))
        early_de_fraction = abs(omega_de * (rho_de_star - 1.0)) / e2_lcdm_star
        if early_de_fraction > EARLY_DE_FRACTION_MAX:
            raise ValueError("CPL early dark energy would invalidate the LCDM sound horizon")

        dm_grid = (C_KM_S / h0) * cumulative_trapezoid(
            self.one_plus_z / np.sqrt(e2), self.u_grid, initial=0.0
        )

        dm_gw = np.interp(self.data.gw_z, self.z_grid, dm_grid)
        gw_model = (1.0 + self.data.gw_z) * dm_gw
        gw_residual = (self.data.gw_distance - gw_model) / self.data.gw_sigma
        with np.errstate(over="ignore", invalid="ignore"):
            gw_chi2 = float(gw_residual @ gw_residual)

        dm_bao = np.interp(self.data.bao_z, self.z_grid, dm_grid)
        e2_bao = np.interp(self.data.bao_z, self.z_grid, e2)
        dh_bao = C_KM_S / (h0 * np.sqrt(e2_bao))
        bao_model = np.empty(len(self.data.bao_z))
        for i, quantity in enumerate(self.data.bao_quantity):
            if quantity == "DM_over_rs":
                bao_model[i] = dm_bao[i] / rdrag
            elif quantity == "DH_over_rs":
                bao_model[i] = dh_bao[i] / rdrag
            elif quantity == "DV_over_rs":
                bao_model[i] = np.cbrt(
                    self.data.bao_z[i] * dm_bao[i] ** 2 * dh_bao[i]
                ) / rdrag
            else:  # guarded by load_observations, retained as a local invariant
                raise ValueError(f"unsupported BAO observable: {quantity}")
        bao_difference = bao_model - self.data.bao_value
        bao_chi2 = float(
            bao_difference @ cho_solve(self.data.bao_cov_factor, bao_difference)
        )

        # The sound horizon is evaluated in the LCDM early background.  Only
        # the late-time distance changes, and the early-DE cut above ensures
        # this compressed treatment is self-consistent.
        delta_integrand = self.one_plus_z * (
            1.0 / np.sqrt(e2) - 1.0 / np.sqrt(e2_lcdm)
        )
        delta_dm = (C_KM_S / h0) * cumulative_trapezoid(
            delta_integrand, self.u_grid, initial=0.0
        )
        dmstar = dmstar_lcdm + float(np.interp(zstar, self.z_grid, delta_dm))
        theta100 = 100.0 * rstar / dmstar
        cmb_chi2 = ((theta100 - CMB_THETA100_MEAN) / CMB_THETA100_SIGMA) ** 2
        bbn_chi2 = ((omega_b - BBN_MEAN) / BBN_SIGMA) ** 2

        result = {
            "GW_absolute": gw_chi2,
            "DESI_DR2_BAO": bao_chi2,
            "BBN_omega_b_h2": float(bbn_chi2),
            "CMB_theta_star": float(cmb_chi2),
            "theta100_model": float(theta100),
            "rdrag_Mpc": float(rdrag),
            "early_de_fraction_at_zstar": float(early_de_fraction),
        }
        if not all(math.isfinite(value) for value in result.values()):
            raise ValueError("non-finite joint likelihood component")
        return result

    def loglike(self, point: np.ndarray) -> float:
        try:
            components = self.components(point)
        except (ValueError, OverflowError, camb.baseconfig.CAMBError):
            return -math.inf
        return -0.5 * sum(
            components[key]
            for key in (
                "GW_absolute",
                "DESI_DR2_BAO",
                "BBN_omega_b_h2",
                "CMB_theta_star",
            )
        )


def find_reference_point(model: CPLJointLikelihood) -> np.ndarray:
    """Find a finite high-likelihood point used only to initialize MCMC."""
    scaled_bounds = list(zip((LOWER - REFERENCE_POINT) / OPTIMIZER_SCALE,
                             (UPPER - REFERENCE_POINT) / OPTIMIZER_SCALE))

    def objective(offset: np.ndarray) -> float:
        value = model.loglike(REFERENCE_POINT + OPTIMIZER_SCALE * offset)
        return -value if np.isfinite(value) else 1.0e30

    starts = (
        REFERENCE_POINT,
        np.asarray([57.4, 0.42, BBN_MEAN, -0.7, -1.0]),
        np.asarray([68.0, 0.30, BBN_MEAN, -0.8, -1.5]),
    )
    best_point = None
    best_value = math.inf
    for start in starts:
        offset = (start - REFERENCE_POINT) / OPTIMIZER_SCALE
        result = minimize(
            objective,
            offset,
            method="Powell",
            bounds=scaled_bounds,
            options={"maxiter": 1500, "xtol": 1.0e-5, "ftol": 1.0e-7},
        )
        if np.isfinite(result.fun) and result.fun < best_value:
            best_value = float(result.fun)
            best_point = REFERENCE_POINT + OPTIMIZER_SCALE * result.x
    if best_point is None or not np.isfinite(model.loglike(best_point)):
        raise RuntimeError("could not find a finite reference point for the joint MCMC")
    return best_point


def run_catalog(
    n_events: int,
    *,
    n_chains: int,
    max_samples: int,
    burn_in: float,
    rmax: float,
    seed: int,
    progress: bool,
    fresh: bool,
) -> None:
    output = output_chain_root(n_events).resolve()
    model = CPLJointLikelihood(load_observations(n_events), TableEarlyBackground())
    reference = find_reference_point(model)
    components = model.components(reference)
    print(f"N_GW={n_events}: reference {dict(zip(PARAMETER_NAMES, reference))}")
    print(
        "  chi2 components: "
        + ", ".join(
            f"{key}={components[key]:.3f}"
            for key in ("GW_absolute", "DESI_DR2_BAO", "BBN_omega_b_h2", "CMB_theta_star")
        )
    )

    cobaya_root = output.parent / f".{output.name}_cobaya"
    if fresh and cobaya_root.is_dir():
        shutil.rmtree(cobaya_root)

    posterior: list[np.ndarray] = []
    posterior_loglikes: list[np.ndarray] = []
    posterior_weights: list[np.ndarray] = []
    for i in range(n_chains):
        chain_seed = seed + 17 * i
        print(f"  chain {i + 1}/{n_chains} (seed={chain_seed})", flush=True)
        samples, loglikes, weights = run_cobaya_chain(
            model.loglike,
            param_names=list(PARAMETER_NAMES),
            prior_bounds=PRIOR_BOUNDS,
            output_path=cobaya_root / f"chain_{i + 1}",
            max_samples=max_samples,
            seed=chain_seed,
            progress=progress,
            ref_point=reference,
            ref_scatter=0.04,
            proposal_scale=0.01,
        )
        burn = int(burn_in * samples.shape[0])
        posterior.append(samples[burn:])
        posterior_loglikes.append(loglikes[burn:])
        posterior_weights.append(weights[burn:])

    rmax_value, rvalues = gelman_rubin_rminus1(posterior, posterior_weights)
    print(
        f"N_GW={n_events}: Gelman--Rubin R-1(max)={rmax_value:.4f}; "
        f"per parameter={dict(zip(PARAMETER_NAMES, rvalues))}"
    )
    if rmax_value > rmax:
        print(f"WARNING: R-1 > {rmax}; resume or increase --nsteps before publication use.")
    write_cosmomc_chains(
        posterior,
        posterior_loglikes,
        output,
        param_names=list(PARAMETER_NAMES),
        prior_bounds=PRIOR_BOUNDS,
        weights=posterior_weights,
    )
    print(f"Wrote joint CPL chains: {output.parent}")


def check_inputs(n_events: list[int]) -> None:
    early = TableEarlyBackground()
    for count in n_events:
        model = CPLJointLikelihood(load_observations(count), early)
        reference = find_reference_point(model)
        components = model.components(reference)
        print(f"N_GW={count}: optimized finite reference")
        print("  " + ", ".join(f"{name}={value:.7g}" for name, value in zip(PARAMETER_NAMES, reference)))
        print(
            "  "
            + ", ".join(
                f"{key}={components[key]:.4f}"
                for key in ("GW_absolute", "DESI_DR2_BAO", "BBN_omega_b_h2", "CMB_theta_star")
            )
        )
    print("Joint CPL inputs and likelihood checks passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-events", type=int, nargs="+", default=list(DEFAULT_N_EVENTS),
                        choices=DEFAULT_N_EVENTS, help="ET catalogue sizes (default: all four)")
    parser.add_argument("--n-chains", type=int, default=N_CHAINS)
    parser.add_argument("--nsteps", type=int, default=N_SAMPLES,
                        help="accepted samples per chain (default: %(default)s)")
    parser.add_argument("--burn-in", type=float, default=BURN_IN_FRAC)
    parser.add_argument("--rmax", type=float, default=RMINUS1_MAX)
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    parser.add_argument("--quick", action="store_true",
                        help="pipeline check: two chains with 800 accepted samples each")
    parser.add_argument("--check", action="store_true",
                        help="validate and optimize reference points without running MCMC")
    parser.add_argument("--fresh", action="store_true",
                        help="discard resumable Cobaya state and restart selected cases")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    counts = list(dict.fromkeys(args.n_events))
    if len(counts) != len(args.n_events):
        parser.error("--n-events values must be distinct")
    if not 0.0 <= args.burn_in < 1.0:
        parser.error("--burn-in must satisfy 0 <= value < 1")
    if args.n_chains < 2 or args.nsteps < 1:
        parser.error("--n-chains must be at least two and --nsteps must be positive")
    if args.check:
        check_inputs(counts)
        return

    n_chains = 2 if args.quick else args.n_chains
    nsteps = 800 if args.quick else args.nsteps
    for i, count in enumerate(counts):
        print("=" * 78)
        print(f"Joint CPL catalogue {i + 1}/{len(counts)}: N_GW={count}")
        run_catalog(
            count,
            n_chains=n_chains,
            max_samples=nsteps,
            burn_in=args.burn_in,
            rmax=args.rmax,
            seed=args.seed + 101 * i,
            progress=not args.no_progress,
            fresh=args.fresh,
        )
    print("Run: python3 plot/plot_gw_w0_wa_4.py --no-show")


if __name__ == "__main__":
    main()
