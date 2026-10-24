"""
Generate a mock GW standard-siren catalog (z, dL_obs, sigma_dL):

- Fiducial cosmology: flat, with H0 = 57.365 km/s/Mpc, Omega_m = 0.423,
  and the fixed binned w(z) model in data/gw_fiducial_w_bins.txt.
- Redshift distribution P(z) ~ 4 pi dC^2 R(z) / [H(z)(1+z)], 0 < z < 5.
- SNR: matched-filter inner product with the ET-D noise PSD
  (data/ET-D.txt, total ASD column), inspiral waveform h(f) = A f^{-7/6},
  three ET interferometers combined in quadrature, f_lower = 1 Hz,
  f_upper = 2 f_LSO, network detection threshold rho > 8.
- BNS masses: per-event truncated Gaussian N(1.33, 0.09) M_sun
  (Galactic DNS population, Ozel & Freire 2016).
- sigma_dL = sqrt[(2 dL/rho)^2 + (0.05 z dL)^2];  dL_mock is drawn
  from N(dL_fid, sigma), with nonpositive distances redrawn.

  python catalog/gw_mock_catalog_ET.py
  -> extends seeds 42..46 from N=1000 to N=3000 in steps of 100.
     Each step draws only 100 new events. Prefixes are chosen in generation
     order, then separately sorted by redshift when saved.

  python catalog/gw_mock_catalog_ET.py --seeds 42 43 --base-n 1100 --max-n 1200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paths import CATALOGS, DATA, setup_sys_path

setup_sys_path()

REALIZATION_ROOT = CATALOGS

# --------------------------------------------------------------
C_KM_S = 299792.458  # km/s

H0 = 57.365
OMEGA_M = 0.423
OMEGA_DE = 1.0 - OMEGA_M  # flat

Z_MAX = 5        # model grid for distance curves
Z_SAMPLE_MAX = 5
N_GRID = 40000
DEFAULT_BASE_N = 1000
DEFAULT_MAX_N = 3000
DEFAULT_STEP = 100
DEFAULT_SEEDS = (42, 43, 44, 45, 46)
RNG_SEED = 42
MIN_MAX_DRAWS = 100_000
MAX_DRAWS_PER_EVENT = 2000

# --- ET-D SNR model -------------------------------------------
G_MSUN_S = 4.92549095e-6      # G M_sun / c^3  [s]
MPC_S = 1.02927125e14         # Mpc / c        [s]

# BNS component masses: Gaussian population from Galactic double NS
# statistics (Ozel & Freire 2016), as in the ET forecast catalogs.
MASS_MEAN = 1.33              # M_sun
MASS_STD = 0.09               # M_sun
MASS_MIN, MASS_MAX = 1.0, 2.0 # truncation range
F_LOWER = 1.0                 # Hz
SNR_THRESHOLD = 8.0           # network detection threshold
N_ET_IFOS = 3                 # three independent ET interferometers


# --------------------------------------------------------------
def _rho_de_ratio_binned(z, zp_lo, zp_hi, w_bins):
    """rho_de(z)/rho_de,0 for piecewise-constant w(z) bins."""
    seg = (1.0 + w_bins) * np.log((1 + zp_hi) / (1 + zp_lo))
    cum = np.cumsum(seg)
    z = np.atleast_1d(z).astype(float)
    rho = np.zeros_like(z)
    for k, zz in enumerate(z):
        if zz > zp_hi[-1]:
            integral = cum[-1]
        elif zz < zp_lo[0]:
            integral = ((1.0 + w_bins[0])* np.log((1.0 + zz) / (1.0 + zp_lo[0])))
        else:
            i = min(np.searchsorted(zp_hi, zz, side="left"),len(w_bins) - 1,)
            integral = cum[i - 1] if i > 0 else 0.0
            integral += ((1.0 + w_bins[i])* np.log((1.0 + zz) / (1.0 + zp_lo[i])))
        rho[k] = np.exp(3.0 * integral)
    return rho


def build_model():
    """
    Build distance / population functions on a z grid.

    Returns dict with: z_grid, dL_fid_curve [Mpc], dL_lcdm [Mpc],
    luminosity_distance(z), comoving_distance(z), E(z), P_z(z).
    """
    z_w = np.loadtxt(DATA / "gw_fiducial_w_bins.txt", comments="#")
    zp_lo, zp_hi, w_bestfit = z_w[:, 1], z_w[:, 2], z_w[:, 4]

    def E(z):
        return np.sqrt(
            OMEGA_M * (1 + np.asarray(z)) ** 3
            + OMEGA_DE * _rho_de_ratio_binned(z, zp_lo, zp_hi, w_bestfit)
        )

    z_grid = np.linspace(0.0, Z_MAX, N_GRID)

    chi_grid = cumulative_trapezoid(1.0 / E(z_grid), z_grid, initial=0)
    dC_grid = (C_KM_S / H0) * chi_grid
    dC_interp = interp1d(z_grid, dC_grid, kind="cubic",
                         bounds_error=False, fill_value="extrapolate")

    def comoving_distance(z):
        return dC_interp(z)

    def luminosity_distance(z):
        return (1 + np.asarray(z)) * comoving_distance(z)

    # LambdaCDM reference (w = -1) with the same H0 and Omega_m.
    Ez_lcdm = np.sqrt(OMEGA_M * (1 + z_grid) ** 3 + OMEGA_DE)
    chi_lcdm = cumulative_trapezoid(1.0 / Ez_lcdm, z_grid, initial=0)
    dL_lcdm = (1 + z_grid) * (C_KM_S / H0) * chi_lcdm

    def merger_rate(z):
        z = np.asarray(z)
        rate = np.zeros_like(z, dtype=float)
        mask1 = z <= 1
        mask2 = (z > 1) & (z < 5)
        rate[mask1] = 1 + 2 * z[mask1]
        rate[mask2] = 0.75 * (5 - z[mask2])
        return rate

    def P_z(z):
        dC = comoving_distance(z)
        return 4 * np.pi * dC**2 * merger_rate(z) / (E(z) * (1 + np.asarray(z)))

    return {
        "z_grid": z_grid,
        "dL_fid_curve": luminosity_distance(z_grid),
        "dL_lcdm": dL_lcdm,
        "E": E,
        "comoving_distance": comoving_distance,
        "luminosity_distance": luminosity_distance,
        "P_z": P_z,
    }


# --------------------------------------------------------------
# ET-D matched-filter SNR (rho, inner product, and f_LSO)
# --------------------------------------------------------------
def _load_noise_moment():
    """
    Build I(f_up) = 4 * int_{f_lower}^{f_up} f^{-7/3} / S_h(f) df from the
    ET-D sensitivity file (column 4 = total ASD, S_h = ASD^2), as a
    cumulative integral interpolated in f_up.
    """
    data = np.loadtxt(DATA / "ET-D.txt")
    if data.ndim != 2 or data.shape[1] < 4:
        raise ValueError(
            f"ET-D noise file must contain at least four columns: {DATA / 'ET-D.txt'}"
        )

    f = data[:, 0]
    asd = data[:, 3]

    mask = f >= F_LOWER
    f, asd = f[mask], asd[mask]
    if f.size < 2:
        raise ValueError(
            f"ET-D noise file has fewer than two samples above {F_LOWER} Hz"
        )
    if not np.all(np.isfinite(f)) or not np.all(np.isfinite(asd)):
        raise ValueError("ET-D frequencies and ASD values must be finite")
    if np.any(np.diff(f) <= 0):
        raise ValueError("ET-D frequencies must be strictly increasing")
    if np.any(asd <= 0):
        raise ValueError("ET-D ASD values must be positive")

    S_h = asd**2

    integrand = 4.0 * f ** (-7.0 / 3.0) / S_h
    I_cum = cumulative_trapezoid(integrand, f, initial=0.0)
    return interp1d(f, I_cum, bounds_error=False,
                    fill_value=(0.0, I_cum[-1]))


def _et_antenna_squared(theta, phi, psi, iota):
    """
    Sum over the three ET interferometers (60-degree arms, rotated by 120
    degrees) of F+^2 (1+cos^2 iota)^2 / 4 + Fx^2 cos^2 iota.
    """
    cos_t = np.cos(theta)
    plus_amp = (1.0 + np.cos(iota) ** 2) / 2.0
    cross_amp = np.cos(iota)

    total = np.zeros_like(np.asarray(theta, dtype=float))
    for j in range(N_ET_IFOS):
        ph = phi + 2.0 * np.pi * j / 3.0
        a = 0.5 * (1.0 + cos_t**2) * np.cos(2 * ph)
        b = cos_t * np.sin(2 * ph)
        f_plus = (np.sqrt(3) / 2) * (a * np.cos(2 * psi) - b * np.sin(2 * psi))
        f_cross = (np.sqrt(3) / 2) * (a * np.sin(2 * psi) + b * np.cos(2 * psi))
        total += (f_plus * plus_amp) ** 2 + (f_cross * cross_amp) ** 2
    return total


def sample_bns_masses(rng, n):
    """Component masses from a truncated Gaussian N(MASS_MEAN, MASS_STD)."""
    def draw(n_left):
        m = rng.normal(MASS_MEAN, MASS_STD, n_left)
        bad = (m < MASS_MIN) | (m > MASS_MAX)
        while np.any(bad):
            m[bad] = rng.normal(MASS_MEAN, MASS_STD, bad.sum())
            bad = (m < MASS_MIN) | (m > MASS_MAX)
        return m
    return draw(n), draw(n)


def snr_et_d(z, dL_mpc, m1, m2, theta, phi, psi, iota, noise_moment):
    """
    Network SNR of the 3 ET-D interferometers for the restricted PN
    inspiral waveform |h(f)| = A f^{-7/6}, with f_upper = 2 f_LSO.
    """
    z = np.asarray(z, dtype=float)
    m1 = np.asarray(m1, dtype=float)
    m2 = np.asarray(m2, dtype=float)

    m_total_s = (m1 + m2) * G_MSUN_S * (1.0 + z)            # observed total mass
    mc = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
    mc_obs = mc * G_MSUN_S * (1.0 + z)                      # observed chirp mass

    f_orb_lso = 1.0 / (6.0 ** 1.5 * 2.0 * np.pi * m_total_s)
    f_upper = 2.0 * f_orb_lso

    dL_s = np.asarray(dL_mpc, dtype=float) * MPC_S
    # Restricted-SPA normalization with
    # Q^2 = F+^2 (1 + cos^2(iota))^2 / 4 + Fx^2 cos^2(iota).
    amp2 = (
        _et_antenna_squared(theta, phi, psi, iota)
        * (5.0 / 24.0) * np.pi ** (-4.0 / 3.0)
        * mc_obs ** (5.0 / 3.0) / dL_s**2
    )
    return np.sqrt(amp2 * noise_moment(f_upper))


def write_catalog(
    out_catalog: Path,
    z_samples: np.ndarray,
    dL_obs: np.ndarray,
    sigma_dL: np.ndarray,
) -> None:
    out_catalog.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        out_catalog,
        np.column_stack([z_samples, dL_obs, sigma_dL]),
        header="z dL_obs[Mpc] sigma_dL[Mpc]",
        comments="",
        fmt="%.18e",
    )


def generate_catalog(
    model, n_events: int, seed=RNG_SEED, *, sort_by_redshift: bool = True
):
    """
    Sample z from P(z) over 0 < z < Z_SAMPLE_MAX, draw random sky/orientation
    angles per event, keep events with ET network SNR > SNR_THRESHOLD, and
    scatter dL_fid with instrument + lensing noise [Mpc]. With
    sort_by_redshift=False the three arrays remain in detection order, so
    nested samples can take a prefix without selecting low-redshift events.
    """
    if not isinstance(n_events, (int, np.integer)) or n_events <= 0:
        raise ValueError("n_events must be a positive integer")
    n_events = int(n_events)

    rng = np.random.default_rng(seed)
    z_grid = model["z_grid"]

    sample_mask = z_grid <= Z_SAMPLE_MAX
    z_s_grid = z_grid[sample_mask]
    if z_s_grid.size < 2:
        raise ValueError(
            "model z_grid must contain at least two samples below Z_SAMPLE_MAX"
        )

    pz = model["P_z"](z_s_grid)
    if np.any(~np.isfinite(pz)) or np.any(pz < 0):
        raise ValueError("redshift probability density must be finite and non-negative")
    trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    normalization = trapezoid(pz, z_s_grid)
    if not np.isfinite(normalization) or normalization <= 0:
        raise ValueError("redshift probability density has invalid normalization")
    pz /= normalization
    cdf = cumulative_trapezoid(pz, z_s_grid, initial=0)
    cdf /= cdf[-1]
    inv_cdf = interp1d(cdf, z_s_grid, bounds_error=False,
                       fill_value=(z_s_grid[0], z_s_grid[-1]))

    noise_moment = _load_noise_moment()

    z_kept, rho_kept = [], []
    n_drawn = 0
    max_draws = max(MIN_MAX_DRAWS, MAX_DRAWS_PER_EVENT * n_events)
    while len(z_kept) < n_events:
        draws_left = max_draws - n_drawn
        if draws_left <= 0:
            raise RuntimeError(
                f"ET detection sampling stopped after {n_drawn} draws: "
                f"kept {len(z_kept)} of {n_events} requested events"
            )
        n_draw = min(max(2 * (n_events - len(z_kept)), 100), draws_left)
        n_drawn += n_draw
        z_try = inv_cdf(rng.random(n_draw))
        dL_try = model["luminosity_distance"](z_try)

        m1, m2 = sample_bns_masses(rng, n_draw)
        theta = np.arccos(rng.uniform(-1, 1, n_draw))
        phi = rng.uniform(0, 2 * np.pi, n_draw)
        psi = rng.uniform(0, np.pi, n_draw)
        iota = np.arccos(rng.uniform(-1, 1, n_draw))

        rho = snr_et_d(z_try, dL_try, m1, m2, theta, phi, psi, iota, noise_moment)
        det = rho > SNR_THRESHOLD
        z_kept.extend(z_try[det])
        rho_kept.extend(rho[det])

    z_arr = np.asarray(z_kept[:n_events])
    rho_arr = np.asarray(rho_kept[:n_events])
    if sort_by_redshift:
        order = np.argsort(z_arr, kind="stable")
        z_samples = z_arr[order]
        rho = rho_arr[order]
    else:
        z_samples = z_arr
        rho = rho_arr

    dL_true = model["luminosity_distance"](z_samples)
    sigma_inst = 2.0 * dL_true / rho
    sigma_lens = 0.05 * z_samples * dL_true
    sigma_dL = np.sqrt(sigma_inst**2 + sigma_lens**2)

    dL_obs = rng.normal(dL_true, sigma_dL)
    # The catalogue likelihood requires positive observed distances. A rare
    # far-tail Gaussian draw can be negative; redraw only that observation,
    # leaving its redshift and uncertainty unchanged.
    invalid = dL_obs <= 0.0
    while np.any(invalid):
        dL_obs[invalid] = rng.normal(dL_true[invalid], sigma_dL[invalid])
        invalid = dL_obs <= 0.0
    return z_samples, dL_obs, sigma_dL


def write_gw_catalog(
    out_catalog: Path,
    *,
    n_events: int,
    seed: int = RNG_SEED,
    model: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if model is None:
        model = build_model()
    z_samples, dL_obs, sigma_dL = generate_catalog(model, n_events=n_events, seed=seed)
    write_catalog(out_catalog, z_samples, dL_obs, sigma_dL)
    return z_samples, dL_obs, sigma_dL


def catalog_path(root: Path, n_events: int, seed: int) -> Path:
    """Return the standard path for one catalog realization."""
    return root / f"N{n_events:04d}" / f"realization_{seed:04d}.txt"


def write_nested_catalogs(
    model,
    *,
    seed: int,
    base_n: int = DEFAULT_BASE_N,
    max_n: int = DEFAULT_MAX_N,
    step: int = DEFAULT_STEP,
    root: Path = REALIZATION_ROOT,
    overwrite_base: bool = False,
) -> list[Path]:
    """Retain the base rows and append one independently drawn block per step.

    Block streams depend on their absolute target size, so starting later at
    N=1100 produces the same N=1200 block as a run starting at N=1000.
    Arrays are extended in generation order; sorting happens only when the
    complete prefix for a target size is written to its catalogue file. The
    base is preserved unless overwrite_base is explicitly requested.
    """
    if seed < 0 or base_n <= 0 or step <= 0 or max_n < base_n:
        raise ValueError("seed must be non-negative and sizes/step must be positive")
    if (max_n - base_n) % step:
        raise ValueError("max_n - base_n must be divisible by step")

    from mcmc.gw_cpl_mcmc import load_gw_catalog

    root = Path(root).expanduser().resolve()
    base_path = catalog_path(root, base_n, seed)
    if base_path.is_file() and not overwrite_base:
        z, dL_obs, sigma_dL = load_gw_catalog(
            base_path, expected_n_events=base_n
        )
    else:
        z, dL_obs, sigma_dL = generate_catalog(
            model, base_n, seed=seed, sort_by_redshift=False
        )
        order = np.argsort(z, kind="stable")
        write_catalog(base_path, z[order], dL_obs[order], sigma_dL[order])

    paths = [base_path]
    for target_n in range(base_n + step, max_n + 1, step):
        # The base stream uses seed directly. A target-size child stream
        # prevents repeats and is unaffected by the requested final size.
        block_seed = np.random.SeedSequence(seed, spawn_key=(1, target_n))
        z_new, dL_new, sigma_new = generate_catalog(
            model, step, seed=block_seed, sort_by_redshift=False
        )
        z = np.concatenate((z, z_new))
        dL_obs = np.concatenate((dL_obs, dL_new))
        sigma_dL = np.concatenate((sigma_dL, sigma_new))
        if z.size != target_n:
            raise RuntimeError("nested GW catalogue has an unexpected event count")
        order = np.argsort(z, kind="stable")
        out_path = catalog_path(root, target_n, seed)
        write_catalog(out_path, z[order], dL_obs[order], sigma_dL[order])
        paths.append(out_path)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate nested ET mock GW catalogues by seed and size.",
    )
    ap.add_argument(
        "--base-n", type=int, default=DEFAULT_BASE_N,
        help="Existing base catalogue size (default: %(default)s).",
    )
    ap.add_argument("--max-n", type=int, default=DEFAULT_MAX_N)
    ap.add_argument("--step", type=int, default=DEFAULT_STEP)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    ap.add_argument(
        "--overwrite-base-seeds", type=int, nargs="*", default=[],
        help="Regenerate these seeds' base catalogues before extending them.",
    )
    ap.add_argument(
        "--root", type=Path, default=REALIZATION_ROOT,
        help="Root directory for standard Nxxxx/realization_yyyy catalogs.",
    )
    args = ap.parse_args()

    seeds = list(dict.fromkeys(args.seeds))
    if not seeds or any(seed < 0 for seed in seeds):
        ap.error("--seeds must contain non-negative integers")
    if args.base_n <= 0 or args.step <= 0 or args.max_n < args.base_n:
        ap.error("--base-n and --step must be positive; --max-n >= --base-n")
    if (args.max_n - args.base_n) % args.step:
        ap.error("--max-n minus --base-n must be divisible by --step")
    if any(seed not in seeds for seed in args.overwrite_base_seeds):
        ap.error("--overwrite-base-seeds must be included in --seeds")

    model = build_model()
    for seed in seeds:
        paths = write_nested_catalogs(
            model, seed=seed, base_n=args.base_n, max_n=args.max_n,
            step=args.step, root=args.root,
            overwrite_base=seed in args.overwrite_base_seeds,
        )
        print(f"seed {seed}: saved {len(paths)} catalogues, {paths[0]} .. {paths[-1]}")


if __name__ == "__main__":
    main()
