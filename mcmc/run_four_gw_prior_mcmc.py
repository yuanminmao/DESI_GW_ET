#!/usr/bin/env python3
"""Run the four production GW-only + correlation-prior MCMC analyses.

This driver is deliberately restricted to the production prior amplitude
``A=1``.  It runs the four catalogues N_GW = 100, 500, 1000 and 3000 in
sequence.  The sampled parameters are H0, Omega_m and w1..w29, and the target
is exactly

    -2 log L = chi2_GW_absolute + chi2_Horndeski_prior(A=1).

DESI, CMB, BBN and omega_b are not used.  Converged products are skipped.
Existing native chains without an accepted summary are resumed automatically.

Examples
--------
Show the state of all four analyses without starting anything::

    python3 mcmc/run_four_gw_prior_mcmc.py --status-only

Run or resume all four analyses with four MPI ranks::

    python3 mcmc/run_four_gw_prior_mcmc.py

Print the commands without executing them::

    python3 mcmc/run_four_gw_prior_mcmc.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SAMPLER = ROOT / "mcmc" / "sample_wz_gw_prior.py"
TABLES = ROOT / "results" / "tables"
CHAIN_ROOT = ROOT / "results" / "chains" / "COBAYA_GW_PRIOR_wz"
DEFAULT_EVENTS = (100, 500, 1000, 3000)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--events", type=int, nargs="+", default=list(DEFAULT_EVENTS),
        help="Catalogue sizes to process (default: 100 500 1000 3000)",
    )
    parser.add_argument(
        "--mpi-processes", type=int, default=4,
        help="Independent MPI chains per catalogue (minimum and default: 4)",
    )
    parser.add_argument("--grid-size", type=int, default=3200)
    parser.add_argument(
        "--max-samples", type=int, default=1_000_000,
        help=(
            "Per-rank accepted-step ceiling. The high default prevents ranks "
            "from reaching unequal ceilings before convergence."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument(
        "--status-only", action="store_true",
        help="Report completion state without launching or changing anything",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without launching them",
    )
    args = parser.parse_args()
    args.events = list(dict.fromkeys(args.events))
    if not args.events or any(value not in DEFAULT_EVENTS for value in args.events):
        parser.error("--events may contain only 100, 500, 1000 and 3000")
    if args.mpi_processes < 4:
        parser.error("--mpi-processes must be at least 4")
    if args.grid_size < 1000 or args.max_samples < 1:
        parser.error("invalid grid size or sample ceiling")
    return args


def summary_path(n_events: int) -> Path:
    return TABLES / f"gw{n_events:04d}_prior_wz_mcmc_summary.json"


def native_chain_files(n_events: int) -> list[Path]:
    directory = CHAIN_ROOT / f"GW{n_events:04d}"
    return sorted(directory.glob("chain.[0-9]*.txt"))


def accepted_summary(n_events: int) -> tuple[bool, str]:
    path = summary_path(n_events)
    if not path.is_file():
        return False, "no accepted summary"
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
        amplitude = float(summary.get("prior_amplitude", 1.0))
        diagnostic = summary["rank_folded_split_rhat"]
        maximum = float(diagnostic["max_rhat_minus_one"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return False, f"invalid summary: {exc}"
    if amplitude != 1.0:
        return False, f"summary records prior amplitude A={amplitude:g}"
    if not diagnostic.get("all_pass") or maximum >= 0.01:
        return False, f"independent max Rhat-1={maximum:.6f} has not passed"
    draws = int(summary.get("total_retained_weighted_draws", 0))
    return True, f"accepted: max Rhat-1={maximum:.6f}, retained draws={draws}"


def sampling_command(n_events: int, args: argparse.Namespace) -> list[str]:
    command = [
        "mpiexec", "-n", str(args.mpi_processes), sys.executable,
        str(SAMPLER), "--run", "--n-events", str(n_events),
        "--prior-amplitude", "1", "--grid-size", str(args.grid_size),
        "--max-samples", str(args.max_samples), "--seed", str(args.seed),
    ]
    if native_chain_files(n_events):
        command.append("--resume")
    return command


def display_command(command: list[str]) -> str:
    return shlex.join(command)


def main() -> None:
    args = parse_args()
    states: dict[int, tuple[bool, str]] = {
        n_events: accepted_summary(n_events) for n_events in args.events
    }
    for n_events, (complete, detail) in states.items():
        label = "COMPLETE" if complete else "PENDING"
        print(f"GW{n_events:04d} A=1: {label} -- {detail}")
    if args.status_only:
        return

    if shutil.which("mpiexec") is None:
        raise SystemExit("mpiexec is not available on PATH")
    for n_events in args.events:
        complete, _ = accepted_summary(n_events)
        if complete:
            print(f"GW{n_events:04d} A=1: skip accepted posterior")
            continue
        command = sampling_command(n_events, args)
        print(f"GW{n_events:04d} A=1 command:\n  {display_command(command)}", flush=True)
        if args.dry_run:
            continue
        completed = subprocess.run(command, cwd=ROOT, check=False)
        if completed.returncode:
            raise SystemExit(
                f"GW{n_events:04d} A=1 stopped with exit code "
                f"{completed.returncode}; later catalogues were not launched"
            )
        accepted, detail = accepted_summary(n_events)
        if not accepted:
            raise SystemExit(
                f"GW{n_events:04d} A=1 ended without an accepted posterior: {detail}"
            )
        print(f"GW{n_events:04d} A=1: {detail}", flush=True)


if __name__ == "__main__":
    main()
