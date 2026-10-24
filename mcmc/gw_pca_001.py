#!/usr/bin/env python3
"""Formal GW-only weak-prior MCMC used for PCA data-Fisher extraction.

This script fixes the modified-Horndeski correlation-prior amplitude to
``A=0.01``. It samples H0, Omega_m and 29 w bins using only the absolute GW
distances plus the weak correlation prior. The production ``A=1`` posterior
is owned separately by ``reconstruct_wz_gw.py``.

Examples::

    python3 mcmc/gw_pca_001.py --check --n-events 3000
    mpiexec -n 4 python3 mcmc/gw_pca_001.py --run --n-events 3000
    mpiexec -n 4 python3 mcmc/gw_pca_001.py --run --n-events 3000 --resume
    python3 mcmc/gw_pca_001.py --postprocess --n-events 3000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcmc.reconstruct_wz_gw import (  # noqa: E402
    FIDUCIAL_H0,
    FIDUCIAL_OMEGA_M,
    GWOnlyLikelihood,
    load_catalog,
    load_injection,
    postprocess,
    prefix_for,
    run_chains,
    sha256,
)
from paths import GW_PCA_001_CHAINS, GW_PCA_001_TABLES  # noqa: E402

PRIOR_AMPLITUDE = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--postprocess", action="store_true")
    mode.add_argument("--check", action="store_true")
    parser.add_argument("--n-events", type=int, default=3000)
    parser.add_argument("--grid-size", type=int, default=3200)
    parser.add_argument("--max-samples", type=int, default=250_000)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.resume and not args.run:
        parser.error("--resume requires --run")
    if args.n_events < 1 or args.grid_size < 1000 or args.max_samples < 1:
        parser.error("invalid event count, grid size or sample limit")
    return args


def main() -> None:
    args = parse_args()
    if args.check:
        model = GWOnlyLikelihood(
            load_catalog(args.n_events),
            grid_size=args.grid_size,
            prior_amplitude=PRIOR_AMPLITUDE,
        )
        point = np.asarray([FIDUCIAL_H0, FIDUCIAL_OMEGA_M, *load_injection()])
        print(json.dumps(model.components(point), indent=2))
        print(f"catalog SHA256: {sha256(model.data.path)}")
    elif args.postprocess:
        result = postprocess(
            prefix_for(args, default_root=GW_PCA_001_CHAINS),
            args.n_events,
            args.grid_size,
            PRIOR_AMPLITUDE,
            table_root=GW_PCA_001_TABLES,
        )
        print(json.dumps(result["diagnostics"], indent=2))
    else:
        run_chains(
            args,
            prior_amplitude=PRIOR_AMPLITUDE,
            default_root=GW_PCA_001_CHAINS,
            table_root=GW_PCA_001_TABLES,
        )


if __name__ == "__main__":
    main()
