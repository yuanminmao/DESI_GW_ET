#!/usr/bin/env python3
"""Copy regenerated publication figures into the self-contained paper tree."""

from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "results" / "figures"
DESTINATION = ROOT / "paper" / "figures"
FIGURES = (
    "wz_posterior_and_cov.png",
    "sensitivity_curves.png",
    "mock_GW_catalog_ET_by_4.png",
    "wz_posterior_cov_gw_vs_desi_bbn_cmb_gw0100_0500_1000_3000_with_injected_w.png",
    "gw_and_desi_bbn_cmb_gw_pca_spectrum.png",
    "wz_pca_modes_pc1-4_gw_vs_desi_bbn_cmb_gw0100_0500_1000_3000.png",
    "gw_w0_wa_4.png",
)


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    copied = 0
    for name in FIGURES:
        source = SOURCE / name
        if not source.is_file():
            print(f"Keeping committed figure (no regenerated file): {name}")
            continue
        shutil.copy2(source, DESTINATION / name)
        copied += 1
        print(f"Updated: paper/figures/{name}")
    print(f"Updated {copied} of {len(FIGURES)} paper figures.")


if __name__ == "__main__":
    main()
