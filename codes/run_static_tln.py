#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

from derive_all_static_tln import (
    DEFAULT_ATOL,
    DEFAULT_HORIZON_OFFSET,
    DEFAULT_MATCH_RADII,
    DEFAULT_RATIOS,
    DEFAULT_RTOL,
    derive_one,
)


HERE = Path(__file__).resolve().parent
MODELS = ("bardeen", "hayward", "fan_wang")
PARITIES = ("polar", "axial")
BASE_ORDER = 22
CHECK_ORDER = 24
LOG_ORDER = 6


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    rows = [
        derive_one(
            model,
            parity,
            ratio,
            BASE_ORDER,
            LOG_ORDER,
            match_radii=DEFAULT_MATCH_RADII,
            horizon_offset=DEFAULT_HORIZON_OFFSET,
            rtol=DEFAULT_RTOL,
            atol=DEFAULT_ATOL,
        )
        for model in MODELS
        for parity in PARITIES
        for ratio in DEFAULT_RATIOS
    ]
    results = HERE / "results"
    write_csv(results / "all_models_static_direct.csv", rows)

    convergence = []
    for model in MODELS:
        for parity in PARITIES:
            low = next(
                row for row in rows
                if row["model"] == model
                and row["parity"] == parity
                and row["ell_over_ell_ext"] == 0.5
            )
            high = derive_one(
                model,
                parity,
                0.5,
                CHECK_ORDER,
                LOG_ORDER,
                match_radii=DEFAULT_MATCH_RADII,
                horizon_offset=DEFAULT_HORIZON_OFFSET,
                rtol=DEFAULT_RTOL,
                atol=DEFAULT_ATOL,
            )
            change = abs(high["k2_static_direct"] - low["k2_static_direct"])
            convergence.append({
                "model": model,
                "parity": parity,
                "ell_over_ell_ext": 0.5,
                f"order_{BASE_ORDER}": low["k2_static_direct"],
                f"order_{CHECK_ORDER}": high["k2_static_direct"],
                "absolute_change": change,
                "fractional_change": change / abs(low["k2_static_direct"]),
            })
    (results / "all_models_static_convergence.json").write_text(
        json.dumps(convergence, indent=2) + "\n"
    )
    controls = {
        "mass": 1.0,
        "models": MODELS,
        "parities": PARITIES,
        "ell_over_ell_ext": DEFAULT_RATIOS,
        "base_series_order": BASE_ORDER,
        "convergence_series_order": CHECK_ORDER,
        "log_order": LOG_ORDER,
        "match_radii_over_M": DEFAULT_MATCH_RADII,
        "horizon_offset_over_M": DEFAULT_HORIZON_OFFSET,
        "relative_tolerance": DEFAULT_RTOL,
        "absolute_tolerance": DEFAULT_ATOL,
        "polar_metric": "-f H0",
        "axial_metric": "h0",
        "log_convention": "log(r/2M)",
    }
    (results / "all_models_static_numerical_controls.json").write_text(
        json.dumps(controls, indent=2) + "\n"
    )
    subprocess.run([sys.executable, str(HERE / "plot_all_models_static_direct.py")], check=True)


if __name__ == "__main__":
    main()
