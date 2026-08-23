#!/usr/bin/env python3

from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import bardeen_polar_nearzone_dynamic_tln as calculation


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FREQUENCIES = [0.002, 0.004, 0.006]


def evaluate(series_order: int, match_radii: list[float], output: Path):
    settings = SimpleNamespace(
        mass=1.0,
        charge_ratio=0.5,
        series_order=series_order,
        omega_order=2,
        log_order=5,
        frequencies=FREQUENCIES,
        match_radii=match_radii,
        horizon_offset=2.0e-6,
        rtol=5.0e-10,
        atol=5.0e-12,
        output_dir=output,
    )
    return calculation.run(settings)


def main() -> None:
    default = evaluate(18, [10.0, 12.0, 14.0, 16.0], ROOT / "results" / "convergence_default")
    window = evaluate(18, [9.0, 11.0, 13.0, 15.0], ROOT / "results" / "convergence_window")
    order = evaluate(16, [10.0, 12.0, 14.0, 16.0], ROOT / "results" / "convergence_order16")
    rows = []
    for base, shifted, reduced in zip(default, window, order):
        uncertainty_real = max(
            abs(base["k_dynamic_real"] - shifted["k_dynamic_real"]),
            abs(base["k_dynamic_real"] - reduced["k_dynamic_real"]),
        )
        uncertainty_imag = max(
            abs(base["k_dynamic_imag"] - shifted["k_dynamic_imag"]),
            abs(base["k_dynamic_imag"] - reduced["k_dynamic_imag"]),
        )
        rows.append(
            {
                "omega_M": base["omega_M"],
                "k_static_reference": base["k_static_reference"],
                "k_dynamic_real": base["k_dynamic_real"],
                "uncertainty_real": uncertainty_real,
                "k_dynamic_imag": base["k_dynamic_imag"],
                "uncertainty_imag": uncertainty_imag,
                "k_dynamic_real_window_alt": shifted["k_dynamic_real"],
                "k_dynamic_real_series16": reduced["k_dynamic_real"],
                "window_fit_rms": base["window_fit_rms"],
                "window_max_residual": base["window_max_residual"],
                "maximum_omega_r": base["maximum_omega_r"],
            }
        )
    output = ROOT / "results" / "bardeen_polar_dynamic_tln_convergence.csv"
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
