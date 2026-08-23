#!/usr/bin/env python3
"""Fit, compare, and plot the directly calculated static Love numbers.

This script reads the direct zero-frequency scan, performs separate
small-regularization and full-range fits, and writes both machine-readable
fit results and a comparison with the small-regularization fits reported by
Coviello, Vellucci, and Lehner (Phys. Rev. D 111, 104073 (2025)).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
MODELS = ("bardeen", "hayward", "fan_wang")
PARITIES = ("polar", "axial")
LABELS = {"bardeen": "Bardeen", "hayward": "Hayward", "fan_wang": "Fan--Wang"}
EXTREMAL = {
    "bardeen": 4 / (3 * np.sqrt(3)),
    "hayward": 4 / (3 * np.sqrt(3)),
    "fan_wang": 8 / 27,
}
FULL_POWERS = {
    ("bardeen", "polar"): [2, 4, 6, 8],
    ("bardeen", "axial"): [4, 6, 8, 10],
    ("hayward", "polar"): [2, 4, 6, 8],
    ("hayward", "axial"): [4, 6, 8, 10],
    ("fan_wang", "polar"): [3, 4, 5, 6],
    ("fan_wang", "axial"): [3, 4, 5, 6],
}
SMALL_POWERS = {
    ("bardeen", "polar"): [2, 4],
    ("bardeen", "axial"): [4, 6],
    ("hayward", "polar"): [2, 4],
    ("hayward", "axial"): [4, 6],
    ("fan_wang", "polar"): [3, 4],
    ("fan_wang", "axial"): [3, 4],
}
COVIELLO = {
    ("bardeen", "polar"): ([2, 4], [0.03, 10.0]),
    ("bardeen", "axial"): ([4], [-5.8]),
    ("hayward", "polar"): ([2], [13.0]),
    ("hayward", "axial"): ([4], [1.8]),
    ("fan_wang", "polar"): ([3], [220.0]),
    ("fan_wang", "axial"): ([3], [95.0]),
}


def polynomial(values, powers, coefficients):
    return sum(coefficient * values**power for power, coefficient in zip(powers, coefficients))


def formula(powers, coefficients, digits=6):
    terms = []
    for index, (power, coefficient) in enumerate(zip(powers, coefficients)):
        sign = "-" if coefficient < 0 else "+"
        magnitude = f"{abs(coefficient):.{digits}g}"
        term = f"{magnitude} x^{power}"
        if index == 0:
            terms.append(("-" if coefficient < 0 else "") + term)
        else:
            terms.append(f" {sign} {term}")
    return "".join(terms)


def main():
    input_path = HERE / "results/all_models_static_direct.csv"
    rows = list(csv.DictReader(input_path.open()))
    summary = {}
    comparison_rows = []
    figure, axes = plt.subplots(2, 3, figsize=(11.2, 6.7), sharex=True)

    for column, model in enumerate(MODELS):
        for row_index, parity in enumerate(PARITIES):
            key = (model, parity)
            selected = [row for row in rows if (row["model"], row["parity"]) == key]
            ratio = np.array([float(row["ell_over_ell_ext"]) for row in selected])
            scale = np.array([float(row["ell_over_M"]) for row in selected])
            direct = np.array([float(row["k2_static_direct"]) for row in selected])
            fit_mask = ratio >= (0.15 if key == ("bardeen", "polar") else 0.1)

            powers = FULL_POWERS[key]
            design = np.column_stack([scale[fit_mask] ** power for power in powers])
            coefficients = np.linalg.lstsq(design, direct[fit_mask], rcond=None)[0]
            full_residual = design @ coefficients - direct[fit_mask]

            small_mask = (ratio <= 0.3) & fit_mask
            small_powers = SMALL_POWERS[key]
            small_design = np.column_stack([scale[small_mask] ** power for power in small_powers])
            small_coefficients = np.linalg.lstsq(
                small_design, direct[small_mask], rcond=None
            )[0]
            small_residual = small_design @ small_coefficients - direct[small_mask]

            coviello_powers, coviello_coefficients = COVIELLO[key]
            x = np.linspace(0, 1, 500)
            ell_over_m = x * EXTREMAL[model]
            fitted = polynomial(ell_over_m, powers, coefficients)
            published = polynomial(ell_over_m, coviello_powers, coviello_coefficients)

            axis = axes[row_index, column]
            axis.plot(x, published, "--", color="#d55e00", lw=1.7,
                      label="Coviello small-$\\ell$ fit")
            axis.plot(x, fitted, color="#0072b2", lw=1.7, label="Direct continuation")
            axis.scatter(ratio, direct, color="#0072b2", edgecolor="white", linewidth=0.5,
                         s=26, zorder=3, label="Direct calculation")
            axis.axhline(0, color="0.35", lw=0.6)
            axis.set_title(f"{LABELS[model]} {parity}")
            axis.grid(alpha=0.18)
            if row_index == 1:
                axis.set_xlabel(r"$\ell/\ell_{\rm ext}$")
            if column == 0:
                axis.set_ylabel(r"$k_2^{\rm static}$")

            name = f"{model}_{parity}"
            summary[name] = {
                "fit_variable": "x=ell/M",
                "fit_powers": powers,
                "fit_coefficients": coefficients.tolist(),
                "fit_rms": float(np.sqrt(np.mean(full_residual**2))),
                "small_ell_powers": small_powers,
                "small_ell_coefficients": small_coefficients.tolist(),
                "small_ell_rms": float(np.sqrt(np.mean(small_residual**2))),
                "coviello_powers": coviello_powers,
                "coviello_coefficients": coviello_coefficients,
            }
            comparison_rows.append({
                "model": model,
                "parity": parity,
                "fit_variable": "x=ell/M",
                "our_small_ell_fit": formula(small_powers, small_coefficients),
                "coviello_small_ell_fit": formula(coviello_powers, coviello_coefficients),
                "our_full_range_fit": formula(powers, coefficients),
                "our_small_ell_rms": float(np.sqrt(np.mean(small_residual**2))),
                "our_full_range_rms": float(np.sqrt(np.mean(full_residual**2))),
            })

    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="upper center", ncol=3, frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    for suffix in ("png", "pdf"):
        figure.savefig(HERE.parent / "figures" / f"all_models_static_direct.{suffix}", dpi=260)
    plt.close(figure)

    results = HERE / "results"
    (results / "all_models_static_fits.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (results / "all_models_static_scaling_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison_rows[0]))
        writer.writeheader()
        writer.writerows(comparison_rows)


if __name__ == "__main__":
    main()
