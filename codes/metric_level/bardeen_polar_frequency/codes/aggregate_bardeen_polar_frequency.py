#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
FREQUENCIES = [0.002, 0.004, 0.006]


def direct_static_fit(ell_over_ell_ext: float) -> float:
    x=ell_over_ell_ext*4.0/(3.0*math.sqrt(3.0))
    return (0.08022747739312462*x**2+10.76112455362971*x**4
            -0.14943707892554278*x**6-0.1809082206985172*x**8)


def load(path: Path) -> dict[float, dict[str, str]]:
    return {float(row["omega_M"]): row for row in csv.DictReader(path.open())}


def main() -> None:
    rbh = load(RESULTS / "bardeen_polar_metric_frequency_raw_q050.csv")
    static = direct_static_fit(0.5)
    rows = [
        {
            "ell_over_ell_ext": 0.5,
            "omega_M": 0.0,
            "k_static": static,
            "metric_ratio_real": float("nan"),
            "metric_ratio_imag": 0.0,
            "metric_ratio_static_real": float("nan"),
            "metric_ratio_static_imag": 0.0,
            "delta_k_real": 0.0,
            "delta_k_imag": 0.0,
            "k_dynamic_real": static,
            "k_dynamic_imag": 0.0,
            "window_fit_rms": 0.0,
            "window_max_residual": 0.0,
        }
    ]
    for omega in FREQUENCIES:
        r = rbh[omega]
        delta_real = float(r["delta_k_real"])
        delta_imag = float(r["delta_k_imag"])
        rows.append(
            {
                "ell_over_ell_ext": 0.5,
                "omega_M": omega,
                "k_static": static,
                "metric_ratio_real": float(r["metric_ratio_real"]),
                "metric_ratio_imag": float(r["metric_ratio_imag"]),
                "metric_ratio_static_real": float(r["metric_ratio_static_real"]),
                "metric_ratio_static_imag": float(r["metric_ratio_static_imag"]),
                "delta_k_real": delta_real,
                "delta_k_imag": delta_imag,
                "k_dynamic_real": static + delta_real,
                "k_dynamic_imag": delta_imag,
                "window_fit_rms": float(r["window_fit_rms"]),
                "window_max_residual": float(r["window_max_residual"]),
            }
        )

    output = RESULTS / "bardeen_polar_metric_frequency_scan.csv"
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "definition": "Direct reconstructed metric response; the independent static Bardeen polar result fixes the zero-frequency constant and the finite-frequency correction is taken from the same Bardeen geometry. No Schwarzschild subtraction is applied.",
        "polar_love_normalization": "k20_polar=-(C_response/C_source)/M^5",
        "ell_over_ell_ext": 0.5,
        "quantitative_max_Momega": 0.004,
        "edge_diagnostic_Momega": 0.006,
        "rows": len(rows),
    }
    (RESULTS / "bardeen_polar_metric_frequency_scan_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    FIGURES.mkdir(parents=True, exist_ok=True)
    omega = [row["omega_M"] for row in rows]
    value = [row["k_dynamic_real"] for row in rows]
    correction = [row["delta_k_real"] for row in rows]
    figure, axis = plt.subplots(figsize=(7.2, 5.0))
    axis.plot(omega, value, "o-", linewidth=1.4)
    axis.axhline(static, linestyle="--", linewidth=1.4, label="Direct static result")
    axis.set_xlabel(r"$M\omega$")
    axis.set_ylabel(r"$\mathrm{Re}\,k_{20}^{\rm polar}(\omega)$")
    axis.ticklabel_format(axis="y", style="plain", useOffset=False)
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    inset = axis.inset_axes([0.13, 0.16, 0.47, 0.42])
    inset.plot(omega, [1.0e4 * x for x in correction], "o-", linewidth=1.1)
    inset.set_xlabel(r"$M\omega$", fontsize=8)
    inset.set_ylabel(r"$10^4[k(\omega)-k(0)]$", fontsize=8)
    inset.tick_params(labelsize=7)
    inset.grid(alpha=0.22)
    figure.tight_layout()
    figure.savefig(FIGURES / "bardeen_polar_metric_dynamic_tln_vs_frequency.png", dpi=320)
    figure.savefig(FIGURES / "bardeen_polar_metric_dynamic_tln_vs_frequency.pdf")
    plt.close(figure)


if __name__ == "__main__":
    main()
