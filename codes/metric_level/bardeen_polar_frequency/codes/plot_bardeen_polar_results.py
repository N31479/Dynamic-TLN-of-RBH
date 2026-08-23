#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_table(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"No rows found in {path}")
    columns: dict[str, np.ndarray] = {}
    for key in rows[0]:
        columns[key] = np.asarray([float(row[key]) for row in rows], dtype=float)
    return columns


def run(input_csv: Path, output_dir: Path) -> None:
    data = read_table(input_csv)
    output_dir.mkdir(parents=True, exist_ok=True)

    omega = data["omega_M"]
    static = data["k_static_reference"]
    real = data["k_dynamic_real"]
    real_error = data["uncertainty_real"]
    imag = data["k_dynamic_imag"]
    imag_error = data["uncertainty_imag"]

    figure, axis = plt.subplots(figsize=(6.8, 4.8))
    axis.axhline(static[0], linestyle="--", label="Direct static limit")
    axis.errorbar(
        omega,
        real,
        yerr=real_error,
        marker="o",
        capsize=4,
        label=r"$\operatorname{Re}\mathcal{K}_{20}^{\mathrm{polar}}(\omega)$",
    )
    axis.set_xlabel(r"$M\omega$")
    axis.set_ylabel(r"$k_{20}^{\rm polar}$")
    axis.set_title(r"Bardeen polar dynamical TLN: $q/q_{\rm ext}=0.5$")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_dir / "bardeen_polar_dynamic_tln_with_uncertainty.png", dpi=220)
    figure.savefig(output_dir / "bardeen_polar_dynamic_tln_with_uncertainty.pdf")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.8, 4.8))
    axis.errorbar(omega, imag, yerr=imag_error, marker="o", capsize=4)
    axis.set_xlabel(r"$M\omega$")
    axis.set_ylabel(r"$\operatorname{Im}\mathcal{K}_{20}^{\mathrm{polar}}$")
    axis.set_title(r"Bardeen polar dissipative response: $q/q_{\rm ext}=0.5$")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(
        output_dir / "bardeen_polar_dissipative_response_with_uncertainty.png",
        dpi=220,
    )
    figure.savefig(
        output_dir / "bardeen_polar_dissipative_response_with_uncertainty.pdf"
    )
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    package = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=package / "results" / "bardeen_polar_dynamic_tln_convergence.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=package / "figures"
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.input_csv, arguments.output_dir)
