#!/usr/bin/env python3


from __future__ import annotations

import json
from math import factorial
from pathlib import Path

import numpy as np
from scipy.special import spherical_jn

import einstein_nled_master as master
import scalar_shell_eft_all_models as shell


def reduced_inside_value(model_name: str, charge: float, omega: float, radius: float, r: float):
    metric, _, _ = shell.geometry(model_name, charge)
    f_r = float(metric(radius))
    root_f = np.sqrt(f_r)
    r_tilde = radius + (r - radius) / root_f
    omega_tilde = omega / root_f
    return r * spherical_jn(shell.ELL, omega_tilde * r_tilde)


def main() -> None:
    rows = []
    maximum_relative_derivative_error = 0.0
    maximum_shell_radius_error = 0.0
    maximum_prefactor_relative_error = 0.0

    for model_name in shell.MODELS:
        extremal = master.extremal_charge(model_name, shell.MASS)
        charge = 0.7 * extremal
        omega = 5.0e-4
        radius = 12.0
        metric, _, _ = shell.geometry(model_name, charge)
        f_r = float(metric(radius))
        root_f = np.sqrt(f_r)

        r_tilde_at_shell = radius + (radius - radius) / root_f
        shell_radius_error = abs(r_tilde_at_shell - radius)
        maximum_shell_radius_error = max(maximum_shell_radius_error, shell_radius_error)

        analytic_value, analytic_derivative = shell.inside_solution(
            model_name, charge, omega, radius
        )
        h = 1.0e-4
        numerical_derivative = (
            reduced_inside_value(model_name, charge, omega, radius, radius + h)
            - reduced_inside_value(model_name, charge, omega, radius, radius - h)
        ) / (2.0 * h)
        derivative_relative_error = abs(analytic_derivative - numerical_derivative) / max(
            abs(analytic_derivative), 1.0e-30
        )
        maximum_relative_derivative_error = max(
            maximum_relative_derivative_error, derivative_relative_error
        )

        x_r = omega * radius / root_f
        expected_value = radius * spherical_jn(shell.ELL, x_r)
        value_relative_error = abs(analytic_value - expected_value) / max(
            abs(expected_value), 1.0e-30
        )

        prefactor_code = (
            4.0
            * np.pi
            * factorial(shell.ELL)
            * 2.0**shell.ELL
            * radius ** (2 * shell.ELL + 2)
            / factorial(2 * shell.ELL + 1)
            * f_r ** ((2 * shell.ELL + 1) / 2.0)
        )
        prefactor_expected = 4.0 * np.pi * radius**6 * f_r ** 2.5 / 15.0
        prefactor_relative_error = abs(prefactor_code - prefactor_expected) / abs(
            prefactor_expected
        )
        maximum_prefactor_relative_error = max(
            maximum_prefactor_relative_error, prefactor_relative_error
        )

        rows.append(
            {
                "model": model_name,
                "ell_over_ell_ext": 0.7,
                "omega_M": omega,
                "R_over_M": radius,
                "f_R": f_r,
                "r_tilde_at_shell": r_tilde_at_shell,
                "shell_radius_absolute_error": shell_radius_error,
                "bessel_argument": x_r,
                "inside_value_relative_error": value_relative_error,
                "inside_derivative_relative_error": derivative_relative_error,
                "quadrupole_prefactor_relative_error": prefactor_relative_error,
            }
        )

    report = {
        "coordinate_definition": "r_tilde = R + (r-R)/sqrt(f_R)",
        "reduced_inside_field": "psi_in = A r j_l(omega_tilde r_tilde)",
        "frequency_definition": "omega_tilde = omega/sqrt(f_R)",
        "quadrupole_normalization": "N_2 = (4 pi/15) R^6 f_R^(5/2)",
        "maximum_shell_radius_absolute_error": maximum_shell_radius_error,
        "maximum_inside_derivative_relative_error": maximum_relative_derivative_error,
        "maximum_quadrupole_prefactor_relative_error": maximum_prefactor_relative_error,
        "checks": rows,
        "status": "passed",
    }
    if maximum_shell_radius_error > 1.0e-14:
        raise AssertionError(json.dumps(report, indent=2))
    if maximum_relative_derivative_error > 1.0e-8:
        raise AssertionError(json.dumps(report, indent=2))
    if maximum_prefactor_relative_error > 1.0e-14:
        raise AssertionError(json.dumps(report, indent=2))

    output = Path(__file__).resolve().parent / "results" / "scalar_shell_junction_check.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
