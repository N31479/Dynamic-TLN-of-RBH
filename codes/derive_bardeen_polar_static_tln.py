#!/usr/bin/env python3
"""Direct quadrupolar static polar response of the Bardeen black hole.

The calculation uses the coupled Einstein--NLED master equations.  It fixes a
unit gravitational tidal source and no independent electromagnetic source at
infinity, imposes regularity at the outer horizon, reconstructs -f H0, and
extracts the finite r^-3 coefficient in the log(r/2M) convention.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import sympy as sp


HERE = Path(__file__).resolve().parent
CHANNEL = HERE / "metric_level" / "bardeen_polar_frequency" / "codes"
sys.path.insert(0, str(CHANNEL))
sys.path.insert(0, str(HERE))

import einstein_nled_master as master
import bardeen_polar_dynamic_frobenius as frobenius
import bardeen_polar_nearzone_dynamic_tln as solver


def _background(mass: float, charge: float, t: sp.Symbol):
    r = t**-2
    denominator = 1 + charge**2 * t**4
    mass_function = mass * denominator ** sp.Rational(-3, 2)
    lagrangian = 3 * mass * charge**2 * t**10 * denominator ** sp.Rational(-5, 2)
    lagrangian_f = (
        sp.Rational(15, 2) * mass * t**2 * denominator ** sp.Rational(-7, 2)
    )
    metric = 1 - 2 * mass_function / r
    return r, mass_function, lagrangian, lagrangian_f, metric


def _field_series(mode, component: int, t: sp.Symbol):
    exponent, coefficients = mode
    exponent = float(exponent)
    if abs(exponent - round(exponent)) < 1.0e-12:
        symbolic_exponent = int(round(exponent))
    else:
        symbolic_exponent = sp.Float(exponent, 16)
    result = 0
    omega_zero = coefficients[0]
    for radial_order in range(omega_zero.shape[0]):
        for log_order in range(omega_zero.shape[1]):
            value = float(np.real(omega_zero[radial_order, log_order, component]))
            if value:
                result += (
                    value
                    * t ** (symbolic_exponent + radial_order)
                    * sp.log(t) ** log_order
                )
    return result


def _metric_series(
    mass: float,
    charge: float,
    psi: sp.Expr,
    phi: sp.Expr,
    t: sp.Symbol,
):
    r, mass_function, lagrangian, lagrangian_f, metric = _background(
        mass, charge, t
    )
    radial_derivative = lambda expression: -sp.Rational(1, 2) * t**3 * sp.diff(
        expression, t
    )
    auxiliary = 6 * mass_function / r - 2 * r**2 * lagrangian
    zeta = (auxiliary + 4) * psi / 2
    electromagnetic = phi / (2 * sp.sqrt(lagrangian_f)) + charge * psi / (2 * r)
    K = (
        8 * metric * charge * lagrangian_f * electromagnetic / r
        - 2 * r * metric * radial_derivative(zeta)
        - 6 * zeta
    ) / (r * (auxiliary + 4))
    H0 = (
        -radial_derivative(zeta)
        - r * radial_derivative(K)
        + 4 * charge * lagrangian_f * electromagnetic / r**2
    )
    return -metric * H0


def _canonical_metric_coefficients(
    mass: float,
    charge: float,
    modes,
    source_transform: np.ndarray,
    response_transform: np.ndarray,
):
    t = sp.symbols("t", positive=True)
    source_psi = sum(
        source_transform[j, 0] * _field_series(modes[j], 0, t) for j in range(2)
    )
    source_phi = sum(
        source_transform[j, 0] * _field_series(modes[j], 1, t) for j in range(2)
    )
    response_psi = sum(
        response_transform[j, 0] * _field_series(modes[j + 2], 0, t)
        for j in range(2)
    )
    response_phi = sum(
        response_transform[j, 0] * _field_series(modes[j + 2], 1, t)
        for j in range(2)
    )
    source_metric = _metric_series(mass, charge, source_psi, source_phi, t)
    response_metric = _metric_series(mass, charge, response_psi, response_phi, t)
    source_power, response_power = -4, 6
    stop = response_power + 2
    source_series = sp.series(source_metric, t, 0, stop).removeO().expand()
    response_series = sp.series(response_metric, t, 0, stop).removeO().expand()
    source_norm = float(
        sp.N(source_series.coeff(t, source_power).subs(sp.log(t), 0))
    )
    response_norm = float(
        sp.N(response_series.coeff(t, response_power).subs(sp.log(t), 0))
    )
    response_polynomial = source_series.coeff(t, response_power)
    # Since t=r^-1/2, this choice is equivalent to log(r/2M).
    log_at_response_scale = -0.5 * np.log(2 * mass)
    local_response = float(
        sp.N(response_polynomial.subs(sp.log(t), log_at_response_scale))
    ) / source_norm
    return source_norm, response_norm, local_response


def _horizon_response(
    model,
    modes,
    source_transform: np.ndarray,
    response_transform: np.ndarray,
    radius: float,
    controls: dict,
):
    horizon_value, horizon_derivative = solver.horizon_basis(
        model, 0.0, radius, **controls
    )
    value = np.zeros((2, 4), dtype=complex)
    derivative = np.zeros((2, 4), dtype=complex)
    for column, (exponent, coefficients) in enumerate(modes):
        v, d = frobenius.evaluate_dynamic_mode(
            coefficients, exponent, 0.0, np.array([radius])
        )
        value[:, column] = v[0]
        derivative[:, column] = d[0]
    value = np.column_stack(
        (value[:, :2] @ source_transform, value[:, 2:] @ response_transform)
    )
    derivative = np.column_stack(
        (
            derivative[:, :2] @ source_transform,
            derivative[:, 2:] @ response_transform,
        )
    )
    asymptotic = np.vstack((value, derivative))
    horizon = np.vstack((horizon_value, horizon_derivative))
    scales = np.linalg.norm(asymptotic, axis=0)
    coefficients = np.linalg.solve(asymptotic / scales, horizon) / scales[:, None]
    weights = np.linalg.solve(
        coefficients[:2], np.array([1.0, 0.0], dtype=complex)
    )
    amplitudes = coefficients @ weights
    return amplitudes[2], amplitudes[3]


def derive_one(
    ell_over_extremal: float,
    mass: float = 1.0,
    series_order: int = 22,
    log_order: int = 3,
):
    charge = ell_over_extremal * master.extremal_charge("bardeen", mass)
    model = master.build_model("bardeen", charge, mass)
    modes = frobenius.build_modes(charge, series_order, 2, log_order)
    source_transform = np.eye(2, dtype=complex)
    response_transform = np.eye(2, dtype=complex)
    source_norm, response_norm, local_response = _canonical_metric_coefficients(
        mass, charge, modes, source_transform, response_transform
    )
    source_transform[:, 0] /= source_norm
    response_transform[:, 0] /= response_norm
    controls = {"horizon_offset": 2.0e-6, "rtol": 5.0e-10, "atol": 5.0e-12}
    radii = np.array([10.0, 12.0, 14.0, 16.0]) * mass
    values = np.array(
        [
            _horizon_response(
                model,
                modes,
                source_transform,
                response_transform,
                radius,
                controls,
            )[0]
            for radius in radii
        ]
    )
    x = radii / mass
    design = np.column_stack((np.ones_like(x), x**-2, x**-4))
    fit = np.linalg.lstsq(design, np.real(values), rcond=None)[0]
    prediction = design @ fit
    horizon_response = float(fit[0])
    coefficient = horizon_response + local_response
    love = -coefficient / mass**5
    return {
        "ell_over_ell_ext": ell_over_extremal,
        "ell": charge,
        "k2_polar_static_direct": love,
        "canonical_local_term": local_response,
        "horizon_response_term": horizon_response,
        "window_fit_rms": float(
            np.sqrt(np.mean((np.real(values) - prediction) ** 2))
        ),
        "series_order": series_order,
        "log_order": log_order,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ratios",
        nargs="+",
        type=float,
        default=[0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.95],
    )
    parser.add_argument("--series-order", type=int, default=22)
    parser.add_argument("--log-order", type=int, default=3)
    parser.add_argument(
        "--output", type=Path, default=HERE / "results/bardeen_polar_static_direct.csv"
    )
    args = parser.parse_args()
    rows = [
        derive_one(ratio, 1.0, args.series_order, args.log_order)
        for ratio in args.ratios
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
