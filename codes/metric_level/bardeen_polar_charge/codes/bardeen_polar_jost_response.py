#!/usr/bin/env python3










from __future__ import annotations

import argparse
import csv
import json
from functools import lru_cache
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import einstein_nled_master as master


def _symbolic_bardeen_polar():
    t = sp.symbols("t", positive=True)
    mass, charge = sp.symbols("mass charge", positive=True)
    radius = t ** -2
    denominator = radius**2 + charge**2
    mass_function = mass * radius**3 * denominator ** (-sp.Rational(3, 2))
    lagrangian = 3 * mass * charge**2 * denominator ** (-sp.Rational(5, 2))
    lagrangian_f = (
        sp.Rational(15, 2)
        * mass
        * radius**6
        * denominator ** (-sp.Rational(7, 2))
    )
    metric = 1 - 2 * mass_function / radius

    def radial_derivative(expression):
        return -sp.Rational(1, 2) * t**3 * sp.diff(expression, t)

    metric_prime = radial_derivative(metric)
    log_lf_prime = radial_derivative(sp.log(lagrangian_f))
    log_lf_second = radial_derivative(log_lf_prime)
    kappa = 1 - sp.Rational(1, 2) * radius * log_lf_prime
    dplus = (
        sp.Rational(1, 2) * metric_prime * log_lf_prime
        + metric
        * (
            sp.Rational(1, 2) * log_lf_second
            + sp.Rational(1, 4) * log_lf_prime**2
        )
    )

    angular = sp.Integer(6)
    lam = sp.Integer(4)
    a = 6 * mass_function / radius - 2 * radius**2 * lagrangian
    b = lam + 4 * lagrangian_f * charge**2 / radius**2
    den = a + lam
    coupling = -sp.sqrt(4 * lam * lagrangian_f) * charge / radius**3

    v11 = (
        angular * lam
        - 2 * metric * lam
        + a * (a - 4 * mass_function / radius)
    ) / (radius**2 * den)
    v11 += 2 * metric * lam * b / (radius**2 * den**2)

    v22 = kappa * angular / radius**2 + dplus
    v22 += (
        4
        * lagrangian_f
        * charge**2
        * (
            lam
            + 1
            - metric
            + 2 * radius**2 * lagrangian
            + 4 * metric * kappa
        )
        / (radius**4 * den)
    )
    v22 += (
        8
        * metric
        * lagrangian_f
        * charge**2
        * b
        / (radius**4 * den**2)
    )

    mixing = (
        lam
        + 1
        - metric
        + 2 * radius**2 * lagrangian
        + 2 * metric * kappa
    ) / den
    mixing += 2 * metric * b / den**2
    v12 = coupling * mixing
    return t, mass, charge, metric, metric_prime, v11, v12, v22


_T, _MASS, _CHARGE, _F, _FP, _V11, _V12, _V22 = _symbolic_bardeen_polar()


def _series_coefficients(expression, substitutions, maximum_power):
    series = sp.series(
        expression.subs(substitutions), _T, 0, maximum_power + 1
    ).removeO().expand()
    coefficients = np.zeros(maximum_power + 1, dtype=complex)
    for term in sp.Add.make_args(series):
        coefficient, exponent = term.as_coeff_exponent(_T)
        power = int(exponent)
        if 0 <= power <= maximum_power:
            coefficients[power] = complex(sp.N(coefficient, 30))
    return coefficients


@lru_cache(maxsize=16)
def bardeen_asymptotic_series(charge: float, order: int):
    substitutions = {_MASS: 1.0, _CHARGE: float(charge)}
    maximum_power = order + 4
    metric = _series_coefficients(_F, substitutions, maximum_power)
    metric_prime = _series_coefficients(_FP, substitutions, maximum_power)
    potential = np.zeros((maximum_power + 1, 2, 2), dtype=complex)
    potential[:, 0, 0] = _series_coefficients(
        _V11, substitutions, maximum_power
    )
    potential[:, 0, 1] = _series_coefficients(
        _V12, substitutions, maximum_power
    )
    potential[:, 1, 0] = potential[:, 0, 1]
    potential[:, 1, 1] = _series_coefficients(
        _V22, substitutions, maximum_power
    )
    return metric, metric_prime, potential


def jost_coefficients(
    omega: float,
    sign: int,
    channel: int,
    metric: np.ndarray,
    metric_prime: np.ndarray,
    potential: np.ndarray,
    order: int,
):
    dimension = potential.shape[1]
    coefficients = np.zeros((order + 1, dimension), dtype=complex)
    coefficients[0, channel] = 1.0
    for total_power in range(3, order + 3):
        index = total_power - 2
        if index > order:
            break
        known = np.zeros(dimension, dtype=complex)

        for power, value in enumerate(metric):
            lower = total_power - power - 4
            if 0 <= lower <= order:
                known += (
                    value
                    * 0.25
                    * lower
                    * (lower + 2)
                    * coefficients[lower]
                )

        for power, value in enumerate(metric_prime):
            lower = total_power - power - 2
            if 0 <= lower <= order:
                known += value * (-0.5 * lower) * coefficients[lower]

        for power, value in enumerate(potential):
            lower = total_power - power
            if 0 <= lower <= order:
                known -= value @ coefficients[lower]

        if index > 0:
            coefficients[index] = known / (sign * 1.0j * omega * index)
    return coefficients


def asymptotic_tortoise_at_radius(
    model: master.Model, radius: float, metric_series: np.ndarray
) -> float:
    

    inverse = np.zeros_like(metric_series, dtype=complex)
    inverse[0] = 1.0 / metric_series[0]
    for power in range(1, inverse.size):
        inverse[power] = -(
            sum(
                metric_series[index] * inverse[power - index]
                for index in range(1, power + 1)
            )
            / metric_series[0]
        )
    correction = 0.0 + 0.0j
    for power, coefficient in enumerate(inverse):
        if power <= 2 or abs(coefficient) == 0.0:
            continue
        exponent = 0.5 * power
        correction += coefficient * radius ** (1.0 - exponent) / (exponent - 1.0)
    return float(
        radius
        + 2.0 * model.mass * np.log(radius / (2.0 * model.mass))
        - correction.real
    )


def evaluate_jost(
    coefficients: np.ndarray,
    radius: float,
    omega: float,
    sign: int,
    metric_value: float,
    tortoise: float,
):
    t = radius ** -0.5
    indices = np.arange(coefficients.shape[0])
    field_series = ((t**indices)[:, None] * coefficients).sum(axis=0)
    derivative_series = (
        -0.5
        * (indices[:, None] * coefficients)
        * (t ** (indices + 2))[:, None]
    ).sum(axis=0)
    phase = np.exp(sign * 1.0j * omega * tortoise)
    field = phase * field_series
    derivative = phase * (
        sign * 1.0j * omega * field_series / metric_value
        + derivative_series
    )
    return field, derivative


def response_matrix_jost(
    model: master.Model,
    omega: float,
    outer_radius: float,
    series_order: int,
    controls: master.SolverControls,
):
    metric_series, metric_prime_series, potential_series = (
        bardeen_asymptotic_series(model.charge, series_order)
    )
    leading_coefficients = np.real(np.diag(potential_series[4]))
    effective_orders = 0.5 * (
        -1.0 + np.sqrt(1.0 + 4.0 * leading_coefficients)
    )

    horizon = master.outer_horizon(model)
    start = horizon + controls.horizon_offset
    match = controls.match_radius
    horizon_slope = float(model.fp(horizon))
    horizon_potential = master.potential_matrix(model, "polar", horizon, 2)
    first_regular = horizon_potential / (horizon_slope - 2.0j * omega)
    identity = np.eye(2, dtype=complex)
    horizon_value = identity + controls.horizon_offset * first_regular
    horizon_derivative = (
        -1.0j * omega * horizon_value / float(model.f(start))
        + first_regular
    )
    horizon_value, horizon_derivative = master._integrate_matrix(
        model,
        "polar",
        omega,
        start,
        match,
        horizon_value,
        horizon_derivative,
        controls,
    )

    outgoing_value = np.zeros((2, 2), dtype=complex)
    outgoing_derivative = np.zeros((2, 2), dtype=complex)
    incoming_value = np.zeros((2, 2), dtype=complex)
    incoming_derivative = np.zeros((2, 2), dtype=complex)
    tortoise = asymptotic_tortoise_at_radius(
        model, outer_radius, metric_series
    )
    metric_value = float(model.f(outer_radius))

    for channel, effective_order in enumerate(effective_orders):
        outgoing_coefficients = jost_coefficients(
            omega,
            +1,
            channel,
            metric_series,
            metric_prime_series,
            potential_series,
            series_order,
        )
        incoming_coefficients = jost_coefficients(
            omega,
            -1,
            channel,
            metric_series,
            metric_prime_series,
            potential_series,
            series_order,
        )
        outgoing_coefficients *= (-1.0j) ** (effective_order + 1.0)
        incoming_coefficients *= (1.0j) ** (effective_order + 1.0)
        outgoing_value[:, channel], outgoing_derivative[:, channel] = (
            evaluate_jost(
                outgoing_coefficients,
                outer_radius,
                omega,
                +1,
                metric_value,
                tortoise,
            )
        )
        incoming_value[:, channel], incoming_derivative[:, channel] = (
            evaluate_jost(
                incoming_coefficients,
                outer_radius,
                omega,
                -1,
                metric_value,
                tortoise,
            )
        )

    outgoing_value, outgoing_derivative = master._integrate_matrix(
        model,
        "polar",
        omega,
        outer_radius,
        match,
        outgoing_value,
        outgoing_derivative,
        controls,
    )
    incoming_value, incoming_derivative = master._integrate_matrix(
        model,
        "polar",
        omega,
        outer_radius,
        match,
        incoming_value,
        incoming_derivative,
        controls,
    )

    horizon_data = np.vstack((horizon_value, horizon_derivative))
    wave_data = np.block(
        [
            [outgoing_value, incoming_value],
            [outgoing_derivative, incoming_derivative],
        ]
    )
    coefficients = np.linalg.solve(wave_data, horizon_data)
    outgoing_amplitude = coefficients[:2]
    incoming_amplitude = coefficients[2:]
    source_amplitude = outgoing_amplitude + incoming_amplitude
    response_amplitude = 1.0j * (
        outgoing_amplitude - incoming_amplitude
    )
    response = response_amplitude @ np.linalg.inv(source_amplitude)
    return response, effective_orders


@lru_cache(maxsize=8)
def schwarzschild_series(order: int):
    t = sp.symbols("t", positive=True)
    radius = t ** -2
    metric_expression = 1 - 2 / radius
    metric_prime_expression = 2 / radius**2
    n = sp.Integer(2)
    potential_expression = 2 * (
        n**2 * (n + 1) * radius**3
        + 3 * n**2 * radius**2
        + 9 * n * radius
        + 9
    ) / (radius**3 * (n * radius + 3) ** 2)

    substitutions = {}

    def coefficients(expression):
        series = sp.series(expression, t, 0, order + 5).removeO().expand()
        result = np.zeros(order + 5, dtype=complex)
        for term in sp.Add.make_args(series):
            coefficient, exponent = term.as_coeff_exponent(t)
            power = int(exponent)
            if 0 <= power < result.size:
                result[power] = complex(sp.N(coefficient, 30))
        return result

    potential = np.zeros((order + 5, 1, 1), dtype=complex)
    potential[:, 0, 0] = coefficients(potential_expression)
    return coefficients(metric_expression), coefficients(metric_prime_expression), potential


def schwarzschild_response_jost(
    omega: float,
    outer_radius: float,
    series_order: int,
    controls: master.SolverControls,
):
    metric_series, metric_prime_series, potential_series = schwarzschild_series(
        series_order
    )
    horizon = 2.0
    start = horizon + controls.horizon_offset
    match = controls.match_radius
    horizon_slope = 0.5
    horizon_potential = float(
        master.schwarzschild_potential("polar", horizon, 1.0, 2)
    )
    first_regular = horizon_potential / (horizon_slope - 2.0j * omega)
    value = 1.0 + controls.horizon_offset * first_regular
    derivative = -1.0j * omega * value / (1.0 - 2.0 / start) + first_regular

    def integrate(begin, end, initial_value, initial_derivative):
        initial_matrix = np.array([[initial_value]], dtype=complex)
        derivative_matrix = np.array([[initial_derivative]], dtype=complex)

        def rhs(radius, state):
            field = state[0]
            field_prime = state[1]
            metric_value = 1.0 - 2.0 / radius
            metric_prime_value = 2.0 / radius**2
            potential_value = float(
                master.schwarzschild_potential("polar", radius, 1.0, 2)
            )
            field_second = -(metric_prime_value / metric_value) * field_prime
            field_second -= (
                omega**2 - metric_value * potential_value
            ) * field / metric_value**2
            return np.array([field_prime, field_second], dtype=complex)

        from scipy.integrate import solve_ivp

        solution = solve_ivp(
            rhs,
            (begin, end),
            np.array([initial_value, initial_derivative], dtype=complex),
            method="DOP853",
            rtol=controls.rtol,
            atol=controls.atol,
            max_step=min(5.0, controls.max_step_phase / omega),
        )
        if not solution.success:
            raise RuntimeError(solution.message)
        return solution.y[:, -1]

    horizon_data = integrate(start, match, value, derivative)
    tortoise = outer_radius + 2.0 * np.log(outer_radius / 2.0 - 1.0)
    metric_value = 1.0 - 2.0 / outer_radius
    outgoing_coefficients = jost_coefficients(
        omega,
        +1,
        0,
        metric_series,
        metric_prime_series,
        potential_series,
        series_order,
    ) * ((-1.0j) ** 3)
    incoming_coefficients = jost_coefficients(
        omega,
        -1,
        0,
        metric_series,
        metric_prime_series,
        potential_series,
        series_order,
    ) * ((1.0j) ** 3)
    outgoing_value, outgoing_derivative = evaluate_jost(
        outgoing_coefficients,
        outer_radius,
        omega,
        +1,
        metric_value,
        tortoise,
    )
    incoming_value, incoming_derivative = evaluate_jost(
        incoming_coefficients,
        outer_radius,
        omega,
        -1,
        metric_value,
        tortoise,
    )
    outgoing_data = integrate(
        outer_radius,
        match,
        outgoing_value[0],
        outgoing_derivative[0],
    )
    incoming_data = integrate(
        outer_radius,
        match,
        incoming_value[0],
        incoming_derivative[0],
    )
    outgoing_amplitude, incoming_amplitude = np.linalg.solve(
        np.column_stack((outgoing_data, incoming_data)), horizon_data
    )
    source_amplitude = outgoing_amplitude + incoming_amplitude
    response_amplitude = 1.0j * (
        outgoing_amplitude - incoming_amplitude
    )
    return complex(response_amplitude / source_amplitude)


def run(args):
    charge = args.charge_ratio * master.extremal_charge("bardeen", 1.0)
    model = master.build_model("bardeen", charge, 1.0)
    controls = master.SolverControls(
        horizon_offset=args.horizon_offset,
        match_radius=args.match_radius,
        asymptotic_cycles=18.0,
        minimum_outer_radius=260.0,
        rtol=args.rtol,
        atol=args.atol,
        max_step_phase=args.max_step_phase,
        max_step_radius=5.0,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_dir = args.output_dir / "results"
    figure_dir = args.output_dir / "figures"
    result_dir.mkdir(exist_ok=True)
    figure_dir.mkdir(exist_ok=True)

    rows = []
    for omega in args.frequencies:
        outer = max(args.minimum_outer_radius, args.outer_cycles / omega)
        bardeen_response, orders = response_matrix_jost(
            model, omega, outer, args.series_order, controls
        )
        direct_response = bardeen_response[0, 0]
        metric_power_ratio = -15.0 * direct_response / (2.0 * omega**5)
        rows.append(
            {
                "omega_M": omega,
                "outer_radius_over_M": outer,
                "bardeen_Rgg_real": bardeen_response[0, 0].real,
                "bardeen_Rgg_imag": bardeen_response[0, 0].imag,
                "direct_Rgg_real": direct_response.real,
                "direct_Rgg_imag": direct_response.imag,
                "unsubtracted_metric_power_ratio_real": metric_power_ratio.real,
                "unsubtracted_metric_power_ratio_imag": metric_power_ratio.imag,
            }
        )

    response_csv = result_dir / "bardeen_polar_jost_metric_response.csv"
    with response_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    convergence_rows = []
    for outer in args.convergence_outer_radii:
        bardeen_response, _ = response_matrix_jost(
            model,
            args.convergence_frequency,
            outer,
            args.series_order,
            controls,
        )
        direct_response = bardeen_response[0, 0]
        convergence_rows.append(
            {
                "outer_radius_over_M": outer,
                "bardeen_Rgg_real": bardeen_response[0, 0].real,
                "bardeen_Rgg_imag": bardeen_response[0, 0].imag,
                "direct_Rgg_real": direct_response.real,
                "direct_Rgg_imag": direct_response.imag,
            }
        )

    convergence_csv = result_dir / "bardeen_polar_jost_outer_convergence.csv"
    with convergence_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(convergence_rows[0])
        )
        writer.writeheader()
        writer.writerows(convergence_rows)

    omega_array = np.array([row["omega_M"] for row in rows])
    delta_array = np.array(
        [row["direct_Rgg_real"] for row in rows]
    )
    figure, axis = plt.subplots(figsize=(6.6, 4.5))
    axis.plot(omega_array, delta_array, "o-")
    axis.set_xlabel(r"$M\omega$")
    axis.set_ylabel(r"$\mathrm{Re}\,\Delta\mathcal{R}_{gg}^{\rm polar}$")
    axis.set_title(
        rf"Bardeen polar metric-wave response: $q/q_{{\rm ext}}={args.charge_ratio:g}$"
    )
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(
        figure_dir / "bardeen_polar_jost_metric_response.png", dpi=220
    )
    figure.savefig(
        figure_dir / "bardeen_polar_jost_metric_response.pdf"
    )
    plt.close(figure)

    outer_array = np.array(
        [row["outer_radius_over_M"] for row in convergence_rows]
    )
    convergence_delta = np.array(
        [row["delta_Rgg_real"] for row in convergence_rows]
    )
    reference = convergence_delta[-1]
    relative = np.abs(
        (convergence_delta - reference)
        / max(abs(reference), 1.0e-30)
    )
    figure, axis = plt.subplots(figsize=(6.6, 4.5))
    axis.semilogy(outer_array, np.maximum(relative, 1.0e-16), "o-")
    axis.set_xlabel(r"outer radius $r_{\rm out}/M$")
    axis.set_ylabel("relative change from largest outer radius")
    axis.set_title(
        rf"Jost-basis convergence at $M\omega={args.convergence_frequency:g}$"
    )
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(
        figure_dir / "bardeen_polar_jost_outer_convergence.png", dpi=220
    )
    figure.savefig(
        figure_dir / "bardeen_polar_jost_outer_convergence.pdf"
    )
    plt.close(figure)

    summary = {
        "model": "bardeen",
        "parity": "polar",
        "charge_ratio": args.charge_ratio,
        "charge": charge,
        "series_order_in_inverse_sqrt_r": args.series_order,
        "effective_asymptotic_orders": [float(value) for value in orders],
        "metric_reconstruction": {
            "source_small_frequency_coefficient": "(2/5) omega^3 r^2",
            "response_small_frequency_coefficient": "-3/(omega^2 r^3)",
            "conversion": "C_response/C_source = -15 B/(2 omega^5 A)",
        },
        "status": (
            "The physical H0 normalization and controlled Jost matching are "
            "implemented.  The table labelled unsubtracted_metric_power_ratio "
            "still contains long-range propagation terms and is not reported as "
            "a dynamical Love number.  An independent finite-part prescription "
            "must be applied before taking the omega-to-zero limit."
        ),
        "controls": {
            "horizon_offset": args.horizon_offset,
            "match_radius": args.match_radius,
            "rtol": args.rtol,
            "atol": args.atol,
            "max_step_phase": args.max_step_phase,
        },
    }
    (result_dir / "bardeen_polar_jost_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--charge-ratio", type=float, default=0.5)
    parser.add_argument(
        "--frequencies",
        type=float,
        nargs="+",
        default=[0.015, 0.020, 0.030, 0.040],
    )
    parser.add_argument("--series-order", type=int, default=14)
    parser.add_argument("--outer-cycles", type=float, default=15.0)
    parser.add_argument("--minimum-outer-radius", type=float, default=400.0)
    parser.add_argument(
        "--convergence-frequency", type=float, default=0.020
    )
    parser.add_argument(
        "--convergence-outer-radii",
        type=float,
        nargs="+",
        default=[500.0, 750.0, 1000.0],
    )
    parser.add_argument("--horizon-offset", type=float, default=1.0e-5)
    parser.add_argument("--match-radius", type=float, default=15.0)
    parser.add_argument("--rtol", type=float, default=3.0e-11)
    parser.add_argument("--atol", type=float, default=3.0e-13)
    parser.add_argument("--max-step-phase", type=float, default=0.20)
    parser.add_argument("--output-dir", type=Path, default=HERE.parent)
    return parser.parse_args()


def main():
    summary = run(parse_arguments())
    print(summary["status"])


if __name__ == "__main__":
    main()
