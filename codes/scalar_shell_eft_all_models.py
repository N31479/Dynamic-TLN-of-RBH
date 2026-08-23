#!/usr/bin/env python3



from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from functools import lru_cache
from math import factorial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import polynomial as nppoly
from scipy.integrate import solve_ivp
from scipy.special import spherical_jn

import einstein_nled_master as master

MASS = 1.0
ELL = 2
MODELS = tuple(os.environ.get("SCALAR_MODELS", "bardeen,hayward,fan_wang").split(","))
RATIOS = np.array([0.10, 0.20, 0.30, 0.40, 0.55, 0.70, 0.85, 0.90, 0.95])
OMEGAS = (1.0e-4, 2.0e-4, 5.0e-4)
SHELL_RADII = np.array([12.0, 16.0, 20.0, 24.0, 30.0, 40.0, 50.0, 60.0, 80.0])
DIRECT_MATCH_RADIUS = 24.0
BASIS_REFERENCE_RADIUS = 120.0
SERIES_ORDER = 28
FREQUENCY_ORDER = 4  
FIT_DEGREE = 3


def geometry(model_name: str, charge: float):
    if abs(charge) < 1.0e-16:
        return (
            lambda radius: 1.0 - 2.0 * MASS / np.asarray(radius),
            lambda radius: 2.0 * MASS / np.asarray(radius) ** 2,
            2.0 * MASS,
        )
    model = master.build_model(model_name, charge, MASS)
    return model.f, model.fp, master.outer_horizon(model)


def scalar_potential(radius, metric, metric_prime):
    radius = np.asarray(radius)
    return metric(radius) * (
        ELL * (ELL + 1) / radius**2 + metric_prime(radius) / radius
    )


def _full_rhs(model_name: str, charge: float, omega: float):
    metric, metric_prime, _ = geometry(model_name, charge)

    def rhs(radius, state):
        psi, derivative = state
        f_value = float(metric(radius))
        fp_value = float(metric_prime(radius))
        coefficient = (
            omega**2 / f_value**2
            - ELL * (ELL + 1) / (f_value * radius**2)
            - fp_value / (f_value * radius)
        )
        return np.array(
            [derivative, -fp_value * derivative / f_value - coefficient * psi],
            dtype=complex,
        )

    return rhs


@lru_cache(maxsize=None)
def outside_profile_cached(
    model_name: str,
    charge: float,
    omega: float,
    radii_key: tuple[float, ...],
    horizon_offset: float = 1.0e-6,
    rtol: float = 1.0e-11,
    atol: float = 1.0e-13,
    max_step: float = 0.15,
):
    radii = np.asarray(radii_key, dtype=float)
    metric, metric_prime, horizon = geometry(model_name, charge)
    start = horizon + horizon_offset
    if radii[0] <= start:
        raise ValueError("all sample radii must lie outside the integration start")

    def rhs_tortoise(radius, state):
        psi, momentum = state
        f_value = float(metric(radius))
        potential = float(scalar_potential(radius, metric, metric_prime))
        return np.array(
            [momentum / f_value, (potential - omega**2) * psi / f_value],
            dtype=complex,
        )

    result = solve_ivp(
        rhs_tortoise,
        (start, float(radii[-1])),
        np.array([1.0 + 0.0j, -1.0j * omega], dtype=complex),
        method="DOP853",
        t_eval=radii,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
    )
    if not result.success:
        raise RuntimeError(result.message)
    values = result.y[0]
    derivatives = result.y[1] / np.asarray(metric(radii), dtype=float)
    return values, derivatives


def outside_profile(model_name: str, charge: float, omega: float, radii, **options):
    radii = np.asarray(radii, dtype=float)
    if radii.ndim != 1 or len(radii) == 0 or np.any(np.diff(radii) <= 0):
        raise ValueError("radii must be a nonempty strictly increasing array")
    return outside_profile_cached(
        model_name,
        float(charge),
        float(omega),
        tuple(float(value) for value in radii),
        **options,
    )


def metric_series(model_name: str, charge: float, order: int) -> np.ndarray:
    coefficients = np.zeros(order + 1)
    coefficients[0] = 1.0
    if model_name == "bardeen":
        binomial = 1.0
        charge_power = 1.0
        for k in range((order - 1) // 2 + 1):
            if k > 0:
                binomial *= (-1.5 - (k - 1)) / k
                charge_power *= charge**2
            index = 1 + 2 * k
            if index <= order:
                coefficients[index] = -2.0 * MASS * binomial * charge_power
    elif model_name == "hayward":
        for k in range((order - 1) // 3 + 1):
            index = 1 + 3 * k
            coefficients[index] = -2.0 * MASS * (-2.0 * MASS * charge**2) ** k
    elif model_name == "fan_wang":
        for k in range(order):
            index = 1 + k
            if index <= order:
                coefficients[index] = (
                    -2.0
                    * MASS
                    * (-1.0) ** k
                    * (k + 1)
                    * (k + 2)
                    * charge**k
                    / 2.0
                )
    else:
        raise ValueError(model_name)
    return coefficients


def _trim(polynomial, tolerance: float = 1.0e-13):
    polynomial = np.asarray(polynomial, dtype=float)
    while len(polynomial) > 1 and abs(polynomial[-1]) < tolerance:
        polynomial = polynomial[:-1]
    return polynomial


def _poly_add(first, second):
    result = np.zeros(max(len(first), len(second)))
    result[: len(first)] += first
    result[: len(second)] += second
    return _trim(result)


def _poly_scale(polynomial, coefficient):
    return _trim(np.asarray(polynomial, dtype=float) * coefficient)


def _poly_derivative(polynomial):
    if len(polynomial) <= 1:
        return np.zeros(1)
    return np.arange(1, len(polynomial), dtype=float) * polynomial[1:]


def _operator_polynomial(polynomial, power: int, metric_index: int, include_l: bool):
    
    polynomial = np.asarray(polynomial, dtype=float)
    first = _poly_derivative(polynomial)
    second = _poly_derivative(first)
    result = np.zeros(max(len(polynomial), len(first), len(second)))
    coefficient = (power + 1) * (power + metric_index)
    if include_l:
        coefficient -= ELL * (ELL + 1)
    result[: len(polynomial)] += coefficient * polynomial
    result[: len(first)] += (2 * power + 1 + metric_index) * first
    result[: len(second)] += second
    return _trim(result)


def _solve_leading_operator(rhs, power: int):
    





    rhs = _trim(rhs)
    indicial = power * (power + 1) - ELL * (ELL + 1)
    if abs(indicial) > 1.0e-12:
        degree = len(rhs) - 1
        matrix = np.zeros((degree + 1, degree + 1))
        for column in range(degree + 1):
            unit = np.zeros(degree + 1)
            unit[column] = 1.0
            image = _operator_polynomial(unit, power, 0, True)
            matrix[: len(image), column] = image
        return _trim(np.linalg.solve(matrix, rhs))

    
    degree = len(rhs)
    matrix = np.zeros((degree + 1, degree + 1))
    target = np.zeros(degree + 1)
    target[: len(rhs)] = rhs
    for column in range(degree + 1):
        unit = np.zeros(degree + 1)
        unit[column] = 1.0
        image = _operator_polynomial(unit, power, 0, True)
        matrix[: len(image), column] = image
    matrix[-1, :] = 0.0
    matrix[-1, 0] = 1.0
    target[-1] = 0.0
    return _trim(np.linalg.solve(matrix, target))


def _inverse_series(coefficients):
    inverse = np.zeros_like(coefficients)
    inverse[0] = 1.0 / coefficients[0]
    for n in range(1, len(coefficients)):
        inverse[n] = -sum(
            coefficients[k] * inverse[n - k] for k in range(1, n + 1)
        ) / coefficients[0]
    return inverse


@lru_cache(maxsize=None)
def perturbative_basis_coefficients(
    model_name: str,
    charge: float,
    branch: str,
    series_order: int = SERIES_ORDER,
    frequency_order: int = FREQUENCY_ORDER,
):
    
    if branch not in {"source", "response"}:
        raise ValueError(branch)
    base_power = -(ELL + 1) if branch == "source" else ELL
    metric_coefficients = metric_series(model_name, charge, series_order)
    inverse_metric = _inverse_series(metric_coefficients)

    levels: list[tuple[int, list[np.ndarray]]] = []

    
    static = [np.array([1.0])]
    for n in range(1, series_order + 1):
        lower = np.zeros(1)
        for k in range(1, n + 1):
            image = _operator_polynomial(
                static[n - k], base_power + n - k, k, False
            )
            lower = _poly_add(lower, _poly_scale(image, metric_coefficients[k]))
        static.append(_solve_leading_operator(_poly_scale(lower, -1.0), base_power + n))
    levels.append((base_power, static))

    
    for frequency_index in range(1, frequency_order + 1):
        start_power = base_power - 2 * frequency_index
        previous_power, previous = levels[frequency_index - 1]
        if previous_power != start_power + 2:
            raise RuntimeError("inconsistent perturbative power bookkeeping")
        current: list[np.ndarray] = []
        for n in range(series_order + 1):
            rhs = np.zeros(1)
            for j in range(n + 1):
                rhs = _poly_add(
                    rhs, _poly_scale(previous[j], -inverse_metric[n - j])
                )
            lower = np.zeros(1)
            for k in range(1, n + 1):
                image = _operator_polynomial(
                    current[n - k], start_power + n - k, k, False
                )
                lower = _poly_add(
                    lower, _poly_scale(image, metric_coefficients[k])
                )
            current.append(
                _solve_leading_operator(
                    _poly_add(rhs, _poly_scale(lower, -1.0)), start_power + n
                )
            )
        levels.append((start_power, current))
    return tuple((power, tuple(tuple(poly) for poly in polynomials)) for power, polynomials in levels)


def _evaluate_level(power0: int, polynomials, radius: float):
    x = 1.0 / radius
    logarithm = np.log(2.0 * MASS * x)
    value = 0.0
    derivative_x = 0.0
    for n, polynomial_tuple in enumerate(polynomials):
        polynomial = np.asarray(polynomial_tuple, dtype=float)
        power = power0 + n
        polynomial_value = nppoly.polyval(logarithm, polynomial)
        polynomial_derivative = nppoly.polyval(
            logarithm, _poly_derivative(polynomial)
        )
        value += x**power * polynomial_value
        derivative_x += x ** (power - 1) * (
            power * polynomial_value + polynomial_derivative
        )
    return value, -x**2 * derivative_x


def perturbative_boundary(model_name, charge, omega, branch, radius):
    levels = perturbative_basis_coefficients(model_name, float(charge), branch)
    value = 0.0
    derivative = 0.0
    for frequency_index, (power, polynomials) in enumerate(levels):
        level_value, level_derivative = _evaluate_level(power, polynomials, radius)
        factor = omega ** (2 * frequency_index)
        value += factor * level_value
        derivative += factor * level_derivative
    return value, derivative


@lru_cache(maxsize=None)
def basis_profile_cached(
    model_name: str,
    charge: float,
    omega: float,
    branch: str,
    radii_key: tuple[float, ...],
    reference_radius: float = BASIS_REFERENCE_RADIUS,
):
    radii = np.asarray(radii_key, dtype=float)
    if radii[-1] >= reference_radius:
        raise ValueError("basis reference radius must exceed all sample radii")
    initial = perturbative_boundary(
        model_name, charge, omega, branch, reference_radius
    )
    descending = radii[::-1]
    result = solve_ivp(
        _full_rhs(model_name, charge, omega),
        (reference_radius, float(descending[-1])),
        np.asarray(initial, dtype=complex),
        method="DOP853",
        t_eval=descending,
        rtol=1.0e-11,
        atol=1.0e-13,
        max_step=0.20,
    )
    if not result.success:
        raise RuntimeError(result.message)
    return result.y[0][::-1], result.y[1][::-1]


def basis_profile(model_name, charge, omega, branch, radii, reference_radius=BASIS_REFERENCE_RADIUS):
    radii = np.asarray(radii, dtype=float)
    return basis_profile_cached(
        model_name,
        float(charge),
        float(omega),
        branch,
        tuple(float(value) for value in radii),
        float(reference_radius),
    )



@lru_cache(maxsize=None)
def basis_pair_profile_cached(
    model_name: str,
    charge: float,
    omega: float,
    radii_key: tuple[float, ...],
    reference_radius: float = BASIS_REFERENCE_RADIUS,
):
    
    radii = np.asarray(radii_key, dtype=float)
    if radii[-1] >= reference_radius:
        raise ValueError("basis reference radius must exceed all sample radii")
    source_initial = perturbative_boundary(
        model_name, charge, omega, "source", reference_radius
    )
    response_initial = perturbative_boundary(
        model_name, charge, omega, "response", reference_radius
    )
    metric, metric_prime, _ = geometry(model_name, charge)

    def rhs(radius, state):
        f_value = float(metric(radius))
        fp_value = float(metric_prime(radius))
        coefficient = (
            omega**2 / f_value**2
            - ELL * (ELL + 1) / (f_value * radius**2)
            - fp_value / (f_value * radius)
        )
        output = np.empty(4, dtype=complex)
        output[0] = state[1]
        output[1] = -fp_value * state[1] / f_value - coefficient * state[0]
        output[2] = state[3]
        output[3] = -fp_value * state[3] / f_value - coefficient * state[2]
        return output

    descending = radii[::-1]
    initial = np.array(
        [source_initial[0], source_initial[1], response_initial[0], response_initial[1]],
        dtype=complex,
    )
    result = solve_ivp(
        rhs,
        (reference_radius, float(descending[-1])),
        initial,
        method="DOP853",
        t_eval=descending,
        rtol=1.0e-10,
        atol=1.0e-12,
        max_step=0.30,
    )
    if not result.success:
        raise RuntimeError(result.message)
    source = result.y[0][::-1]
    source_derivative = result.y[1][::-1]
    response = result.y[2][::-1]
    response_derivative = result.y[3][::-1]
    return source, source_derivative, response, response_derivative


def basis_pair_profile(
    model_name, charge, omega, radii, reference_radius=BASIS_REFERENCE_RADIUS
):
    radii = np.asarray(radii, dtype=float)
    return basis_pair_profile_cached(
        model_name,
        float(charge),
        float(omega),
        tuple(float(value) for value in radii),
        float(reference_radius),
    )


def solution_bundle(model_name, charge, omega, radii=SHELL_RADII):
    
    radii = np.asarray(radii, dtype=float)
    full, full_derivative = outside_profile(model_name, charge, omega, radii)
    source, source_derivative, response, response_derivative = basis_pair_profile(
        model_name, charge, omega, radii
    )
    return {
        "radii": radii,
        "full": full,
        "full_derivative": full_derivative,
        "source": source,
        "source_derivative": source_derivative,
        "response": response,
        "response_derivative": response_derivative,
    }


def response_from_bundle(bundle, match_radius=DIRECT_MATCH_RADIUS):
    radii = bundle["radii"]
    index = int(np.argmin(np.abs(radii - match_radius)))
    if abs(radii[index] - match_radius) > 1.0e-10:
        raise ValueError("match radius is not present in the bundle grid")
    matrix = np.array(
        [
            [bundle["source"][index], bundle["response"][index]],
            [
                bundle["source_derivative"][index],
                bundle["response_derivative"][index],
            ],
        ],
        dtype=complex,
    )
    vector = np.array(
        [bundle["full"][index], bundle["full_derivative"][index]],
        dtype=complex,
    )
    coefficients = np.linalg.solve(matrix, vector)
    return coefficients[1] / coefficients[0]


def renormalized_profile_from_bundle(model_name, charge, bundle):
    




    radii = bundle["radii"]
    values = []
    for index, radius in enumerate(radii):
        prefactor = shell_prefactor(model_name, charge, float(radius))
        source_log_derivative = (
            bundle["source_derivative"][index] / bundle["source"][index]
        )
        full_log_derivative = (
            bundle["full_derivative"][index] / bundle["full"][index]
        )
        values.append(prefactor * (source_log_derivative - full_log_derivative))
    return np.asarray(values)


def extrapolate_profile_values(radii, values, degree=FIT_DEGREE):
    radii = np.asarray(radii, dtype=float)
    values = np.asarray(values, dtype=complex)
    x = 2.0 * MASS / radii
    design = np.column_stack([x**power for power in range(degree + 1)])
    real_parameters, _, _, _ = np.linalg.lstsq(design, values.real, rcond=None)
    imag_parameters, _, _, _ = np.linalg.lstsq(design, values.imag, rcond=None)
    parameters = real_parameters + 1.0j * imag_parameters
    fitted = design @ parameters
    residual = values - fitted
    response = complex(parameters[0])
    return {
        "response": response,
        "fit": fitted,
        "relative_rms_residual": float(
            np.sqrt(np.mean(np.abs(residual) ** 2))
            / max(abs(response), 1.0e-30)
        ),
        "maximum_absolute_residual": float(np.max(np.abs(residual))),
        "coefficients": [[float(v.real), float(v.imag)] for v in parameters],
    }

def direct_response(model_name: str, charge: float, omega: float, match_radius=DIRECT_MATCH_RADIUS, **solver_options):
    radii = np.array([float(match_radius)])
    full, full_derivative = outside_profile(
        model_name, charge, omega, radii, **solver_options
    )
    source, source_derivative = basis_profile(
        model_name, charge, omega, "source", radii
    )
    response, response_derivative = basis_profile(
        model_name, charge, omega, "response", radii
    )
    coefficients = np.linalg.solve(
        np.array(
            [
                [source[0], response[0]],
                [source_derivative[0], response_derivative[0]],
            ],
            dtype=complex,
        ),
        np.array([full[0], full_derivative[0]], dtype=complex),
    )
    return coefficients[1] / coefficients[0]


def inside_solution(model_name: str, charge: float, omega: float, radius: float):
    
    metric, _, _ = geometry(model_name, charge)
    f_value = float(metric(radius))
    if f_value <= 0.0:
        raise ValueError("the shell must lie outside the outer horizon")
    root_f = np.sqrt(f_value)
    argument = omega * radius / root_f
    bessel = spherical_jn(ELL, argument)
    bessel_prime = spherical_jn(ELL, argument, derivative=True)
    psi = radius * bessel
    derivative = bessel + (omega * radius / f_value) * bessel_prime
    return psi, derivative


def shell_prefactor(model_name: str, charge: float, radius: float):
    metric, _, _ = geometry(model_name, charge)
    f_value = float(metric(radius))
    prefactor = (
        4.0
        * np.pi
        * factorial(ELL)
        * 2.0**ELL
        * radius ** (2 * ELL + 2)
        / factorial(2 * ELL + 1)
    )
    return prefactor * f_value ** ((2 * ELL + 1) / 2.0)


def derivative_jump_response(model_name, charge, omega, radii, exterior_values, exterior_derivatives):
    responses = []
    for radius, outside, outside_derivative in zip(
        radii, exterior_values, exterior_derivatives
    ):
        inside, inside_derivative = inside_solution(
            model_name, charge, omega, float(radius)
        )
        logarithmic_jump = inside_derivative / inside - outside_derivative / outside
        responses.append(
            shell_prefactor(model_name, charge, float(radius)) * logarithmic_jump
        )
    return np.asarray(responses)


def renormalized_shell_profile(model_name: str, charge: float, omega: float, radii=SHELL_RADII):
    
    radii = np.asarray(radii, dtype=float)
    full, full_derivative = outside_profile(model_name, charge, omega, radii)
    source, source_derivative = basis_profile(
        model_name, charge, omega, "source", radii
    )

    
    
    full_jump = derivative_jump_response(
        model_name, charge, omega, radii, full, full_derivative
    )
    source_jump = derivative_jump_response(
        model_name, charge, omega, radii, source, source_derivative
    )
    return full_jump - source_jump


def relative_rms(model: np.ndarray, target: np.ndarray) -> float:
    scale = max(float(np.sqrt(np.mean(np.abs(target) ** 2))), 1.0e-30)
    return float(np.sqrt(np.mean(np.abs(model - target) ** 2)) / scale)


def run():
    root = Path(__file__).resolve().parent
    results_dir = root / "results"
    results_dir.mkdir(exist_ok=True)
    rows = []
    summary = {}

    for model_name in MODELS:
        print(f"running {model_name}", flush=True)
        extremal = master.extremal_charge(model_name, MASS)
        model_summary = {"fit_degree": FIT_DEGREE, "points": {}}
        direct_values_by_frequency = {}
        shell_values_by_frequency = {}

        for omega in OMEGAS:
            direct_values = []
            shell_values = []
            fit_residuals = []
            for ratio in RATIOS:
                charge = ratio * extremal
                bundle = solution_bundle(model_name, charge, omega, SHELL_RADII)
                direct = response_from_bundle(bundle)
                shell_profile = (3.0 / (4.0 * np.pi)) * (
                    renormalized_profile_from_bundle(model_name, charge, bundle)
                )
                shell_fit = extrapolate_profile_values(
                    SHELL_RADII, shell_profile, FIT_DEGREE
                )
                shell = shell_fit["response"]
                direct_values.append(direct)
                shell_values.append(shell)
                fit_residuals.append(shell_fit["relative_rms_residual"])
                rows.append(
                    {
                        "model": model_name,
                        "omega_M": omega,
                        "ell_over_ell_ext": ratio,
                        "direct_real": direct.real,
                        "direct_imag": direct.imag,
                        "shell_real": shell.real,
                        "shell_imag": shell.imag,
                        "real_residual": shell.real - direct.real,
                        "imag_residual": shell.imag - direct.imag,
                        "shell_fit_relative_rms": shell_fit["relative_rms_residual"],
                    }
                )
            direct_array = np.asarray(direct_values)
            shell_array = np.asarray(shell_values)
            direct_values_by_frequency[omega] = direct_array
            shell_values_by_frequency[omega] = shell_array
            model_summary["points"][f"{omega:.1e}"] = {
                "real_relative_rms": relative_rms(shell_array.real, direct_array.real),
                "maximum_absolute_real_residual": float(
                    np.max(np.abs(shell_array.real - direct_array.real))
                ),
                "complex_relative_rms": relative_rms(shell_array, direct_array),
                "maximum_shell_fit_relative_rms": float(max(fit_residuals)),
            }

        
        charge = 0.70 * extremal
        omega = 5.0e-4

        control_radii = np.array([18.0, 24.0, 30.0])
        control_bundle = solution_bundle(model_name, charge, omega, control_radii)
        direct_radius_values = []
        for control_radius in control_radii:
            direct_radius_values.append(
                response_from_bundle(control_bundle, float(control_radius))
            )

        reference_radius_values = []
        single_radius = np.array([DIRECT_MATCH_RADIUS])
        full, full_derivative = outside_profile(
            model_name, charge, omega, single_radius
        )
        for reference_radius in (100.0, 120.0, 140.0):
            source, source_derivative, response, response_derivative = basis_pair_profile(
                model_name, charge, omega, single_radius, reference_radius
            )
            coefficients = np.linalg.solve(
                np.array(
                    [[source[0], response[0]], [source_derivative[0], response_derivative[0]]],
                    dtype=complex,
                ),
                np.array([full[0], full_derivative[0]], dtype=complex),
            )
            reference_radius_values.append(
                coefficients[1] / coefficients[0]
            )

        
        master_radii = SHELL_RADII
        master_bundle = solution_bundle(model_name, charge, omega, master_radii)
        master_profile = (3.0 / (4.0 * np.pi)) * (
            renormalized_profile_from_bundle(model_name, charge, master_bundle)
        )
        window_results = {}
        for label, lower in (("12-80M", 12.0), ("16-80M", 16.0), ("20-80M", 20.0)):
            mask = master_radii >= lower
            fit = extrapolate_profile_values(
                master_radii[mask], master_profile[mask], FIT_DEGREE
            )
            window_results[label] = {
                "response": [fit["response"].real, fit["response"].imag],
                "relative_rms_residual": fit["relative_rms_residual"],
            }

        model_summary["controls"] = {
            "test_point": {"ell_over_ell_ext": 0.70, "omega_M": omega},
            "direct_match_radius_values": [
                [value.real, value.imag] for value in direct_radius_values
            ],
            "basis_reference_radius_values": [
                [value.real, value.imag] for value in reference_radius_values
            ],
            "shell_windows": window_results,
        }
        summary[model_name] = model_summary

        
        
        outside_profile_cached.cache_clear()
        basis_profile_cached.cache_clear()
        basis_pair_profile_cached.cache_clear()
        perturbative_basis_coefficients.cache_clear()

    csv_path = results_dir / "scalar_shell_eft_all_models_direct.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (results_dir / "scalar_shell_eft_all_models_direct.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    figure, axes = plt.subplots(3, 2, figsize=(10.8, 11.5))
    for row_index, model_name in enumerate(MODELS):
        for omega in OMEGAS:
            selected = sorted(
                [
                    row
                    for row in rows
                    if row["model"] == model_name
                    and np.isclose(row["omega_M"], omega)
                ],
                key=lambda row: row["ell_over_ell_ext"],
            )
            ratio = np.array([row["ell_over_ell_ext"] for row in selected])
            direct = np.array([row["direct_real"] for row in selected])
            shell = np.array([row["shell_real"] for row in selected])
            axes[row_index, 0].plot(
                ratio,
                direct,
                "o-",
                ms=3.3,
                lw=1.25,
                label=rf"direct, $M\omega={omega:g}$",
            )
            axes[row_index, 0].plot(
                ratio,
                shell,
                "--",
                lw=1.35,
                label=rf"shell, $M\omega={omega:g}$",
            )
            axes[row_index, 1].plot(
                ratio,
                shell - direct,
                "o-",
                ms=3.3,
                lw=1.25,
                label=rf"$M\omega={omega:g}$",
            )
        axes[row_index, 0].set_title(model_name.replace("_", "-").title())
        axes[row_index, 0].set_ylabel(r"renormalized scalar response $\Lambda_0$")
        axes[row_index, 1].set_ylabel("shell minus direct")
        for axis in axes[row_index]:
            axis.axhline(0.0, lw=0.7)
            axis.set_xlabel(r"$\ell/\ell_{\rm ext}$")
            axis.grid(alpha=0.22)
            axis.legend(frameon=False, fontsize=7.2)
    figure.suptitle(
        "Scalar shell EFT: direct response vs same-background-renormalized shell response", y=0.995
    )
    figure.tight_layout()
    figure_dir = root.parent / "figures"
    figure_dir.mkdir(exist_ok=True)
    figure.savefig(figure_dir / "scalar_shell_eft_all_models.png", dpi=240)
    figure.savefig(figure_dir / "scalar_shell_eft_all_models.pdf")
    plt.close(figure)


def run_all_models_in_fresh_processes():
    





    root = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="scalar_shell_partials_") as temporary:
        partial = Path(temporary)
        for model_name in ("bardeen", "hayward", "fan_wang"):
            environment = os.environ.copy()
            environment["SCALAR_MODELS"] = model_name
            subprocess.run([sys.executable, str(Path(__file__).resolve())], cwd=root, env=environment, check=True)
            shutil.copy2(root / "results" / "scalar_shell_eft_all_models_direct.csv", partial / f"{model_name}.csv")
            shutil.copy2(root / "results" / "scalar_shell_eft_all_models_direct.json", partial / f"{model_name}.json")
        environment = os.environ.copy()
        environment["SCALAR_PARTIAL_DIR"] = str(partial)
        subprocess.run([sys.executable, str(root / "merge_scalar_shell_results.py")], cwd=root, env=environment, check=True)


if __name__ == "__main__":
    if "SCALAR_MODELS" not in os.environ and len(MODELS) > 1:
        run_all_models_in_fresh_processes()
    else:
        run()
