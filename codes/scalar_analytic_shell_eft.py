#!/usr/bin/env python3


from __future__ import annotations

import json
from pathlib import Path

import sympy as sp


def metric_function(model_name: str, radius, mass, charge):
    
    if model_name == "bardeen":
        return 1 - 2 * mass * radius**2 / (radius**2 + charge**2) ** sp.Rational(3, 2)
    if model_name == "hayward":
        return 1 - 2 * mass * radius**2 / (radius**3 + 2 * mass * charge**2)
    if model_name == "fan_wang":
        return 1 - 2 * mass * radius**2 / (radius + charge) ** 3
    raise ValueError(model_name)


def static_source_series(model_name: str, order: int = 7):
    x, mass, charge = sp.symbols("x M q", positive=True)
    metric = metric_function(model_name, 1 / x, mass, charge)

    metric_series = sp.series(metric, x, 0, order + 1).removeO().expand()
    metric_coefficients = [metric_series.coeff(x, k) for k in range(order + 1)]
    multipole = 2
    leading_power = -(multipole + 1)
    constants = [sp.Integer(0)] * (order + 1)
    logarithms = [sp.Integer(0)] * (order + 1)
    constants[0] = sp.Integer(1)

    for n in range(1, order + 1):
        power = leading_power + n
        lower_log = sp.Integer(0)
        lower_constant = sp.Integer(0)
        for k in range(1, n + 1):
            previous_power = leading_power + n - k
            factor = (
                previous_power * (previous_power + 1)
                + k * previous_power
                + k
            )
            lower_log += metric_coefficients[k] * logarithms[n - k] * factor
            lower_constant += metric_coefficients[k] * (
                constants[n - k] * factor
                + logarithms[n - k] * (2 * previous_power + 1 + k)
            )
        indicial = power * (power + 1) - multipole * (multipole + 1)
        if indicial == 0:
            logarithms[n] = sp.simplify(-lower_constant / (2 * power + 1))
        else:
            logarithms[n] = sp.simplify(-lower_log / indicial)
            constants[n] = sp.simplify(
                -((2 * power + 1) * logarithms[n] + lower_constant) / indicial
            )
    return x, mass, charge, constants, logarithms


def verify_static_series(model_name, mass, charge, constants, logarithms):
    





    x = sp.symbols("x", positive=True)
    radius = sp.symbols("r", positive=True)
    leading_power = -3
    series_x = sp.Integer(0)
    for n, (constant, logarithm) in enumerate(zip(constants[:6], logarithms[:6])):
        power = leading_power + n
        series_x += x**power * (constant + logarithm * sp.log(2 * mass * x))
    psi = series_x.subs(x, 1 / radius)
    metric = metric_function(model_name, radius, mass, charge)
    radial_equation = (
        metric * sp.diff(psi, radius, 2)
        + sp.diff(metric, radius) * sp.diff(psi, radius)
        - (6 / radius**2 + sp.diff(metric, radius) / radius) * psi
    )
    residual_x = radial_equation.subs(radius, 1 / x)
    residual_series = sp.series(residual_x, x, 0, 5).removeO().expand()
    if sp.simplify(residual_series) != 0:
        raise AssertionError(
            f"static scalar series substitution failed for {model_name}: "
            f"{sp.factor(residual_series)}"
        )
    return residual_series


def main() -> None:
    expected = {
        "bardeen": lambda m, q: 3 * m * q**2 * (5 * q**2 - 4 * m**2) / 5,
        "hayward": lambda m, q: 32 * m**3 * q**2 / 5,
        "fan_wang": lambda m, q: 8 * m * q**2 * (6 * m**2 + 20 * m * q + 15 * q**2) / 5,
    }
    report = {}
    for model_name in expected:
        _, mass, charge, constants, logarithms = static_source_series(model_name)
        
        
        lambda_log = sp.factor(-logarithms[5])
        reference = sp.factor(expected[model_name](mass, charge))
        if sp.simplify(lambda_log - reference) != 0:
            raise AssertionError(f"logarithmic coefficient failed for {model_name}")
        static_residual = verify_static_series(
            model_name, mass, charge, constants, logarithms
        )

        lambda_finite = sp.symbols("lambda_fin")
        log_radius = sp.symbols("L")
        dynamic_particular = (
            lambda_log * log_radius / 6
            + (lambda_finite - lambda_log / 6) / 6
        )
        
        
        radius = sp.symbols("r", positive=True)
        trial = lambda_log * sp.log(radius) / 6 + (
            lambda_finite - lambda_log / 6
        ) / 6
        target = -(lambda_log * sp.log(radius) + lambda_finite) / radius**2
        residual = sp.simplify(sp.diff(trial, radius, 2) - 6 * trial / radius**2 - target)
        if residual != 0:
            raise AssertionError(f"dynamic particular solution failed for {model_name}")

        report[model_name] = {
            "lambda_log": str(lambda_log),
            "scheme_independent_static_shell_log_coefficient": str(
                sp.factor(4 * sp.pi * lambda_log / 3)
            ),
            "scheme_dependent_static_shell_finite_part": str(
                sp.factor(4 * sp.pi * (5 * lambda_finite - lambda_log) / 15)
            ),
            "scheme_independent_omega2_log_coefficient": str(
                sp.factor(lambda_log / 6)
            ),
            "omega2_scale_invariant_particular_combination": str(
                sp.factor(
                    lambda_log * log_radius / 6
                    + (lambda_finite - lambda_log / 6) / 6
                )
            ),
            "scheme_dependent_omega2_finite_data": str(
                lambda_finite / 6 + sp.Symbol("lambda_2") / sp.Symbol("r") ** 2
            ),
            "note_on_rational_companion": (
                "The -lambda_log/36 term is fixed in this displayed particular "
                "solution but is not an independently scheme-invariant Wilson coefficient."
            ),
            "static_source_log_x_coefficient": str(sp.factor(logarithms[5])),
            "dynamic_particular": str(sp.factor(dynamic_particular)),
            "source_series_constants_through_resonance": [
                str(sp.factor(value)) for value in constants[:6]
            ],
            "source_series_logarithms_through_resonance": [
                str(sp.factor(value)) for value in logarithms[:6]
            ],
            "static_exact_equation_residual_through_checked_order": str(static_residual),
        }
    report["status"] = "passed"
    output = Path(__file__).resolve().parent / "results" / "scalar_analytic_shell_eft.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
