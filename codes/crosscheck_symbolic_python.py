#!/usr/bin/env python3


from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import sympy as sp

import einstein_nled_master as master


MODELS = ("bardeen", "hayward", "fan_wang")


def symbolic_potentials(model_name: str):
    r, ell, mass, charge = sp.symbols("r ell mass charge", positive=True)
    if model_name == "bardeen":
        m = mass * r**3 / (r**2 + charge**2) ** sp.Rational(3, 2)
        lag = 3 * mass * charge**2 / (r**2 + charge**2) ** sp.Rational(5, 2)
        lag_f = sp.Rational(15, 2) * mass * r**6 / (
            r**2 + charge**2
        ) ** sp.Rational(7, 2)
    elif model_name == "hayward":
        denominator = r**3 + 2 * mass * charge**2
        m = mass * r**3 / denominator
        lag = 6 * mass**2 * charge**2 / denominator**2
        lag_f = 18 * mass**2 * r**7 / denominator**3
    elif model_name == "fan_wang":
        m = mass * r**3 / (r + charge) ** 3
        lag = 3 * mass * charge / (r + charge) ** 4
        lag_f = 6 * mass * r**5 / (charge * (r + charge) ** 5)
    else:
        raise ValueError(model_name)

    f = 1 - 2 * m / r
    log_prime = sp.diff(sp.log(lag_f), r)
    log_second = sp.diff(sp.log(lag_f), r, 2)
    kappa = 1 - r * log_prime / 2
    d_minus = -sp.diff(f, r) * log_prime / 2 + f * (
        -log_second / 2 + log_prime**2 / 4
    )
    d_plus = sp.diff(f, r) * log_prime / 2 + f * (
        log_second / 2 + log_prime**2 / 4
    )
    angular = ell * (ell + 1)
    lam = (ell - 1) * (ell + 2)
    coupling = -charge * sp.sqrt(4 * lam * lag_f) / r**3

    axial = sp.Matrix([
        [angular / r**2 - 6 * m / r**3 + 2 * lag, coupling],
        [coupling, angular / r**2 + d_minus + 4 * charge**2 * lag_f / r**4],
    ])
    a = 6 * m / r - 2 * r**2 * lag
    b = lam + 4 * lag_f * charge**2 / r**2
    denominator = a + lam
    polar_11 = (
        angular * lam - 2 * f * lam + a * (a - 4 * m / r)
    ) / (r**2 * denominator) + 2 * f * lam * b / (r**2 * denominator**2)
    polar_22 = kappa * angular / r**2 + d_plus + (
        4 * lag_f * charge**2
        * (lam + 1 - f + 2 * r**2 * lag + 4 * f * kappa)
        / (r**4 * denominator)
    ) + 8 * f * lag_f * charge**2 * b / (r**4 * denominator**2)
    mixing = (
        (lam + 1 - f + 2 * r**2 * lag + 2 * f * kappa) / denominator
        + 2 * f * b / denominator**2
    )
    polar = sp.Matrix([
        [polar_11, coupling * mixing],
        [coupling * mixing, polar_22],
    ])
    return axial, polar, (r, ell, mass, charge), (m, lag)


def check_model(model_name: str) -> dict:
    axial, polar, symbols, background = symbolic_potentials(model_name)
    r, ell, mass, charge = symbols
    m, lag = background
    evaluators = {
        "axial": sp.lambdify(symbols, axial, "numpy"),
        "polar": sp.lambdify(symbols, polar, "numpy"),
    }
    rng = np.random.default_rng(271828)
    maximum_relative_error = {"axial": 0.0, "polar": 0.0}
    samples = 60
    q_ext = master.extremal_charge(model_name, 1.0)
    for _ in range(samples):
        ratio = float(rng.uniform(0.05, 0.97))
        numerical_model = master.build_model(model_name, ratio * q_ext)
        horizon = master.outer_horizon(numerical_model)
        radius = float(rng.uniform(1.03 * horizon, 60.0))
        for parity in ("axial", "polar"):
            symbolic = np.asarray(
                evaluators[parity](radius, 2, 1.0, numerical_model.charge),
                dtype=float,
            )
            numerical = master.potential_matrix(numerical_model, parity, radius, 2)
            scale = np.maximum(np.abs(numerical), 1.0e-14)
            error = float(np.max(np.abs(symbolic - numerical) / scale))
            maximum_relative_error[parity] = max(maximum_relative_error[parity], error)

    identity = sp.simplify(sp.diff(m, r) - r**2 * lag) == 0
    symmetry = {
        "axial": sp.simplify(axial[0, 1] - axial[1, 0]) == 0,
        "polar": sp.simplify(polar[0, 1] - polar[1, 0]) == 0,
    }
    if not identity or not all(symmetry.values()):
        raise AssertionError(f"symbolic identity failed for {model_name}")
    if max(maximum_relative_error.values()) > 3.0e-12:
        raise AssertionError(f"symbolic/Python mismatch for {model_name}")
    return {
        "background_identity_exact": identity,
        "potential_symmetry_exact": symmetry,
        "random_samples": samples,
        "maximum_relative_potential_error": maximum_relative_error,
    }


def main() -> None:
    report = {model_name: check_model(model_name) for model_name in MODELS}
    report["status"] = "passed"
    path = Path(__file__).resolve().parent / "results" / "symbolic_python_crosscheck.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
