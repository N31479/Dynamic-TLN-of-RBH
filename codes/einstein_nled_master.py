#!/usr/bin/env python3


from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.special import jv, jvp, yv, yvp


Array = np.ndarray


@dataclass(frozen=True)
class Model:
    name: str
    mass: float
    charge: float
    extremal_charge: float
    f: Callable[[Array], Array]
    fp: Callable[[Array], Array]
    mass_function: Callable[[Array], Array]
    lagrangian: Callable[[Array], Array]
    lagrangian_f: Callable[[Array], Array]
    kappa: Callable[[Array], Array]
    dminus: Callable[[Array], Array]
    dplus: Callable[[Array], Array]


@dataclass(frozen=True)
class SolverControls:
    multipole: int = 2
    horizon_offset: float = 2.0e-5
    match_radius: float = 18.0
    asymptotic_cycles: float = 70.0
    minimum_outer_radius: float = 800.0
    rtol: float = 2.0e-10
    atol: float = 2.0e-12
    max_step_phase: float = 0.20
    max_step_radius: float = 0.80


def extremal_charge(model_name: str, mass: float) -> float:
    if model_name in {"bardeen", "hayward"}:
        return 4.0 * mass / (3.0 * np.sqrt(3.0))
    if model_name == "fan_wang":
        return 8.0 * mass / 27.0
    raise ValueError(f"Unknown model: {model_name}")


def build_model(model_name: str, charge: float, mass: float = 1.0) -> Model:
    

    if mass <= 0.0:
        raise ValueError("The mass must be positive.")
    qext = extremal_charge(model_name, mass)
    if not 0.0 < charge < qext:
        raise ValueError(f"Require 0 < charge < {qext:.12g} for {model_name}.")

    if model_name == "bardeen":
        def mass_function(radius: Array) -> Array:
            r = np.asarray(radius, dtype=float)
            return mass * r**3 / (r**2 + charge**2) ** 1.5

        def lagrangian(radius: Array) -> Array:
            r = np.asarray(radius, dtype=float)
            return 3.0 * mass * charge**2 / (r**2 + charge**2) ** 2.5

        def lagrangian_f(radius: Array) -> Array:
            r = np.asarray(radius, dtype=float)
            return 7.5 * mass * r**6 / (r**2 + charge**2) ** 3.5

        def log_lf_derivatives(radius: Array) -> tuple[Array, Array]:
            r = np.asarray(radius, dtype=float)
            denominator = r**2 + charge**2
            first = 6.0 / r - 7.0 * r / denominator
            second = -6.0 / r**2 - 7.0 * (charge**2 - r**2) / denominator**2
            return first, second
    elif model_name == "hayward":
        def mass_function(radius: Array) -> Array:
            r = np.asarray(radius, dtype=float)
            return mass * r**3 / (r**3 + 2.0 * mass * charge**2)

        def lagrangian(radius: Array) -> Array:
            r = np.asarray(radius, dtype=float)
            denominator = r**3 + 2.0 * mass * charge**2
            return 6.0 * mass**2 * charge**2 / denominator**2

        def lagrangian_f(radius: Array) -> Array:
            r = np.asarray(radius, dtype=float)
            denominator = r**3 + 2.0 * mass * charge**2
            return 18.0 * mass**2 * r**7 / denominator**3

        def log_lf_derivatives(radius: Array) -> tuple[Array, Array]:
            r = np.asarray(radius, dtype=float)
            denominator = r**3 + 2.0 * mass * charge**2
            first = 7.0 / r - 9.0 * r**2 / denominator
            second = -7.0 / r**2 - 18.0 * r / denominator + 27.0 * r**4 / denominator**2
            return first, second
    elif model_name == "fan_wang":
        def mass_function(radius: Array) -> Array:
            r = np.asarray(radius, dtype=float)
            return mass * r**3 / (r + charge) ** 3

        def lagrangian(radius: Array) -> Array:
            r = np.asarray(radius, dtype=float)
            return 3.0 * mass * charge / (r + charge) ** 4

        def lagrangian_f(radius: Array) -> Array:
            r = np.asarray(radius, dtype=float)
            return 6.0 * mass * r**5 / (charge * (r + charge) ** 5)

        def log_lf_derivatives(radius: Array) -> tuple[Array, Array]:
            r = np.asarray(radius, dtype=float)
            first = 5.0 / r - 5.0 / (r + charge)
            second = -5.0 / r**2 + 5.0 / (r + charge) ** 2
            return first, second
    else:
        raise ValueError(f"Unknown model: {model_name}")

    def metric(radius: Array) -> Array:
        r = np.asarray(radius, dtype=float)
        return 1.0 - 2.0 * mass_function(r) / r

    def metric_prime(radius: Array) -> Array:
        r = np.asarray(radius, dtype=float)
        return 2.0 * mass_function(r) / r**2 - 2.0 * r * lagrangian(r)

    def kappa(radius: Array) -> Array:
        r = np.asarray(radius, dtype=float)
        first, _ = log_lf_derivatives(r)
        return 1.0 - 0.5 * r * first

    def dplus(radius: Array) -> Array:
        r = np.asarray(radius, dtype=float)
        first, second = log_lf_derivatives(r)
        return 0.5 * metric_prime(r) * first + metric(r) * (0.5 * second + 0.25 * first**2)

    def dminus(radius: Array) -> Array:
        r = np.asarray(radius, dtype=float)
        first, second = log_lf_derivatives(r)
        return -0.5 * metric_prime(r) * first + metric(r) * (-0.5 * second + 0.25 * first**2)

    return Model(
        name=model_name,
        mass=mass,
        charge=charge,
        extremal_charge=qext,
        f=metric,
        fp=metric_prime,
        mass_function=mass_function,
        lagrangian=lagrangian,
        lagrangian_f=lagrangian_f,
        kappa=kappa,
        dminus=dminus,
        dplus=dplus,
    )


def outer_horizon(model: Model) -> float:
    grid = np.geomspace(1.0e-8 * model.mass, 10.0 * model.mass, 20000)
    values = model.f(grid)
    changes = np.where(values[:-1] * values[1:] < 0.0)[0]
    if changes.size == 0:
        raise RuntimeError(f"Could not bracket the outer horizon for {model.name}.")
    index = int(changes[-1])
    return float(brentq(lambda radius: float(model.f(radius)), grid[index], grid[index + 1]))


def potential_matrix(model: Model, parity: str, radius: Array, multipole: int = 2) -> Array:
    

    r = np.asarray(radius, dtype=float)
    scalar_input = r.ndim == 0
    r = np.atleast_1d(r)
    ell = float(multipole)
    angular = ell * (ell + 1.0)
    lam = (ell - 1.0) * (ell + 2.0)
    f = model.f(r)
    m = model.mass_function(r)
    lag = model.lagrangian(r)
    lf = model.lagrangian_f(r)
    q = model.charge
    coupling = -np.sqrt(np.maximum(4.0 * lam * lf, 0.0)) * q / r**3

    matrix = np.zeros((r.size, 2, 2), dtype=float)
    if parity == "axial":
        matrix[:, 0, 0] = angular / r**2 - 6.0 * m / r**3 + 2.0 * lag
        matrix[:, 0, 1] = coupling
        matrix[:, 1, 0] = coupling
        matrix[:, 1, 1] = angular / r**2 + model.dminus(r) + 4.0 * q**2 * lf / r**4
    elif parity == "polar":
        kap = model.kappa(r)
        a = 6.0 * m / r - 2.0 * r**2 * lag
        b = lam + 4.0 * lf * q**2 / r**2
        denominator = a + lam
        v11 = (
            angular * lam
            - 2.0 * f * lam
            + a * (a - 4.0 * m / r)
        ) / (r**2 * denominator)
        v11 += 2.0 * f * lam * b / (r**2 * denominator**2)
        v22 = kap * angular / r**2 + model.dplus(r)
        v22 += (
            4.0
            * lf
            * q**2
            * (lam + 1.0 - f + 2.0 * r**2 * lag + 4.0 * f * kap)
            / (r**4 * denominator)
        )
        v22 += 8.0 * f * lf * q**2 * b / (r**4 * denominator**2)
        w = (lam + 1.0 - f + 2.0 * r**2 * lag + 2.0 * f * kap) / denominator
        w += 2.0 * f * b / denominator**2
        matrix[:, 0, 0] = v11
        matrix[:, 0, 1] = coupling * w
        matrix[:, 1, 0] = coupling * w
        matrix[:, 1, 1] = v22
    else:
        raise ValueError("parity must be 'polar' or 'axial'.")

    return matrix[0] if scalar_input else matrix


def schwarzschild_potential(parity: str, radius: Array, mass: float, multipole: int = 2) -> Array:
    r = np.asarray(radius, dtype=float)
    ell = float(multipole)
    angular = ell * (ell + 1.0)
    f = 1.0 - 2.0 * mass / r
    if parity == "axial":
        return angular / r**2 - 6.0 * mass / r**3
    if parity == "polar":
        n = (ell - 1.0) * (ell + 2.0) / 2.0
        numerator = 2.0 * (
            n**2 * (n + 1.0) * r**3
            + 3.0 * n**2 * mass * r**2
            + 9.0 * n * mass**2 * r
            + 9.0 * mass**3
        )
        return numerator / (r**3 * (n * r + 3.0 * mass) ** 2)
    raise ValueError("parity must be 'polar' or 'axial'.")


def _riccati_pair(order: float, z: float) -> tuple[float, float, float, float]:
    nu = order + 0.5
    prefactor = np.sqrt(np.pi * z / 2.0)
    jhat = prefactor * jv(nu, z)
    yhat = prefactor * yv(nu, z)
    derivative_prefactor = prefactor / (2.0 * z)
    jhat_z = derivative_prefactor * jv(nu, z) + prefactor * jvp(nu, z)
    yhat_z = derivative_prefactor * yv(nu, z) + prefactor * yvp(nu, z)
    return float(jhat), float(jhat_z), float(yhat), float(yhat_z)


def _asymptotic_tortoise(radius: float, mass: float) -> float:
    

    return radius + 2.0 * mass * np.log(radius / (2.0 * mass))


def _integrate_matrix(
    model: Model,
    parity: str,
    omega: float,
    start: float,
    stop: float,
    value: Array,
    derivative: Array,
    controls: SolverControls,
) -> tuple[Array, Array]:
    initial = np.concatenate([value.ravel(), derivative.ravel()]).astype(complex)

    def rhs(radius: float, state: Array) -> Array:
        field = state[:4].reshape(2, 2)
        field_prime = state[4:].reshape(2, 2)
        f = float(model.f(radius))
        fp = float(model.fp(radius))
        potential = potential_matrix(model, parity, radius, controls.multipole)
        field_second = -(fp / f) * field_prime
        field_second -= ((omega**2 * np.eye(2) - f * potential) @ field) / f**2
        return np.concatenate([field_prime.ravel(), field_second.ravel()])

    maximum_step = min(controls.max_step_radius, controls.max_step_phase / omega)
    solution = solve_ivp(
        rhs,
        (start, stop),
        initial,
        method="DOP853",
        rtol=controls.rtol,
        atol=controls.atol,
        max_step=maximum_step,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    final = solution.y[:, -1]
    return final[:4].reshape(2, 2), final[4:].reshape(2, 2)


def response_matrix(
    model: Model,
    parity: str,
    omega: float,
    controls: SolverControls = SolverControls(),
) -> Array:
    





    if omega <= 0.0:
        raise ValueError("omega must be positive.")
    horizon = outer_horizon(model)
    start = horizon + controls.horizon_offset
    match = max(controls.match_radius, horizon + 5.0)
    outer = max(controls.minimum_outer_radius, controls.asymptotic_cycles / omega)

    fph = float(model.fp(horizon))
    vh = potential_matrix(model, parity, horizon, controls.multipole)
    first_regular = vh / (fph - 2.0j * omega)
    identity = np.eye(2, dtype=complex)
    horizon_value = identity + controls.horizon_offset * first_regular
    f_start = float(model.f(start))
    horizon_derivative = -1.0j * omega * horizon_value / f_start + first_regular
    h_value, h_derivative = _integrate_matrix(
        model,
        parity,
        omega,
        start,
        match,
        horizon_value,
        horizon_derivative,
        controls,
    )

    asymptotic_v = potential_matrix(model, parity, outer, controls.multipole)
    c_em = float(outer**2 * asymptotic_v[1, 1])
    electromagnetic_order = 0.5 * (-1.0 + np.sqrt(max(1.0 + 4.0 * c_em, 1.0e-14)))
    orders = [float(controls.multipole), electromagnetic_order]
    source_value = np.zeros((2, 2), dtype=complex)
    source_derivative = np.zeros((2, 2), dtype=complex)
    response_value = np.zeros((2, 2), dtype=complex)
    response_derivative = np.zeros((2, 2), dtype=complex)
    z = omega * _asymptotic_tortoise(outer, model.mass)
    dz_dr = omega / float(model.f(outer))
    for channel, order in enumerate(orders):
        jhat, jhat_z, yhat, yhat_z = _riccati_pair(order, z)
        source_value[channel, channel] = jhat
        source_derivative[channel, channel] = dz_dr * jhat_z
        response_value[channel, channel] = yhat
        response_derivative[channel, channel] = dz_dr * yhat_z

    j_value, j_derivative = _integrate_matrix(
        model,
        parity,
        omega,
        outer,
        match,
        source_value,
        source_derivative,
        controls,
    )
    y_value, y_derivative = _integrate_matrix(
        model,
        parity,
        omega,
        outer,
        match,
        response_value,
        response_derivative,
        controls,
    )

    horizon_data = np.vstack([h_value, h_derivative])
    asymptotic_data = np.block([[j_value, y_value], [j_derivative, y_derivative]])
    coefficients = np.linalg.solve(asymptotic_data, horizon_data)
    source_coefficients = coefficients[:2, :]
    response_coefficients = coefficients[2:, :]
    return response_coefficients @ np.linalg.inv(source_coefficients)


def schwarzschild_response(
    parity: str,
    omega: float,
    mass: float,
    controls: SolverControls,
) -> complex:
    

    ell = controls.multipole
    horizon = 2.0 * mass
    start = horizon + controls.horizon_offset
    match = max(controls.match_radius, horizon + 5.0)
    outer = max(controls.minimum_outer_radius, controls.asymptotic_cycles / omega)
    fph = 1.0 / (2.0 * mass)
    vh = float(schwarzschild_potential(parity, horizon, mass, ell))
    regular_first = vh / (fph - 2.0j * omega)
    f_start = 1.0 - 2.0 * mass / start
    h0 = 1.0 + controls.horizon_offset * regular_first
    hp0 = -1.0j * omega * h0 / f_start + regular_first

    def rhs(radius: float, state: Array) -> Array:
        f = 1.0 - 2.0 * mass / radius
        fp = 2.0 * mass / radius**2
        potential = float(schwarzschild_potential(parity, radius, mass, ell))
        return np.array(
            [
                state[1],
                -(fp / f) * state[1] - (omega**2 - f * potential) * state[0] / f**2,
            ],
            dtype=complex,
        )

    maximum_step = min(controls.max_step_radius, controls.max_step_phase / omega)

    def integrate(begin: float, end: float, initial: Array) -> Array:
        result = solve_ivp(
            rhs,
            (begin, end),
            np.asarray(initial, dtype=complex),
            method="DOP853",
            rtol=controls.rtol,
            atol=controls.atol,
            max_step=maximum_step,
        )
        if not result.success:
            raise RuntimeError(result.message)
        return result.y[:, -1]

    horizon_data = integrate(start, match, np.array([h0, hp0]))
    z = omega * _asymptotic_tortoise(outer, mass)
    dz_dr = omega / (1.0 - 2.0 * mass / outer)
    jhat, jhat_z, yhat, yhat_z = _riccati_pair(float(ell), z)
    source_data = integrate(outer, match, np.array([jhat, dz_dr * jhat_z]))
    response_data = integrate(outer, match, np.array([yhat, dz_dr * yhat_z]))
    amplitudes = np.linalg.solve(
        np.column_stack([source_data, response_data]), horizon_data
    )
    return complex(amplitudes[1] / amplitudes[0])


def canonical_gravitational_response(
    model: Model,
    parity: str,
    omega: float,
    controls: SolverControls = SolverControls(),
) -> complex:
    """Return the direct coupled-system gravitational response.

    No Schwarzschild response is subtracted.  Schwarzschild calculations are
    retained elsewhere only as independent validation/reference checks.
    """
    return complex(response_matrix(model, parity, omega, controls)[0, 0])


def independent_static_continuation(model: Model, parity: str) -> float:
    """Return the full-range continuation of the directly computed static TLN.

    The coefficients are fitted only to the zero-frequency coupled-system
    calculation archived in ``results/all_models_static_direct.csv``.  No
    published static fit or external normalization enters this function.
    """
    x = model.charge / model.mass
    if parity == "polar":
        if model.name == "bardeen":
            return (0.08022747734516422 * x**2
                    + 10.76112455405768 * x**4
                    - 0.14943708011041934 * x**6
                    - 0.18090821966611434 * x**8)
        if model.name == "hayward":
            return (12.806535897920313 * x**2
                    - 0.508024965615953 * x**4
                    - 0.05285559611179657 * x**6
                    - 0.07816243496995176 * x**8)
        if model.name == "fan_wang":
            return (215.47136484956638 * x**3
                    + 95.2369896663257 * x**4
                    + 1.4652862135271463 * x**5
                    - 48.49048508096146 * x**6)
    elif parity == "axial":
        if model.name == "bardeen":
            return (-5.7638546264096995 * x**4
                    - 1.0095588588541715 * x**6
                    - 0.016395640985604535 * x**8
                    - 0.4595953178610712 * x**10)
        if model.name == "hayward":
            return (1.7997609254140432 * x**4
                    + 0.10161615525955954 * x**6
                    + 0.009910501154038777 * x**8
                    + 0.04608944056790019 * x**10)
        if model.name == "fan_wang":
            return (95.33769325898247 * x**3
                    + 1.8052121138624708 * x**4
                    - 12.97487698310544 * x**5
                    + 12.740535818693607 * x**6)
    raise ValueError("Unsupported model/parity combination.")


def run_scan(args: argparse.Namespace) -> list[dict[str, float | str]]:
    controls = SolverControls(
        horizon_offset=args.horizon_offset,
        match_radius=args.match_radius,
        asymptotic_cycles=args.asymptotic_cycles,
        minimum_outer_radius=args.minimum_outer_radius,
        rtol=args.rtol,
        atol=args.atol,
    )
    rows: list[dict[str, float | str]] = []
    for model_name in args.models:
        qext = extremal_charge(model_name, args.mass)
        for ratio in args.charge_ratios:
            model = build_model(model_name, ratio * qext, args.mass)
            for parity in args.parities:
                for omega in args.omegas:
                    value = canonical_gravitational_response(model, parity, omega, controls)
                    row = {
                        "model": model_name,
                        "parity": parity,
                        "charge_ratio": ratio,
                        "charge": model.charge,
                        "omega_M": omega * args.mass,
                        "response_real": float(value.real),
                        "response_imag": float(value.imag),
                        "response_over_omega3_real": float(value.real / omega**3),
                        "independent_static_continuation": independent_static_continuation(model, parity),
                    }
                    rows.append(row)
                    print(
                        f"{model_name:9s} {parity:5s} q/qext={ratio:.3f} "
                        f"omegaM={omega * args.mass:.4f} "
                        f"Rgg={value.real:+.8e}{value.imag:+.8e}i"
                    )
    return rows


def write_csv(rows: list[dict[str, float | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_rows(rows: list[dict[str, float | str]], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    models = sorted({str(row["model"]) for row in rows})
    parities = sorted({str(row["parity"]) for row in rows})
    for parity in parities:
        figure, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 4.3), squeeze=False)
        for column, model_name in enumerate(models):
            axis = axes[0, column]
            selected = [
                row for row in rows
                if row["model"] == model_name and row["parity"] == parity
            ]
            omegas = sorted({float(row["omega_M"]) for row in selected})
            for omega in omegas:
                points = sorted(
                    [row for row in selected if float(row["omega_M"]) == omega],
                    key=lambda row: float(row["charge_ratio"]),
                )
                axis.plot(
                    [float(row["charge_ratio"]) for row in points],
                    [float(row["response_over_omega3_real"]) for row in points],
                    "o-",
                    label=rf"$\omega M={omega:g}$",
                )
            axis.axhline(0.0, color="0.5", linewidth=0.8)
            axis.set_title(model_name.replace("_", " ").title())
            axis.set_xlabel(r"$q/q_{\rm ext}$")
            axis.grid(alpha=0.25)
        axes[0, 0].set_ylabel(rf"$\mathrm{{Re}}\,[\Delta\mathcal{{R}}_{{gg}}^{{{parity}}}/(M\omega)^3]$")
        axes[0, -1].legend(frameon=False, fontsize=9)
        figure.tight_layout()
        figure.savefig(output / f"canonical_response_{parity}.png", dpi=220)
        figure.savefig(output / f"canonical_response_{parity}.pdf")
        plt.close(figure)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["bardeen", "hayward", "fan_wang"])
    parser.add_argument("--parities", nargs="+", default=["polar", "axial"])
    parser.add_argument("--mass", type=float, default=1.0)
    parser.add_argument("--charge-ratios", type=float, nargs="+", default=[0.2, 0.4, 0.6, 0.8, 0.95])
    parser.add_argument("--omegas", type=float, nargs="+", default=[0.01, 0.02, 0.03])
    parser.add_argument("--horizon-offset", type=float, default=2.0e-5)
    parser.add_argument("--match-radius", type=float, default=18.0)
    parser.add_argument("--asymptotic-cycles", type=float, default=70.0)
    parser.add_argument("--minimum-outer-radius", type=float, default=800.0)
    parser.add_argument("--rtol", type=float, default=2.0e-10)
    parser.add_argument("--atol", type=float, default=2.0e-12)
    parser.add_argument("--output", type=Path, default=Path("results/dynamic_tln.csv"))
    parser.add_argument("--figure-dir", type=Path, default=Path("figures"))
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    rows = run_scan(args)
    write_csv(rows, args.output)
    plot_rows(rows, args.figure_dir)


if __name__ == "__main__":
    main()
