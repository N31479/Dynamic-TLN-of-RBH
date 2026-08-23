#!/usr/bin/env python3









from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import jv, jvp, yv, yvp

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import einstein_nled_master as master


def _bardeen_background_derivatives(
    radius: np.ndarray, mass: float, charge: float
) -> dict[str, np.ndarray]:
    r = np.asarray(radius, dtype=float)
    d = r * r + charge * charge
    m = mass * r**3 * d ** (-1.5)
    mp = 3.0 * mass * charge**2 * r**2 * d ** (-2.5)
    mpp = 3.0 * mass * charge**2 * (
        2.0 * r * d ** (-2.5) - 5.0 * r**3 * d ** (-3.5)
    )
    lag = 3.0 * mass * charge**2 * d ** (-2.5)
    lagp = -15.0 * mass * charge**2 * r * d ** (-3.5)
    lagpp = -15.0 * mass * charge**2 * (
        d ** (-3.5) - 7.0 * r**2 * d ** (-4.5)
    )
    lagf = 7.5 * mass * r**6 * d ** (-3.5)
    lagfp = lagf * (6.0 / r - 7.0 * r / d)
    f = 1.0 - 2.0 * m / r
    fp = -2.0 * mp / r + 2.0 * m / r**2
    fpp = -2.0 * mpp / r + 4.0 * mp / r**2 - 4.0 * m / r**3
    a = 6.0 * m / r - 2.0 * r**2 * lag
    ap = 6.0 * mp / r - 6.0 * m / r**2 - 4.0 * r * lag - 2.0 * r**2 * lagp
    app = (
        6.0 * mpp / r
        - 12.0 * mp / r**2
        + 12.0 * m / r**3
        - 4.0 * lag
        - 8.0 * r * lagp
        - 2.0 * r**2 * lagpp
    )
    return {
        "f": f,
        "fp": fp,
        "fpp": fpp,
        "m": m,
        "lag": lag,
        "lagf": lagf,
        "lagfp": lagfp,
        "a": a,
        "ap": ap,
        "app": app,
    }


def _riccati_pair(order: float, z: np.ndarray) -> tuple[np.ndarray, ...]:
    nu = order + 0.5
    prefactor = np.sqrt(np.pi * z / 2.0)
    jhat = prefactor * jv(nu, z)
    yhat = prefactor * yv(nu, z)
    derivative_prefactor = prefactor / (2.0 * z)
    jhat_z = derivative_prefactor * jv(nu, z) + prefactor * jvp(nu, z)
    yhat_z = derivative_prefactor * yv(nu, z) + prefactor * yvp(nu, z)
    return jhat, jhat_z, yhat, yhat_z


def _integrate_asymptotic_bases(
    model: master.Model,
    omega: float,
    controls: master.SolverControls,
    radii: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    outer = max(controls.minimum_outer_radius, controls.asymptotic_cycles / omega)
    potential_outer = master.potential_matrix(model, "polar", outer, controls.multipole)
    electromagnetic_coefficient = float(outer**2 * potential_outer[1, 1])
    electromagnetic_order = 0.5 * (
        -1.0 + np.sqrt(max(1.0 + 4.0 * electromagnetic_coefficient, 1.0e-14))
    )
    orders = (float(controls.multipole), electromagnetic_order)
    z = omega * master._asymptotic_tortoise(outer, model.mass)
    dz_dr = omega / float(model.f(outer))

    source_value = np.zeros((2, 2), dtype=complex)
    source_derivative = np.zeros((2, 2), dtype=complex)
    response_value = np.zeros((2, 2), dtype=complex)
    response_derivative = np.zeros((2, 2), dtype=complex)
    for channel, order in enumerate(orders):
        jhat, jhat_z, yhat, yhat_z = master._riccati_pair(order, z)
        source_value[channel, channel] = jhat
        source_derivative[channel, channel] = dz_dr * jhat_z
        response_value[channel, channel] = yhat
        response_derivative[channel, channel] = dz_dr * yhat_z

    initial_value = np.column_stack((source_value, response_value))
    initial_derivative = np.column_stack((source_derivative, response_derivative))
    initial = np.concatenate((initial_value.ravel(), initial_derivative.ravel()))

    def rhs(radius: float, state: np.ndarray) -> np.ndarray:
        field = state[:8].reshape(2, 4)
        field_prime = state[8:].reshape(2, 4)
        f = float(model.f(radius))
        fp = float(model.fp(radius))
        potential = master.potential_matrix(model, "polar", radius, controls.multipole)
        field_second = -(fp / f) * field_prime
        field_second -= (
            (omega**2 * np.eye(2) - f * potential) @ field
        ) / f**2
        return np.concatenate((field_prime.ravel(), field_second.ravel()))

    descending_radii = np.asarray(radii[::-1], dtype=float)
    result = solve_ivp(
        rhs,
        (outer, float(radii[0])),
        initial,
        t_eval=descending_radii,
        method="DOP853",
        rtol=controls.rtol,
        atol=controls.atol,
        max_step=min(0.8, controls.max_step_phase / omega),
    )
    if not result.success:
        raise RuntimeError(result.message)

    data = result.y[:, ::-1]
    values = data[:8].T.reshape(radii.size, 2, 4)
    derivatives = data[8:].T.reshape(radii.size, 2, 4)
    response = master.response_matrix(model, "polar", omega, controls)
    return values, derivatives, response, outer, electromagnetic_order


def reconstruct_h0(
    model: master.Model,
    omega: float,
    radii: np.ndarray,
    values: np.ndarray,
    derivatives: np.ndarray,
) -> np.ndarray:
    










    ell = 2.0
    angular = ell * (ell + 1.0)
    lam = (ell - 1.0) * (ell + 2.0)
    sqrt_lam = np.sqrt(lam)
    background = _bardeen_background_derivatives(radii, model.mass, model.charge)
    f = background["f"]
    fp = background["fp"]
    lagf = background["lagf"]
    lagfp = background["lagfp"]
    a = background["a"]
    ap = background["ap"]
    app = background["app"]
    A = a + lam
    denominator = radii * A
    denominator_prime = A + radii * ap

    potentials = master.potential_matrix(model, "polar", radii, 2)
    reconstructed = np.zeros((radii.size, values.shape[2]), dtype=complex)

    for column in range(values.shape[2]):
        psi = values[:, 0, column]
        phi_master = values[:, 1, column]
        psi_prime = derivatives[:, 0, column]
        phi_master_prime = derivatives[:, 1, column]

        second = np.empty((radii.size, 2), dtype=complex)
        for index in range(radii.size):
            vector = values[index, :, column]
            vector_prime = derivatives[index, :, column]
            second[index] = -(fp[index] / f[index]) * vector_prime
            second[index] -= (
                (omega**2 * np.eye(2) - f[index] * potentials[index]) @ vector
            ) / f[index] ** 2
        psi_second = second[:, 0]

        zeta_factor = A / sqrt_lam
        zeta_factor_prime = ap / sqrt_lam
        zeta_factor_second = app / sqrt_lam
        zeta = zeta_factor * psi
        zeta_prime = zeta_factor_prime * psi + zeta_factor * psi_prime
        zeta_second = (
            zeta_factor_second * psi
            + 2.0 * zeta_factor_prime * psi_prime
            + zeta_factor * psi_second
        )

        sqrt_lagf = np.sqrt(lagf)
        matter_potential = (
            phi_master / (2.0 * sqrt_lagf)
            + model.charge * psi / (radii * sqrt_lam)
        )
        matter_potential_prime = (
            phi_master_prime / (2.0 * sqrt_lagf)
            - phi_master * lagfp / (4.0 * lagf * sqrt_lagf)
            + model.charge
            * (psi_prime / radii - psi / radii**2)
            / sqrt_lam
        )

        numerator = (
            8.0 * model.charge * f * lagf * matter_potential / radii
            - 2.0 * radii * f * zeta_prime
            - angular * zeta
        )
        numerator_prime = (
            8.0
            * model.charge
            * (
                (fp * lagf + f * lagfp) * matter_potential / radii
                + f * lagf * matter_potential_prime / radii
                - f * lagf * matter_potential / radii**2
            )
            - 2.0
            * (
                (f + radii * fp) * zeta_prime
                + radii * f * zeta_second
            )
            - angular * zeta_prime
        )
        k_prime = (
            numerator_prime * denominator
            - numerator * denominator_prime
        ) / denominator**2
        reconstructed[:, column] = (
            -zeta_prime
            - radii * k_prime
            + 4.0 * lagf * model.charge * matter_potential / radii**2
        )

    return reconstructed


def flat_space_metric_bases(
    radii: np.ndarray, omega: float
) -> tuple[np.ndarray, np.ndarray]:
    z = omega * radii
    jhat, jhat_z, yhat, yhat_z = _riccati_pair(2.0, z)
    source = ((3.0 - z**2) * jhat + z * jhat_z) / radii
    response = ((3.0 - z**2) * yhat + z * yhat_z) / radii
    return source, response


def source_normalization(
    radii: np.ndarray, field: np.ndarray, omega: float, mass: float, charge: float
) -> complex:
    source, _ = flat_space_metric_bases(radii, omega)
    template = source * (
        1.0
        - 2.0 * mass / radii
        + 1.5 * mass * charge**2 / radii**3
        - 0.5 * mass**2 * charge**2 / radii**4
    )
    design = np.column_stack((template, source / radii**2, source / radii**3))
    scales = np.linalg.norm(design, axis=0)
    coefficients = np.linalg.lstsq(
        design / scales, field, rcond=1.0e-13
    )[0] / scales
    return complex(coefficients[0])


def schwarzschild_normalization_check(omega: float) -> dict[str, float]:
    radius = np.array([1.0e-3 / omega, 2.0e-3 / omega, 4.0e-3 / omega])
    source, response = flat_space_metric_bases(radius, omega)
    source_coefficient = float(np.mean(source / radius**2))
    response_coefficient = float(np.mean(response * radius**3))
    expected_source = 2.0 * omega**3 / 5.0
    expected_response = -3.0 / omega**2
    return {
        "omega": omega,
        "source_coefficient_numeric": source_coefficient,
        "source_coefficient_expected": expected_source,
        "source_relative_error": abs(source_coefficient / expected_source - 1.0),
        "response_coefficient_numeric": response_coefficient,
        "response_coefficient_expected": expected_response,
        "response_relative_error": abs(response_coefficient / expected_response - 1.0),
        "metric_ratio_prefactor": -15.0 / (2.0 * omega**5),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    controls = master.SolverControls(
        horizon_offset=args.horizon_offset,
        match_radius=args.match_radius,
        asymptotic_cycles=args.asymptotic_cycles,
        minimum_outer_radius=args.minimum_outer_radius,
        rtol=args.rtol,
        atol=args.atol,
        max_step_phase=args.max_step_phase,
        max_step_radius=5.0,
    )
    charge = args.charge_ratio * master.extremal_charge("bardeen", args.mass)
    model = master.build_model("bardeen", charge, args.mass)
    radii = np.linspace(args.radius_min, args.radius_max, args.points)
    values, derivatives, response_matrix, outer, electromagnetic_order = (
        _integrate_asymptotic_bases(model, args.omega, controls, radii)
    )
    h0_bases = reconstruct_h0(model, args.omega, radii, values, derivatives)

    canonical_coefficients = np.array(
        [1.0, 0.0, response_matrix[0, 0], response_matrix[1, 0]],
        dtype=complex,
    )
    h0 = h0_bases @ canonical_coefficients
    fit_mask = (radii >= args.fit_min) & (radii <= args.fit_max)
    normalization = source_normalization(
        radii[fit_mask], h0[fit_mask], args.omega, args.mass, charge
    )
    normalized_h0 = h0 / normalization

    source_basis, response_basis = flat_space_metric_bases(radii, args.omega)
    check = schwarzschild_normalization_check(args.omega)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_dir = args.output_dir / "results"
    figure_dir = args.output_dir / "figures"
    result_dir.mkdir(exist_ok=True)
    figure_dir.mkdir(exist_ok=True)

    csv_path = result_dir / "bardeen_polar_reconstructed_h0.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "radius_over_M",
                "omega_M",
                "H0_real",
                "H0_imag",
                "H0_normalized_real",
                "H0_normalized_imag",
                "flat_source_basis",
                "flat_response_basis",
            ]
        )
        for index, radius in enumerate(radii):
            writer.writerow(
                [
                    radius / args.mass,
                    args.omega * args.mass,
                    h0[index].real,
                    h0[index].imag,
                    normalized_h0[index].real,
                    normalized_h0[index].imag,
                    source_basis[index],
                    response_basis[index],
                ]
            )

    figure, axis = plt.subplots(figsize=(7.0, 4.7))
    axis.plot(radii / args.mass, normalized_h0.real, label=r"$\mathrm{Re}\,H_0$")
    axis.plot(radii / args.mass, normalized_h0.imag, label=r"$\mathrm{Im}\,H_0$")
    axis.set_xlabel(r"$r/M$")
    axis.set_ylabel(r"source-normalized $H_0(\omega,r)$")
    axis.set_title(
        rf"Bardeen polar metric reconstruction: $q/q_{{\rm ext}}={args.charge_ratio:g}$, "
        rf"$M\omega={args.omega * args.mass:g}$"
    )
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(figure_dir / "bardeen_polar_reconstructed_h0.png", dpi=220)
    figure.savefig(figure_dir / "bardeen_polar_reconstructed_h0.pdf")
    plt.close(figure)

    result = {
        "model": "bardeen",
        "parity": "polar",
        "mass": args.mass,
        "charge": charge,
        "charge_ratio": args.charge_ratio,
        "omega": args.omega,
        "outer_radius": outer,
        "electromagnetic_asymptotic_order": electromagnetic_order,
        "source_normalization": [normalization.real, normalization.imag],
        "canonical_response_column": [
            [response_matrix[0, 0].real, response_matrix[0, 0].imag],
            [response_matrix[1, 0].real, response_matrix[1, 0].imag],
        ],
        "schwarzschild_metric_normalization_check": check,
        "controls": asdict(controls),
        "extraction_status": (
            "H0 is reconstructed from the coupled master fields.  The r^-3 Love "
            "coefficient is not reported by this first-pass calculation because a "
            "controlled asymptotic-series source/response matcher is required to "
            "separate it from lower-order long-range terms."
        ),
    }
    with (result_dir / "bardeen_polar_metric_reconstruction_check.json").open("w") as stream:
        json.dump(result, stream, indent=2)
    return result


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mass", type=float, default=1.0)
    parser.add_argument("--charge-ratio", type=float, default=0.50)
    parser.add_argument("--omega", type=float, default=0.02)
    parser.add_argument("--radius-min", type=float, default=6.0)
    parser.add_argument("--radius-max", type=float, default=55.0)
    parser.add_argument("--fit-min", type=float, default=10.0)
    parser.add_argument("--fit-max", type=float, default=35.0)
    parser.add_argument("--points", type=int, default=900)
    parser.add_argument("--horizon-offset", type=float, default=2.0e-5)
    parser.add_argument("--match-radius", type=float, default=15.0)
    parser.add_argument("--asymptotic-cycles", type=float, default=18.0)
    parser.add_argument("--minimum-outer-radius", type=float, default=260.0)
    parser.add_argument("--rtol", type=float, default=3.0e-9)
    parser.add_argument("--atol", type=float, default=3.0e-11)
    parser.add_argument("--max-step-phase", type=float, default=0.30)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE.parent,
    )
    return parser.parse_args()


def main() -> None:
    result = run(parse_arguments())
    check = result["schwarzschild_metric_normalization_check"]
    print(
        "Schwarzschild metric-basis check: "
        f"source error={check['source_relative_error']:.3e}, "
        f"response error={check['response_relative_error']:.3e}"
    )
    print("Reconstructed Bardeen polar H0 profile written to the output directory.")


if __name__ == "__main__":
    main()
