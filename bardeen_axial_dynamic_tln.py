#!/usr/bin/env python3
"""
Axial dynamical tidal response of the Bardeen regular black hole.

This script computes the frequency-dependent axial response coefficient beta(omega)
for quadrupolar perturbations of the Bardeen regular black-hole geometry.  It solves
the coupled axial gravitational--matter boundary-value problem, extracts the far-zone
source/response coefficients, and plots beta as a function of ell_B/ell_ext.

The implementation is intended as reproducibility code for the paper
"Dynamical tidal response of regular black holes: Perturbative analysis and shell
EFT interpretation".

Notes
-----
* Units: M = 1 by default.
* The Bardeen extremal scale is ell_ext = 4 M / (3 sqrt(3)).
* The extraction follows the convention used in the paper's numerical scripts; the
  response is divided by `normalization_scale` to match the plotting convention.
* The script deliberately plots raw numerical curves/points and does not apply spline
  interpolation or smoothing.

Example
-------
    python bardeen_axial_dynamic_tln.py
    python bardeen_axial_dynamic_tln.py --omegas 1e-4 1e-2 2e-2 3e-2 --n-chi 10
    python bardeen_axial_dynamic_tln.py --no-show --output data/axial_bardeen.csv --figure figures/axial_bardeen.png
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_bvp
from scipy.optimize import brentq
from scipy.special import spherical_yn


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class AxialConfig:
    """Numerical configuration for the axial Bardeen BVP."""

    mass: float = 1.0
    ell_harmonic: int = 2
    z_switch: float = 10.0
    r_min_offset: float = 5.0e-5
    bvp_tol: float = 1.0e-3
    max_nodes: int = 300_000
    n_mesh: int = 2_500
    normalization_scale: float = 1.0e7

    @property
    def ell_ext(self) -> float:
        """Extremal Bardeen length scale."""
        return 4.0 * self.mass / (3.0 * np.sqrt(3.0))

    @property
    def schwarzschild_radius(self) -> float:
        return 2.0 * self.mass


# -----------------------------------------------------------------------------
# Bardeen background and NLED derivatives
# -----------------------------------------------------------------------------


def horizon_equation(r: np.ndarray | float, mass: float, ell_b: float) -> np.ndarray | float:
    """Equation whose outer positive root gives the Bardeen horizon."""
    return (np.asarray(r) ** 2 + ell_b**2) ** 1.5 - 2.0 * mass * np.asarray(r) ** 2


def get_outer_horizon(mass: float, ell_b: float) -> float:
    """Return the outer horizon radius for a subextremal Bardeen black hole.

    A logarithmic scan is used to locate all sign changes of the horizon equation;
    the largest positive root is returned.
    """

    if mass <= 0:
        raise ValueError("mass must be positive.")
    if ell_b < 0:
        raise ValueError("ell_b must be non-negative.")

    def equation(r: np.ndarray | float) -> np.ndarray | float:
        return horizon_equation(r, mass, ell_b)

    def scan_and_bracket(r_max: float, n_points: int = 30_000) -> tuple[float, float] | None:
        grid = np.geomspace(1.0e-12 * mass, r_max, n_points)
        values = equation(grid)
        signs = np.sign(values)
        signs[signs == 0.0] = 1.0
        sign_changes = np.where(signs[:-1] * signs[1:] < 0.0)[0]
        if sign_changes.size == 0:
            return None
        idx = int(sign_changes[-1])
        return float(grid[idx]), float(grid[idx + 1])

    for r_max in (2.5 * mass, 10.0 * mass, 50.0 * mass, 200.0 * mass):
        bracket = scan_and_bracket(r_max)
        if bracket is not None:
            return float(brentq(lambda r: float(equation(r)), *bracket))

    raise RuntimeError(
        "Failed to bracket the outer horizon. Check that ell_b is subextremal "
        "or increase the scan range."
    )


def bardeen_derivatives(
    r: np.ndarray | float,
    mass: float,
    ell_b: float,
) -> tuple[np.ndarray | float, np.ndarray | float, np.ndarray | float, np.ndarray | float]:
    """Return f, f', L_F and L_FF for the Bardeen background.

    The returned nonlinear-electrodynamics derivatives follow the conventions used
    in the axial perturbation equations of the numerical calculation.
    """

    r_arr = np.atleast_1d(r).astype(float)
    r2 = r_arr * r_arr
    ell2 = ell_b * ell_b
    denom = r2 + ell2

    mass_profile = mass * r_arr**3 * denom ** (-1.5)
    f_metric = 1.0 - 2.0 * mass_profile / r_arr

    dm_dr = 3.0 * mass * ell2 * r2 * denom ** (-2.5)
    df_dr = (2.0 * mass_profile / r2) - (2.0 * dm_dr / r_arr)

    if ell_b < 1.0e-12:
        l_f = np.zeros_like(r_arr)
        l_ff = np.zeros_like(r_arr)
    else:
        d_l_dr = -15.0 * mass * ell2 * r_arr * denom ** (-3.5)
        d_f_inv_dr = -2.0 * ell2 / r_arr**5
        l_f = d_l_dr / d_f_inv_dr

        d_lf_dr = 7.5 * mass * (
            6.0 * r_arr**5 * denom ** (-3.5)
            - 7.0 * r_arr**7 * denom ** (-4.5)
        )
        l_ff = d_lf_dr / d_f_inv_dr

    if np.isscalar(r):
        return float(f_metric[0]), float(df_dr[0]), float(l_f[0]), float(l_ff[0])
    return f_metric, df_dr, l_f, l_ff


def far_field_slice(r_grid: np.ndarray, start_fraction: float = 0.60, end_trim: int = 5) -> slice:
    """Return the far-field region used for source/response fitting."""
    start = int(start_fraction * len(r_grid))
    stop = -end_trim if end_trim > 0 else None
    return slice(start, stop)


# -----------------------------------------------------------------------------
# Axial dynamical response solver
# -----------------------------------------------------------------------------


def solve_beta_axial_dynamic(
    ell_b: float,
    omega: float,
    config: AxialConfig = AxialConfig(),
) -> float:
    """Compute the axial dynamical response coefficient beta(omega).

    Parameters
    ----------
    ell_b:
        Bardeen regularization length.
    omega:
        Driving frequency in units of 1/M.
    config:
        Numerical solver configuration.

    Returns
    -------
    float
        The normalized axial response coefficient beta(omega). Returns np.nan if
        the boundary-value solver fails to converge.
    """

    if omega <= 0.0:
        raise ValueError("omega must be positive for the dynamical calculation.")

    mass = config.mass
    ell_ext = config.ell_ext
    chi = ell_b / ell_ext
    r_horizon = get_outer_horizon(mass, ell_b)

    r_min = r_horizon + config.r_min_offset
    r_max = 20.0 + 108.0 * chi

    ell = float(config.ell_harmonic)
    ell_factor = ell * (ell + 1.0)

    def ode_system(r: np.ndarray, y: np.ndarray) -> np.ndarray:
        h0, h0_prime, u1, u1_prime = y
        f_metric, df_dr, l_f, l_ff = bardeen_derivatives(r, mass, ell_b)
        f_metric = np.maximum(f_metric, 1.0e-12)

        omega_term = omega**2 / f_metric**2

        coupling_a = np.zeros_like(r)
        coupling_b = np.zeros_like(r)
        mask = np.abs(l_f) > 1.0e-14
        if np.any(mask):
            coupling_a[mask] = -2.0 * ell_b**2 * l_ff[mask] / (
                r[mask] ** 5 * l_f[mask]
            )
            coupling_b[mask] = 2.0 * ell_b**2 * l_ff[mask] / (
                r[mask] ** 6 * l_f[mask]
            )

        u1_pp_static = -(
            coupling_a * u1_prime
            + (coupling_b - ell_factor / (r**2 * f_metric)) * u1
            - (ell_factor * ell_b / (r**3 * f_metric)) * h0
        )
        u1_pp = u1_pp_static - omega_term * u1

        axial_potential = (
            2.0 * r**2 * f_metric
            + (ell - 1.0) * (ell + 2.0) * r**2
            + 4.0 * ell_b**2 * l_f
        ) / r**4
        h0_pp_static = (axial_potential * h0 + (4.0 * ell_b * l_f / r**3) * u1) / f_metric
        h0_pp = h0_pp_static - omega_term * h0

        return np.vstack((h0_prime, h0_pp, u1_prime, u1_pp))

    def static_boundary_conditions(ya: np.ndarray, yb: np.ndarray) -> np.ndarray:
        h0_a, _, u1_a, _ = ya
        _, _, u1_b, u1_prime_b = yb
        _, h0_prime_b, _, _ = yb
        return np.array(
            [
                h0_a - 1.0,
                u1_a + (ell_b / r_horizon) * h0_a,
                h0_prime_b - 3.0 * r_max**2,
                u1_prime_b + 2.0 * u1_b / r_max,
            ]
        )

    def wave_boundary_conditions(ya: np.ndarray, yb: np.ndarray) -> np.ndarray:
        h0_a, _, u1_a, _ = ya
        h0_b, h0_prime_b, u1_b, u1_prime_b = yb

        horizon_h_condition = h0_a - 1.0
        horizon_u_condition = u1_a + (ell_b / r_horizon) * h0_a

        z_inf = omega * r_max
        y_bessel = spherical_yn(config.ell_harmonic, z_inf)
        dy_bessel = spherical_yn(config.ell_harmonic, z_inf, derivative=True)
        source_normalization = 1.0 / (omega * r_max**2)

        far_h_condition = omega * dy_bessel * h0_b - y_bessel * h0_prime_b - source_normalization
        far_u_condition = omega * dy_bessel * u1_b - y_bessel * u1_prime_b

        return np.array(
            [
                horizon_h_condition,
                horizon_u_condition,
                far_h_condition,
                far_u_condition,
            ]
        )

    use_quasi_static_boundary = (omega * r_max) < config.z_switch

    r_grid = np.linspace(r_min, r_max, config.n_mesh)
    initial_guess = np.zeros((4, r_grid.size))
    initial_guess[0] = (r_grid / r_min) ** 3
    initial_guess[1] = 3.0 * r_grid**2 / r_min**3
    initial_guess[2] = -1.0e-6 * (r_grid / r_max) ** (-2)
    initial_guess[3] = 2.0e-6 * r_max**2 / r_grid**3

    result = solve_bvp(
        ode_system,
        static_boundary_conditions if use_quasi_static_boundary else wave_boundary_conditions,
        r_grid,
        initial_guess,
        tol=config.bvp_tol,
        max_nodes=config.max_nodes,
    )

    if not result.success:
        return float("nan")

    fit_region = far_field_slice(result.x)
    r_fit = result.x[fit_region]
    h0_fit = result.y[0][fit_region]
    u1_fit = result.y[2][fit_region]

    basis = np.vstack([r_fit**3, r_fit**(-2)]).T
    h_coeffs, *_ = np.linalg.lstsq(basis, h0_fit, rcond=None)
    u_coeffs, *_ = np.linalg.lstsq(basis, u1_fit, rcond=None)

    tidal_amplitude = h_coeffs[0]
    response_amplitude = u_coeffs[1]

    if abs(tidal_amplitude) < 1.0e-18:
        return float("nan")

    beta = response_amplitude / tidal_amplitude
    return float(beta / config.normalization_scale)


# -----------------------------------------------------------------------------
# Driver utilities
# -----------------------------------------------------------------------------


def compute_axial_scan(
    chi_values: Iterable[float],
    omegas: Iterable[float],
    config: AxialConfig,
) -> list[dict[str, float]]:
    """Run the axial-response scan and return rows suitable for CSV output."""

    rows: list[dict[str, float]] = []
    for omega in omegas:
        print(f"\n### omega = {omega:.6g} ###")
        print(f"{'ell_B/ell_ext':<16} | {'beta(omega)':<16}")
        print("-" * 38)

        for chi in chi_values:
            ell_b = chi * config.ell_ext
            try:
                beta = solve_beta_axial_dynamic(ell_b, omega, config=config)
            except Exception as exc:  # keep long scans alive if one point fails
                beta = float("nan")
                print(f"{chi:<16.6f} | ERROR: {exc}")
            else:
                print(f"{chi:<16.6f} | {beta:<16.8e}")

            rows.append(
                {
                    "omega": float(omega),
                    "ell_B_over_ell_ext": float(chi),
                    "ell_B": float(ell_b),
                    "beta": float(beta),
                }
            )
    return rows


def write_csv(rows: list[dict[str, float]], output_path: Path) -> None:
    """Write scan results to a CSV file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["omega", "ell_B_over_ell_ext", "ell_B", "beta"]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_axial_scan(
    rows: list[dict[str, float]],
    config: AxialConfig,
    figure_path: Path | None = None,
    show: bool = True,
) -> None:
    """Plot beta(omega) versus ell_B/ell_ext without smoothing."""

    plt.figure(figsize=(9.0, 7.0))
    omegas = sorted({row["omega"] for row in rows})

    for omega in omegas:
        omega_rows = [row for row in rows if row["omega"] == omega]
        chi = np.array([row["ell_B_over_ell_ext"] for row in omega_rows], dtype=float)
        beta = np.array([row["beta"] for row in omega_rows], dtype=float)
        mask = np.isfinite(beta)
        if not np.any(mask):
            continue
        plt.plot(
            chi[mask],
            beta[mask],
            "o-",
            linewidth=2.0,
            markersize=5.0,
            label=rf"$\omega M={omega:g}$",
        )

    chi_ref = np.linspace(0.01, 0.99, 600)
    ell_ref = chi_ref * config.ell_ext
    beta_static_ref = -5.8 * config.mass * ell_ref**4
    plt.plot(
        chi_ref,
        beta_static_ref,
        "k--",
        linewidth=2.5,
        label=rf"$-5.8\,M\,\ell_B^4$ static limit",
    )

    plt.xlabel(r"$\ell_B/\ell_{\rm ext}$", fontsize=22)
    plt.ylabel(r"$\beta(\omega)$", fontsize=22)
    plt.title("Axial dynamical response of the Bardeen geometry", fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=16)
    plt.tick_params(axis="both", which="major", labelsize=16)
    plt.tight_layout()

    if figure_path is not None:
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(figure_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {figure_path}")

    if show:
        plt.show()
    else:
        plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute the axial dynamical response beta(omega) for the Bardeen regular black hole.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mass", type=float, default=1.0, help="Black-hole mass M.")
    parser.add_argument(
        "--omegas",
        type=float,
        nargs="+",
        default=[1.0e-4, 1.0e-2, 2.0e-2, 3.0e-2],
        help="Frequencies omega M to scan.",
    )
    parser.add_argument("--chi-min", type=float, default=0.01, help="Minimum ell_B/ell_ext.")
    parser.add_argument("--chi-max", type=float, default=0.99, help="Maximum ell_B/ell_ext.")
    parser.add_argument("--n-chi", type=int, default=10, help="Number of ell_B/ell_ext samples.")
    parser.add_argument("--z-switch", type=float, default=10.0, help="Use quasi-static boundary conditions if omega*r_max < z_switch.")
    parser.add_argument("--bvp-tol", type=float, default=1.0e-3, help="Tolerance passed to scipy.integrate.solve_bvp.")
    parser.add_argument("--max-nodes", type=int, default=300_000, help="Maximum BVP mesh nodes.")
    parser.add_argument("--n-mesh", type=int, default=2_500, help="Initial BVP mesh size.")
    parser.add_argument(
        "--normalization-scale",
        type=float,
        default=1.0e7,
        help="Scale factor applied to the extracted response coefficient.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional CSV output path.")
    parser.add_argument("--figure", type=Path, default=None, help="Optional figure output path.")
    parser.add_argument("--no-show", action="store_true", help="Do not display the plot interactively.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not (0.0 < args.chi_min < args.chi_max < 1.0):
        raise ValueError("Require 0 < chi_min < chi_max < 1 for subextremal Bardeen black holes.")
    if args.n_chi < 2:
        raise ValueError("n-chi must be at least 2.")

    config = AxialConfig(
        mass=args.mass,
        z_switch=args.z_switch,
        bvp_tol=args.bvp_tol,
        max_nodes=args.max_nodes,
        n_mesh=args.n_mesh,
        normalization_scale=args.normalization_scale,
    )

    chi_values = np.linspace(args.chi_min, args.chi_max, args.n_chi)

    print("Bardeen axial dynamical tidal response")
    print("=" * 55)
    print(f"M       = {config.mass:g}")
    print(f"ell_ext = {config.ell_ext:.12g}")
    print(f"z_switch = {config.z_switch:g}")

    rows = compute_axial_scan(chi_values, args.omegas, config)

    if args.output is not None:
        write_csv(rows, args.output)
        print(f"Saved data to {args.output}")

    plot_axial_scan(rows, config, figure_path=args.figure, show=not args.no_show)


if __name__ == "__main__":
    main()
