#!/usr/bin/env python3
"""
Bardeen direct tensor response and tensor Shell-EFT response.

Both sides of the comparison use the same spin-2 Regge-Wheeler potential

    V_T = f [l(l+1)/r^2 - 6m(r)/r^3 + 2m'(r)/r^2].

The direct branch extracts the source-response ratio from the asymptotic
solution. The shell branch integrates the same ingoing exterior solution,
matches it at a finite radius to the regular flat-interior tensor solution,
subtracts the Schwarzschild shell response, and fits the finite renormalized
coefficient.

A direct normalization and a tensor-shell normalization/scheme shift place
the two quantities in the same static convention. They are determined once at
the lowest frequency over ``CHI_CAL`` and are then held fixed for every
deformation and frequency.

This is a probe/test-tensor calculation. It is not the full coupled axial
gravitational-electromagnetic perturbation problem of a Bardeen black hole.
"""

from __future__ import annotations

from math import factorial

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_bvp, solve_ivp
from scipy.interpolate import UnivariateSpline
from scipy.linalg import lstsq
from scipy.optimize import brentq
from scipy.special import spherical_jn, spherical_yn


# =============================================================================
# Global parameters
# =============================================================================

M = 1.0
ELL = 2
RTOL = 1.0e-11
ATOL = 1.0e-13

R_S = 2.0 * M
L_EXT = 4.0 * M / (3.0 * np.sqrt(3.0))
SHELL_R_MIN = 6.0 * M
SHELL_R_MAX = 30.0 * M
SHELL_R_POINTS = 17
SHELL_NEAR_ZONE_MAX = 5.0e-2

frac_vals = np.linspace(0.01, 0.99, 20)
omega_list = [1.0e-4, 2.0e-4]
OUTPUT_PNG = "bardeen_tensor_shell_eft_matched.png"

# Once-and-for-all calibration settings.
OMEGA_CAL = min(omega_list)
CHI_CAL = np.linspace(0.45, 0.90, 6)


# =============================================================================
# Bardeen geometry
# =============================================================================

def f_bardeen(
    r: np.ndarray | float,
    mass: float,
    ell_b: float,
) -> np.ndarray | float:
    """Return the Bardeen metric function."""
    return (
        1.0
        - 2.0 * mass * r**2
        / (r**2 + ell_b**2) ** 1.5
    )


def dfdr_bardeen(
    r: np.ndarray | float,
    mass: float,
    ell_b: float,
) -> np.ndarray | float:
    """Return the radial derivative of the Bardeen metric function."""
    rp2 = r * r + ell_b * ell_b
    return -2.0 * mass * (
        2.0 * r * rp2 ** (-1.5)
        - 3.0 * r**3 * rp2 ** (-2.5)
    )


def mass_bardeen(
    r: np.ndarray | float,
    mass: float,
    ell_b: float,
) -> np.ndarray | float:
    """Return the effective Bardeen mass profile."""
    return (
        mass * r**3
        / (r**2 + ell_b**2) ** 1.5
    )


def dm_bardeen(
    r: np.ndarray | float,
    mass: float,
    ell_b: float,
) -> np.ndarray | float:
    """Return the derivative of the effective Bardeen mass profile."""
    return (
        3.0 * mass * ell_b**2 * r**2
        / (r**2 + ell_b**2) ** 2.5
    )


def V_RW_tensor(
    r: np.ndarray | float,
    mass: float,
    ell_b: float,
    ell: int,
) -> np.ndarray | float:
    """Return the spin-2 Regge-Wheeler potential on the Bardeen background."""
    f_value = f_bardeen(
        r,
        mass,
        ell_b,
    )
    mass_profile = mass_bardeen(
        r,
        mass,
        ell_b,
    )
    mass_profile_derivative = dm_bardeen(
        r,
        mass,
        ell_b,
    )

    return f_value * (
        ell * (ell + 1) / r**2
        - 6.0 * mass_profile / r**3
        + 2.0 * mass_profile_derivative / r**2
    )


def get_outer_horizon(
    mass: float,
    ell_b: float,
) -> float:
    """Locate the outer Bardeen horizon by bracketing the last sign change."""

    def horizon_function(radius: float) -> float:
        return f_bardeen(radius, mass, ell_b)

    grid = np.linspace(
        1.0e-6 * mass,
        80.0 * mass,
        60_000,
    )
    values = horizon_function(grid)
    crossings = np.where(
        values[:-1] * values[1:] < 0.0
    )[0]

    if len(crossings) == 0:
        raise RuntimeError("No horizon found.")

    lower = grid[crossings[-1]]
    upper = grid[crossings[-1] + 1]

    return float(
        brentq(
            horizon_function,
            lower,
            upper,
        )
    )


def bardeen_derivs(
    r: np.ndarray | float,
    mass: float,
    ell_b: float,
) -> tuple[
    np.ndarray | float,
    np.ndarray | float,
    np.ndarray | float,
    np.ndarray | float,
]:
    """
    Return f, f', and two zero placeholders used by the direct BVP code.
    """
    r_in = np.atleast_1d(r).astype(float)

    f_value = f_bardeen(
        r_in,
        mass,
        ell_b,
    )
    fp_value = dfdr_bardeen(
        r_in,
        mass,
        ell_b,
    )

    l_f = np.zeros_like(r_in)
    l_ff = np.zeros_like(r_in)

    if np.isscalar(r):
        return (
            float(f_value[0]),
            float(fp_value[0]),
            float(l_f[0]),
            float(l_ff[0]),
        )

    return f_value, fp_value, l_f, l_ff


def far_field_slice(
    r: np.ndarray,
) -> slice:
    """Return the far-field slice used in the asymptotic response fit."""
    return slice(
        int(0.6 * len(r)),
        -5,
    )


def build_bessel_matrix(
    r_fit: np.ndarray,
    omega: float,
    L: int = 2,
) -> np.ndarray:
    """Return the spherical-Bessel source-response design matrix."""
    z = omega * r_fit
    j_value = spherical_jn(L, z)
    y_value = spherical_yn(L, z)

    return np.vstack(
        [j_value, y_value]
    ).T


# =============================================================================
# 1. Direct numerical Bardeen test-tensor response
# =============================================================================

def solve_dynamic_exact_bc_tensor_bardeen_raw(
    mass: float,
    ell_param: float,
    omega: float,
) -> float:
    """
    Return the raw direct source-response ratio.
    """
    ratio = ell_param / L_EXT
    r_h = get_outer_horizon(
        mass,
        ell_param,
    )

    r_min = r_h + 0.5e-4
    r_max = 20.0 + 127.0 * ratio

    l_harm = 2

    def radial_system(
        r: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        f_value, fp_value, _, _ = bardeen_derivs(
            r,
            mass,
            ell_param,
        )
        f_value = np.maximum(
            f_value,
            1.0e-7,
        )

        potential = V_RW_tensor(
            r,
            mass,
            ell_param,
            l_harm,
        )

        coefficient = (
            omega**2 - potential
        ) / f_value**2
        fp_over_f = fp_value / f_value

        (
            re_psi,
            re_psi_prime,
            im_psi,
            im_psi_prime,
        ) = y

        re_psi_double_prime = (
            -fp_over_f * re_psi_prime
            - coefficient * re_psi
        )
        im_psi_double_prime = (
            -fp_over_f * im_psi_prime
            - coefficient * im_psi
        )

        return np.vstack(
            (
                re_psi_prime,
                re_psi_double_prime,
                im_psi_prime,
                im_psi_double_prime,
            )
        )

    def boundary_conditions(
        y_inner: np.ndarray,
        y_outer: np.ndarray,
    ) -> np.ndarray:
        f_min, _, _, _ = bardeen_derivs(
            np.array([r_min]),
            mass,
            ell_param,
        )
        k_omega = omega / f_min[0]

        bc_1 = (
            y_inner[1]
            - k_omega * y_inner[2]
        )
        bc_2 = (
            y_inner[3]
            + k_omega * y_inner[0]
        )

        z_inf = omega * r_max
        y_value = spherical_yn(
            l_harm,
            z_inf,
        )
        y_prime_value = spherical_yn(
            l_harm,
            z_inf,
            derivative=True,
        )
        target = (
            1.0
            / (omega * r_max**2)
        )

        bc_3 = (
            omega * y_prime_value * y_outer[0]
            - y_value * y_outer[1]
            - target
        )
        bc_4 = (
            omega * y_prime_value * y_outer[2]
            - y_value * y_outer[3]
        )

        return np.array(
            [bc_1, bc_2, bc_3, bc_4]
        )

    x_initial = np.linspace(
        r_min,
        r_max,
        2_000,
    )
    y_initial = np.zeros(
        (4, x_initial.size)
    )
    z_initial = omega * x_initial
    y_initial[0] = spherical_jn(
        l_harm,
        z_initial,
    )

    result = solve_bvp(
        radial_system,
        boundary_conditions,
        x_initial,
        y_initial,
        tol=1.0e-6,
        max_nodes=50_000,
    )

    if not result.success:
        return np.nan

    fit_slice = far_field_slice(
        result.x
    )
    r_fit = result.x[fit_slice]
    psi_complex = (
        result.y[0][fit_slice]
        + 1j * result.y[2][fit_slice]
    )

    z_max = omega * np.max(
        r_fit
    )

    if z_max < 0.1:
        source_column = r_fit**l_harm
        response_column = r_fit ** (-(l_harm + 1))
        design_matrix = np.vstack(
            [source_column, response_column]
        ).T

        coefficients = lstsq(
            design_matrix,
            np.real(psi_complex),
            lapack_driver="gelsy",
        )[0]

        c_tidal = coefficients[0]
        c_response = coefficients[1]

        if abs(c_tidal) < 1.0e-20:
            return 0.0

        return float(
            -c_response / c_tidal
        )

    design_matrix = build_bessel_matrix(
        r_fit,
        omega,
        L=l_harm,
    )
    coefficients = lstsq(
        design_matrix,
        psi_complex,
        lapack_driver="gelsy",
    )[0]

    a_complex = coefficients[0]
    b_complex = coefficients[1]
    response_ratio = (
        b_complex / a_complex
    )
    alpha_complex = (
        45.0
        * response_ratio
        / omega**5
    )

    return float(
        np.real(alpha_complex)
    )


# =============================================================================
# 2. Tensor Shell-EFT response
# =============================================================================

def rhs_RW_dynamic(
    r: float,
    y: np.ndarray,
    omega: float,
    mass: float,
    ell_b: float,
    ell: int,
) -> list[complex]:
    """Return the first-order tensor Regge-Wheeler radial system."""
    psi, momentum = y

    f_value = f_bardeen(
        r,
        mass,
        ell_b,
    )
    potential = V_RW_tensor(
        r,
        mass,
        ell_b,
        ell,
    )

    if abs(f_value) < 1.0e-16:
        f_value = (
            np.sign(f_value)
            * 1.0e-16
        )

    dpsi_dr = (
        momentum / f_value
    )
    dmomentum_dr = (
        (potential - omega**2)
        * psi
        / f_value
    )

    return [
        dpsi_dr,
        dmomentum_dr,
    ]


def solve_outside_RW_dynamic(
    omega: float,
    mass: float,
    ell_b: float,
    ell: int,
    r_shell: float,
    eps: float = 1.0e-6,
) -> tuple[complex, complex]:
    """Integrate the ingoing exterior tensor solution to the shell."""
    r_h = get_outer_horizon(
        mass,
        ell_b,
    )
    r_0 = r_h * (
        1.0 + eps
    )

    psi_0 = 1.0 + 0.0j
    momentum_0 = (
        -1j * omega * psi_0
    )

    real_solution = solve_ivp(
        fun=lambda r, y: rhs_RW_dynamic(
            r,
            y,
            omega,
            mass,
            ell_b,
            ell,
        ),
        t_span=(r_0, r_shell),
        y0=[
            psi_0.real,
            momentum_0.real,
        ],
        method="DOP853",
        rtol=RTOL,
        atol=ATOL,
    )

    imaginary_solution = solve_ivp(
        fun=lambda r, y: rhs_RW_dynamic(
            r,
            y,
            omega,
            mass,
            ell_b,
            ell,
        ),
        t_span=(r_0, r_shell),
        y0=[
            psi_0.imag,
            momentum_0.imag,
        ],
        method="DOP853",
        rtol=RTOL,
        atol=ATOL,
    )

    if (
        not real_solution.success
        or not imaginary_solution.success
    ):
        raise RuntimeError(
            "Outside dynamic integration failed."
        )

    psi_at_shell = (
        real_solution.y[0, -1]
        + 1j * imaginary_solution.y[0, -1]
    )
    momentum_at_shell = (
        real_solution.y[1, -1]
        + 1j * imaginary_solution.y[1, -1]
    )

    f_at_shell = f_bardeen(
        r_shell,
        mass,
        ell_b,
    )
    dpsi_dr_at_shell = (
        momentum_at_shell
        / f_at_shell
    )

    return (
        psi_at_shell,
        dpsi_dr_at_shell,
    )


def psi_inside_RW_dynamic(
    r_shell: float,
    omega: float,
    mass: float,
    ell_b: float,
    ell: int,
) -> tuple[float, float]:
    """Return the regular flat-interior tensor solution at the shell."""
    f_at_shell = f_bardeen(
        r_shell,
        mass,
        ell_b,
    )

    if f_at_shell <= 0:
        raise ValueError(
            "Shell must be outside horizon: f(R)>0"
        )

    sqrt_f = np.sqrt(
        f_at_shell
    )
    r_tilde = (
        r_shell / sqrt_f
    )
    omega_tilde = (
        omega / sqrt_f
    )

    argument = (
        omega_tilde * r_tilde
    )

    j_value = spherical_jn(
        ell,
        argument,
    )
    j_prime_value = spherical_jn(
        ell,
        argument,
        derivative=True,
    )

    psi_inside = (
        r_tilde * j_value
    )
    dpsi_dr_tilde = (
        j_value
        + r_tilde
        * j_prime_value
        * omega_tilde
    )
    dpsi_dr_inside = (
        dpsi_dr_tilde / sqrt_f
    )

    return (
        psi_inside,
        dpsi_dr_inside,
    )


def F_shell_RW_dynamic(
    omega: float,
    mass: float,
    ell_b: float,
    ell: int,
    r_shell: float,
) -> complex:
    """Return the finite-radius tensor shell response."""
    (
        psi_outside,
        dpsi_dr_outside,
    ) = solve_outside_RW_dynamic(
        omega,
        mass,
        ell_b,
        ell,
        r_shell,
    )

    (
        psi_inside,
        dpsi_dr_inside,
    ) = psi_inside_RW_dynamic(
        r_shell,
        omega,
        mass,
        ell_b,
        ell,
    )

    matching_scale = (
        psi_inside / psi_outside
    )
    dpsi_dr_outside *= matching_scale

    derivative_jump = (
        dpsi_dr_inside
        - dpsi_dr_outside
    )
    f_at_shell = f_bardeen(
        r_shell,
        mass,
        ell_b,
    )

    prefactor = (
        4.0
        * np.pi
        * factorial(ell)
        * 2.0**ell
        * r_shell ** (2 * ell + 2)
        / factorial(2 * ell + 1)
    )
    prefactor *= (
        np.sqrt(f_at_shell)
        ** (2 * ell + 1)
    )

    return prefactor * (
        derivative_jump / psi_inside
    )


def make_R_list(
    chi: float,
    ell_b: float,
    nR: int = SHELL_R_POINTS,
) -> np.ndarray:
    """Return the fixed shell-radius grid defining the subtraction scheme."""
    del chi, ell_b

    return np.linspace(
        SHELL_R_MIN,
        SHELL_R_MAX,
        nR,
    )


def fit_delta_running_dynamic(
    omega: float,
    ell_b: float,
    r_list_local: np.ndarray,
) -> tuple[complex, complex]:
    """Fit the logarithmic and finite shell-response coefficients."""
    if np.max(
        np.abs(
            omega
            * r_list_local
        )
    ) > SHELL_NEAR_ZONE_MAX:
        raise ValueError(
            "Shell fit requires max(|omega*R|) <= "
            f"{SHELL_NEAR_ZONE_MAX:.3g}."
        )

    log_x = np.log(
        R_S / r_list_local
    )
    x_1 = (
        R_S / r_list_local
    )
    x_2 = x_1**2
    x_3 = x_1**3

    f_bardeen_values = np.array(
        [
            F_shell_RW_dynamic(
                omega,
                M,
                ell_b,
                ELL,
                radius,
            )
            for radius in r_list_local
        ],
        dtype=complex,
    )

    f_schwarzschild_values = np.array(
        [
            F_shell_RW_dynamic(
                omega,
                M,
                0.0,
                ELL,
                radius,
            )
            for radius in r_list_local
        ],
        dtype=complex,
    )

    delta_f = (
        f_bardeen_values
        - f_schwarzschild_values
    )

    design_matrix = np.vstack(
        [
            log_x,
            np.ones_like(log_x),
            x_1,
            x_2,
            x_3,
        ]
    ).T

    def stable_solve(
        data: np.ndarray,
    ) -> np.ndarray:
        return lstsq(
            design_matrix,
            data,
            lapack_driver="gelsy",
        )[0]

    parameters_real = stable_solve(
        delta_f.real
    )
    parameters_imaginary = stable_solve(
        delta_f.imag
    )

    delta_a = (
        parameters_real[0]
        + 1j * parameters_imaginary[0]
    )
    delta_b = (
        parameters_real[1]
        + 1j * parameters_imaginary[1]
    )

    return delta_a, delta_b


def shell_tensor_response_raw(
    delta_a: complex,
    delta_b: complex,
    beta: float,
    ell: int = ELL,
) -> complex:
    """
    Return the finite tensor Shell-EFT coefficient in the chosen scheme.

    The finite scheme is specified by ``B -> B + beta*A``. The factor
    ``3/(4*pi)`` is the fixed quadrupolar shell-response convention.
    """
    del ell

    return (
        3.0 / (4.0 * np.pi)
    ) * (
        delta_b
        + beta * delta_a
    )


def solve_shell_running_raw(
    mass: float,
    ell_b: float,
    omega: float,
) -> tuple[complex, complex]:
    """Return the logarithmic and finite tensor-shell fit coefficients."""
    del mass

    chi = ell_b / L_EXT
    r_list_local = make_R_list(
        chi,
        ell_b,
        nR=SHELL_R_POINTS,
    )

    return fit_delta_running_dynamic(
        omega,
        ell_b,
        r_list_local,
    )


def solve_shell_tensor_raw(
    mass: float,
    ell_b: float,
    omega: float,
    beta: float,
) -> float:
    """Return the raw renormalized tensor Shell-EFT response."""
    delta_a, delta_b = solve_shell_running_raw(
        mass,
        ell_b,
        omega,
    )

    return float(
        np.real(
            shell_tensor_response_raw(
                delta_a,
                delta_b,
                beta,
                ell=ELL,
            )
        )
    )


# =============================================================================
# 3. Once-and-for-all calibration
# =============================================================================

def coviello_static_bardeen(
    mass: float,
    ell_b: float,
) -> float:
    """
    Return the leading small-deformation s=l=2 benchmark.

    Lambda = (42/5) M ell_b^4.
    """
    return (
        42.0 / 5.0
    ) * mass * ell_b**4


def through_origin_factor(
    raw_values: np.ndarray,
    target_values: np.ndarray,
) -> float:
    """Return the least-squares scale factor for a fit through the origin."""
    raw_array = np.asarray(
        raw_values,
        dtype=float,
    )
    target_array = np.asarray(
        target_values,
        dtype=float,
    )

    finite_mask = (
        np.isfinite(raw_array)
        & np.isfinite(target_array)
    )

    raw_array = raw_array[
        finite_mask
    ]
    target_array = target_array[
        finite_mask
    ]

    denominator = np.dot(
        raw_array,
        raw_array,
    )

    if denominator <= 0.0:
        raise RuntimeError(
            "Cannot determine calibration factor."
        )

    return float(
        np.dot(
            raw_array,
            target_array,
        )
        / denominator
    )


def relative_rms(
    model: np.ndarray,
    target: np.ndarray,
) -> float:
    """Return the RMS residual normalized by the RMS target amplitude."""
    model_array = np.asarray(
        model,
        dtype=float,
    )
    target_array = np.asarray(
        target,
        dtype=float,
    )

    scale = max(
        np.sqrt(
            np.mean(
                target_array**2
            )
        ),
        1.0e-30,
    )

    return float(
        np.sqrt(
            np.mean(
                (
                    model_array
                    - target_array
                ) ** 2
            )
        )
        / scale
    )


def calibrate_once() -> tuple[float, float, float]:
    """
    Determine the direct normalization and tensor-shell scheme once.

    The tensor-shell fit permits the finite renormalization
    ``B -> B + beta*A``. Both ``beta`` and the overall normalization are
    fitted at ``OMEGA_CAL`` and then held fixed for every deformation and
    frequency.
    """
    raw_numerical_values = []
    raw_shell_a_values = []
    raw_shell_b_values = []
    target_values = []

    print(
        "\nComputing once-and-for-all calibration..."
    )
    print(
        f"omega_cal = {OMEGA_CAL:.3e}"
    )

    for chi in CHI_CAL:
        ell_b = chi * L_EXT

        raw_numerical_values.append(
            solve_dynamic_exact_bc_tensor_bardeen_raw(
                M,
                ell_b,
                OMEGA_CAL,
            )
        )

        delta_a, delta_b = solve_shell_running_raw(
            M,
            ell_b,
            OMEGA_CAL,
        )
        shell_convention = 3.0 / (4.0 * np.pi)
        raw_shell_a_values.append(
            shell_convention
            * np.real(delta_a)
        )
        raw_shell_b_values.append(
            shell_convention
            * np.real(delta_b)
        )

        target_values.append(
            coviello_static_bardeen(
                M,
                ell_b,
            )
        )

    raw_numerical_array = np.asarray(
        raw_numerical_values,
        dtype=float,
    )
    raw_shell_a_array = np.asarray(
        raw_shell_a_values,
        dtype=float,
    )
    raw_shell_b_array = np.asarray(
        raw_shell_b_values,
        dtype=float,
    )
    target_array = np.asarray(
        target_values,
        dtype=float,
    )

    z_numerical = through_origin_factor(
        raw_numerical_array,
        target_array,
    )
    shell_design_matrix = np.column_stack(
        [
            raw_shell_b_array,
            raw_shell_a_array,
        ]
    )
    shell_coefficients = lstsq(
        shell_design_matrix,
        target_array,
        lapack_driver="gelsy",
    )[0]
    z_shell = float(
        shell_coefficients[0]
    )
    if abs(z_shell) < 1.0e-30:
        raise RuntimeError(
            "Cannot determine the tensor-shell finite scheme."
        )
    beta_shell = float(
        shell_coefficients[1]
        / z_shell
    )
    raw_shell_array = (
        raw_shell_b_array
        + beta_shell
        * raw_shell_a_array
    )

    numerical_residual = relative_rms(
        z_numerical
        * raw_numerical_array,
        target_array,
    )
    shell_residual = relative_rms(
        z_shell
        * raw_shell_array,
        target_array,
    )

    print(
        f"Z_NUM   = {z_numerical:.12e}"
    )
    print(
        f"Z_SHELL = {z_shell:.12e}"
    )
    print(
        f"BETA_SHELL = {beta_shell:.12e}"
    )
    print(
        "Direct calibration relative RMS residual "
        f"= {numerical_residual:.3e}"
    )
    print(
        "Tensor-shell calibration relative RMS residual "
        f"= {shell_residual:.3e}"
    )

    return z_numerical, z_shell, beta_shell


# =============================================================================
# 4. Frequency and deformation scan
# =============================================================================

def run_comparison() -> None:
    """Run the calibrated comparison and display the final plot."""
    (
        z_numerical,
        z_shell,
        beta_shell,
    ) = calibrate_once()

    plt.figure(
        figsize=(9.2, 6.5)
    )

    print(
        "\nBardeen comparison: calibrated direct tensor "
        "vs once-calibrated tensor Shell-EFT"
    )
    print(
        "=" * 100
    )

    for omega in omega_list:
        print(
            f"\n### omega = {omega:.3e} ###"
        )
        print(
            f"{'chi':<10} | "
            f"{'direct':<18} | "
            f"{'tensor shell':<18} | "
            f"{'diff':<18}"
        )
        print(
            "-" * 78
        )

        direct_values = []
        shell_values = []

        for chi in frac_vals:
            ell_b = chi * L_EXT

            try:
                direct_raw = (
                    solve_dynamic_exact_bc_tensor_bardeen_raw(
                        M,
                        ell_b,
                        omega,
                    )
                )
                direct_response = (
                    z_numerical
                    * direct_raw
                )
            except Exception as exc:
                direct_response = np.nan
                print(
                    f"direct error at chi={chi:.3f}: "
                    f"{exc}"
                )

            try:
                shell_raw = solve_shell_tensor_raw(
                    M,
                    ell_b,
                    omega,
                    beta_shell,
                )
                shell_response = (
                    z_shell
                    * shell_raw
                )
            except Exception as exc:
                shell_response = np.nan
                print(
                    f"shell error at chi={chi:.3f}: "
                    f"{exc}"
                )

            direct_values.append(
                direct_response
            )
            shell_values.append(
                shell_response
            )

            print(
                f"{chi:<10.3f} | "
                f"{direct_response:<18.8e} | "
                f"{shell_response:<18.8e} | "
                f"{(direct_response-shell_response):<18.8e}"
            )

        direct_array = np.asarray(
            direct_values,
            dtype=float,
        )
        shell_array = np.asarray(
            shell_values,
            dtype=float,
        )

        direct_mask = np.isfinite(
            direct_array
        )
        shell_mask = np.isfinite(
            shell_array
        )

        if np.sum(direct_mask) > 3:
            x_smooth = np.linspace(
                frac_vals[direct_mask].min(),
                frac_vals[direct_mask].max(),
                300,
            )
            direct_spline = UnivariateSpline(
                frac_vals[direct_mask],
                direct_array[direct_mask],
                s=0,
            )(x_smooth)

            plt.plot(
                x_smooth,
                direct_spline,
                lw=2.0,
                label=(
                    rf"Direct tensor, "
                    rf"$\omega={omega}$"
                ),
            )
        else:
            plt.plot(
                frac_vals,
                direct_array,
                "o-",
                label=(
                    rf"Direct, "
                    rf"$\omega={omega}$"
                ),
            )

        if np.sum(shell_mask) > 3:
            x_smooth = np.linspace(
                frac_vals[shell_mask].min(),
                frac_vals[shell_mask].max(),
                300,
            )
            shell_spline = UnivariateSpline(
                frac_vals[shell_mask],
                shell_array[shell_mask],
                s=0,
            )(x_smooth)

            plt.plot(
                x_smooth,
                shell_spline,
                "--",
                lw=2.2,
                label=(
                    rf"Tensor Shell-EFT, "
                    rf"$\omega={omega}$"
                ),
            )
        else:
            plt.plot(
                frac_vals,
                shell_array,
                "s--",
                label=(
                    rf"Tensor Shell-EFT, "
                    rf"$\omega={omega}$"
                ),
            )

    plt.xlabel(
        r"$\ell_B/\ell_{\rm ext}$",
        fontsize=22,
    )
    plt.ylabel(
        r"Calibrated response",
        fontsize=22,
    )
    plt.title(
        "Bardeen: direct tensor vs tensor Shell-EFT",
        fontsize=16,
    )
    plt.grid(
        True,
        alpha=0.35,
    )
    plt.legend(
        fontsize=15,
    )
    plt.tick_params(
        axis="both",
        which="major",
        labelsize=16,
    )
    plt.tight_layout()
    plt.savefig(
        OUTPUT_PNG,
        dpi=220,
        bbox_inches="tight",
    )
    print(
        f"\nSaved figure to {OUTPUT_PNG}"
    )
    plt.show()


def main() -> int:
    """Execute the calibrated Bardeen comparison."""
    run_comparison()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
