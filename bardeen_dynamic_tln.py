"""Dynamic tidal Love number for the Bardeen regular black hole.

This module contains the numerical boundary-value solver used to compute the
frequency-dependent quadrupolar tidal response of the Bardeen geometry.  The
implementation is written as research code: the differential system is kept
close to the equations used in the manuscript, while solver parameters,
post-processing and plotting are separated from the core numerical routine.

Units and conventions
---------------------
All quantities are evaluated in geometrized units with ``G = c = 1``.  The
black-hole mass is denoted by ``M`` and the Bardeen length scale by ``ell_b``.
The extremal value is

    ell_ext = 4 M / (3 sqrt(3)).

The default output is the dimensionless normalized response used in the
accompanying paper, obtained by dividing the raw response by ``1e7`` (merely a choice of normalization).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_bvp
from scipy.linalg import lstsq
from scipy.interpolate import UnivariateSpline
from scipy.optimize import brentq
from scipy.special import spherical_jn, spherical_yn


FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class SolverConfig:
    """Numerical controls for the dynamic TLN boundary-value problem."""

    mass: float = 1.0
    harmonic_index: int = 2
    bvp_tolerance: float = 1.0e-6
    max_nodes: int = 50_000
    initial_mesh_points: int = 2_000
    horizon_offset: float = 5.0e-5
    far_field_start_fraction: float = 0.60
    low_frequency_threshold: float = 0.10
    love_normalization: float = 1.0e7
    min_lf_abs: float = 1.0e-12
    min_f: float = 1.0e-7

    def __post_init__(self) -> None:
        if self.mass <= 0:
            raise ValueError("mass must be positive")
        if self.harmonic_index < 2:
            raise ValueError("harmonic_index must be >= 2 for tidal perturbations")
        if not (0.0 < self.far_field_start_fraction < 1.0):
            raise ValueError("far_field_start_fraction must lie between 0 and 1")
        if self.bvp_tolerance <= 0:
            raise ValueError("bvp_tolerance must be positive")
        if self.initial_mesh_points < 50:
            raise ValueError("initial_mesh_points is too small for a stable BVP mesh")


@dataclass(frozen=True)
class LoveNumberResult:
    """One computed dynamic Love-number value."""

    ell_ratio: float
    ell_b: float
    omega: float
    alpha: float
    success: bool = True
    message: str = ""


def extremal_length(mass: float) -> float:
    """Return the extremal Bardeen length ``ell_ext = 4M/(3 sqrt(3))``."""

    if mass <= 0:
        raise ValueError("mass must be positive")
    return 4.0 * mass / (3.0 * np.sqrt(3.0))


def get_outer_horizon(mass: float, ell_b: float) -> float:
    """Return the outer horizon of the Bardeen metric.

    The horizon is the largest positive zero of

        f(r) = 1 - 2 M r^2 / (r^2 + ell_b^2)^(3/2).

    For ``0 <= ell_b < ell_ext`` there are two horizons and the larger root is
    returned.  The extremal point is numerically delicate, so scans should stay
    slightly below ``ell_b / ell_ext = 1``.
    """

    if mass <= 0:
        raise ValueError("mass must be positive")
    if ell_b < 0:
        raise ValueError("ell_b must be non-negative")
    if ell_b == 0:
        return 2.0 * mass

    def horizon_function(r: float) -> float:
        return (r * r + ell_b * ell_b) ** 1.5 - 2.0 * mass * r * r

    ell_ext = extremal_length(mass)
    if ell_b > ell_ext:
        raise ValueError(
            f"ell_b={ell_b:.8g} exceeds the extremal value ell_ext={ell_ext:.8g}; "
            "no black-hole horizon exists."
        )

    # Locate all sign changes on a dense positive grid and take the largest root.
    r_min = max(1.0e-8 * mass, 1.0e-8)
    r_max = 4.0 * mass + 4.0 * ell_b
    grid = np.geomspace(r_min, r_max, 4_000)
    values = np.array([horizon_function(x) for x in grid])
    sign_change_indices = np.where(values[:-1] * values[1:] <= 0.0)[0]

    roots: list[float] = []
    for idx in sign_change_indices:
        a, b = grid[idx], grid[idx + 1]
        try:
            roots.append(brentq(horizon_function, a, b))
        except ValueError:
            continue

    if not roots:
        # At exact extremality the two roots merge and a sign-change search can fail.
        r_ext = np.sqrt(2.0) * ell_b
        residual = abs(horizon_function(r_ext))
        if residual < 1.0e-7 * mass**3:
            return r_ext
        raise RuntimeError("could not locate the Bardeen outer horizon")

    return float(max(roots))


def bardeen_background(
    r: ArrayLike,
    mass: float,
    ell_b: float,
) -> tuple[FloatArray | float, FloatArray | float, FloatArray | float, FloatArray | float]:
    """Return ``f``, ``f'``, ``L_F`` and ``L_FF`` for the Bardeen background."""

    scalar_input = np.isscalar(r)
    r_arr = np.atleast_1d(np.asarray(r, dtype=float))

    if np.any(r_arr <= 0):
        raise ValueError("all radial points must be positive")
    if mass <= 0:
        raise ValueError("mass must be positive")
    if ell_b < 0:
        raise ValueError("ell_b must be non-negative")

    r2 = r_arr * r_arr
    ell2 = ell_b * ell_b
    denom = r2 + ell2

    m_of_r = mass * r_arr**3 * denom ** (-1.5)
    f = 1.0 - 2.0 * m_of_r / r_arr

    dm_dr = 3.0 * mass * ell2 * r2 * denom ** (-2.5)
    df_dr = 2.0 * m_of_r / r2 - 2.0 * dm_dr / r_arr

    if ell_b < 1.0e-12:
        l_f = np.zeros_like(r_arr)
        l_ff = np.zeros_like(r_arr)
    else:
        d_l_dr = -15.0 * mass * ell2 * r_arr * denom ** (-3.5)
        d_finv_dr = -2.0 * ell2 / r_arr**5
        l_f = d_l_dr / d_finv_dr
        d_l_f_dr = 7.5 * mass * (
            6.0 * r_arr**5 * denom ** (-3.5)
            - 7.0 * r_arr**7 * denom ** (-4.5)
        )
        l_ff = d_l_f_dr / d_finv_dr

    if scalar_input:
        return f[0], df_dr[0], l_f[0], l_ff[0]
    return f, df_dr, l_f, l_ff


def far_field_slice(radial_grid: FloatArray, start_fraction: float = 0.60) -> slice:
    """Return the default far-field fitting window."""

    if radial_grid.ndim != 1:
        raise ValueError("radial_grid must be one-dimensional")
    if len(radial_grid) < 20:
        raise ValueError("radial_grid is too short for a stable far-field fit")
    if not (0.0 < start_fraction < 1.0):
        raise ValueError("start_fraction must lie between 0 and 1")
    return slice(int(start_fraction * len(radial_grid)), -5)


def build_bessel_matrix(r_fit: FloatArray, omega: float, harmonic_index: int = 2) -> FloatArray:
    """Build the spherical-Bessel design matrix for far-field matching."""

    if omega <= 0:
        raise ValueError("omega must be positive")
    z = omega * r_fit
    return np.vstack(
        [spherical_jn(harmonic_index, z), spherical_yn(harmonic_index, z)]
    ).T


def _system_rhs(
    r: FloatArray,
    y: FloatArray,
    *,
    mass: float,
    ell_b: float,
    omega: float,
    config: SolverConfig,
) -> FloatArray:
    """Right-hand side of the coupled real/imaginary BVP system."""

    f, fp, l_f, l_ff = bardeen_background(r, mass, ell_b)
    f = np.maximum(np.asarray(f), config.min_f)
    fp = np.asarray(fp)
    l_f = np.asarray(l_f)
    l_ff = np.asarray(l_ff)

    harmonic = float(config.harmonic_index)
    lambda_l = harmonic * (harmonic + 1.0)
    r2 = r * r
    r4 = r2 * r2

    delta = (harmonic - 1.0) * (harmonic + 2.0) * r2 + 4.0 * ell_b**2 * l_f
    k_num = r2 * f * (lambda_l - 2.0 * f) + r4 * fp**2
    k_h = -k_num / (delta * f)
    k_hp = -r4 * fp / delta
    k_u = 4.0 * ell_b * lambda_l * l_f / delta
    k_up = 4.0 * ell_b * r * l_f * (r * fp + 2.0 * f) / delta

    ratio_l = np.zeros_like(r)
    mask_lf = np.abs(l_f) > config.min_lf_abs
    if np.any(mask_lf):
        ratio_l[mask_lf] = ell_b**2 * l_ff[mask_lf] / (r4[mask_lf] * l_f[mask_lf])

    p_coef = fp / f
    if ell_b > 1.0e-12 and np.any(mask_lf):
        p_coef[mask_lf] -= (
            2.0 * ell_b**2 * l_ff[mask_lf] / (r[mask_lf] ** 5 * l_f[mask_lf])
        )

    q_coef = -lambda_l / (r2 * f) * (1.0 + ratio_l)
    r_coef = ell_b / (r2 * f) * (1.0 + ratio_l)
    corrected_lf = l_f * (1.0 + ratio_l)

    eta1 = (4.0 * ell_b**2 / delta) * fp * corrected_lf - fp - 2.0 * f / r
    eta2_core = lambda_l / r2 + fp**2 / f - 2.0 * f / r2
    eta2 = (
        (4.0 * ell_b**2 / delta) * corrected_lf * eta2_core
        - 2.0 * fp / r
        + fp**2 / f
        - 2.0 * f / r2
        + (lambda_l + 2.0) / r2
        - 4.0 * ell_b**2 * l_f / r4
    )

    j3_bracket = (
        4.0 * ell_b**4 * l_f * l_ff / (delta * r4)
        + 4.0 * ell_b**2 * l_f**2 / delta
        + l_f
    )
    j_up = -4.0 * ell_b / r2 * (2.0 * f * l_f / r - (fp + 2.0 * f / r) * j3_bracket)
    j_u = -4.0 * ell_b * 24.0 * corrected_lf / (r2 * delta)

    denom_u4 = 1.0 + r_coef * k_up
    omega2_over_f2 = omega**2 / f**2

    rhs_u_real = -(
        p_coef * y[3]
        + q_coef * y[2]
        + r_coef * (k_h * y[0] + k_hp * y[1] + k_u * y[2])
    )
    re_upp = rhs_u_real / denom_u4
    re_hpp_static = (eta1 * y[1] + eta2 * y[0] - (j_up * y[3] + j_u * y[2])) / f
    re_hpp = re_hpp_static - omega2_over_f2 * y[0]

    rhs_u_imag = -(
        p_coef * y[7]
        + q_coef * y[6]
        + r_coef * (k_h * y[4] + k_hp * y[5] + k_u * y[6])
    )
    im_upp = rhs_u_imag / denom_u4
    im_hpp_static = (eta1 * y[5] + eta2 * y[4] - (j_up * y[7] + j_u * y[6])) / f
    im_hpp = im_hpp_static - omega2_over_f2 * y[4]

    return np.vstack((y[1], re_hpp, y[3], re_upp, y[5], im_hpp, y[7], im_upp))


def solve_dynamic_love_number(
    ell_b: float,
    omega: float,
    config: SolverConfig | None = None,
    *,
    return_nan_on_failure: bool = False,
) -> float:
    """Compute the normalized dynamic quadrupolar Love number.

    Parameters
    ----------
    ell_b:
        Bardeen length scale.
    omega:
        Driving frequency.  The dimensionless frequency plotted in the paper is
        ``omega * M``; the default ``M=1`` therefore coincides with ``omega``.
    config:
        Solver and normalization settings.
    return_nan_on_failure:
        If true, failed BVP solves return ``np.nan`` instead of raising.
    """

    config = config or SolverConfig()
    mass = config.mass

    if ell_b < 0:
        raise ValueError("ell_b must be non-negative")
    if omega <= 0:
        raise ValueError("omega must be positive")

    try:
        ell_ext = extremal_length(mass)
        ell_ratio = ell_b / ell_ext
        r_h = get_outer_horizon(mass, ell_b)

        r_min = r_h + config.horizon_offset
        r_max = 0.8 + 125.0 * ell_ratio

        def fun(r: FloatArray, y: FloatArray) -> FloatArray:
            return _system_rhs(r, y, mass=mass, ell_b=ell_b, omega=omega, config=config)

        def bc(ya: FloatArray, yb: FloatArray) -> FloatArray:
            f_min, _, _, _ = bardeen_background(np.array([r_min]), mass, ell_b)
            k_wave = omega / np.asarray(f_min)[0]

            # Near-horizon ingoing-wave conditions for real and imaginary parts.
            bc1 = ya[1] - k_wave * ya[4]
            bc2 = ya[5] + k_wave * ya[0]
            bc3 = ya[3] - k_wave * ya[6]
            bc4 = ya[7] + k_wave * ya[2]

            # Far-field normalization in the spherical-Bessel basis.
            z_inf = omega * r_max
            y_val = spherical_yn(config.harmonic_index, z_inf)
            yp_val = spherical_yn(config.harmonic_index, z_inf, derivative=True)
            target = 1.0 / (omega * r_max**2)

            bc5 = omega * yp_val * yb[0] - y_val * yb[1] - target
            bc6 = omega * yp_val * yb[4] - y_val * yb[5]
            bc7 = yb[2]
            bc8 = yb[6]

            return np.array([bc1, bc2, bc3, bc4, bc5, bc6, bc7, bc8])

        radial_grid = np.linspace(r_min, r_max, config.initial_mesh_points)
        initial_guess = np.zeros((8, radial_grid.size))
        initial_guess[0] = spherical_jn(config.harmonic_index, omega * radial_grid)

        result = solve_bvp(
            fun,
            bc,
            radial_grid,
            initial_guess,
            tol=config.bvp_tolerance,
            max_nodes=config.max_nodes,
        )
        if not result.success:
            raise RuntimeError(result.message)

        fit_window = far_field_slice(result.x, config.far_field_start_fraction)
        r_fit = result.x[fit_window]
        h_complex = result.y[0][fit_window] + 1j * result.y[4][fit_window]
        z_max = omega * np.max(r_fit)

        if z_max < config.low_frequency_threshold:
            # Static/low-frequency matching: H ~ C_tidal r^L + C_resp r^(-L-1).
            harmonic = config.harmonic_index
            design = np.vstack([r_fit**harmonic, r_fit ** (-(harmonic + 1))]).T
            coeffs = lstsq(design, np.real(h_complex), lapack_driver="gelsy")[0]
            c_tidal, c_response = coeffs
            if abs(c_tidal) < 1.0e-16:
                return 0.0
            return float(-c_response / (config.love_normalization * c_tidal))

        # Finite-frequency matching in the spherical-Bessel basis.
        design = build_bessel_matrix(r_fit, omega, config.harmonic_index)
        coeffs = lstsq(design, h_complex, lapack_driver="gelsy")[0]
        a_coeff, b_coeff = coeffs
        alpha_complex = 45.0 * (b_coeff / a_coeff) / omega**5
        return float(np.real(alpha_complex) / config.love_normalization)

    except Exception:
        if return_nan_on_failure:
            return float("nan")
        raise


def scan_length_ratios(
    ell_ratios: Iterable[float],
    omegas: Iterable[float],
    config: SolverConfig | None = None,
) -> list[LoveNumberResult]:
    """Compute ``alpha(ell_b/ell_ext)`` for several frequencies."""

    config = config or SolverConfig()
    ell_ext = extremal_length(config.mass)
    output: list[LoveNumberResult] = []

    for omega in omegas:
        for ell_ratio in ell_ratios:
            ell_b = float(ell_ratio) * ell_ext
            try:
                alpha = solve_dynamic_love_number(ell_b, float(omega), config)
                output.append(LoveNumberResult(float(ell_ratio), ell_b, float(omega), alpha))
            except Exception as exc:  # keep scans reproducible even when a point fails
                output.append(
                    LoveNumberResult(
                        float(ell_ratio), ell_b, float(omega), float("nan"), False, str(exc)
                    )
                )
    return output


def scan_frequencies(
    ell_ratios: Iterable[float],
    omegas: Iterable[float],
    config: SolverConfig | None = None,
) -> list[LoveNumberResult]:
    """Compute ``alpha(omega)`` for fixed values of ``ell_b/ell_ext``."""

    return scan_length_ratios(ell_ratios=ell_ratios, omegas=omegas, config=config)


def results_to_array(results: list[LoveNumberResult]) -> FloatArray:
    """Convert a list of results to a numeric array for saving."""

    return np.array(
        [[r.ell_ratio, r.ell_b, r.omega, r.alpha, float(r.success)] for r in results],
        dtype=float,
    )


def save_results_csv(results: list[LoveNumberResult], output_path: str | Path) -> None:
    """Save scan results as a CSV file."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    header = "ell_ratio,ell_b,omega,alpha,success"
    np.savetxt(output, results_to_array(results), delimiter=",", header=header, comments="")


def _valid_points(results: list[LoveNumberResult], omega: float | None = None, ell_ratio: float | None = None):
    selected = [r for r in results if r.success and np.isfinite(r.alpha)]
    if omega is not None:
        selected = [r for r in selected if np.isclose(r.omega, omega)]
    if ell_ratio is not None:
        selected = [r for r in selected if np.isclose(r.ell_ratio, ell_ratio)]
    return selected


def plot_love_vs_length(
    results: list[LoveNumberResult],
    output_path: str | Path,
    *,
    config: SolverConfig | None = None,
    analytic_coefficient: float = 10.0,
    smooth: bool = True,
) -> None:
    """Plot the dynamic Love number against ``ell_b / ell_ext``."""

    import matplotlib.pyplot as plt

    config = config or SolverConfig()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9.0, 7.0))
    omegas = sorted({r.omega for r in results})

    for omega in omegas:
        points = sorted(_valid_points(results, omega=omega), key=lambda r: r.ell_ratio)
        if not points:
            continue
        x = np.array([p.ell_ratio for p in points])
        y = np.array([p.alpha for p in points])
        if smooth and len(x) > 3:
            x_smooth = np.linspace(x.min(), x.max(), 300)
            y_smooth = UnivariateSpline(x, y, s=0)(x_smooth)
            ax.plot(x_smooth, y_smooth, linewidth=2.0, label=rf"$\omega M={omega:g}$")
        else:
            ax.plot(x, y, "-o", linewidth=2.0, label=rf"$\omega M={omega:g}$")

    ell_ext = extremal_length(config.mass)
    all_ratios = np.array([r.ell_ratio for r in results])
    if all_ratios.size:
        x_analytic = np.linspace(np.nanmin(all_ratios), np.nanmax(all_ratios), 600)
        ell_phys = x_analytic * ell_ext
        alpha_analytic = analytic_coefficient * config.mass * ell_phys**4
        ax.plot(
            x_analytic,
            alpha_analytic,
            "k--",
            linewidth=2.0,
            label=rf"${analytic_coefficient:g}\,M\ell_B^4$",
        )

    ax.set_xlabel(r"$\ell_B/\ell_{\rm ext}$", fontsize=18)
    ax.set_ylabel(r"Love number $\alpha$", fontsize=18)
    ax.set_title("Dynamic quadrupolar Love number: Bardeen geometry", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=13)
    ax.tick_params(axis="both", which="major", labelsize=13)
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def plot_frequency_response(
    results: list[LoveNumberResult],
    output_path: str | Path,
    *,
    smooth: bool = True,
) -> None:
    """Plot ``alpha(omega)`` for fixed ``ell_b / ell_ext`` values."""

    import matplotlib.pyplot as plt

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    ell_ratios = sorted({r.ell_ratio for r in results})

    for ell_ratio in ell_ratios:
        points = sorted(_valid_points(results, ell_ratio=ell_ratio), key=lambda r: r.omega)
        if not points:
            continue
        x = np.array([p.omega for p in points])
        y = np.array([p.alpha for p in points])
        if smooth and len(x) > 3:
            x_smooth = np.linspace(x.min(), x.max(), 400)
            y_smooth = UnivariateSpline(x, y, s=0)(x_smooth)
            ax.plot(
                x_smooth,
                y_smooth,
                linewidth=2.0,
                label=rf"$\ell_B/\ell_{{\rm ext}}={ell_ratio:.2f}$",
            )
        else:
            ax.plot(
                x,
                y,
                "-o",
                linewidth=2.0,
                label=rf"$\ell_B/\ell_{{\rm ext}}={ell_ratio:.2f}$",
            )

    ax.set_xlabel(r"Frequency $\omega M$", fontsize=18)
    ax.set_ylabel(r"Love number $\alpha(\omega)$", fontsize=18)
    ax.set_title(r"Dynamic Love number versus frequency", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=13)
    ax.tick_params(axis="both", which="major", labelsize=13)
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Command-line interface
# -----------------------------------------------------------------------------

def _parse_float_list(value: str) -> list[float]:
    """Parse comma-separated floats or ``linspace:start,stop,num``."""

    value = value.strip()
    if value.startswith("linspace:"):
        payload = value.removeprefix("linspace:")
        start, stop, num = payload.split(",")
        return np.linspace(float(start), float(stop), int(num)).tolist()
    return [float(item) for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute dynamic quadrupolar Love numbers for the Bardeen geometry."
    )
    parser.add_argument(
        "--mode",
        choices=("length", "frequency", "all"),
        default="all",
        help="Scan length ratios, scan frequencies, or run both scans.",
    )
    parser.add_argument("--mass", type=float, default=1.0, help="Black-hole mass M.")
    parser.add_argument(
        "--ell-ratios",
        default="linspace:0.01,0.99,10",
        help="Comma-separated ell_B/ell_ext values or linspace:start,stop,num.",
    )
    parser.add_argument(
        "--omegas",
        default="0.0001,0.01,0.03",
        help="Comma-separated omega values or linspace:start,stop,num for the length scan.",
    )
    parser.add_argument(
        "--freq-ell-ratios",
        default="0.1,0.3,0.5,0.9",
        help="Fixed ell_B/ell_ext values for the frequency scan.",
    )
    parser.add_argument(
        "--freq-omegas",
        default="linspace:0.1,0.5,20",
        help="Omega grid for the frequency scan.",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory for CSV outputs.",
    )
    parser.add_argument(
        "--figure-dir",
        default="figures",
        help="Directory for figure outputs.",
    )
    parser.add_argument("--tol", type=float, default=1.0e-6, help="solve_bvp tolerance.")
    parser.add_argument("--max-nodes", type=int, default=50_000, help="Maximum BVP nodes.")
    parser.add_argument(
        "--mesh-points", type=int, default=2_000, help="Number of initial radial mesh points."
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    config = SolverConfig(
        mass=args.mass,
        bvp_tolerance=args.tol,
        max_nodes=args.max_nodes,
        initial_mesh_points=args.mesh_points,
    )

    output_dir = Path(args.output_dir)
    figure_dir = Path(args.figure_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in {"length", "all"}:
        ell_ratios = _parse_float_list(args.ell_ratios)
        omegas = _parse_float_list(args.omegas)
        print("Running length-ratio scan")
        print(f"ell_B/ell_ext = {ell_ratios}")
        print(f"omega M = {omegas}")

        length_results = scan_length_ratios(ell_ratios, omegas, config)
        save_results_csv(length_results, output_dir / "love_vs_length.csv")
        plot_love_vs_length(length_results, figure_dir / "love_vs_length.png", config=config)
        print(f"Saved {output_dir / 'love_vs_length.csv'}")
        print(f"Saved {figure_dir / 'love_vs_length.png'}")

    if args.mode in {"frequency", "all"}:
        ell_ratios = _parse_float_list(args.freq_ell_ratios)
        omegas = _parse_float_list(args.freq_omegas)
        print("Running frequency scan")
        print(f"ell_B/ell_ext = {ell_ratios}")
        print(f"omega M = [{omegas[0]}, ..., {omegas[-1]}] with {len(omegas)} points")

        frequency_results = scan_frequencies(ell_ratios, omegas, config)
        save_results_csv(frequency_results, output_dir / "love_vs_frequency.csv")
        plot_frequency_response(frequency_results, figure_dir / "love_vs_frequency.png")
        print(f"Saved {output_dir / 'love_vs_frequency.csv'}")
        print(f"Saved {figure_dir / 'love_vs_frequency.png'}")


if __name__ == "__main__":
    main()
