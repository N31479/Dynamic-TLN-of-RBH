#!/usr/bin/env python3
"""
Bardeen probe-response and shell-EFT comparison.

This module implements the numerical calculation used to compare two definitions of
Bardeen-sector probe tidal response:

1. Direct test-tensor response from the exterior Regge--Wheeler boundary-value problem.
2. Renormalized shell-EFT response obtained by finite-radius matching and subtraction
   of the Schwarzschild contribution.

The implementation follows the original research script but separates geometry,
solvers, extraction routines, and scan utilities so that the calculation is easier to
reuse and cite.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.integrate import solve_bvp, solve_ivp
from scipy.interpolate import UnivariateSpline
from scipy.linalg import lstsq
from scipy.optimize import brentq
from scipy.special import spherical_jn, spherical_yn


@dataclass(frozen=True)
class Constants:
    """Numerical and physical constants used in the Bardeen comparison scan."""

    mass: float = 1.0
    ell: int = 2
    rtol: float = 1.0e-11
    atol: float = 1.0e-13
    bvp_tol: float = 1.0e-6
    max_nodes: int = 50_000

    @property
    def r_s(self) -> float:
        """Schwarzschild radius for the chosen mass normalization."""
        return 2.0 * self.mass

    @property
    def ell_ext(self) -> float:
        """Extremal Bardeen regularization length for the chosen mass."""
        return 4.0 * self.mass / (3.0 * np.sqrt(3.0))


DEFAULTS = Constants()


@dataclass(frozen=True)
class ComparisonResult:
    """Container for one comparison datum."""

    omega: float
    chi: float
    ell_b: float
    direct_response: float
    shell_response: float

    @property
    def difference(self) -> float:
        """Direct minus shell-EFT response."""
        return self.direct_response - self.shell_response


# -----------------------------------------------------------------------------
# Bardeen geometry
# -----------------------------------------------------------------------------


def f_bardeen(r: np.ndarray | float, mass: float, ell_b: float) -> np.ndarray | float:
    """Metric function f(r) for the Bardeen geometry."""
    r_arr = np.asarray(r, dtype=float)
    value = 1.0 - 2.0 * mass * r_arr**2 / (r_arr**2 + ell_b**2) ** 1.5
    return float(value) if np.isscalar(r) else value


def dfdr_bardeen(r: np.ndarray | float, mass: float, ell_b: float) -> np.ndarray | float:
    """Radial derivative of the Bardeen metric function."""
    r_arr = np.asarray(r, dtype=float)
    rp2 = r_arr * r_arr + ell_b * ell_b
    value = -2.0 * mass * (2.0 * r_arr * rp2 ** (-1.5) - 3.0 * r_arr**3 * rp2 ** (-2.5))
    return float(value) if np.isscalar(r) else value


def mass_bardeen(r: np.ndarray | float, mass: float, ell_b: float) -> np.ndarray | float:
    """Effective mass profile m(r) for the Bardeen geometry."""
    r_arr = np.asarray(r, dtype=float)
    value = mass * r_arr**3 / (r_arr**2 + ell_b**2) ** 1.5
    return float(value) if np.isscalar(r) else value


def dm_bardeen(r: np.ndarray | float, mass: float, ell_b: float) -> np.ndarray | float:
    """Derivative of the Bardeen effective mass profile."""
    r_arr = np.asarray(r, dtype=float)
    value = 3.0 * mass * ell_b**2 * r_arr**2 / (r_arr**2 + ell_b**2) ** 2.5
    return float(value) if np.isscalar(r) else value


def get_outer_horizon(mass: float, ell_b: float, *, r_max: float | None = None, grid_size: int = 60_000) -> float:
    """Find the outer Bardeen horizon by bracketing roots of f(r)."""
    if r_max is None:
        r_max = 80.0 * mass

    grid = np.linspace(1.0e-6 * mass, r_max, grid_size)
    values = f_bardeen(grid, mass, ell_b)
    crossings = np.where(values[:-1] * values[1:] < 0.0)[0]

    if len(crossings) == 0:
        raise RuntimeError("No outer horizon found. Check that ell_b is below extremality.")

    index = crossings[-1]
    a, b = grid[index], grid[index + 1]
    return float(brentq(lambda x: f_bardeen(x, mass, ell_b), a, b))


def _bardeen_derivs_for_bvp(r: np.ndarray | float, mass: float, ell_b: float):
    """Return f, f', and placeholder NLED derivatives for compatibility with BVP code."""
    r_in = np.atleast_1d(r).astype(float)
    f = f_bardeen(r_in, mass, ell_b)
    fp = dfdr_bardeen(r_in, mass, ell_b)
    lf = np.zeros_like(r_in)
    lff = np.zeros_like(r_in)

    if np.isscalar(r):
        return float(f[0]), float(fp[0]), float(lf[0]), float(lff[0])
    return f, fp, lf, lff


def far_field_slice(r: np.ndarray) -> slice:
    """Slice selecting the far-field part of a numerical solution for matching."""
    return slice(int(0.6 * len(r)), -5)


def build_bessel_matrix(r_fit: np.ndarray, omega: float, ell: int = 2) -> np.ndarray:
    """Matrix with spherical Bessel basis columns j_l(omega r), y_l(omega r)."""
    z = omega * r_fit
    return np.vstack([spherical_jn(ell, z), spherical_yn(ell, z)]).T


# -----------------------------------------------------------------------------
# Direct test-tensor response
# -----------------------------------------------------------------------------


def solve_dynamic_exact_bc_tensor_bardeen(
    mass: float,
    ell_b: float,
    omega: float,
    *,
    constants: Constants = DEFAULTS,
) -> float:
    """
    Direct BVP extraction of the Bardeen test-tensor dynamical response.

    The response follows the normalization convention of the research script: in the
    low-frequency power-law regime the extracted ratio is divided by 1e9.
    """
    ratio = ell_b / constants.ell_ext
    r_h = get_outer_horizon(mass, ell_b)

    r_min = r_h + 0.5e-4
    r_max = 20.0 + 127.0 * ratio

    ell = constants.ell
    lam = ell * (ell + 1)

    def fun(r, y):
        f, fp, _, _ = _bardeen_derivs_for_bvp(r, mass, ell_b)
        f = np.maximum(f, 1.0e-7)

        m = mass_bardeen(r, mass, ell_b)
        dm_dr = dm_bardeen(r, mass, ell_b)
        potential = f * (lam / r**2 - 6.0 * m / r**3 + 2.0 * dm_dr / r**2)

        coeff = (omega**2 - potential) / f**2
        fp_over_f = fp / f

        re_psi, re_p, im_psi, im_p = y
        re_pp = -fp_over_f * re_p - coeff * re_psi
        im_pp = -fp_over_f * im_p - coeff * im_psi

        return np.vstack((re_p, re_pp, im_p, im_pp))

    def bc(ya, yb):
        f_min, _, _, _ = _bardeen_derivs_for_bvp(np.array([r_min]), mass, ell_b)
        kw = omega / f_min[0]

        bc1 = ya[1] - kw * ya[2]
        bc2 = ya[3] + kw * ya[0]

        z_inf = omega * r_max
        y_val = spherical_yn(ell, z_inf)
        yp_val = spherical_yn(ell, z_inf, derivative=True)
        target = 1.0 / (omega * r_max**2)

        bc3 = omega * yp_val * yb[0] - y_val * yb[1] - target
        bc4 = omega * yp_val * yb[2] - y_val * yb[3]

        return np.array([bc1, bc2, bc3, bc4])

    x_init = np.linspace(r_min, r_max, 2000)
    y_init = np.zeros((4, x_init.size))
    y_init[0] = spherical_jn(ell, omega * x_init)

    result = solve_bvp(fun, bc, x_init, y_init, tol=constants.bvp_tol, max_nodes=constants.max_nodes)
    if not result.success:
        return np.nan

    sl = far_field_slice(result.x)
    r_fit = result.x[sl]
    psi_complex = result.y[0][sl] + 1j * result.y[2][sl]
    z_max = omega * np.max(r_fit)

    if z_max < 0.1:
        source = r_fit**ell
        response = r_fit ** (-(ell + 1))
        matrix = np.vstack([source, response]).T
        coeffs = lstsq(matrix, np.real(psi_complex), lapack_driver="gelsy")[0]
        c_tidal, c_resp = coeffs
        if abs(c_tidal) < 1.0e-20:
            return 0.0
        return float(-c_resp / (1.0e9 * c_tidal))

    matrix = build_bessel_matrix(r_fit, omega, ell=ell)
    coeffs = lstsq(matrix, psi_complex, lapack_driver="gelsy")[0]
    a_cplx, b_cplx = coeffs
    alpha_cplx = 45.0 * (b_cplx / a_cplx) / omega**5
    return float(np.real(alpha_cplx))


# -----------------------------------------------------------------------------
# Shell-EFT response
# -----------------------------------------------------------------------------


def v_rw_scalar(r: float, mass: float, ell_b: float, ell: int) -> float:
    """Regge--Wheeler-type potential used in the shell-EFT probe calculation."""
    f = f_bardeen(r, mass, ell_b)
    fp = dfdr_bardeen(r, mass, ell_b)
    return float(f * (ell * (ell + 1) / r**2 + fp / r))


def rhs_rw_dynamic(r: float, y: Iterable[float], omega: float, mass: float, ell_b: float, ell: int):
    """First-order dynamic Regge--Wheeler system in r coordinates."""
    psi, p = y
    f = f_bardeen(r, mass, ell_b)
    potential = v_rw_scalar(r, mass, ell_b, ell)

    if abs(f) < 1.0e-16:
        f = np.sign(f) * 1.0e-16

    return [p / f, ((potential - omega**2) * psi) / f]


def solve_outside_rw_dynamic(
    omega: float,
    mass: float,
    ell_b: float,
    ell: int,
    r_shell: float,
    *,
    eps: float = 1.0e-6,
    constants: Constants = DEFAULTS,
) -> tuple[complex, complex]:
    """Integrate the exterior shell-EFT Regge--Wheeler problem to the shell radius."""
    r_h = get_outer_horizon(mass, ell_b)
    r0 = r_h * (1.0 + eps)

    psi0 = 1.0 + 0.0j
    p0 = -1j * omega * psi0

    sol_re = solve_ivp(
        fun=lambda r, y: rhs_rw_dynamic(r, y, omega, mass, ell_b, ell),
        t_span=(r0, r_shell),
        y0=[psi0.real, p0.real],
        method="DOP853",
        rtol=constants.rtol,
        atol=constants.atol,
    )
    sol_im = solve_ivp(
        fun=lambda r, y: rhs_rw_dynamic(r, y, omega, mass, ell_b, ell),
        t_span=(r0, r_shell),
        y0=[psi0.imag, p0.imag],
        method="DOP853",
        rtol=constants.rtol,
        atol=constants.atol,
    )

    if not sol_re.success or not sol_im.success:
        raise RuntimeError("Outside dynamic integration failed.")

    psi_r = sol_re.y[0, -1] + 1j * sol_im.y[0, -1]
    p_r = sol_re.y[1, -1] + 1j * sol_im.y[1, -1]
    dpsi_dr_r = p_r / f_bardeen(r_shell, mass, ell_b)
    return psi_r, dpsi_dr_r


def psi_inside_rw_dynamic(r_shell: float, omega: float, mass: float, ell_b: float, ell: int) -> tuple[float, float]:
    """Regular flat interior shell solution and radial derivative at the shell."""
    f_r = f_bardeen(r_shell, mass, ell_b)
    if f_r <= 0.0:
        raise ValueError("Shell must be outside the horizon: f(R)>0.")

    sqrt_f = np.sqrt(f_r)
    r_tilde = r_shell / sqrt_f
    omega_tilde = omega / sqrt_f
    x = omega_tilde * r_tilde

    j_l = spherical_jn(ell, x)
    dj_l_dx = spherical_jn(ell, x, derivative=True)

    psi_in = r_tilde * j_l
    dpsi_dr_tilde = j_l + r_tilde * dj_l_dx * omega_tilde
    dpsi_dr_in = dpsi_dr_tilde / sqrt_f
    return float(psi_in), float(dpsi_dr_in)


def f_shell_rw_dynamic(omega: float, mass: float, ell_b: float, ell: int, r_shell: float) -> complex:
    """Finite-radius shell response before running/finite-part fitting."""
    psi_out, dpsi_out = solve_outside_rw_dynamic(omega, mass, ell_b, ell, r_shell)
    psi_in, dpsi_in = psi_inside_rw_dynamic(r_shell, omega, mass, ell_b, ell)

    scale = psi_in / psi_out
    dpsi_out *= scale

    jump = dpsi_in - dpsi_out
    f_r = f_bardeen(r_shell, mass, ell_b)

    pref = 4.0 * np.pi * factorial(ell) * (2.0**ell) * r_shell ** (2 * ell + 2) / factorial(2 * ell + 1)
    pref *= np.sqrt(f_r) ** (2 * ell + 1)
    return pref * (jump / psi_in)


def make_r_list(chi: float, mass: float, n_r: int = 13) -> np.ndarray:
    """Shell-radius list used to fit the finite-radius shell response."""
    r_min = 6.0 * mass
    r_max = r_min + 160.0 * chi**2
    return np.linspace(r_min, r_max, n_r)


def fit_delta_running_dynamic(
    omega: float,
    ell_b: float,
    r_list: np.ndarray,
    *,
    constants: Constants = DEFAULTS,
    ridge: float = 1.0e-26,
) -> tuple[complex, complex]:
    """Fit the Bardeen-minus-Schwarzschild shell response to log and power terms."""
    mass = constants.mass
    ell = constants.ell
    log_x = np.log(constants.r_s / r_list)
    x1 = constants.r_s / r_list
    x2 = x1**2
    x3 = x1**3

    f_b = np.array([f_shell_rw_dynamic(omega, mass, ell_b, ell, r) for r in r_list], dtype=complex)
    f_s = np.array([f_shell_rw_dynamic(omega, mass, 0.0, ell, r) for r in r_list], dtype=complex)
    delta_f = f_b - f_s

    design = np.vstack([log_x, np.ones_like(log_x), x1, x2, x3]).T

    def ridge_solve(y):
        xtx = design.T @ design
        xty = design.T @ y
        return np.linalg.solve(xtx + ridge * np.eye(design.shape[1]), xty)

    params_re = ridge_solve(delta_f.real)
    params_im = ridge_solve(delta_f.imag)

    d_a = params_re[0] + 1j * params_im[0]
    d_b = params_re[1] + 1j * params_im[1]
    return d_a, d_b


def bar_f_from_dadb(d_a: complex, d_b: complex, *, normalization: float = 0.8 / 1.0e3) -> complex:
    """Convert fitted log/finite parts to the plotted shell response convention."""
    beta = 0.0  # retained explicitly for future scheme variations
    return normalization * (d_b + beta * d_a)


def solve_shell_barF_bardeen(mass: float, ell_b: float, omega: float, *, constants: Constants = DEFAULTS) -> float:
    """Compute the renormalized shell-EFT response in the script's normalization."""
    chi = ell_b / constants.ell_ext
    r_list = make_r_list(chi, mass, n_r=13)
    d_a, d_b = fit_delta_running_dynamic(omega, ell_b, r_list, constants=constants, ridge=1.0e-26)
    return float(np.real(bar_f_from_dadb(d_a, d_b)))


# -----------------------------------------------------------------------------
# Scan/plot utilities
# -----------------------------------------------------------------------------


def run_comparison(
    chi_values: Iterable[float],
    omega_values: Iterable[float],
    *,
    constants: Constants = DEFAULTS,
) -> pd.DataFrame:
    """Run the direct-vs-shell comparison scan and return a tidy DataFrame."""
    rows: list[ComparisonResult] = []
    mass = constants.mass

    for omega in omega_values:
        for chi in chi_values:
            ell_b = chi * constants.ell_ext
            try:
                direct = solve_dynamic_exact_bc_tensor_bardeen(mass, ell_b, omega, constants=constants)
            except Exception:
                direct = np.nan
            try:
                shell = solve_shell_barF_bardeen(mass, ell_b, omega, constants=constants)
            except Exception:
                shell = np.nan

            rows.append(ComparisonResult(omega, chi, ell_b, direct, shell))

    return pd.DataFrame(
        [
            {
                "omega": row.omega,
                "chi": row.chi,
                "ell_b": row.ell_b,
                "direct_response": row.direct_response,
                "shell_response": row.shell_response,
                "difference": row.difference,
            }
            for row in rows
        ]
    )


def save_comparison_plot(df: pd.DataFrame, output_path: str | Path) -> Path:
    """Save a publication-style comparison plot from a scan DataFrame."""
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9.2, 6.5))

    for omega, sub in df.groupby("omega"):
        sub = sub.sort_values("chi")
        chi = sub["chi"].to_numpy(dtype=float)
        direct = sub["direct_response"].to_numpy(dtype=float)
        shell = sub["shell_response"].to_numpy(dtype=float)

        mask_d = np.isfinite(direct)
        mask_s = np.isfinite(shell)

        if np.sum(mask_d) > 3:
            x_smooth = np.linspace(chi[mask_d].min(), chi[mask_d].max(), 300)
            y_smooth = UnivariateSpline(chi[mask_d], direct[mask_d], s=0)(x_smooth)
            plt.plot(x_smooth, y_smooth, lw=2.0, label=rf"Direct test tensor, $\omega={omega}$")
        else:
            plt.plot(chi, direct, "o-", label=rf"Direct, $\omega={omega}$")

        if np.sum(mask_s) > 3:
            x_smooth = np.linspace(chi[mask_s].min(), chi[mask_s].max(), 300)
            y_smooth = UnivariateSpline(chi[mask_s], shell[mask_s], s=0)(x_smooth)
            plt.plot(x_smooth, y_smooth, "--", lw=2.2, label=rf"Shell EFT, $\omega={omega}$")
        else:
            plt.plot(chi, shell, "s--", label=rf"Shell, $\omega={omega}$")

    plt.xlabel(r"$\ell_B/\ell_{\rm ext}$", fontsize=22)
    plt.ylabel(r"Response", fontsize=22)
    plt.title(r"Bardeen: direct test tensor vs shell EFT response", fontsize=16)
    plt.grid(True, alpha=0.35)
    plt.legend(fontsize=14)
    plt.tick_params(axis="both", which="major", labelsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


# -----------------------------------------------------------------------------
# Command-line interface
# -----------------------------------------------------------------------------

import argparse


def _parse_float_list(text: str) -> list[float]:
    """Parse a comma-separated list of floating-point values."""
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Run Bardeen direct test-tensor vs shell-EFT response comparison scans."
    )
    parser.add_argument("--mass", type=float, default=1.0, help="Black-hole mass normalization. Default: 1.0")
    parser.add_argument("--ell", type=int, default=2, help="Multipole number. Default: 2")
    parser.add_argument(
        "--chi-min",
        type=float,
        default=0.01,
        help="Minimum chi = ell_B/ell_ext for the scan. Default: 0.01",
    )
    parser.add_argument(
        "--chi-max",
        type=float,
        default=0.99,
        help="Maximum chi = ell_B/ell_ext for the scan. Default: 0.99",
    )
    parser.add_argument(
        "--n-chi",
        type=int,
        default=20,
        help="Number of chi values. Default: 20",
    )
    parser.add_argument(
        "--omega",
        type=str,
        default="1e-4,2e-4",
        help="Comma-separated frequency list. Default: 1e-4,2e-4",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for CSV and figure output. Default: outputs",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Save only the CSV file and skip figure generation.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    constants = Constants(mass=args.mass, ell=args.ell)
    chi_values = np.linspace(args.chi_min, args.chi_max, args.n_chi)
    omega_values = _parse_float_list(args.omega)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = run_comparison(chi_values, omega_values, constants=constants)
    csv_path = args.output_dir / "bardeen_shell_eft_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved data: {csv_path}")

    if not args.no_plot:
        fig_path = args.output_dir / "bardeen_shell_eft_comparison.png"
        save_comparison_plot(df, fig_path)
        print(f"Saved figure: {fig_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
