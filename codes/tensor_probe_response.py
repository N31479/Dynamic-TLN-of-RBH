#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


@dataclass(frozen=True)
class Controls:
    mass: float = 1.0
    multipole: int = 2
    horizon_offset: float = 2.0e-6
    rtol: float = 3.0e-11
    atol: float = 3.0e-13
    max_step: float = 0.15
    radial_order: int = 26
    omega_order: int = 3
    log_order: int = 5
    match_radii: tuple[float, ...] = (10.0, 12.0, 14.0, 16.0)


MODELS = ("bardeen", "hayward", "fan_wang")
DISPLAY_FREQUENCIES = np.array([0.002, 0.004, 0.006], dtype=float)
CHARGE_RATIOS = np.array([0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99], dtype=float)


def extremal_length(model: str, mass: float) -> float:
    if model in ("bardeen", "hayward"):
        return 4.0 * mass / (3.0 * np.sqrt(3.0))
    if model == "fan_wang":
        return 8.0 * mass / 27.0
    raise ValueError(model)


def mass_function(model: str, length: float, radius: np.ndarray | float, mass: float) -> np.ndarray:
    r = np.asarray(radius, dtype=float)
    if model == "bardeen":
        return mass * r**3 / (r**2 + length**2) ** 1.5
    if model == "hayward":
        return mass * r**3 / (r**3 + 2.0 * mass * length**2)
    if model == "fan_wang":
        return mass * r**3 / (r + length) ** 3
    if model == "schwarzschild":
        return np.full_like(r, mass, dtype=float)
    raise ValueError(model)


def mass_prime(model: str, length: float, radius: np.ndarray | float, mass: float) -> np.ndarray:
    r = np.asarray(radius, dtype=float)
    if model == "bardeen":
        return 3.0 * mass * length**2 * r**2 / (r**2 + length**2) ** 2.5
    if model == "hayward":
        d = r**3 + 2.0 * mass * length**2
        return 6.0 * mass**2 * length**2 * r**2 / d**2
    if model == "fan_wang":
        return 3.0 * mass * length * r**2 / (r + length) ** 4
    if model == "schwarzschild":
        return np.zeros_like(r, dtype=float)
    raise ValueError(model)


def metric(model: str, length: float, radius: np.ndarray | float, mass: float) -> np.ndarray:
    r = np.asarray(radius, dtype=float)
    return 1.0 - 2.0 * mass_function(model, length, r, mass) / r


def metric_prime(model: str, length: float, radius: np.ndarray | float, mass: float) -> np.ndarray:
    r = np.asarray(radius, dtype=float)
    m = mass_function(model, length, r, mass)
    mp = mass_prime(model, length, r, mass)
    return 2.0 * m / r**2 - 2.0 * mp / r


def potential_bracket(model: str, length: float, radius: np.ndarray | float, mass: float, multipole: int) -> np.ndarray:
    r = np.asarray(radius, dtype=float)
    m = mass_function(model, length, r, mass)
    mp = mass_prime(model, length, r, mass)
    return multipole * (multipole + 1.0) / r**2 - 6.0 * m / r**3 + 2.0 * mp / r**2


def outer_horizon(model: str, length: float, mass: float) -> float:
    if model == "schwarzschild":
        return 2.0 * mass
    grid = np.geomspace(1.0e-8 * mass, 10.0 * mass, 20000)
    values = metric(model, length, grid, mass)
    changes = np.where(values[:-1] * values[1:] < 0.0)[0]
    if changes.size == 0:
        raise RuntimeError(f"No outer horizon for {model} at length={length}")
    i = int(changes[-1])
    return float(brentq(lambda r: float(metric(model, length, r, mass)), grid[i], grid[i + 1]))


def conv(a: np.ndarray, b: np.ndarray, order: int) -> np.ndarray:
    out = np.zeros(order + 1, dtype=complex)
    for i, x in enumerate(a):
        if i > order:
            break
        for j, y in enumerate(b):
            if i + j > order:
                break
            out[i + j] += x * y
    return out


def f_series(model: str, length: float, mass: float, order: int) -> np.ndarray:
    f = np.zeros(order + 1, dtype=complex)
    f[0] = 1.0
    if model == "bardeen":
        coefficient = 1.0
        k = 0
        while 2 + 4 * k <= order:
            if k > 0:
                coefficient *= (-1.5 - (k - 1)) / k
            f[2 + 4 * k] += -2.0 * mass * coefficient * length ** (2 * k)
            k += 1
        return f
    if model == "hayward":
        k = 0
        while 2 + 6 * k <= order:
            f[2 + 6 * k] += -2.0 * mass * (-2.0 * mass * length**2) ** k
            k += 1
        return f
    if model == "fan_wang":
        k = 0
        while 2 + 2 * k <= order:
            coefficient = (-1.0) ** k * (k + 1.0) * (k + 2.0) / 2.0
            f[2 + 2 * k] += -2.0 * mass * coefficient * length**k
            k += 1
        return f
    if model == "schwarzschild":
        if order >= 2:
            f[2] = -2.0 * mass
        return f
    raise ValueError(model)


def derivative_r_series(a: np.ndarray, order: int) -> np.ndarray:
    out = np.zeros(order + 1, dtype=complex)
    for n, value in enumerate(a):
        if n + 2 <= order:
            out[n + 2] += -0.5 * n * value
    return out


def operator_series(model: str, length: float, controls: Controls, order: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    f = f_series(model, length, controls.mass, order + 8)
    fp = derivative_r_series(f, order + 8)
    m = np.zeros(order + 9, dtype=complex)
    for n in range(2, len(f)):
        if n - 2 < len(m):
            m[n - 2] += -0.5 * f[n]
    mp = derivative_r_series(m, order + 8)
    u = np.zeros(order + 9, dtype=complex)
    if len(u) > 4:
        u[4] += controls.multipole * (controls.multipole + 1.0)
    for n, value in enumerate(m):
        if n + 6 < len(u):
            u[n + 6] += -6.0 * value
    for n, value in enumerate(mp):
        if n + 4 < len(u):
            u[n + 4] += 2.0 * value
    n = order
    f2 = conv(f, f, n)
    ffp = conv(f, fp, n)
    fu = conv(f, u, n)
    return f2, ffp, fu


def dynamic_mode(model: str, length: float, exponent: int, controls: Controls) -> np.ndarray:
    radial_order = controls.radial_order
    omega_order = controls.omega_order
    log_order = controls.log_order
    f2, ffp, fu = operator_series(model, length, controls, radial_order + 4 * omega_order + 8)
    c = np.zeros((omega_order + 1, radial_order + 1, log_order + 1), dtype=complex)
    c[0, 0, 0] = 1.0
    for k in range(omega_order + 1):
        base = exponent - 4 * k
        for n in range(radial_order + 1):
            if k == 0 and n == 0:
                continue
            e = base + n
            known = np.zeros(log_order + 1, dtype=complex)
            for p, coefficient in enumerate(f2):
                mm = n - p
                if mm < 0 or mm > radial_order or (p == 0 and mm == n):
                    continue
                em = base + mm
                for j in range(log_order + 1):
                    known[j] += coefficient * 0.25 * em * (em + 2.0) * c[k, mm, j]
                    if j + 1 <= log_order:
                        known[j] += coefficient * 0.5 * (em + 1.0) * (j + 1) * c[k, mm, j + 1]
                    if j + 2 <= log_order:
                        known[j] += coefficient * 0.25 * (j + 2) * (j + 1) * c[k, mm, j + 2]
            for p, coefficient in enumerate(ffp):
                mm = n + 2 - p
                if mm < 0 or mm > radial_order:
                    continue
                em = base + mm
                for j in range(log_order + 1):
                    known[j] += -0.5 * coefficient * em * c[k, mm, j]
                    if j + 1 <= log_order:
                        known[j] += -0.5 * coefficient * (j + 1) * c[k, mm, j + 1]
            for p, coefficient in enumerate(fu):
                mm = n + 4 - p
                if mm < 0 or mm > radial_order or (p == 4 and mm == n):
                    continue
                known -= coefficient * c[k, mm]
            if k > 0:
                known += c[k - 1, n]
            a = 0.25 * e * (e + 2.0) - controls.multipole * (controls.multipole + 1.0)
            ap = 0.5 * (e + 1.0)
            block = np.zeros((log_order + 1, log_order + 1), dtype=complex)
            rhs = -known
            for j in range(log_order + 1):
                block[j, j] = a
                if j + 1 <= log_order:
                    block[j, j + 1] = ap * (j + 1)
                if j + 2 <= log_order:
                    block[j, j + 2] = 0.25 * (j + 2) * (j + 1)
            if abs(a) < 1.0e-12:
                constraint = np.zeros((1, log_order + 1), dtype=complex)
                constraint[0, 0] = 1.0
                solution = np.linalg.lstsq(np.vstack([block, constraint]), np.concatenate([rhs, [0.0]]), rcond=1.0e-12)[0]
                residual = np.linalg.norm(block @ solution - rhs) / (1.0 + np.linalg.norm(rhs))
                if residual > 5.0e-8:
                    raise RuntimeError(f"Insufficient logarithmic order for {model}, k={k}, n={n}")
            else:
                solution = np.linalg.solve(block, rhs)
            c[k, n] = solution
    return c


def evaluate_mode(c: np.ndarray, exponent: int, omega: float, radii: np.ndarray, mass: float) -> tuple[np.ndarray, np.ndarray]:
    rr = np.atleast_1d(np.asarray(radii, dtype=float))
    val = np.zeros(len(rr), dtype=complex)
    der = np.zeros(len(rr), dtype=complex)
    w2 = omega**2
    for ir, radius in enumerate(rr):
        x = radius**-0.5
        lx = np.log(x * np.sqrt(2.0 * mass))
        for k in range(c.shape[0]):
            base = exponent - 4 * k
            wk = w2**k
            for n in range(c.shape[1]):
                e = base + n
                poly = 0.0j
                dpoly = 0.0j
                for j in range(c.shape[2]):
                    poly += c[k, n, j] * lx**j
                    if j:
                        dpoly += j * c[k, n, j] * lx ** (j - 1)
                val[ir] += wk * x**e * poly
                der[ir] += -0.5 * wk * x ** (e + 2) * (e * poly + dpoly)
    return val, der


def horizon_solution(model: str, length: float, omega: float, controls: Controls) -> tuple[np.ndarray, np.ndarray]:
    horizon = outer_horizon(model, length, controls.mass)
    start = horizon + controls.horizon_offset
    fprime = float(metric_prime(model, length, horizon, controls.mass))
    uh = float(potential_bracket(model, length, horizon, controls.mass, controls.multipole))
    regular = uh / (fprime - 2.0j * omega)
    value = 1.0 + controls.horizon_offset * regular
    fstart = float(metric(model, length, start, controls.mass))
    derivative = regular if omega == 0.0 else -1.0j * omega * value / fstart + regular

    def rhs(radius: float, state: np.ndarray) -> np.ndarray:
        f = float(metric(model, length, radius, controls.mass))
        fp = float(metric_prime(model, length, radius, controls.mass))
        u = float(potential_bracket(model, length, radius, controls.mass, controls.multipole))
        return np.array([state[1], -(fp / f) * state[1] - (omega**2 - f * u) * state[0] / f**2], dtype=complex)

    end = max(controls.match_radii)
    solution = solve_ivp(rhs, (start, end), np.array([value, derivative], dtype=complex), method="DOP853", rtol=controls.rtol, atol=controls.atol, max_step=controls.max_step, dense_output=True)
    if not solution.success:
        raise RuntimeError(solution.message)
    radii = np.asarray(controls.match_radii, dtype=float)
    values = solution.sol(radii)
    return values[0], values[1]


def extrapolate_ratio(radii: np.ndarray, ratios: np.ndarray, omega: float) -> tuple[complex, float]:
    if omega == 0.0:
        design = np.column_stack([np.ones_like(radii), radii**-2])
    else:
        design = np.column_stack([np.ones_like(radii), (omega * radii) ** 6, radii**-2])
    coeff_real = np.linalg.lstsq(design, ratios.real, rcond=None)[0]
    coeff_imag = np.linalg.lstsq(design, ratios.imag, rcond=None)[0]
    fit = design @ coeff_real + 1.0j * (design @ coeff_imag)
    residual = float(np.max(np.abs(fit - ratios)))
    return complex(coeff_real[0], coeff_imag[0]), residual


def direct_ratio(model: str, length: float, omega: float, controls: Controls, source_mode: np.ndarray, response_mode: np.ndarray) -> tuple[complex, float, list[complex]]:
    radii = np.asarray(controls.match_radii, dtype=float)
    field, derivative = horizon_solution(model, length, omega, controls)
    source, source_derivative = evaluate_mode(source_mode, -6, omega, radii, controls.mass)
    response, response_derivative = evaluate_mode(response_mode, 4, omega, radii, controls.mass)
    ratios = []
    for i in range(len(radii)):
        matrix = np.array([[source[i], response[i]], [source_derivative[i], response_derivative[i]]], dtype=complex)
        coefficients = np.linalg.solve(matrix, np.array([field[i], derivative[i]], dtype=complex))
        ratios.append(complex(coefficients[1] / coefficients[0]))
    ratio, residual = extrapolate_ratio(radii, np.asarray(ratios, dtype=complex), omega)
    return ratio, residual, ratios


def compute_scan(controls: Controls) -> tuple[list[dict[str, float | str]], dict[str, object]]:
    rows: list[dict[str, float | str]] = []
    summary: dict[str, object] = {"definition": "Independent static tensor-probe source-response ratio and finite-frequency intermediate-zone continuation in the log(r/2M) asymptotic convention.", "controls": asdict(controls), "models": {}}
    for model in MODELS:
        lext = extremal_length(model, controls.mass)
        model_summary = {}
        for ratio in CHARGE_RATIOS:
            length = ratio * lext
            source_mode = dynamic_mode(model, length, -6, controls)
            response_mode = dynamic_mode(model, length, 4, controls)
            c0, static_residual, _ = direct_ratio(model, length, 0.0, controls, source_mode, response_mode)
            static_value = c0.real / controls.mass**5
            values = {}
            for omega in DISPLAY_FREQUENCIES:
                cw, residual, local = direct_ratio(model, length, float(omega), controls, source_mode, response_mode)
                dynamic = static_value + (cw.real - c0.real) / controls.mass**5
                rows.append({"model": model, "ell_over_ell_ext": float(ratio), "ell_over_M": float(length / controls.mass), "omega_M": float(omega * controls.mass), "static_response_over_M5": float(static_value), "direct_ratio_zero_real": float(c0.real), "direct_ratio_real": float(cw.real), "direct_ratio_imag": float(cw.imag), "dynamic_response_over_M5": float(dynamic), "finite_frequency_correction_over_M5": float((cw.real - c0.real) / controls.mass**5), "static_match_residual": float(static_residual), "dynamic_match_residual": float(residual), "maximum_omega_r": float(omega * max(controls.match_radii))})
                values[f"{omega:.6f}"] = {"direct_ratio": [cw.real, cw.imag], "dynamic_response_over_M5": dynamic, "match_residual": residual, "local_ratios": [[z.real, z.imag] for z in local]}
            model_summary[f"{ratio:.6f}"] = {"static_direct_ratio": [c0.real, c0.imag], "static_response_over_M5": static_value, "static_match_residual": static_residual, "frequencies": values}
        if model == "fan_wang":
            check_ratio = 1.0e-3
            check_length = check_ratio * lext
            check_source = dynamic_mode(model, check_length, -6, controls)
            check_response = dynamic_mode(model, check_length, 4, controls)
            check_value, check_residual, _ = direct_ratio(model, check_length, 0.0, controls, check_source, check_response)
            measured_coefficient = check_value.real / (controls.mass**3 * check_length**2)
            expected_coefficient = 128.0 / 25.0
            relative_error = abs(measured_coefficient / expected_coefficient - 1.0)
            if relative_error > 5.0e-3:
                raise RuntimeError("Fan-Wang static tensor response failed the small-length analytic check")
            model_summary["static_small_length_check"] = {
                "ell_over_ell_ext": check_ratio,
                "measured_coefficient_of_M3_ell2": measured_coefficient,
                "expected_coefficient_of_M3_ell2": expected_coefficient,
                "relative_error": relative_error,
                "match_residual": check_residual,
            }
        summary["models"][model] = model_summary
    return rows, summary


def write_csv(rows: list[dict[str, float | str]], path: Path) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_model(rows: list[dict[str, float | str]], model: str, output_dir: Path) -> None:
    selected = [row for row in rows if row["model"] == model]
    figure, axis = plt.subplots(figsize=(6.4, 4.9))
    static_points = sorted({float(row["ell_over_ell_ext"]): float(row["static_response_over_M5"]) for row in selected}.items())
    axis.plot([x for x, _ in static_points], [y for _, y in static_points], "--", linewidth=2.0, label=r"Static ($M\omega=0$)")
    markers = ("o", "s", "^")
    for marker, omega in zip(markers, DISPLAY_FREQUENCIES):
        points = sorted([row for row in selected if abs(float(row["omega_M"]) - omega) < 1.0e-12], key=lambda item: float(item["ell_over_ell_ext"]))
        axis.plot([float(row["ell_over_ell_ext"]) for row in points], [float(row["dynamic_response_over_M5"]) for row in points], marker=marker, markersize=4.2, linewidth=1.65, label=rf"$M\omega={omega:.3f}$")
    labels = {"bardeen": "Bardeen", "hayward": "Hayward", "fan_wang": "Fan-Wang"}
    subscripts = {"bardeen": "B", "hayward": "H", "fan_wang": "FW"}
    label = labels[model]
    subscript = subscripts[model]
    axis.set_xlabel(rf"$\ell_{subscript}/\ell_{{\rm ext}}$")
    axis.set_ylabel(r"$\alpha_T(\omega)/M^5$")
    axis.set_xlim(0.0, 1.0)
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, loc="best")
    axis.set_title(f"Tensor-probe response on the {label} background")
    inset = inset_axes(axis, width="48%", height="42%", loc="center left", bbox_to_anchor=(0.10, -0.02, 1, 1), bbox_transform=axis.transAxes, borderpad=0.8)
    for marker, omega in zip(markers, DISPLAY_FREQUENCIES):
        points = sorted([row for row in selected if abs(float(row["omega_M"]) - omega) < 1.0e-12], key=lambda item: float(item["ell_over_ell_ext"]))
        inset.plot([float(row["ell_over_ell_ext"]) for row in points], [1.0e4 * float(row["finite_frequency_correction_over_M5"]) for row in points], marker=marker, markersize=2.8, linewidth=1.0)
    inset.set_xlabel(r"$\ell/\ell_{\rm ext}$", fontsize=7)
    inset.set_ylabel(r"$10^4[\alpha_T(\omega)-\alpha_T(0)]/M^5$", fontsize=7)
    inset.tick_params(labelsize=7)
    inset.grid(alpha=0.2)
    figure.subplots_adjust(left=0.12, right=0.98, bottom=0.14, top=0.90)
    stem = f"Tensor_Field_{label}"
    figure.savefig(output_dir / f"{stem}.png", dpi=300)
    figure.savefig(output_dir / f"{stem}.pdf")
    plt.close(figure)


def plot_combined(rows: list[dict[str, float | str]], output_dir: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15.8, 4.35), sharex=True)
    markers = ("o", "s", "^")
    labels = {"bardeen": "Bardeen", "hayward": "Hayward", "fan_wang": "Fan-Wang"}
    subscripts = {"bardeen": "B", "hayward": "H", "fan_wang": "FW"}
    for axis, model in zip(axes, MODELS):
        selected = [row for row in rows if row["model"] == model]
        static_points = sorted({float(row["ell_over_ell_ext"]): float(row["static_response_over_M5"]) for row in selected}.items())
        axis.plot([x for x, _ in static_points], [y for _, y in static_points], "--", linewidth=2.0, label="Static")
        for marker, omega in zip(markers, DISPLAY_FREQUENCIES):
            points = sorted([row for row in selected if abs(float(row["omega_M"]) - omega) < 1.0e-12], key=lambda item: float(item["ell_over_ell_ext"]))
            axis.plot([float(row["ell_over_ell_ext"]) for row in points], [float(row["dynamic_response_over_M5"]) for row in points], marker=marker, markersize=3.7, linewidth=1.5, label=rf"$M\omega={omega:.3f}$")
        label = labels[model]
        subscript = subscripts[model]
        axis.set_title(label)
        axis.set_xlabel(rf"$\ell_{subscript}/\ell_{{\rm ext}}$")
        axis.set_xlim(0.0, 1.0)
        axis.grid(alpha=0.25)
        inset = inset_axes(axis, width="48%", height="42%", loc="center left", bbox_to_anchor=(0.10, -0.02, 1, 1), bbox_transform=axis.transAxes, borderpad=0.8)
        for marker, omega in zip(markers, DISPLAY_FREQUENCIES):
            points = sorted([row for row in selected if abs(float(row["omega_M"]) - omega) < 1.0e-12], key=lambda item: float(item["ell_over_ell_ext"]))
            inset.plot([float(row["ell_over_ell_ext"]) for row in points], [1.0e4 * float(row["finite_frequency_correction_over_M5"]) for row in points], marker=marker, markersize=2.5, linewidth=0.95)
        inset.set_xlabel(r"$\ell/\ell_{\rm ext}$", fontsize=7)
        inset.set_ylabel(r"$10^4\Delta\alpha_T/M^5$", fontsize=7)
        inset.tick_params(labelsize=7)
        inset.grid(alpha=0.2)
    axes[0].set_ylabel(r"$\alpha_T(\omega)/M^5$")
    axes[-1].legend(frameon=False, loc="best")
    figure.subplots_adjust(left=0.065, right=0.99, bottom=0.14, top=0.90, wspace=0.24)
    figure.savefig(output_dir / "tensor_probe_all_models.png", dpi=300)
    figure.savefig(output_dir / "tensor_probe_all_models.pdf")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("."))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.output_root.resolve()
    result_dir = root / "codes" / "results"
    figure_dir = root / "figures"
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    controls = Controls()
    rows, summary = compute_scan(controls)
    write_csv(rows, result_dir / "tensor_probe_response.csv")
    with (result_dir / "tensor_probe_validation.json").open("w") as stream:
        json.dump(summary, stream, indent=2)
    old = result_dir / "tensor_probe_schwarzschild_response.json"
    if old.exists():
        old.unlink()
    for model in MODELS:
        plot_model(rows, model, figure_dir)
    plot_combined(rows, figure_dir)
    print(json.dumps({"definition": summary["definition"], "rows": len(rows), "figures": ["Tensor_Field_Bardeen", "Tensor_Field_Hayward", "Tensor_Field_Fan-Wang", "tensor_probe_all_models"]}, indent=2))


if __name__ == "__main__":
    main()
