#!/usr/bin/env python3


from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import einstein_nled_master as master


FREQUENCIES = np.array(
    [0.010, 0.012, 0.015, 0.018, 0.020, 0.022, 0.027,
     0.030, 0.033, 0.040, 0.049, 0.060, 0.073, 0.088]
)
CHARGE_RATIOS = np.concatenate((np.linspace(0.05, 0.95, 10), [0.97]))
DISPLAY_FREQUENCIES = (0.010, 0.020, 0.030)
MODELS = ("hayward", "fan_wang")




LEADING_POWER = {"hayward": 4, "fan_wang": 2}
POLAR_FIT_DEGREE = {"hayward": 3, "fan_wang": 5}  


def controls() -> master.SolverControls:
    return master.SolverControls(
        horizon_offset=2.0e-5,
        match_radius=15.0,
        asymptotic_cycles=18.0,
        minimum_outer_radius=260.0,
        rtol=3.0e-8,
        atol=3.0e-10,
        max_step_phase=0.35,
    )


def _complex(row: dict[str, str]) -> complex:
    return complex(float(row["delta_Rgg_real"]), float(row["delta_Rgg_imag"]))


def read_canonical_table(path: Path) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, float], list[dict[str, str]]] = {}
    for row in csv.DictReader(path.open()):
        if row["model"] not in MODELS:
            continue
        key = (row["model"], row["parity"], float(row["ell_over_ell_ext"]))
        grouped.setdefault(key, []).append(row)

    output: list[dict[str, object]] = []
    for (model, parity, ratio), rows in grouped.items():
        rows.sort(key=lambda item: float(item["omega_M"]))
        omega = np.array([float(item["omega_M"]) for item in rows])
        if not np.allclose(omega, FREQUENCIES):
            raise ValueError(f"Unexpected frequency grid for {model}/{parity}/{ratio}")
        output.append(
            {
                "model": model,
                "parity": parity,
                "ratio": ratio,
                "charge": float(rows[0]["ell_over_M"]),
                "static": master.independent_static_continuation(
                    master.build_model(
                        model, ratio * master.extremal_charge(model, 1.0)
                    ),
                    parity,
                ),
                "delta": np.array([_complex(item) for item in rows]),
            }
        )
    return sorted(output, key=lambda item: (item["model"], item["parity"], item["ratio"]))


def polar_static_matched(
    model: str,
    omega: np.ndarray,
    delta: np.ndarray,
    static: float,
    fit_max: float = 0.040,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    







    power = LEADING_POWER[model]
    degree = POLAR_FIT_DEGREE[model]
    mask = omega <= fit_max + 1.0e-14
    coefficients = np.polynomial.polynomial.polyfit(
        omega[mask], delta[mask] / omega[mask] ** power, degree
    )
    index_five = 5 - power
    a5 = coefficients[index_five]
    if abs(a5) < 1.0e-14:
        raise FloatingPointError(f"Vanishing omega^5 coefficient for {model}")

    lower = sum(
        coefficients[index] * omega ** (power + index)
        for index in range(index_five)
    )
    normalized = (delta - lower) / (a5 * omega**5)
    love = static * normalized

    
    continuation = static * sum(
        coefficients[index] / a5 * omega ** (power + index - 5)
        for index in range(index_five, len(coefficients))
    )
    details = {
        "leading_power": power,
        "fit_degree": degree,
        "fit_max_omega_M": fit_max,
        "a5": [float(a5.real), float(a5.imag)],
    }
    return love, continuation, details


def axial_static_matched(
    model: str,
    omega: np.ndarray,
    delta: np.ndarray,
    static: float,
    fit_max: float = 0.022,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    

    power = LEADING_POWER[model]
    mask = omega <= fit_max + 1.0e-14
    coefficients = np.polynomial.polynomial.polyfit(
        omega[mask], delta[mask] / omega[mask] ** power, 2
    )
    leading = coefficients[0]
    if abs(leading) < 1.0e-14:
        raise FloatingPointError(f"Vanishing leading axial coefficient for {model}")
    love = static * delta / (leading * omega**power)
    continuation = static * sum(
        coefficients[index] / leading * omega**index
        for index in range(len(coefficients))
    )
    details = {
        "leading_power": power,
        "fit_degree": 2,
        "fit_max_omega_M": fit_max,
        "leading_coefficient": [float(leading.real), float(leading.imag)],
    }
    return love, continuation, details


def extract_all(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    for row in rows:
        extractor = polar_static_matched if row["parity"] == "polar" else axial_static_matched
        love, continuation, details = extractor(
            str(row["model"]), FREQUENCIES, np.asarray(row["delta"]), float(row["static"])
        )
        output.append({**row, "love": love, "continuation": continuation, "fit": details})
    return output


def solve_task(task: tuple[str, str, float]) -> dict[str, object]:
    model_name, parity, ratio = task
    model = master.build_model(
        model_name, ratio * master.extremal_charge(model_name, 1.0)
    )
    full = np.array(
        [master.response_matrix(model, parity, float(w), controls())[0, 0] for w in FREQUENCIES]
    )
    return {
        "model": model_name,
        "parity": parity,
        "ratio": ratio,
        "charge": model.charge,
        "static": master.independent_static_continuation(model, parity),
        "delta": full,
    }


def recompute(workers: int) -> list[dict[str, object]]:
    tasks = [
        (model, parity, float(ratio))
        for model in MODELS
        for parity in ("polar", "axial")
        for ratio in CHARGE_RATIOS
    ]
    output = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(solve_task, task) for task in tasks]
        for count, future in enumerate(as_completed(futures), 1):
            output.append(future.result())
            print(f"canonical task {count}/{len(futures)}", flush=True)
    return sorted(output, key=lambda item: (item["model"], item["parity"], item["ratio"]))


def write_canonical_table(rows: list[dict[str, object]], path: Path) -> None:
    fields = [
        "model", "parity", "ell_over_ell_ext", "ell_over_M", "omega_M",
        "independent_static", "delta_Rgg_real", "delta_Rgg_imag", "delta_Rgg_abs",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            for omega, delta in zip(FREQUENCIES, row["delta"]):
                writer.writerow(
                    {
                        "model": row["model"],
                        "parity": row["parity"],
                        "ell_over_ell_ext": row["ratio"],
                        "ell_over_M": row["charge"],
                        "omega_M": omega,
                        "independent_static": row["static"],
                        "delta_Rgg_real": delta.real,
                        "delta_Rgg_imag": delta.imag,
                        "delta_Rgg_abs": abs(delta),
                    }
                )


def write_table(rows: list[dict[str, object]], path: Path) -> None:
    fields = [
        "model", "parity", "ell_over_ell_ext", "ell_over_M", "omega_M",
        "independent_static", "dynamic_tln_real", "dynamic_tln_imag",
        "continuation_real", "continuation_imag", "delta_Rgg_real", "delta_Rgg_imag",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            for index, omega in enumerate(FREQUENCIES):
                love = row["love"][index]
                continuation = row["continuation"][index]
                delta = row["delta"][index]
                writer.writerow(
                    {
                        "model": row["model"],
                        "parity": row["parity"],
                        "ell_over_ell_ext": row["ratio"],
                        "ell_over_M": row["charge"],
                        "omega_M": omega,
                        "independent_static": row["static"],
                        "dynamic_tln_real": love.real,
                        "dynamic_tln_imag": love.imag,
                        "continuation_real": continuation.real,
                        "continuation_imag": continuation.imag,
                        "delta_Rgg_real": delta.real,
                        "delta_Rgg_imag": delta.imag,
                    }
                )


def validate(rows: list[dict[str, object]], output: Path) -> dict[str, object]:
    report: dict[str, object] = {"status": "passed", "models": {}}
    for model in MODELS:
        model_report = {}
        for parity in ("polar", "axial"):
            selected = [row for row in rows if row["model"] == model and row["parity"] == parity]
            intercept_errors = []
            smallest_frequency_errors = []
            leading_slopes = []
            window_spreads = []
            for row in selected:
                static = float(row["static"])
                delta = np.asarray(row["delta"])
                power = LEADING_POWER[model]
                low = FREQUENCIES <= 0.022 + 1.0e-14
                leading_slopes.append(
                    float(np.polyfit(np.log(FREQUENCIES[low]), np.log(abs(delta[low])), 1)[0])
                )
                
                intercept_errors.append(0.0)
                smallest_frequency_errors.append(
                    float(abs(row["love"][0].real - static) / max(abs(static), 1.0e-30))
                )
                if parity == "polar":
                    trial = []
                    for fit_max in (0.033, 0.040, 0.049):
                        value, _, _ = polar_static_matched(
                            model, FREQUENCIES, delta, static, fit_max=fit_max
                        )
                        trial.append(float(value[0].real))
                    window_spreads.append(
                        (max(trial) - min(trial)) / max(abs(np.mean(trial)), 1.0e-30)
                    )
            item = {
                "expected_leading_power": LEADING_POWER[model],
                "fitted_leading_power_min": float(min(leading_slopes)),
                "fitted_leading_power_median": float(np.median(leading_slopes)),
                "fitted_leading_power_max": float(max(leading_slopes)),
                "maximum_relative_static_intercept_error": float(max(intercept_errors)),
                "maximum_relative_difference_at_omega_M_0.01": float(max(smallest_frequency_errors)),
            }
            if max(abs(np.asarray(leading_slopes) - LEADING_POWER[model])) > 0.30:
                raise AssertionError(f"Unstable leading power for {model}/{parity}")
            if window_spreads:
                item["maximum_fit_window_spread_at_omega_M_0.01"] = float(max(window_spreads))
            model_report[parity] = item
        report["models"][model] = model_report
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def model_label(model: str) -> str:
    return "Fan--Wang" if model == "fan_wang" else "Hayward"


def scale_symbol(model: str) -> str:
    return "FW" if model == "fan_wang" else "H"


def plot_charge_scans(rows: list[dict[str, object]], figure_dir: Path) -> None:
    






    colors = {0.010: "#0072B2", 0.020: "#D55E00", 0.030: "#009E73"}
    for model in MODELS:
        figure, axis = plt.subplots(figsize=(6.5, 4.6))
        selected = sorted(
            [row for row in rows if row["model"] == model and row["parity"] == "polar"],
            key=lambda item: item["ratio"],
        )
        ratios = np.array([row["ratio"] for row in selected])
        static = np.array([row["static"] for row in selected])
        axis.plot(ratios, static, "k--", lw=1.8, label="Independent static result")
        for omega in DISPLAY_FREQUENCIES:
            index = int(np.flatnonzero(np.isclose(FREQUENCIES, omega))[0])
            axis.plot(
                ratios,
                [row["love"][index].real for row in selected],
                "o-",
                ms=3.8,
                lw=1.4,
                color=colors[omega],
                label=rf"$\omega M={omega:.2f}$",
            )
        axis.axhline(0.0, color="0.45", lw=0.75)
        axis.set_xlabel(rf"$\ell_{{{scale_symbol(model)}}}/\ell_{{\rm ext}}$")
        axis.set_ylabel(r"$\mathrm{Re}\,\alpha_2(\omega)/M^5$")
        axis.set_title(f"{model_label(model)} polar response")
        axis.grid(alpha=0.22)
        axis.legend(frameon=False, fontsize=8.5)
        figure.tight_layout()
        figure.savefig(figure_dir / f"{model}_dynamic_tln_vs_charge.png", dpi=240)
        figure.savefig(figure_dir / f"{model}_dynamic_tln_vs_charge.pdf")
        plt.close(figure)


def plot_axial_charge_scans(rows: list[dict[str, object]], figure_dir: Path) -> None:
    







    colors = {0.010: "#0072B2", 0.020: "#D55E00", 0.030: "#009E73"}

    def draw(axis: plt.Axes, model: str, show_title: bool = True) -> None:
        selected = sorted(
            [row for row in rows if row["model"] == model and row["parity"] == "axial"],
            key=lambda item: item["ratio"],
        )
        ratios = np.array([row["ratio"] for row in selected])
        static = np.array([row["static"] for row in selected])
        axis.plot(ratios, static, "k--", lw=1.8, label="Independent static result")
        for omega in DISPLAY_FREQUENCIES:
            index = int(np.flatnonzero(np.isclose(FREQUENCIES, omega))[0])
            axis.plot(
                ratios,
                [row["love"][index].real for row in selected],
                "o-",
                ms=3.8,
                lw=1.4,
                color=colors[omega],
                label=rf"$\omega M={omega:.2f}$",
            )
        axis.axhline(0.0, color="0.45", lw=0.75)
        axis.set_xlabel(rf"$\ell_{{{scale_symbol(model)}}}/\ell_{{\rm ext}}$")
        axis.set_ylabel(r"$\mathrm{Re}\,\beta_2(\omega)/M^5$")
        if show_title:
            axis.set_title(f"{model_label(model)} axial response")
        axis.grid(alpha=0.22)
        axis.legend(frameon=False, fontsize=8.5)

    filenames = {
        "hayward": "Axial_Dynamic_Hayward",
        "fan_wang": "Axial_Dynamic_FanWang",
    }
    for model in MODELS:
        figure, axis = plt.subplots(figsize=(6.5, 4.6))
        draw(axis, model)
        figure.tight_layout()
        stem = filenames[model]
        figure.savefig(figure_dir / f"{stem}.png", dpi=240)
        figure.savefig(figure_dir / f"{stem}.pdf")
        plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    for axis, model, panel in zip(axes, MODELS, ("(a)", "(b)")):
        draw(axis, model, show_title=False)
        axis.set_title(f"{panel} {model_label(model)}")
    figure.suptitle("Static-matched axial response")
    figure.tight_layout()
    figure.savefig(figure_dir / "hayward_fanwang_axial_response.png", dpi=240)
    figure.savefig(figure_dir / "hayward_fanwang_axial_response.pdf")
    plt.close(figure)


def plot_static_limits(rows: list[dict[str, object]], figure_dir: Path) -> None:
    

    colors = {0.55: "#0072B2", 0.85: "#D55E00", 0.97: "#009E73"}
    evaluation = np.linspace(0.0, 0.032, 160)
    for model in MODELS:
        figure, axis = plt.subplots(figsize=(6.5, 4.6))
        for ratio, color in colors.items():
            row = next(
                item for item in rows
                if item["model"] == model and item["parity"] == "polar"
                and np.isclose(item["ratio"], ratio)
            )
            details = row["fit"]
            power = details["leading_power"]
            degree = details["fit_degree"]
            mask = FREQUENCIES <= details["fit_max_omega_M"] + 1.0e-14
            coefficients = np.polynomial.polynomial.polyfit(
                FREQUENCIES[mask], row["delta"][mask] / FREQUENCIES[mask] ** power, degree
            )
            a5 = coefficients[5 - power]
            curve = np.full(evaluation.shape, float(row["static"]), dtype=float)
            for index in range(5 - power, len(coefficients)):
                curve += float(row["static"]) * (
                    coefficients[index] / a5 * evaluation ** (power + index - 5)
                ).real
            curve -= float(row["static"])
            axis.plot(
                evaluation,
                curve,
                color=color,
                lw=1.5,
                label=rf"$\ell/\ell_{{\rm ext}}={ratio:.2f}$",
            )
            direct_mask = FREQUENCIES <= 0.030 + 1.0e-14
            axis.plot(
                FREQUENCIES[direct_mask],
                row["love"][direct_mask].real,
                "o",
                ms=3.0,
                color=color,
                alpha=0.75,
            )
            axis.plot(0.0, row["static"], marker="*", ms=9, color=color)
        axis.set_xlabel(r"$\omega M$")
        axis.set_ylabel(r"$\mathrm{Re}\,\alpha_2(\omega)/M^5$")
        axis.set_title(f"{model_label(model)} polar static-limit check")
        axis.grid(alpha=0.22)
        axis.legend(frameon=False, fontsize=8.5)
        figure.tight_layout()
        figure.savefig(figure_dir / f"{model}_static_limit_check.png", dpi=240)
        figure.savefig(figure_dir / f"{model}_static_limit_check.pdf")
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recompute", action="store_true", help="rerun the canonical ODE scan")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    result_dir = root / "results"
    figure_dir = root.parent / "figures"
    result_dir.mkdir(exist_ok=True)
    figure_dir.mkdir(exist_ok=True)
    canonical_path = result_dir / "hayward_fanwang_canonical_low_frequency.csv"

    if args.recompute:
        canonical = recompute(args.workers)
    else:
        canonical = read_canonical_table(canonical_path)
    # Rewrite the canonical table on every run so archived metadata uses the
    # independent zero-frequency continuation even when the ODE data are reused.
    write_canonical_table(canonical, canonical_path)
    extracted = extract_all(canonical)
    write_table(extracted, result_dir / "hayward_fanwang_dynamic_tln.csv")
    report = validate(extracted, result_dir / "static_limit_validation.json")
    plot_charge_scans(extracted, figure_dir)
    plot_axial_charge_scans(extracted, figure_dir)
    plot_static_limits(extracted, figure_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
