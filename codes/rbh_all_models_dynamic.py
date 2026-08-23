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


LOW_FREQUENCIES = np.array(
    [0.010, 0.012, 0.015, 0.018, 0.020, 0.022, 0.027,
     0.030, 0.033, 0.040, 0.049, 0.060, 0.073, 0.088]
)
BROAD_FREQUENCIES = np.linspace(0.10, 0.80, 57)
CHARGE_RATIOS = np.concatenate((np.linspace(0.05, 0.95, 10), [0.97]))
BROAD_RATIOS = np.array([0.60, 0.90, 0.97])
DISPLAY_FREQUENCIES = (0.010, 0.020, 0.030)
MODELS = ("bardeen", "hayward", "fan_wang")


def static_reference(model, parity: str) -> float:
    return master.independent_static_continuation(model, parity)


def controls(strict: bool = False) -> master.SolverControls:
    return master.SolverControls(
        horizon_offset=1.0e-5 if strict else 2.0e-5,
        match_radius=18.0 if strict else 15.0,
        asymptotic_cycles=28.0 if strict else 18.0,
        minimum_outer_radius=400.0 if strict else 260.0,
        rtol=8.0e-9 if strict else 3.0e-8,
        atol=8.0e-11 if strict else 3.0e-10,
        max_step_phase=0.25 if strict else 0.35,
    )


def complex_counterterms(omega: np.ndarray, response: np.ndarray) -> dict[str, complex]:
    mask = omega <= 0.088 + 1.0e-14
    x = omega[mask] / 0.05
    raw = -45.0 * response[mask] / omega[mask] ** 5
    design = np.column_stack((x**-2, x**-1, np.log(x), np.ones_like(x), x))
    fitted, _, _, _ = np.linalg.lstsq(design, raw, rcond=None)
    return dict(zip(("x^-2", "x^-1", "log(x)", "1", "x"), fitted))


def leading_power(model_name: str) -> int:
    
    return {"bardeen": 3, "hayward": 4, "fan_wang": 2}[model_name]


def axial_leading_power(model_name: str) -> int:
    return leading_power(model_name)


def polar_response(
    model_name: str, omega: np.ndarray, response: np.ndarray, static: float
) -> np.ndarray:
    
    if model_name == "bardeen":
        counterterms = complex_counterterms(omega, response)
        x = omega / 0.05
        background = (
            counterterms["x^-2"] * x**-2
            + counterterms["x^-1"] * x**-1
            + counterterms["log(x)"] * np.log(x)
            + counterterms["1"]
            + counterterms["x"] * x
        )
        return static - 45.0 * response / omega**5 - background

    power = leading_power(model_name)
    degree = {"hayward": 3, "fan_wang": 5}[model_name]
    low = omega <= 0.040 + 1.0e-14
    coefficients = np.polynomial.polynomial.polyfit(
        omega[low], response[low] / omega[low] ** power, degree
    )
    index_five = 5 - power
    a5 = coefficients[index_five]
    if abs(a5) < 1.0e-14:
        raise FloatingPointError(f"vanishing omega^5 coefficient for {model_name}")
    lower = sum(
        coefficients[index] * omega ** (power + index)
        for index in range(index_five)
    )
    return static * (response - lower) / (a5 * omega**5)


def static_normalized_axial(
    model_name: str, omega: np.ndarray, response: np.ndarray, static: float
) -> np.ndarray:
    





    power = axial_leading_power(model_name)
    low = omega <= 0.022 + 1.0e-14
    leading = np.polynomial.polynomial.polyfit(
        omega[low], response[low] / omega[low] ** power, 2
    )[0]
    return static * response / (leading * omega**power)


def solve_low_task(task: tuple[str, str, float]) -> dict:
    model_name, parity, ratio = task
    model = master.build_model(
        model_name, ratio * master.extremal_charge(model_name, 1.0)
    )
    full = np.array(
        [
            master.response_matrix(model, parity, float(omega), controls())[0, 0]
            for omega in LOW_FREQUENCIES
        ]
    )
    delta = full
    static = static_reference(model, parity)
    counterterms = complex_counterterms(LOW_FREQUENCIES, delta) if model_name == "bardeen" else {}
    finite = (
        polar_response(model_name, LOW_FREQUENCIES, delta, static)
        if parity == "polar"
        else static_normalized_axial(model_name, LOW_FREQUENCIES, delta, static)
    )
    axial_normalized = (
        static_normalized_axial(model_name, LOW_FREQUENCIES, delta, static)
        if parity == "axial" else np.full_like(delta, np.nan)
    )
    return {
        "model": model_name,
        "parity": parity,
        "ratio": ratio,
        "charge": model.charge,
        "static": static,
        "delta": [[value.real, value.imag] for value in delta],
        "finite": [[value.real, value.imag] for value in finite],
        "axial_normalized": [[value.real, value.imag] for value in axial_normalized],
        "counterterms": {
            name: [value.real, value.imag] for name, value in counterterms.items()
        },
    }


def solve_broad_task(task: tuple[str, float, bool]) -> dict:
    model_name, ratio, strict = task
    model = master.build_model(
        model_name, ratio * master.extremal_charge(model_name, 1.0)
    )
    response = np.array(
        [
            master.response_matrix(model, "polar", float(omega), controls(strict))[0, 0]
            for omega in BROAD_FREQUENCIES
        ]
    )
    return {
        "model": model_name,
        "ratio": ratio,
        "strict": strict,
        "delta": [[value.real, value.imag] for value in response],
    }


def write_low_csv(results: list[dict], path: Path) -> None:
    rows = []
    for result in results:
        for index, omega in enumerate(LOW_FREQUENCIES):
            rows.append(
                {
                    "model": result["model"],
                    "parity": result["parity"],
                    "ell_over_ell_ext": result["ratio"],
                    "ell_over_M": result["charge"],
                    "omega_M": omega,
                    "static_reference": result["static"],
                    "finite_real": result["finite"][index][0],
                    "finite_imag": result["finite"][index][1],
                    "axial_leading_power": (
                        axial_leading_power(result["model"])
                        if result["parity"] == "axial" else ""
                    ),
                    "axial_normalized_real": result["axial_normalized"][index][0],
                    "axial_normalized_imag": result["axial_normalized"][index][1],
                    "delta_Rgg_real": result["delta"][index][0],
                    "delta_Rgg_imag": result["delta"][index][1],
                    "delta_Rgg_abs": abs(complex(*result["delta"][index])),
                }
            )
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_broad_csv(results: list[dict], path: Path) -> None:
    rows = []
    for result in results:
        for omega, value in zip(BROAD_FREQUENCIES, result["delta"]):
            response = complex(*value)
            rows.append(
                {
                    "model": result["model"],
                    "ell_over_ell_ext": result["ratio"],
                    "omega_M": omega,
                    "strict_controls": int(result["strict"]),
                    "delta_Rgg_real": response.real,
                    "delta_Rgg_imag": response.imag,
                    "delta_Rgg_abs": abs(response),
                }
            )
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_low_csv(path: Path) -> list[dict]:
    grouped: dict[tuple[str, str, float], list[dict[str, str]]] = {}
    for row in csv.DictReader(path.open()):
        key = (row["model"], row["parity"], float(row["ell_over_ell_ext"]))
        grouped.setdefault(key, []).append(row)
    output = []
    for (model_name, parity, ratio), rows in grouped.items():
        rows.sort(key=lambda row: float(row["omega_M"]))
        delta = np.array(
            [complex(float(row["delta_Rgg_real"]), float(row["delta_Rgg_imag"])) for row in rows]
        )
        model = master.build_model(
            model_name, ratio * master.extremal_charge(model_name, 1.0)
        )
        static = master.independent_static_continuation(model, parity)
        normalized = (
            static_normalized_axial(model_name, LOW_FREQUENCIES, delta, static)
            if parity == "axial" else np.full_like(delta, np.nan)
        )
        finite = (
            polar_response(model_name, LOW_FREQUENCIES, delta, static)
            if parity == "polar" else normalized
        )
        output.append(
            {
                "model": model_name,
                "parity": parity,
                "ratio": ratio,
                "charge": float(rows[0]["ell_over_M"]),
                "static": static,
                "delta": [[value.real, value.imag] for value in delta],
                "finite": [[value.real, value.imag] for value in finite],
                "axial_normalized": [[value.real, value.imag] for value in normalized],
                "counterterms": {},
            }
        )
    return sorted(output, key=lambda row: (row["model"], row["parity"], row["ratio"]))


def model_label(model_name: str) -> str:
    return model_name.replace("_", "-").title()


def validate_axial_scaling(results: list[dict], path: Path) -> dict:
    
    report = {}
    for model_name in MODELS:
        slopes = []
        for row in results:
            if row["model"] != model_name or row["parity"] != "axial":
                continue
            response = np.array([complex(*value) for value in row["delta"]])
            low = LOW_FREQUENCIES <= 0.022 + 1.0e-14
            slope = np.polyfit(
                np.log(LOW_FREQUENCIES[low]), np.log(np.abs(response[low])), 1
            )[0]
            slopes.append(float(slope))
        expected = axial_leading_power(model_name)
        report[model_name] = {
            "expected_integer_power": expected,
            "fitted_power_minimum": float(np.min(slopes)),
            "fitted_power_median": float(np.median(slopes)),
            "fitted_power_maximum": float(np.max(slopes)),
        }
        if max(abs(np.asarray(slopes) - expected)) > 0.30:
            raise AssertionError(f"unstable axial power for {model_name}: {slopes}")
    report["status"] = "passed"
    path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def scale_symbol(model_name: str) -> str:
    return {"bardeen": "B", "hayward": "H", "fan_wang": "FW"}[model_name]


def plot_low(results: list[dict], output_dir: Path) -> None:
    colors = {0.010: "#0072B2", 0.020: "#E69F00", 0.030: "#009E73"}
    for model_name in MODELS:
        selected = sorted(
            [row for row in results if row["model"] == model_name and row["parity"] == "polar"],
            key=lambda row: row["ratio"],
        )
        ratios = np.array([row["ratio"] for row in selected])
        static = np.array([row["static"] for row in selected])
        figure, axis = plt.subplots(figsize=(6.5, 4.6))
        axis.plot(ratios, static, "k--", lw=1.7, label="Independent static result")
        for omega in DISPLAY_FREQUENCIES:
            index = int(np.flatnonzero(np.isclose(LOW_FREQUENCIES, omega))[0])
            values = [row["finite"][index][0] for row in selected]
            axis.plot(ratios, values, "o-", ms=3.8, lw=1.4, color=colors[omega], label=rf"$\omega M={omega:.2f}$")
        axis.axhline(0.0, color="0.45", lw=0.75)
        symbol = scale_symbol(model_name)
        axis.set_xlabel(rf"$\ell_{{{symbol}}}/\ell_{{\rm ext}}$")
        axis.set_ylabel(r"$\mathrm{Re}\,\alpha_2(\omega)/M^5$")
        axis.set_title(f"{model_label(model_name)} polar finite-frequency response")
        axis.grid(alpha=0.22)
        axis.legend(frameon=False, fontsize=8.5)
        figure.tight_layout()
        filename = {"bardeen": "Dynamic_TLN_Bardeen.png", "hayward": "Dynamic_TLN_Hayward.png", "fan_wang": "Dynamic_TLN_FanWang.png"}[model_name]
        figure.savefig(output_dir / filename, dpi=240)
        plt.close(figure)

        axial = sorted(
            [row for row in results if row["model"] == model_name and row["parity"] == "axial"],
            key=lambda row: row["ratio"],
        )
        ratios = np.array([row["ratio"] for row in axial])
        static = np.array([row["static"] for row in axial])
        figure, axis = plt.subplots(figsize=(6.5, 4.6))
        axis.plot(ratios, static, "k--", lw=1.7, label="Independent static result")
        for omega in DISPLAY_FREQUENCIES:
            index = int(np.flatnonzero(np.isclose(LOW_FREQUENCIES, omega))[0])
            values = [row["axial_normalized"][index][0] for row in axial]
            axis.plot(ratios, values, "o-", ms=3.8, lw=1.4, color=colors[omega], label=rf"$\omega M={omega:.2f}$")
        axis.axhline(0.0, color="0.45", lw=0.75)
        axis.set_xlabel(rf"$\ell_{{{symbol}}}/\ell_{{\rm ext}}$")
        axis.set_ylabel(r"$\mathrm{Re}\,\beta_2^{\rm norm}(\omega)/M^5$")
        axis.set_title(f"{model_label(model_name)} axial response")
        axis.grid(alpha=0.22)
        axis.legend(frameon=False, fontsize=8.5)
        figure.tight_layout()
        axial_filename = {"bardeen": "Axial_Dynamic_Bardeen.png", "hayward": "Axial_Dynamic_Hayward.png", "fan_wang": "Axial_Dynamic_FanWang.png"}[model_name]
        figure.savefig(output_dir / axial_filename, dpi=240)
        plt.close(figure)


def plot_broad(results: list[dict], output_dir: Path) -> dict:
    summary = {"classification": "broad polar scattering maxima", "cases": []}
    standard = [row for row in results if not row["strict"]]
    strict = {(row["model"], row["ratio"]): row for row in results if row["strict"]}
    colors = plt.cm.viridis(np.linspace(0.10, 0.90, len(BROAD_RATIOS)))
    for model_name in MODELS:
        figure, axes = plt.subplots(1, 2, figsize=(10.6, 4.2))
        for color, ratio in zip(colors, BROAD_RATIOS):
            row = next(item for item in standard if item["model"] == model_name and np.isclose(item["ratio"], ratio))
            values = np.array([complex(*value) for value in row["delta"]])
            axes[0].plot(BROAD_FREQUENCIES, values.real, "o-", ms=2.8, lw=1.2, color=color, label=rf"$\ell/\ell_{{\rm ext}}={ratio:.2f}$")
            axes[1].plot(BROAD_FREQUENCIES, abs(values), "o-", ms=2.8, lw=1.2, color=color, label=rf"$\ell/\ell_{{\rm ext}}={ratio:.2f}$")
            peak = int(np.argmax(abs(values)))
            strict_row = strict[(model_name, ratio)]
            strict_values = np.array([complex(*value) for value in strict_row["delta"]])
            # Compare the complex responses at the frequency where the
            # standard-control response magnitude peaks.
            relative = abs(strict_values[peak] - values[peak]) / max(abs(strict_values[peak]), 1.0e-30)
            summary["cases"].append(
                {
                    "model": model_name,
                    "ell_over_ell_ext": ratio,
                    "sampled_peak_omega_M": float(BROAD_FREQUENCIES[peak]),
                    "peak_abs": float(abs(values[peak])),
                    "relative_peak_response_change": float(relative),
                }
            )
        axes[0].axhline(0.0, color="0.45", lw=0.75)
        axes[0].set_ylabel(r"$\mathrm{Re}\,\mathcal{R}_{gg}^{\rm polar}$")
        axes[1].set_ylabel(r"$|\mathcal{R}_{gg}^{\rm polar}|$")
        for axis in axes:
            axis.set_xlabel(r"$\omega M$")
            axis.grid(alpha=0.22)
            axis.legend(frameon=False, fontsize=8.0)
        axes[0].set_title("Dispersive response")
        axes[1].set_title("Broad scattering maximum")
        figure.suptitle(f"{model_label(model_name)} canonical polar response")
        figure.tight_layout()
        filename = {"bardeen": "freq_response_Bardeen.png", "hayward": "freq_response_Hayward.png", "fan_wang": "freq_response_FanWang.png"}[model_name]
        figure.savefig(output_dir / filename, dpi=240)
        plt.close(figure)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=Path(".."))
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument(
        "--broad-only",
        action="store_true",
        help="Skip the legacy low-frequency master-field scan and compute only the broad polar resonance data.",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output_dir = args.output_dir if args.output_dir.is_absolute() else (root / args.output_dir).resolve()
    result_dir = root / "results"
    result_dir.mkdir(exist_ok=True)

    if args.plot_only:
        low_results = load_low_csv(result_dir / "all_models_low_frequency.csv")
        write_low_csv(low_results, result_dir / "all_models_low_frequency.csv")
        validate_axial_scaling(
            low_results, result_dir / "axial_low_frequency_scaling.json"
        )
        plot_low(low_results, output_dir)
        return

    if not args.broad_only:
        low_tasks = [
            (model_name, parity, float(ratio))
            for model_name in MODELS for parity in ("polar", "axial") for ratio in CHARGE_RATIOS
        ]
        low_results = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(solve_low_task, task) for task in low_tasks]
            for count, future in enumerate(as_completed(futures), start=1):
                low_results.append(future.result())
                print(f"low-frequency task {count}/{len(futures)}", flush=True)
        low_results.sort(key=lambda row: (row["model"], row["parity"], row["ratio"]))
        write_low_csv(low_results, result_dir / "all_models_low_frequency.csv")
        validate_axial_scaling(
            low_results, result_dir / "axial_low_frequency_scaling.json"
        )
        plot_low(low_results, output_dir)

    broad_tasks = [
        (model_name, float(ratio), False)
        for model_name in MODELS for ratio in BROAD_RATIOS
    ] + [
        (model_name, float(ratio), True)
        for model_name in MODELS for ratio in BROAD_RATIOS
    ]
    broad_results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(solve_broad_task, task) for task in broad_tasks]
        for count, future in enumerate(as_completed(futures), start=1):
            broad_results.append(future.result())
            print(f"broad-frequency task {count}/{len(futures)}", flush=True)
    broad_results.sort(key=lambda row: (row["model"], row["strict"], row["ratio"]))
    write_broad_csv(broad_results, result_dir / "all_models_broad_frequency.csv")
    summary = plot_broad(broad_results, output_dir)
    (result_dir / "all_models_convergence.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
