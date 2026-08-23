#!/usr/bin/env python3


from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import einstein_nled_master as master


Array = np.ndarray


@dataclass(frozen=True)
class EvolutionControls:
    spacing: float = 0.10
    outer_radius: float = 180.0
    horizon_offset: float = 1.0e-8
    final_time: float = 150.0
    courant: float = 0.45
    pulse_offset: float = 12.0
    pulse_width: float = 3.0
    extraction_offset: float = 25.0


def inverse_tortoise_grid(
    metric: Callable[[Array], Array],
    horizon: float,
    controls: EvolutionControls,
) -> tuple[Array, Array]:
    def reach_outer(_: float, radius: Array) -> float:
        return float(radius[0] - controls.outer_radius)

    reach_outer.terminal = True
    reach_outer.direction = 1
    solution = solve_ivp(
        lambda _x, radius: [float(metric(radius[0]))],
        (0.0, 1500.0),
        [horizon + controls.horizon_offset],
        events=reach_outer,
        dense_output=True,
        rtol=1.0e-11,
        atol=1.0e-13,
        max_step=0.1,
    )
    if not solution.success or solution.t_events[0].size != 1:
        raise RuntimeError("The inverse tortoise integration did not reach the outer boundary.")
    length = float(solution.t_events[0][0])
    points = int(length / controls.spacing) + 1
    tortoise = np.linspace(0.0, length, points)
    return tortoise, solution.sol(tortoise)[0]


def evolve(
    metric: Callable[[Array], Array],
    potential: Callable[[Array], Array],
    horizon: float,
    controls: EvolutionControls,
) -> tuple[Array, Array, dict[str, float]]:
    tortoise, radius = inverse_tortoise_grid(metric, horizon, controls)
    spacing = float(tortoise[1] - tortoise[0])
    values = np.asarray(potential(radius), dtype=float)
    if values.ndim == 1:
        values = values[:, None, None]
    weighted = np.asarray(metric(radius))[:, None, None] * values
    channels = values.shape[1]
    largest_potential = np.linalg.eigvalsh(weighted)[:, -1]
    peak_index = int(np.argmax(largest_potential))
    peak_position = float(tortoise[peak_index])
    pulse_center = peak_position + controls.pulse_offset
    extraction_position = peak_position + controls.extraction_offset
    extraction_index = int(np.argmin(abs(tortoise - extraction_position)))

    initial = np.zeros((channels, tortoise.size), dtype=float)
    initial[0] = np.exp(-((tortoise - pulse_center) / controls.pulse_width) ** 2)

    def acceleration(field: Array) -> Array:
        result = np.zeros_like(field)
        result[:, 1:-1] = (
            field[:, 2:] - 2.0 * field[:, 1:-1] + field[:, :-2]
        ) / spacing**2
        result[:, 1:-1] -= np.einsum(
            "nij,jn->in", weighted[1:-1], field[:, 1:-1]
        )
        return result

    time_step = controls.courant * spacing
    steps = int(controls.final_time / time_step)
    time_step = controls.final_time / steps
    previous = initial.copy()
    current = initial + 0.5 * time_step**2 * acceleration(initial)
    absorbing = (time_step - spacing) / (time_step + spacing)
    times: list[float] = []
    signal: list[float] = []
    for step in range(1, steps):
        following = 2.0 * current - previous + time_step**2 * acceleration(current)
        following[:, 0] = current[:, 1] + absorbing * (
            following[:, 1] - current[:, 0]
        )
        following[:, -1] = current[:, -2] + absorbing * (
            following[:, -2] - current[:, -1]
        )
        previous, current = current, following
        if step % 2 == 0:
            times.append((step + 1) * time_step)
            signal.append(float(current[0, extraction_index]))
    metadata = {
        "spacing": spacing,
        "outer_radius": controls.outer_radius,
        "horizon_offset": controls.horizon_offset,
        "potential_peak_tortoise": peak_position,
        "potential_peak_radius": float(radius[peak_index]),
        "extraction_tortoise": float(tortoise[extraction_index]),
    }
    return np.asarray(times), np.asarray(signal), metadata


def fit_ringdown(
    time: Array,
    signal: Array,
    start: float,
    stop: float,
) -> dict[str, float]:
    mask = (time >= start) & (time <= stop)
    shifted = time[mask] - start
    data = signal[mask]
    if data.size < 100:
        raise ValueError("The ringdown fit window is too short.")

    def residual(parameters: Array) -> Array:
        amplitude, damping, frequency, phase = parameters
        model = amplitude * np.exp(-damping * shifted) * np.cos(
            frequency * shifted + phase
        )
        return model - data

    best = None
    scale = float(max(np.max(abs(data)), 1.0e-10))
    for frequency_guess in np.linspace(0.30, 0.55, 6):
        fit = least_squares(
            residual,
            [data[0], 0.09, frequency_guess, 0.0],
            bounds=([-2.0 * scale, 0.0, 0.20, -20.0], [2.0 * scale, 0.40, 0.70, 20.0]),
            max_nfev=4000,
        )
        relative_residual = float(np.linalg.norm(fit.fun) / np.linalg.norm(data))
        if best is None or relative_residual < best[0]:
            best = (relative_residual, fit.x)
    if best is None:
        raise RuntimeError("The ringdown fit failed.")
    relative_residual, parameters = best
    return {
        "amplitude": float(parameters[0]),
        "damping_rate_M": float(parameters[1]),
        "frequency_M": float(parameters[2]),
        "phase": float(parameters[3]),
        "relative_fit_residual": relative_residual,
        "fit_start_M": start,
        "fit_stop_M": stop,
    }


def bardeen_problem(charge_ratio: float):
    model = master.build_model(
        "bardeen", charge_ratio * master.extremal_charge("bardeen", 1.0)
    )
    return (
        model.f,
        lambda radius: master.potential_matrix(model, "polar", radius, 2),
        master.outer_horizon(model),
    )


def schwarzschild_problem():
    return (
        lambda radius: 1.0 - 2.0 / np.asarray(radius),
        lambda radius: master.schwarzschild_potential("polar", radius, 1.0, 2),
        2.0,
    )


def load_real_axis_peaks(path: Path) -> dict[float, float]:
    records = json.loads(path.read_text())
    return {
        float(record["charge_ratio"]): float(record["omega_peak_M"])
        for record in records
        if record["parity"] == "polar"
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_case(problem, controls: EvolutionControls, fit_window: tuple[float, float]):
    time, signal, metadata = evolve(*problem, controls)
    fit = fit_ringdown(time, signal, *fit_window)
    return time, signal, metadata, fit


def run_diagnostic(args: argparse.Namespace):
    root = Path(__file__).resolve().parent
    peaks = load_real_axis_peaks(root / "results/resonance_summary.json")
    reference = EvolutionControls()
    reference_window = (60.0, 120.0)
    control_variants = [
        ("spacing", 0.08, EvolutionControls(spacing=0.08)),
        ("spacing", 0.12, EvolutionControls(spacing=0.12)),
        ("outer_radius", 160.0, EvolutionControls(outer_radius=160.0)),
        ("outer_radius", 220.0, EvolutionControls(outer_radius=220.0)),
        ("horizon_offset", 5.0e-9, EvolutionControls(horizon_offset=5.0e-9)),
        ("horizon_offset", 2.0e-8, EvolutionControls(horizon_offset=2.0e-8)),
    ]
    fit_windows = [(55.0, 115.0), (65.0, 125.0), (70.0, 140.0)]

    schwarz_time, schwarz_signal, _, schwarz_fit = run_case(
        schwarzschild_problem(), reference, (65.0, 140.0)
    )
    schwarz_reference = 0.37367168 - 0.08896232j
    schwarz_computed = complex(
        schwarz_fit["frequency_M"], -schwarz_fit["damping_rate_M"]
    )
    schwarz_error = abs(schwarz_computed - schwarz_reference) / abs(schwarz_reference)
    benchmark_passed = schwarz_error <= args.schwarzschild_tolerance
    print(
        f"Schwarzschild: omega M={schwarz_computed.real:.8f}{schwarz_computed.imag:+.8f}i, "
        f"relative error={schwarz_error:.3e}",
        flush=True,
    )

    rows: list[dict] = []
    convergence: list[dict] = []
    waveforms: dict[float, tuple[Array, Array, dict]] = {}
    for ratio in args.charge_ratios:
        ratio = float(ratio)
        problem = bardeen_problem(ratio)
        time, signal, metadata, fit = run_case(problem, reference, reference_window)
        waveforms[ratio] = (time, signal, fit)
        reference_frequency = complex(fit["frequency_M"], -fit["damping_rate_M"])
        shifts: list[float] = []
        control_modes = [reference_frequency]

        for name, value, controls in control_variants:
            _, _, _, variant_fit = run_case(problem, controls, reference_window)
            variant_frequency = complex(
                variant_fit["frequency_M"], -variant_fit["damping_rate_M"]
            )
            shift = abs(variant_frequency - reference_frequency)
            shifts.append(shift)
            control_modes.append(variant_frequency)
            convergence.append(
                {
                    "charge_ratio": ratio,
                    "control": name,
                    "control_value": value,
                    "frequency_M": variant_frequency.real,
                    "damping_rate_M": -variant_frequency.imag,
                    "absolute_complex_shift": shift,
                    "relative_fit_residual": variant_fit["relative_fit_residual"],
                }
            )
        for window in fit_windows:
            variant_fit = fit_ringdown(time, signal, *window)
            variant_frequency = complex(
                variant_fit["frequency_M"], -variant_fit["damping_rate_M"]
            )
            shift = abs(variant_frequency - reference_frequency)
            shifts.append(shift)
            control_modes.append(variant_frequency)
            convergence.append(
                {
                    "charge_ratio": ratio,
                    "control": "fit_window_start",
                    "control_value": window[0],
                    "frequency_M": variant_frequency.real,
                    "damping_rate_M": -variant_frequency.imag,
                    "absolute_complex_shift": shift,
                    "relative_fit_residual": variant_fit["relative_fit_residual"],
                }
            )

        peak = peaks[ratio]
        alignment = abs(reference_frequency.real - peak)
        detuning = alignment / (-reference_frequency.imag)
        robust_detuning = max(
            abs(peak - mode.real) / (-mode.imag) for mode in control_modes
        )
        resonance_aligned = bool(benchmark_passed and detuning <= 1.0)
        rows.append(
            {
                "charge_ratio": ratio,
                "real_axis_peak_omega_M": peak,
                "ringdown_frequency_M": reference_frequency.real,
                "ringdown_damping_rate_M": -reference_frequency.imag,
                "quality_factor": reference_frequency.real / (-2.0 * reference_frequency.imag),
                "peak_ringdown_separation": alignment,
                "linewidth_detuning": detuning,
                "robust_linewidth_detuning": robust_detuning,
                "qnm_resonance_aligned": resonance_aligned,
                "controls_preserve_alignment": bool((robust_detuning <= 1.0) == resonance_aligned),
                "maximum_control_shift": max(shifts),
                "relative_fit_residual": fit["relative_fit_residual"],
                "potential_peak_radius_M": metadata["potential_peak_radius"],
                "schwarzschild_benchmark_passed": benchmark_passed,
            }
        )
        print(
            f"q/qext={ratio:.2f}: omega M={reference_frequency.real:.8f}"
            f"{reference_frequency.imag:+.8f}i, peak separation={alignment:.3e}, "
            f"max control shift={max(shifts):.3e}",
            flush=True,
        )

    summary = {
        "method": "time-domain evolution and single-mode damped-sinusoid ringdown fit",
        "schwarzschild_benchmark": {
            "reference_frequency_M": schwarz_reference.real,
            "reference_damping_rate_M": -schwarz_reference.imag,
            "computed_frequency_M": schwarz_computed.real,
            "computed_damping_rate_M": -schwarz_computed.imag,
            "relative_complex_error": schwarz_error,
            "passed": benchmark_passed,
        },
        "criterion": "A real-frequency maximum is QNM-resonance aligned when its nominal detuning from the fitted real QNM frequency is no larger than one damping width. Control fits test whether the classification is preserved. This is not a pole identification.",
        "all_qnm_resonance_aligned": bool(rows and all(row["qnm_resonance_aligned"] for row in rows)),
        "cases": rows,
    }
    return rows, convergence, summary, waveforms, (schwarz_time, schwarz_signal, schwarz_fit)


def plot_results(rows: list[dict], output: Path) -> None:
    root = Path(__file__).resolve().parent
    broad_path = root.parents[1] / "results" / "all_models_broad_frequency.csv"
    with broad_path.open(newline="") as stream:
        broad_rows = [
            row for row in csv.DictReader(stream)
            if row["model"] == "bardeen" and int(row["strict_controls"]) == 0
        ]

    figure, axes = plt.subplots(1, 2, figsize=(10.6, 4.4))
    ordered_rows = sorted(rows, key=lambda row: float(row["charge_ratio"]))
    ratios = [float(row["charge_ratio"]) for row in ordered_rows]
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(ratios)))

    for color, ratio, ringdown_row in zip(colors, ratios, ordered_rows):
        case = sorted(
            (
                row for row in broad_rows
                if abs(float(row["ell_over_ell_ext"]) - ratio) < 1.0e-12
            ),
            key=lambda row: float(row["omega_M"]),
        )
        axes[0].plot(
            [float(row["omega_M"]) for row in case],
            [float(row["delta_Rgg_abs"]) for row in case],
            "o-",
            color=color,
            markersize=2.8,
            linewidth=1.2,
            label=rf"$\ell/\ell_{{\rm ext}}={ratio:.2f}$",
        )
        axes[0].axvline(
            float(ringdown_row["real_axis_peak_omega_M"]),
            color=color,
            alpha=0.25,
            linewidth=0.9,
        )

    axes[0].set(
        xlabel=r"$M\omega$",
        ylabel=r"$|\mathcal{R}_{gg}^{\rm polar}|$",
        title="Broad canonical polar response",
    )
    axes[0].grid(alpha=0.22)
    axes[0].legend(frameon=False, fontsize=8)

    positions = np.arange(len(ordered_rows))
    peaks = np.array([float(row["real_axis_peak_omega_M"]) for row in ordered_rows])
    modes = np.array([float(row["ringdown_frequency_M"]) for row in ordered_rows])
    errors = np.array([float(row["maximum_control_shift"]) for row in ordered_rows])
    axes[1].plot(positions, peaks, "D-", label="real-axis maximum")
    axes[1].errorbar(
        positions,
        modes,
        yerr=errors,
        fmt="o-",
        capsize=3,
        label="polar ringdown frequency",
    )
    axes[1].set_xticks(positions, [f"{ratio:.2f}" for ratio in ratios])
    axes[1].set(
        xlabel=r"$\ell/\ell_{\rm ext}$",
        ylabel=r"$M\omega$",
        title="QNM-resonance alignment",
    )
    axes[1].grid(alpha=0.22)
    axes[1].legend(frameon=False, fontsize=8)

    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--charge-ratios", nargs="+", type=float, default=[0.60, 0.90, 0.97])
    parser.add_argument("--schwarzschild-tolerance", type=float, default=2.0e-2)
    parser.add_argument("--result-dir", type=Path, default=Path("results"))
    parser.add_argument("--figure", type=Path, default=Path("figures/bardeen_polar_ringdown_test.png"))
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate the two-panel figure from the archived numerical tables.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    root = Path(__file__).resolve().parent
    result_dir = args.result_dir if args.result_dir.is_absolute() else root / args.result_dir
    figure = args.figure if args.figure.is_absolute() else root / args.figure
    if args.plot_only:
        with (result_dir / "bardeen_polar_ringdown.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        plot_results(rows, figure)
        print(f"Wrote {figure}")
        return
    rows, convergence, summary, waveforms, schwarz = run_diagnostic(args)
    write_csv(result_dir / "bardeen_polar_ringdown.csv", rows)
    write_csv(result_dir / "bardeen_polar_ringdown_convergence.csv", convergence)
    (result_dir / "bardeen_polar_ringdown_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    plot_results(rows, figure)
    print(f"Wrote {result_dir / 'bardeen_polar_ringdown_summary.json'}")
    print(f"Wrote {figure}")


if __name__ == "__main__":
    main()
