#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
METRIC = ROOT / "metric_level"
TABLES = ROOT.parent / "manuscript_tables"
CRITERION = (
    "A real-frequency maximum is QNM-resonance aligned when "
    "abs(omega_peak-omega_R) <= abs(omega_I). The nominal linewidth detuning "
    "determines the classification; control fits test whether it is preserved. "
    "This operational label is not a complex-pole identification."
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def control_modes(path: Path, keys: tuple[str, ...], damping_key: str) -> dict[tuple, list[tuple[float, float]]]:
    grouped: dict[tuple, list[tuple[float, float]]] = {}
    for row in read_csv(path):
        key = tuple(row[item] for item in keys)
        grouped.setdefault(key, []).append((float(row["frequency_M"]), float(row[damping_key])))
    return grouped


def classify(peak: float, omega_r: float, gamma: float, controls: list[tuple[float, float]]) -> tuple[float, float, bool, bool]:
    detuning = abs(peak - omega_r) / gamma
    values = [detuning] + [abs(peak - frequency) / damping for frequency, damping in controls]
    robust = max(values)
    aligned = detuning <= 1.0
    return detuning, robust, aligned, (robust <= 1.0) == aligned


def update_hayward_fanwang_polar() -> list[dict]:
    base = METRIC / "hayward_fanwang_polar" / "results"
    controls = control_modes(base / "resonance_convergence.csv", ("model", "ell_over_ell_ext"), "damping_M")
    output = []
    for row in read_csv(base / "resonance_test.csv"):
        model, ratio = row["model"], row["ell_over_ell_ext"]
        peak = float(row["real_axis_peak_M"])
        omega_r, gamma = float(row["ringdown_frequency_M"]), float(row["ringdown_damping_M"])
        detuning, robust, aligned, preserved = classify(peak, omega_r, gamma, controls[(model, ratio)])
        clean = {key: value for key, value in row.items() if key != "resonance_supported"}
        clean.update(linewidth_detuning=detuning, robust_linewidth_detuning=robust,
                     qnm_resonance_aligned=aligned, controls_preserve_alignment=preserved)
        output.append(clean)
    write_csv(base / "resonance_test.csv", output)
    write_csv(base / "hayward_resonance_test.csv", [row for row in output if row["model"] == "hayward"])
    write_csv(base / "fan_wang_resonance_test.csv", [row for row in output if row["model"] == "fan_wang"])
    (base / "resonance_summary.json").write_text(json.dumps({
        "criterion": CRITERION,
        "all_qnm_resonance_aligned": all(row["qnm_resonance_aligned"] for row in output),
        "cases": output,
    }, indent=2) + "\n")
    return output


def update_hayward_fanwang_axial() -> list[dict]:
    base = METRIC / "hayward_fanwang_axial" / "results"
    controls = control_modes(base / "hayward_fanwang_axial_ringdown_convergence.csv", ("model", "ell_over_ell_ext"), "damping_M")
    output = []
    for row in read_csv(base / "hayward_fanwang_axial_ringdown.csv"):
        model, ratio = row["model"], row["ell_over_ell_ext"]
        peak = float(row["real_axis_peak_M"])
        omega_r, gamma = float(row["ringdown_frequency_M"]), float(row["ringdown_damping_M"])
        detuning, robust, aligned, preserved = classify(peak, omega_r, gamma, controls[(model, ratio)])
        clean = {key: value for key, value in row.items() if key != "resonance_supported"}
        clean.update(linewidth_detuning=detuning, robust_linewidth_detuning=robust,
                     qnm_resonance_aligned=aligned, controls_preserve_alignment=preserved)
        output.append(clean)
    write_csv(base / "hayward_fanwang_axial_ringdown.csv", output)
    schwarzschild = {key: output[0][key] for key in ["schwarzschild_relative_error"]}
    (base / "hayward_fanwang_axial_resonance_summary.json").write_text(json.dumps({
        "criterion": CRITERION,
        "schwarzschild": schwarzschild,
        "any_qnm_resonance_aligned": any(row["qnm_resonance_aligned"] for row in output),
        "cases": output,
    }, indent=2) + "\n")
    return output


def update_bardeen_axial() -> list[dict]:
    base = METRIC / "bardeen_axial" / "results"
    controls = control_modes(base / "bardeen_axial_ringdown_convergence.csv", ("ell_over_ell_ext",), "damping_M")
    peak_rows = json.loads((base / "bardeen_axial_broad_response_summary.json").read_text())
    peaks = {str(float(row["ell_over_ell_ext"])): float(row["peak_omega_M"]) for row in peak_rows}
    output = []
    for row in read_csv(base / "bardeen_axial_ringdown.csv"):
        ratio = str(float(row["ell_over_ell_ext"]))
        peak = peaks[ratio]
        omega_r, gamma = float(row["ringdown_frequency_M"]), float(row["ringdown_damping_M"])
        detuning, robust, aligned, preserved = classify(peak, omega_r, gamma, controls[(row["ell_over_ell_ext"],)])
        clean = {key: value for key, value in row.items() if key != "resonance_supported"}
        clean.update(real_axis_peak_M=peak, peak_ringdown_separation=abs(peak - omega_r),
                     linewidth_detuning=detuning, robust_linewidth_detuning=robust,
                     qnm_resonance_aligned=aligned, controls_preserve_alignment=preserved)
        output.append(clean)
    write_csv(base / "bardeen_axial_ringdown.csv", output)
    (base / "bardeen_axial_resonance_summary.json").write_text(json.dumps({
        "criterion": CRITERION,
        "any_qnm_resonance_aligned": any(row["qnm_resonance_aligned"] for row in output),
        "cases": output,
    }, indent=2) + "\n")
    return output


def update_bardeen_polar() -> list[dict]:
    base = METRIC / "bardeen_polar_resonance" / "results"
    controls = control_modes(base / "bardeen_polar_ringdown_convergence.csv", ("charge_ratio",), "damping_rate_M")
    output = []
    for row in read_csv(base / "bardeen_polar_ringdown.csv"):
        ratio = row["charge_ratio"]
        peak = float(row["real_axis_peak_omega_M"])
        omega_r, gamma = float(row["ringdown_frequency_M"]), float(row["ringdown_damping_rate_M"])
        detuning, robust, aligned, preserved = classify(peak, omega_r, gamma, controls[(ratio,)])
        keep = ["charge_ratio", "real_axis_peak_omega_M", "ringdown_frequency_M",
                "ringdown_damping_rate_M", "quality_factor", "peak_ringdown_separation",
                "maximum_control_shift", "relative_fit_residual", "potential_peak_radius_M",
                "schwarzschild_benchmark_passed"]
        clean = {key: row[key] for key in keep}
        clean.update(linewidth_detuning=detuning, robust_linewidth_detuning=robust,
                     qnm_resonance_aligned=aligned, controls_preserve_alignment=preserved)
        output.append(clean)
    write_csv(base / "bardeen_polar_ringdown.csv", output)
    (base / "bardeen_polar_ringdown_summary.json").write_text(json.dumps({
        "criterion": CRITERION,
        "all_qnm_resonance_aligned": all(row["qnm_resonance_aligned"] for row in output),
        "cases": output,
    }, indent=2) + "\n")
    return output


def aggregate(groups: list[tuple[str, str, list[dict]]]) -> None:
    rows = []
    for geometry, parity, data in groups:
        for row in data:
            name = geometry or row.get("model", "").replace("_", " ").title()
            ratio = row.get("ell_over_ell_ext", row.get("charge_ratio"))
            peak = row.get("real_axis_peak_M", row.get("real_axis_peak_omega_M"))
            gamma = row.get("ringdown_damping_M", row.get("ringdown_damping_rate_M"))
            rows.append({
                "geometry": name, "parity": parity, "ell_over_ell_ext": ratio,
                "peak_frequency_M": peak, "ringdown_frequency_M": row["ringdown_frequency_M"],
                "ringdown_damping_M": gamma, "linewidth_detuning": row["linewidth_detuning"],
                "robust_linewidth_detuning": row["robust_linewidth_detuning"],
                "qnm_resonance_aligned": row["qnm_resonance_aligned"],
                "controls_preserve_alignment": row["controls_preserve_alignment"],
            })
    order = {"Bardeen": 0, "Hayward": 1, "Fan Wang": 2}
    rows.sort(key=lambda row: (row["parity"] == "axial", order[row["geometry"]], float(row["ell_over_ell_ext"])))
    write_csv(TABLES / "qnm_resonance_alignment.csv", rows)
    (TABLES / "qnm_resonance_alignment.json").write_text(json.dumps({
        "criterion": CRITERION, "cases": rows,
    }, indent=2) + "\n")


def main() -> None:
    bardeen_polar = update_bardeen_polar()
    other_polar = update_hayward_fanwang_polar()
    bardeen_axial = update_bardeen_axial()
    other_axial = update_hayward_fanwang_axial()
    aggregate([
        ("Bardeen", "polar", bardeen_polar), ("", "polar", other_polar),
        ("Bardeen", "axial", bardeen_axial), ("", "axial", other_axial),
    ])


if __name__ == "__main__":
    main()
