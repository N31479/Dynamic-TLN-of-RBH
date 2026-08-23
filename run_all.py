#!/usr/bin/env python3
"""Regenerate and verify every figure and numerical table used by main.tex."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CODES = ROOT / "codes"
FIGURES = ROOT / "figures"
TABLES = ROOT / "manuscript_tables"
LOGS = ROOT / "run_logs"

REQUIRED_PACKAGES = ("numpy", "scipy", "sympy", "matplotlib", "mpmath")

MANUSCRIPT_FIGURES = (
    "scalar_shell_eft_all_models.pdf",
    "all_models_static_direct.pdf",
    "bardeen_polar_metric_dynamic_tln_vs_ell_ratio.pdf",
    "bardeen_polar_metric_dynamic_tln_vs_frequency.pdf",
    "bardeen_polar_resonance_test.png",
    "hayward_polar_metric_dynamic_tln_vs_ell_ratio.pdf",
    "hayward_polar_metric_dynamic_tln_vs_frequency.pdf",
    "hayward_polar_resonance_test.pdf",
    "fan_wang_polar_metric_dynamic_tln_vs_ell_ratio.pdf",
    "fan_wang_polar_metric_dynamic_tln_vs_frequency.pdf",
    "fan_wang_polar_resonance_test.pdf",
    "bardeen_axial_metric_dynamic_tln_vs_ell_ratio.pdf",
    "bardeen_axial_metric_dynamic_tln_vs_frequency.pdf",
    "bardeen_axial_resonance_test.pdf",
    "hayward_axial_metric_dynamic_tln_vs_ell_ratio.pdf",
    "hayward_axial_metric_dynamic_tln_vs_frequency.pdf",
    "hayward_axial_resonance_test.pdf",
    "fan_wang_axial_metric_dynamic_tln_vs_ell_ratio.pdf",
    "fan_wang_axial_metric_dynamic_tln_vs_frequency.pdf",
    "fan_wang_axial_resonance_test.pdf",
    "tensor_probe_all_models.pdf",
)

MANUSCRIPT_TABLES = (
    "scalar_scheme_summary.json",
    "scalar_shell_convergence.json",
    "static_scaling_comparison.csv",
    "static_direct.csv",
    "static_convergence.json",
    "dynamic_tln_convergence.csv",
    "polar_master_convergence.json",
    "axial_master_convergence.csv",
    "nearzone_amplitudes.csv",
    "nearzone_basis_orders.csv",
    "qnm_resonance_alignment.csv",
    "qnm_resonance_alignment.json",
)


def initialize_output_directories() -> None:
    """Create top-level output directories required by clean code-only runs."""
    for directory in (FIGURES, TABLES, LOGS):
        directory.mkdir(parents=True, exist_ok=True)


def dependency_check() -> None:
    missing = [name for name in REQUIRED_PACKAGES if importlib.util.find_spec(name) is None]
    if missing:
        names = ", ".join(missing)
        raise SystemExit(
            f"Missing Python packages: {names}\n"
            f"Install them with: {sys.executable} -m pip install -r requirements.txt"
        )


def run_step(name: str, command: list[str], cwd: Path = ROOT) -> None:
    LOGS.mkdir(exist_ok=True)
    log_path = LOGS / f"{name}.log"
    print(f"\n[{name}] {' '.join(command)}", flush=True)
    started = time.time()
    environment = os.environ.copy()
    environment.setdefault("MPLBACKEND", "Agg")
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            print(line, end="", flush=True)
        return_code = process.wait()
    if return_code:
        raise RuntimeError(f"Step {name} failed; see {log_path}")
    print(f"[{name}] completed in {time.time() - started:.1f} s", flush=True)


def python_step(name: str, script: Path, *arguments: str, cwd: Path = ROOT) -> None:
    run_step(name, [sys.executable, str(script), *arguments], cwd=cwd)


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def prepare_bardeen_frequency_inputs() -> None:
    package = CODES / "metric_level" / "bardeen_polar_charge"
    raw = package / "results" / "raw_scans_recomputed"
    frequency_results = CODES / "metric_level" / "bardeen_polar_frequency" / "results"
    frequency_results.mkdir(parents=True, exist_ok=True)
    mapping = {
        raw / "ellratio_0p50" / "results" / "bardeen_polar_dynamic_tln.csv":
            frequency_results / "bardeen_polar_metric_frequency_raw_q050.csv",
    }
    for source, destination in mapping.items():
        copy_file(source, destination)


def prepare_bardeen_polar_peaks() -> None:
    source = CODES / "results" / "all_models_convergence.json"
    report = json.loads(source.read_text(encoding="utf-8"))
    cases = []
    for row in report["cases"]:
        if row["model"] != "bardeen":
            continue
        cases.append(
            {
                "charge_ratio": row["ell_over_ell_ext"],
                "omega_peak_M": row["sampled_peak_omega_M"],
                "parity": "polar",
                "peak_abs": row["peak_abs"],
                "relative_peak_response_change": row["relative_peak_response_change"],
            }
        )
    if len(cases) != 3:
        raise RuntimeError("Could not construct the three Bardeen polar peak entries")
    destination = CODES / "metric_level" / "bardeen_polar_resonance" / "results"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "resonance_summary.json").write_text(
        json.dumps(cases, indent=2) + "\n", encoding="utf-8"
    )


def prepare_shared_broad_polar_data() -> None:
    source = CODES / "results" / "all_models_broad_frequency.csv"
    destination = (
        CODES / "metric_level" / "hayward_fanwang_polar" / "results"
        / "all_models_broad_frequency.csv"
    )
    copy_file(source, destination)


def prepare_nearzone_bardeen_data() -> None:
    package = CODES / "metric_level" / "bardeen_polar_charge" / "results"
    rebuilt = package / "raw_scans_recomputed"
    canonical = package / "raw_scans"
    canonical.mkdir(parents=True, exist_ok=True)
    for ratio in (0.50, 0.80, 0.95):
        tag = f"ellratio_{ratio:.2f}".replace(".", "p")
        source = rebuilt / tag / "results" / "bardeen_polar_window_data.csv"
        destination = canonical / tag / "bardeen_polar_window_data.csv"
        copy_file(source, destination)


def prepare_bardeen_axial_broad_results() -> None:
    """Merge the three one-ratio broad scans for the ringdown analysis."""
    package = CODES / "metric_level" / "bardeen_axial"
    results = package / "results"
    rows = []
    for ratio in (0.6, 0.9, 0.97):
        source = results / f"bardeen_axial_broad_q{round(100 * ratio):03d}.csv"
        with source.open(newline="", encoding="utf-8") as stream:
            rows.extend(csv.DictReader(stream))
    rows.sort(key=lambda row: (
        float(row["ell_over_ell_ext"]),
        int(row["strict_controls"]),
        float(row["omega_M"]),
    ))
    fieldnames = list(rows[0])
    with (results / "bardeen_axial_broad_response.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for ratio in (0.6, 0.9, 0.97):
        standard = [
            row for row in rows
            if float(row["ell_over_ell_ext"]) == ratio
            and int(row["strict_controls"]) == 0
        ]
        strict = next(
            row for row in rows
            if float(row["ell_over_ell_ext"]) == ratio
            and int(row["strict_controls"]) == 1
        )
        peak = max(standard, key=lambda row: float(row["delta_Rgg_abs"]))
        peak_value = complex(
            float(peak["delta_Rgg_real"]), float(peak["delta_Rgg_imag"])
        )
        strict_value = complex(
            float(strict["delta_Rgg_real"]), float(strict["delta_Rgg_imag"])
        )
        summary.append({
            "ell_over_ell_ext": ratio,
            "peak_omega_M": float(peak["omega_M"]),
            "peak_abs": float(peak["delta_Rgg_abs"]),
            "relative_peak_response_change": abs(strict_value - peak_value)
            / max(abs(strict_value), 1.0e-30),
        })
    (results / "bardeen_axial_broad_response_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


def run_bardeen_axial_calculation() -> None:
    package = CODES / "metric_level" / "bardeen_axial"
    scripts = package / "codes"
    scan = package / "scan"
    scan.mkdir(parents=True, exist_ok=True)
    short_ratios = (0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.9, 0.99)
    dense_ratios = (0.5, 0.8, 0.95)
    short_frequencies = (0.002, 0.004, 0.006)
    dense_frequencies = (0.001, 0.002, 0.003, 0.004, 0.005, 0.006)
    for ratio in short_ratios:
        tag = f"q{round(100 * ratio):03d}"
        python_step(
            f"bardeen_axial_metric_{tag}",
            scripts / "bardeen_axial_metric_solver.py",
            "--charge-ratio", str(ratio),
            "--frequencies", *(str(value) for value in short_frequencies),
            "--output-dir", str(scan / tag),
        )
    for ratio in dense_ratios:
        tag = f"q{round(100 * ratio):03d}"
        python_step(
            f"bardeen_axial_metric_{tag}",
            scripts / "bardeen_axial_metric_solver.py",
            "--charge-ratio", str(ratio),
            "--frequencies", *(str(value) for value in dense_frequencies),
            "--output-dir", str(scan / tag),
        )
    # q=0 is repeated on the dense grid as the Schwarzschild-limit validation/reference run.
    python_step(
        "bardeen_axial_metric_q000_dense",
        scripts / "bardeen_axial_metric_solver.py",
        "--charge-ratio", "0.0",
        "--frequencies", *(str(value) for value in dense_frequencies),
        "--output-dir", str(scan / "q000"),
    )
    python_step("bardeen_axial_aggregate", scripts / "aggregate_and_plot.py")
    for ratio in (0.6, 0.9, 0.97):
        python_step(
            f"bardeen_axial_broad_q{round(100 * ratio):03d}",
            scripts / "bardeen_axial_broad_one.py",
            "--ratio", str(ratio),
        )
    prepare_bardeen_axial_broad_results()
    python_step("bardeen_axial_ringdown", scripts / "bardeen_axial_ringdown.py")
    python_step("bardeen_axial_plots", scripts / "finalize_plots.py")


def run_hayward_fanwang_axial_calculation() -> None:
    package = CODES / "metric_level" / "hayward_fanwang_axial"
    scripts = package / "codes"
    for name in (
        "run_metric_scans.py",
        "aggregate_metric_scans.py",
        "plot_metric_scans.py",
        "axial_broad_response.py",
        "axial_ringdown.py",
        "plot_resonance.py",
    ):
        python_step(f"hayward_fanwang_axial_{Path(name).stem}", scripts / name)


def copy_manuscript_figures() -> None:
    sources = {
        "bardeen_polar_metric_dynamic_tln_vs_ell_ratio": CODES / "metric_level" / "bardeen_polar_charge" / "figures",
        "bardeen_polar_metric_dynamic_tln_vs_frequency": CODES / "metric_level" / "bardeen_polar_frequency" / "figures",
        "bardeen_polar_resonance_test": CODES / "metric_level" / "bardeen_polar_resonance" / "figures",
        "hayward_polar_metric_dynamic_tln_vs_ell_ratio": CODES / "metric_level" / "hayward_fanwang_polar" / "figures",
        "hayward_polar_metric_dynamic_tln_vs_frequency": CODES / "metric_level" / "hayward_fanwang_polar" / "figures",
        "hayward_polar_resonance_test": CODES / "metric_level" / "hayward_fanwang_polar" / "figures",
        "fan_wang_polar_metric_dynamic_tln_vs_ell_ratio": CODES / "metric_level" / "hayward_fanwang_polar" / "figures",
        "fan_wang_polar_metric_dynamic_tln_vs_frequency": CODES / "metric_level" / "hayward_fanwang_polar" / "figures",
        "fan_wang_polar_resonance_test": CODES / "metric_level" / "hayward_fanwang_polar" / "figures",
        "bardeen_axial_metric_dynamic_tln_vs_ell_ratio": CODES / "metric_level" / "bardeen_axial" / "figures",
        "bardeen_axial_metric_dynamic_tln_vs_frequency": CODES / "metric_level" / "bardeen_axial" / "figures",
        "bardeen_axial_resonance_test": CODES / "metric_level" / "bardeen_axial" / "figures",
        "hayward_axial_metric_dynamic_tln_vs_ell_ratio": CODES / "metric_level" / "hayward_fanwang_axial" / "figures",
        "hayward_axial_metric_dynamic_tln_vs_frequency": CODES / "metric_level" / "hayward_fanwang_axial" / "figures",
        "hayward_axial_resonance_test": CODES / "metric_level" / "hayward_fanwang_axial" / "figures",
        "fan_wang_axial_metric_dynamic_tln_vs_ell_ratio": CODES / "metric_level" / "hayward_fanwang_axial" / "figures",
        "fan_wang_axial_metric_dynamic_tln_vs_frequency": CODES / "metric_level" / "hayward_fanwang_axial" / "figures",
        "fan_wang_axial_resonance_test": CODES / "metric_level" / "hayward_fanwang_axial" / "figures",
    }
    # The Bardeen polar diagnostic script names the same manuscript diagnostic "ringdown".
    ringdown = sources["bardeen_polar_resonance_test"] / "bardeen_polar_ringdown_test"
    for suffix in (".png", ".pdf"):
        if ringdown.with_suffix(suffix).is_file():
            copy_file(ringdown.with_suffix(suffix), FIGURES / f"bardeen_polar_resonance_test{suffix}")
    for stem, directory in sources.items():
        if stem == "bardeen_polar_resonance_test":
            continue
        for suffix in (".png", ".pdf"):
            source = directory / f"{stem}{suffix}"
            if source.is_file():
                copy_file(source, FIGURES / source.name)


def export_table_inputs(strict: bool = True) -> None:
    TABLES.mkdir(exist_ok=True)
    sources = {
        "scalar_scheme_summary.json": CODES / "results" / "scalar_analytic_shell_eft.json",
        "scalar_shell_convergence.json": CODES / "results" / "scalar_shell_eft_convergence.json",
        "static_scaling_comparison.csv": CODES / "results" / "all_models_static_scaling_comparison.csv",
        "static_direct.csv": CODES / "results" / "all_models_static_direct.csv",
        "static_convergence.json": CODES / "results" / "all_models_static_convergence.json",
        "dynamic_tln_convergence.csv": CODES / "results" / "dynamic_tln_convergence.csv",
        "polar_master_convergence.json": CODES / "results" / "all_models_convergence.json",
        "axial_master_convergence.csv": CODES / "results" / "axial_master_response_convergence.csv",
        "nearzone_amplitudes.csv": CODES / "results" / "numerical_nearzone_asymptotics.csv",
    }
    for name, source in sources.items():
        if source.is_file():
            copy_file(source, TABLES / name)
        elif strict:
            raise FileNotFoundError(source)
    with (TABLES / "nearzone_basis_orders.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("sector", "models", "radial_order", "log_order", "frequency_order"))
        writer.writerow(("polar", "Bardeen; Hayward; Fan-Wang", 18, 5, "O(omega^4)"))
        writer.writerow(("axial", "Bardeen", 22, 5, "O(omega^4)"))
        writer.writerow(("axial", "Hayward; Fan-Wang", 22, 6, "O(omega^4)"))


def verify_outputs() -> None:
    missing_figures = [name for name in MANUSCRIPT_FIGURES if not (FIGURES / name).is_file()]
    missing_tables = [name for name in MANUSCRIPT_TABLES if not (TABLES / name).is_file()]
    if missing_figures:
        raise RuntimeError("Missing manuscript figures: " + ", ".join(missing_figures))
    if missing_tables:
        raise RuntimeError("Missing manuscript table files: " + ", ".join(missing_tables))
    table_files = tuple(TABLES / name for name in MANUSCRIPT_TABLES)
    empty = [path for path in [*(FIGURES / name for name in MANUSCRIPT_FIGURES), *table_files] if path.stat().st_size == 0]
    if empty:
        raise RuntimeError("Empty output files: " + ", ".join(str(path) for path in empty))
    solver_files = list(CODES.rglob("einstein_nled_master.py"))
    if solver_files != [CODES / "einstein_nled_master.py"]:
        raise RuntimeError(
            "Expected exactly one canonical einstein_nled_master.py; found: "
            + ", ".join(str(path) for path in solver_files)
        )

    scalar_path = CODES / "results" / "scalar_shell_eft_all_models_direct.csv"
    with scalar_path.open(newline="", encoding="utf-8") as stream:
        scalar_reader = csv.DictReader(stream)
        scalar_rows = list(scalar_reader)
    expected_scalar_schema = {
        "model", "ell_over_ell_ext", "omega_M",
        "direct_real", "direct_imag", "shell_real", "shell_imag",
        "real_residual", "imag_residual", "shell_fit_relative_rms",
    }
    if set(scalar_reader.fieldnames or ()) != expected_scalar_schema:
        raise RuntimeError("Unexpected direct scalar CSV schema")
    scalar_counts = {
        model: sum(row["model"] == model for row in scalar_rows)
        for model in ("bardeen", "hayward", "fan_wang")
    }
    if len(scalar_rows) != 81 or scalar_counts != {
        "bardeen": 27, "hayward": 27, "fan_wang": 27
    }:
        raise RuntimeError(
            "Direct scalar CSV must contain 81 rows: 27 for each of the three models"
        )

    polar = json.loads((TABLES / "polar_master_convergence.json").read_text(encoding="utf-8"))
    polar_cases = polar.get("cases", [])
    if len(polar_cases) != 9 or any(
        "relative_peak_response_change" not in row for row in polar_cases
    ):
        raise RuntimeError("Polar peak-response table must contain nine cases with the current schema")

    with (TABLES / "axial_master_convergence.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        axial_reader = csv.DictReader(stream)
        axial_rows = list(axial_reader)
    if len(axial_rows) != 9 or "relative_peak_response_change" not in (
        axial_reader.fieldnames or ()
    ):
        raise RuntimeError("Axial peak-response table must contain nine cases with the current schema")

    report = {
        "status": "passed",
        "manuscript_figures_verified": len(MANUSCRIPT_FIGURES),
        "manuscript_table_inputs_verified": len(table_files),
        "scalar_direct_rows_verified": len(scalar_rows),
        "canonical_solver_files_verified": len(solver_files),
        "table_manifest_verified": True,
    }
    (ROOT / "run_all_verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nVerification passed: {len(MANUSCRIPT_FIGURES)} figures and {len(table_files)} table files.")


def pre_calculation_validations() -> None:
    # These checks do not require numerical CSV/JSON products from the production scans.
    python_step("schwarzschild_reduction", CODES / "check_schwarzschild_reduction.py")
    python_step("symbolic_crosscheck", CODES / "crosscheck_symbolic_python.py")
    python_step("structural_validation", CODES / "validate_all_models.py")
    python_step("scalar_junction", CODES / "check_scalar_shell_junction.py")
    python_step("scalar_source_subtraction", CODES / "check_scalar_shell_source_subtraction.py")


def post_metric_validations() -> None:
    # This check reads CSV files generated by the metric-level production scans, so it
    # must run only after those scans have completed in a clean codes-only checkout.
    python_step("polar_metric_normalization", CODES / "check_polar_metric_normalization.py")


def full_calculation(workers: int, resume: bool = False, skip_validations: bool = False) -> None:
    if not resume:
        python_step("static_tln", CODES / "run_static_tln.py")
        python_step("scalar_shell", CODES / "scalar_shell_eft_all_models.py")
        python_step("scalar_analytic", CODES / "scalar_analytic_shell_eft.py")
        python_step("scalar_shell_convergence", CODES / "scalar_shell_eft_convergence.py")
        python_step("tensor_probe", CODES / "tensor_probe_response.py", "--output-root", str(ROOT))
    export_table_inputs(strict=False)

    print("\nGenerating the manuscript metric-level dynamical TLN plots before the broad resonance scans.", flush=True)
    bardeen_charge = CODES / "metric_level" / "bardeen_polar_charge"
    python_step("bardeen_polar_metric", bardeen_charge / "codes" / "run_bardeen_polar_charge_scan.py")
    prepare_bardeen_frequency_inputs()
    python_step(
        "bardeen_polar_frequency",
        CODES / "metric_level" / "bardeen_polar_frequency" / "codes" / "aggregate_bardeen_polar_frequency.py",
    )

    hf_polar = CODES / "metric_level" / "hayward_fanwang_polar"
    python_step("hayward_fanwang_polar_metric", hf_polar / "codes" / "run_metric_scans.py")

    run_bardeen_axial_calculation()
    run_hayward_fanwang_axial_calculation()

    if not skip_validations:
        post_metric_validations()
    python_step("dynamic_tln_convergence", CODES / "dynamic_tln_convergence.py")
    python_step("axial_master_convergence", CODES / "resonance_checks" / "axial_master_response_convergence.py")
    prepare_nearzone_bardeen_data()
    python_step("nearzone_amplitudes", CODES / "numerical_nearzone_asymptotics.py")
    copy_manuscript_figures()
    export_table_inputs(strict=False)
    print("\nAll metric-level dynamical TLN plots are now available in figures/.", flush=True)

    print("\nStarting the slower broad polar resonance scan.", flush=True)
    python_step(
        "broad_polar",
        CODES / "rbh_all_models_dynamic.py",
        "--workers", str(workers),
        "--output-dir", str(FIGURES),
        "--broad-only",
    )
    prepare_shared_broad_polar_data()
    python_step("hayward_fanwang_polar_resonance", hf_polar / "codes" / "resonance_test.py")
    prepare_bardeen_polar_peaks()
    bardeen_resonance = CODES / "metric_level" / "bardeen_polar_resonance"
    python_step("bardeen_polar_resonance", bardeen_resonance / "check_polar_ringdown.py")
    python_step("qnm_resonance_alignment", CODES / "compile_qnm_resonance_alignment.py")


def reuse_existing_results() -> None:
    python_step("plot_static", CODES / "plot_all_models_static_direct.py")
    python_step("scalar_shell_convergence", CODES / "scalar_shell_eft_convergence.py")
    python_step("bardeen_polar_charge_plot", CODES / "metric_level" / "bardeen_polar_charge" / "codes" / "plot_bardeen_polar_charge_scan.py")
    python_step("bardeen_polar_frequency_plot", CODES / "metric_level" / "bardeen_polar_frequency" / "codes" / "aggregate_bardeen_polar_frequency.py")
    python_step("hayward_fanwang_polar_plot", CODES / "metric_level" / "hayward_fanwang_polar" / "codes" / "aggregate_metric_scans.py")
    python_step("bardeen_axial_plot", CODES / "metric_level" / "bardeen_axial" / "codes" / "finalize_plots.py")
    python_step("hayward_fanwang_axial_plot", CODES / "metric_level" / "hayward_fanwang_axial" / "codes" / "plot_metric_scans.py")
    python_step("hayward_fanwang_axial_resonance_plot", CODES / "metric_level" / "hayward_fanwang_axial" / "codes" / "plot_resonance.py")
    python_step("dynamic_tln_convergence", CODES / "dynamic_tln_convergence.py")
    python_step("axial_master_convergence", CODES / "resonance_checks" / "axial_master_response_convergence.py")
    python_step("nearzone_amplitudes", CODES / "numerical_nearzone_asymptotics.py")
    python_step("qnm_resonance_alignment", CODES / "compile_qnm_resonance_alignment.py")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument(
        "--reuse-results",
        action="store_true",
        help="Rebuild plots/tables from archived numerical CSV/JSON files instead of rerunning ODE scans.",
    )
    parser.add_argument("--skip-validations", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue after already completed static, scalar-shell, and tensor-probe stages.",
    )
    args = parser.parse_args()
    dependency_check()
    initialize_output_directories()
    if not args.skip_validations and not args.resume:
        pre_calculation_validations()
    if args.reuse_results:
        reuse_existing_results()
    else:
        full_calculation(args.workers, resume=args.resume, skip_validations=args.skip_validations)
    copy_manuscript_figures()
    # Final manuscript figures use the direct regular-black-hole response definitions.
    python_step("scalar_shell_direct", CODES / "scalar_shell_eft_all_models_direct.py")
    python_step("metric_tln_direct_plots", CODES / "plot_direct_partII.py")
    export_table_inputs()
    python_step("release_manifest_verify", CODES / "release_manifest.py", "--verify")
    verify_outputs()


if __name__ == "__main__":
    main()
