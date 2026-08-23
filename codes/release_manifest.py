#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODES = ROOT / "codes"
FIGURES = ROOT / "figures"
TABLES = ROOT / "manuscript_tables"
MANIFEST = ROOT / "release_manifest.json"

TABLE_FILES = (
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

FIGURE_PRODUCERS = {
    "scalar_shell_eft_all_models.pdf": (
        "codes/scalar_shell_eft_all_models_direct.py",
        ("codes/results/scalar_shell_eft_all_models_direct.csv",),
        "same-background scalar source subtraction",
    ),
    "all_models_static_direct.pdf": (
        "codes/plot_all_models_static_direct.py",
        ("codes/results/all_models_static_direct.csv",),
        "direct static regular-black-hole response",
    ),
    "tensor_probe_all_models.pdf": (
        "codes/tensor_probe_response.py",
        ("codes/results/tensor_probe_response.csv",),
        "direct tensor-probe response with independently reproduced static limit",
    ),
}

for model, label in (
    ("bardeen", "Bardeen"),
    ("hayward", "Hayward"),
    ("fan_wang", "Fan-Wang"),
):
    for parity in ("polar", "axial"):
        data = f"codes/results/direct_metric/{model}_{parity}_unsubtracted.csv"
        for variable in ("ell_ratio", "frequency"):
            name = f"{model}_{parity}_metric_dynamic_tln_vs_{variable}.pdf"
            FIGURE_PRODUCERS[name] = (
                "codes/plot_direct_partII.py",
                (data,),
                "unsubtracted direct metric source-response ratio",
            )
        suffix = ".png" if model == "bardeen" and parity == "polar" else ".pdf"
        resonance = f"{model}_{parity}_resonance_test{suffix}"
        FIGURE_PRODUCERS[resonance] = (
            "run_resonance.py",
            (
                "manuscript_tables/qnm_resonance_alignment.csv",
                "manuscript_tables/qnm_resonance_alignment.json",
            ),
            f"{label} {parity} real-axis response peak compared with the QNM damping width",
        )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def table_cells(path: Path):
    if path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.reader(stream)
            return list(reader)
    return json.loads(path.read_text(encoding="utf-8"))


def file_record(relative: str) -> dict:
    path = ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": relative, "sha256": sha256(path), "size_bytes": path.stat().st_size}


def parameter_grid(relative: str) -> dict:
    path = ROOT / relative
    if path.suffix != ".csv":
        return {}
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    keys = (
        "model", "parity", "ell_over_ell_ext", "omega_M", "Momega",
        "strict_controls", "series_order", "omega_order", "log_order",
    )
    output = {}
    for key in keys:
        values = sorted({row[key] for row in rows if key in row}, key=str)
        if values:
            output[key] = values
    return output


def build_manifest() -> dict:
    tables = {}
    for name in TABLE_FILES:
        path = TABLES / name
        tables[name] = {
            "sha256": sha256(path),
            "cells": table_cells(path),
        }

    figures = {}
    for name, (producer, inputs, convention) in sorted(FIGURE_PRODUCERS.items()):
        output = FIGURES / name
        figures[name] = {
            "command": "python3 run_all.py",
            "stage_script": producer,
            "code_sha256": {
                "run_all.py": sha256(ROOT / "run_all.py"),
                producer: sha256(ROOT / producer),
                "codes/einstein_nled_master.py": sha256(
                    CODES / "einstein_nled_master.py"
                ),
            },
            "parameters": {
                "source": "explicit run_all.py arguments and stage-script constants",
                "archived_input_grids": {
                    item: parameter_grid(item) for item in inputs if parameter_grid(item)
                },
            },
            "response_convention": convention,
            "inputs": [file_record(item) for item in inputs],
            "output": file_record(f"figures/{name}"),
        }

    scalar_csv = CODES / "results" / "scalar_shell_eft_all_models_direct.csv"
    with scalar_csv.open(newline="", encoding="utf-8") as stream:
        scalar_rows = list(csv.DictReader(stream))
    model_counts = {
        model: sum(row["model"] == model for row in scalar_rows)
        for model in ("bardeen", "hayward", "fan_wang")
    }

    return {
        "canonical_command": "python3 run_all.py",
        "response_conventions": {
            "dynamical_metric_tln": "unsubtracted direct regular-black-hole source-response ratio",
            "scalar_shell_eft": "same-background source subtraction only",
            "relative_peak_response_change": "|R_strict-R_standard|/|R_strict| at the standard-control response-magnitude peak",
        },
        "canonical_solver": file_record("codes/einstein_nled_master.py"),
        "requirements": file_record("requirements.txt"),
        "scalar_direct_output": {
            "path": "codes/results/scalar_shell_eft_all_models_direct.csv",
            "row_count": len(scalar_rows),
            "model_counts": model_counts,
            "sha256": sha256(scalar_csv),
        },
        "tables": tables,
        "figures": figures,
    }


def verify_manifest(manifest: dict) -> None:
    expected_solver = manifest["canonical_solver"]
    if sha256(ROOT / expected_solver["path"]) != expected_solver["sha256"]:
        raise RuntimeError("Canonical solver hash differs from the release manifest")

    for name, expected in manifest["tables"].items():
        path = TABLES / name
        actual_cells = table_cells(path)
        if actual_cells != expected["cells"]:
            raise RuntimeError(f"Printed-cell source differs from the release manifest: {name}")
        if sha256(path) != expected["sha256"]:
            raise RuntimeError(f"Table-file hash differs from the release manifest: {name}")

    scalar = manifest["scalar_direct_output"]
    path = ROOT / scalar["path"]
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    counts = {
        model: sum(row["model"] == model for row in rows)
        for model in ("bardeen", "hayward", "fan_wang")
    }
    if len(rows) != 81 or counts != {model: 27 for model in counts}:
        raise RuntimeError("Direct scalar output must contain 81 rows: 27 for each model")
    if sha256(path) != scalar["sha256"]:
        raise RuntimeError("Direct scalar output hash differs from the release manifest")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.write == args.verify:
        raise SystemExit("Choose exactly one of --write or --verify")
    if args.write:
        MANIFEST.write_text(json.dumps(build_manifest(), indent=2) + "\n", encoding="utf-8")
        print(MANIFEST)
    else:
        verify_manifest(json.loads(MANIFEST.read_text(encoding="utf-8")))
        print("Release manifest verification passed")


if __name__ == "__main__":
    main()
