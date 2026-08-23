#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent


def keyed_rows(path: Path, *, reference: bool) -> dict[tuple[str, float, float], dict[str, str]]:
    rows = {}
    with path.open() as stream:
        for row in csv.DictReader(stream):
            if not reference and row["model"] != "bardeen":
                continue
            key = (
                row["parity"],
                round(float(row["ell_over_ell_ext"]), 12),
                round(float(row["omega_M"]), 12),
            )
            rows[key] = row
    return rows


def main() -> None:
    generated = keyed_rows(
        ROOT / "results" / "all_models_low_frequency.csv", reference=False
    )
    reference = keyed_rows(
        ROOT / "reference" / "bardeen_dynamic_vs_charge_verified.csv",
        reference=True,
    )
    if set(generated) != set(reference):
        missing = sorted(set(reference) - set(generated))
        extra = sorted(set(generated) - set(reference))
        raise AssertionError(f"Bardeen grid mismatch: missing={missing[:3]}, extra={extra[:3]}")

    errors = {
        "canonical_delta_real": [],
        "canonical_delta_imag": [],
        "displayed_response": [],
    }
    for key in sorted(reference):
        new = generated[key]
        old = reference[key]
        errors["canonical_delta_real"].append(
            abs(float(new["delta_Rgg_real"]) - float(old["delta_Rgg_real"]))
        )
        errors["canonical_delta_imag"].append(
            abs(float(new["delta_Rgg_imag"]) - float(old["delta_Rgg_imag"]))
        )
        if key[0] == "polar":
            errors["displayed_response"].append(
                abs(float(new["finite_real"]) - float(old["calibrated_real"]))
            )
        else:
            errors["displayed_response"].append(
                abs(
                    float(new["axial_normalized_real"])
                    - float(old["static_normalized_real"])
                )
            )

    report = {
        "dynamic_rows_compared": len(reference),
        "dynamic_maximum_absolute_errors": {
            name: float(np.max(values)) for name, values in errors.items()
        },
    }
    tolerance = 2.0e-10
    if max(report["dynamic_maximum_absolute_errors"].values()) > tolerance:
        raise AssertionError(json.dumps(report, indent=2))
    report["dynamic_tolerance"] = tolerance

    report["scalar_shell_validation"] = (
        "performed independently by check_scalar_shell_junction.py and "
        "scalar_shell_eft_convergence.py"
    )
    report["status"] = "passed"
    output = ROOT / "results" / "bardeen_consistency_validation.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
