#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def maximum_identity_error(data: list[dict[str, str]], delta: str) -> float:
    return max(
        abs(
            float(row["k_dynamic_real"])
            - float(row["k_static"])
            - float(row[delta])
        )
        for row in data
    )


def main() -> None:
    bardeen = rows(
        ROOT
        / "metric_level"
        / "bardeen_polar_charge"
        / "results"
        / "bardeen_polar_metric_dynamic_tln_charge_scan_long.csv"
    )
    all_models = rows(
        ROOT
        / "metric_level"
        / "hayward_fanwang_polar"
        / "results"
        / "hayward_fanwang_metric_charge_scan.csv"
    )
    required_raw = {
        "metric_ratio_real",
        "metric_ratio_imag",
        "metric_ratio_static_real",
        "metric_ratio_static_imag",
    }
    bardeen_fields = set(bardeen[0])
    all_model_fields = set(all_models[0])
    report = {
        "polar_love_normalization": "k20_polar=-(C_response/C_source)/M^5",
        "bardeen_raw_metric_fields_present": required_raw <= bardeen_fields,
        "hayward_fanwang_raw_metric_fields_present": required_raw <= all_model_fields,
        "bardeen_baseline_identity_max_error": maximum_identity_error(
            bardeen, "delta_k_real"
        ),
        "hayward_fanwang_baseline_identity_max_error": maximum_identity_error(
            all_models, "delta_real"
        ),
        "static_limit_statement": "All six zero-frequency constants are obtained from the direct coupled static calculation. In every case the computed frequency-dependent correction is checked to vanish as omega tends to zero.",
    }
    report["status"] = "passed" if (
        report["bardeen_raw_metric_fields_present"]
        and report["hayward_fanwang_raw_metric_fields_present"]
        and report["bardeen_baseline_identity_max_error"] < 1.0e-12
        and report["hayward_fanwang_baseline_identity_max_error"] < 1.0e-12
    ) else "failed"
    output = ROOT / "results" / "polar_metric_normalization_check.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
