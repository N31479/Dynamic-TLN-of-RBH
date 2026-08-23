#!/usr/bin/env python3
''

from __future__ import annotations

import csv
from pathlib import Path

from run_metric_scans import plot


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def load_numeric_csv(path: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            rows.append(
                {
                    key: value if key == "model" else float(value)
                    for key, value in row.items()
                }
            )
    return rows


def main() -> None:
    charge_rows = load_numeric_csv(
        RESULTS / "hayward_fanwang_metric_charge_scan.csv"
    )
    frequency_rows = load_numeric_csv(
        RESULTS / "hayward_fanwang_metric_frequency_scan.csv"
    )
    plot(charge_rows, frequency_rows)


if __name__ == "__main__":
    main()
