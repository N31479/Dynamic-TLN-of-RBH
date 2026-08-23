#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import bardeen_polar_nearzone_dynamic_tln as calculation


def read_positive(path: Path) -> dict[float, complex]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {
        round(float(row["omega_M"]), 12): complex(
            float(row["k_dynamic_real"]), float(row["k_dynamic_imag"])
        )
        for row in rows
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    positive = read_positive(args.positive_table)
    with tempfile.TemporaryDirectory(prefix="bardeen_frequency_check_") as directory:
        temporary = Path(directory)
        settings = SimpleNamespace(
            mass=1.0,
            charge_ratio=args.charge_ratio,
            series_order=args.series_order,
            omega_order=args.omega_order,
            log_order=args.log_order,
            frequencies=[-abs(value) for value in args.frequencies],
            match_radii=args.match_radii,
            horizon_offset=args.horizon_offset,
            rtol=args.rtol,
            atol=args.atol,
            output_dir=temporary,
        )
        negative_rows = calculation.run(settings)

    checks: list[dict[str, object]] = []
    for row in negative_rows:
        omega = abs(float(row["omega_M"]))
        key = round(omega, 12)
        if key not in positive:
            raise RuntimeError(f"No positive-frequency reference for M omega={omega}")
        plus = positive[key]
        minus = complex(row["k_dynamic_real"], row["k_dynamic_imag"])
        checks.append(
            {
                "omega_M": omega,
                "k_plus": [plus.real, plus.imag],
                "k_minus": [minus.real, minus.imag],
                "absolute_conjugation_error": abs(minus - plus.conjugate()),
            }
        )

    result: dict[str, object] = {
        "relation": "K(-omega)=K(omega)^* for the real nonrotating background",
        "checks": checks,
        "maximum_absolute_conjugation_error": max(
            float(item["absolute_conjugation_error"]) for item in checks
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def parse_args() -> argparse.Namespace:
    package = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--positive-table",
        type=Path,
        default=package / "results" / "bardeen_polar_dynamic_tln.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=package / "results" / "bardeen_polar_frequency_symmetry_check.json",
    )
    parser.add_argument("--frequencies", nargs="+", type=float, default=[0.002, 0.004])
    parser.add_argument("--charge-ratio", type=float, default=0.5)
    parser.add_argument("--series-order", type=int, default=18)
    parser.add_argument("--omega-order", type=int, default=2)
    parser.add_argument("--log-order", type=int, default=5)
    parser.add_argument("--match-radii", nargs="+", type=float, default=[10, 12, 14, 16])
    parser.add_argument("--horizon-offset", type=float, default=2e-6)
    parser.add_argument("--rtol", type=float, default=5e-10)
    parser.add_argument("--atol", type=float, default=5e-12)
    return parser.parse_args()


if __name__ == "__main__":
    output = run(parse_args())
    print(json.dumps(output, indent=2))
