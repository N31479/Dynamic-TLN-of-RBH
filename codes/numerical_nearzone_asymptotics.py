#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CODES = ROOT / "codes"
RESULTS = CODES / "results"


def read_rows(path: Path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def close(value: str, target: float, tol: float = 5.0e-12) -> bool:
    return abs(float(value) - target) <= tol


def extrapolate(radii, values, omega):
    r = np.asarray(radii, dtype=float)
    y = np.asarray(values, dtype=complex)
    design = np.column_stack([np.ones_like(r), (omega * r) ** 6, 1.0 / r**2])
    scales = np.linalg.norm(design, axis=0)
    real = np.linalg.lstsq(design / scales, y.real, rcond=1.0e-13)[0] / scales
    imag = np.linalg.lstsq(design / scales, y.imag, rcond=1.0e-13)[0] / scales
    return complex(real[0], imag[0])


def select(rows, *, model=None, ratio=None, omega=None):
    out = []
    for row in rows:
        if model is not None and row.get("model") != model:
            continue
        if ratio is not None and not close(row["ell_over_ell_ext"], ratio):
            continue
        if omega is not None and not close(row["omega_M"], omega):
            continue
        out.append(row)
    return out


def summarize_rows(rows, radius_key, omega):
    radii = [float(row[radius_key]) for row in rows]
    grav = [complex(float(row["response_real"]), float(row["response_imag"])) for row in rows]
    em = [complex(float(row["em_response_real"]), float(row["em_response_imag"])) for row in rows]
    return extrapolate(radii, grav, omega), extrapolate(radii, em, omega)


def bardeen_polar(ratio, omega):
    tag = f"{ratio:.2f}".replace(".", "p")
    path = CODES / "metric_level" / "bardeen_polar_charge" / "results" / "raw_scans" / f"ellratio_{tag}" / "bardeen_polar_window_data.csv"
    rows = [row for row in read_rows(path) if close(row["omega_M"], omega)]
    return summarize_rows(rows, "match_radius_over_M", omega)


def combined_table(path, model, ratio, omega):
    rows = select(read_rows(path), model=model, ratio=ratio, omega=omega)
    return summarize_rows(rows, "match_radius", omega)


def bardeen_axial(ratio, omega):
    path = CODES / "metric_level" / "bardeen_axial" / "results" / "bardeen_axial_metric_windows.csv"
    rows = select(read_rows(path), ratio=ratio, omega=omega)
    return summarize_rows(rows, "match_radius", omega)


def hayward_fanwang_polar(model, ratio, omega):
    path = CODES / "metric_level" / "hayward_fanwang_polar" / "results" / "metric_window_diagnostics.csv"
    return combined_table(path, model, ratio, omega)


def hayward_fanwang_axial(model, ratio, omega):
    path = CODES / "metric_level" / "hayward_fanwang_axial" / "results" / "hayward_fanwang_axial_metric_windows.csv"
    return combined_table(path, model, ratio, omega)


def build_table(ratios, frequencies):
    rows = []
    for parity in ("polar", "axial"):
        for model in ("bardeen", "hayward", "fan_wang"):
            for ratio in ratios:
                for omega in frequencies:
                    if model == "bardeen" and parity == "polar":
                        cg, ce = bardeen_polar(ratio, omega)
                    elif model == "bardeen":
                        cg, ce = bardeen_axial(ratio, omega)
                    elif parity == "polar":
                        cg, ce = hayward_fanwang_polar(model, ratio, omega)
                    else:
                        cg, ce = hayward_fanwang_axial(model, ratio, omega)
                    rows.append(
                        {
                            "model": model,
                            "parity": parity,
                            "ell_over_ell_ext": ratio,
                            "omega_M": omega,
                            "c_g_real": cg.real,
                            "c_g_imag": cg.imag,
                            "c_e_real": ce.real,
                            "c_e_imag": ce.imag,
                        }
                    )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.50, 0.80, 0.95])
    parser.add_argument("--frequencies", nargs="+", type=float, default=[0.002, 0.004])
    parser.add_argument("--output", type=Path, default=RESULTS / "numerical_nearzone_asymptotics.csv")
    args = parser.parse_args()
    rows = build_table(args.ratios, args.frequencies)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            f"{row['model']:9s} {row['parity']:5s} "
            f"q={row['ell_over_ell_ext']:.2f} Momega={row['omega_M']:.3f} "
            f"cg={row['c_g_real']:+.7e}{row['c_g_imag']:+.7e}i "
            f"ce={row['c_e_real']:+.7e}{row['c_e_imag']:+.7e}i"
        )


if __name__ == "__main__":
    main()
