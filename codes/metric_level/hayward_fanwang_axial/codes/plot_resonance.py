#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)


def read_numeric_csv(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key, value in list(row.items()):
            try:
                row[key] = float(value)
            except ValueError:
                pass
    return rows


def plot_model(model: str, broad_rows: list[dict], ringdown_rows: list[dict]) -> None:
    ratios = (0.60, 0.90, 0.97)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(ratios)))
    model_ringdown = sorted(
        (row for row in ringdown_rows if row["model"] == model),
        key=lambda row: row["ell_over_ell_ext"],
    )

    figure, axes = plt.subplots(1, 2, figsize=(10.6, 4.4))
    for color, ratio in zip(colors, ratios):
        case = sorted(
            (
                row for row in broad_rows
                if row["model"] == model
                and abs(row["ell_over_ell_ext"] - ratio) < 1.0e-12
                and int(row["strict_controls"]) == 0
            ),
            key=lambda row: row["omega_M"],
        )
        axes[0].plot(
            [row["omega_M"] for row in case],
            [row["delta_Rgg_abs"] for row in case],
            "o-",
            color=color,
            markersize=2.8,
            linewidth=1.2,
            label=rf"$\ell/\ell_{{\rm ext}}={ratio:.2f}$",
        )
        peak = next(
            row["real_axis_peak_M"]
            for row in model_ringdown
            if abs(row["ell_over_ell_ext"] - ratio) < 1.0e-12
        )
        axes[0].axvline(peak, color=color, alpha=0.25, linewidth=0.9)

    axes[0].set(
        xlabel=r"$M\omega$",
        ylabel=r"$|\mathcal{R}_{gg}^{\rm axial}|$",
        title="Broad canonical axial response",
    )
    axes[0].grid(alpha=0.22)
    axes[0].legend(frameon=False, fontsize=8)

    positions = np.arange(len(ratios))
    peaks = np.array([row["real_axis_peak_M"] for row in model_ringdown])
    modes = np.array([row["ringdown_frequency_M"] for row in model_ringdown])
    errors = np.array([row["maximum_control_shift"] for row in model_ringdown])
    axes[1].plot(positions, peaks, "D-", label="real-axis maximum")
    axes[1].errorbar(
        positions,
        modes,
        yerr=errors,
        fmt="o-",
        capsize=3,
        label="axial ringdown frequency",
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
    stem = FIGURES / f"{model}_axial_resonance_test"
    figure.savefig(stem.with_suffix(".png"), dpi=300)
    figure.savefig(stem.with_suffix(".pdf"))
    plt.close(figure)


def main() -> None:
    broad_rows = read_numeric_csv(RESULTS / "hayward_fanwang_axial_broad_response.csv")
    ringdown_rows = read_numeric_csv(RESULTS / "hayward_fanwang_axial_ringdown.csv")
    for model in ("hayward", "fan_wang"):
        plot_model(model, broad_rows, ringdown_rows)


if __name__ == "__main__":
    main()
