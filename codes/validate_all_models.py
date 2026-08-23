#!/usr/bin/env python3


from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import einstein_nled_master as master


MODELS = ("bardeen", "hayward", "fan_wang")


def relative_error(value: np.ndarray, reference: np.ndarray) -> float:
    scale = np.maximum(np.abs(reference), 1.0e-30)
    return float(np.max(np.abs(value - reference) / scale))


def validate_model(model_name: str) -> dict[str, float]:
    charge = 0.73 * master.extremal_charge(model_name, 1.0)
    model = master.build_model(model_name, charge)
    horizon = master.outer_horizon(model)
    radii = np.geomspace(1.02 * horizon, 80.0, 300)
    step = 1.0e-3 * radii
    mass_derivative = (
        -model.mass_function(radii + 2.0 * step)
        + 8.0 * model.mass_function(radii + step)
        - 8.0 * model.mass_function(radii - step)
        + model.mass_function(radii - 2.0 * step)
    ) / (12.0 * step)
    identity_error = relative_error(mass_derivative, radii**2 * model.lagrangian(radii))

    symmetry = 0.0
    for parity in ("polar", "axial"):
        potential = master.potential_matrix(model, parity, radii)
        symmetry = max(symmetry, float(np.max(np.abs(potential - potential.swapaxes(1, 2)))))

    small_charge = 1.0e-7 * master.extremal_charge(model_name, 1.0)
    near_schwarzschild = master.build_model(model_name, small_charge)
    check_radii = np.geomspace(2.05, 50.0, 200)
    axial = master.potential_matrix(near_schwarzschild, "axial", check_radii)[:, 0, 0]
    polar = master.potential_matrix(near_schwarzschild, "polar", check_radii)[:, 0, 0]
    axial_reference = master.schwarzschild_potential("axial", check_radii, 1.0)
    polar_reference = master.schwarzschild_potential("polar", check_radii, 1.0)
    return {
        "background_identity_max_relative_error": identity_error,
        "potential_symmetry_max_absolute_error": symmetry,
        "regge_wheeler_limit_max_relative_error": relative_error(axial, axial_reference),
        "zerilli_limit_max_relative_error": relative_error(polar, polar_reference),
    }


def main() -> None:
    output = {model_name: validate_model(model_name) for model_name in MODELS}
    path = Path(__file__).resolve().parent / "results" / "structural_validation.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
