#!/usr/bin/env python3

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import einstein_nled_master as master
import scalar_shell_eft_all_models as scalar


def pair(z):
    return [float(z.real), float(z.imag)]


def run():
    model = 'bardeen'
    ratio = 0.70
    omega = 1e-4
    charge = ratio * master.extremal_charge(model, scalar.MASS)
    radii = np.array([12., 18., 24., 30., 40., 60., 80.])

    bundle = scalar.solution_bundle(model, charge, omega, radii)
    full_jump = scalar.derivative_jump_response(
        model, charge, omega, radii, bundle['full'], bundle['full_derivative']
    )
    source_jump = scalar.derivative_jump_response(
        model, charge, omega, radii, bundle['source'], bundle['source_derivative']
    )
    manual_same_background = (3.0 / (4.0 * np.pi)) * (full_jump - source_jump)
    renormalized = (3.0 / (4.0 * np.pi)) * scalar.renormalized_profile_from_bundle(
        model, charge, bundle
    )

    difference = renormalized - manual_same_background
    scale = max(float(np.sqrt(np.mean(np.abs(renormalized) ** 2))), 1e-30)
    relative_rms = float(np.sqrt(np.mean(np.abs(difference) ** 2)) / scale)

    report = {
        'model': model,
        'ell_over_ell_ext': ratio,
        'omega_M': omega,
        'radii_M': radii.tolist(),
        'definition': 'Same-background shell source subtraction: full shell jump minus source-only shell jump, both evaluated on the identical regular-black-hole geometry.',
        'renormalized_same_background_profile': [pair(z) for z in renormalized],
        'manual_full_minus_source_profile': [pair(z) for z in manual_same_background],
        'relative_rms_difference': relative_rms,
    }
    out = Path(__file__).resolve().parent / 'results' / 'scalar_shell_source_subtraction_check.json'
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    run()
