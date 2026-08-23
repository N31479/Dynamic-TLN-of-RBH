#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
PATHS = {
    "bardeen_polar": HERE / "metric_level/bardeen_polar_frequency/codes",
    "bardeen_axial": HERE / "metric_level/bardeen_axial/codes",
    "other_polar": HERE / "metric_level/hayward_fanwang_polar/codes",
    "other_axial": HERE / "metric_level/hayward_fanwang_axial/codes",
}
for path in reversed(list(PATHS.values())):
    sys.path.insert(0, str(path))
sys.path.insert(0, str(HERE))

import einstein_nled_master as master
import bardeen_polar_dynamic_frobenius as bp_basis
import bardeen_polar_nearzone_dynamic_tln as bp_solver
import bardeen_axial_nearzone_basis as ba_basis
import bardeen_axial_metric_solver as ba_solver
import rbh_polar_nearzone_basis as op_basis
import rbh_polar_metric_solver as op_solver
import rbh_axial_nearzone_basis as oa_basis
import rbh_axial_metric_solver as oa_solver


PMIN, PMAX = -14, 16
DEFAULT_RATIOS = [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.95]
DEFAULT_MATCH_RADII = [10.0, 12.0, 14.0, 16.0]
DEFAULT_HORIZON_OFFSET = 2.0e-6
DEFAULT_RTOL = 5.0e-10
DEFAULT_ATOL = 5.0e-12


def _clean(series):
    return {p: np.asarray(v, complex) for p, v in series.items()
            if PMIN <= p <= PMAX and np.any(np.abs(v) > 1.0e-30)}


def _add(a, b):
    out = {p: v.copy() for p, v in a.items()}
    for p, value in b.items():
        if p in out:
            width = max(len(out[p]), len(value))
            merged = np.zeros(width, complex)
            merged[:len(out[p])] += out[p]
            merged[:len(value)] += value
            out[p] = merged
        else:
            out[p] = value.copy()
    return _clean(out)


def _scale(a, value):
    return _clean({p: value * coefficients for p, coefficients in a.items()})


def _shift(a, power):
    return _clean({p + power: value.copy() for p, value in a.items()})


def _mul(a, b):
    out = {}
    for pa, va in a.items():
        for pb, vb in b.items():
            power = pa + pb
            if not PMIN <= power <= PMAX:
                continue
            value = np.convolve(va, vb)
            if power in out:
                width = max(len(out[power]), len(value))
                merged = np.zeros(width, complex)
                merged[:len(out[power])] += out[power]
                merged[:len(value)] += value
                out[power] = merged
            else:
                out[power] = value
    return _clean(out)


def _inverse(a):
    if any(np.any(np.abs(v[1:]) > 1.0e-30) for v in a.values()):
        raise ValueError("Only log-free background series may be inverted")
    p0 = min(a)
    a0 = a[p0][0]
    coefficients = {p - p0: v[0] for p, v in a.items()}
    inverse = {0: 1.0 / a0}
    for n in range(1, PMAX - PMIN + 1):
        inverse[n] = -sum(
            coefficients.get(k, 0.0) * inverse.get(n - k, 0.0)
            for k in range(1, n + 1)
        ) / a0
    return _clean({n - p0: np.array([value], complex) for n, value in inverse.items()})


def _radial_derivative(a):
    out = {}
    for power, polynomial in a.items():
        value = np.zeros(len(polynomial), complex)
        value += -0.5 * power * polynomial
        if len(polynomial) > 1:
            value[:-1] += -0.5 * np.arange(1, len(polynomial)) * polynomial[1:]
        out[power + 2] = value
    return _clean(out)


def _binomial(amplitude, step, exponent):
    out = {}
    coefficient = 1.0
    for n in range((PMAX - PMIN) // step + 2):
        power = n * step
        if power > PMAX:
            break
        if n:
            coefficient *= (exponent - n + 1) / n
        out[power] = np.array([coefficient * amplitude**n], complex)
    return out


def _background_series(model_name: str, mass: float, charge: float):
    if model_name == "bardeen":
        mass_function = _scale(_binomial(charge**2, 4, -1.5), mass)
        lagrangian = _shift(_scale(_binomial(charge**2, 4, -2.5), 3 * mass * charge**2), 10)
        sqrt_lf = _shift(_scale(_binomial(charge**2, 4, -1.75), np.sqrt(7.5 * mass)), 1)
    elif model_name == "hayward":
        amplitude = 2 * mass * charge**2
        mass_function = _scale(_binomial(amplitude, 6, -1.0), mass)
        lagrangian = _shift(_scale(_binomial(amplitude, 6, -2.0), 6 * mass**2 * charge**2), 12)
        sqrt_lf = _shift(_scale(_binomial(amplitude, 6, -1.5), 3 * np.sqrt(2) * mass), 2)
    elif model_name == "fan_wang":
        mass_function = _scale(_binomial(charge, 2, -3.0), mass)
        lagrangian = _shift(_scale(_binomial(charge, 2, -4.0), 3 * mass * charge), 8)
        sqrt_lf = _scale(_binomial(charge, 2, -2.5), np.sqrt(6 * mass / charge))
    else:
        raise ValueError(model_name)
    metric = _add({0: np.array([1.0], complex)}, _scale(_shift(mass_function, 2), -2.0))
    return mass_function, lagrangian, sqrt_lf, metric


def _field_series(mode, component: int, maximum_power: int):
    exponent, coefficients = mode
    exponent = float(exponent)
    if abs(exponent - round(exponent)) >= 1.0e-10:
        raise ValueError(f"Noninteger asymptotic exponent {exponent}")
    exponent = int(round(exponent))
    result = {}
    omega_zero = coefficients[0]
    for radial_order in range(omega_zero.shape[0]):
        if float(exponent) + radial_order > maximum_power:
            break
        for log_order in range(omega_zero.shape[1]):
            value = complex(omega_zero[radial_order, log_order, component])
            if value:
                power = exponent + radial_order
                if power not in result:
                    result[power] = np.zeros(omega_zero.shape[1], complex)
                result[power][log_order] += value
    return _clean(result)


def _polar_metric_series(
    model_name: str,
    mass: float,
    charge: float,
    psi,
    phi,
):
    mass_function, lagrangian, sqrt_lf, metric = _background_series(model_name, mass, charge)
    lf = _mul(sqrt_lf, sqrt_lf)
    auxiliary = _add(_scale(_shift(mass_function, 2), 6.0), _scale(_shift(lagrangian, -4), -2.0))
    denominator = _add(auxiliary, {0: np.array([4.0], complex)})
    zeta = _scale(_mul(denominator, psi), 0.5)
    electromagnetic = _add(_scale(_mul(phi, _inverse(sqrt_lf)), 0.5), _scale(_shift(psi, 2), 0.5 * charge))
    numerator = _add(
        _add(_scale(_shift(_mul(_mul(metric, lf), electromagnetic), 2), 8 * charge),
             _scale(_shift(_mul(metric, _radial_derivative(zeta)), -2), -2.0)),
        _scale(zeta, -6.0),
    )
    K = _mul(numerator, _inverse(_shift(denominator, -2)))
    H0 = _add(
        _add(_scale(_radial_derivative(zeta), -1.0), _scale(_shift(_radial_derivative(K), -2), -1.0)),
        _scale(_shift(_mul(lf, electromagnetic), 4), 4 * charge),
    )
    return _scale(_mul(metric, H0), -1.0)


def _axial_metric_series(
    model_name: str,
    mass: float,
    charge: float,
    psi,
):
    _, _, _, metric = _background_series(model_name, mass, charge)
    return _scale(_mul(metric, _add(psi, _shift(_radial_derivative(psi), -2))), 1 / np.sqrt(2))


def _electromagnetic_series(model_name, parity, charge, psi, phi):
    _, _, sqrt_lf, _ = _background_series(model_name, 1.0, charge)
    first = _mul(phi, _inverse(sqrt_lf))
    if parity == "polar":
        return _add(_scale(first, 0.5), _scale(_shift(psi, 2), 0.5 * charge))
    return _scale(first, 0.25)


def _exact_degenerate_transform(model_name, parity, charge, modes):
    if parity == "polar":
        grav_source_power, grav_response_power = -4, 6
        em_source_power, em_response_power = -6, 4
    else:
        grav_source_power, grav_response_power = -6, 4
        em_source_power, em_response_power = -6, 4
    source = np.zeros((2, 2), complex)
    response = np.zeros((2, 2), complex)
    for branch, matrix, offset, gp, ep in (
        ("source", source, 0, grav_source_power, em_source_power),
        ("response", response, 2, grav_response_power, em_response_power),
    ):
        for column in range(2):
            mode = modes[offset + column]
            psi = _field_series(mode, 0, 12)
            phi = _field_series(mode, 1, 12)
            if parity == "polar":
                grav = _polar_metric_series(model_name, 1.0, charge, psi, phi)
            else:
                grav = _axial_metric_series(model_name, 1.0, charge, psi)
            electromagnetic = _electromagnetic_series(
                model_name, parity, charge, psi, phi
            )
            matrix[0, column] = grav.get(gp, np.zeros(1))[0]
            matrix[1, column] = electromagnetic.get(ep, np.zeros(1))[0]
    return np.linalg.inv(source), np.linalg.inv(response)


def _sector(model_name: str, parity: str, charge: float, order: int, logs: int):
    if model_name == "bardeen" and parity == "polar":
        modes = bp_basis.build_modes(charge, order, 2, logs)
        model = master.build_model(model_name, charge, 1.0)
        transform = (np.eye(2, dtype=complex), np.eye(2, dtype=complex))
        return model, modes, transform, bp_solver
    if model_name == "bardeen":
        model = ba_solver.build_model(charge, 1.0)
        modes = ba_basis.build_modes(charge, order, 2, logs, 1.0)
        Ts, Tr, _, _ = ba_solver.physical_transform(model, modes)
        return model, modes, (Ts, Tr), ba_solver
    if parity == "polar":
        model = master.build_model(model_name, charge, 1.0)
        modes = op_basis.build_modes(model_name, charge, order, 2, logs, 1.0)
        Ts, Tr, _, _ = op_solver.physical_transform(model, modes)
        return model, modes, (Ts, Tr), op_solver
    model = oa_solver.build_model(model_name, charge, 1.0)
    modes = oa_basis.build_modes(model_name, charge, order, 2, logs, 1.0)
    Ts, Tr, _, _ = oa_solver.physical_transform(model, modes)
    return model, modes, (Ts, Tr), oa_solver


def _metric_coefficients(
    model_name: str,
    parity: str,
    charge: float,
    modes,
    transform,
):
    Ts, Tr = transform
    if parity == "polar":
        source_power, response_power = -4, 6
        maximum_power = response_power + 6
    else:
        source_power, response_power = -6, 4
        maximum_power = response_power + 6
    source_psi = {}
    response_psi = {}
    for j in range(2):
        if abs(Ts[j, 0]) > 1.0e-14:
            source_psi = _add(
                source_psi,
                _scale(_field_series(modes[j], 0, maximum_power), Ts[j, 0]),
            )
        if abs(Tr[j, 0]) > 1.0e-14:
            response_psi = _add(
                response_psi,
                _scale(_field_series(modes[j + 2], 0, maximum_power), Tr[j, 0]),
            )
    if parity == "polar":
        source_phi = {}
        response_phi = {}
        for j in range(2):
            if abs(Ts[j, 0]) > 1.0e-14:
                source_phi = _add(
                    source_phi,
                    _scale(_field_series(modes[j], 1, maximum_power), Ts[j, 0]),
                )
            if abs(Tr[j, 0]) > 1.0e-14:
                response_phi = _add(
                    response_phi,
                    _scale(_field_series(modes[j + 2], 1, maximum_power), Tr[j, 0]),
                )
        source_metric = _polar_metric_series(model_name, 1.0, charge, source_psi, source_phi)
        response_metric = _polar_metric_series(
            model_name, 1.0, charge, response_psi, response_phi
        )
    else:
        source_metric = _axial_metric_series(model_name, 1.0, charge, source_psi)
        response_metric = _axial_metric_series(model_name, 1.0, charge, response_psi)
    source_norm = float(np.real(source_metric[source_power][0]))
    response_norm = float(np.real(response_metric[response_power][0]))
    response_polynomial = source_metric.get(response_power, np.zeros(1, complex))
    log_value = -0.5 * np.log(2.0)
    local_response = float(np.real(sum(
        coefficient * log_value**order
        for order, coefficient in enumerate(response_polynomial)
    ))) / source_norm
    return source_norm, response_norm, local_response


def _horizon_term(
    model,
    modes,
    transform,
    solver,
    radii,
    horizon_offset=DEFAULT_HORIZON_OFFSET,
    rtol=DEFAULT_RTOL,
    atol=DEFAULT_ATOL,
):
    controls = {"horizon_offset": horizon_offset, "rtol": rtol, "atol": atol}
    Ts, Tr = transform
    values = []
    for radius in radii:
        horizon_value, horizon_derivative = solver.horizon_basis(
            model, 0.0, radius, **controls
        )
        value = np.zeros((2, 4), complex)
        derivative = np.zeros((2, 4), complex)
        for column, (exponent, coefficients) in enumerate(modes):
            if solver is bp_solver:
                v, d = bp_basis.evaluate_dynamic_mode(
                    coefficients, exponent, 0.0, np.array([radius])
                )
            elif solver is ba_solver:
                v, d = ba_basis.evaluate(coefficients, exponent, 0.0, [radius])
            elif solver is op_solver:
                v, d = op_basis.evaluate(coefficients, exponent, 0.0, [radius])
            else:
                v, d = oa_basis.evaluate(coefficients, exponent, 0.0, [radius])
            value[:, column] = v[0]
            derivative[:, column] = d[0]
        value = np.column_stack((value[:, :2] @ Ts, value[:, 2:] @ Tr))
        derivative = np.column_stack(
            (derivative[:, :2] @ Ts, derivative[:, 2:] @ Tr)
        )
        asymptotic = np.vstack((value, derivative))
        horizon = np.vstack((horizon_value, horizon_derivative))
        scales = np.linalg.norm(asymptotic, axis=0)
        coefficients = np.linalg.solve(asymptotic / scales, horizon) / scales[:, None]
        weights = np.linalg.solve(coefficients[:2], np.array([1.0, 0.0], complex))
        amplitudes = coefficients @ weights
        values.append(amplitudes[2])
    values = np.asarray(values)
    x = np.asarray(radii, float)
    design = np.column_stack((np.ones_like(x), x**-2, x**-4))
    fit = np.linalg.lstsq(design, np.real(values), rcond=None)[0]
    prediction = design @ fit
    return float(fit[0]), float(np.sqrt(np.mean((np.real(values) - prediction) ** 2)))


def derive_one(
    model_name: str,
    parity: str,
    ratio: float,
    order: int,
    logs: int,
    match_radii=DEFAULT_MATCH_RADII,
    horizon_offset=DEFAULT_HORIZON_OFFSET,
    rtol=DEFAULT_RTOL,
    atol=DEFAULT_ATOL,
):
    charge = ratio * master.extremal_charge(model_name, 1.0)
    model, modes, transform, solver = _sector(model_name, parity, charge, order, logs)
    if model_name == "fan_wang":
        transform = _exact_degenerate_transform(model_name, parity, charge, modes)
    source_norm, response_norm, local = _metric_coefficients(
        model_name, parity, charge, modes, transform
    )
    Ts, Tr = (transform[0].copy(), transform[1].copy())
    Ts[:, 0] /= source_norm
    Tr[:, 0] /= response_norm
    horizon, rms = _horizon_term(
        model,
        modes,
        (Ts, Tr),
        solver,
        match_radii,
        horizon_offset=horizon_offset,
        rtol=rtol,
        atol=atol,
    )
    metric_ratio = horizon + local
    love = (-metric_ratio if parity == "polar" else metric_ratio)
    return {
        "model": model_name,
        "parity": parity,
        "ell_over_ell_ext": ratio,
        "ell_over_M": charge,
        "k2_static_direct": love,
        "metric_response_ratio": metric_ratio,
        "canonical_local_term": local,
        "horizon_response_term": horizon,
        "window_fit_rms": rms,
        "series_order": order,
        "log_order": logs,
        "match_radii_over_M": ";".join(f"{radius:g}" for radius in match_radii),
        "horizon_offset_over_M": horizon_offset,
        "relative_tolerance": rtol,
        "absolute_tolerance": atol,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=["bardeen", "hayward", "fan_wang"], default=["bardeen", "hayward", "fan_wang"])
    parser.add_argument("--parities", nargs="+", choices=["polar", "axial"], default=["polar", "axial"])
    parser.add_argument("--ratios", nargs="+", type=float, default=DEFAULT_RATIOS)
    parser.add_argument("--series-order", type=int, default=22)
    parser.add_argument("--log-order", type=int, default=6)
    parser.add_argument("--match-radii", nargs="+", type=float, default=DEFAULT_MATCH_RADII)
    parser.add_argument("--horizon-offset", type=float, default=DEFAULT_HORIZON_OFFSET)
    parser.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    parser.add_argument("--output", type=Path, default=HERE / "results/all_models_static_direct.csv")
    args = parser.parse_args()
    rows = [derive_one(
                model,
                parity,
                ratio,
                args.series_order,
                args.log_order,
                match_radii=args.match_radii,
                horizon_offset=args.horizon_offset,
                rtol=args.rtol,
                atol=args.atol,
            )
            for model in args.models for parity in args.parities for ratio in args.ratios]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
