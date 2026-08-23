#!/usr/bin/env python3


from __future__ import annotations

import json
from pathlib import Path

import sympy as sp


def main() -> None:
    r, M, lam = sp.symbols("r M lambda", positive=True)
    f = 1 - 2 * M / r
    angular = lam + 2

    axial_entry = angular / r**2 - 6 * M / r**3
    regge_wheeler_over_f = angular / r**2 - 6 * M / r**3

    a = 6 * M / r
    b = lam
    polar_entry = (
        angular * lam - 2 * f * lam + a * (a - 4 * M / r)
    ) / (r**2 * (a + lam)) + 2 * f * lam * b / (
        r**2 * (a + lam) ** 2
    )
    zerilli_over_f = (
        lam**2 * (lam + 2) * r**3
        + 6 * lam**2 * M * r**2
        + 36 * lam * M**2 * r
        + 72 * M**3
    ) / (r**3 * (lam * r + 6 * M) ** 2)

    n = lam / 2
    zerilli_standard = 2 * f * (
        n**2 * (n + 1) * r**3
        + 3 * n**2 * M * r**2
        + 9 * n * M**2 * r
        + 9 * M**3
    ) / (r**3 * (n * r + 3 * M) ** 2)

    checks = {
        "regge_wheeler_entry_exact": sp.simplify(
            axial_entry - regge_wheeler_over_f
        ) == 0,
        "zerilli_entry_exact": sp.factor(polar_entry - zerilli_over_f) == 0,
        "zerilli_standard_form_exact": sp.factor(
            f * zerilli_over_f - zerilli_standard
        ) == 0,
    }
    if not all(checks.values()):
        raise AssertionError(checks)

    output = {**checks, "status": "passed"}
    result_path = Path(__file__).resolve().parent / "results" / (
        "schwarzschild_reduction_exact.json"
    )
    result_path.parent.mkdir(exist_ok=True)
    result_path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
