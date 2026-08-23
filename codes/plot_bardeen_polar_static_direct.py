#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
data = np.genfromtxt(
    HERE / "results/bardeen_polar_static_direct.csv",
    delimiter=",",
    names=True,
)
ratio = data["ell_over_ell_ext"]
ell = data["ell"]
direct = data["k2_polar_static_direct"]

# The two smallest points are excluded from the finite polynomial fit because
# the response is obtained by subtracting two O(1) terms there.
mask = ratio >= 0.15
design = np.column_stack([ell[mask] ** (2 * n) for n in range(1, 5)])
coefficients = np.linalg.lstsq(design, direct[mask], rcond=None)[0]

x = np.linspace(0.0, 1.0, 500)
qext = 4.0 / (3.0 * np.sqrt(3.0))
q = x * qext
coviello = 0.03 * q**2 + 10.0 * q**4
higher = sum(coefficients[n - 1] * q ** (2 * n) for n in range(1, 5))

fig, ax = plt.subplots(figsize=(6.4, 4.3))
ax.plot(x, coviello, "--", color="#d55e00", lw=2.0, label="Coviello two-term fit")
ax.plot(x, higher, color="#0072b2", lw=2.0, label="Direct higher-order fit")
ax.scatter(ratio, direct, color="#0072b2", edgecolor="white", linewidth=0.6,
           s=34, zorder=3, label="Direct static calculation")
ax.set_xlabel(r"$\ell_B/\ell_{\rm ext}$")
ax.set_ylabel(r"$k_{2,+}^{B,\rm static}$")
ax.set_xlim(0.0, 1.0)
ax.set_ylim(bottom=0.0)
ax.legend(frameon=False)
ax.grid(alpha=0.18)
fig.tight_layout()
for suffix in ("png", "pdf"):
    fig.savefig(HERE.parent / "figures" / f"bardeen_polar_static_direct.{suffix}", dpi=240)

summary = {
    "fit_basis": ["ell^2", "ell^4", "ell^6", "ell^8"],
    "coefficients_M_equals_1": coefficients.tolist(),
    "fit_rms": float(np.sqrt(np.mean((design @ coefficients - direct[mask]) ** 2))),
}
(HERE / "results/bardeen_polar_static_fit.json").write_text(
    __import__("json").dumps(summary, indent=2) + "\n"
)
