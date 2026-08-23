#!/usr/bin/env python3
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'results' / 'bardeen_polar_metric_dynamic_tln_charge_scan_long.csv'
FIG = ROOT / 'figures'
FIG.mkdir(parents=True, exist_ok=True)

rows = list(csv.DictReader(DATA.open()))
frequencies = sorted({float(r['Momega']) for r in rows})
qext = 4.0 / (3.0 * np.sqrt(3.0))


x_smooth = np.linspace(0.0, 1.0, 500)
ell_smooth = qext * x_smooth
k_static_smooth = (0.08022747739312462*ell_smooth**2
                   +10.76112455362971*ell_smooth**4
                   -0.14943707892554278*ell_smooth**6
                   -0.1809082206985172*ell_smooth**8)

fig, ax = plt.subplots(figsize=(7.2, 5.0))
ax.plot(x_smooth, k_static_smooth, '--', linewidth=1.8, label=r'Static ($M\omega=0$)')
main_lines = []
for omega, marker in zip(frequencies, ['o', 's', '^']):
    subset = sorted((r for r in rows if abs(float(r['Momega'])-omega) < 1e-12),
                    key=lambda r: float(r['ell_over_ell_ext']))
    x = np.array([float(r['ell_over_ell_ext']) for r in subset])
    y = np.array([float(r['k_dynamic_real']) for r in subset])
    line, = ax.plot(x, y, marker=marker, markersize=4.2, linewidth=1.35,
                    label=rf'$M\omega={omega:.3f}$')
    main_lines.append((omega, line, x, y, np.array([float(r['k_static']) for r in subset])))
ax.set_xlabel(r'$\ell/\ell_{\rm ext}$')
ax.set_ylabel(r'$k_{20}^{\rm polar}(\omega)$')
ax.set_xlim(0.0, 1.0)
ax.set_ylim(bottom=-0.03)
ax.grid(alpha=0.25)
ax.legend(frameon=False, loc='upper left')

ins = inset_axes(ax, width='48%', height='43%', loc='center left',
                 bbox_to_anchor=(0.10, -0.02, 1, 1), bbox_transform=ax.transAxes,
                 borderpad=0.8)
for omega, line, x, y, ks in main_lines:
    ins.plot(x, 1.0e4*(y-ks), marker=line.get_marker(), markersize=3.0,
             linewidth=1.05, color=line.get_color())
ins.set_xlabel(r'$\ell/\ell_{\rm ext}$', fontsize=8)
ins.set_ylabel(r'$10^{4}[k(\omega)-k(0)]$', fontsize=8)
ins.tick_params(labelsize=7)
ins.grid(alpha=0.22)
ins.set_xlim(0.0, 1.0)

fig.tight_layout()
fig.savefig(FIG / 'bardeen_polar_metric_dynamic_tln_vs_ell_ratio.png', dpi=320)
fig.savefig(FIG / 'bardeen_polar_metric_dynamic_tln_vs_ell_ratio.pdf')
plt.close(fig)

fig, ax = plt.subplots(figsize=(7.0, 4.8))
for omega, line, x, y, ks in main_lines:
    ax.plot(x, y-ks, marker=line.get_marker(), markersize=4.2, linewidth=1.35,
            label=rf'$M\omega={omega:.3f}$')
ax.axhline(0.0, linewidth=1.0)
ax.set_xlabel(r'$\ell/\ell_{\rm ext}$')
ax.set_ylabel(r'$k_{20}^{\rm polar}(\omega)-k_{20}^{\rm polar}(0)$')
ax.set_xlim(0.0, 1.0)
ax.grid(alpha=0.25)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(FIG / 'bardeen_polar_metric_dynamic_correction_vs_ell_ratio.png', dpi=320)
fig.savefig(FIG / 'bardeen_polar_metric_dynamic_correction_vs_ell_ratio.pdf')
plt.close(fig)
