#!/usr/bin/env python3

from __future__ import annotations
import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOLVER = HERE / 'bardeen_polar_nearzone_dynamic_tln.py'

RATIOS = [1.0e-5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
FREQUENCIES = [0.002, 0.004, 0.006]


def direct_static_fit(ratio: float) -> float:
    x=ratio*4.0/(3.0*math.sqrt(3.0))
    return (0.08022747739312462*x**2+10.76112455362971*x**4
            -0.14943707892554278*x**6-0.1809082206985172*x**8)


def tag(ratio: float) -> str:
    if ratio < 1.0e-4:
        return 'schwarzschild_limit'
    return f'ellratio_{ratio:.2f}'.replace('.', 'p')


def run_solver(ratio: float, output: Path) -> None:
    cmd = [
        sys.executable, str(SOLVER),
        '--charge-ratio', str(ratio),
        '--series-order', '18',
        '--omega-order', '2',
        '--log-order', '5',
        '--frequencies', *[str(x) for x in FREQUENCIES],
        '--match-radii', '10', '12', '14', '16',
        '--rtol', '5e-10',
        '--atol', '5e-12',
        '--output-dir', str(output),
    ]
    subprocess.run(cmd, check=True)


def aggregate(scan_root: Path) -> None:
    qext = 4.0 / (3.0 * math.sqrt(3.0))
    rows = []
    for ratio in RATIOS:
        file = scan_root / tag(ratio) / 'results' / 'bardeen_polar_dynamic_tln.csv'
        for r in csv.DictReader(file.open()):
            omega = float(r['omega_M'])
            kstatic = 0.0 if ratio <= 1.0e-4 else direct_static_fit(ratio)
            # The solver's k_dynamic is the direct metric response for this geometry.
            kreal = float(r['k_dynamic_real'])
            kimag = float(r['k_dynamic_imag'])
            rows.append({
                'ell_over_ell_ext': 0.0 if ratio <= 1.0e-4 else ratio,
                'ell': 0.0 if ratio <= 1.0e-4 else ratio*qext,
                'Momega': omega,
                'k_static': kstatic,
                'metric_ratio_real': float(r['metric_ratio_real']),
                'metric_ratio_imag': float(r['metric_ratio_imag']),
                'metric_ratio_static_real': float(r['metric_ratio_static_real']),
                'metric_ratio_static_imag': float(r['metric_ratio_static_imag']),
                'k_dynamic_real': kreal, 'k_dynamic_imag': kimag,
                'delta_k_real': kreal-kstatic, 'delta_k_imag': kimag,
                'window_fit_rms': float(r['window_fit_rms']),
                'window_max_residual': float(r['window_max_residual']),
                'maximum_omega_r': float(r['maximum_omega_r']),
            })
    rows.sort(key=lambda x: (x['Momega'], x['ell_over_ell_ext']))
    results = ROOT / 'results'
    results.mkdir(parents=True, exist_ok=True)
    long_file = results / 'bardeen_polar_metric_dynamic_tln_charge_scan_long.csv'
    with long_file.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    ratios = sorted({x['ell_over_ell_ext'] for x in rows})
    wide = []
    for ratio in ratios:
        selected = [x for x in rows if x['ell_over_ell_ext'] == ratio]
        row = {'ell_over_ell_ext': ratio, 'ell': ratio*qext, 'k_static': selected[0]['k_static']}
        for x in selected:
            suffix = str(x['Momega']).replace('.', 'p')
            row[f'k_real_Momega_{suffix}'] = x['k_dynamic_real']
            row[f'k_imag_Momega_{suffix}'] = x['k_dynamic_imag']
            row[f'delta_k_real_Momega_{suffix}'] = x['delta_k_real']
        wide.append(row)
    with (results/'bardeen_polar_metric_dynamic_tln_charge_scan_wide.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(wide[0]))
        writer.writeheader(); writer.writerows(wide)

    summary = {
        'model': 'Bardeen', 'sector': 'polar', 'multipole': 2, 'mass': 1.0,
        'ell_ext': qext, 'ell_over_ell_ext_values': ratios,
        'Momega_values': FREQUENCIES,
        'definition': 'Direct metric near-zone response of reconstructed -f H0 with unit gravitational source and zero independent electromagnetic source.',
        'static_baseline': 'Direct coupled Bardeen polar static result.',
        'polar_love_normalization': 'k20_polar=-(C_response/C_source)/M^5.',
        'formula': 'k_dyn(ell,omega)=k_static(ell)+[k_direct(ell,omega)-k_direct(ell,0)]. No Schwarzschild subtraction is applied.',
        'near_zone_basis': 'O(omega^4), radial order 18, log order 5, radii 10M,12M,14M,16M.',
        'controlled_range': 'Momega <= 0.006; Momega=0.006 is the edge diagnostic.',
    }
    (results/'bardeen_polar_metric_dynamic_tln_charge_scan_summary.json').write_text(json.dumps(summary, indent=2)+'\n')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scan-root', type=Path, default=ROOT/'results'/'raw_scans_recomputed')
    parser.add_argument('--skip-solvers', action='store_true')
    args = parser.parse_args()
    args.scan_root.mkdir(parents=True, exist_ok=True)
    if not args.skip_solvers:
        for ratio in RATIOS:
            run_solver(ratio, args.scan_root/tag(ratio))
    aggregate(args.scan_root)
    subprocess.run([sys.executable, str(HERE/'plot_bardeen_polar_charge_scan.py')], check=True)


if __name__ == '__main__':
    main()
