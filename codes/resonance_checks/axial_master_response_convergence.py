#!/usr/bin/env python3
from __future__ import annotations
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_rows(path, model_name=None):
    rows = []
    with path.open(newline='') as handle:
        for row in csv.DictReader(handle):
            rows.append({
                'model': row.get('model') or model_name,
                'ell_over_ell_ext': float(row['ell_over_ell_ext']),
                'omega_M': float(row['omega_M']),
                'strict_controls': int(row['strict_controls']),
                'delta_Rgg_real': float(row['delta_Rgg_real']),
                'delta_Rgg_imag': float(row['delta_Rgg_imag']),
                'delta_Rgg_abs': float(row['delta_Rgg_abs']),
            })
    return rows


def summarize(rows):
    output = []
    models = ['bardeen', 'hayward', 'fan_wang']
    ratios = [0.6, 0.9, 0.97]
    for model in models:
        for ratio in ratios:
            base = [r for r in rows if r['model'] == model and r['ell_over_ell_ext'] == ratio and r['strict_controls'] == 0]
            strict = [r for r in rows if r['model'] == model and r['ell_over_ell_ext'] == ratio and r['strict_controls'] == 1]
            peak = max(base, key=lambda r: r['delta_Rgg_abs'])
            strict_peak = min(strict, key=lambda r: abs(r['omega_M'] - peak['omega_M']))
            z0 = complex(peak['delta_Rgg_real'], peak['delta_Rgg_imag'])
            z1 = complex(strict_peak['delta_Rgg_real'], strict_peak['delta_Rgg_imag'])
            # Compare the complex responses at the standard-control
            # response-magnitude peak; this is not a peak-height difference.
            change = abs(z1 - z0) / max(abs(z1), 1e-30)
            output.append({
                'model': model,
                'ell_over_ell_ext': ratio,
                'peak_omega_M': peak['omega_M'],
                'peak_abs': peak['delta_Rgg_abs'],
                'relative_peak_response_change': change,
            })
    return output


def main():
    bardeen = ROOT / 'metric_level' / 'bardeen_axial' / 'results' / 'bardeen_axial_broad_response.csv'
    other = ROOT / 'metric_level' / 'hayward_fanwang_axial' / 'results' / 'hayward_fanwang_axial_broad_response.csv'
    rows = load_rows(bardeen, 'bardeen') + load_rows(other)
    summary = summarize(rows)
    json_path = ROOT / 'results' / 'axial_master_response_convergence.json'
    csv_path = ROOT / 'results' / 'axial_master_response_convergence.csv'
    json_path.write_text(json.dumps({'classification': 'broad axial scattering maxima', 'cases': summary}, indent=2) + '\n')
    with csv_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
