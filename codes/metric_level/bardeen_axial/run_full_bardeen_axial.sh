#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT/codes"
rm -rf "$ROOT/scan"
mkdir -p "$ROOT/scan"
for ratio in 0.0 0.1 0.2 0.3 0.4 0.6 0.7 0.9 0.99; do
  tag=$(python -c "print(f'{round(float(\"$ratio\")*100):03d}')")
  python bardeen_axial_metric_solver.py --charge-ratio "$ratio" \
    --frequencies 0.002 0.004 0.006 --output-dir "$ROOT/scan/q$tag"
done
for ratio in 0.5 0.8 0.95; do
  tag=$(python -c "print(f'{round(float(\"$ratio\")*100):03d}')")
  python bardeen_axial_metric_solver.py --charge-ratio "$ratio" \
    --frequencies 0.001 0.002 0.003 0.004 0.005 0.006 \
    --output-dir "$ROOT/scan/q$tag"
done
python bardeen_axial_metric_solver.py --charge-ratio 0.0 \
  --frequencies 0.001 0.002 0.003 0.004 0.005 0.006 \
  --output-dir "$ROOT/scan/q000"
python aggregate_and_plot.py
for ratio in 0.6 0.9 0.97; do
  python bardeen_axial_broad_one.py --ratio "$ratio"
done
python bardeen_axial_ringdown.py
python finalize_plots.py
