#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
python "$ROOT/codes/bardeen_polar_nearzone_dynamic_tln.py" \
  --charge-ratio 0.5 \
  --series-order 18 \
  --omega-order 2 \
  --log-order 5 \
  --frequencies 0.001 0.002 0.003 0.004 0.005 0.006 0.007 0.008 \
  --match-radii 10 12 14 16 \
  --rtol 5e-10 \
  --atol 5e-12 \
  --output-dir "$ROOT"

python "$ROOT/codes/bardeen_polar_metric_projection_check.py" \
  --charge-ratio 0.5 \
  --omega 0.004 \
  --match-radius 14 \
  --series-order 18 \
  --omega-order 2 \
  --log-order 5 \
  --rtol 5e-10 \
  --atol 5e-12 \
  --output-dir "$ROOT"

python "$ROOT/codes/plot_bardeen_polar_results.py"
