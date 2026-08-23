#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT/codes"
python run_metric_scans.py
python aggregate_metric_scans.py
python plot_metric_scans.py
python axial_broad_response.py
python axial_ringdown.py
python plot_resonance.py
