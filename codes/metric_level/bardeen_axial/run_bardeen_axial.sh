#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/codes"
python aggregate_and_plot.py
python finalize_plots.py
