#!/bin/bash
set -e
model=$1; ratio=$2; shift 2
out=/mnt/data/Hayward_FanWang_Polar_Metric_Dynamic_TLN/results/raw/${model}_${ratio//./p}
python /mnt/data/Hayward_FanWang_Polar_Metric_Dynamic_TLN/codes/rbh_polar_metric_solver.py --model "$model" --charge-ratio "$ratio" --frequencies "$@" --match-radii 10 12 14 16 --output-dir "$out" > "$out.log"
