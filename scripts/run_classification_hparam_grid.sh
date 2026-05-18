#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-src}"

for config in configs/classification_grid/*.yaml; do
  echo "Running ${config}"
  python -m cvhw2 train-classification --config "${config}"
done

python scripts/summarize_classification_hparam_grid.py
