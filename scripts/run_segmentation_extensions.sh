#!/usr/bin/env bash
set -euo pipefail

nohup env PYTHONPATH=src python -m cvhw2 train-segmentation \
  --config configs/segmentation_dice_smooth_0_1_60.yaml \
  > smooth_60.log 2>&1 &

nohup env PYTHONPATH=src python -m cvhw2 train-segmentation \
  --config configs/segmentation_focal_60.yaml \
  > focal_60.log 2>&1 &

echo "Started segmentation extension runs."
echo "Smooth log: smooth_60.log"
echo "Focal log: focal_60.log"
