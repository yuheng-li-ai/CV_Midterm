#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=src

python -m cvhw2 train-classification --config configs/classification_resnet18.yaml
python -m cvhw2 train-classification --config configs/classification_random.yaml
python -m cvhw2 train-classification --config configs/classification_se_resnet18.yaml
python -m cvhw2 train-classification --config configs/classification_resnet34.yaml
python -m cvhw2 train-classification --config configs/classification_label_smoothing.yaml
python -m cvhw2 train-classification --config configs/classification_freeze_unfreeze.yaml

python -m cvhw2 train-segmentation --config configs/segmentation_ce.yaml
python -m cvhw2 train-segmentation --config configs/segmentation_dice.yaml
python -m cvhw2 train-segmentation --config configs/segmentation_combo.yaml
python -m cvhw2 train-segmentation --config configs/segmentation_focal.yaml
python -m cvhw2 train-segmentation --config configs/segmentation_smooth_0_1.yaml

python -m cvhw2 convert-visdrone --source data/VisDrone --out data/visdrone_yolo
python -m cvhw2 train-detection --config configs/detection_yolov8n.yaml
python -m cvhw2 train-detection --config configs/detection_yolov8s.yaml
