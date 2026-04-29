#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${1:-$ROOT_DIR/data/VisDrone}"
YOLO_DIR="${2:-$ROOT_DIR/data/visdrone_yolo}"

mkdir -p "$DATA_DIR"

declare -a FILES=(
  "VisDrone2019-DET-train.zip"
  "VisDrone2019-DET-val.zip"
  "VisDrone2019-DET-test-dev.zip"
)

BASE_URL="https://github.com/ultralytics/assets/releases/download/v0.0.0"

for file in "${FILES[@]}"; do
  url="$BASE_URL/$file"
  target="$DATA_DIR/$file"
  split_dir="${target%.zip}"

  if [[ ! -f "$target" ]]; then
    echo "Downloading $file"
    curl -L --fail --retry 5 --retry-delay 5 --continue-at - "$url" -o "$target"
  else
    echo "Found existing archive: $target"
  fi

  if [[ ! -d "$split_dir" ]]; then
    echo "Extracting $file"
    unzip -q "$target" -d "$DATA_DIR"
  else
    echo "Found existing extracted split: $split_dir"
  fi
done

echo "Converting VisDrone DET annotations to YOLO format"
cd "$ROOT_DIR"
PYTHONPATH=src python -m cvhw2 convert-visdrone --source "$DATA_DIR" --out "$YOLO_DIR"

echo "Done: $YOLO_DIR/visdrone.yaml"
