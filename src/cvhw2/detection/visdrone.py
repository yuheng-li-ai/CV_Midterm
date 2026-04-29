from __future__ import annotations

from pathlib import Path

import yaml
from PIL import Image
from tqdm import tqdm

from cvhw2.utils.config import ensure_dir


VISDRONE_CLASSES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]


def convert_visdrone(source: str, out: str) -> Path:
    source_path = Path(source)
    out_path = ensure_dir(out)
    split_map = {
        "train": "VisDrone2019-DET-train",
        "val": "VisDrone2019-DET-val",
        "test": "VisDrone2019-DET-test-dev",
    }

    for split, dirname in split_map.items():
        split_dir = source_path / dirname
        image_dir = split_dir / "images"
        anno_dir = split_dir / "annotations"
        if not image_dir.exists():
            continue
        out_images = ensure_dir(out_path / "images" / split)
        out_labels = ensure_dir(out_path / "labels" / split)
        for image_path in tqdm(sorted(image_dir.glob("*.jpg")), desc=f"convert {split}"):
            target_image = out_images / image_path.name
            if not target_image.exists():
                target_image.symlink_to(image_path.resolve())
            label_path = out_labels / f"{image_path.stem}.txt"
            anno_path = anno_dir / f"{image_path.stem}.txt"
            _convert_annotation(image_path, anno_path, label_path)

    yaml_path = out_path / "visdrone.yaml"
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "path": str(out_path.resolve()),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": {i: name for i, name in enumerate(VISDRONE_CLASSES)},
            },
            f,
            allow_unicode=True,
            sort_keys=False,
        )
    return yaml_path


def _convert_annotation(image_path: Path, anno_path: Path, label_path: Path) -> None:
    width, height = Image.open(image_path).size
    rows: list[str] = []
    if anno_path.exists():
        with anno_path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 8:
                    continue
                x, y, w, h = map(float, parts[:4])
                category = int(parts[5])
                if category < 1 or category > len(VISDRONE_CLASSES):
                    continue
                cx = (x + w / 2.0) / width
                cy = (y + h / 2.0) / height
                nw = w / width
                nh = h / height
                rows.append(f"{category - 1} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    with label_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(rows))
        if rows:
            f.write("\n")
