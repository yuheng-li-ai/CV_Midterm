from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a qualitative segmentation comparison grid.")
    parser.add_argument("--target-dir", required=True, help="directory containing *_target.png files")
    parser.add_argument("--pred", nargs="+", required=True, help="label=directory pairs containing *_pred.png files")
    parser.add_argument("--out", required=True, help="output PNG")
    parser.add_argument("--rows", type=int, default=8, help="number of examples to include")
    args = parser.parse_args()

    target_dir = Path(args.target_dir)
    pred_dirs = [_parse_pair(pair) for pair in args.pred]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    font = ImageFont.load_default()
    labels = ["Ground Truth"] + [label for label, _ in pred_dirs]
    samples = sorted(target_dir.glob("*_target.png"))[: args.rows]
    if not samples:
        raise FileNotFoundError(f"No target masks found in {target_dir}")

    first = Image.open(samples[0]).convert("RGB")
    cell_w, cell_h = first.size
    header_h = 28
    pad = 8
    grid_w = len(labels) * cell_w + (len(labels) + 1) * pad
    grid_h = header_h + len(samples) * cell_h + (len(samples) + 1) * pad
    canvas = Image.new("RGB", (grid_w, grid_h), "white")
    draw = ImageDraw.Draw(canvas)

    for col, label in enumerate(labels):
        x = pad + col * (cell_w + pad)
        draw.text((x, 8), label, fill=(0, 0, 0), font=font)

    for row, target_path in enumerate(samples):
        stem = target_path.name.replace("_target.png", "")
        y = header_h + pad + row * (cell_h + pad)
        images = [target_path]
        for _, pred_dir in pred_dirs:
            pred_path = pred_dir / f"{stem}_pred.png"
            if not pred_path.exists():
                raise FileNotFoundError(f"Missing prediction mask: {pred_path}")
            images.append(pred_path)
        for col, image_path in enumerate(images):
            x = pad + col * (cell_w + pad)
            image = Image.open(image_path).convert("RGB")
            canvas.paste(image, (x, y))

    canvas.save(out)


def _parse_pair(pair: str) -> tuple[str, Path]:
    if "=" not in pair:
        raise ValueError(f"Expected label=path, got {pair!r}")
    label, path = pair.split("=", 1)
    return label, Path(path)


if __name__ == "__main__":
    main()
