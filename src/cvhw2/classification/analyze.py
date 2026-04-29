from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import datasets
from torchvision.transforms import functional as F
from tqdm import tqdm

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from cvhw2.classification.data import build_classification_loaders
from cvhw2.classification.models import build_classifier
from cvhw2.utils.config import ensure_dir, load_config
from cvhw2.utils.torch import get_device


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


@torch.no_grad()
def analyze_classification(config_path: str, checkpoint: str, split: str, out_dir: str, max_examples: int = 24) -> None:
    cfg = load_config(config_path)
    device = get_device(cfg.get("device", "auto"))
    out_path = ensure_dir(out_dir)
    _, val_loader, test_loader = build_classification_loaders({**cfg, "download": False})
    loader = val_loader if split == "val" else test_loader

    model = build_classifier(cfg).to(device)
    payload = torch.load(checkpoint, map_location=device)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state)
    model.eval()

    class_names = datasets.OxfordIIITPet(root=cfg.get("data_root", "data"), split="trainval", target_types="category", download=False).classes
    num_classes = int(cfg.get("num_classes", len(class_names)))
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    mistakes: list[tuple[torch.Tensor, int, int, float]] = []
    correct = 0
    total = 0

    for images, targets in tqdm(loader, desc=f"analyze {split}", leave=False):
        logits = model(images.to(device))
        probs = logits.softmax(dim=1).cpu()
        preds = probs.argmax(dim=1)
        for image, target, pred, prob in zip(images.cpu(), targets.cpu(), preds, probs):
            t = int(target)
            p = int(pred)
            confusion[t, p] += 1
            correct += int(t == p)
            total += 1
            if t != p and len(mistakes) < max_examples:
                mistakes.append((image, t, p, float(prob[p])))

    accuracy = correct / max(total, 1)
    np.savetxt(out_path / f"{split}_confusion_matrix.csv", confusion, fmt="%d", delimiter=",")
    _plot_confusion(confusion, class_names, out_path / f"{split}_confusion_matrix.png")
    _save_mistakes(mistakes, class_names, out_path / f"{split}_mistakes.png")
    with (out_path / f"{split}_summary.md").open("w", encoding="utf-8") as f:
        f.write(f"# Classification Analysis: {split}\n\n")
        f.write(f"- Checkpoint: `{checkpoint}`\n")
        f.write(f"- Samples: {total}\n")
        f.write(f"- Accuracy: {accuracy:.6f}\n")
        f.write(f"- Confusion matrix: `{split}_confusion_matrix.png`\n")
        f.write(f"- Mistake grid: `{split}_mistakes.png`\n")


def _plot_confusion(confusion: np.ndarray, class_names: list[str], out: Path) -> None:
    row_sums = confusion.sum(axis=1, keepdims=True)
    normalized = np.divide(confusion, np.maximum(row_sums, 1), where=row_sums != 0)
    plt.figure(figsize=(14, 12))
    plt.imshow(normalized, cmap="magma", vmin=0, vmax=1)
    plt.colorbar(label="row-normalized frequency")
    plt.xticks(range(len(class_names)), class_names, rotation=90, fontsize=6)
    plt.yticks(range(len(class_names)), class_names, fontsize=6)
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close()


def _save_mistakes(mistakes: list[tuple[torch.Tensor, int, int, float]], class_names: list[str], out: Path) -> None:
    if not mistakes:
        Image.new("RGB", (512, 128), "white").save(out)
        return
    cell_w, cell_h = 180, 220
    cols = 4
    rows = int(np.ceil(len(mistakes) / cols))
    canvas = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for idx, (image, target, pred, score) in enumerate(mistakes):
        x = (idx % cols) * cell_w
        y = (idx // cols) * cell_h
        img = _denormalize(image)
        img = F.resize(img, [150, 150])
        canvas.paste(F.to_pil_image(img), (x + 15, y + 5))
        true_name = class_names[target]
        pred_name = class_names[pred]
        draw.text((x + 8, y + 162), f"true: {true_name[:22]}", fill=(0, 0, 0), font=font)
        draw.text((x + 8, y + 182), f"pred: {pred_name[:22]}", fill=(180, 0, 0), font=font)
        draw.text((x + 8, y + 202), f"conf: {score:.2f}", fill=(0, 0, 0), font=font)
    canvas.save(out)


def _denormalize(image: torch.Tensor) -> torch.Tensor:
    return (image * STD + MEAN).clamp(0, 1)
