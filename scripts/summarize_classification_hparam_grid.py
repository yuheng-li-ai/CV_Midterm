from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import yaml


ROOT = Path("runs/classification_hparam_grid")
CONFIG_ROOT = Path("configs/classification_grid")
OUT_DIR = Path("assets/classification_hparam_grid")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for config_path in sorted(CONFIG_ROOT.glob("*.yaml")):
        with config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        metrics_path = Path(cfg["run_dir"]) / "metrics.csv"
        if not metrics_path.exists():
            continue
        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = list(csv.DictReader(f))
        if not metrics:
            continue
        val_rows = [row for row in metrics if row.get("val_accuracy")]
        best = max(val_rows, key=lambda row: float(row["val_accuracy"]))
        test_rows = [row for row in metrics if row.get("test_accuracy")]
        test_accuracy = float(test_rows[-1]["test_accuracy"]) if test_rows else float("nan")
        rows.append(
            {
                "name": cfg["name"],
                "label_smoothing": float(cfg["label_smoothing"]),
                "head_lr": float(cfg["head_lr"]),
                "backbone_lr": float(cfg["backbone_lr"]),
                "best_epoch": int(best["epoch"]),
                "best_val_accuracy": float(best["val_accuracy"]),
                "test_accuracy": test_accuracy,
            }
        )

    rows.sort(key=lambda row: (row["label_smoothing"], row["head_lr"]))
    summary_csv = OUT_DIR / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "label_smoothing",
                "head_lr",
                "backbone_lr",
                "best_epoch",
                "best_val_accuracy",
                "test_accuracy",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary_md = OUT_DIR / "summary.md"
    with summary_md.open("w", encoding="utf-8") as f:
        f.write("# Classification Hyperparameter Grid Search\n\n")
        f.write("Fixed setting: pretrained ResNet-18, image size 224, batch size 32, backbone_lr 1e-4, weight_decay 1e-4, 8 epochs.\n\n")
        f.write("| label_smoothing | head_lr | best_epoch | best_val_accuracy | test_accuracy |\n")
        f.write("|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                f"| {row['label_smoothing']:.2f} | {row['head_lr']:.4g} | {row['best_epoch']} | "
                f"{row['best_val_accuracy']:.4f} | {row['test_accuracy']:.4f} |\n"
            )
        if rows:
            best = max(rows, key=lambda row: row["best_val_accuracy"])
            f.write("\n")
            f.write(
                "Best validation setting: "
                f"label_smoothing={best['label_smoothing']:.2f}, head_lr={best['head_lr']:.4g}, "
                f"best_val_accuracy={best['best_val_accuracy']:.4f}.\n"
            )

    if rows:
        _plot_heatmap(rows, metric="best_val_accuracy", out_path=OUT_DIR / "val_accuracy_heatmap.png")
        _plot_heatmap(rows, metric="test_accuracy", out_path=OUT_DIR / "test_accuracy_heatmap.png")
    print(f"Wrote {summary_csv} and {summary_md}")


def _plot_heatmap(rows: list[dict], metric: str, out_path: Path) -> None:
    label_smoothing_values = sorted({row["label_smoothing"] for row in rows})
    head_lr_values = sorted({row["head_lr"] for row in rows})
    value_by_pair = {(row["label_smoothing"], row["head_lr"]): row[metric] for row in rows}
    grid = [
        [value_by_pair.get((label_smoothing, head_lr), float("nan")) for head_lr in head_lr_values]
        for label_smoothing in label_smoothing_values
    ]

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    image = ax.imshow(grid, cmap="viridis", vmin=min(min(row) for row in grid), vmax=max(max(row) for row in grid))
    ax.set_xticks(range(len(head_lr_values)), [f"{value:g}" for value in head_lr_values])
    ax.set_yticks(range(len(label_smoothing_values)), [f"{value:.2f}" for value in label_smoothing_values])
    ax.set_xlabel("Head learning rate")
    ax.set_ylabel("Label smoothing")
    ax.set_title(metric.replace("_", " ").title())
    for y, label_smoothing in enumerate(label_smoothing_values):
        for x, head_lr in enumerate(head_lr_values):
            value = value_by_pair.get((label_smoothing, head_lr))
            if value is not None:
                ax.text(x, y, f"{value:.3f}", ha="center", va="center", color="white", fontsize=9)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
