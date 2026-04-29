from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def plot_metrics_csv(csv_path: str | Path, out_path: str | Path, columns: list[str]) -> None:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        return

    x = df["epoch"] if "epoch" in df.columns else range(len(df))
    plotted = False
    plt.figure(figsize=(8, 5))
    for col in columns:
        if col in df.columns and df[col].notna().any():
            plt.plot(x, df[col], marker="o", label=col)
            plotted = True

    if not plotted:
        plt.close()
        return

    plt.xlabel("epoch")
    plt.ylabel("metric")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
