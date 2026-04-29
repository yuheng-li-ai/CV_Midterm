from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot metrics.csv curves for reports")
    parser.add_argument("csv", help="path to metrics.csv")
    parser.add_argument("--out", required=True, help="output png")
    parser.add_argument("--columns", nargs="+", required=True, help="metric columns to plot")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    x = df["epoch"] if "epoch" in df.columns else range(len(df))
    plt.figure(figsize=(8, 5))
    for col in args.columns:
        if col in df.columns:
            plt.plot(x, df[col], marker="o", label=col)
    plt.xlabel("epoch")
    plt.ylabel("metric")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=180)


if __name__ == "__main__":
    main()
