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
    parser = argparse.ArgumentParser(description="Summarize multiple metrics.csv files into a report-ready table and bar plot.")
    parser.add_argument("metrics", nargs="+", help="metrics.csv paths")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--metric", required=True, help="metric column to compare, e.g. best_val_accuracy or best_val_miou")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path_str in args.metrics:
        path = Path(path_str)
        df = pd.read_csv(path)
        value = _extract_metric(df, args.metric)
        rows.append({"run": path.parent.name, "metrics": str(path), args.metric: value})

    result = pd.DataFrame(rows).sort_values(args.metric, ascending=False)
    result.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "summary.md").write_text(_to_markdown(result), encoding="utf-8")

    plt.figure(figsize=(max(6, len(result) * 1.4), 4.5))
    plt.bar(result["run"], result[args.metric])
    plt.ylabel(args.metric)
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / f"{args.metric}.png", dpi=180)


def _extract_metric(df: pd.DataFrame, metric: str) -> float:
    if metric in df.columns and df[metric].notna().any():
        return float(df[metric].dropna().iloc[-1])
    if metric.startswith("best_"):
        base = metric.removeprefix("best_")
        if base in df.columns and df[base].notna().any():
            return float(df[base].max())
    raise KeyError(f"Metric {metric!r} not found in columns: {list(df.columns)}")


def _to_markdown(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
