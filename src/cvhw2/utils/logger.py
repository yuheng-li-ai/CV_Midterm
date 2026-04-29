from __future__ import annotations

import csv
import importlib
import os
from pathlib import Path
from typing import Any

from .config import ensure_dir


class ExperimentLogger:
    def __init__(self, run_dir: str | Path, backend: str = "csv", project: str = "cvhw2", name: str | None = None):
        self.run_dir = ensure_dir(run_dir)
        self.backend = backend.lower()
        self.csv_path = self.run_dir / "metrics.csv"
        self._rows: list[dict[str, Any]] = []
        self._fieldnames: list[str] = []
        self._external = None

        if self.backend in {"wandb", "swanlab"}:
            if self.backend == "swanlab":
                os.environ.setdefault("SWANLAB_SAVE_DIR", str(self.run_dir / ".swanlab"))
                os.environ.setdefault("SWANLAB_LOG_DIR", str(self.run_dir / "swanlog"))
                os.environ.setdefault("SWANLAB_MODE", "offline")
            module = importlib.import_module(self.backend)
            self._external = module
            if self.backend == "wandb":
                module.init(project=project, name=name, dir=str(self.run_dir), config={"run_dir": str(self.run_dir)})
            else:
                module.init(project=project, experiment_name=name, logdir=str(self.run_dir), config={"run_dir": str(self.run_dir)})

    def log(self, metrics: dict[str, Any]) -> None:
        flat = {k: _to_scalar(v) for k, v in metrics.items()}
        for key in flat:
            if key not in self._fieldnames:
                self._fieldnames.append(key)
        self._rows.append(flat)
        with self.csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            writer.writeheader()
            writer.writerows(self._rows)
        if self._external is not None:
            self._external.log(flat)

    def close(self) -> None:
        if self._external is not None and self.backend == "wandb":
            self._external.finish()


def _to_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value
