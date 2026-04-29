from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .config import ensure_dir


def get_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def save_checkpoint(path: str | Path, **payload: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(payload, path)
