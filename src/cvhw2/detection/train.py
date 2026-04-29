from __future__ import annotations

import os
from pathlib import Path

from cvhw2.utils.config import load_config


def train_detection(config_path: str) -> None:
    cfg = load_config(config_path)
    os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("ultralytics is required for detection. Install with: pip install ultralytics") from exc

    model = YOLO(cfg.get("model", "yolov8n.pt"))
    project_dir = Path(cfg.get("project_dir", "runs/detection")).resolve()
    model.train(
        data=cfg["data"],
        epochs=int(cfg.get("epochs", 50)),
        imgsz=int(cfg.get("imgsz", 640)),
        batch=int(cfg.get("batch", 16)),
        device=cfg.get("device", "0"),
        project=str(project_dir),
        name=cfg.get("name", "yolov8n_visdrone"),
        pretrained=bool(cfg.get("pretrained", True)),
        optimizer=cfg.get("optimizer", "auto"),
        lr0=float(cfg.get("lr0", 0.01)),
        workers=int(cfg.get("workers", 4)),
        patience=int(cfg.get("patience", 20)),
        resume=bool(cfg.get("resume", False)),
    )
