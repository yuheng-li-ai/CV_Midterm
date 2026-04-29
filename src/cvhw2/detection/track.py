from __future__ import annotations

import os
import json
from pathlib import Path

import numpy as np

from cvhw2.utils.config import ensure_dir


def track_video(
    weights: str,
    video: str,
    out: str,
    line: tuple[int, int, int, int],
    tracker: str = "bytetrack.yaml",
    conf: float = 0.25,
    iou: float = 0.5,
    export_frames: list[int] | None = None,
) -> None:
    os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
    try:
        import cv2
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("tracking requires ultralytics and opencv-python") from exc

    out_path = Path(out)
    ensure_dir(out_path.parent)
    export_set = set(export_frames or [])
    frame_dir = ensure_dir(out_path.parent / "occlusion_frames")

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    model = YOLO(weights)
    seen_side: dict[int, float] = {}
    counted: set[int] = set()
    total_count = 0
    unique_ids: set[int] = set()
    frame_records: list[dict[str, int]] = []
    crossings: list[dict[str, int | str]] = []
    x1, y1, x2, y2 = line

    results = model.track(source=video, stream=True, persist=True, tracker=tracker, conf=conf, iou=iou, verbose=False)
    for frame_idx, result in enumerate(results):
        frame = result.orig_img.copy()
        boxes = result.boxes
        tracked_count = 0
        if boxes is not None and boxes.id is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            ids = boxes.id.cpu().numpy().astype(int)
            classes = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            names = result.names
            tracked_count = len(ids)
            for box, track_id, cls_id, score in zip(xyxy, ids, classes, confs):
                unique_ids.add(int(track_id))
                bx1, by1, bx2, by2 = box.astype(int)
                cx = int((bx1 + bx2) / 2)
                cy = int((by1 + by2) / 2)
                side = _signed_side(cx, cy, x1, y1, x2, y2)
                if track_id in seen_side and track_id not in counted and seen_side[track_id] * side < 0:
                    counted.add(track_id)
                    total_count += 1
                    crossings.append(
                        {
                            "frame": frame_idx,
                            "track_id": int(track_id),
                            "class": str(names.get(cls_id, cls_id)),
                            "count_after_crossing": total_count,
                        }
                    )
                seen_side[track_id] = side

                color = _color_for_id(track_id)
                label = f"ID {track_id} {names.get(cls_id, cls_id)} {score:.2f}"
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
                cv2.circle(frame, (cx, cy), 3, color, -1)
                cv2.putText(frame, label, (bx1, max(15, by1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(frame, f"Line count: {total_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        if frame_idx in export_set:
            cv2.imwrite(str(frame_dir / f"frame_{frame_idx:06d}.jpg"), frame)
        writer.write(frame)
        frame_records.append({"frame": frame_idx, "tracked_count": int(tracked_count), "line_count": int(total_count)})
    writer.release()
    summary = {
        "video": str(video),
        "out": str(out_path),
        "weights": str(weights),
        "tracker": tracker,
        "conf": conf,
        "iou": iou,
        "line": [x1, y1, x2, y2],
        "frames": len(frame_records),
        "fps": fps,
        "width": width,
        "height": height,
        "unique_track_ids": len(unique_ids),
        "final_line_count": total_count,
        "crossings": crossings,
        "frame_records": frame_records,
        "exported_frames": sorted(export_set),
    }
    with (out_path.with_suffix(".json")).open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def _signed_side(px: int, py: int, x1: int, y1: int, x2: int, y2: int) -> float:
    return float((x2 - x1) * (py - y1) - (y2 - y1) * (px - x1))


def _color_for_id(track_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(track_id)
    color = rng.integers(60, 255, size=3)
    return int(color[0]), int(color[1]), int(color[2])
