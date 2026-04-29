from __future__ import annotations

import torch


@torch.no_grad()
def accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return (pred == target).float().mean().item()


@torch.no_grad()
def mean_iou(logits: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int | None = None) -> float:
    pred = logits.argmax(dim=1)
    ious: list[torch.Tensor] = []
    for cls in range(num_classes):
        if ignore_index is not None and cls == ignore_index:
            continue
        pred_mask = pred == cls
        target_mask = target == cls
        intersection = (pred_mask & target_mask).sum()
        union = (pred_mask | target_mask).sum()
        if union > 0:
            ious.append(intersection.float() / union.float())
    if not ious:
        return 0.0
    return torch.stack(ious).mean().item()
