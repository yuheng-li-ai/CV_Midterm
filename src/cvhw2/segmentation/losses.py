from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class DiceLoss(nn.Module):
    def __init__(self, num_classes: int = 3, smooth: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = logits.softmax(dim=1)
        target_onehot = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        intersection = (probs * target_onehot).sum(dims)
        cardinality = probs.sum(dims) + target_onehot.sum(dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class CombinedSegmentationLoss(nn.Module):
    def __init__(self, mode: str, num_classes: int = 3, smooth: float = 1.0):
        super().__init__()
        self.mode = mode.lower()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss(num_classes=num_classes, smooth=smooth)
        if self.mode not in {"ce", "dice", "combo", "focal", "ce_dice_focal"}:
            raise ValueError(f"Unsupported segmentation loss: {mode}")

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.mode == "ce":
            return self.ce(logits, target)
        if self.mode == "dice":
            return self.dice(logits, target)
        if self.mode == "focal":
            return focal_loss(logits, target)
        if self.mode == "ce_dice_focal":
            return self.ce(logits, target) + self.dice(logits, target) + focal_loss(logits, target)
        return self.ce(logits, target) + self.dice(logits, target)


def focal_loss(logits: torch.Tensor, target: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    ce = F.cross_entropy(logits, target, reduction="none")
    pt = torch.exp(-ce)
    return ((1.0 - pt) ** gamma * ce).mean()
