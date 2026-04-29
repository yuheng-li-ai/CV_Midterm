from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import functional as F


class PetSegmentationTransform:
    def __init__(self, image_size: int, train: bool):
        self.image_size = image_size
        self.train = train

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        image = F.resize(image, [self.image_size, self.image_size], interpolation=F.InterpolationMode.BILINEAR)
        mask = F.resize(mask, [self.image_size, self.image_size], interpolation=F.InterpolationMode.NEAREST)
        if self.train and torch.rand(()) < 0.5:
            image = F.hflip(image)
            mask = F.hflip(mask)
        image_tensor = F.to_tensor(image)
        image_tensor = F.normalize(image_tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        mask_np = np.array(mask, dtype=np.int64)
        mask_np = np.clip(mask_np, 1, 3) - 1
        return image_tensor, torch.from_numpy(mask_np).long()


class OxfordPetSegmentation(datasets.OxfordIIITPet):
    def __init__(self, *args, joint_transform: PetSegmentationTransform, **kwargs):
        super().__init__(*args, target_types="segmentation", **kwargs)
        self.joint_transform = joint_transform

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(self._images[index]).convert("RGB")
        mask = Image.open(self._segs[index])
        return self.joint_transform(image, mask)


def build_segmentation_loaders(cfg: dict) -> tuple[DataLoader, DataLoader, DataLoader]:
    data_root = Path(cfg.get("data_root", "data"))
    image_size = int(cfg.get("image_size", 224))
    batch_size = int(cfg.get("batch_size", 16))
    num_workers = int(cfg.get("num_workers", 4))
    val_ratio = float(cfg.get("val_ratio", 0.15))
    seed = int(cfg.get("seed", 42))
    download = bool(cfg.get("download", False))

    trainval = OxfordPetSegmentation(
        root=str(data_root),
        split="trainval",
        joint_transform=PetSegmentationTransform(image_size=image_size, train=True),
        download=download,
    )
    test = OxfordPetSegmentation(
        root=str(data_root),
        split="test",
        joint_transform=PetSegmentationTransform(image_size=image_size, train=False),
        download=download,
    )
    val_len = max(1, int(len(trainval) * val_ratio))
    train_len = len(trainval) - val_len
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(trainval, [train_len, val_len], generator=generator)
    val_set.dataset = OxfordPetSegmentation(
        root=str(data_root),
        split="trainval",
        joint_transform=PetSegmentationTransform(image_size=image_size, train=False),
        download=False,
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader
