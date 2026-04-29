from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def build_classification_loaders(cfg: dict) -> tuple[DataLoader, DataLoader, DataLoader]:
    data_root = Path(cfg.get("data_root", "data"))
    image_size = int(cfg.get("image_size", 224))
    batch_size = int(cfg.get("batch_size", 32))
    num_workers = int(cfg.get("num_workers", 4))
    val_ratio = float(cfg.get("val_ratio", 0.15))
    seed = int(cfg.get("seed", 42))
    download = bool(cfg.get("download", False))

    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    trainval = datasets.OxfordIIITPet(
        root=str(data_root),
        split="trainval",
        target_types="category",
        transform=train_tf,
        download=download,
    )
    test = datasets.OxfordIIITPet(
        root=str(data_root),
        split="test",
        target_types="category",
        transform=eval_tf,
        download=download,
    )

    val_len = max(1, int(len(trainval) * val_ratio))
    train_len = len(trainval) - val_len
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(trainval, [train_len, val_len], generator=generator)
    val_set.dataset = datasets.OxfordIIITPet(
        root=str(data_root),
        split="trainval",
        target_types="category",
        transform=eval_tf,
        download=False,
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader
