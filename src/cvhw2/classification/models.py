from __future__ import annotations

import torch
from torch import nn
from torchvision import models


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        weight = self.pool(x).view(b, c)
        weight = self.fc(weight).view(b, c, 1, 1)
        return x * weight


class SEResNetClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, channels: int, num_classes: int):
        super().__init__()
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.se = SEBlock(channels)
        self.avgpool = backbone.avgpool
        self.fc = nn.Linear(channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.se(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def build_classifier(cfg: dict) -> nn.Module:
    name = cfg.get("model", "resnet18").lower()
    num_classes = int(cfg.get("num_classes", 37))
    pretrained = bool(cfg.get("pretrained", True))
    attention = cfg.get("attention", "none").lower()

    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        channels = model.fc.in_features
    elif name == "resnet34":
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        model = models.resnet34(weights=weights)
        channels = model.fc.in_features
    else:
        raise ValueError(f"Unsupported classifier: {name}")

    if attention == "se":
        return SEResNetClassifier(model, channels=channels, num_classes=num_classes)
    if attention in {"none", ""}:
        model.fc = nn.Linear(channels, num_classes)
        return model
    raise ValueError(f"Unsupported attention module: {attention}")


def parameter_groups(model: nn.Module, backbone_lr: float, head_lr: float, weight_decay: float) -> list[dict]:
    head_names = ("fc.", "se.")
    head_params = []
    backbone_params = []
    for name, param in model.named_parameters():
        if name.startswith(head_names) or ".fc." in name or ".se." in name:
            head_params.append(param)
        else:
            backbone_params.append(param)
    return [
        {"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay},
        {"params": head_params, "lr": head_lr, "weight_decay": weight_decay},
    ]


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    head_names = ("fc.", "se.")
    for name, param in model.named_parameters():
        is_head = name.startswith(head_names) or ".fc." in name or ".se." in name
        param.requires_grad = trainable or is_head
