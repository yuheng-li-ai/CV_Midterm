from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

from cvhw2.classification.data import build_classification_loaders
from cvhw2.classification.models import build_classifier, parameter_groups, set_backbone_trainable
from cvhw2.utils.config import ensure_dir, load_config
from cvhw2.utils.logger import ExperimentLogger
from cvhw2.utils.metrics import accuracy
from cvhw2.utils.plots import plot_metrics_csv
from cvhw2.utils.seed import seed_everything
from cvhw2.utils.torch import get_device, save_checkpoint


def train_classification(config_path: str) -> None:
    cfg = load_config(config_path)
    seed_everything(int(cfg.get("seed", 42)))
    device = get_device(cfg.get("device", "auto"))

    run_dir = ensure_dir(cfg.get("run_dir", "runs/classification/default"))
    ckpt_dir = ensure_dir(cfg.get("checkpoint_dir", "checkpoints/classification/default"))
    logger = ExperimentLogger(
        run_dir,
        backend=cfg.get("log_backend", "csv"),
        project=cfg.get("project", "cvhw2-classification"),
        name=cfg.get("name"),
    )

    train_loader, val_loader, test_loader = build_classification_loaders(cfg)
    model = build_classifier(cfg).to(device)
    freeze_backbone_epochs = int(cfg.get("freeze_backbone_epochs", 0))
    if freeze_backbone_epochs > 0:
        set_backbone_trainable(model, trainable=False)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(cfg.get("label_smoothing", 0.0)))
    optimizer = torch.optim.AdamW(
        parameter_groups(
            model,
            backbone_lr=float(cfg.get("backbone_lr", 1e-4)),
            head_lr=float(cfg.get("head_lr", 1e-3)),
            weight_decay=float(cfg.get("weight_decay", 1e-4)),
        )
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(cfg.get("epochs", 20)))

    best_acc = -1.0
    epochs = int(cfg.get("epochs", 20))
    for epoch in range(1, epochs + 1):
        if freeze_backbone_epochs > 0:
            set_backbone_trainable(model, trainable=epoch > freeze_backbone_epochs)
        train_loss, train_acc = _run_epoch(model, train_loader, criterion, device, optimizer, max_batches=cfg.get("max_train_batches"))
        val_loss, val_acc = _run_epoch(model, val_loader, criterion, device, max_batches=cfg.get("max_eval_batches"))

        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "lr_backbone": optimizer.param_groups[0]["lr"],
            "lr_head": optimizer.param_groups[1]["lr"],
        }
        logger.log(metrics)
        save_checkpoint(Path(ckpt_dir) / "last.pt", epoch=epoch, model=model.state_dict(), config=cfg, metrics=metrics)
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(Path(ckpt_dir) / "best.pt", epoch=epoch, model=model.state_dict(), config=cfg, metrics=metrics)
        scheduler.step()

    best_path = Path(ckpt_dir) / "best.pt"
    if best_path.exists():
        best_payload = torch.load(best_path, map_location=device)
        model.load_state_dict(best_payload["model"])
    test_loss, test_acc = _run_epoch(model, test_loader, criterion, device, max_batches=cfg.get("max_eval_batches"))
    logger.log({"epoch": epochs, "test_loss": test_loss, "test_accuracy": test_acc, "best_val_accuracy": best_acc})
    logger.close()
    try:
        plot_metrics_csv(
            run_dir / "metrics.csv",
            Path(cfg.get("curve_path", run_dir / "curves.png")),
            ["train_loss", "val_loss", "test_loss", "train_accuracy", "val_accuracy", "test_accuracy"],
        )
    except Exception as exc:  # pragma: no cover - plotting must not invalidate a completed training run.
        print(f"Warning: failed to plot classification curves: {exc}")


def _run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    max_batches: int | None = None,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    loss_sum = 0.0
    acc_sum = 0.0
    n = 0
    if max_batches is not None:
        max_batches = int(max_batches)
    for batch_idx, (images, targets) in enumerate(tqdm(loader, leave=False)):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, targets)
            if is_train:
                loss.backward()
                optimizer.step()
        batch = images.size(0)
        loss_sum += loss.item() * batch
        acc_sum += accuracy(logits.detach(), targets) * batch
        n += batch
    return loss_sum / max(n, 1), acc_sum / max(n, 1)
