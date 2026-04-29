from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch import nn
from tqdm import tqdm

from cvhw2.segmentation.data import build_segmentation_loaders
from cvhw2.segmentation.losses import CombinedSegmentationLoss
from cvhw2.segmentation.models import UNet
from cvhw2.utils.config import ensure_dir, load_config
from cvhw2.utils.logger import ExperimentLogger
from cvhw2.utils.metrics import mean_iou
from cvhw2.utils.plots import plot_metrics_csv
from cvhw2.utils.seed import seed_everything
from cvhw2.utils.torch import get_device, save_checkpoint


PALETTE = torch.tensor([[128, 0, 0], [0, 128, 0], [0, 0, 128]], dtype=torch.uint8)


def train_segmentation(config_path: str) -> None:
    cfg = load_config(config_path)
    seed_everything(int(cfg.get("seed", 42)))
    device = get_device(cfg.get("device", "auto"))

    run_dir = ensure_dir(cfg.get("run_dir", "runs/segmentation/default"))
    ckpt_dir = ensure_dir(cfg.get("checkpoint_dir", "checkpoints/segmentation/default"))
    logger = ExperimentLogger(
        run_dir,
        backend=cfg.get("log_backend", "csv"),
        project=cfg.get("project", "cvhw2-segmentation"),
        name=cfg.get("name"),
    )

    train_loader, val_loader, test_loader = build_segmentation_loaders(cfg)
    num_classes = int(cfg.get("num_classes", 3))
    model = UNet(num_classes=num_classes, base_channels=int(cfg.get("base_channels", 32))).to(device)
    criterion = CombinedSegmentationLoss(cfg.get("loss", "ce"), num_classes=num_classes, smooth=float(cfg.get("dice_smooth", 1.0)))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("lr", 1e-3)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(cfg.get("epochs", 30)))

    best_miou = -1.0
    epochs = int(cfg.get("epochs", 30))
    for epoch in range(1, epochs + 1):
        train_loss, train_miou = _run_epoch(
            model,
            train_loader,
            criterion,
            device,
            num_classes,
            optimizer,
            max_batches=cfg.get("max_train_batches"),
        )
        val_loss, val_miou = _run_epoch(
            model,
            val_loader,
            criterion,
            device,
            num_classes,
            max_batches=cfg.get("max_eval_batches"),
        )
        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_miou": train_miou,
            "val_loss": val_loss,
            "val_miou": val_miou,
            "lr": optimizer.param_groups[0]["lr"],
        }
        logger.log(metrics)
        save_checkpoint(Path(ckpt_dir) / "last.pt", epoch=epoch, model=model.state_dict(), config=cfg, metrics=metrics)
        if val_miou > best_miou:
            best_miou = val_miou
            save_checkpoint(Path(ckpt_dir) / "best.pt", epoch=epoch, model=model.state_dict(), config=cfg, metrics=metrics)
        scheduler.step()

    best_path = Path(ckpt_dir) / "best.pt"
    if best_path.exists():
        best_payload = torch.load(best_path, map_location=device)
        model.load_state_dict(best_payload["model"])
    test_loss, test_miou = _run_epoch(model, test_loader, criterion, device, num_classes, max_batches=cfg.get("max_eval_batches"))
    logger.log({"epoch": epochs, "test_loss": test_loss, "test_miou": test_miou, "best_val_miou": best_miou})
    save_visualizations(model, test_loader, device, run_dir / "visualizations", max_images=int(cfg.get("save_vis", 8)))
    logger.close()
    try:
        plot_metrics_csv(
            run_dir / "metrics.csv",
            Path(cfg.get("curve_path", run_dir / "curves.png")),
            ["train_loss", "val_loss", "test_loss", "train_miou", "val_miou", "test_miou"],
        )
    except Exception as exc:  # pragma: no cover - plotting must not invalidate a completed training run.
        print(f"Warning: failed to plot segmentation curves: {exc}")


def _run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    optimizer: torch.optim.Optimizer | None = None,
    max_batches: int | None = None,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    loss_sum = 0.0
    miou_sum = 0.0
    n = 0
    if max_batches is not None:
        max_batches = int(max_batches)
    for batch_idx, (images, masks) in enumerate(tqdm(loader, leave=False)):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, masks)
            if is_train:
                loss.backward()
                optimizer.step()
        batch = images.size(0)
        loss_sum += loss.item() * batch
        miou_sum += mean_iou(logits.detach(), masks, num_classes=num_classes) * batch
        n += batch
    return loss_sum / max(n, 1), miou_sum / max(n, 1)


@torch.no_grad()
def save_visualizations(model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device, out_dir: Path, max_images: int) -> None:
    ensure_dir(out_dir)
    model.eval()
    saved = 0
    for images, masks in loader:
        images = images.to(device)
        preds = model(images).argmax(dim=1).cpu()
        for pred, target in zip(preds, masks):
            pred_rgb = PALETTE[pred].numpy()
            target_rgb = PALETTE[target].numpy()
            Image.fromarray(pred_rgb).save(out_dir / f"{saved:03d}_pred.png")
            Image.fromarray(target_rgb).save(out_dir / f"{saved:03d}_target.png")
            saved += 1
            if saved >= max_images:
                return
