"""Benchmark training script for WetMamba and baseline models.

Runs all 8 benchmark models on the PPR dataset with consistent hyperparameters.
Supports JSON logging (HPC) and optional WandB tracking.

Usage:
    # Single model
    python -m research_paper.train_benchmark --model wetmamba --epochs 100

    # All baselines
    python -m research_paper.train_benchmark --model all --epochs 100

    # Ablation variants
    python -m research_paper.train_benchmark --model wetmamba --ablation no_dag

    # WandB tracking
    python -m research_paper.train_benchmark --model wetmamba --wandb --wandb-project wetmamba

Models:
    wetmamba, unetplusplus, deeplabv3plus, segformer, swin_unet,
    unetmamba, prithvi_linear

Ablation variants (--ablation):
    no_dag          — WetMamba without Depression-Aware Gating
    no_temporal     — WetMamba without temporal SSM (concat epochs)
    no_mamba        — WetMamba with ViT decoder instead of Mamba
    no_pretrained   — WetMamba with random init encoder
    no_lora         — WetMamba with full fine-tuning
    no_weak_filter  — WetMamba trained on unfiltered NWI labels
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BENCHMARK_MODELS = [
    "wetmamba",
    "unetplusplus",
    "deeplabv3plus",
    "segformer",
    "swin_unet",
    "unetmamba",
    "prithvi_linear",
]

ABLATION_VARIANTS = {
    "no_dag": {"use_dag": False},
    "no_temporal": {"use_temporal": False},
    "no_pretrained": {"use_pretrained": False},
    "no_lora": {"use_lora": False},
}


@dataclass
class TrainConfig:
    """Training configuration."""

    model_name: str = "wetmamba"
    ablation: Optional[str] = None
    num_classes: int = 3
    input_channels: int = 10
    tile_size: int = 256
    batch_size: int = 8
    epochs: int = 100
    lr: float = 1e-4
    allow_proxy: bool = False
    weight_decay: float = 1e-4
    loss_function: str = "unified_focal"
    use_class_weights: bool = True
    ignore_index: int = 255
    max_class_weight: float = 50.0
    focal_gamma: float = 2.0
    ufl_lambda: float = 0.5
    ufl_gamma: float = 0.75
    ufl_delta: float = 0.6
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    num_workers: int = 4

    # Data
    data_dir: str = ""
    train_split: float = 0.8
    n_epochs_temporal: int = 2

    # Logging
    output_dir: str = "experiments"
    use_wandb: bool = False
    wandb_project: str = "wetmamba"
    log_interval: int = 10
    save_interval: int = 10

    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True

    def experiment_name(self) -> str:
        name = self.model_name
        if self.ablation:
            name = f"{name}_{self.ablation}"
        wt = "w" if self.use_class_weights else "nw"
        return f"{name}_{self.loss_function}_{wt}_ep{self.epochs}_bs{self.batch_size}"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int,
) -> Dict[str, float]:
    """Compute segmentation metrics.

    Args:
        pred: Predicted labels, shape (N,).
        target: Ground truth labels, shape (N,).
        num_classes: Number of classes.

    Returns:
        Dict with OA, mIoU, per-class IoU, per-class F1.
    """
    # Confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for p, t in zip(pred.flat, target.flat):
        if 0 <= t < num_classes:
            cm[t, p] += 1

    # Overall accuracy
    oa = np.diag(cm).sum() / max(cm.sum(), 1)

    # Per-class IoU and F1
    iou_per_class = {}
    f1_per_class = {}
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp

        iou = tp / max(tp + fp + fn, 1)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        iou_per_class[f"iou_class_{c}"] = float(iou)
        f1_per_class[f"f1_class_{c}"] = float(f1)

    miou = np.mean(list(iou_per_class.values()))
    mf1 = np.mean(list(f1_per_class.values()))

    return {
        "overall_accuracy": float(oa),
        "mean_iou": float(miou),
        "mean_f1": float(mf1),
        **iou_per_class,
        **f1_per_class,
    }


# ---------------------------------------------------------------------------
# JSON Logger
# ---------------------------------------------------------------------------

class JSONLogger:
    """Append-only JSON lines logger for HPC environments."""

    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, data: Dict[str, Any]) -> None:
        """Append one JSON line."""
        data["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        with open(self.log_path, "a") as f:
            f.write(json.dumps(data) + "\n")


# ---------------------------------------------------------------------------
# Dataset — loads GeoTIFF tiles from the data pipeline
# ---------------------------------------------------------------------------

class WetlandTileDataset(Dataset):
    """Loads paired image/label GeoTIFF tiles from export_training_tiles().

    The data pipeline outputs:
        <data_dir>/images/tile_RRRRRR_CCCCCC.tif  — multi-band composite
        <data_dir>/labels/tile_RRRRRR_CCCCCC.tif  — single-band labels

    Composite band layout (per epoch: 4 NAIP + 2 indices = 6):
        Epoch 1: R, G, B, NIR, NDVI, NDWI
        Epoch 2: R, G, B, NIR, NDVI, NDWI
        DEM elevation (1 band)
        Depression depth (1 band — last band, extracted for DAG)

    For train/val split, provide separate directories or use the split arg
    to partition a single tile directory.

    Args:
        data_dir: Path to tile directory (containing images/ and labels/).
        split: "train" or "val". Used for train/val partitioning.
        tile_size: Expected spatial dimensions.
        input_channels: Bands per epoch fed to model (excl. depression).
        n_epochs: Number of temporal NAIP epochs in the composite.
        val_fraction: Fraction of tiles for validation. Defaults to 0.2.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        tile_size: int = 256,
        input_channels: int = 7,
        n_epochs: int = 2,
        val_fraction: float = 0.2,
    ) -> None:
        self.tile_size = tile_size
        self.input_channels = input_channels
        self.n_epochs = n_epochs
        self.split = split

        data_path = Path(data_dir)

        # Support two directory layouts:
        # Layout A: data_dir/train/images/ + data_dir/val/images/ (pre-split)
        # Layout B: data_dir/images/ + data_dir/labels/ (single dir, auto-split)
        split_dir = data_path / split / "images"
        flat_dir = data_path / "images"

        if split_dir.exists():
            img_dir = split_dir
            self.label_dir = data_path / split / "labels"
        elif flat_dir.exists():
            img_dir = flat_dir
            self.label_dir = data_path / "labels"
        else:
            self.tile_paths: List[Path] = []
            self.label_dir = data_path / "labels"
            logger.warning("No tiles at %s or %s, using synthetic data", split_dir, flat_dir)
            return

        # Collect and sort tile paths
        all_tiles = sorted(img_dir.glob("*.tif"))

        if split_dir.exists():
            # Already pre-split
            self.tile_paths = all_tiles
        else:
            # Auto-split by deterministic index
            n_val = max(1, int(len(all_tiles) * val_fraction))
            if split == "val":
                self.tile_paths = all_tiles[:n_val]
            else:
                self.tile_paths = all_tiles[n_val:]

        logger.info("Loaded %d %s tiles from %s", len(self.tile_paths), split, img_dir)

    def __len__(self) -> int:
        return max(len(self.tile_paths), 100)  # min 100 for synthetic fallback

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx < len(self.tile_paths):
            return self._load_tile(idx)
        return self._synthetic_tile()

    def _load_tile(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load a GeoTIFF image/label tile pair.

        Composite band layout (2-epoch example, 14 bands total):
            [0:6]   Epoch 1: R, G, B, NIR, NDVI, NDWI
            [6:12]  Epoch 2: R, G, B, NIR, NDVI, NDWI
            [12]    DEM elevation
            [13]    Depression depth  ← extracted for DAG

        Extracts depression depth (last band) separately for DAG module.
        DEM stays in the composite as a shared feature.
        Per-epoch bands (6 each) are kept concatenated — WetMamba._parse_epochs
        splits them by input_channels for temporal modeling.
        """
        import rasterio

        img_path = self.tile_paths[idx]
        lbl_path = self.label_dir / img_path.name

        with rasterio.open(img_path) as src:
            composite = src.read().astype(np.float32)  # (C_total, H, W)

        with rasterio.open(lbl_path) as src:
            label = src.read(1).astype(np.int64)  # (H, W)

        # Last band = depression depth → separate for DAG module
        depression = composite[-1:, :, :]  # (1, H, W)
        composite = composite[:-1, :, :]    # (C_total-1, H, W)

        return {
            "composite": torch.from_numpy(composite),
            "depression": torch.from_numpy(depression),
            "label": torch.from_numpy(label),
        }

    def _synthetic_tile(self) -> Dict[str, torch.Tensor]:
        """Generate synthetic tile for testing/development."""
        C = self.input_channels * self.n_epochs
        composite = torch.randn(C, self.tile_size, self.tile_size)
        depression = torch.rand(1, self.tile_size, self.tile_size) * 2.0
        label = torch.randint(0, 6, (self.tile_size, self.tile_size))
        return {"composite": composite, "depression": depression, "label": label}


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    epoch: int,
    json_logger: JSONLogger,
    wandb_run: Optional[Any] = None,
) -> float:
    """Train for one epoch. Returns mean loss."""
    model.train()
    device = config.device
    total_loss = 0.0
    n_batches = 0

    scaler = torch.amp.GradScaler(enabled=config.mixed_precision and device == "cuda")

    for batch_idx, batch in enumerate(loader):
        composite = batch["composite"].to(device)
        depression = batch["depression"].to(device)
        label = batch["label"].to(device)

        optimizer.zero_grad()

        with torch.amp.autocast(
            device_type="cuda" if device == "cuda" else "cpu",
            enabled=config.mixed_precision,
        ):
            logits = model(composite, depression_depth=depression)
            loss = criterion(logits, label)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n_batches += 1

        if batch_idx % config.log_interval == 0:
            log_data = {
                "epoch": epoch,
                "batch": batch_idx,
                "loss": loss.item(),
                "lr": optimizer.param_groups[0]["lr"],
            }
            json_logger.log(log_data)
            if wandb_run is not None:
                wandb_run.log(log_data)

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    config: TrainConfig,
) -> Dict[str, float]:
    """Validate and compute metrics. Returns dict with loss + metrics."""
    model.eval()
    device = config.device
    total_loss = 0.0
    all_preds: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []

    for batch in loader:
        composite = batch["composite"].to(device)
        depression = batch["depression"].to(device)
        label = batch["label"].to(device)

        logits = model(composite, depression_depth=depression)
        loss = criterion(logits, label)
        total_loss += loss.item()

        preds = logits.argmax(dim=1).cpu().numpy()
        targets = label.cpu().numpy()
        all_preds.append(preds.flatten())
        all_targets.append(targets.flatten())

    all_preds_arr = np.concatenate(all_preds)
    all_targets_arr = np.concatenate(all_targets)

    metrics = compute_metrics(all_preds_arr, all_targets_arr, config.num_classes)
    metrics["val_loss"] = total_loss / max(len(loader), 1)
    return metrics


def build_optimizer(
    model: nn.Module,
    config: TrainConfig,
) -> torch.optim.Optimizer:
    """Build optimizer with optional differential learning rates."""
    if hasattr(model, "get_param_groups"):
        groups = model.get_param_groups()
        param_groups = [
            {"params": g["params"], "lr": config.lr * g["lr_scale"]}
            for g in groups
            if len(g["params"]) > 0
        ]
    else:
        param_groups = [{"params": model.parameters(), "lr": config.lr}]

    return torch.optim.AdamW(
        param_groups,
        lr=config.lr,
        weight_decay=config.weight_decay,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
) -> torch.optim.lr_scheduler._LRScheduler:
    """Build learning rate scheduler with warmup."""
    if config.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.epochs - config.warmup_epochs,
            eta_min=config.lr * 0.01,
        )
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)


# ---------------------------------------------------------------------------
# Loss builder
# ---------------------------------------------------------------------------

def _build_criterion(config: TrainConfig, dataset: "WetlandTileDataset") -> nn.Module:
    """Build loss function, optionally with inverse-frequency class weights."""
    ignore_index = config.ignore_index

    if config.use_class_weights:
        counts = np.zeros(config.num_classes, dtype=np.float64)
        for i in range(min(len(dataset.tile_paths), 500)):
            import rasterio
            lbl_path = dataset.label_dir / dataset.tile_paths[i].name
            try:
                with rasterio.open(lbl_path) as src:
                    lbl = src.read(1).ravel()
            except Exception:
                continue
            for c in range(config.num_classes):
                counts[c] += np.sum(lbl == c)
        counts = np.where(counts == 0, 1, counts)
        inv_freq = counts.sum() / (config.num_classes * counts)
        inv_freq = np.clip(inv_freq, 1.0, config.max_class_weight)
        weights = torch.tensor(inv_freq, dtype=torch.float32).to(config.device)
    else:
        weights = None

    lf = config.loss_function
    if lf == "crossentropy":
        return nn.CrossEntropyLoss(weight=weights, ignore_index=ignore_index)
    if lf == "focal":
        try:
            from geoai.losses import FocalLoss
            return FocalLoss(gamma=config.focal_gamma, weight=weights, ignore_index=ignore_index)
        except ImportError:
            return nn.CrossEntropyLoss(weight=weights, ignore_index=ignore_index)
    if lf in ("unified_focal", "ce_dice"):
        try:
            from geoai.losses import UnifiedFocalLoss
            return UnifiedFocalLoss(
                lmbda=config.ufl_lambda,
                gamma=config.ufl_gamma,
                delta=config.ufl_delta,
                weight=weights,
                ignore_index=ignore_index,
            )
        except ImportError:
            return nn.CrossEntropyLoss(weight=weights, ignore_index=ignore_index)
    if lf == "dice":
        try:
            from geoai.losses import DiceLoss
            return DiceLoss(weight=weights, ignore_index=ignore_index)
        except ImportError:
            return nn.CrossEntropyLoss(weight=weights, ignore_index=ignore_index)
    return nn.CrossEntropyLoss(weight=weights, ignore_index=ignore_index)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(config: TrainConfig) -> Dict[str, Any]:
    """Full training pipeline for a single model.

    Returns:
        Final validation metrics dict.
    """
    from research_paper.models.baselines import build_model

    exp_name = config.experiment_name()
    output_dir = Path(config.output_dir) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    json_logger = JSONLogger(output_dir / "train_log.jsonl")

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)

    logger.info("Starting experiment: %s", exp_name)

    # Build model with optional ablation overrides
    model_kwargs: Dict[str, Any] = {}
    if config.ablation and config.ablation in ABLATION_VARIANTS:
        model_kwargs.update(ABLATION_VARIANTS[config.ablation])

    model = build_model(
        config.model_name,
        num_classes=config.num_classes,
        input_channels=config.input_channels,
        allow_proxy=config.allow_proxy,
        **model_kwargs,
    )
    model = model.to(config.device)

    # Log param count
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Params: %dM total, %dM trainable", n_params // 1e6, n_trainable // 1e6)
    json_logger.log({"params_total": n_params, "params_trainable": n_trainable})

    # Data
    train_dataset = WetlandTileDataset(
        config.data_dir, "train", config.tile_size,
        config.input_channels, config.n_epochs_temporal,
    )
    val_dataset = WetlandTileDataset(
        config.data_dir, "val", config.tile_size,
        config.input_channels, config.n_epochs_temporal,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        shuffle=True, num_workers=config.num_workers,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size,
        shuffle=False, num_workers=config.num_workers,
        pin_memory=True,
    )

    # Loss, optimizer, scheduler
    criterion = _build_criterion(config, train_dataset)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    # WandB
    wandb_run = None
    if config.use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=config.wandb_project,
                name=exp_name,
                config=asdict(config),
            )
        except ImportError:
            logger.warning("wandb not installed, skipping")

    # Training loop
    best_miou = 0.0
    for epoch in range(config.epochs):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer,
            config, epoch, json_logger, wandb_run,
        )

        val_metrics = validate(model, val_loader, criterion, config)
        scheduler.step()

        # Log epoch summary
        epoch_log = {
            "epoch": epoch,
            "train_loss": train_loss,
            **val_metrics,
        }
        json_logger.log(epoch_log)
        if wandb_run is not None:
            wandb_run.log(epoch_log)

        logger.info(
            "Epoch %d/%d — train_loss: %.4f, val_mIoU: %.4f, val_OA: %.4f",
            epoch, config.epochs, train_loss,
            val_metrics["mean_iou"], val_metrics["overall_accuracy"],
        )

        # Save best model
        if val_metrics["mean_iou"] > best_miou:
            best_miou = val_metrics["mean_iou"]
            torch.save(
                {"model_state_dict": model.state_dict(), "epoch": epoch, **val_metrics},
                output_dir / "best_model.pt",
            )

        # Periodic checkpoint
        if (epoch + 1) % config.save_interval == 0:
            torch.save(
                {"model_state_dict": model.state_dict(), "epoch": epoch},
                output_dir / f"checkpoint_ep{epoch}.pt",
            )

    # Final results
    final_metrics = validate(model, val_loader, criterion, config)
    final_metrics["best_miou"] = best_miou

    with open(output_dir / "final_results.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

    if wandb_run is not None:
        wandb_run.finish()

    return final_metrics


def run_all_benchmarks(config: TrainConfig) -> None:
    """Run all benchmark models sequentially."""
    results = {}
    for model_name in BENCHMARK_MODELS:
        logger.info("=" * 60)
        logger.info("Running benchmark: %s", model_name)
        logger.info("=" * 60)

        model_config = TrainConfig(**{
            **asdict(config),
            "model_name": model_name,
            "ablation": None,
        })
        metrics = train(model_config)
        results[model_name] = metrics

    # Summary table
    summary_path = Path(config.output_dir) / "benchmark_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Benchmark summary saved to %s", summary_path)


def run_ablation_study(config: TrainConfig) -> None:
    """Run all ablation variants for WetMamba."""
    results = {}

    # Full WetMamba first
    full_config = TrainConfig(**{**asdict(config), "model_name": "wetmamba", "ablation": None})
    results["wetmamba_full"] = train(full_config)

    # Each ablation variant
    for ablation_name in ABLATION_VARIANTS:
        logger.info("Running ablation: %s", ablation_name)
        abl_config = TrainConfig(**{
            **asdict(config),
            "model_name": "wetmamba",
            "ablation": ablation_name,
        })
        results[f"wetmamba_{ablation_name}"] = train(abl_config)

    summary_path = Path(config.output_dir) / "ablation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Ablation summary saved to %s", summary_path)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="WetMamba Benchmark Training")

    # Model
    parser.add_argument("--model", default="wetmamba", help="Model name or 'all'")
    parser.add_argument("--ablation", default=None, choices=list(ABLATION_VARIANTS) + [None, "all"])

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--tile-size", type=int, default=256)

    # Data
    parser.add_argument("--data-dir", required=True, help="Path to tile directory")
    parser.add_argument("--n-epochs-temporal", type=int, default=2)

    # Logging
    parser.add_argument("--output-dir", default="experiments")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="wetmamba")

    # Hardware
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=4)

    # Safety
    parser.add_argument(
        "--allow-proxy", action="store_true",
        help="Allow CNN proxy fallback when FM weights unavailable. "
             "Off by default so benchmarks fail loudly on missing deps.",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config = TrainConfig(
        model_name=args.model,
        ablation=args.ablation if args.ablation != "all" else None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        tile_size=args.tile_size,
        data_dir=args.data_dir,
        n_epochs_temporal=args.n_epochs_temporal,
        output_dir=args.output_dir,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        device=args.device,
        num_workers=args.num_workers,
        allow_proxy=args.allow_proxy,
    )

    if args.model == "all":
        run_all_benchmarks(config)
    elif args.ablation == "all":
        run_ablation_study(config)
    else:
        train(config)


if __name__ == "__main__":
    main()
