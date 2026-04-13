#!/usr/bin/env python3
"""HPC runner for WetMamba benchmark experiments.

Runs all benchmark models + ablation study on pre-generated PPR tiles.
Assumes tiles already exist from the data pipeline (run_ppr_hpc.py).

Quick start (interactive)
-------------------------
    python research_paper/run_benchmark_hpc.py \
        --data-dir /scratch/$USER/wetlands/tiles \
        --output-dir /scratch/$USER/wetlands/experiments \
        --model wetmamba --epochs 100

Run all benchmarks
------------------
    python research_paper/run_benchmark_hpc.py \
        --data-dir /scratch/$USER/wetlands/tiles \
        --output-dir /scratch/$USER/wetlands/experiments \
        --model all --epochs 100

Run ablation study
------------------
    python research_paper/run_benchmark_hpc.py \
        --data-dir /scratch/$USER/wetlands/tiles \
        --output-dir /scratch/$USER/wetlands/experiments \
        --model wetmamba --ablation all --epochs 100

SLURM submission
----------------
    sbatch research_paper/slurm_benchmark.sh
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="WetMamba benchmark training on HPC.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--data-dir", required=True, help="Path to tile directory (with train/ and val/ subdirs)")
    parser.add_argument("--output-dir", default="experiments", help="Output directory for results")
    parser.add_argument("--model", default="wetmamba", help="Model name or 'all' for full benchmark")
    parser.add_argument("--ablation", default=None, help="Ablation variant or 'all'")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--n-epochs-temporal", type=int, default=2, help="Number of NAIP temporal epochs")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument("--wandb-project", default="wetmamba")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    # Logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger(__name__)

    # Verify data dir
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        log.error("Data dir not found: %s", data_dir)
        log.error("Run the data pipeline first: python research_paper/run_ppr_hpc.py")
        raise SystemExit(1)

    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    if train_dir.exists():
        n_train = len(list(train_dir.glob("*.npz")))
        log.info("Train tiles: %d", n_train)
    else:
        log.warning("No train/ subdir found — will use synthetic data")

    if val_dir.exists():
        n_val = len(list(val_dir.glob("*.npz")))
        log.info("Val tiles: %d", n_val)

    # GPU info
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                log.info("GPU %d: %s (%.1f GB)", i, torch.cuda.get_device_name(i),
                         torch.cuda.get_device_properties(i).total_mem / 1e9)
            device = "cuda"
        else:
            log.warning("No GPU detected — training will be slow")
            device = "cpu"
    except ImportError:
        log.error("PyTorch not installed")
        raise SystemExit(1)

    # Import training module
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root))

    from research_paper.train_benchmark import (
        TrainConfig,
        run_ablation_study,
        run_all_benchmarks,
        train,
    )

    config = TrainConfig(
        model_name=args.model,
        ablation=args.ablation if args.ablation != "all" else None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        tile_size=args.tile_size,
        data_dir=str(data_dir),
        n_epochs_temporal=args.n_epochs_temporal,
        output_dir=args.output_dir,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        device=device,
        num_workers=args.num_workers,
    )

    log.info("=" * 60)
    log.info("WetMamba Benchmark — %s", config.experiment_name())
    log.info("=" * 60)

    t0 = time.time()

    try:
        if args.model == "all":
            run_all_benchmarks(config)
        elif args.ablation == "all":
            run_ablation_study(config)
        else:
            results = train(config)
            log.info("Final mIoU: %.4f, OA: %.4f", results["mean_iou"], results["overall_accuracy"])
    except KeyboardInterrupt:
        log.warning("Interrupted after %.0fs", time.time() - t0)
        raise SystemExit(130)
    except Exception:
        log.exception("Failed after %.0fs", time.time() - t0)
        raise SystemExit(1)

    log.info("Completed in %.1f min", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
