#!/usr/bin/env python3
"""HPC runner for the PPR wetland mapping experiment.

Downloads all data fresh from APIs (NAIP, 3DEP DEM, USFWS NWI)
and runs the full pipeline: composites → weak labels → training → evaluation.

Quick start
-----------
    # 1. Clone the repo
    git clone https://github.com/jayakumarpujar/wetlands-mapping-with-geoai.git
    cd wetlands-mapping-with-geoai

    # 2. Install dependencies
    pip install -r requirements.txt

    # 3. Run
    python research_paper/run_ppr_hpc.py --output-root /scratch/$USER/wetlands

SLURM example (save as slurm_wetlands.sh and run with sbatch)
--------------------------------------------------------------
    #!/bin/bash
    #SBATCH --job-name=wetlands_ppr
    #SBATCH --partition=gpu
    #SBATCH --gres=gpu:1
    #SBATCH --cpus-per-task=8
    #SBATCH --mem=128G
    #SBATCH --time=12:00:00
    #SBATCH --output=logs/wetlands_%j.out
    #SBATCH --error=logs/wetlands_%j.err

    module load python/3.11 cuda/12.1
    source $HOME/venvs/wetlands/bin/activate

    cd $HOME/wetlands-mapping-with-geoai
    git pull

    python research_paper/run_ppr_hpc.py \\
        --output-root /scratch/$USER/wetlands \\
        --num-epochs 100 \\
        --batch-size 16 \\
        --num-workers $SLURM_CPUS_PER_TASK
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run the PPR wetland mapping experiment on HPC.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- Required ---
    p.add_argument(
        "--output-root",
        required=True,
        metavar="DIR",
        help="Root directory for all outputs (naip/, composites/, tiles/, models/, results/).",
    )

    # --- Training hyperparameters ---
    train = p.add_argument_group("training hyperparameters")
    train.add_argument(
        "--num-epochs", type=int, default=100, metavar="N",
        help="Number of training epochs (default: 100).",
    )
    train.add_argument(
        "--batch-size", type=int, default=32, metavar="N",
        help="Training batch size (default: 32, V100-32GB FP32 headroom).",
    )
    train.add_argument(
        "--learning-rate", type=float, default=3e-4, metavar="LR",
        help="Initial learning rate (default: 3e-4; 1e-3 collapses on PPR).",
    )
    train.add_argument(
        "--weight-decay", type=float, default=1e-4, metavar="WD",
        help="Optimizer weight decay (default: 1e-4).",
    )
    train.add_argument(
        "--num-workers", type=int, default=4, metavar="N",
        help="DataLoader worker processes (default: 4).",
    )
    train.add_argument(
        "--tile-size", type=int, default=256, metavar="PX",
        help="Training tile size in pixels (default: 256).",
    )
    train.add_argument(
        "--tile-stride", type=int, default=None, metavar="PX",
        help=(
            "Stride between tiles in pixels (default: tile_size = no "
            "overlap). Set to tile_size/2 for 50%% overlap and ~4x more "
            "tiles when NAIP coverage is small."
        ),
    )
    train.add_argument(
        "--val-split", type=float, default=0.2, metavar="F",
        help="Validation fraction 0-1 (default: 0.2).",
    )

    # --- Class-imbalance mitigation ---
    imb = p.add_argument_group("class imbalance mitigation")
    imb.add_argument(
        "--loss-function", default="unified_focal", metavar="NAME",
        choices=[
            "crossentropy", "focal", "dice", "tversky",
            "unified_focal", "ce_dice",
        ],
        help=(
            "Loss function. unified_focal (default) blends focal CE + "
            "focal Tversky; ce_dice is a plain CE+Dice alias. Both handle "
            "severe imbalance (~95%% upland in PPR) better than pure CE."
        ),
    )
    imb.add_argument(
        "--no-class-weights", action="store_true",
        help="Disable inverse-frequency class weights (not recommended).",
    )
    imb.add_argument(
        "--ignore-index", type=int, default=-100, metavar="I",
        help=(
            "Class index to exclude from loss (default: -100, ignore none). "
            "Previous default 0 excluded upland and caused gradient "
            "starvation + collapse to trivial predictions."
        ),
    )
    imb.add_argument(
        "--max-class-weight", type=float, default=50.0, metavar="W",
        help="Cap on per-class weight after inverse-frequency (default: 50).",
    )
    imb.add_argument(
        "--focal-gamma", type=float, default=2.0,
        help="Focal loss focusing parameter (default: 2.0).",
    )
    imb.add_argument(
        "--ufl-lambda", type=float, default=0.5,
        help="UnifiedFocalLoss lambda [0=Tversky, 1=CE] (default: 0.5).",
    )
    imb.add_argument(
        "--ufl-gamma", type=float, default=0.75,
        help="UnifiedFocalLoss focusing parameter (default: 0.75).",
    )
    imb.add_argument(
        "--ufl-delta", type=float, default=0.6,
        help="UnifiedFocalLoss Tversky FN weight (default: 0.6).",
    )
    imb.add_argument(
        "--min-wetland-fraction", type=float, default=0.05, metavar="F",
        help=(
            "Drop tiles with less than this fraction of wetland pixels "
            "(default: 0.05). PPR is ~95%% upland — raise this to "
            "concentrate training on informative tiles."
        ),
    )
    imb.add_argument(
        "--oversample-threshold", type=float, default=0.20, metavar="F",
        help=(
            "Duplicate tiles with wetland fraction >= this value. "
            "Default 0.20. Set to 1.01 to disable."
        ),
    )
    imb.add_argument(
        "--oversample-factor", type=int, default=3, metavar="N",
        help="Copies for oversampled wetland-rich tiles (default: 3).",
    )

    # --- Auto-fallback on training collapse ---
    fb = p.add_argument_group("auto-fallback on training collapse")
    fb.add_argument(
        "--no-auto-fallback", action="store_true",
        help="Disable auto-retry with harder focal loss when val IoU collapses.",
    )
    fb.add_argument(
        "--collapse-check-epoch", type=int, default=10, metavar="N",
        help="Epoch after which collapse is evaluated (default: 10).",
    )
    fb.add_argument(
        "--collapse-miou-threshold", type=float, default=0.05, metavar="F",
        help="Val IoU below this past check epoch triggers fallback (default: 0.05).",
    )
    fb.add_argument(
        "--fallback-loss-function", default="focal", metavar="NAME",
        choices=[
            "crossentropy", "focal", "dice", "tversky",
            "unified_focal", "ce_dice",
        ],
        help="Loss to retry with on collapse (default: focal).",
    )
    fb.add_argument(
        "--fallback-focal-gamma", type=float, default=3.0,
        help="Focal gamma for fallback retry — harder focus (default: 3.0).",
    )
    fb.add_argument(
        "--fallback-max-class-weight", type=float, default=100.0,
        help="Raised class-weight cap for fallback retry (default: 100).",
    )

    # --- Pre-uploaded data (skip API downloads) ---
    data = p.add_argument_group("pre-uploaded data (skip API downloads)")
    data.add_argument(
        "--dem-tiles",
        nargs="+",
        metavar="FILE",
        help="Paths to pre-downloaded USGS DEM .tif tiles (skips 3DEP API).",
    )
    data.add_argument(
        "--nwi-path",
        metavar="FILE",
        help="Path to pre-downloaded NWI .gpkg or .shp file (skips USFWS API).",
    )
    data.add_argument(
        "--dem-resolution",
        type=int,
        default=10,
        choices=[1, 3, 10, 30, 60],
        metavar="M",
        help="3DEP DEM resolution in meters when downloading via API "
             "(default: 10). Full PPR bbox at 1m is ~50GB and will timeout; "
             "use 3 or 10 for full-extent downloads.",
    )

    # --- Logging ---
    p.add_argument(
        "--log-file", metavar="FILE",
        help="Write logs to this file in addition to stdout. "
             "Defaults to <output-root>/logs/experiment_<timestamp>.log.",
    )
    p.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable DEBUG-level logging.",
    )

    return p


def _setup_logging(output_root: Path, log_file: str | None, verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s %(levelname)-8s %(name)s  %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    if log_file:
        log_path = Path(log_file)
    else:
        logs_dir = output_root / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        log_path = logs_dir / f"experiment_{ts}.log"

    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_path),
    ]
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers)

    # Quiet noisy third-party loggers
    for noisy in ("rasterio", "fiona", "pyproj", "urllib3", "requests"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.info("Log file: %s", log_path)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    _setup_logging(output_root, args.log_file, args.verbose)
    log = logging.getLogger(__name__)

    log.info("Output root   : %s", output_root)
    log.info("Epochs        : %d", args.num_epochs)
    log.info("Batch size    : %d", args.batch_size)
    log.info("LR            : %g", args.learning_rate)
    log.info("Weight decay  : %g", args.weight_decay)
    log.info("Workers       : %d", args.num_workers)
    log.info("Tile size     : %d px", args.tile_size)
    log.info("Val split     : %.0f%%", args.val_split * 100)
    log.info("Loss          : %s", args.loss_function)
    log.info("Class weights : %s", not args.no_class_weights)
    log.info("Ignore index  : %d", args.ignore_index)
    log.info("Min wetland frac   : %.3f", args.min_wetland_fraction)
    log.info("Oversample thresh  : %.3f (x%d)",
             args.oversample_threshold, args.oversample_factor)

    # Import here so import errors are shown clearly
    try:
        repo_root = Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(repo_root))
        from research_paper.run_experiment import run_ppr_experiment
    except ImportError as exc:
        log.error("Import failed: %s", exc)
        log.error("Run: pip install -r requirements.txt")
        raise SystemExit(1)

    overrides = {
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "num_workers": args.num_workers,
        "tile_size": args.tile_size,
        "tile_stride": args.tile_stride,
        "val_split": args.val_split,
        "loss_function": args.loss_function,
        "use_class_weights": not args.no_class_weights,
        "ignore_index": args.ignore_index,
        "max_class_weight": args.max_class_weight,
        "focal_gamma": args.focal_gamma,
        "ufl_lambda": args.ufl_lambda,
        "ufl_gamma": args.ufl_gamma,
        "ufl_delta": args.ufl_delta,
        "min_wetland_fraction": args.min_wetland_fraction,
        "oversample_wetland_threshold": args.oversample_threshold,
        "oversample_factor": args.oversample_factor,
    }
    overrides["auto_fallback"] = not args.no_auto_fallback
    overrides["collapse_check_epoch"] = args.collapse_check_epoch
    overrides["collapse_miou_threshold"] = args.collapse_miou_threshold
    overrides["fallback_loss_function"] = args.fallback_loss_function
    overrides["fallback_focal_gamma"] = args.fallback_focal_gamma
    overrides["fallback_max_class_weight"] = args.fallback_max_class_weight
    if args.dem_tiles:
        overrides["pre_downloaded_dem_tiles"] = args.dem_tiles
    if args.nwi_path:
        overrides["pre_downloaded_nwi"] = args.nwi_path
    overrides["dem_resolution"] = args.dem_resolution

    log.info("Starting experiment ...")
    t0 = time.time()

    try:
        results = run_ppr_experiment(
            output_root=str(output_root),
            overrides=overrides,
        )
    except KeyboardInterrupt:
        log.warning("Interrupted after %.0fs.", time.time() - t0)
        raise SystemExit(130)
    except Exception:
        log.exception("Experiment failed after %.0fs.", time.time() - t0)
        raise SystemExit(1)

    log.info("Done in %.1fs. Results: %s", time.time() - t0, results.get("output_path"))


if __name__ == "__main__":
    main()
