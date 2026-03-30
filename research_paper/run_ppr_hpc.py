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
        "--num-epochs", type=int, default=50, metavar="N",
        help="Number of training epochs (default: 50).",
    )
    train.add_argument(
        "--batch-size", type=int, default=8, metavar="N",
        help="Training batch size (default: 8).",
    )
    train.add_argument(
        "--learning-rate", type=float, default=1e-3, metavar="LR",
        help="Initial learning rate (default: 1e-3).",
    )
    train.add_argument(
        "--num-workers", type=int, default=4, metavar="N",
        help="DataLoader worker processes — set to CPU count (default: 4).",
    )
    train.add_argument(
        "--tile-size", type=int, default=256, metavar="PX",
        help="Training tile size in pixels (default: 256).",
    )
    train.add_argument(
        "--val-split", type=float, default=0.2, metavar="F",
        help="Validation fraction 0-1 (default: 0.2).",
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

    log.info("Output root : %s", output_root)
    log.info("Epochs      : %d", args.num_epochs)
    log.info("Batch size  : %d", args.batch_size)
    log.info("LR          : %g", args.learning_rate)
    log.info("Workers     : %d", args.num_workers)
    log.info("Tile size   : %d px", args.tile_size)
    log.info("Val split   : %.0f%%", args.val_split * 100)

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
        "num_workers": args.num_workers,
        "tile_size": args.tile_size,
        "val_split": args.val_split,
    }
    if args.dem_tiles:
        overrides["pre_downloaded_dem_tiles"] = args.dem_tiles
    if args.nwi_path:
        overrides["pre_downloaded_nwi"] = args.nwi_path

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
