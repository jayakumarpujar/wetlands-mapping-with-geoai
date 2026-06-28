#!/usr/bin/env python3
"""Systematic experiment sweep: architecture × loss × class-weights.

Trains all combinations on pre-existing tiles, evaluates on test tiles,
and writes a unified comparison JSON + summary table.

Usage (HPC, GPU node):
    python research_paper/run_experiment_sweep.py \
        --train-tiles /home/john_lab/shared/Jay/wetlands_data/tiles \
        --test-tiles  /home/john_lab/shared/Jay/wetlands_testdata/tiles \
        --output-root /home/john_lab/shared/Jay/wetlands_data/sweep \
        -v

SLURM example:
    #SBATCH --partition=gpu --gres=gpu:1 --mem=64G --time=24:00:00
    python research_paper/run_experiment_sweep.py \
        --train-tiles $DATA/tiles --test-tiles $TEST/tiles \
        --output-root $DATA/sweep -v
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Experiment grid
# ---------------------------------------------------------------------------

ARCHITECTURES = [
    {"architecture": "unetplusplus", "encoder_name": "resnet50"},
    {"architecture": "deeplabv3plus", "encoder_name": "resnet50"},
]

LOSS_CONFIGS = [
    {
        "name": "unified_focal_weighted",
        "loss_function": "unified_focal",
        "use_class_weights": True,
        "description": "Focal CE + Tversky blend with inverse-freq weights (recommended)",
    },
    {
        "name": "unified_focal_unweighted",
        "loss_function": "unified_focal",
        "use_class_weights": False,
        "description": "Focal CE + Tversky blend without class weights",
    },
    {
        "name": "focal_weighted",
        "loss_function": "focal",
        "use_class_weights": True,
        "description": "Standard focal loss with inverse-freq weights",
    },
    {
        "name": "crossentropy_weighted",
        "loss_function": "crossentropy",
        "use_class_weights": True,
        "description": "Plain CE with inverse-freq weights",
    },
    {
        "name": "dice_weighted",
        "loss_function": "dice",
        "use_class_weights": True,
        "description": "Dice loss with inverse-freq weights (optimises overlap)",
    },
]

# ---------------------------------------------------------------------------
# Fixed hyperparameters (tuned for PPR on V100-32GB)
# ---------------------------------------------------------------------------

FIXED_HPARAMS = {
    "learning_rate": 3e-4,
    "weight_decay": 1e-4,
    "num_epochs": 100,
    "batch_size": 32,
    "val_split": 0.2,
    "num_classes": 3,
    "in_channels": 10,
    "ignore_index": 255,
    "seed": 42,
    "max_class_weight": 50.0,
    "focal_alpha": 1.0,
    "focal_gamma": 2.0,
    "ufl_lambda": 0.5,
    "ufl_gamma": 0.75,
    "ufl_delta": 0.6,
    "collapse_check_epoch": 10,
    "collapse_miou_threshold": 0.10,
}


@dataclass(frozen=True)
class ExperimentRun:
    run_id: str
    architecture: str
    encoder_name: str
    loss_name: str
    loss_function: str
    use_class_weights: bool
    description: str


def _build_runs() -> List[ExperimentRun]:
    runs = []
    for arch in ARCHITECTURES:
        for loss_cfg in LOSS_CONFIGS:
            run_id = f"{arch['architecture']}_{arch['encoder_name']}_{loss_cfg['name']}"
            runs.append(ExperimentRun(
                run_id=run_id,
                architecture=arch["architecture"],
                encoder_name=arch["encoder_name"],
                loss_name=loss_cfg["name"],
                loss_function=loss_cfg["loss_function"],
                use_class_weights=loss_cfg["use_class_weights"],
                description=loss_cfg["description"],
            ))
    return runs


# ---------------------------------------------------------------------------
# Train one run
# ---------------------------------------------------------------------------

def train_one(
    run: ExperimentRun,
    train_tiles: Path,
    output_dir: Path,
    num_workers: int,
) -> Dict[str, Any]:
    from research_paper.wetland import train_wetland_model

    model_dir = output_dir / "models" / run.run_id
    best_model = model_dir / "best_model.pth"

    if best_model.exists():
        log.info("  SKIP (already trained): %s", best_model)
        return {"model_path": str(best_model), "skipped": True}

    log.info("  Training %s ...", run.run_id)
    t0 = time.time()

    result = train_wetland_model(
        tiles_dir=str(train_tiles),
        output_dir=str(model_dir),
        architecture=run.architecture,
        encoder_name=run.encoder_name,
        num_classes=FIXED_HPARAMS["num_classes"],
        in_channels=FIXED_HPARAMS["in_channels"],
        num_epochs=FIXED_HPARAMS["num_epochs"],
        batch_size=FIXED_HPARAMS["batch_size"],
        learning_rate=FIXED_HPARAMS["learning_rate"],
        weight_decay=FIXED_HPARAMS["weight_decay"],
        loss_function=run.loss_function,
        use_class_weights=run.use_class_weights,
        ignore_index=FIXED_HPARAMS["ignore_index"],
        val_split=FIXED_HPARAMS["val_split"],
        seed=FIXED_HPARAMS["seed"],
        max_class_weight=FIXED_HPARAMS["max_class_weight"],
        focal_alpha=FIXED_HPARAMS["focal_alpha"],
        focal_gamma=FIXED_HPARAMS["focal_gamma"],
        ufl_lambda=FIXED_HPARAMS["ufl_lambda"],
        ufl_gamma=FIXED_HPARAMS["ufl_gamma"],
        ufl_delta=FIXED_HPARAMS["ufl_delta"],
        num_workers=num_workers,
        collapse_check_epoch=FIXED_HPARAMS["collapse_check_epoch"],
        collapse_miou_threshold=FIXED_HPARAMS["collapse_miou_threshold"],
    )

    elapsed = time.time() - t0
    log.info("  Trained in %.0fs → %s", elapsed, result.get("model_path"))
    return {**result, "train_time_s": round(elapsed, 1), "skipped": False}


# ---------------------------------------------------------------------------
# Evaluate one run
# ---------------------------------------------------------------------------

def evaluate_one(
    run: ExperimentRun,
    model_path: str,
    test_tiles: Path,
    output_dir: Path,
    num_workers: int,
) -> Dict[str, Any]:
    from research_paper.evaluate_tiles import evaluate

    results_dir = output_dir / "results" / run.run_id
    metrics_file = results_dir / "test_metrics.json"

    if metrics_file.exists():
        log.info("  SKIP eval (already done): %s", metrics_file)
        with open(metrics_file) as f:
            return json.load(f)

    log.info("  Evaluating %s ...", run.run_id)
    metrics = evaluate(
        model_path=model_path,
        tiles_dir=str(test_tiles),
        output_dir=str(results_dir),
        architecture=run.architecture,
        encoder_name=run.encoder_name,
        in_channels=FIXED_HPARAMS["in_channels"],
        num_classes=FIXED_HPARAMS["num_classes"],
        batch_size=64,
        num_workers=num_workers,
    )
    return metrics


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _build_summary(
    all_results: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    summary_path = output_dir / "sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info("Summary JSON: %s", summary_path)

    header = (
        f"{'Run ID':<55} {'Arch':<15} {'Loss':<25} {'Wt':>3} "
        f"{'OA':>6} {'mIoU':>6} {'Upland':>7} {'Water':>7} {'Emerg':>7} "
        f"{'Time':>6}"
    )
    sep = "-" * len(header)

    lines = ["\n" + sep, header, sep]
    for r in sorted(all_results, key=lambda x: -x.get("mean_iou", 0)):
        pc = r.get("per_class", {})
        lines.append(
            f"{r['run_id']:<55} {r['architecture']:<15} "
            f"{r['loss_name']:<25} {'Y' if r['use_class_weights'] else 'N':>3} "
            f"{r.get('overall_accuracy', 0):>6.4f} "
            f"{r.get('mean_iou', 0):>6.4f} "
            f"{pc.get('Upland', {}).get('iou', 0):>7.4f} "
            f"{pc.get('Water', {}).get('iou', 0):>7.4f} "
            f"{pc.get('Emergent', {}).get('iou', 0):>7.4f} "
            f"{r.get('train_time_s', 0) / 60:>5.0f}m"
        )
    lines.append(sep)

    table = "\n".join(lines)
    log.info(table)

    table_path = output_dir / "sweep_table.txt"
    with open(table_path, "w") as f:
        f.write(table + "\n")
    log.info("Table: %s", table_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run systematic experiment sweep: arch × loss × weights",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--train-tiles", required=True, metavar="DIR",
        help="Directory with training images/ and labels/ (from export_training_tiles).",
    )
    parser.add_argument(
        "--test-tiles", required=True, metavar="DIR",
        help="Directory with test images/ and labels/ (from export_training_tiles).",
    )
    parser.add_argument(
        "--output-root", required=True, metavar="DIR",
        help="Root directory for sweep outputs (models/, results/, summary).",
    )
    parser.add_argument(
        "--num-workers", type=int, default=4,
        help="DataLoader workers (default: 4).",
    )
    parser.add_argument(
        "--skip-train", action="store_true",
        help="Skip training, only evaluate existing models.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="DEBUG-level logging.",
    )
    args = parser.parse_args()

    # Ensure repo root is on sys.path (handles running as a script)
    repo_root = str(Path(__file__).resolve().parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    for noisy in ("rasterio", "fiona", "pyproj", "urllib3", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    train_tiles = Path(args.train_tiles)
    test_tiles = Path(args.test_tiles)
    output_dir = Path(args.output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / "sweep.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-8s %(message)s", "%Y-%m-%d %H:%M:%S"
    ))
    logging.getLogger().addHandler(file_handler)

    runs = _build_runs()
    log.info("=" * 70)
    log.info("EXPERIMENT SWEEP: %d runs (%d archs × %d loss configs)",
             len(runs), len(ARCHITECTURES), len(LOSS_CONFIGS))
    log.info("  Train tiles : %s", train_tiles)
    log.info("  Test tiles  : %s", test_tiles)
    log.info("  Output      : %s", output_dir)
    log.info("  Fixed LR    : %g", FIXED_HPARAMS["learning_rate"])
    log.info("  Epochs      : %d", FIXED_HPARAMS["num_epochs"])
    log.info("  Batch size  : %d", FIXED_HPARAMS["batch_size"])
    log.info("  ignore_index: %d", FIXED_HPARAMS["ignore_index"])
    log.info("=" * 70)

    with open(output_dir / "sweep_config.json", "w") as f:
        json.dump({
            "fixed_hparams": FIXED_HPARAMS,
            "architectures": ARCHITECTURES,
            "loss_configs": LOSS_CONFIGS,
            "runs": [r.run_id for r in runs],
        }, f, indent=2)

    all_results: List[Dict[str, Any]] = []
    t_total = time.time()

    for i, run in enumerate(runs, 1):
        log.info("")
        log.info("=" * 70)
        log.info("[%d/%d] %s", i, len(runs), run.run_id)
        log.info("  %s | weights=%s", run.description, run.use_class_weights)
        log.info("=" * 70)

        result_entry: Dict[str, Any] = {
            "run_id": run.run_id,
            "architecture": run.architecture,
            "encoder_name": run.encoder_name,
            "loss_name": run.loss_name,
            "loss_function": run.loss_function,
            "use_class_weights": run.use_class_weights,
        }

        # --- Train ---
        model_path = None
        if not args.skip_train:
            try:
                train_result = train_one(
                    run, train_tiles, output_dir, args.num_workers,
                )
                model_path = train_result.get("model_path")
                result_entry["train_time_s"] = train_result.get("train_time_s", 0)
                result_entry["skipped_train"] = train_result.get("skipped", False)
            except Exception:
                log.exception("  TRAIN FAILED for %s — skipping.", run.run_id)
                result_entry["train_error"] = True
                all_results.append(result_entry)
                continue
        else:
            candidate = output_dir / "models" / run.run_id / "best_model.pth"
            if candidate.exists():
                model_path = str(candidate)
            else:
                log.warning("  No model found for %s, skipping eval.", run.run_id)
                all_results.append(result_entry)
                continue

        # --- Evaluate ---
        if model_path:
            try:
                metrics = evaluate_one(
                    run, model_path, test_tiles, output_dir, args.num_workers,
                )
                result_entry.update(metrics)
            except Exception:
                log.exception("  EVAL FAILED for %s", run.run_id)
                result_entry["eval_error"] = True

        all_results.append(result_entry)

        _build_summary(all_results, output_dir)

    elapsed = time.time() - t_total
    log.info("")
    log.info("=" * 70)
    log.info("SWEEP COMPLETE: %d runs in %.0f min", len(runs), elapsed / 60)
    log.info("=" * 70)

    _build_summary(all_results, output_dir)


if __name__ == "__main__":
    main()
