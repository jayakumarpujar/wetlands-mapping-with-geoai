#!/usr/bin/env python

"""PPR Wetland Mapping Experiment — End-to-end pipeline.

Runs the full Phase 1-4 pipeline on the Prairie Pothole Region study area,
trains multiple architectures, and generates accuracy metrics for the paper.

Usage:
    python -m research_paper.run_experiment --output-root ./experiment_output
    python -m research_paper.run_experiment --help

References:
    Wu et al. (2019) RSE: LiDAR + multi-temporal NAIP wetland mapping
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from research_paper.wetland import (
    COWARDIN_CLASSES,
    EXPERIMENT_DEFAULTS,
    PPR_STUDY_AREA,
    build_experiment_config,
    compare_with_nwi,
    compute_spectral_indices,
    create_wetland_composite,
    download_3dep_dem,
    download_naip_timeseries,
    download_nwi,
    merge_dem_tiles,
    export_training_tiles,
    extract_surface_depressions,
    format_results_table,
    generate_weak_labels,
    map_wetland_dynamics,
    predict_wetlands,
    reclassify_nwi,
    save_experiment_results,
    train_wetland_model,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------


def run_data_download(config: Dict[str, Any]) -> Dict[str, Any]:
    """Phase 1a: Download NAIP, DEM, and NWI data."""
    paths = config["paths"]
    bbox = config["study_area"]["bbox"]
    years = config["study_area"]["naip_years"]

    # Retry settings from overrides (useful for flaky Colab networks)
    retry_kwargs = {
        "max_retries": config["training"].get("download_max_retries", 5),
        "timeout": config["training"].get("download_timeout", 300),
    }

    # Support pre-downloaded NAIP files (skip STAC API)
    pre_naip = config["training"].get("pre_downloaded_naip")
    if pre_naip:
        logger.info("Using %d pre-downloaded NAIP year(s)", len(pre_naip))
        naip_files = {int(yr): [str(p) for p in paths_list]
                      for yr, paths_list in pre_naip.items()}
    else:
        logger.info("Downloading NAIP timeseries for years %s ...", years)
        naip_files = download_naip_timeseries(
            bbox=bbox,
            output_dir=paths["naip_dir"],
            years=years,
            **retry_kwargs,
        )

    # Support pre-downloaded DEM tiles (skip 3DEP API)
    pre_dem_tiles = config["training"].get("pre_downloaded_dem_tiles")
    pre_dem_path = config["training"].get("pre_downloaded_dem")
    if pre_dem_path:
        logger.info("Using pre-downloaded DEM: %s", pre_dem_path)
        dem_path = str(pre_dem_path)
    elif pre_dem_tiles:
        logger.info("Merging %d pre-downloaded DEM tiles ...", len(pre_dem_tiles))
        dem_path = merge_dem_tiles(
            tile_paths=pre_dem_tiles,
            output_path=paths["dem_path"],
            bbox=bbox,
            overwrite=True,
        )
    else:
        logger.info("Downloading 3DEP DEM ...")
        dem_path = download_3dep_dem(
            bbox=bbox,
            output_path=paths["dem_path"],
            resolution=config["training"].get("dem_resolution", 1),
            **retry_kwargs,
        )

    logger.info("Downloading NWI ...")
    nwi_path = download_nwi(
        bbox=bbox,
        output_path=paths["nwi_path"],
        **retry_kwargs,
    )

    return {
        "naip_files": naip_files,
        "dem_path": dem_path,
        "nwi_path": nwi_path,
    }


def run_composites(
    config: Dict[str, Any], download_result: Dict[str, Any]
) -> Dict[str, List[str]]:
    """Phase 1b: Compute indices, depressions, and create composites."""
    paths = config["paths"]
    composites_dir = Path(paths["composites_dir"])
    composites_dir.mkdir(parents=True, exist_ok=True)

    dem_path = download_result["dem_path"]
    naip_files = download_result["naip_files"]

    # Extract surface depressions from DEM
    logger.info("Extracting surface depressions ...")
    depression_path = str(composites_dir / "depression_depth.tif")
    extract_surface_depressions(
        dem_path=dem_path,
        output_path=depression_path,
        min_depth=config["training"].get("depression_min_depth", 0.1),
    )

    # Build per-epoch composites
    composite_paths = []
    ndvi_paths = []
    ndwi_paths = []

    for year, year_files in sorted(naip_files.items()):
        if not year_files:
            logger.warning("No NAIP files for year %d, skipping.", year)
            continue

        # Use the first file per year (covers the study area)
        naip_path = year_files[0]

        # Compute spectral indices (multi-band: NDVI=band1, NDWI=band2)
        indices_path = str(composites_dir / f"indices_{year}.tif")
        compute_spectral_indices(
            naip_path=naip_path,
            output_path=indices_path,
        )

        # Extract single-band NDVI and NDWI for temporal stability filtering
        import rasterio

        ndvi_path = str(composites_dir / f"ndvi_{year}.tif")
        ndwi_path = str(composites_dir / f"ndwi_{year}.tif")
        with rasterio.open(indices_path) as src:
            profile = src.profile.copy()
            profile.update(count=1)
            ndvi_data = src.read(1)
            ndwi_data = src.read(2)
            with rasterio.open(ndvi_path, "w", **profile) as dst:
                dst.write(ndvi_data, 1)
            with rasterio.open(ndwi_path, "w", **profile) as dst:
                dst.write(ndwi_data, 1)

        ndvi_paths.append(ndvi_path)
        ndwi_paths.append(ndwi_path)

        # Create multi-band composite
        composite_path = str(composites_dir / f"composite_{year}.tif")
        create_wetland_composite(
            naip_paths=[naip_path],
            dem_path=dem_path,
            output_path=composite_path,
            include_depressions=True,
        )
        composite_paths.append(composite_path)

    return {
        "composite_paths": composite_paths,
        "ndvi_paths": ndvi_paths,
        "ndwi_paths": ndwi_paths,
        "depression_path": depression_path,
    }


def run_weak_labels(
    config: Dict[str, Any],
    download_result: Dict[str, Any],
    composite_result: Dict[str, List[str]],
) -> Dict[str, str]:
    """Phase 2: Generate weak labels and export training tiles."""
    paths = config["paths"]
    composites_dir = Path(paths["composites_dir"])
    tiles_dir = paths["tiles_dir"]

    if not composite_result["composite_paths"]:
        raise RuntimeError(
            "No composite rasters were produced. "
            "Check that NAIP files were downloaded for the configured years."
        )

    # Reclassify NWI to Cowardin classes
    nwi_raster_path = str(composites_dir / "nwi_raster.tif")
    reclassify_nwi(
        nwi_path=download_result["nwi_path"],
        raster_template=composite_result["composite_paths"][0],
        output_path=nwi_raster_path,
    )

    # Generate weak labels with depression + temporal filtering
    weak_label_path = str(composites_dir / "weak_labels.tif")

    generate_weak_labels(
        nwi_raster_path=nwi_raster_path,
        depression_path=composite_result["depression_path"],
        ndvi_paths=composite_result["ndvi_paths"],
        ndwi_paths=composite_result["ndwi_paths"],
        output_path=weak_label_path,
    )

    # Export training tiles
    tile_result = export_training_tiles(
        composite_path=composite_result["composite_paths"][0],
        label_path=weak_label_path,
        output_dir=tiles_dir,
        tile_size=config["training"]["tile_size"],
    )

    logger.info("Exported %d training tiles.", tile_result["num_tiles"])

    return {
        "nwi_raster_path": nwi_raster_path,
        "weak_label_path": weak_label_path,
        "tiles_dir": tiles_dir,
    }


def run_training(
    config: Dict[str, Any], label_result: Dict[str, str]
) -> List[Dict[str, Any]]:
    """Phase 3: Train models for each architecture configuration."""
    paths = config["paths"]
    training = config["training"]
    models_dir = Path(paths["models_dir"])

    trained_models = []
    for arch_cfg in config["architectures"]:
        arch_name = arch_cfg["architecture"]
        encoder = arch_cfg.get("encoder_name", "resnet50")
        model_dir = str(models_dir / f"{arch_name}_{encoder}")

        logger.info("Training %s with %s encoder ...", arch_name, encoder)

        result = train_wetland_model(
            tiles_dir=label_result["tiles_dir"],
            output_dir=model_dir,
            architecture=arch_name,
            encoder_name=encoder,
            num_classes=training["num_classes"],
            in_channels=training["in_channels"],
            num_epochs=training["num_epochs"],
            batch_size=training["batch_size"],
            learning_rate=training["learning_rate"],
            loss_function=training["loss_function"],
            use_class_weights=training["use_class_weights"],
            val_split=training["val_split"],
            seed=training["seed"],
            overwrite=True,
        )
        trained_models.append(
            {
                **result,
                "display_name": f"{arch_name} ({encoder})",
            }
        )

    return trained_models


def run_inference(
    config: Dict[str, Any],
    trained_models: List[Dict[str, Any]],
    composite_result: Dict[str, List[str]],
) -> List[Dict[str, Any]]:
    """Phase 4a: Run inference for each model on each epoch composite."""
    paths = config["paths"]
    training = config["training"]
    predictions_dir = Path(paths["predictions_dir"])
    predictions_dir.mkdir(parents=True, exist_ok=True)

    all_predictions = []
    for model_info in trained_models:
        model_predictions = []
        arch = model_info["architecture"]
        encoder = model_info["encoder_name"]

        for composite_path in composite_result["composite_paths"]:
            epoch_name = Path(composite_path).stem
            pred_path = str(
                predictions_dir / f"pred_{arch}_{encoder}_{epoch_name}.tif"
            )

            logger.info("Predicting %s with %s ...", epoch_name, arch)
            predict_wetlands(
                model_path=model_info["model_path"],
                composite_path=composite_path,
                output_path=pred_path,
                architecture=arch,
                encoder_name=encoder,
                num_classes=training["num_classes"],
                in_channels=training["in_channels"],
                overwrite=True,
            )
            model_predictions.append(pred_path)

        all_predictions.append(
            {
                "model_info": model_info,
                "prediction_paths": model_predictions,
            }
        )

    return all_predictions


def run_evaluation(
    config: Dict[str, Any],
    all_predictions: List[Dict[str, Any]],
    label_result: Dict[str, str],
) -> List[Dict[str, Any]]:
    """Phase 4b: Evaluate predictions, compute dynamics, and compare with NWI."""
    paths = config["paths"]
    results_dir = Path(paths["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for pred_info in all_predictions:
        model = pred_info["model_info"]
        pred_paths = pred_info["prediction_paths"]

        # Accuracy assessment against NWI reference
        accuracy = compare_with_nwi(
            prediction_path=pred_paths[0],
            reference_path=label_result["nwi_raster_path"],
        )

        # Dynamics mapping (if multiple epochs)
        dynamics_result = None
        if len(pred_paths) >= 2:
            dynamics_path = str(
                results_dir
                / f"dynamics_{model['architecture']}_{model['encoder_name']}.tif"
            )
            dynamics_result = map_wetland_dynamics(
                prediction_paths=pred_paths,
                output_path=dynamics_path,
                overwrite=True,
            )

        result = {
            "name": model.get("display_name", model["architecture"]),
            "architecture": model["architecture"],
            "encoder_name": model["encoder_name"],
            **accuracy,
        }
        if dynamics_result:
            result["dynamics"] = dynamics_result["statistics"]

        all_results.append(result)

    return all_results


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def run_ppr_experiment(
    output_root: str = "./ppr_experiment",
    overrides: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Run the full PPR wetland mapping experiment.

    Executes Phase 1-4 in sequence, trains both U-Net++ and DeepLabV3+,
    and saves structured results for the paper.

    Args:
        output_root: Root directory for all outputs.
        overrides: Optional training hyperparameter overrides.

    Returns:
        Dict with ``config``, ``results``, and ``output_path`` keys.
    """
    t0 = time.time()

    # Build configuration
    config = build_experiment_config(
        study_area=PPR_STUDY_AREA,
        output_root=output_root,
        overrides=overrides,
    )
    logger.info(
        "Experiment config: %d architectures, %d epochs",
        len(config["architectures"]),
        config["training"]["num_epochs"],
    )

    # Phase 1: Data download and composites
    download_result = run_data_download(config)
    composite_result = run_composites(config, download_result)

    # Phase 2: Weak labels and tiles
    label_result = run_weak_labels(config, download_result, composite_result)

    # Phase 3: Training
    trained_models = run_training(config, label_result)

    # Phase 4: Inference and evaluation
    all_predictions = run_inference(config, trained_models, composite_result)
    all_results = run_evaluation(config, all_predictions, label_result)

    # Save results
    results_path = str(Path(config["paths"]["results_dir"]) / "experiment_results.json")
    save_experiment_results(
        results=all_results,
        output_path=results_path,
        config=config,
    )

    # Print summary
    table = format_results_table(all_results)
    elapsed = time.time() - t0
    logger.info("Experiment completed in %.1f seconds.", elapsed)
    print("\n" + table + "\n")

    return {
        "config": config,
        "results": all_results,
        "output_path": results_path,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Run PPR wetland mapping experiment."
    )
    parser.add_argument(
        "--output-root",
        default="./ppr_experiment",
        help="Root directory for experiment outputs (default: ./ppr_experiment)",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help="Override number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override learning rate",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    overrides = {}
    if args.num_epochs is not None:
        overrides["num_epochs"] = args.num_epochs
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        overrides["learning_rate"] = args.learning_rate

    run_ppr_experiment(
        output_root=args.output_root,
        overrides=overrides or None,
    )


if __name__ == "__main__":
    main()
