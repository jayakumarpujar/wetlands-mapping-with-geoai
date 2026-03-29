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

import numpy as np

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
        # Check if NAIP files already exist on disk (resume after crash)
        naip_dir = Path(paths["naip_dir"])
        existing_naip: Dict[int, list] = {}
        for year in years:
            year_dir = naip_dir / str(year)
            if year_dir.exists():
                tifs = sorted(str(f) for f in year_dir.glob("*.tif"))
                if tifs:
                    existing_naip[year] = tifs
        if existing_naip and len(existing_naip) == len(years):
            logger.info("NAIP already exists for all years, reusing %s",
                        {y: len(f) for y, f in existing_naip.items()})
            naip_files = existing_naip
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
    elif Path(paths["dem_path"]).exists():
        logger.info("DEM already exists, reusing: %s", paths["dem_path"])
        dem_path = str(paths["dem_path"])
    else:
        logger.info("Downloading 3DEP DEM ...")
        dem_path = download_3dep_dem(
            bbox=bbox,
            output_path=paths["dem_path"],
            resolution=config["training"].get("dem_resolution", 1),
            **retry_kwargs,
        )

    # Support pre-downloaded NWI file (skip USFWS API)
    pre_nwi = config["training"].get("pre_downloaded_nwi")
    if pre_nwi:
        logger.info("Using pre-downloaded NWI: %s", pre_nwi)
        nwi_path = str(pre_nwi)
    elif Path(paths["nwi_path"]).exists():
        logger.info("NWI already exists, reusing: %s", paths["nwi_path"])
        nwi_path = str(paths["nwi_path"])
    else:
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
    years = sorted(naip_files.keys())

    training_composite_path = str(composites_dir / "training_composite.tif")
    depression_path = str(composites_dir / "depression_depth.tif")
    # Check for the mosaic marker — old single-tile composites must be rebuilt
    mosaic_marker = composites_dir / ".mosaic_complete"
    if Path(training_composite_path).exists() and mosaic_marker.exists():
        print("  Training composite (mosaicked) already exists, skipping.", flush=True)
        composite_paths = [str(p) for p in sorted(composites_dir.glob("composite_*.tif"))]
        ndvi_paths = [str(p) for p in sorted(composites_dir.glob("ndvi_*.tif"))]
        ndwi_paths = [str(p) for p in sorted(composites_dir.glob("ndwi_*.tif"))]
        if not Path(depression_path).exists():
            depression_path = None
        return {
            "composite_paths": composite_paths,
            "training_composite_path": training_composite_path,
            "ndvi_paths": ndvi_paths,
            "ndwi_paths": ndwi_paths,
            "depression_path": depression_path,
        }

    # Extract surface depressions from DEM
    depression_path = str(composites_dir / "depression_depth.tif")
    if Path(depression_path).exists():
        print("  Depression raster already exists, reusing.", flush=True)
    else:
        logger.info("Extracting surface depressions ...")
        extract_surface_depressions(
            dem_path=dem_path,
            output_path=depression_path,
            min_depth=config["training"].get("depression_min_depth", 0.1),
        )

    # Build per-epoch NAIP mosaics and indices
    # IMPORTANT: mosaic ALL tiles per year to cover the full study area.
    # Using a single tile leaves most wetlands outside the footprint.
    import rasterio
    from rasterio.merge import merge as rasterio_merge
    from rasterio.warp import Resampling, reproject

    composite_paths = []
    ndvi_paths = []
    ndwi_paths = []

    for year, year_files in sorted(naip_files.items()):
        if not year_files:
            logger.warning("No NAIP files for year %d, skipping.", year)
            continue

        # Mosaic all NAIP tiles for this year into a single raster
        mosaic_path = str(composites_dir / f"naip_mosaic_{year}.tif")
        if Path(mosaic_path).exists() and mosaic_marker.exists():
            with rasterio.open(mosaic_path) as _ms:
                print(f"  NAIP mosaic for {year} already exists ({_ms.width}x{_ms.height}), reusing.", flush=True)
        else:
            print(f"  Mosaicking {len(year_files)} NAIP tiles for {year} ...", flush=True)
            src_files = [rasterio.open(f) for f in year_files]
            mosaic_arr, mosaic_transform = rasterio_merge(src_files)
            mosaic_profile = src_files[0].profile.copy()
            mosaic_profile.update(
                height=mosaic_arr.shape[1],
                width=mosaic_arr.shape[2],
                transform=mosaic_transform,
                count=mosaic_arr.shape[0],
            )
            for s in src_files:
                s.close()
            with rasterio.open(mosaic_path, "w", **mosaic_profile) as dst:
                dst.write(mosaic_arr)
            print(f"  Mosaic {year}: {mosaic_arr.shape[2]}x{mosaic_arr.shape[1]} pixels", flush=True)

        # Compute spectral indices on the mosaic
        indices_path = str(composites_dir / f"indices_{year}.tif")
        ndvi_path = str(composites_dir / f"ndvi_{year}.tif")
        ndwi_path = str(composites_dir / f"ndwi_{year}.tif")

        # Check if existing indices match the mosaic dimensions
        indices_stale = False
        if Path(ndvi_path).exists() and Path(mosaic_path).exists():
            with rasterio.open(ndvi_path) as idx_src, rasterio.open(mosaic_path) as mos_src:
                if idx_src.width != mos_src.width or idx_src.height != mos_src.height:
                    print(f"  Indices for {year} are stale (wrong size), recomputing.", flush=True)
                    indices_stale = True

        if Path(ndvi_path).exists() and Path(ndwi_path).exists() and not indices_stale and mosaic_marker.exists():
            print(f"  Indices for {year} already exist, reusing.", flush=True)
        else:
            print(f"  Computing indices for {year} ...", flush=True)
            compute_spectral_indices(
                naip_path=mosaic_path,
                output_path=indices_path,
                overwrite=True,
            )
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

        # Create multi-band composite from mosaic
        composite_path = str(composites_dir / f"composite_{year}.tif")
        if Path(composite_path).exists() and mosaic_marker.exists():
            print(f"  Composite for {year} already exists, reusing.", flush=True)
        else:
            print(f"  Creating composite for {year} ...", flush=True)
            create_wetland_composite(
                naip_paths=[mosaic_path],
                dem_path=dem_path,
                output_path=composite_path,
                include_depressions=True,
                overwrite=True,
            )
        composite_paths.append(composite_path)

    # Build the 10-band training composite matching the research design:
    # Bands: NAIP_2015(R,G,B,NIR) + NDVI_2015 + NDWI_2015 + NDVI_2017 + NDWI_2017 + DEM + Depression
    # This stacks the primary epoch's NAIP with temporal indices from both
    # epochs plus topographic features into a single training input.
    training_composite_path = str(composites_dir / "training_composite.tif")

    if Path(training_composite_path).exists() and mosaic_marker.exists():
        print("  Training composite already exists, reusing.", flush=True)
    elif len(ndvi_paths) >= 2 and len(ndwi_paths) >= 2:
        print("  Building 10-band training composite from mosaics ...", flush=True)

        # Use the first year's MOSAIC as the spatial reference (full study area)
        first_year = sorted(naip_files.keys())[0]
        first_mosaic = str(composites_dir / f"naip_mosaic_{first_year}.tif")
        with rasterio.open(first_mosaic) as ref_src:
            ref_profile = ref_src.profile.copy()
            ref_transform = ref_src.transform
            ref_crs = ref_src.crs
            ref_h, ref_w = ref_src.height, ref_src.width

            # Bands 1-4: NAIP (R, G, B, NIR)
            bands = [ref_src.read(b).astype(np.float32) for b in range(1, min(ref_src.count, 4) + 1)]

        # Helper to read a raster aligned to the reference grid
        def _read_aligned(path):
            with rasterio.open(path) as src:
                if (src.height, src.width) != (ref_h, ref_w) or src.crs != ref_crs:
                    arr = np.empty((ref_h, ref_w), dtype=np.float32)
                    reproject(
                        source=rasterio.band(src, 1),
                        destination=arr,
                        dst_transform=ref_transform,
                        dst_crs=ref_crs,
                        resampling=Resampling.bilinear,
                    )
                    return arr
                return src.read(1).astype(np.float32)

        # Bands 5-6: NDVI and NDWI from first epoch
        bands.append(_read_aligned(ndvi_paths[0]))
        bands.append(_read_aligned(ndwi_paths[0]))

        # Bands 7-8: NDVI and NDWI from second epoch
        bands.append(_read_aligned(ndvi_paths[1]))
        bands.append(_read_aligned(ndwi_paths[1]))

        # Band 9: DEM elevation
        bands.append(_read_aligned(dem_path))

        # Band 10: Depression depth
        bands.append(_read_aligned(depression_path))

        ref_profile.update(dtype="float32", count=len(bands), nodata=None)
        with rasterio.open(training_composite_path, "w", **ref_profile) as dst:
            for i, band in enumerate(bands, start=1):
                dst.write(band, i)

        # Mark that this composite was built from mosaics (not single tiles)
        mosaic_marker.touch()
        print(
            f"  Created {len(bands)}-band training composite -> {training_composite_path}",
            flush=True,
        )
    elif composite_paths:
        # Fallback: single epoch — use that composite directly
        training_composite_path = composite_paths[0]
        print(
            f"  Single epoch: using {training_composite_path} as training composite",
            flush=True,
        )
    else:
        training_composite_path = None

    return {
        "composite_paths": composite_paths,
        "training_composite_path": training_composite_path,
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

    training_composite = composite_result.get("training_composite_path")
    if not training_composite and not composite_result["composite_paths"]:
        raise RuntimeError(
            "No composite rasters were produced. "
            "Check that NAIP files were downloaded for the configured years."
        )
    # Use the training composite (10-band) if available, else first per-epoch composite
    composite_for_tiles = training_composite or composite_result["composite_paths"][0]

    # Reclassify NWI to Cowardin classes (use same grid as training composite)
    # Always regenerate — ensures alignment with current training composite
    nwi_raster_path = str(composites_dir / "nwi_raster.tif")
    reclassify_nwi(
        nwi_path=download_result["nwi_path"],
        raster_template=composite_for_tiles,
        output_path=nwi_raster_path,
        overwrite=True,
    )

    # Generate weak labels with depression + temporal filtering
    # Always regenerate — thresholds may have changed between runs
    weak_label_path = str(composites_dir / "weak_labels.tif")
    generate_weak_labels(
        nwi_raster_path=nwi_raster_path,
        depression_path=composite_result["depression_path"],
        ndvi_paths=composite_result["ndvi_paths"],
        ndwi_paths=composite_result["ndwi_paths"],
        output_path=weak_label_path,
        overwrite=True,
    )

    # Export training tiles from the training composite
    tile_result = export_training_tiles(
        composite_path=composite_for_tiles,
        label_path=weak_label_path,
        output_dir=tiles_dir,
        tile_size=config["training"]["tile_size"],
        overwrite=True,
    )
    num_tiles = tile_result["num_tiles"]

    if num_tiles == 0:
        raise RuntimeError(
            "No training tiles were generated. Check that NWI covers the study area "
            "and that depression/stability thresholds are not too strict."
        )

    # Auto-detect actual band count from the training composite
    import rasterio as _rio
    with _rio.open(composite_for_tiles) as _src:
        actual_channels = _src.count
    config["training"]["in_channels"] = actual_channels

    print(
        f"  Exported {num_tiles} training tiles ({actual_channels} bands).",
        flush=True,
    )

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

        print(
            f"  Training {arch_name} ({encoder}) ...",
            flush=True,
        )

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
    """Phase 4a: Run inference for each model on the training composite.

    The model was trained on the 10-band training composite (NAIP + temporal
    indices + topography), so inference must use the same band structure.
    Per-epoch 8-band composites are incompatible with the trained model.
    """
    paths = config["paths"]
    training = config["training"]
    predictions_dir = Path(paths["predictions_dir"])
    predictions_dir.mkdir(parents=True, exist_ok=True)

    # Use the training composite for inference (same bands as training)
    training_composite = composite_result.get("training_composite_path")
    if not training_composite:
        training_composite = composite_result["composite_paths"][0]

    all_predictions = []
    for model_info in trained_models:
        arch = model_info["architecture"]
        encoder = model_info["encoder_name"]

        pred_path = str(
            predictions_dir / f"pred_{arch}_{encoder}.tif"
        )

        print(f"  Predicting with {arch} ({encoder}) ...", flush=True)
        predict_wetlands(
            model_path=model_info["model_path"],
            composite_path=training_composite,
            output_path=pred_path,
            architecture=arch,
            encoder_name=encoder,
            num_classes=training["num_classes"],
            in_channels=training["in_channels"],
            overwrite=True,
        )

        all_predictions.append(
            {
                "model_info": model_info,
                "prediction_paths": [pred_path],
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

    def _elapsed() -> str:
        return f"[{time.time() - t0:.0f}s]"

    # Build configuration
    config = build_experiment_config(
        study_area=PPR_STUDY_AREA,
        output_root=output_root,
        overrides=overrides,
    )
    print(
        f"{_elapsed()} Experiment: {len(config['architectures'])} architectures, "
        f"{config['training']['num_epochs']} epochs, output={output_root}",
        flush=True,
    )

    # Phase 1: Data download and composites
    print(f"{_elapsed()} Phase 1a: Downloading / loading data ...", flush=True)
    download_result = run_data_download(config)
    print(f"{_elapsed()} Phase 1a complete.", flush=True)

    print(f"{_elapsed()} Phase 1b: Computing composites & indices ...", flush=True)
    composite_result = run_composites(config, download_result)
    print(
        f"{_elapsed()} Phase 1b complete: {len(composite_result['composite_paths'])} composites.",
        flush=True,
    )

    # Phase 2: Weak labels and tiles
    print(f"{_elapsed()} Phase 2: Generating weak labels & tiles ...", flush=True)
    label_result = run_weak_labels(config, download_result, composite_result)
    print(f"{_elapsed()} Phase 2 complete.", flush=True)

    # Phase 3: Training
    print(f"{_elapsed()} Phase 3: Training models ...", flush=True)
    trained_models = run_training(config, label_result)
    print(
        f"{_elapsed()} Phase 3 complete: trained {len(trained_models)} model(s).",
        flush=True,
    )

    # Phase 4: Inference and evaluation
    print(f"{_elapsed()} Phase 4: Inference & evaluation ...", flush=True)
    all_predictions = run_inference(config, trained_models, composite_result)
    all_results = run_evaluation(config, all_predictions, label_result)
    print(f"{_elapsed()} Phase 4 complete.", flush=True)

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
    print(f"\nExperiment completed in {elapsed:.1f} seconds.\n", flush=True)
    print(table + "\n", flush=True)

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
