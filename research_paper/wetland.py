"""Wetland mapping data pipeline for multi-temporal NAIP + LiDAR research.

Phase 1 implementation: data download, spectral index computation,
depression extraction, and composite creation.

This module is standalone research code that imports from the geoai package
as a dependency. It is NOT part of the geoai package itself.

References:
    - Wu et al. (2019) RSE: LiDAR + multi-temporal NAIP wetland mapping
    - Igwe et al. (2026) RSASE: weakly supervised CNN wetland mapping
"""

from __future__ import annotations

import heapq
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from geoai.download import download_naip

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COWARDIN_CLASSES: Dict[int, str] = {
    0: "Upland",
    1: "Water",
    2: "Emergent",
    3: "Forested",
    4: "Scrub-Shrub",
    5: "Other",
}
"""Cowardin-based wetland classification schema (7-class simplified)."""

NWI_CODE_TO_CLASS: Dict[str, int] = {
    "L": 1,     # Lacustrine -> Water
    "R": 1,     # Riverine -> Water
    "PAB": 1,   # Palustrine Aquatic Bed -> Water
    "PUB": 1,   # Palustrine Unconsolidated Bottom -> Water
    "POW": 1,   # Palustrine Open Water -> Water
    "PEM": 2,   # Palustrine Emergent -> Emergent
    "PFO": 3,   # Palustrine Forested -> Forested
    "PSS": 4,   # Palustrine Scrub-Shrub -> Scrub-Shrub
}
"""Mapping from NWI Cowardin code prefixes to target class IDs."""

NAIP_BANDS: Dict[int, str] = {
    1: "Red",
    2: "Green",
    3: "Blue",
    4: "NIR",
}
"""NAIP band number to name mapping (1-indexed)."""

SPECTRAL_INDICES: Dict[str, str] = {
    "ndvi": "(NIR - Red) / (NIR + Red)",
    "ndwi": "(Green - NIR) / (Green + NIR)",
}
"""Supported spectral indices and their formulas."""

SUPPORTED_3DEP_RESOLUTIONS: frozenset = frozenset({1, 3, 10, 30})
"""Supported 3DEP DEM resolutions in meters."""

SUPPORTED_ARCHITECTURES: frozenset = frozenset(
    {
        "unet",
        "unetplusplus",
        "deeplabv3",
        "deeplabv3plus",
        "fpn",
        "pspnet",
        "linknet",
        "manet",
        "pan",
        "upernet",
    }
)
"""Supported segmentation architectures (via segmentation-models-pytorch)."""

PPR_STUDY_AREA: Dict[str, Any] = {
    "name": "Prairie Pothole Region, Central North Dakota",
    "bbox": (-100.55, 46.65, -99.15, 47.60),
    "naip_years": [2015, 2017],
    "huc8_codes": ["10160001", "10160002", "10130103"],
    "description": (
        "26 HUC-10 watersheds across James Headwaters, Pipestem, and "
        "Apple Creek subbasins — same study area as Wu 2019 RSE."
    ),
}
"""PPR study area configuration matching Wu 2019 RSE."""

EXPERIMENT_DEFAULTS: Dict[str, Any] = {
    "tile_size": 256,
    "num_epochs": 50,
    "batch_size": 8,
    "learning_rate": 1e-3,
    "val_split": 0.2,
    "num_classes": 6,
    "in_channels": 10,
    "loss_function": "focal",
    "use_class_weights": True,
    "seed": 42,
    "dem_resolution": 1,
    "depression_min_depth": 0.1,
    "architectures": [
        {"architecture": "unetplusplus", "encoder_name": "resnet50"},
        {"architecture": "deeplabv3plus", "encoder_name": "resnet50"},
    ],
}
"""Default experiment hyperparameters for the wetland mapping paper."""

__all__ = [
    "COWARDIN_CLASSES",
    "NWI_CODE_TO_CLASS",
    "NAIP_BANDS",
    "SPECTRAL_INDICES",
    "download_naip_timeseries",
    "download_3dep_dem",
    "download_nwi",
    "compute_spectral_indices",
    "extract_surface_depressions",
    "create_wetland_composite",
    "reclassify_nwi",
    "generate_weak_labels",
    "export_training_tiles",
    "SUPPORTED_ARCHITECTURES",
    "train_wetland_model",
    "predict_wetlands",
    "map_wetland_dynamics",
    "compare_with_nwi",
    "PPR_STUDY_AREA",
    "EXPERIMENT_DEFAULTS",
    "build_experiment_config",
    "format_results_table",
    "save_experiment_results",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_bbox(bbox: Tuple[float, float, float, float]) -> None:
    """Validate a bounding box tuple.

    Args:
        bbox: (min_lon, min_lat, max_lon, max_lat).

    Raises:
        ValueError: If bbox does not have 4 values or min >= max.
        TypeError: If values are not numeric.
    """
    if len(bbox) != 4:
        raise ValueError(f"bbox must have 4 values (min_lon, min_lat, max_lon, max_lat), got {len(bbox)}")

    min_lon, min_lat, max_lon, max_lat = (float(v) for v in bbox)

    if min_lon >= max_lon:
        raise ValueError(
            f"min_lon ({min_lon}) must be less than max_lon ({max_lon})"
        )
    if min_lat >= max_lat:
        raise ValueError(
            f"min_lat ({min_lat}) must be less than max_lat ({max_lat})"
        )


def _validate_index_names(
    indices: Optional[Sequence[str]],
) -> List[str]:
    """Validate spectral index names.

    Args:
        indices: List of index names, or None for all supported indices.

    Returns:
        Validated list of index names.

    Raises:
        ValueError: If any index name is not in SPECTRAL_INDICES.
    """
    if indices is None:
        return list(SPECTRAL_INDICES.keys())

    for name in indices:
        if name not in SPECTRAL_INDICES:
            raise ValueError(
                f"Unknown spectral index: {name}. "
                f"Supported indices: {list(SPECTRAL_INDICES.keys())}"
            )
    return list(indices)


def _parse_nwi_code(code: Optional[str]) -> int:
    """Parse an NWI Cowardin code to a target class ID.

    Args:
        code: NWI attribute code string (e.g., 'PEM1Ch', 'L1UBHh').

    Returns:
        Integer class ID from COWARDIN_CLASSES. Returns 5 (Other) for
        unrecognized or empty codes.
    """
    if not code:
        return 5

    upper = code.upper()

    # Check single-char system codes first (L, R)
    if upper[0] in ("L", "R"):
        return NWI_CODE_TO_CLASS[upper[0]]

    # Check 3-char palustrine codes (PEM, PFO, PSS, PAB, PUB, POW)
    if len(upper) >= 3:
        prefix3 = upper[:3]
        if prefix3 in NWI_CODE_TO_CLASS:
            return NWI_CODE_TO_CLASS[prefix3]

    return 5


def _compute_index(name: str, bands: Dict[str, np.ndarray]) -> np.ndarray:
    """Compute a spectral index from band arrays.

    Args:
        name: Index name ('ndvi' or 'ndwi').
        bands: Dict with keys 'red', 'green', 'nir' mapping to 2D arrays.

    Returns:
        2D array with index values in [-1, 1].

    Raises:
        ValueError: If index name is unknown.
    """
    if name == "ndvi":
        nir = bands["nir"].astype(np.float64)
        red = bands["red"].astype(np.float64)
        denom = nir + red
        with np.errstate(divide="ignore", invalid="ignore"):
            result = np.where(denom != 0, (nir - red) / denom, 0.0)
    elif name == "ndwi":
        green = bands["green"].astype(np.float64)
        nir = bands["nir"].astype(np.float64)
        denom = green + nir
        with np.errstate(divide="ignore", invalid="ignore"):
            result = np.where(denom != 0, (green - nir) / denom, 0.0)
    else:
        raise ValueError(
            f"Unknown spectral index: {name}. "
            f"Supported: {list(SPECTRAL_INDICES.keys())}"
        )

    return np.clip(result, -1.0, 1.0)


def _fill_depressions(
    dem: np.ndarray,
    nodata: Optional[float] = None,
) -> np.ndarray:
    """Fill surface depressions in a DEM using a priority-flood algorithm.

    Based on the priority-flood approach (Barnes et al. 2014). Border cells
    seed a priority queue; interior cells are raised to at least the lowest
    neighbor already processed.

    Args:
        dem: 2D numpy array of elevation values.
        nodata: Value to treat as no-data. NaN is always treated as no-data.

    Returns:
        2D array with depressions filled. No-data cells become NaN.

    Raises:
        ValueError: If dem is empty or not 2D.
    """
    if dem.size == 0:
        raise ValueError("DEM array is empty")
    if dem.ndim != 2:
        raise ValueError(f"DEM must be 2D, got {dem.ndim}D")

    filled = dem.astype(np.float64).copy()

    # Mark nodata cells
    nodata_mask = np.isnan(filled)
    if nodata is not None:
        nodata_mask |= filled == nodata
    filled[nodata_mask] = np.nan

    rows, cols = filled.shape
    visited = np.zeros((rows, cols), dtype=bool)
    visited[nodata_mask] = True

    # Priority queue: (elevation, row, col)
    pq: list = []

    # Seed with border cells
    for r in range(rows):
        for c in range(cols):
            if nodata_mask[r, c]:
                continue
            if r == 0 or r == rows - 1 or c == 0 or c == cols - 1:
                heapq.heappush(pq, (filled[r, c], r, c))
                visited[r, c] = True

    # 8-connected neighbors
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    while pq:
        elev, r, c = heapq.heappop(pq)
        for dr, dc in neighbors:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc]:
                visited[nr, nc] = True
                if np.isnan(filled[nr, nc]):
                    continue
                # Fill: raise neighbor to at least current cell elevation
                if filled[nr, nc] < elev:
                    filled[nr, nc] = elev
                heapq.heappush(pq, (filled[nr, nc], nr, nc))

    return filled


# ---------------------------------------------------------------------------
# Public API — Phase 1 Data Pipeline
# ---------------------------------------------------------------------------


def merge_dem_tiles(
    tile_paths: List[Union[str, Path]],
    output_path: Union[str, Path],
    bbox: Optional[Tuple[float, float, float, float]] = None,
    overwrite: bool = False,
) -> str:
    """Merge multiple DEM GeoTIFF tiles into a single file.

    Useful when DEM tiles are manually downloaded from USGS 3DEP
    (e.g. 1/3 arc-second NED tiles) and need to be combined before
    use in the pipeline.

    Args:
        tile_paths: List of paths to DEM GeoTIFF tiles.
        output_path: Path for the merged output GeoTIFF.
        bbox: Optional (min_lon, min_lat, max_lon, max_lat) to clip
              the merged result. If None, the full extent is kept.
        overwrite: Re-create if output exists.

    Returns:
        Path to the merged DEM file.

    Raises:
        FileNotFoundError: If any tile path does not exist.
        FileExistsError: If output exists and overwrite is False.
    """
    import rasterio
    from rasterio.merge import merge

    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output file exists: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Validate all tiles exist
    for tp in tile_paths:
        if not Path(tp).exists():
            raise FileNotFoundError(f"DEM tile not found: {tp}")

    logger.info("Merging %d DEM tiles ...", len(tile_paths))
    datasets = [rasterio.open(str(tp)) for tp in tile_paths]
    try:
        mosaic, transform = merge(datasets)
        profile = datasets[0].profile.copy()
        profile.update(
            driver="GTiff",
            height=mosaic.shape[1],
            width=mosaic.shape[2],
            transform=transform,
        )
    finally:
        for ds in datasets:
            ds.close()

    # Optionally clip to bbox
    if bbox is not None:
        _validate_bbox(bbox)
        from rasterio.windows import from_bounds

        min_lon, min_lat, max_lon, max_lat = bbox
        window = from_bounds(min_lon, min_lat, max_lon, max_lat, transform)
        row_off = max(0, int(window.row_off))
        col_off = max(0, int(window.col_off))
        height = min(int(window.height), mosaic.shape[1] - row_off)
        width = min(int(window.width), mosaic.shape[2] - col_off)

        mosaic = mosaic[:, row_off:row_off + height, col_off:col_off + width]

        from rasterio.transform import Affine
        new_transform = Affine(
            transform.a, transform.b, transform.c + col_off * transform.a,
            transform.d, transform.e, transform.f + row_off * transform.e,
        )
        profile.update(
            height=height,
            width=width,
            transform=new_transform,
        )

    with rasterio.open(str(output_path), "w", **profile) as dst:
        dst.write(mosaic)

    logger.info(
        "Merged DEM saved to %s (shape=%s)", output_path, mosaic.shape
    )
    return str(output_path)


def download_naip_timeseries(
    bbox: Tuple[float, float, float, float],
    output_dir: Union[str, Path],
    years: Optional[List[int]] = None,
    max_items_per_year: int = 10,
    overwrite: bool = False,
    **kwargs: Any,
) -> Dict[int, List[str]]:
    """Download NAIP imagery for multiple years.

    Wraps geoai's ``download_naip`` for each requested year and returns
    a dict mapping year to downloaded file paths. Years with no available
    imagery are omitted from the result.

    Args:
        bbox: (min_lon, min_lat, max_lon, max_lat) in WGS84.
        output_dir: Root directory for downloads; per-year subdirs are created.
        years: List of years to download. Defaults to [2015, 2017].
        max_items_per_year: Max STAC items per year.
        overwrite: Re-download existing files.

    Returns:
        Dict mapping year (int) to list of downloaded file paths.

    Raises:
        ValueError: If bbox is invalid.
    """
    _validate_bbox(bbox)

    if years is None:
        years = [2015, 2017]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    max_retries = kwargs.pop("max_retries", 3)

    results: Dict[int, List[str]] = {}
    for year in years:
        year_dir = str(output_dir / str(year))
        last_error: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(
                    "Downloading NAIP for year %d (attempt %d/%d) to %s",
                    year, attempt, max_retries, year_dir,
                )
                files = download_naip(
                    bbox=bbox,
                    output_dir=year_dir,
                    year=year,
                    max_items=max_items_per_year,
                    overwrite=overwrite,
                    **kwargs,
                )
                if files:
                    results[year] = files
                break
            except (TimeoutError, OSError, ConnectionError) as exc:
                last_error = exc
                logger.warning(
                    "NAIP download for year %d attempt %d/%d failed: %s",
                    year, attempt, max_retries, exc,
                )
                if attempt < max_retries:
                    import time
                    wait = min(30, 5 * attempt)
                    logger.info("Retrying in %ds ...", wait)
                    time.sleep(wait)
        else:
            raise TimeoutError(
                f"NAIP download for year {year} failed after "
                f"{max_retries} attempts"
            ) from last_error

    return results


def download_3dep_dem(
    bbox: Tuple[float, float, float, float],
    output_path: Union[str, Path],
    resolution: int = 10,
    overwrite: bool = False,
    **kwargs: Any,
) -> str:
    """Download a 3DEP DEM for the given bounding box.

    Args:
        bbox: (min_lon, min_lat, max_lon, max_lat) in WGS84.
        output_path: Path for the output GeoTIFF.
        resolution: DEM resolution in meters (1, 3, 10, or 30).
        overwrite: Re-download if file exists.

    Returns:
        Path to the downloaded DEM file.

    Raises:
        ValueError: If bbox is invalid or resolution is unsupported.
        FileExistsError: If output exists and overwrite is False.
    """
    _validate_bbox(bbox)

    if resolution not in SUPPORTED_3DEP_RESOLUTIONS:
        raise ValueError(
            f"Unsupported resolution: {resolution}m. "
            f"Supported resolutions: {sorted(SUPPORTED_3DEP_RESOLUTIONS)}"
        )

    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output file exists: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    import py3dep

    max_retries = kwargs.get("max_retries", 5)
    timeout = kwargs.get("timeout", 300)

    # Set timeout for the underlying async_retriever / pygeoogc calls
    os.environ.setdefault("HYRIVER_CACHE_DISABLE", "true")
    os.environ["HYRIVER_CACHE_EXPIRE"] = "0"

    # Resolutions to try: requested first, then coarser fallbacks
    _fallback_resolutions = sorted(
        [r for r in SUPPORTED_3DEP_RESOLUTIONS if r >= resolution]
    )
    if not _fallback_resolutions:
        _fallback_resolutions = [resolution]

    last_error: Optional[Exception] = None
    for res in _fallback_resolutions:
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(
                    "3DEP download attempt %d/%d (res=%dm, timeout=%ds) ...",
                    attempt, max_retries, res, timeout,
                )
                dem = py3dep.get_dem(bbox, resolution=res, crs="EPSG:4326")
                if res != resolution:
                    logger.warning(
                        "Used fallback resolution %dm instead of %dm",
                        res, resolution,
                    )
                break
            except (TimeoutError, OSError, ConnectionError) as exc:
                last_error = exc
                logger.warning(
                    "3DEP download attempt %d/%d (res=%dm) failed: %s",
                    attempt, max_retries, res, exc,
                )
                if attempt < max_retries:
                    import time
                    wait = min(60, 10 * attempt)
                    logger.info("Retrying in %ds ...", wait)
                    time.sleep(wait)
        else:
            # All retries exhausted for this resolution, try next
            logger.warning(
                "All %d attempts failed at %dm resolution, "
                "trying coarser resolution ...",
                max_retries, res,
            )
            continue
        break  # Success — exit the resolution loop
    else:
        raise TimeoutError(
            f"3DEP DEM download failed after {max_retries} attempts "
            f"across resolutions {_fallback_resolutions}. "
            f"The 3DEP service may be down — try again later or increase "
            f"max_retries via overrides: "
            f"{{'dem_max_retries': 10, 'dem_timeout': 600}}"
        ) from last_error

    dem.rio.to_raster(str(output_path))
    logger.info("Saved 3DEP DEM (%dm) to %s", resolution, output_path)

    return str(output_path)


def download_nwi(
    bbox: Tuple[float, float, float, float],
    output_path: Union[str, Path],
    overwrite: bool = False,
    **kwargs: Any,
) -> str:
    """Download National Wetlands Inventory polygons for a bounding box.

    Uses the USFWS NWI Web Map Services / ArcGIS REST endpoint to fetch
    wetland polygons intersecting the bbox.

    Args:
        bbox: (min_lon, min_lat, max_lon, max_lat) in WGS84.
        output_path: Path for the output GeoPackage (.gpkg).
        overwrite: Re-download if file exists.

    Returns:
        Path to the saved NWI file.

    Raises:
        ValueError: If bbox is invalid.
        FileExistsError: If output exists and overwrite is False.
    """
    _validate_bbox(bbox)

    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output file exists: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    import geopandas as gpd
    import requests

    min_lon, min_lat, max_lon, max_lat = bbox
    url = (
        "https://fwsprimary.wim.usgs.gov/server/rest/services/"
        "Wetlands/MapServer/0/query"
    )
    params = {
        "geometry": f"{min_lon},{min_lat},{max_lon},{max_lat}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "outSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*",
        "f": "geojson",
        "resultRecordCount": 10000,
    }

    max_retries = kwargs.get("max_retries", 3)
    timeout = kwargs.get("timeout", 120)

    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(
                "Querying NWI for bbox %s (attempt %d/%d) ...",
                bbox, attempt, max_retries,
            )
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            break
        except (TimeoutError, OSError, ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ConnectionError) as exc:
            last_error = exc
            logger.warning(
                "NWI download attempt %d/%d failed: %s",
                attempt, max_retries, exc,
            )
            if attempt < max_retries:
                import time
                wait = min(30, 5 * attempt)
                logger.info("Retrying in %ds ...", wait)
                time.sleep(wait)
    else:
        raise TimeoutError(
            f"NWI download failed after {max_retries} attempts"
        ) from last_error

    gdf = gpd.GeoDataFrame.from_features(resp.json()["features"], crs="EPSG:4326")
    gdf.to_file(str(output_path), driver="GPKG")
    logger.info("Saved %d NWI polygons to %s", len(gdf), output_path)

    return str(output_path)


def compute_spectral_indices(
    naip_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    indices: Optional[List[str]] = None,
    overwrite: bool = False,
    **kwargs: Any,
) -> str:
    """Compute spectral indices from a NAIP GeoTIFF.

    Reads a 4-band NAIP image (R, G, B, NIR) and computes the requested
    spectral indices, writing them as bands in a single output GeoTIFF.

    Args:
        naip_path: Path to 4-band NAIP GeoTIFF.
        output_path: Path for the output indices GeoTIFF. Defaults to
            ``<naip_path>_indices.tif``.
        indices: List of index names to compute. Defaults to all supported.
        overwrite: Overwrite existing output file.

    Returns:
        Path to the output GeoTIFF.

    Raises:
        FileNotFoundError: If naip_path does not exist.
        FileExistsError: If output exists and overwrite is False.
    """
    import rasterio

    naip_path = Path(naip_path)
    if not naip_path.exists():
        raise FileNotFoundError(f"NAIP file not found: {naip_path}")

    index_names = _validate_index_names(indices)

    if output_path is None:
        output_path = naip_path.with_name(naip_path.stem + "_indices.tif")
    output_path = Path(output_path)

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output file exists: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(naip_path) as src:
        red = src.read(1).astype(np.float64)
        green = src.read(2).astype(np.float64)
        nir = src.read(4).astype(np.float64)

        bands_dict = {"red": red, "green": green, "nir": nir}

        profile = src.profile.copy()
        profile.update(
            dtype="float32",
            count=len(index_names),
            nodata=None,
        )

        with rasterio.open(output_path, "w", **profile) as dst:
            for i, name in enumerate(index_names, start=1):
                result = _compute_index(name, bands_dict)
                dst.write(result.astype(np.float32), i)

    logger.info("Computed %s -> %s", index_names, output_path)
    return str(output_path)


def extract_surface_depressions(
    dem_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    min_depth: float = 0.1,
    min_area: float = 0.0,
    overwrite: bool = False,
    **kwargs: Any,
) -> str:
    """Extract surface depressions from a DEM.

    Uses a priority-flood fill algorithm to identify topographic depressions.
    Depression depth is computed as (filled_DEM - original_DEM).

    Args:
        dem_path: Path to the input DEM GeoTIFF.
        output_path: Path for output depression depth GeoTIFF. Defaults to
            ``<dem_path>_depressions.tif``.
        min_depth: Minimum depression depth in DEM units (default 0.1m).
        min_area: Minimum depression area in square map units (default 0.0).
        overwrite: Overwrite existing output.

    Returns:
        Path to the depression depth GeoTIFF.

    Raises:
        FileNotFoundError: If dem_path does not exist.
        ValueError: If min_depth or min_area is negative.
        FileExistsError: If output exists and overwrite is False.
    """
    import rasterio

    dem_path = Path(dem_path)
    if not dem_path.exists():
        raise FileNotFoundError(f"DEM file not found: {dem_path}")

    if min_depth < 0:
        raise ValueError(f"min_depth must be >= 0, got {min_depth}")
    if min_area < 0:
        raise ValueError(f"min_area must be >= 0, got {min_area}")

    if output_path is None:
        output_path = dem_path.with_name(dem_path.stem + "_depressions.tif")
    output_path = Path(output_path)

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output file exists: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float64)
        nodata = src.nodata

        filled = _fill_depressions(dem, nodata=nodata)
        depth = filled - dem

        # Apply nodata mask
        nodata_mask = np.isnan(filled)
        depth[nodata_mask] = -9999.0

        # Apply minimum depth threshold
        valid = ~nodata_mask
        depth[valid & (depth < min_depth)] = 0.0

        profile = src.profile.copy()
        profile.update(
            dtype="float32",
            count=1,
            nodata=-9999.0,
        )

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(depth.astype(np.float32), 1)

    logger.info("Extracted depressions (min_depth=%.2f) -> %s", min_depth, output_path)
    return str(output_path)


def create_wetland_composite(
    naip_paths: Union[str, Path, List[Union[str, Path]]],
    dem_path: Union[str, Path],
    output_path: Union[str, Path],
    indices: Optional[List[str]] = None,
    include_depressions: bool = True,
    overwrite: bool = False,
    **kwargs: Any,
) -> str:
    """Create a multi-band wetland analysis composite.

    Stacks NAIP bands, spectral indices, DEM elevation, and optionally
    depression depth into a single multi-band GeoTIFF.

    Args:
        naip_paths: Path(s) to NAIP GeoTIFF(s). Can be a single path or list.
        dem_path: Path to DEM GeoTIFF.
        output_path: Path for the output composite GeoTIFF.
        indices: Spectral indices to compute per NAIP image.
        include_depressions: Include depression depth band from DEM.
        overwrite: Overwrite existing output.

    Returns:
        Path to the composite GeoTIFF.

    Raises:
        ValueError: If naip_paths is empty.
        FileNotFoundError: If any input file does not exist.
        FileExistsError: If output exists and overwrite is False.
    """
    import rasterio
    from rasterio.warp import Resampling, reproject

    # Normalize naip_paths to list
    if isinstance(naip_paths, (str, Path)):
        naip_paths = [naip_paths]
    naip_paths = [Path(p) for p in naip_paths]

    if not naip_paths:
        raise ValueError("naip_paths must not be empty")

    dem_path = Path(dem_path)
    if not dem_path.exists():
        raise FileNotFoundError(f"DEM file not found: {dem_path}")

    for p in naip_paths:
        if not p.exists():
            raise FileNotFoundError(f"NAIP file not found: {p}")

    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output file exists: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    index_names = _validate_index_names(indices)

    # Collect all bands
    all_bands: List[np.ndarray] = []

    # Use the first NAIP file as the spatial reference
    with rasterio.open(naip_paths[0]) as ref_src:
        ref_profile = ref_src.profile.copy()
        ref_transform = ref_src.transform
        ref_crs = ref_src.crs
        ref_height = ref_src.height
        ref_width = ref_src.width

    for naip_path in naip_paths:
        with rasterio.open(naip_path) as src:
            # Read NAIP bands (R, G, B, NIR)
            for b in range(1, src.count + 1):
                all_bands.append(src.read(b).astype(np.float32))

            # Compute spectral indices
            red = src.read(1).astype(np.float64)
            green = src.read(2).astype(np.float64)
            nir = src.read(4).astype(np.float64)
            bands_dict = {"red": red, "green": green, "nir": nir}

            for idx_name in index_names:
                idx_arr = _compute_index(idx_name, bands_dict)
                all_bands.append(idx_arr.astype(np.float32))

    # Add DEM elevation band (reproject to match NAIP if needed)
    with rasterio.open(dem_path) as dem_src:
        dem_data = np.empty((ref_height, ref_width), dtype=np.float32)
        reproject(
            source=rasterio.band(dem_src, 1),
            destination=dem_data,
            src_transform=dem_src.transform,
            src_crs=dem_src.crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=Resampling.bilinear,
        )
        all_bands.append(dem_data)

        # Add depression depth band
        if include_depressions:
            dem_arr = dem_src.read(1).astype(np.float64)
            filled = _fill_depressions(dem_arr, nodata=dem_src.nodata)
            dep_depth = (filled - dem_arr).astype(np.float32)
            dep_depth[np.isnan(filled)] = 0.0

            # Reproject depression depth to match NAIP grid
            dep_reproj = np.empty((ref_height, ref_width), dtype=np.float32)
            reproject(
                source=dep_depth,
                destination=dep_reproj,
                src_transform=dem_src.transform,
                src_crs=dem_src.crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=Resampling.bilinear,
            )
            all_bands.append(dep_reproj)

    # Write composite
    ref_profile.update(
        dtype="float32",
        count=len(all_bands),
        nodata=None,
    )

    with rasterio.open(output_path, "w", **ref_profile) as dst:
        for i, band in enumerate(all_bands, start=1):
            dst.write(band, i)

    logger.info(
        "Created %d-band composite -> %s",
        len(all_bands),
        output_path,
    )
    return str(output_path)


# ---------------------------------------------------------------------------
# Public API — Phase 2 Weak Label Generation
# ---------------------------------------------------------------------------


def reclassify_nwi(
    nwi_path: Union[str, Path],
    raster_template: Union[str, Path],
    output_path: Union[str, Path],
    attribute_field: str = "ATTRIBUTE",
    overwrite: bool = False,
    **kwargs: Any,
) -> str:
    """Rasterize NWI polygons into a classified raster matching a template grid.

    Reads NWI vector polygons, parses Cowardin codes from the attribute field,
    and burns class IDs into a raster aligned to the template.

    Args:
        nwi_path: Path to NWI vector file (GeoPackage, Shapefile, etc.).
        raster_template: Path to a reference raster that defines the output
            grid (CRS, resolution, extent).
        output_path: Path for the output classified raster.
        attribute_field: Column name containing Cowardin codes.
        overwrite: Overwrite existing output file.

    Returns:
        Path to the output reclassified raster.

    Raises:
        FileNotFoundError: If nwi_path or raster_template does not exist.
        FileExistsError: If output exists and overwrite is False.
    """
    import rasterio
    from rasterio.features import rasterize

    nwi_path = Path(nwi_path)
    raster_template = Path(raster_template)
    output_path = Path(output_path)

    if not nwi_path.exists():
        raise FileNotFoundError(f"NWI file not found: {nwi_path}")
    if not raster_template.exists():
        raise FileNotFoundError(f"Raster template not found: {raster_template}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output file exists: {output_path}")

    import geopandas as gpd

    output_path.parent.mkdir(parents=True, exist_ok=True)

    gdf = gpd.read_file(str(nwi_path))

    with rasterio.open(raster_template) as tmpl:
        tmpl_crs = tmpl.crs
        tmpl_transform = tmpl.transform
        tmpl_height = tmpl.height
        tmpl_width = tmpl.width

    # Reproject NWI to template CRS if needed
    if gdf.crs and not gdf.crs.equals(tmpl_crs):
        gdf = gdf.to_crs(tmpl_crs)

    # Parse Cowardin codes and assign class IDs
    shapes = []
    for _, row in gdf.iterrows():
        code = row.get(attribute_field, None)
        class_id = _parse_nwi_code(code)
        if class_id > 0 and row.geometry is not None:
            shapes.append((row.geometry, class_id))

    # Rasterize
    if shapes:
        out_arr = rasterize(
            shapes,
            out_shape=(tmpl_height, tmpl_width),
            transform=tmpl_transform,
            fill=0,
            dtype=np.uint8,
            all_touched=True,
        )
    else:
        out_arr = np.zeros((tmpl_height, tmpl_width), dtype=np.uint8)

    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "count": 1,
        "height": tmpl_height,
        "width": tmpl_width,
        "crs": tmpl_crs,
        "transform": tmpl_transform,
    }
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(out_arr, 1)

    logger.info(
        "Reclassified NWI (%d polygons, %d classes) -> %s",
        len(shapes),
        len(set(v for _, v in shapes)) if shapes else 0,
        output_path,
    )
    return str(output_path)


def generate_weak_labels(
    nwi_raster_path: Union[str, Path],
    depression_path: Union[str, Path],
    ndvi_paths: List[Union[str, Path]],
    ndwi_paths: List[Union[str, Path]],
    output_path: Union[str, Path],
    depression_threshold: float = 0.0,
    stability_threshold: float = 0.05,
    min_component_fraction: float = 0.5,
    overwrite: bool = False,
    **kwargs: Any,
) -> str:
    """Generate filtered weak labels from NWI + depression + temporal stability.

    Implements the four-step label filtering strategy:
      1. Start with reclassified NWI raster labels.
      2. Depression filter: remove wetland labels outside topographic depressions.
      3. Temporal stability filter: remove labels where NDVI/NDWI change
         exceeds the threshold across epochs.
      4. Object-level confidence: for connected components of same-class
         pixels, retain only those where >= min_component_fraction of pixels
         survived steps 2-3.

    Args:
        nwi_raster_path: Path to reclassified NWI raster (from reclassify_nwi).
        depression_path: Path to depression depth raster.
        ndvi_paths: List of NDVI raster paths for each epoch.
        ndwi_paths: List of NDWI raster paths for each epoch (same order).
        output_path: Path for the output filtered label raster.
        depression_threshold: Minimum depression depth to retain a wetland
            label (default 0.0 = any depression).
        stability_threshold: Maximum abs change in NDVI/NDWI between
            consecutive epochs to consider a pixel "stable" (default 0.05).
        min_component_fraction: Minimum fraction of pixels in a connected
            component that must be reliable to retain the component (default 0.5).
        overwrite: Overwrite existing output file.

    Returns:
        Path to the filtered weak label raster.

    Raises:
        FileNotFoundError: If any input file does not exist.
        ValueError: If ndvi_paths and ndwi_paths have different lengths,
            or if thresholds are negative.
    """
    import rasterio
    from scipy import ndimage

    nwi_raster_path = Path(nwi_raster_path)
    depression_path = Path(depression_path)
    output_path = Path(output_path)

    if not nwi_raster_path.exists():
        raise FileNotFoundError(f"NWI raster not found: {nwi_raster_path}")
    if not depression_path.exists():
        raise FileNotFoundError(f"Depression raster not found: {depression_path}")

    if len(ndvi_paths) != len(ndwi_paths):
        raise ValueError(
            f"ndvi_paths ({len(ndvi_paths)}) and ndwi_paths ({len(ndwi_paths)}) "
            f"must have the same length"
        )
    if stability_threshold < 0:
        raise ValueError(f"stability_threshold must be >= 0, got {stability_threshold}")
    if depression_threshold < 0:
        raise ValueError(f"depression_threshold must be >= 0, got {depression_threshold}")
    if not (0 <= min_component_fraction <= 1):
        raise ValueError(
            f"min_component_fraction must be in [0, 1], got {min_component_fraction}"
        )

    for p in ndvi_paths:
        if not Path(p).exists():
            raise FileNotFoundError(f"NDVI file not found: {p}")
    for p in ndwi_paths:
        if not Path(p).exists():
            raise FileNotFoundError(f"NDWI file not found: {p}")

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output file exists: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Read inputs
    with rasterio.open(nwi_raster_path) as src:
        labels = src.read(1).astype(np.uint8)
        profile = src.profile.copy()

    with rasterio.open(depression_path) as src:
        dep_data = src.read(1).astype(np.float64)
        dep_nodata = src.nodata

    # Step 1: Start with NWI labels
    reliable = labels > 0  # mask of pixels with any label

    # Step 2: Depression filter — keep only wetland pixels in depressions
    dep_valid = dep_data > depression_threshold
    if dep_nodata is not None:
        dep_valid &= dep_data != dep_nodata
    reliable &= dep_valid

    # Step 3: Temporal stability filter
    if len(ndvi_paths) >= 2:
        for i in range(len(ndvi_paths) - 1):
            with rasterio.open(ndvi_paths[i]) as s1, rasterio.open(ndvi_paths[i + 1]) as s2:
                ndvi1 = s1.read(1).astype(np.float64)
                ndvi2 = s2.read(1).astype(np.float64)
            ndvi_change = np.abs(ndvi2 - ndvi1)
            reliable &= ndvi_change <= stability_threshold

        for i in range(len(ndwi_paths) - 1):
            with rasterio.open(ndwi_paths[i]) as s1, rasterio.open(ndwi_paths[i + 1]) as s2:
                ndwi1 = s1.read(1).astype(np.float64)
                ndwi2 = s2.read(1).astype(np.float64)
            ndwi_change = np.abs(ndwi2 - ndwi1)
            reliable &= ndwi_change <= stability_threshold

    # Step 4: Object-level confidence filtering
    # Label connected components per class
    output_labels = np.zeros_like(labels)
    unique_classes = [c for c in np.unique(labels) if c > 0]

    for cls in unique_classes:
        cls_mask = labels == cls
        # Find connected components for this class
        struct = ndimage.generate_binary_structure(2, 1)  # 4-connected
        comp_labels, num_comps = ndimage.label(cls_mask, structure=struct)

        for comp_id in range(1, num_comps + 1):
            comp_mask = comp_labels == comp_id
            total_pixels = np.sum(comp_mask)
            reliable_pixels = np.sum(comp_mask & reliable)

            if total_pixels > 0 and (reliable_pixels / total_pixels) >= min_component_fraction:
                output_labels[comp_mask] = cls

    # Write output
    profile.update(dtype="uint8", count=1, nodata=0)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(output_labels.astype(np.uint8), 1)

    retained = np.sum(output_labels > 0)
    original = np.sum(labels > 0)
    logger.info(
        "Generated weak labels: %d/%d pixels retained (%.1f%%) -> %s",
        retained,
        original,
        100 * retained / max(original, 1),
        output_path,
    )
    return str(output_path)


def export_training_tiles(
    composite_path: Union[str, Path],
    label_path: Union[str, Path],
    output_dir: Union[str, Path],
    tile_size: int = 256,
    stride: Optional[int] = None,
    min_valid_fraction: float = 0.5,
    overwrite: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Export paired image/label tiles for CNN training.

    Extracts fixed-size tiles from a composite raster and corresponding
    label raster using a sliding window approach. Tiles with insufficient
    labeled pixels are filtered out.

    Args:
        composite_path: Path to the multi-band composite GeoTIFF.
        label_path: Path to the single-band label GeoTIFF.
        output_dir: Root output directory. Creates ``images/`` and ``labels/``
            subdirectories.
        tile_size: Tile height and width in pixels (default 256).
        stride: Step size between tiles. Defaults to tile_size (no overlap).
        min_valid_fraction: Minimum fraction of non-zero label pixels
            required to keep a tile (default 0.5). Set to 0.0 to keep all.
        overwrite: If True, overwrite existing output directory.

    Returns:
        Dict with keys ``num_tiles``, ``output_dir``, ``tile_size``.

    Raises:
        FileNotFoundError: If composite_path or label_path does not exist.
        ValueError: If tile_size <= 0 or min_valid_fraction not in [0, 1].
        FileExistsError: If output_dir exists and overwrite is False.
    """
    import rasterio
    from rasterio.windows import Window

    composite_path = Path(composite_path)
    label_path = Path(label_path)
    output_dir = Path(output_dir)

    if not composite_path.exists():
        raise FileNotFoundError(f"Composite file not found: {composite_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"Label file not found: {label_path}")
    if tile_size <= 0:
        raise ValueError(f"tile_size must be > 0, got {tile_size}")
    if not (0 <= min_valid_fraction <= 1):
        raise ValueError(
            f"min_valid_fraction must be in [0, 1], got {min_valid_fraction}"
        )

    if stride is None:
        stride = tile_size

    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"Output directory exists: {output_dir}")

    img_dir = output_dir / "images"
    lbl_dir = output_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    tile_count = 0

    with rasterio.open(composite_path) as comp_src, rasterio.open(label_path) as lbl_src:
        height = comp_src.height
        width = comp_src.width

        for row_off in range(0, height - tile_size + 1, stride):
            for col_off in range(0, width - tile_size + 1, stride):
                window = Window(col_off, row_off, tile_size, tile_size)

                lbl_tile = lbl_src.read(1, window=window)

                # Filter by valid fraction
                valid_frac = np.count_nonzero(lbl_tile) / lbl_tile.size
                if valid_frac < min_valid_fraction:
                    continue

                comp_tile = comp_src.read(window=window)

                # Compute tile transform
                tile_transform = rasterio.windows.transform(window, comp_src.transform)

                # Write image tile
                img_profile = comp_src.profile.copy()
                img_profile.update(
                    height=tile_size,
                    width=tile_size,
                    transform=tile_transform,
                )
                tile_name = f"tile_{row_off:06d}_{col_off:06d}.tif"
                with rasterio.open(img_dir / tile_name, "w", **img_profile) as dst:
                    dst.write(comp_tile)

                # Write label tile
                lbl_profile = lbl_src.profile.copy()
                lbl_profile.update(
                    height=tile_size,
                    width=tile_size,
                    transform=tile_transform,
                )
                with rasterio.open(lbl_dir / tile_name, "w", **lbl_profile) as dst:
                    dst.write(lbl_tile, 1)

                tile_count += 1

    logger.info(
        "Exported %d tiles (%dx%d, stride=%d) -> %s",
        tile_count,
        tile_size,
        tile_size,
        stride,
        output_dir,
    )
    return {
        "num_tiles": tile_count,
        "output_dir": str(output_dir),
        "tile_size": tile_size,
    }


# ---------------------------------------------------------------------------
# Public API — Phase 3 Model Training
# ---------------------------------------------------------------------------


def train_wetland_model(
    tiles_dir: Union[str, Path],
    output_dir: Union[str, Path],
    architecture: str = "unetplusplus",
    encoder_name: str = "resnet50",
    num_classes: int = 6,
    in_channels: int = 14,
    num_epochs: int = 50,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    loss_function: str = "focal",
    use_class_weights: bool = True,
    val_split: float = 0.2,
    seed: int = 42,
    encoder_weights: Optional[str] = "imagenet",
    ignore_index: int = 0,
    focal_alpha: float = 1.0,
    focal_gamma: float = 2.0,
    max_class_weight: float = 50.0,
    device: Optional[str] = None,
    overwrite: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Train a wetland segmentation model using multi-temporal NAIP + LiDAR tiles.

    Wraps ``geoai.landcover_train.train_segmentation_landcover`` with
    wetland-specific defaults: U-Net++ architecture, ResNet-50 encoder,
    14-band input (2 NAIP epochs x 6 bands + DEM + depression depth),
    6 Cowardin classes, focal loss with class weights.

    Args:
        tiles_dir: Root directory containing ``images/`` and ``labels/``
            subdirectories (output of ``export_training_tiles``).
        output_dir: Directory for model checkpoints and training logs.
        architecture: Segmentation architecture name. Must be one of
            ``SUPPORTED_ARCHITECTURES``. Default ``"unetplusplus"``.
        encoder_name: Backbone encoder name (timm/SMP compatible).
            Default ``"resnet50"``.
        num_classes: Number of output classes including background.
            Default 6 (Cowardin schema).
        in_channels: Number of input bands. Default 14.
        num_epochs: Maximum training epochs. Default 50.
        batch_size: Training batch size. Default 8.
        learning_rate: Initial learning rate. Default 1e-3.
        weight_decay: Weight decay for optimizer. Default 1e-4.
        loss_function: Loss function name: ``"focal"`` or
            ``"crossentropy"``. Default ``"focal"``.
        use_class_weights: Compute and apply inverse-frequency class
            weights. Default True.
        val_split: Fraction of data for validation. Default 0.2.
        seed: Random seed for reproducibility. Default 42.
        encoder_weights: Pretrained encoder weights. Default ``"imagenet"``.
        ignore_index: Class index to ignore in loss. Default 0 (Upland).
        focal_alpha: Focal loss alpha parameter. Default 1.0.
        focal_gamma: Focal loss gamma parameter. Default 2.0.
        max_class_weight: Maximum class weight cap. Default 50.0.
        device: Device string (e.g. ``"cuda"``, ``"cpu"``). Auto-detected
            if None.
        overwrite: Allow overwriting existing output_dir. Default False.

    Returns:
        Dict with keys:
            - ``model_path``: Path to the best saved model weights.
            - ``architecture``: Architecture used.
            - ``encoder_name``: Encoder used.
            - ``num_classes``: Number of classes.
            - ``in_channels``: Number of input channels.
            - ``output_dir``: Path to the output directory.

    Raises:
        FileNotFoundError: If tiles_dir or its subdirectories don't exist.
        FileExistsError: If output_dir exists and overwrite is False.
        ValueError: If architecture, num_classes, in_channels, or
            val_split are invalid.
    """
    import torch
    from geoai.landcover_train import train_segmentation_landcover

    tiles_dir = Path(tiles_dir)
    output_dir = Path(output_dir)

    # --- Validation ---
    if not tiles_dir.exists():
        raise FileNotFoundError(f"Tiles directory not found: {tiles_dir}")

    images_dir = tiles_dir / "images"
    labels_dir = tiles_dir / "labels"

    if not images_dir.exists():
        raise FileNotFoundError(
            f"Images subdirectory not found: {images_dir}"
        )
    if not labels_dir.exists():
        raise FileNotFoundError(
            f"Labels subdirectory not found: {labels_dir}"
        )

    if num_classes <= 0:
        raise ValueError(f"num_classes must be > 0, got {num_classes}")
    if in_channels <= 0:
        raise ValueError(f"in_channels must be > 0, got {in_channels}")
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(
            f"Unsupported architecture: {architecture!r}. "
            f"Choose from: {sorted(SUPPORTED_ARCHITECTURES)}"
        )
    _supported_losses = frozenset({"focal", "crossentropy"})
    if loss_function not in _supported_losses:
        raise ValueError(
            f"Unsupported loss_function: {loss_function!r}. "
            f"Choose from: {sorted(_supported_losses)}"
        )
    if not (0 < val_split < 1):
        raise ValueError(f"val_split must be in (0, 1), got {val_split}")
    if num_epochs <= 0:
        raise ValueError(f"num_epochs must be > 0, got {num_epochs}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be > 0, got {learning_rate}")

    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"Output directory exists: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Resolve device ---
    if device is None:
        resolved_device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        resolved_device = torch.device(device)

    logger.info(
        "Training wetland model: arch=%s, encoder=%s, classes=%d, "
        "channels=%d, epochs=%d, device=%s",
        architecture,
        encoder_name,
        num_classes,
        in_channels,
        num_epochs,
        resolved_device,
    )

    # --- Train via geoai ---
    model = train_segmentation_landcover(
        images_dir=str(images_dir),
        labels_dir=str(labels_dir),
        output_dir=str(output_dir),
        architecture=architecture,
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        num_channels=in_channels,
        num_classes=num_classes,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=seed,
        val_split=val_split,
        loss_function=loss_function,
        use_class_weights=use_class_weights,
        ignore_index=ignore_index,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
        max_class_weight=max_class_weight,
        device=resolved_device,
        verbose=True,
        save_best_only=True,
        **kwargs,
    )

    # Locate best model file
    best_model_path = output_dir / "best_model.pth"
    if not best_model_path.exists():
        best_model_path = output_dir / "final_model.pth"
    if not best_model_path.exists():
        raise RuntimeError(
            f"Training completed but no model checkpoint found in {output_dir}. "
            "Expected 'best_model.pth' or 'final_model.pth'."
        )

    logger.info("Training complete. Model saved to %s", best_model_path)

    return {
        "model_path": str(best_model_path),
        "architecture": architecture,
        "encoder_name": encoder_name,
        "num_classes": num_classes,
        "in_channels": in_channels,
        "output_dir": str(output_dir),
    }


# ---------------------------------------------------------------------------
# Public API — Phase 4 Inference & Dynamics
# ---------------------------------------------------------------------------


def predict_wetlands(
    model_path: Union[str, Path],
    composite_path: Union[str, Path],
    output_path: Union[str, Path],
    architecture: str = "unetplusplus",
    encoder_name: str = "resnet50",
    num_classes: int = 6,
    in_channels: int = 14,
    tile_size: int = 256,
    overlap: int = 128,
    batch_size: int = 4,
    encoder_weights: Optional[str] = None,
    device: Optional[str] = None,
    overwrite: bool = False,
    **kwargs: Any,
) -> str:
    """Run wetland segmentation inference on a composite raster.

    Loads a trained model checkpoint, tiles the input composite using a
    sliding window with overlap, runs inference, and stitches predictions
    into a single classified raster using argmax blending.

    Args:
        model_path: Path to saved model weights (``.pth`` file).
        composite_path: Path to multi-band composite GeoTIFF (from
            ``create_wetland_composite``).
        output_path: Path for the output classified GeoTIFF.
        architecture: Segmentation architecture name. Must match the
            architecture used during training. Default ``"unetplusplus"``.
        encoder_name: Encoder backbone name. Default ``"resnet50"``.
        num_classes: Number of output classes. Default 6.
        in_channels: Number of input bands. Default 14.
        tile_size: Inference tile size in pixels. Default 256.
        overlap: Overlap between adjacent tiles in pixels. Default 128.
        batch_size: Number of tiles per inference batch. Default 4.
        encoder_weights: Encoder pretrained weights (use None for inference).
        device: Device string. Auto-detected if None.
        overwrite: Overwrite existing output file.

    Returns:
        Path to the output classified GeoTIFF.

    Raises:
        FileNotFoundError: If model_path or composite_path does not exist.
        FileExistsError: If output_path exists and overwrite is False.
        ValueError: If tile_size <= 0 or overlap >= tile_size.
    """
    import rasterio
    import segmentation_models_pytorch as smp
    import torch

    model_path = Path(model_path)
    composite_path = Path(composite_path)
    output_path = Path(output_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not composite_path.exists():
        raise FileNotFoundError(f"Composite file not found: {composite_path}")
    if tile_size <= 0:
        raise ValueError(f"tile_size must be > 0, got {tile_size}")
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0, got {overlap}")
    if overlap >= tile_size:
        raise ValueError(
            f"overlap must be < tile_size, got overlap={overlap}, tile_size={tile_size}"
        )
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(
            f"Unsupported architecture: {architecture!r}. "
            f"Choose from: {sorted(SUPPORTED_ARCHITECTURES)}"
        )
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output file exists: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resolve device
    if device is None:
        resolved_device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        resolved_device = torch.device(device)

    # Build model and load weights
    model = smp.create_model(
        architecture,
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
    )
    state_dict = torch.load(model_path, map_location=resolved_device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(resolved_device)
    model.eval()

    # Run sliding-window inference
    with rasterio.open(composite_path) as src:
        height = src.height
        width = src.width
        profile = src.profile.copy()

        # Accumulate softmax probabilities for blending
        prob_acc = np.zeros((num_classes, height, width), dtype=np.float64)
        count_acc = np.zeros((height, width), dtype=np.float64)

        stride = tile_size - overlap

        # Build row/column offsets ensuring full coverage
        def _offsets(total: int, tile: int, step: int) -> List[int]:
            offsets = list(range(0, max(total - tile, 0) + 1, step))
            last = max(0, total - tile)
            if not offsets or offsets[-1] != last:
                offsets.append(last)
            return offsets

        row_offsets = _offsets(height, tile_size, stride)
        col_offsets = _offsets(width, tile_size, stride)
        windows = [(r, c) for r in row_offsets for c in col_offsets]

        # Process in batches
        for batch_start in range(0, len(windows), batch_size):
            batch_windows = windows[batch_start: batch_start + batch_size]
            tiles = []

            for r, c in batch_windows:
                from rasterio.windows import Window

                win = Window(c, r, tile_size, tile_size)
                tile = src.read(window=win).astype(np.float32)

                # Handle edge tiles smaller than tile_size
                actual_h, actual_w = tile.shape[1], tile.shape[2]
                if actual_h < tile_size or actual_w < tile_size:
                    padded = np.zeros(
                        (in_channels, tile_size, tile_size), dtype=np.float32
                    )
                    padded[:, :actual_h, :actual_w] = tile
                    tile = padded

                # Clip or pad channels
                if tile.shape[0] > in_channels:
                    tile = tile[:in_channels]
                elif tile.shape[0] < in_channels:
                    pad = np.zeros(
                        (in_channels - tile.shape[0], tile_size, tile_size),
                        dtype=np.float32,
                    )
                    tile = np.concatenate([tile, pad], axis=0)

                tiles.append(tile)

            batch_tensor = torch.from_numpy(np.stack(tiles)).to(resolved_device)

            with torch.no_grad():
                logits = model(batch_tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()

            # Accumulate predictions
            for idx, (r, c) in enumerate(batch_windows):
                h_end = min(r + tile_size, height)
                w_end = min(c + tile_size, width)
                th = h_end - r
                tw = w_end - c

                prob_acc[:, r:h_end, c:w_end] += probs[idx, :, :th, :tw]
                count_acc[r:h_end, c:w_end] += 1.0

    # Avoid division by zero
    count_acc = np.maximum(count_acc, 1.0)
    avg_probs = prob_acc / count_acc[np.newaxis, :, :]

    # Argmax to get class predictions
    prediction = np.argmax(avg_probs, axis=0).astype(np.uint8)

    # Write output
    profile.update(dtype="uint8", count=1, nodata=None)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(prediction, 1)

    logger.info("Inference complete -> %s", output_path)
    return str(output_path)


def map_wetland_dynamics(
    prediction_paths: List[Union[str, Path]],
    output_path: Union[str, Path],
    overwrite: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Map wetland dynamics (gain/loss/stable) across multiple time periods.

    Compares classified wetland predictions from two or more epochs and
    produces a change map and pixel-level statistics.

    Change codes in the output raster (for the last pair of epochs):
        - 0: Stable non-wetland (upland in both epochs)
        - 1: Stable wetland (wetland in both epochs)
        - 2: Wetland gain (non-wetland -> wetland)
        - 3: Wetland loss (wetland -> non-wetland)

    Args:
        prediction_paths: List of classified raster paths, one per epoch,
            in chronological order. Must have at least 2.
        output_path: Path for the output dynamics raster.
        overwrite: Overwrite existing output file.

    Returns:
        Dict with keys:
            - ``output_path``: Path to the dynamics raster.
            - ``statistics``: Dict with ``wetland_gain_pixels``,
              ``wetland_loss_pixels``, ``stable_wetland_pixels``,
              ``stable_nonwetland_pixels``.

    Raises:
        ValueError: If fewer than 2 prediction paths are provided.
        FileNotFoundError: If any prediction path does not exist.
        FileExistsError: If output_path exists and overwrite is False.
    """
    import rasterio

    if len(prediction_paths) < 2:
        raise ValueError(
            f"Need at least 2 prediction paths, got {len(prediction_paths)}"
        )

    prediction_paths = [Path(p) for p in prediction_paths]
    output_path = Path(output_path)

    for p in prediction_paths:
        if not p.exists():
            raise FileNotFoundError(f"Prediction file not found: {p}")

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output file exists: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Compare last two epochs for the dynamics map
    early_path = prediction_paths[-2]
    late_path = prediction_paths[-1]

    with rasterio.open(early_path) as src:
        early = src.read(1)
        profile = src.profile.copy()

    with rasterio.open(late_path) as src:
        late = src.read(1)

    if early.shape != late.shape:
        raise ValueError(
            f"Shape mismatch between epochs: {early.shape} vs {late.shape}. "
            "Predictions must be co-registered."
        )

    # Binary wetland masks (any class > 0 is wetland)
    early_wet = early > 0
    late_wet = late > 0

    # Compute dynamics
    # 0 = stable non-wetland, 1 = stable wetland, 2 = gain, 3 = loss
    dynamics = np.zeros_like(early, dtype=np.uint8)
    dynamics[early_wet & late_wet] = 1    # stable wetland
    dynamics[~early_wet & late_wet] = 2   # gain
    dynamics[early_wet & ~late_wet] = 3   # loss
    # ~early_wet & ~late_wet remains 0    # stable non-wetland

    # Write output
    profile.update(dtype="uint8", count=1, nodata=None)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(dynamics, 1)

    # Compute statistics
    stats = {
        "wetland_gain_pixels": int(np.sum(dynamics == 2)),
        "wetland_loss_pixels": int(np.sum(dynamics == 3)),
        "stable_wetland_pixels": int(np.sum(dynamics == 1)),
        "stable_nonwetland_pixels": int(np.sum(dynamics == 0)),
    }

    logger.info(
        "Wetland dynamics: gain=%d, loss=%d, stable_wet=%d, stable_dry=%d -> %s",
        stats["wetland_gain_pixels"],
        stats["wetland_loss_pixels"],
        stats["stable_wetland_pixels"],
        stats["stable_nonwetland_pixels"],
        output_path,
    )

    return {
        "output_path": str(output_path),
        "statistics": stats,
    }


def compare_with_nwi(
    prediction_path: Union[str, Path],
    reference_path: Union[str, Path],
    **kwargs: Any,
) -> Dict[str, Any]:
    """Compare wetland predictions against a reference raster (e.g. NWI).

    Computes per-class and overall accuracy metrics including IoU, F1,
    precision, recall, and a confusion matrix.

    Args:
        prediction_path: Path to classified prediction raster.
        reference_path: Path to reference classification raster (e.g.
            from ``reclassify_nwi``).

    Returns:
        Dict with keys:
            - ``overall_accuracy``: Fraction of correctly classified pixels.
            - ``mean_iou``: Mean IoU across all present classes.
            - ``per_class_iou``: Dict mapping class ID to IoU.
            - ``per_class_f1``: Dict mapping class ID to F1 score.
            - ``per_class_precision``: Dict mapping class ID to precision.
            - ``per_class_recall``: Dict mapping class ID to recall.
            - ``confusion_matrix``: 2D list (row=reference, col=prediction).
            - ``num_classes``: Number of unique classes found.

    Raises:
        FileNotFoundError: If prediction_path or reference_path does not exist.
    """
    import rasterio

    prediction_path = Path(prediction_path)
    reference_path = Path(reference_path)

    if not prediction_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {prediction_path}")
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference file not found: {reference_path}")

    with rasterio.open(prediction_path) as src:
        pred = src.read(1).astype(np.int32)

    with rasterio.open(reference_path) as src:
        ref = src.read(1).astype(np.int32)

    if pred.shape != ref.shape:
        raise ValueError(
            f"Shape mismatch: prediction={pred.shape}, reference={ref.shape}. "
            "Rasters must be co-registered."
        )

    # Find all unique classes across both
    all_classes = sorted(set(np.unique(pred)) | set(np.unique(ref)))
    num_classes = len(all_classes)
    class_to_idx = {c: i for i, c in enumerate(all_classes)}

    # Build confusion matrix (row=reference, col=prediction) — vectorized
    remap = np.vectorize(class_to_idx.__getitem__)
    ref_idx = remap(ref.ravel())
    pred_idx = remap(pred.ravel())
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (ref_idx, pred_idx), 1)

    # Overall accuracy
    overall_accuracy = float(np.trace(cm)) / float(np.sum(cm)) if np.sum(cm) > 0 else 0.0

    # Per-class metrics
    per_class_iou = {}
    per_class_f1 = {}
    per_class_precision = {}
    per_class_recall = {}

    for cls in all_classes:
        idx = class_to_idx[cls]
        tp = cm[idx, idx]
        fp = np.sum(cm[:, idx]) - tp
        fn = np.sum(cm[idx, :]) - tp

        # IoU = TP / (TP + FP + FN)
        denom_iou = tp + fp + fn
        iou = float(tp) / float(denom_iou) if denom_iou > 0 else 0.0
        per_class_iou[int(cls)] = iou

        # Precision = TP / (TP + FP)
        denom_prec = tp + fp
        precision = float(tp) / float(denom_prec) if denom_prec > 0 else 0.0
        per_class_precision[int(cls)] = precision

        # Recall = TP / (TP + FN)
        denom_rec = tp + fn
        recall = float(tp) / float(denom_rec) if denom_rec > 0 else 0.0
        per_class_recall[int(cls)] = recall

        # F1 = 2 * precision * recall / (precision + recall)
        denom_f1 = precision + recall
        f1 = 2.0 * precision * recall / denom_f1 if denom_f1 > 0 else 0.0
        per_class_f1[int(cls)] = f1

    # Mean IoU
    iou_values = list(per_class_iou.values())
    mean_iou = float(np.mean(iou_values)) if iou_values else 0.0

    logger.info(
        "Comparison: OA=%.4f, mIoU=%.4f, classes=%d",
        overall_accuracy,
        mean_iou,
        num_classes,
    )

    return {
        "overall_accuracy": overall_accuracy,
        "mean_iou": mean_iou,
        "per_class_iou": per_class_iou,
        "per_class_f1": per_class_f1,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "confusion_matrix": cm.tolist(),
        "num_classes": num_classes,
    }


# ---------------------------------------------------------------------------
# Phase 5: Paper Experiments
# ---------------------------------------------------------------------------


def build_experiment_config(
    study_area: Dict[str, Any],
    output_root: Union[str, Path],
    architectures: Optional[List[Dict[str, Any]]] = None,
    overrides: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Build a complete experiment configuration for a study area.

    Merges ``EXPERIMENT_DEFAULTS`` with the given study area, architecture
    list, and any per-run overrides into a single configuration dict that
    can drive ``run_experiment.py``.

    Args:
        study_area: Dict with at least ``bbox`` and ``naip_years`` keys.
        output_root: Root directory for all experiment outputs.
        architectures: List of architecture configs, each a dict with at
            least an ``architecture`` key. Defaults to
            ``EXPERIMENT_DEFAULTS["architectures"]``.
        overrides: Optional dict of training hyperparameter overrides
            (e.g. ``{"num_epochs": 100}``).

    Returns:
        Dict with keys: ``study_area``, ``output_root``, ``paths``,
        ``training``, ``architectures``.

    Raises:
        ValueError: If ``study_area`` is missing ``bbox`` or ``naip_years``,
            or if an architecture is not in ``SUPPORTED_ARCHITECTURES``.
    """
    if "bbox" not in study_area:
        raise ValueError("study_area must contain a 'bbox' key")
    if "naip_years" not in study_area:
        raise ValueError("study_area must contain a 'naip_years' key")

    _validate_bbox(study_area["bbox"])

    output_root = Path(output_root)

    # Resolve architectures
    if architectures is None:
        archs = [dict(a) for a in EXPERIMENT_DEFAULTS["architectures"]]
    else:
        archs = [dict(a) for a in architectures]

    # Validate architectures
    for arch_cfg in archs:
        arch_name = arch_cfg.get("architecture", "")
        if arch_name not in SUPPORTED_ARCHITECTURES:
            raise ValueError(
                f"Unsupported architecture: {arch_name!r}. "
                f"Choose from: {sorted(SUPPORTED_ARCHITECTURES)}"
            )

    # Build training config from defaults + overrides
    training = {
        k: v
        for k, v in EXPERIMENT_DEFAULTS.items()
        if k != "architectures"
    }
    if overrides:
        valid_keys = set(training.keys()) | {
            "download_max_retries", "download_timeout",
            "pre_downloaded_dem", "pre_downloaded_dem_tiles",
        }
        unknown = set(overrides) - valid_keys
        if unknown:
            raise ValueError(
                f"Unknown override keys: {sorted(unknown)}. "
                f"Valid keys: {sorted(valid_keys)}"
            )
        training.update(overrides)

    # Build output paths
    paths = {
        "naip_dir": str(output_root / "naip"),
        "dem_path": str(output_root / "dem" / "dem.tif"),
        "nwi_path": str(output_root / "nwi" / "nwi.gpkg"),
        "composites_dir": str(output_root / "composites"),
        "tiles_dir": str(output_root / "tiles"),
        "models_dir": str(output_root / "models"),
        "predictions_dir": str(output_root / "predictions"),
        "results_dir": str(output_root / "results"),
    }

    return {
        "study_area": study_area,
        "output_root": str(output_root),
        "paths": paths,
        "training": training,
        "architectures": archs,
    }


def format_results_table(
    results: List[Dict[str, Any]],
    class_names: Optional[Dict[int, str]] = None,
    **kwargs: Any,
) -> str:
    """Format experiment results as a human-readable Markdown table.

    Args:
        results: List of result dicts, each with at least
            ``name``, ``overall_accuracy``, ``mean_iou``,
            ``per_class_iou``, and ``per_class_f1``.
        class_names: Optional mapping of class ID to name.
            Defaults to ``COWARDIN_CLASSES``.

    Returns:
        Markdown-formatted table string.

    Raises:
        ValueError: If results is empty or missing required keys.
    """
    if not results:
        raise ValueError("results must not be empty")

    required = {"overall_accuracy", "mean_iou", "per_class_iou", "per_class_f1"}
    for r in results:
        missing = required - set(r.keys())
        if missing:
            raise ValueError(
                f"Result missing required keys: {sorted(missing)}"
            )

    if class_names is None:
        class_names = dict(COWARDIN_CLASSES)

    # Collect all class IDs across results
    all_classes = sorted(
        set().union(*(set(r["per_class_iou"].keys()) for r in results))
    )

    # Build header
    header_parts = ["| Model | OA | mIoU"]
    for cls in all_classes:
        name = class_names.get(cls, class_names.get(int(cls), f"Class {cls}"))
        header_parts.append(f"IoU {name}")
        header_parts.append(f"F1 {name}")
    header = " | ".join(header_parts) + " |"

    # Separator
    sep_parts = ["---"] * (3 + 2 * len(all_classes))
    separator = "| " + " | ".join(sep_parts) + " |"

    # Rows
    rows = []
    for r in results:
        name = r.get("name", "Model")
        oa = f"{r['overall_accuracy']:.4f}"
        miou = f"{r['mean_iou']:.4f}"
        parts = [f"| {name} | {oa} | {miou}"]
        for cls in all_classes:
            iou = r["per_class_iou"].get(cls, r["per_class_iou"].get(int(cls), 0.0))
            f1 = r["per_class_f1"].get(cls, r["per_class_f1"].get(int(cls), 0.0))
            parts.append(f"{iou:.4f}")
            parts.append(f"{f1:.4f}")
        rows.append(" | ".join(parts) + " |")

    return "\n".join([header, separator] + rows)


def save_experiment_results(
    results: List[Dict[str, Any]],
    output_path: Union[str, Path],
    config: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> str:
    """Save experiment results to a JSON file with a summary table.

    Args:
        results: List of result dicts from ``compare_with_nwi`` augmented
            with ``name``, ``architecture``, and ``encoder_name``.
        output_path: Path to write the JSON file.
        config: Optional experiment configuration to include.

    Returns:
        Path to the saved JSON file.

    Raises:
        ValueError: If results is empty.
    """
    import json

    if not results:
        raise ValueError("results must not be empty")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate summary table
    summary_table = format_results_table(results)

    payload: Dict[str, Any] = {
        "results": results,
        "summary_table": summary_table,
    }
    if config is not None:
        payload["config"] = config

    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    logger.info("Experiment results saved to %s", output_path)
    return str(output_path)
