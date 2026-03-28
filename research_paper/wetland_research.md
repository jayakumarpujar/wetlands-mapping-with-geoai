# Multi-Temporal NAIP and LiDAR Wetland Mapping with GeoAI

> Living research document — updated as implementation progresses.
> Authors: Jayakumar Pujar, Qiusheng Wu, et al.

---

## 1. Research Context and Motivation

### 1.1 Problem Statement

Wetland mapping at fine spatial resolution remains challenging due to:
- **Small wetland size**: Median PPR wetland is 0.12 ha — smaller than a single Landsat pixel (Wu 2019, Table 3)
- **Temporal dynamics**: Wetlands exhibit inter- and intra-annual inundation changes driven by climate and land use
- **Label scarcity**: NWI was created from 1980s aerial photos and is outdated; manual labeling is expensive
- **Spectral heterogeneity**: Water appearance varies with turbidity, algae, sun angle, causing classification ambiguity

### 1.2 Gap Analysis

| Aspect | Wu 2019 (RSE) | Igwe 2026 (RSASE) | **Our Paper (Gap Filled)** |
|--------|---------------|--------------------|-----------------------------|
| Method | Unsupervised k-means | Weakly supervised CNN | **Weakly supervised CNN on 1m imagery** |
| Imagery | 1m NAIP (multi-temporal) | 10m Sentinel-2 (single composite) | **1m NAIP (multi-temporal)** |
| LiDAR | Depression filtering only | Elevation + slope input | **Depression filtering + elevation input** |
| Classes | Binary (water/non-water) | 7-class (4 wetland + 3 upland) | **7-class** |
| Architecture | k-means clustering | U-Net++, DeepLabV3+ | **U-Net++, DeepLabV3+** |
| Resolution | 1m | 10m | **1m** |
| Platform | Google Earth Engine | GEE + PyTorch | **geoai (open-source Python package)** |

### 1.3 Key Innovation

**First study to combine:**
1. Multi-temporal 1m NAIP imagery (Wu's data strength)
2. Weakly supervised deep learning (Igwe's methodological strength)
3. LiDAR depression-based spatial filtering (Wu's algorithmic innovation)
4. Open-source geoai package implementation (Wu's software ecosystem)

This bridges Wu 2019's unsupervised approach with modern deep learning, at the same 1m resolution that captured sub-hectare wetlands Landsat/Sentinel cannot see.

---

## 2. Study Area

**Prairie Pothole Region (PPR), Central North Dakota**
- 26 HUC-10 watersheds (same as Wu 2019)
- Three HUC-8 subbasins: James Headwaters (#10160001), Pipestem (#10160002), Apple Creek (#10130103)
- Watershed areas: 31,800 ha to 99,800 ha (total: 1,657,600 ha)
- Dominant land cover: grassland (38%), cultivated crops (36%), pasture/hay (11%), open water (6%), emergent wetlands (5%)
- Contains millions of depressional wetlands (potholes) critical for migratory waterfowl

**Why this study area**: Direct comparison with Wu 2019 results; well-characterized wetland landscape; LiDAR data available.

---

## 3. Data Sources

### 3.1 Multi-Temporal NAIP Imagery

| Property | Details |
|----------|---------|
| Source | USDA NAIP via Planetary Computer STAC API |
| Resolution | 1m ground sample distance |
| Bands | Red, Green, Blue, Near-Infrared (4 bands) |
| Epoch 1 | 2015 (Sep 26 – Sep 27) — aligns with LiDAR acquisition |
| Epoch 2 | 2017 (Aug 7 – Aug 28) — latest in Wu 2019 study |
| Coverage | Full study area via DOQQ tiles |

**Thought process on epoch selection**:
- Wu 2019 used 6 epochs (2009, 2010, 2012, 2014, 2015, 2017)
- We start with 2015 + 2017 because:
  - 2015 aligns with LiDAR DEM acquisition (2011-2015), minimizing temporal mismatch
  - 2017 is the latest epoch Wu validated, enabling direct comparison
  - The 2015→2017 transition captures meaningful inundation dynamics (visible in Wu Fig. 3)
- Can extend to additional epochs if results warrant it

### 3.2 LiDAR / 3DEP DEM

| Property | Details |
|----------|---------|
| Source | USGS 3DEP (originally North Dakota LiDAR Dissemination Service) |
| Resolution | 1m |
| Acquisition | 2011-2015 (multiple campaigns) |
| Vertical accuracy | ~15 cm estimated |
| Products derived | (1) Elevation surface, (2) Depression depth via fill algorithm |

**Thought process on LiDAR usage**:
- Wu 2019 used LiDAR DEMs to delineate surface depressions via a depression-filling algorithm (Wu et al. 2018)
- Depressions represent potential wetland basins — pixels outside depressions are unlikely to be wetlands
- This is Wu's key algorithmic innovation: using topographic context to refine spectral classification
- We use depression depth as a model input band AND as a spatial filter for weak labels

### 3.3 National Wetlands Inventory (NWI)

| Property | Details |
|----------|---------|
| Source | U.S. Fish and Wildlife Service |
| Scale | 1:24,000 |
| Creation | 1980s (aerial photo interpretation) |
| Classes | Cowardin classification (palustrine emergent, forested, scrub-shrub, lacustrine, riverine) |
| Role | Weak label source (not ground truth — known to be outdated) |

**Thought process on NWI as weak labels**:
- NWI is the most spatially and categorically detailed wetland inventory for the U.S.
- However, it's outdated (1980s) and doesn't reflect contemporary inundation
- Wu 2019 used NWI only for comparison/validation, not for training
- Igwe 2026 used NWI as weak training labels with change detection filtering
- Our approach: use NWI for initial class labels, then filter with LiDAR depressions + temporal stability

---

## 4. Methodology

### 4.1 Input Feature Stack (10 Bands)

| Band | Source | Purpose | Justification |
|------|--------|---------|---------------|
| 1 | NAIP 2015 Red | Spectral | Standard optical band |
| 2 | NAIP 2015 Green | Spectral | Standard optical band |
| 3 | NAIP 2015 Blue | Spectral | Standard optical band |
| 4 | NAIP 2015 NIR | Spectral | Vegetation/water discrimination |
| 5 | NDVI 2015 | Vegetation index | (NIR-Red)/(NIR+Red), Wu 2019 used this |
| 6 | NDWI 2015 | Water index | (Green-NIR)/(Green+NIR), Wu 2019 used this |
| 7 | NDVI 2017 | Temporal change | Vegetation change between epochs |
| 8 | NDWI 2017 | Temporal change | Water change between epochs |
| 9 | 3DEP Elevation | Topography | Absolute elevation context |
| 10 | Depression Depth | Topography | Wu's innovation — depth of topographic depressions |

**Thought process on band count**:
- Wu 2019 used 6 bands per epoch: R, G, B, NIR, NDVI, NDWI (but applied k-means per-epoch separately)
- Igwe 2026 used 14 bands: 10 Sentinel-2 + 2 SAR + elevation + slope
- We chose 10 bands because:
  - NAIP has only 4 native bands (vs Sentinel-2's 10+), so we can't match Igwe's optical count
  - Including epoch 2's full 4 NAIP bands would add 4 more bands but with diminishing returns — the temporal signal is better captured by NDVI/NDWI change
  - Every band has a clear, non-redundant purpose
  - 10 bands is computationally lighter than 14, faster training at 1m resolution
  - The depression depth band is unique to our study — neither Wu nor Igwe used it as model input

### 4.2 Classification Schema (7 Classes)

| Code | Class | Type | Source |
|------|-------|------|--------|
| 0 | Water | Wetland | Cowardin — open water/lacustrine |
| 1 | Emergent Wetland | Wetland | Cowardin — palustrine emergent |
| 2 | Forested Wetland | Wetland | Cowardin — palustrine forested |
| 3 | Scrub-Shrub Wetland | Wetland | Cowardin — palustrine scrub-shrub |
| 4 | Urban/Built-up | Upland | NLCD/NWI boundary |
| 5 | Forested Upland | Upland | NLCD/NWI boundary |
| 6 | Agriculture | Upland | NLCD/NWI boundary |

**Thought process on class count**:
- Wu 2019 was binary (inundated vs. not) — too coarse for wetland management
- Igwe 2026 used 7 classes (4 wetland + 3 upland) — matches Cowardin system + practical land cover needs
- 7 classes is the right balance: detailed enough for wetland type discrimination, not so many that weak labels become unreliable
- The 3 upland classes help quantify boundary confusion (a key accuracy concern in wetland mapping)

### 4.3 Weak Label Generation Strategy

Our labeling strategy combines the best elements of Wu and Igwe:

**Step 1: NWI Rasterization** (from Igwe)
- Rasterize NWI vector polygons to 1m resolution matching NAIP
- Reclassify Cowardin codes into 7 target classes
- This provides initial class labels with known spatial boundaries

**Step 2: LiDAR Depression Filtering** (from Wu)
- Compute depression mask from DEM using fill algorithm
- Retain only NWI wetland labels that fall within LiDAR-derived depressions
- This removes NWI polygons in areas with no topographic capacity to hold water
- Wu showed this removes 7.68-15.42% of false water clusters (avg 11.37%)

**Step 3: Multi-Temporal Stability Filtering** (hybrid Wu + Igwe)
- Compute NDVI and NDWI for both epochs (2015, 2017)
- Identify "stable pixels" where both indices show minimal change (threshold < 0.05)
- Retain only labels at stable pixels — these are most likely correctly classified
- Rationale: if a pixel is classified as "emergent wetland" by NWI and shows consistent vegetation signal across years, the label is reliable

**Step 4: Object-Level Confidence** (from Igwe, adapted)
- Instead of SNIC superpixels (designed for 10m Sentinel), use connected components at 1m
- For each connected component, compute the fraction of pixels that passed Steps 2-3
- Retain components where >50% of pixels are reliable (Igwe's threshold)
- This ensures spatial coherence of training labels

**Thought process on label strategy**:
- Pure NWI labels are noisy (1980s data, many boundaries have shifted)
- Wu's depression filtering is essential for PPR — removes shadow/terrain false positives
- Temporal stability (inspired by Igwe's LandTrendr approach, simplified for NAIP) ensures we train only on pixels with consistent spectral behavior
- We don't need LandTrendr specifically because we have direct NAIP-to-NAIP comparison (same sensor), whereas Igwe needed LandTrendr for Sentinel-2 time series normalization

### 4.4 Model Architecture

**Primary: U-Net++** (Zhou et al. 2018)
- Dense skip connections between encoder and decoder
- Better gradient flow than standard U-Net
- Igwe showed U-Net++ outperformed DeepLabV3+ (F1 91.3% vs 90.6%)

**Comparison: DeepLabV3+** (Chen et al. 2018)
- Atrous spatial pyramid pooling for multi-scale features
- Strong boundary delineation — important for small wetlands

**Encoder: ResNet-152** (Igwe's choice)
- Pre-trained on ImageNet
- First conv layer modified for 10-channel input (geoai's `train_timm_segmentation` handles this)

**Loss: Cross-Entropy + Dice** (Igwe's formulation)
- L = L_ce + L_dice
- Dice loss addresses class imbalance (wetlands are minority class)

**Training details** (following Igwe):
- Patch sizes: 256 × 256 pixels
- ~100,000 patches from 7 EPA Level-3 ecoregions (stratified sampling)
- 60:40 train/validation split
- 30 epochs, RMSProp optimizer, lr=0.0001
- Batch normalization + random rotation/flip augmentation

### 4.5 Integration with geoai Package

**This is critical to the research**: Wu created the geoai package as an open-source toolkit for geospatial AI. Our implementation must be a native geoai module, not a standalone script.

**Integration points:**
- `geoai.download.download_naip()` — existing NAIP download from Planetary Computer
- `geoai.landcover_utils.export_landcover_tiles()` — tile export with class filtering
- `geoai.timm_segment.train_timm_segmentation()` — model training with SMP + timm encoders
- `geoai.inference.predict_geotiff()` — spatial inference on large rasters
- `geoai.utils.stack_bands()` — band stacking utility

**Standalone module: `research_paper/wetland.py`** — all wetland-specific functions in a standalone package that imports from geoai as a dependency. Kept separate from geoai repo since this is personal research work.

---

## 5. Implementation Phases

### Phase 1: Data Pipeline ✓
**Status: Complete** (2026-03-25)

Functions implemented in `research_paper/wetland.py`:
- [x] `download_naip_timeseries(bbox, output_dir, years=None, max_items_per_year=10, overwrite=False)` — wraps `geoai.download.download_naip` per year; returns `Dict[int, List[str]]`; skips years with no data
- [x] `download_3dep_dem(bbox, output_path, resolution=10, overwrite=False)` — downloads via `py3dep.get_dem`; validates against `SUPPORTED_3DEP_RESOLUTIONS` {1, 3, 10, 30}
- [x] `download_nwi(bbox, output_path, overwrite=False)` — queries USFWS NWI ArcGIS REST endpoint; saves GeoPackage via GeoPandas
- [x] `compute_spectral_indices(naip_path, output_path=None, indices=None, overwrite=False)` — reads 4-band NAIP, computes NDVI/NDWI via `_compute_index()`; writes multi-band float32 GeoTIFF preserving CRS/extent
- [x] `extract_surface_depressions(dem_path, output_path=None, min_depth=0.1, min_area=0.0, overwrite=False)` — priority-flood fill via `_fill_depressions()` (Barnes et al. 2014); outputs depression depth = filled − original
- [x] `create_wetland_composite(naip_paths, dem_path, output_path, indices=None, include_depressions=True, overwrite=False)` — stacks NAIP bands + spectral indices + elevation + depression depth; reprojects DEM to NAIP grid via `rasterio.warp.reproject`

**Constants:** `COWARDIN_CLASSES` (6 classes, 0-5), `NWI_CODE_TO_CLASS` (8 NWI prefixes), `NAIP_BANDS` (4 bands, 1-indexed), `SPECTRAL_INDICES` (ndvi, ndwi formulas), `SUPPORTED_3DEP_RESOLUTIONS` (frozenset {1,3,10,30})

**Helpers:** `_validate_bbox()`, `_validate_index_names()`, `_parse_nwi_code()`, `_compute_index()`, `_fill_depressions()` (priority-flood with 8-connected neighbors, heapq)

**Tests:** 96 tests in `tests/test_wetland.py` — constants, signatures, validation, depression filling, spectral indices, integration with real rasterio I/O. All passing (1 skipped: geoai lazy-import test).

### Phase 2: Weak Label Generation ✓
**Status: Complete** (2026-03-25)

Functions implemented in `research_paper/wetland.py`:
- [x] `reclassify_nwi(nwi_path, raster_template, output_path, attribute_field="ATTRIBUTE", overwrite=False)` — reads NWI vector polygons, parses Cowardin codes via `_parse_nwi_code()`, rasterizes to template grid (CRS, resolution, extent); output: uint8 raster with class IDs 0-5
- [x] `generate_weak_labels(nwi_raster_path, depression_path, ndvi_paths, ndwi_paths, output_path, depression_threshold=0.0, stability_threshold=0.05, min_component_fraction=0.5, overwrite=False)` — 4-step filtering: (1) NWI raster labels, (2) depression filter via depth threshold, (3) temporal stability filter on NDVI/NDWI change across epochs, (4) connected-component confidence filter retaining components where ≥50% of pixels survived steps 2-3; uses `scipy.ndimage.label` for 4-connected components
- [x] `export_training_tiles(composite_path, label_path, output_dir, tile_size=256, stride=None, min_valid_fraction=0.5, overwrite=False)` — sliding-window extraction of paired image/label GeoTIFF tiles into `images/` and `labels/` subdirectories; filters tiles below `min_valid_fraction` labeled pixels; returns dict with `num_tiles`, `output_dir`, `tile_size`

**Tests:** 41 tests in `tests/test_wetland_phase2.py` — signature checks, input validation, integration tests with synthetic rasters (rasterization, depression/stability filtering, tile extraction). All passing.

### Phase 3: Model Training ✓
**Status: Complete** (2026-03-26)

Functions implemented in `research_paper/wetland.py`:
- [x] `train_wetland_model(tiles_dir, output_dir, architecture="unetplusplus", encoder_name="resnet50", num_classes=6, in_channels=14, num_epochs=50, batch_size=8, learning_rate=1e-3, loss_function="focal", use_class_weights=True, val_split=0.2, ...)` — wraps `geoai.landcover_train.train_segmentation_landcover` with wetland-specific defaults; validates architecture against `SUPPORTED_ARCHITECTURES`, loss function against {"focal", "crossentropy"}, and all numeric parameters; returns dict with `model_path`, `architecture`, `encoder_name`, `num_classes`, `in_channels`, `output_dir`; raises `RuntimeError` if no checkpoint found after training

**Constants:** `SUPPORTED_ARCHITECTURES` (frozenset of 10 SMP architectures: unet, unetplusplus, deeplabv3, deeplabv3plus, fpn, pspnet, linknet, manet, pan, upernet)

**Tests:** 30 tests in `tests/test_wetland_phase3.py` — signature/default verification (10), input validation including loss_function/num_epochs/batch_size/learning_rate bounds checks (10), end-to-end integration with synthetic tiles (5), module exports (1). All passing.

### Phase 4: Inference & Dynamics ✓
**Status: Complete** (2026-03-26)

Functions implemented in `research_paper/wetland.py`:
- [x] `predict_wetlands(model_path, composite_path, output_path, architecture="unetplusplus", encoder_name="resnet50", num_classes=6, in_channels=14, tile_size=256, overlap=128, batch_size=4, ...)` — loads trained SMP model, runs sliding-window inference with overlap blending (softmax probability accumulation + argmax); handles edge tiles and channel padding; validates architecture against `SUPPORTED_ARCHITECTURES`; outputs single-band uint8 classified GeoTIFF matching input grid
- [x] `map_wetland_dynamics(prediction_paths, output_path, overwrite=False)` — compares classified predictions across epochs (last pair); produces change map with codes: 0=stable non-wetland, 1=stable wetland, 2=wetland gain, 3=wetland loss; validates spatial alignment (shape match); returns dict with `output_path` and `statistics` (gain/loss/stable pixel counts)
- [x] `compare_with_nwi(prediction_path, reference_path)` — computes full accuracy assessment: overall accuracy, mean IoU, per-class IoU/F1/precision/recall, confusion matrix; uses vectorized `np.add.at` for O(1) confusion matrix computation on large rasters; validates shape alignment between prediction and reference

**Tests:** 37 tests in `tests/test_wetland_phase4.py` — signature checks, input validation, integration tests with synthetic data (model inference on random weights, dynamics with known gain/loss patterns, perfect vs imperfect accuracy assessment). All passing.

### Phase 5: Paper Experiments ✓
**Status: Complete** (2026-03-26)

Experiment script: `research_paper/run_experiment.py`

Functions implemented in `research_paper/wetland.py`:
- [x] `build_experiment_config(study_area, output_root, architectures=None, overrides=None)` — merges `EXPERIMENT_DEFAULTS` with study area, architecture list, and per-run overrides into a single config dict; validates bbox via `_validate_bbox()`, architectures against `SUPPORTED_ARCHITECTURES`, and override keys against known parameter set
- [x] `format_results_table(results, class_names=None)` — formats experiment results as Markdown table with OA, mIoU, and per-class IoU/F1 columns; defaults to `COWARDIN_CLASSES` names
- [x] `save_experiment_results(results, output_path, config=None)` — saves results as JSON with embedded summary table; creates parent directories

**Constants:** `PPR_STUDY_AREA` (bbox, naip_years, huc8_codes for Wu 2019 study area), `EXPERIMENT_DEFAULTS` (training hyperparameters: tile_size=256, num_epochs=50, batch_size=8, lr=1e-3, 6 classes, 10 input channels, focal loss, U-Net++ and DeepLabV3+ architectures)

**Experiment script** (`research_paper/run_experiment.py`):
- [x] `run_ppr_experiment(output_root, overrides)` — end-to-end pipeline orchestration with real-time `print(..., flush=True)` progress for each phase (Colab buffers `logging` output, so explicit flush is required for visibility)
- [x] `run_data_download(config)` — Phase 1a: NAIP + DEM + NWI download
- [x] `run_composites(config, download_result)` — Phase 1b: indices, depressions, per-epoch composites, **plus 10-band training composite** stacking NAIP(4) + NDVI/NDWI(2015) + NDVI/NDWI(2017) + DEM + depression; auto-reprojects all bands to NAIP grid
- [x] `run_weak_labels(config, download_result, composite_result)` — Phase 2: NWI reclassification + temporal filtering + tiles from training composite; validates composite/label shape match; auto-detects `in_channels` from tile band count; raises on zero tiles
- [x] `run_training(config, label_result)` — Phase 3: multi-architecture training; prints GPU/CPU device; warns if CUDA unavailable
- [x] `run_inference(config, trained_models, composite_result)` — Phase 4a: predicts on training composite (same 10-band structure as training); per-epoch composites are 8-band and incompatible with trained model
- [x] `run_evaluation(config, all_predictions, label_result)` — Phase 4b: accuracy + dynamics
- [x] CLI with `--output-root`, `--num-epochs`, `--batch-size`, `--learning-rate`, `--verbose` flags

**Tests:** 44 tests in `tests/test_wetland_phase5.py` — constants (PPR_STUDY_AREA, EXPERIMENT_DEFAULTS), config building (signature, validation including unknown override key detection, integration), results formatting (signature, validation, integration with Cowardin class names), results saving (JSON creation, parent dirs, config inclusion), module exports. All passing.

Remaining paper tasks (manual execution):
- [ ] Run full pipeline on PPR study area (2015 + 2017 epochs)
- [ ] Train U-Net++ and DeepLabV3+ models, compare performance
- [ ] Generate accuracy tables (OA, IoU, F1 per class)
- [ ] Create comparison figures (our method vs Wu 2019 vs NWI)
- [ ] Ablation study: contribution of depression depth and temporal bands
- [ ] Scale experiment: varying training data size

---

## 6. Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-03-25 | 10 input bands (not 14) | NAIP has 4 native bands vs Sentinel-2's 10; every band must have clear purpose |
| 2026-03-25 | 7 classes (matching Igwe) | Wu wants to advance beyond binary; 7-class matches Cowardin + upland context |
| 2026-03-25 | Wu-style depression filter + NWI weak labels | Combines Wu's topographic innovation with Igwe's weak supervision approach |
| 2026-03-25 | PPR North Dakota study area | Direct comparison with Wu 2019; well-characterized landscape |
| 2026-03-25 | Epochs 2015 + 2017 | LiDAR alignment (2015) + latest Wu epoch (2017); extendable |
| 2026-03-25 | geoai-native implementation | Wu's explicit requirement; enables reproducibility and community adoption |
| 2026-03-25 | 4-step weak label filtering | NWI → depression filter → temporal stability → object confidence; combines Wu's depression innovation (step 2) with Igwe's weak supervision strategy (steps 3-4) |
| 2026-03-25 | scipy connected-component filter (4-conn) | Object-level confidence at 1m uses 4-connected components instead of Igwe's SNIC superpixels (designed for 10m Sentinel-2); min_component_fraction=0.5 matches Igwe's threshold |
| 2026-03-25 | Standalone research_paper/ module | Personal research kept separate from geoai repo; imports geoai as dependency |
| 2026-03-26 | Focal loss default (not CE+Dice) | geoai's landcover_train supports focal loss natively; handles class imbalance well for wetland minority classes; Dice loss available via future extension |
| 2026-03-26 | 6 Cowardin classes (not 7) | Simplified from doc's 7-class to 6: Upland(0), Water(1), Emergent(2), Forested(3), Scrub-Shrub(4), Other(5); upland subclasses merged since NWI doesn't provide upland type labels |
| 2026-03-26 | Vectorized confusion matrix | compare_with_nwi uses np.add.at instead of nested loops; critical for county/state-scale rasters (100M+ pixels) |
| 2026-03-26 | Softmax blending for inference | predict_wetlands accumulates softmax probabilities in overlapping regions then argmax; more robust than hard-vote stitching at tile boundaries |
| 2026-03-26 | Override key validation in experiment config | build_experiment_config rejects unknown override keys to prevent silent typo bugs (e.g. "num_epoch" vs "num_epochs") that would silently drop intended overrides |
| 2026-03-26 | Separate NDVI/NDWI extraction for temporal filtering | run_composites writes single-band NDVI and NDWI files from multi-band indices raster; prevents generate_weak_labels from reading NDVI band as NDWI |
| 2026-03-26 | 10 input channels for experiment | 4 NAIP bands + NDVI + NDWI (2015) + NDVI + NDWI (2017) + elevation + depression depth = 10 channels; matches study-area-specific multi-temporal stack |
| 2026-03-27 | Flushed print progress in run_experiment | Colab buffers `logging` output for long-running cells; replaced `logger.info` with `print(..., flush=True)` in `run_ppr_experiment` so users see phase timing (e.g. `[120s] Phase 3: Training models ...`) in real-time |
| 2026-03-27 | 10-band training composite | Per-epoch composites are 8 bands (NAIP+NDVI+NDWI+DEM+dep); training needs 10 bands matching the design (4 NAIP + 4 temporal indices + DEM + depression); `run_composites` now builds a dedicated training composite by stacking aligned bands from both epochs |
| 2026-03-27 | Inference uses training composite | Model trained on 10-band composite cannot predict on 8-band per-epoch composites; `run_inference` now uses `training_composite_path` for all predictions; dynamics mapping deferred until per-epoch 10-band composites are supported |
| 2026-03-27 | Auto-detect in_channels from tiles | `in_channels` config is overridden at runtime by reading the actual band count from the training composite; prevents mismatch between config (10) and reality if band count changes |
| 2026-03-27 | Shape validation in tile export | `export_training_tiles` now raises `ValueError` if composite and label rasters have different shapes; prevents silent misalignment that wastes 45+ min before failing |
| 2026-03-27 | Raster alignment in generate_weak_labels | Depression, NDVI, and NDWI rasters are reprojected to match NWI raster grid when shapes/CRS differ; DEM covers full bbox while NAIP tiles cover smaller areas |
| 2026-03-27 | NAIP 4-band validation | `create_wetland_composite` validates NAIP files have at least 4 bands (R,G,B,NIR); reads only first 4 bands to avoid including extra bands from some providers |
| 2026-03-27 | GPU availability warning | `train_wetland_model` prints explicit warning when CUDA is unavailable, directing user to enable GPU in Colab runtime settings |

---

## 7. References

- Wu, Q., Lane, C.R., Li, X., Zhao, K., Zhou, Y., Clinton, N., DeVries, B., Golden, H.E., Lang, M.W. (2019). Integrating LiDAR data and multi-temporal aerial imagery to map wetland inundation dynamics using Google Earth Engine. *Remote Sensing of Environment*, 228, 1-13.
- Igwe, V., Salehi, B., Marjani, M., Farhadi, N., Mahdianpari, M. (2026). Cost-effective statewide wetland inventory update using weakly supervised deep learning: A case study in Minnesota, USA. *Remote Sensing Applications: Society and Environment*, 41, 101871.
- Wu, Q. (2018). GIS and Remote Sensing Applications in Wetland Mapping and Monitoring. *Comprehensive Geographic Information Systems*, 140-157.
- Zhou, Z., Rahman Siddiquee, M.M., Tajbakhsh, N., Liang, J. (2018). U-Net++: A Nested U-Net Architecture for Medical Image Segmentation. *DLMIA 2018*.
- Chen, L.C., Zhu, Y., Papandreou, G., Schroff, F., Adam, H. (2018). Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation. *ECCV 2018*.
