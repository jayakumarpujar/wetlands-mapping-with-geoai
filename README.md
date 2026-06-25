# Wetland Mapping Using GeoAI: Multi-Temporal NAIP and LiDAR with Weakly Supervised Deep Learning

A research project implementing weakly supervised deep learning for high-resolution (1m) wetland mapping in the Prairie Pothole Region using multi-temporal NAIP imagery and LiDAR-derived features, built on the [geoai](https://github.com/opengeos/geoai) open-source package.

**Authors:** Jayakumar Pujar, Qiusheng Wu

## Research Summary

This project bridges two methodological approaches for wetland mapping:

- **Wu et al. (2019, RSE)** demonstrated that multi-temporal 1m NAIP + LiDAR depression filtering can map sub-hectare wetlands that Landsat/Sentinel cannot resolve, but used unsupervised k-means clustering.
- **Igwe et al. (2026, RSASE)** applied weakly supervised CNNs (U-Net++, DeepLabV3+) for cost-effective wetland inventory, but at 10m Sentinel-2 resolution.

**Our contribution:** Combine multi-temporal 1m NAIP imagery with weakly supervised deep learning and LiDAR depression-based spatial filtering to map wetlands at 1m resolution using a 3-class Cowardin schema (Upland, Water, Emergent). Beyond the U-Net++/DeepLabV3+ baselines, we propose **WetMamba** — a Prithvi-EO geospatial foundation-model encoder with a Mamba state-space decoder and a Depression-Aware Gating module that injects the LiDAR depression prior directly into the network.

## Study Area

Prairie Pothole Region, Central North Dakota -- 26 HUC-10 watersheds across three HUC-8 subbasins (James Headwaters, Pipestem, Apple Creek). Same study area as Wu 2019 RSE.

## Pipeline Architecture

The experiment is organized into 5 phases:

| Phase | Description | Key Functions |
|-------|-------------|---------------|
| **1. Data Pipeline** | Download NAIP, 3DEP DEM, NWI; compute spectral indices; create composites | `download_naip_timeseries`, `compute_spectral_indices`, `create_wetland_composite` |
| **2. Weak Labels** | Reclassify NWI to Cowardin classes (PFO/PSS → `IGNORE_INDEX`); generate weak labels with depression + temporal filtering | `generate_weak_labels`, `reclassify_nwi`, `export_training_tiles` |
| **3. Model Training** | Train U-Net++/DeepLabV3+ baselines and WetMamba with unified-focal loss + class weights | `train_wetland_model` |
| **4. Inference & Evaluation** | Predict wetlands, map dynamics, compare with NWI reference | `predict_wetlands`, `map_wetland_dynamics`, `compare_with_nwi` |
| **5. Experiment Orchestration** | End-to-end pipeline, config management, result formatting | `run_experiment.py`, `build_experiment_config` |

## Technical Details

- **Input:** 10-channel composite (4 NAIP bands + NDVI + NDWI per epoch + elevation + depression depth)
- **Architectures:** U-Net++ and DeepLabV3+ baselines via [segmentation-models-pytorch](https://github.com/qubvel-org/segmentation_models.pytorch), plus **WetMamba** (Prithvi-EO-2.0-300M encoder + Mamba SSM decoder + Depression-Aware Gating + temporal SSM fusion) and isolating baselines (Prithvi-linear, UNetMamba, SegFormer, Swin-UNet) in `research_paper/models/`
- **Encoder:** ResNet-50 (ImageNet pretrained) for SMP baselines; Prithvi-EO foundation model (LoRA-tuned) for WetMamba
- **Loss:** `unified_focal` (focal CE + focal Tversky blend) with inverse-frequency class weights and a collapse-guard auto-fallback
- **Classes:** Upland (0), Water (1), Emergent (2). Forested/Scrub-Shrub (PFO/PSS) and unrecognized NWI codes are masked via `IGNORE_INDEX = 255` (excluded from loss and metrics) — they are near-absent in treeless PPR and were unlearnable under the prior 4-class schema (IoU 0.097)
- **Tile size:** 256x256 pixels
- **NAIP epochs:** 2015, 2017

## Recent Changes (2026-06-24)

Driven by the first geographic test run (western PPR, North Dakota):

- **3-class schema (was 4).** The 4-class run showed "Other" (PFO+PSS forested/scrub-shrub
  wetland) collapse — IoU 0.097 on just 0.24% of pixels. PPR is treeless, so these types are
  near-absent and unlearnable. The schema is now **Upland / Water / Emergent**, and PFO/PSS +
  unrecognized NWI codes map to **`IGNORE_INDEX = 255`** (a uint8-safe sentinel, masked from
  both the loss and evaluation rather than relabeled as Upland).
- **Fixed an evaluation reporting bug.** `evaluate_tiles.py` previously printed class names
  (`upland/emergent/forested/pond`) that did not match the trained schema; metrics were
  numerically correct but mislabeled. It now imports `COWARDIN_CLASSES` as the single source
  of truth. (Corrected interpretation: the "strong" class was open **Water**, real Emergent
  was 0.53, and the 0.097 collapse was the dropped "Other" class.)
- **Synced docs to code.** `wetland_research.md` schema, hyperparameters (ResNet-50, AdamW,
  lr 3e-4, 100 epochs, 80:20, unified-focal), API signatures, and results section were
  corrected; added a **South Dakota multi-state** future-work plan (§8.7) and decision-log
  entries.
- **Expected impact:** removing the dead class lifts reported mean IoU from 0.546 toward ~0.70.
  A retrain on the regenerated 3-class labels is required to populate the final numbers.
- Added `fiona` as an explicit dependency (needed by NWI reclassification).

## Project Structure

```
.
├── research_paper/
│   ├── __init__.py              # Package init
│   ├── wetland.py               # Core module (all 5 phases, ~2800 lines)
│   ├── models/                  # WetMamba + baselines (wetmamba.py, baselines.py,
│   │                            #   mamba_decoder.py, temporal_ssm.py, dag_module.py)
│   ├── train_benchmark.py       # Multi-model benchmark training pipeline
│   ├── evaluate_tiles.py        # Tile-based test evaluation (3-class)
│   ├── run_experiment.py        # CLI pipeline orchestrator
│   └── wetland_research.md      # Living research document
├── tests/
│   ├── test_wetland.py          # Phase 1 tests (96 tests)
│   ├── test_wetland_phase2.py   # Phase 2 tests (41 tests)
│   ├── test_wetland_phase3.py   # Phase 3 tests (30 tests)
│   ├── test_wetland_phase4.py   # Phase 4 tests (37 tests)
│   ├── test_wetland_phase5.py   # Phase 5 tests (44 tests)
│   └── test_models.py           # Model architecture tests (34 tests)
├── notebooks/
│   └── run_experiment.ipynb     # Google Colab notebook (coming soon)
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation

```bash
git clone https://github.com/jayakumarpujar/wetlands-mapping-using-geoai-ResearchProject-.git
cd wetlands-mapping-using-geoai-ResearchProject-
pip install -r requirements.txt
```

## Usage

### Run the full experiment

```bash
python -m research_paper.run_experiment --output-root ./ppr_experiment --verbose
```

### With custom hyperparameters

```bash
python -m research_paper.run_experiment \
    --output-root ./ppr_experiment \
    --num-epochs 100 \
    --batch-size 16 \
    --learning-rate 0.0005 \
    --verbose
```

### Run tests

```bash
pytest tests/ -v
```

## Dependencies

- **[geoai](https://github.com/opengeos/geoai)** -- Geospatial AI package (data download, processing)
- **[segmentation-models-pytorch](https://github.com/qubvel-org/segmentation_models.pytorch)** -- CNN architectures
- **PyTorch** -- Deep learning framework
- **rasterio, geopandas, numpy, scikit-learn** -- Geospatial and ML utilities

## Test Coverage

~283 tests; 280 passing, 1 skipped. Two `generate_weak_labels` integration tests
(depression/stability filter) are known pre-existing failures unrelated to the 3-class
change and are tracked separately.

| Phase | Tests | Coverage |
|-------|-------|----------|
| Phase 1: Data Pipeline | 96 | Data download, spectral indices, composites |
| Phase 2: Weak Labels | 41 | NWI reclassification (incl. `IGNORE_INDEX`), weak label generation |
| Phase 3: Training | 30 | Model training, loss functions, class weights |
| Phase 4: Inference | 37 | Prediction, dynamics, NWI comparison |
| Phase 5: Experiments | 44 | Config, result formatting, orchestration |
| Models | 34 | WetMamba + baseline architecture shapes/forward |

> Note: NWI reclassification requires the `fiona` package (`pip install fiona`).

## References

1. Wu, Q., Lane, C.R., Li, X., Zhao, K., Zhou, Y., Clinton, N., DeVries, B., Golden, H.E. and Lang, M.W. (2019). Integrating LiDAR data and multi-temporal aerial imagery to map wetland inundation dynamics using Google Earth Engine. *Remote Sensing of Environment*, 228, pp.1-13.

2. Igwe, O.M., Lane, C.R., Golden, H.E. and Wu, Q. (2026). Cost-effective statewide wetland inventory update using weakly supervised deep learning: A case study in Minnesota, USA. *Remote Sensing Applications: Society and Environment*.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- Built on the [geoai](https://github.com/opengeos/geoai) open-source package by Dr. Qiusheng Wu
- Study area and methodology informed by Wu et al. (2019) RSE
- Weakly supervised approach adapted from Igwe et al. (2026)
