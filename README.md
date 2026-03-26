# Wetland Mapping Using GeoAI: Multi-Temporal NAIP and LiDAR with Weakly Supervised Deep Learning

A research project implementing weakly supervised deep learning for high-resolution (1m) wetland mapping in the Prairie Pothole Region using multi-temporal NAIP imagery and LiDAR-derived features, built on the [geoai](https://github.com/opengeos/geoai) open-source package.

**Authors:** Jayakumar Pujar, Qiusheng Wu

## Research Summary

This project bridges two methodological approaches for wetland mapping:

- **Wu et al. (2019, RSE)** demonstrated that multi-temporal 1m NAIP + LiDAR depression filtering can map sub-hectare wetlands that Landsat/Sentinel cannot resolve, but used unsupervised k-means clustering.
- **Igwe et al. (2026, RSASE)** applied weakly supervised CNNs (U-Net++, DeepLabV3+) for cost-effective wetland inventory, but at 10m Sentinel-2 resolution.

**Our contribution:** Combine multi-temporal 1m NAIP imagery with weakly supervised deep learning and LiDAR depression-based spatial filtering to map wetlands at 1m resolution using 6-class Cowardin classification.

## Study Area

Prairie Pothole Region, Central North Dakota -- 26 HUC-10 watersheds across three HUC-8 subbasins (James Headwaters, Pipestem, Apple Creek). Same study area as Wu 2019 RSE.

## Pipeline Architecture

The experiment is organized into 5 phases:

| Phase | Description | Key Functions |
|-------|-------------|---------------|
| **1. Data Pipeline** | Download NAIP, 3DEP DEM, NWI; compute spectral indices; create composites | `download_naip_timeseries`, `compute_spectral_indices`, `create_wetland_composite` |
| **2. Weak Labels** | Reclassify NWI to Cowardin classes; generate weak labels with depression + temporal filtering | `generate_weak_labels`, `reclassify_nwi`, `export_training_tiles` |
| **3. Model Training** | Train U-Net++ and DeepLabV3+ with focal loss and class weights | `train_wetland_model` |
| **4. Inference & Evaluation** | Predict wetlands, map dynamics, compare with NWI reference | `predict_wetlands`, `map_wetland_dynamics`, `compare_with_nwi` |
| **5. Experiment Orchestration** | End-to-end pipeline, config management, result formatting | `run_experiment.py`, `build_experiment_config` |

## Technical Details

- **Input:** 10-channel composite (4 NAIP bands + NDVI + NDWI per epoch + elevation + depression depth)
- **Architectures:** U-Net++ and DeepLabV3+ via [segmentation-models-pytorch](https://github.com/qubvel-org/segmentation_models.pytorch)
- **Encoder:** ResNet-50 (ImageNet pretrained)
- **Loss:** Focal loss with inverse-frequency class weights
- **Classes:** Upland (0), Water (1), Emergent (2), Forested (3), Scrub-Shrub (4), Other Wetland (5)
- **Tile size:** 256x256 pixels
- **NAIP epochs:** 2015, 2017

## Project Structure

```
.
├── research_paper/
│   ├── __init__.py              # Package init
│   ├── wetland.py               # Core module (all 5 phases, ~2000 lines)
│   ├── run_experiment.py        # CLI pipeline orchestrator
│   └── wetland_research.md      # Living research document
├── tests/
│   ├── test_wetland.py          # Phase 1 tests (96 tests)
│   ├── test_wetland_phase2.py   # Phase 2 tests (41 tests)
│   ├── test_wetland_phase3.py   # Phase 3 tests (30 tests)
│   ├── test_wetland_phase4.py   # Phase 4 tests (37 tests)
│   └── test_wetland_phase5.py   # Phase 5 tests (44 tests)
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

248 tests across all 5 phases, all passing:

| Phase | Tests | Coverage |
|-------|-------|----------|
| Phase 1: Data Pipeline | 96 | Data download, spectral indices, composites |
| Phase 2: Weak Labels | 41 | NWI reclassification, weak label generation |
| Phase 3: Training | 30 | Model training, loss functions, class weights |
| Phase 4: Inference | 37 | Prediction, dynamics, NWI comparison |
| Phase 5: Experiments | 44 | Config, result formatting, orchestration |

## References

1. Wu, Q., Lane, C.R., Li, X., Zhao, K., Zhou, Y., Clinton, N., DeVries, B., Golden, H.E. and Lang, M.W. (2019). Integrating LiDAR data and multi-temporal aerial imagery to map wetland inundation dynamics using Google Earth Engine. *Remote Sensing of Environment*, 228, pp.1-13.

2. Igwe, O.M., Lane, C.R., Golden, H.E. and Wu, Q. (2026). Cost-effective statewide wetland inventory update using weakly supervised deep learning: A case study in Minnesota, USA. *Remote Sensing Applications: Society and Environment*.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- Built on the [geoai](https://github.com/opengeos/geoai) open-source package by Dr. Qiusheng Wu
- Study area and methodology informed by Wu et al. (2019) RSE
- Weakly supervised approach adapted from Igwe et al. (2026)
