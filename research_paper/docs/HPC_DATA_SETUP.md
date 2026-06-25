# HPC Data Setup — Manual Download & Upload

When HPC has no internet, download NAIP, DEM, and NWI locally, then upload.

## Study Areas

| Split | Region | BBox (WGS84) | Size |
|-------|--------|--------------|------|
| **Train / Val** | Central PPR ND | `-100.55, 46.65, -99.15, 47.60` | ~155 × 106 km |
| **Test** | Western PPR ND | `-101.9, 46.65, -100.55, 47.60` | ~145 × 106 km |

- NAIP years: 2015, 2017 (both splits)
- **No spatial overlap** — test east edge (`-100.55`) == train west edge (`-100.55`)
- Same biome/ecoregion → valid geographic generalization test

## Target Layout on HPC

### Training data (`$TRAIN_ROOT`)

```
$TRAIN_ROOT/
├── naip/
│   ├── 2015/*.tif   (~368 tiles, ~25 GB)
│   └── 2017/*.tif   (~368 tiles, ~25 GB)
├── dem/
│   └── dem_ppr_1m.tif   (~5 GB, single merged raster)
└── nwi/
    └── ND_Wetlands.gpkg (~3 GB, statewide — shared with test)
```

Total ~55 GB.

### Test data (`$TEST_ROOT`)

```
$TEST_ROOT/
├── naip/
│   ├── 2015/*.tif   (~340 tiles, ~23 GB)
│   └── 2017/*.tif   (~340 tiles, ~23 GB)
├── dem/
│   └── dem_ppr_test_10m.tif   (~1 GB, 10m resolution)
└── nwi/
    └── ND_Wetlands.gpkg (symlink or copy from $TRAIN_ROOT/nwi/)
```

Total ~47 GB.

## Download Scripts (local)

### Training — `~/Downloads/wetlands_dl/`

- `download_naip.py` — bbox `(-100.55, 46.65, -99.15, 47.60)`
- `download_dem.py` — 3DEP 1m for train bbox
- `download_nwi.sh` — USFWS ND statewide geopackage

### Test — `~/Downloads/wetlands_dl_test/`

- `download_naip.py` — bbox `(-101.9, 46.65, -100.55, 47.60)`, includes no-overlap assertion
- `download_dem.py` — 3DEP 10m for test bbox (1m times out on full extent)
- `download_nwi.sh` — same ND statewide file; skips if already downloaded

### Dependencies

```bash
pip install planetary-computer pystac-client py3dep tqdm requests
```

### NAIP

Resumable — skips existing `.tif`. ~42 s/tile.
- Train: ~368 tiles × 2 years ≈ 9 hours
- Test: ~340 tiles × 2 years ≈ 8 hours

### DEM

py3dep fetches merged raster. 30-60 min per bbox.
Test uses 10m (not 1m) to avoid 3DEP API timeout on full western extent.

### NWI

Working URL (as of 2026-04):
```
https://documentst.ecosphere.fws.gov/wetlands/data/State-Downloads/ND_geopackage_wetlands.zip
```

**Note**: old `www.fws.gov/wetlands/downloads/State/` returns 404. Use `documentst.ecosphere.fws.gov`.

## Upload to HPC

### Training data

```bash
rsync -avP ~/Downloads/wetlands_dl/naip/ user@hpc:$TRAIN_ROOT/naip/
rsync -avP ~/Downloads/wetlands_dl/dem/  user@hpc:$TRAIN_ROOT/dem/
rsync -avP ~/Downloads/wetlands_dl/nwi/*.gpkg user@hpc:$TRAIN_ROOT/nwi/
```

### Test data

```bash
rsync -avP ~/Downloads/wetlands_dl_test/naip/ user@hpc:$TEST_ROOT/naip/
rsync -avP ~/Downloads/wetlands_dl_test/dem/  user@hpc:$TEST_ROOT/dem/
# NWI is statewide — reuse training copy
ln -s $TRAIN_ROOT/nwi/ND_Wetlands.gpkg $TEST_ROOT/nwi/ND_Wetlands.gpkg
```

## Verify on HPC

### Training

```bash
ls $TRAIN_ROOT/naip/2015/*.tif | wc -l   # ~368
ls $TRAIN_ROOT/naip/2017/*.tif | wc -l   # ~368
ls -lh $TRAIN_ROOT/nwi/*.gpkg             # 500MB - 3GB
du -sh $TRAIN_ROOT                        # ~55 GB
```

### Test

```bash
ls $TEST_ROOT/naip/2015/*.tif | wc -l    # ~340
ls $TEST_ROOT/naip/2017/*.tif | wc -l    # ~340
ls -lh $TEST_ROOT/dem/*.tif               # ~1 GB
du -sh $TEST_ROOT                         # ~47 GB
```

## Run Pipeline (pre-uploaded mode)

Pipeline auto-detects NAIP if `naip/{2015,2017}/*.tif` populated (see [run_experiment.py:77-89](../run_experiment.py#L77-L89) resume path). DEM/NWI require explicit CLI flags.

### Training run

```bash
python research_paper/run_ppr_hpc.py \
  --output-root $TRAIN_ROOT \
  --dem-tiles $TRAIN_ROOT/dem/*.tif \
  --nwi-path $TRAIN_ROOT/nwi/ND_Wetlands.gpkg \
  --loss-function crossentropy --no-class-weights \
  --learning-rate 1e-4 --oversample-threshold 1.01 \
  --min-wetland-fraction 0.05 --tile-stride 128 \
  --num-epochs 50 --batch-size 16 \
  --collapse-miou-threshold 0.35 --collapse-check-epoch 10
```

### Test tile generation (no training)

Generate test tiles then evaluate the trained model against NWI reference:

```bash
# Step 1 — generate composite + weak label tiles for test region
python research_paper/run_ppr_hpc.py \
  --output-root $TEST_ROOT \
  --dem-tiles $TEST_ROOT/dem/*.tif \
  --nwi-path $TEST_ROOT/nwi/ND_Wetlands.gpkg \
  --min-wetland-fraction 0.05 --tile-stride 128 \
  --num-epochs 0

# Step 2 — run inference + score
python - <<'EOF'
import sys
sys.path.insert(0, ".")
from research_paper.wetland import predict_wetlands, compare_with_nwi

predict_wetlands(
    model_path="research_paper/best_model.pth",
    composite_path="$TEST_ROOT/composites/composite.tif",
    output_path="$TEST_ROOT/results/prediction.tif",
    architecture="unetplusplus",
    encoder_name="resnet50",
    in_channels=10,
    num_classes=3,
)

metrics = compare_with_nwi(
    prediction_path="$TEST_ROOT/results/prediction.tif",
    reference_path="$TEST_ROOT/composites/weak_labels.tif",
)
import json
print(json.dumps(metrics, indent=2))
EOF
```

## Critical Flags

| Flag | Why |
|------|-----|
| `--tile-stride 128` | 50% overlap → 4× more tiles (NAIP coverage small relative to bbox) |
| `--min-wetland-fraction 0.05` | Drops mostly-upland tiles |
| `--oversample-threshold 1.01` | Disables oversample duplication |
| `--collapse-miou-threshold 0.35` | Auto-retry with focal loss if Val IoU < 0.35 by epoch 10 |
| `--dem-tiles` | Maps to `pre_downloaded_dem_tiles` override → skips 3DEP API |
| `--nwi-path` | Maps to `pre_downloaded_nwi` override → skips USFWS API |

## Band Normalization (critical)

Composite writer at [run_experiment.py:289-360](../run_experiment.py#L289-L360) normalizes all 10 bands to [0,1]:

| Band | Source | Normalization |
|------|--------|--------------|
| 1-4  | NAIP RGBNIR (0-255) | `/ 255` |
| 5-6  | NDVI/NDWI 2015 (-1..1) | `(x + 1) / 2` |
| 7-8  | NDVI/NDWI 2017 (-1..1) | `(x + 1) / 2` |
| 9    | DEM (~500 m elev) | `(x - min) / range` |
| 10   | Depression depth (0-5 m) | `/ 5` |

**Why**: geoai dataset loader applies `if image.max() > 1.0: image /= 255`. Without per-band pre-normalization, DEM at 500 m triggered the /255 shortcut, crushing NDVI/NDWI/depth to ~0 → model trained on essentially NAIP-only → collapse to trivial upland prediction.

## Known Issues & Fixes (commit log)

| Commit | Issue | Fix |
|--------|-------|-----|
| `57fba3b` | Training bands crushed to 0 | Per-band norm in run_experiment.py composite writer |
| `111cd86` | Dead code path in wetland.py | Per-band norm in `create_wetland_composite` (fallback) |
| `32ab8d1` | Tiles exported with empty NAIP | Skip tiles where `naip_bands.max() == 0` |
| `0fc2576` | Too few tiles (62) | `--tile-stride` CLI for dense overlap |
| `14580fc` | NAIP cap = 10/year too small | Raised to 500/year |

## Pre-flight Troubleshooting

**Symptom**: Val mean IoU plateaus at a low value (num_classes=3 → a collapse-to-upland
prediction caps mean IoU near upland_IoU/3, the trivial baseline)
- Check normalization applied: inspect composite band stats, all should be in [0,1]
- Check NAIP coverage: `gdalinfo` on composite, verify band 1 nonzero across extent
- Check tile count: expect 3000-8000 post-filter with `--tile-stride 128`

**Symptom**: Tile count too low (<500)
- NAIP coverage gap → verify `naip/{year}/` has ~300+ `.tif`
- Raise `--tile-stride` divisor or lower `--min-wetland-fraction`

**Symptom**: Out of memory
- Lower `--batch-size` (V100-32GB fits 16 at 256×256)
- Lower `--tile-size` to 192
