#!/bin/bash
# Run this on the HPC LOGIN node (has internet) BEFORE submitting any sweep job.
# Caches Prithvi-EO-2.0-300M weights so GPU compute nodes (offline) can load them.
#
# Usage:
#   bash slurm/cache_prithvi.sh

HF_CACHE=/home/john_lab/shared/Jay/hf_cache

mkdir -p $HF_CACHE

echo "Downloading Prithvi-EO-2.0-300M → $HF_CACHE"
python -c "
from huggingface_hub import snapshot_download
cache = '$HF_CACHE'
model_id = 'ibm-nasa-geospatial/Prithvi-EO-2.0-300M'
print('Downloading all Prithvi files...')
path = snapshot_download(repo_id=model_id, cache_dir=cache)
print('Done. Snapshot at:', path)
"

echo ""
echo "Verify with:"
echo "  ls $HF_CACHE/models--ibm-nasa-geospatial--Prithvi-EO-2.0-300M/snapshots/"
