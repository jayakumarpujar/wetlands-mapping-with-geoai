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
from transformers import AutoModel, AutoConfig
cache = '$HF_CACHE'
model_id = 'ibm-nasa-geospatial/Prithvi-EO-2.0-300M'
print('Fetching config...')
AutoConfig.from_pretrained(model_id, trust_remote_code=True, cache_dir=cache)
print('Fetching model weights...')
AutoModel.from_pretrained(model_id, trust_remote_code=True, cache_dir=cache)
print('Done. Cache at:', cache)
"

echo ""
echo "Verify with:"
echo "  ls $HF_CACHE/models--ibm-nasa-geospatial--Prithvi-EO-2.0-300M/snapshots/"
