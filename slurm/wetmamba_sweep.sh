#!/bin/bash
#SBATCH --job-name=wetmamba_sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=/home/john_lab/shared/Jay/logs/wetmamba_sweep_%j.log
#SBATCH --error=/home/john_lab/shared/Jay/logs/wetmamba_sweep_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO=/home/john_lab/shared/Jay/wetlands-mapping-with-geoai
DATA=/home/john_lab/shared/Jay/wetlands_data
TEST=/home/john_lab/shared/Jay/wetlands_testdata
SWEEP_OUT=/home/john_lab/shared/Jay/wetlands_data/sweep
HF_CACHE=/home/john_lab/shared/Jay/hf_cache

# ── Environment ───────────────────────────────────────────────────────────────
export HF_HOME=$HF_CACHE
export TRANSFORMERS_OFFLINE=1      # use pre-cached Prithvi weights
export PYTHONUNBUFFERED=1

mkdir -p /home/john_lab/shared/Jay/logs

# ── Pull latest code ──────────────────────────────────────────────────────────
cd $REPO
git pull --ff-only

# ── Verify Prithvi cache exists ───────────────────────────────────────────────
if [ ! -d "$HF_CACHE/models--ibm-nasa-geospatial--Prithvi-EO-2.0-300M" ]; then
    echo "ERROR: Prithvi cache not found at $HF_CACHE"
    echo "Run this on the LOGIN node first (has internet):"
    echo "  python -c \"from transformers import AutoModel; AutoModel.from_pretrained('ibm-nasa-geospatial/Prithvi-EO-2.0-300M', trust_remote_code=True, cache_dir='$HF_CACHE')\""
    exit 1
fi

echo "================================================================"
echo "WetMamba Sweep — Job $SLURM_JOB_ID"
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Repo: $REPO"
echo "Data: $DATA"
echo "Output: $SWEEP_OUT"
echo "================================================================"

# ── Run sweep (WetMamba only — CNN runs already done, auto-skipped) ───────────
python research_paper/run_experiment_sweep.py \
    --train-tiles $DATA/tiles \
    --test-tiles  $TEST/tiles \
    --output-root $SWEEP_OUT \
    --wetmamba-only \
    --num-workers 4 \
    -v

echo "================================================================"
echo "Sweep finished — exit code $?"
echo "Results: $SWEEP_OUT/sweep_summary.json"
echo "================================================================"
