#!/bin/bash
#SBATCH --job-name=full_sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#SBATCH --output=/home/john_lab/shared/Jay/logs/full_sweep_%j.log
#SBATCH --error=/home/john_lab/shared/Jay/logs/full_sweep_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

# Full sweep: 10 CNN runs + 5 WetMamba runs = 15 total (~80h worst case).
# All runs are resumable — already-trained models are skipped automatically.

REPO=/home/john_lab/shared/Jay/wetlands-mapping-with-geoai
DATA=/home/john_lab/shared/Jay/wetlands_data
TEST=/home/john_lab/shared/Jay/wetlands_testdata
SWEEP_OUT=/home/john_lab/shared/Jay/wetlands_data/sweep
HF_CACHE=/home/john_lab/shared/Jay/hf_cache

export HF_HOME=$HF_CACHE
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

mkdir -p /home/john_lab/shared/Jay/logs

cd $REPO
git pull --ff-only

if [ ! -d "$HF_CACHE/models--ibm-nasa-geospatial--Prithvi-EO-2.0-300M" ]; then
    echo "ERROR: Prithvi cache not found at $HF_CACHE — pre-cache on login node first"
    exit 1
fi

echo "================================================================"
echo "Full Sweep — Job $SLURM_JOB_ID"
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "================================================================"

python research_paper/run_experiment_sweep.py \
    --train-tiles $DATA/tiles \
    --test-tiles  $TEST/tiles \
    --output-root $SWEEP_OUT \
    --num-workers 4 \
    -v

echo "Sweep complete — exit code $?"
