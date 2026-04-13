#!/bin/bash
#SBATCH --job-name=wetmamba_bench
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/wetmamba_%j.out
#SBATCH --error=logs/wetmamba_%j.err

# ============================================================
# WetMamba Benchmark Training — SLURM Script
# ============================================================
#
# Usage:
#   # Single model (WetMamba)
#   sbatch research_paper/slurm_benchmark.sh
#
#   # All benchmarks (override MODEL)
#   sbatch --export=MODEL=all research_paper/slurm_benchmark.sh
#
#   # Ablation study
#   sbatch --export=MODEL=wetmamba,ABLATION=all research_paper/slurm_benchmark.sh
#
#   # Custom GPU (e.g., A100)
#   sbatch --gres=gpu:a100:1 --mem=128G research_paper/slurm_benchmark.sh
#
# Environment variables (override via --export or edit below):
#   MODEL       — model name or "all" (default: wetmamba)
#   ABLATION    — ablation variant or "all" (default: empty)
#   DATA_DIR    — path to tiles (default: /scratch/$USER/wetlands/tiles)
#   OUTPUT_DIR  — results directory (default: /scratch/$USER/wetlands/experiments)
#   EPOCHS      — training epochs (default: 100)
#   BATCH_SIZE  — batch size (default: 8)
#   LR          — learning rate (default: 1e-4)
# ============================================================

set -euo pipefail

# Defaults (override via SLURM --export)
MODEL="${MODEL:-wetmamba}"
ABLATION="${ABLATION:-}"
DATA_DIR="${DATA_DIR:-/scratch/$USER/wetlands/tiles}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/$USER/wetlands/experiments}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-1e-4}"

# --- Environment setup ---
# Adjust these to match your HPC module system
module load python/3.11 cuda/12.1 2>/dev/null || true
source "$HOME/venvs/wetlands/bin/activate" 2>/dev/null || true

# Ensure repo is on PYTHONPATH
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

mkdir -p logs

# --- GPU diagnostics ---
echo "=========================================="
echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $SLURM_NODELIST"
echo "GPUs      : $SLURM_GPUS_ON_NODE"
echo "CPUs      : $SLURM_CPUS_PER_TASK"
echo "Model     : $MODEL"
echo "Ablation  : ${ABLATION:-none}"
echo "Data      : $DATA_DIR"
echo "Output    : $OUTPUT_DIR"
echo "Epochs    : $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "=========================================="
nvidia-smi || true
echo "=========================================="

# --- Build command ---
CMD="python research_paper/run_benchmark_hpc.py \
    --data-dir $DATA_DIR \
    --output-dir $OUTPUT_DIR \
    --model $MODEL \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --num-workers $SLURM_CPUS_PER_TASK \
    --n-epochs-temporal 2"

# Add ablation flag if set
if [ -n "$ABLATION" ]; then
    CMD="$CMD --ablation $ABLATION"
fi

echo "Running: $CMD"
eval "$CMD"

echo "Job $SLURM_JOB_ID completed successfully."
