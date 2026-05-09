#!/bin/bash -l
# Cross-site replication — Cell F: forward direction × 24 reddit-strong-tier tasks.
# Pairs with cls cell A (qsub_stage2b_myriad.sh) for direct cross-site comparison.

#$ -l h_rt=12:0:0
#$ -l mem=64G
#$ -l gpu=1
#$ -wd /home/ucab352/Scratch/p79
#$ -N cellf_fwd_reddit
#$ -o /home/ucab352/Scratch/p79/logs/qsub_cellf_fwd_reddit.$JOB_ID.out
#$ -e /home/ucab352/Scratch/p79/logs/qsub_cellf_fwd_reddit.$JOB_ID.err
#$ -j n

mkdir -p /home/ucab352/Scratch/p79/logs

set -euo pipefail
REPO_DIR="/home/ucab352/Scratch/p79"
cd "$REPO_DIR"

echo "[$(date '+%H:%M:%S')] Job $JOB_ID start (cell F: fwd × reddit-strong) on $(hostname)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

module unload gcc-libs python python3 2>/dev/null || true
module load pytorch/2.1.0/gpu

export PYTHONUSERBASE="$HOME/Scratch/python_user"
export PYTHONPATH="$PYTHONUSERBASE/lib/python3.9/site-packages:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

echo "[$(date '+%H:%M:%S')] Repo HEAD: $(git rev-parse --short HEAD)"

HF_REVISION="ebb281ec70b05090aa6165b016eac8ec08e71b17"
HF_SNAPSHOT_DIR="$HF_HOME/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/$HF_REVISION"
[ -f "$HF_SNAPSHOT_DIR/config.json" ] || { echo "FATAL: HF model snapshot missing"; exit 1; }

OUT_DIR="$REPO_DIR/results/mechanistic/stage2b_cellf_fwd_reddit_strong_myriad"
mkdir -p "$OUT_DIR"

echo "[$(date '+%H:%M:%S')] Launching FORWARD × reddit-strong-tier 24 task..."
python3 scripts/mechanistic/run_stage2b_continuation_pilot.py \
    --site reddit \
    --n-tasks 24 \
    --step 2 \
    --max-new-tokens 50 \
    --source-mode som \
    --target-mode phantom_som \
    --tier strong \
    --output-dir "$OUT_DIR" \
    --archived-run-dir "$REPO_DIR/results/mechanistic/archive_subset_b1_reddit"

echo "[$(date '+%H:%M:%S')] DONE → $OUT_DIR"
ls -la "$OUT_DIR/"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv
