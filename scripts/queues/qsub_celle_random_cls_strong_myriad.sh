#!/bin/bash -l
# 2x2+ random-injection control — Cell E: forward direction × 24 strong-tier
# tasks × Gaussian-noise source hidden (mean+std matched per-layer).
#
# Paper §5 reviewer Q: "Is L17 mid-layer disruption from source-content
# specificity, or any non-zero injection?" Expected null at all layers if
# mechanism is source-content-specific (paper claim valid). If random-inject
# also produces L11-L17 dip → mechanism non-specific (claim weakened).
#
# Pairs with cell A (qsub_stage2b_myriad.sh, fwd × strong, real source) for
# direct A vs E comparison: same task subset, same direction, only difference
# is hidden state content (real vs Gaussian).

#$ -l h_rt=12:0:0
#$ -l mem=64G
#$ -l gpu=1
#$ -wd /home/ucab352/Scratch/p79
#$ -N celle_random_cls
#$ -o /home/ucab352/Scratch/p79/logs/qsub_celle_random_cls.$JOB_ID.out
#$ -e /home/ucab352/Scratch/p79/logs/qsub_celle_random_cls.$JOB_ID.err
#$ -j n

mkdir -p /home/ucab352/Scratch/p79/logs

set -euo pipefail
REPO_DIR="/home/ucab352/Scratch/p79"
cd "$REPO_DIR"

echo "[$(date '+%H:%M:%S')] Job $JOB_ID start (random-inject cell E) on $(hostname)"
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
if [ ! -f "$HF_SNAPSHOT_DIR/config.json" ]; then
  echo "FATAL: HF model snapshot missing at $HF_SNAPSHOT_DIR/config.json"
  exit 1
fi

OUT_DIR="$REPO_DIR/results/mechanistic/stage2b_celle_random_cls_strong_myriad"
mkdir -p "$OUT_DIR"

echo "[$(date '+%H:%M:%S')] Launching FORWARD × strong-tier 24 task with RANDOM source hidden..."
python3 scripts/mechanistic/run_stage2b_continuation_pilot.py \
    --site classifieds \
    --n-tasks 24 \
    --step 2 \
    --max-new-tokens 50 \
    --source-mode som \
    --target-mode phantom_som \
    --tier strong \
    --random-inject \
    --output-dir "$OUT_DIR" \
    --archived-run-dir "$REPO_DIR/results/mechanistic/archive_subset_b1_cls"

echo "[$(date '+%H:%M:%S')] DONE → $OUT_DIR"
ls -la "$OUT_DIR/"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv
