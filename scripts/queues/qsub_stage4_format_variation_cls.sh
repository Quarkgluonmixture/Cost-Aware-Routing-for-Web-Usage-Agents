#!/bin/bash -l
# Stage 4 H1 test: text format variation across 8 industry-relevant indexed-list styles
# (SoM / Browser-Use @ / AppAgent / Tarsier / numbered / XML + 2 controls hash-id / plain-sentence).
# 24 cls strong-tier tasks × 2 steps × (8 variants + 2 baselines) = 480 forward passes.
#
# Output → results/mechanistic/stage4_format_variation_b1_cls/hidden_states.npz
# Used for: testing H1 (pretraining co-occurrence shortcut hypothesis) —
# do all marks-like variants trigger image-axis-peak shift to L17+, or only [SOM_MARKS]?

#$ -l h_rt=12:0:0
#$ -l mem=64G
#$ -l gpu=1
#$ -wd /home/ucab352/Scratch/p79
#$ -N stage4fv_cls
#$ -o /home/ucab352/Scratch/p79/logs/qsub_stage4fv_cls.$JOB_ID.out
#$ -e /home/ucab352/Scratch/p79/logs/qsub_stage4fv_cls.$JOB_ID.err
#$ -j n

mkdir -p /home/ucab352/Scratch/p79/logs

set -euo pipefail
REPO_DIR="/home/ucab352/Scratch/p79"
cd "$REPO_DIR"

echo "[$(date '+%H:%M:%S')] Job $JOB_ID start on $(hostname)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

module unload gcc-libs python python3 2>/dev/null || true
module load pytorch/2.1.0/gpu

export PYTHONUSERBASE="$HOME/Scratch/python_user"
export PYTHONPATH="$PYTHONUSERBASE/lib/python3.9/site-packages:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

if [ ! -d "$REPO_DIR/results/mechanistic/archive_subset_b1_cls" ]; then
  echo "FATAL: archive_subset_b1_cls missing"
  exit 1
fi

OUT_DIR="$REPO_DIR/results/mechanistic/stage4_format_variation_b1_cls"
mkdir -p "$OUT_DIR"

echo "[$(date '+%H:%M:%S')] Stage 4 H1: 24 tasks × 2 steps × 10 modes = 480 forward passes"

python3 scripts/mechanistic/run_stage4_format_variation_extract.py \
    --archived-run-dir "$REPO_DIR/results/mechanistic/archive_subset_b1_cls" \
    --output "$OUT_DIR/hidden_states.npz" \
    --tier strong \
    --n-tasks 24 \
    --steps 2,5

echo "[$(date '+%H:%M:%S')] DONE"
ls -la "$OUT_DIR/"
