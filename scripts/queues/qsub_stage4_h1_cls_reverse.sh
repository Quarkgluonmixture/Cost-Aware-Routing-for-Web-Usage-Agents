#!/bin/bash -l
# P4: H1 test on cls reverse-tier (15 tasks) — selection-bias defense.
# Same 8 format variants as stage4fv_cls + dom + som baselines.
# Output → results/mechanistic/stage4_format_variation_b1_cls_reverse/hidden_states.npz

#$ -l h_rt=12:0:0
#$ -l mem=64G
#$ -l gpu=1
#$ -wd /home/ucab352/Scratch/p79
#$ -N stage4fv_clsrev
#$ -o /home/ucab352/Scratch/p79/logs/qsub_stage4fv_clsrev.$JOB_ID.out
#$ -e /home/ucab352/Scratch/p79/logs/qsub_stage4fv_clsrev.$JOB_ID.err
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

OUT_DIR="$REPO_DIR/results/mechanistic/stage4_format_variation_b1_cls_reverse"
mkdir -p "$OUT_DIR"

python3 scripts/mechanistic/run_stage4_format_variation_extract.py \
    --archived-run-dir "$REPO_DIR/results/mechanistic/archive_subset_b1_cls" \
    --output "$OUT_DIR/hidden_states.npz" \
    --tier reverse \
    --n-tasks 15 \
    --steps 2,5

echo "[$(date '+%H:%M:%S')] DONE"
ls -la "$OUT_DIR/"
