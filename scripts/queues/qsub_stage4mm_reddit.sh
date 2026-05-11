#!/bin/bash -l
# P5b: Method 4.2 multimode on reddit strong-tier 24 tasks — cross-site Mirage signature.
# Output → results/mechanistic/stage4_multimode_b1_reddit/hidden_states.npz

#$ -l h_rt=12:0:0
#$ -l mem=64G
#$ -l gpu=1
#$ -wd /home/ucab352/Scratch/p79
#$ -N stage4mm_red
#$ -o /home/ucab352/Scratch/p79/logs/qsub_stage4mm_red.$JOB_ID.out
#$ -e /home/ucab352/Scratch/p79/logs/qsub_stage4mm_red.$JOB_ID.err
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

OUT_DIR="$REPO_DIR/results/mechanistic/stage4_multimode_b1_reddit"
mkdir -p "$OUT_DIR"
OUT_NPZ="$OUT_DIR/hidden_states.npz"

python3 scripts/mechanistic/run_stage4_multimode_extract.py \
    --site reddit \
    --n-tasks 24 \
    --steps 2 5 \
    --archived-run-dir "$REPO_DIR/results/mechanistic/archive_subset_b1_reddit" \
    --output "$OUT_NPZ" \
    --modes dom phantom_text phantom_prompt phantom_som som vision

# Sentinel for Phase 0
echo "Stage 4 multimode reddit extraction complete" > "$OUT_DIR/pilot_summary.md"
echo "Modes: 6 (dom / phantom_text / phantom_prompt / phantom_som / som / vision)" >> "$OUT_DIR/pilot_summary.md"
echo "Tasks: 24 reddit strong-tier × 2 steps × 6 modes = 288 examples" >> "$OUT_DIR/pilot_summary.md"
ls -la "$OUT_DIR/" >> "$OUT_DIR/pilot_summary.md"

echo "[$(date '+%H:%M:%S')] DONE"
ls -la "$OUT_DIR/"
