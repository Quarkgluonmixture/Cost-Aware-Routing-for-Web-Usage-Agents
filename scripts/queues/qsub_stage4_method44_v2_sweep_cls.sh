#!/bin/bash -l
# Stage 4 Method 4.4 v2 train/eval split sweep (cls strong-tier).
#
# Pipeline audit P0-4 fix (2026-05-13): direction now fit on 16 train tasks
# (split_seed=20260513), evaluated on 8 held-out tasks. Reviewer-3 demands
# this — old script fit + evaluated on same 24 tasks → in-sample inflation.
# Output JSON has per_task_eval (paper-grade headline) + per_task_in_sample
# (training cohort for reviewer comparison only). MD compares generalization
# gap at hero (L*, α*) cell.
#
# Compute estimate: 24 tasks × 2 steps × 6 layers × 5 α + 2 baselines per cell
#   = 24 × 2 × 32 = 1536 generations × 15 tokens ≈ 23K tokens.
# Myriad A100/V100 ~30-50 tok/s → ~10-15 min compute + 3 min model load = ~20 min wall.
# h_rt 4h budget covers slow queue / cold cache.
#
# Why Myriad not DGX: DGX seonglae 96% GPU contention 2026-05-13, 10min smoke
# timed out with 0 progress. Myriad qsub guarantees dedicated GPU.

#$ -l h_rt=4:0:0
#$ -l mem=64G
#$ -l gpu=1
#$ -wd /home/ucab352/Scratch/p79
#$ -N stage4mm44_cls_v2split
#$ -o /home/ucab352/Scratch/p79/logs/qsub_stage4mm44_cls_v2split.$JOB_ID.out
#$ -e /home/ucab352/Scratch/p79/logs/qsub_stage4mm44_cls_v2split.$JOB_ID.err
#$ -j n

mkdir -p /home/ucab352/Scratch/p79/logs

set -euo pipefail
REPO_DIR="/home/ucab352/Scratch/p79"
cd "$REPO_DIR"

echo "[$(date '+%H:%M:%S')] Job $JOB_ID start on $(hostname)"
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv

module unload gcc-libs python python3 2>/dev/null || true
module load pytorch/2.1.0/gpu

export PYTHONUSERBASE="$HOME/Scratch/python_user"
export PYTHONPATH="$PYTHONUSERBASE/lib/python3.9/site-packages:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

echo "[$(date '+%H:%M:%S')] Repo HEAD: $(git rev-parse --short HEAD)"

# Verify inputs present
NPZ="$REPO_DIR/results/mechanistic/stage4_multimode_b1_cls/hidden_states_v2_fixed.npz"
ARCHIVE="$REPO_DIR/results/mechanistic/archive_subset_b1_cls"
if [ ! -f "$NPZ" ]; then
  echo "FATAL: NPZ missing at $NPZ"
  exit 1
fi
if [ ! -d "$ARCHIVE" ]; then
  echo "FATAL: archive_subset_b1_cls missing at $ARCHIVE"
  exit 1
fi
if [ ! -f "$ARCHIVE/manifest.json" ]; then
  echo "FATAL: manifest.json missing"
  exit 1
fi

HF_REVISION="ebb281ec70b05090aa6165b016eac8ec08e71b17"
HF_SNAPSHOT_DIR="$HF_HOME/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/$HF_REVISION"
if [ ! -f "$HF_SNAPSHOT_DIR/config.json" ]; then
  echo "FATAL: HF model snapshot missing at $HF_SNAPSHOT_DIR"
  exit 1
fi

echo "[$(date '+%H:%M:%S')] Method 4.4 v2 sweep: --limit 24 (16 train / 8 eval @ seed 20260513)"
echo "[$(date '+%H:%M:%S')]   layers=11,17,23,29,33,34   alphas=1,2,5,10,20   max_new_tokens=15"
echo "[$(date '+%H:%M:%S')]   also_report_in_sample=True (two-column eval vs in-sample)"

python3 scripts/mechanistic/run_stage4_method44_v2_sweep.py \
    --limit 24 \
    --n-train-tasks 16 \
    --split-seed 20260513

# Sentinel for auto_pull Phase 0 done-check
OUT_DIR="$REPO_DIR/results/mechanistic/stage4_multimode_b1_cls"
touch "$OUT_DIR/method44_v2_sweep_DONE.marker"

echo "[$(date '+%H:%M:%S')] Stage 4 cls Method 4.4 v2 split sweep DONE"
ls -la "$OUT_DIR/method44_v2_sweep"* 2>/dev/null || true
ls -la "$REPO_DIR/docs/checkpoints/stage4_method44_v2_results.md" 2>/dev/null || true
nvidia-smi --query-gpu=memory.used,memory.free --format=csv
