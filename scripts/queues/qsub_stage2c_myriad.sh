#!/bin/bash -l
# Mechanistic Stage 2C (REVERSE direction) — 15 reverse mirage task on Myriad.
#
# Tests forward-vs-reverse asymmetry hypothesis (笔记 §111.5b paper-grade evidence).
# Forward (SoM→P-SoM L11) flips output 93% match (笔记 §111 task 0).
# Reverse (P-SoM→SoM) expected to be null at all layers if asymmetric encoding holds.
#
# Submission:
#   ssh myriad
#   cd ~/Scratch/p79
#   qsub scripts/queues/qsub_stage2c_myriad.sh
#
# Can be submitted IN PARALLEL with qsub_stage2b_myriad.sh (different jobs, different GPUs).
#
# Output:
#   results/mechanistic/stage2c_reverse_curated_b1_cls_myriad/
#     ├── env_snapshot.json + run_manifest.json
#     ├── patching_continuation_results.json
#     ├── patching_continuation_curves.png
#     └── pilot_summary.md

# ============================================================================
# SGE directives
# ============================================================================

#$ -l h_rt=24:0:0          # 24h wallclock (15 task × ~50 min/task)
#$ -l mem=64G
#$ -l gpu=1
#$ -ac allow=L,U,V         # Allow L (40GB) + U/V (80GB) — let scheduler pick least busy
#$ -wd /home/ucab352/Scratch/p79
#$ -N stage2c_reverse_b1_cls
#$ -o /home/ucab352/Scratch/p79/logs/qsub_stage2c_b1_cls.$JOB_ID.out
#$ -e /home/ucab352/Scratch/p79/logs/qsub_stage2c_b1_cls.$JOB_ID.err
#$ -j n

mkdir -p /home/ucab352/Scratch/p79/logs

# ============================================================================
# Environment
# ============================================================================

set -euo pipefail

REPO_DIR="/home/ucab352/Scratch/p79"
cd "$REPO_DIR"

echo "[$(date '+%H:%M:%S')] Job $JOB_ID (reverse direction) start on $(hostname)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# Load Myriad pre-built PyTorch (auto-loads python/3.9.6 + cuda/11.8 + cudnn + gcc-libs)
module unload python python3 2>/dev/null || true
module load pytorch/2.1.0/gpu

# pip install --user packages live here
export PYTHONUSERBASE="$HOME/Scratch/python_user"
# PYTHONPATH prepend so pinned urllib3<2 wins over module's v2.3.0
export PYTHONPATH="$PYTHONUSERBASE/lib/python3.9/site-packages:${PYTHONPATH:-}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

echo "[$(date '+%H:%M:%S')] Repo HEAD: $(git rev-parse --short HEAD)"

# ============================================================================
# Sanity
# ============================================================================

n_reverse=$(python3 -c "import json; print(len(json.load(open('$REPO_DIR/results/mechanistic/archive_subset_b1_cls/manifest.json'))['reverse']))")
echo "[$(date '+%H:%M:%S')] Dataset: $n_reverse reverse mirage candidates"

# ============================================================================
# Run Stage 2C reverse direction
# ============================================================================

OUT_DIR="$REPO_DIR/results/mechanistic/stage2c_reverse_curated_b1_cls_myriad"
mkdir -p "$OUT_DIR"

echo "[$(date '+%H:%M:%S')] Launching Stage 2C reverse (15 task × all layers × 50 max_new_tokens)..."

# --reverse flag swaps source ↔ target: patches phantom_som's hidden into som run
# "Removing image content" probe — expected null at all layers per §111.5b
python3 scripts/mechanistic/run_stage2b_continuation_pilot.py \
    --reverse \
    --site classifieds \
    --n-tasks 15 \
    --step 2 \
    --max-new-tokens 50 \
    --source-mode som \
    --target-mode phantom_som \
    --output-dir "$OUT_DIR" \
    --archived-run-dir "$REPO_DIR/results/visualwebarena/phase1/B1_phantom_som_classifieds_20260428"

# ============================================================================
# Done
# ============================================================================

echo "[$(date '+%H:%M:%S')] Stage 2C reverse DONE"
echo "[$(date '+%H:%M:%S')] Output: $OUT_DIR"
ls -la "$OUT_DIR/"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv
