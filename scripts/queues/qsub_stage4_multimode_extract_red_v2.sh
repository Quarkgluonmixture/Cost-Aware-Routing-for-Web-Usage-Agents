#!/bin/bash -l
# Stage 4 Method 4.2 v2 fixed extract (reddit strong-tier).
# Cross-site replication of qsub_stage4_multimode_extract_cls_v2.sh.
#
# Reddit archive has 47 strong + 48 reverse — without --tier strong, lexicographic
# glob would heavily contaminate. v2 fixes (Bug 1 + Bug 2 + Bug 5 + provenance).

#$ -l h_rt=12:0:0
#$ -l mem=64G
#$ -l gpu=1
#$ -wd /home/ucab352/Scratch/p79
#$ -N stage4mm_red_v2
#$ -o /home/ucab352/Scratch/p79/logs/qsub_stage4mm_red_v2.$JOB_ID.out
#$ -e /home/ucab352/Scratch/p79/logs/qsub_stage4mm_red_v2.$JOB_ID.err
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

if [ ! -d "$REPO_DIR/results/mechanistic/archive_subset_b1_reddit" ]; then
  echo "FATAL: archive_subset_b1_reddit missing"
  exit 1
fi

HF_REVISION="ebb281ec70b05090aa6165b016eac8ec08e71b17"
HF_SNAPSHOT_DIR="$HF_HOME/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/$HF_REVISION"
if [ ! -f "$HF_SNAPSHOT_DIR/config.json" ]; then
  echo "FATAL: HF model snapshot missing at $HF_SNAPSHOT_DIR"
  exit 1
fi

OUT_DIR="$REPO_DIR/results/mechanistic/stage4_multimode_b1_reddit"
mkdir -p "$OUT_DIR"
OUT_NPZ="$OUT_DIR/hidden_states_v2_fixed.npz"

echo "[$(date '+%H:%M:%S')] Stage 4 v2 fixed reddit: --tier strong, 24 tasks × step 2 × 6 modes = 144 fwd passes"

python3 scripts/mechanistic/run_stage4_multimode_extract.py \
    --site reddit \
    --tier strong \
    --n-tasks 24 \
    --steps 2 \
    --archived-run-dir "$REPO_DIR/results/mechanistic/archive_subset_b1_reddit" \
    --output "$OUT_NPZ" \
    --model-revision "$HF_REVISION" \
    --modes dom phantom_text phantom_prompt phantom_som som vision

# Sentinel for auto_pull Phase 0 done-check
touch "$OUT_DIR/pilot_summary.md"

echo "[$(date '+%H:%M:%S')] Stage 4 reddit v2 fixed DONE"
ls -la "$OUT_DIR/"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv
