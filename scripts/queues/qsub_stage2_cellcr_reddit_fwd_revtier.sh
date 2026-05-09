#!/bin/bash -l
# Cell C-r: reddit forward × reverse-tier × N=15 (2x2 selection-bias control).
# Parallels cls Cell C (stage2b_2x2_fwd_revtasks_myriad). Mirrors §117.5/§121.

#$ -l h_rt=24:0:0
#$ -l mem=64G
#$ -l gpu=1
#$ -wd /home/ucab352/Scratch/p79
#$ -N cellcr_redfwd_rev
#$ -o /home/ucab352/Scratch/p79/logs/qsub_cellcr_redfwd_rev.$JOB_ID.out
#$ -e /home/ucab352/Scratch/p79/logs/qsub_cellcr_redfwd_rev.$JOB_ID.err
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

if [ ! -f "$REPO_DIR/results/mechanistic/archive_subset_b1_reddit/manifest.json" ]; then
  echo "FATAL: archive_subset_b1_reddit/manifest.json missing"
  exit 1
fi

HF_REVISION="ebb281ec70b05090aa6165b016eac8ec08e71b17"
HF_SNAPSHOT_DIR="$HF_HOME/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/$HF_REVISION"
if [ ! -f "$HF_SNAPSHOT_DIR/config.json" ]; then
  echo "FATAL: HF model snapshot missing at $HF_SNAPSHOT_DIR"
  exit 1
fi

OUT_DIR="$REPO_DIR/results/mechanistic/stage2b_cellcr_reddit_fwd_revtier_myriad"
mkdir -p "$OUT_DIR"

echo "[$(date '+%H:%M:%S')] Cell C-r: reddit forward × reverse-tier × N=15"

# NO --reverse (forward direction); --tier reverse forces reverse-tier task subset
python3 scripts/mechanistic/run_stage2b_continuation_pilot.py \
    --site reddit \
    --n-tasks 15 \
    --step 2 \
    --max-new-tokens 50 \
    --tier reverse \
    --source-mode som \
    --target-mode phantom_som \
    --output-dir "$OUT_DIR" \
    --archived-run-dir "$REPO_DIR/results/mechanistic/archive_subset_b1_reddit"

echo "[$(date '+%H:%M:%S')] Cell C-r DONE"
ls -la "$OUT_DIR/"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv
