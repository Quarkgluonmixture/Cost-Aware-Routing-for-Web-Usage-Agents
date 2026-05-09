#!/bin/bash -l
# Cell D-r: reddit reverse × strong-tier × N=24 (2x2 selection-bias control).
# Parallels cls Cell D (stage2c_2x2_rev_strongtasks_myriad). Mirrors §117.5/§121.

#$ -l h_rt=36:0:0
#$ -l mem=64G
#$ -l gpu=1
#$ -wd /home/ucab352/Scratch/p79
#$ -N celldr_redrev_str
#$ -o /home/ucab352/Scratch/p79/logs/qsub_celldr_redrev_str.$JOB_ID.out
#$ -e /home/ucab352/Scratch/p79/logs/qsub_celldr_redrev_str.$JOB_ID.err
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

OUT_DIR="$REPO_DIR/results/mechanistic/stage2c_celldr_reddit_rev_strongtier_myriad"
mkdir -p "$OUT_DIR"

echo "[$(date '+%H:%M:%S')] Cell D-r: reddit reverse × strong-tier × N=24"

# --reverse (reverse direction); --tier strong forces strong-tier task subset (override default)
python3 scripts/mechanistic/run_stage2b_continuation_pilot.py \
    --site reddit \
    --n-tasks 24 \
    --step 2 \
    --max-new-tokens 50 \
    --reverse \
    --tier strong \
    --source-mode som \
    --target-mode phantom_som \
    --output-dir "$OUT_DIR" \
    --archived-run-dir "$REPO_DIR/results/mechanistic/archive_subset_b1_reddit"

echo "[$(date '+%H:%M:%S')] Cell D-r DONE"
ls -la "$OUT_DIR/"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv
