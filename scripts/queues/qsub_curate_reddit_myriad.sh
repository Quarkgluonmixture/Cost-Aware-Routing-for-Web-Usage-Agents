#!/bin/bash -l
# Cross-site replication step 1 — curate mirage tasks on B1 reddit SoM run.
#
# Reads B1_3mode_reddit_20260413/phase1_som_router_0/ archive, runs B1 model
# on each task to compute composite mirage score per (src, tgt) text pair,
# writes candidates.jsonl + candidates.md ranked by score.
#
# Output: results/mechanistic/curate_mirage_b1_reddit/{candidates.jsonl, .md, summary.json}
#
# Then DGX-side: run scripts/mechanistic/extract_archive_subset.py on the
# resulting candidates.jsonl to produce archive_subset_b1_reddit/ with
# manifest.json + per-task data → enables reddit Stage 2 patching cells F+G.

#$ -l h_rt=12:0:0
#$ -l mem=64G
#$ -l gpu=1
#$ -wd /home/ucab352/Scratch/p79
#$ -N curate_reddit
#$ -o /home/ucab352/Scratch/p79/logs/qsub_curate_reddit.$JOB_ID.out
#$ -e /home/ucab352/Scratch/p79/logs/qsub_curate_reddit.$JOB_ID.err
#$ -j n

mkdir -p /home/ucab352/Scratch/p79/logs

set -euo pipefail
REPO_DIR="/home/ucab352/Scratch/p79"
cd "$REPO_DIR"

echo "[$(date '+%H:%M:%S')] Job $JOB_ID start (curate reddit) on $(hostname)"
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

ARCHIVED_RUN="$REPO_DIR/results/visualwebarena/phase1/B1_3mode_reddit_20260413"
if [ ! -d "$ARCHIVED_RUN/phase1_som_router_0" ]; then
  echo "FATAL: B1 reddit SoM archive missing at $ARCHIVED_RUN/phase1_som_router_0/"
  exit 1
fi

OUT_DIR="$REPO_DIR/results/mechanistic/curate_mirage_b1_reddit"
mkdir -p "$OUT_DIR"

echo "[$(date '+%H:%M:%S')] Launching curate_mirage_tasks.py on reddit (210 tasks max)..."
python3 scripts/mechanistic/curate_mirage_tasks.py \
    --site reddit \
    --step 2 \
    --max-new-tokens 50 \
    --source-mode som \
    --target-mode phantom_som \
    --output-dir "$OUT_DIR" \
    --archived-run-dir "$ARCHIVED_RUN" \
    --artifacts-subdir phase1_som_router_0

echo "[$(date '+%H:%M:%S')] DONE → $OUT_DIR"
ls -la "$OUT_DIR/"
echo "[$(date '+%H:%M:%S')] Next: DGX-side run extract_archive_subset.py to produce archive_subset_b1_reddit/"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv
