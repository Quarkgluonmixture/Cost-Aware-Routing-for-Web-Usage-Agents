#!/bin/bash -l
# 2x2 cross-subset control — Cell C: FORWARD direction × 15 REVERSE-tier tasks.
#
# Pairs with the existing forward × 24 strong (cell A, qsub_stage2b_myriad.sh)
# and reverse × 15 reverse (cell B, qsub_stage2c_myriad.sh) and a separate
# qsub_2x2_rev_strongtasks_myriad.sh (cell D). Together cell A+B+C+D form the
# 2x2 design: direction (fwd/rev) × task subset (strong/reverse). This rules
# out task-selection-bias artifact in the apparent "reverse also disrupts"
# finding (see 笔记 §111.7+ analysis 2026-05-09).
#
# Submission:
#   qsub scripts/queues/qsub_2x2_fwd_revtasks_myriad.sh
#
# Output:
#   results/mechanistic/stage2b_2x2_fwd_revtasks_myriad/

#$ -l h_rt=12:0:0          # 12h wallclock (15 task × ~50 min/task A100 fast path)
#$ -l mem=64G
#$ -l gpu=1
#$ -wd /home/ucab352/Scratch/p79
#$ -N 2x2_fwd_revtasks
#$ -o /home/ucab352/Scratch/p79/logs/qsub_2x2_fwd_revtasks.$JOB_ID.out
#$ -e /home/ucab352/Scratch/p79/logs/qsub_2x2_fwd_revtasks.$JOB_ID.err
#$ -j n

mkdir -p /home/ucab352/Scratch/p79/logs

set -euo pipefail
REPO_DIR="/home/ucab352/Scratch/p79"
cd "$REPO_DIR"

echo "[$(date '+%H:%M:%S')] Job $JOB_ID start (fwd × reverse-tier 15-task) on $(hostname)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

module unload gcc-libs python python3 2>/dev/null || true
module load pytorch/2.1.0/gpu

export PYTHONUSERBASE="$HOME/Scratch/python_user"
export PYTHONPATH="$PYTHONUSERBASE/lib/python3.9/site-packages:${PYTHONPATH:-}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

echo "[$(date '+%H:%M:%S')] Repo HEAD: $(git rev-parse --short HEAD)"

# Fail-fast HF cache check (B-81e pre-flight)
HF_REVISION="ebb281ec70b05090aa6165b016eac8ec08e71b17"
HF_SNAPSHOT_DIR="$HF_HOME/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/$HF_REVISION"
if [ ! -f "$HF_SNAPSHOT_DIR/config.json" ]; then
  echo "FATAL: HF model snapshot missing at $HF_SNAPSHOT_DIR/config.json"
  exit 1
fi

n_reverse=$(python3 -c "import json; print(len(json.load(open('$REPO_DIR/results/mechanistic/archive_subset_b1_cls/manifest.json'))['reverse']))")
echo "[$(date '+%H:%M:%S')] Dataset: $n_reverse reverse-tier candidates (forward direction will run on these)"

OUT_DIR="$REPO_DIR/results/mechanistic/stage2b_2x2_fwd_revtasks_myriad"
mkdir -p "$OUT_DIR"

echo "[$(date '+%H:%M:%S')] Launching FORWARD × REVERSE-tier 15 task..."
python3 scripts/mechanistic/run_stage2b_continuation_pilot.py \
    --site classifieds \
    --n-tasks 15 \
    --step 2 \
    --max-new-tokens 50 \
    --source-mode som \
    --target-mode phantom_som \
    --tier reverse \
    --output-dir "$OUT_DIR" \
    --archived-run-dir "$REPO_DIR/results/mechanistic/archive_subset_b1_cls"

echo "[$(date '+%H:%M:%S')] DONE → $OUT_DIR"
ls -la "$OUT_DIR/"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv
