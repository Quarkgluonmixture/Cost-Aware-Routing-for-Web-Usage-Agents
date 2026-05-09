#!/bin/bash -l
# 2x2 cross-subset control — Cell D: REVERSE direction × 24 STRONG-tier tasks.
#
# Pairs with cell A (qsub_stage2b_myriad.sh, fwd × strong 24), cell B
# (qsub_stage2c_myriad.sh, rev × reverse 15), cell C (qsub_2x2_fwd_revtasks_myriad.sh,
# fwd × reverse 15). Together: 2x2 direction × task-subset design to rule out
# selection-bias artifact in apparent "reverse also disrupts" finding (笔记 §111.7+).
#
# Submission:
#   qsub scripts/queues/qsub_2x2_rev_strongtasks_myriad.sh
#
# Output:
#   results/mechanistic/stage2c_2x2_rev_strongtasks_myriad/

#$ -l h_rt=18:0:0          # 18h wallclock (24 task × ~50 min/task A100 fast path)
#$ -l mem=64G
#$ -l gpu=1
#$ -wd /home/ucab352/Scratch/p79
#$ -N celld_rev_strongtasks
#$ -o /home/ucab352/Scratch/p79/logs/qsub_celld_rev_strongtasks.$JOB_ID.out
#$ -e /home/ucab352/Scratch/p79/logs/qsub_celld_rev_strongtasks.$JOB_ID.err
#$ -j n

mkdir -p /home/ucab352/Scratch/p79/logs

set -euo pipefail
REPO_DIR="/home/ucab352/Scratch/p79"
cd "$REPO_DIR"

echo "[$(date '+%H:%M:%S')] Job $JOB_ID start (rev × strong-tier 24-task) on $(hostname)"
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

n_strong=$(python3 -c "import json; print(len(json.load(open('$REPO_DIR/results/mechanistic/archive_subset_b1_cls/manifest.json'))['strong']))")
echo "[$(date '+%H:%M:%S')] Dataset: $n_strong strong-tier candidates (reverse direction will run on these)"

OUT_DIR="$REPO_DIR/results/mechanistic/stage2c_2x2_rev_strongtasks_myriad"
mkdir -p "$OUT_DIR"

echo "[$(date '+%H:%M:%S')] Launching REVERSE × STRONG-tier 24 task..."
python3 scripts/mechanistic/run_stage2b_continuation_pilot.py \
    --reverse \
    --site classifieds \
    --n-tasks 24 \
    --step 2 \
    --max-new-tokens 50 \
    --source-mode som \
    --target-mode phantom_som \
    --tier strong \
    --output-dir "$OUT_DIR" \
    --archived-run-dir "$REPO_DIR/results/mechanistic/archive_subset_b1_cls"

echo "[$(date '+%H:%M:%S')] DONE → $OUT_DIR"
ls -la "$OUT_DIR/"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv
