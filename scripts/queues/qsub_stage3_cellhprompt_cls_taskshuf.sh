#!/bin/bash -l
# Cell H-prompt-cls-TASKSHUF: cls forward × strong × P-SoM → P-text × N=24 ×
# task-shuffled content-specificity control (Exp 5 axis-2).
#
# Codex methodology audit 2026-05-12: Gaussian random-injection is WEAK
# specificity baseline because variance-matched noise breaks residual norm
# regardless of axis content. Task-shuffle = derangement permutation so that
# target task T_i uses SOURCE from task T_j != T_i with same source mode (=
# same residual-stream statistics) but different content.
#
# Expected paper-grade outcome: if axis-2 effect is content-specific,
# task-shuffled L11-L17 displacement << real-source (cellhprompt_cls)
# 0.20-0.30 magnitude. If task-shuffled ≈ real-source → axis-2 effect
# is non-content-specific (any source-mode hidden state injection
# produces the displacement).

#$ -l h_rt=24:0:0
#$ -l mem=64G
#$ -l gpu=1
#$ -wd /home/ucab352/Scratch/p79
#$ -N cellhprm_cls_tsh
#$ -o /home/ucab352/Scratch/p79/logs/qsub_cellhprm_cls_tsh.$JOB_ID.out
#$ -e /home/ucab352/Scratch/p79/logs/qsub_cellhprm_cls_tsh.$JOB_ID.err
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

if [ ! -f "$REPO_DIR/results/mechanistic/archive_subset_b1_cls/manifest.json" ]; then
  echo "FATAL: archive_subset_b1_cls/manifest.json missing"
  exit 1
fi

HF_REVISION="ebb281ec70b05090aa6165b016eac8ec08e71b17"
HF_SNAPSHOT_DIR="$HF_HOME/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/$HF_REVISION"
if [ ! -f "$HF_SNAPSHOT_DIR/config.json" ]; then
  echo "FATAL: HF model snapshot missing at $HF_SNAPSHOT_DIR"
  exit 1
fi

OUT_DIR="$REPO_DIR/results/mechanistic/stage3_cellhprompt_cls_fwd_ptext_taskshuf_myriad"
mkdir -p "$OUT_DIR"

echo "[$(date '+%H:%M:%S')] Cell H-prompt-cls-TASKSHUF: P-SoM (shuffled task) → P-text × N=24"
echo "  Codex audit Bug 6 / G3 content-specificity defuse"

python3 scripts/mechanistic/run_stage2b_continuation_pilot.py \
    --site classifieds \
    --n-tasks 24 \
    --step 2 \
    --max-new-tokens 50 \
    --source-mode phantom_som \
    --target-mode phantom_text \
    --task-shuffle \
    --task-shuffle-seed 42 \
    --output-dir "$OUT_DIR" \
    --archived-run-dir "$REPO_DIR/results/mechanistic/archive_subset_b1_cls"

echo "[$(date '+%H:%M:%S')] Cell H-prompt-cls-TASKSHUF DONE"
ls -la "$OUT_DIR/"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv
