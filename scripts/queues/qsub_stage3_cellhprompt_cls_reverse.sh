#!/bin/bash -l
# Cell H-prompt-cls-REVERSE: cls reverse direction × strong × P-text → P-SoM ×
# N=24 (asymmetry test of axis-2 cellhprompt patching).
#
# Original cellhprompt_cls (359511) tested forward: P-SoM (clean source, with
# image) → P-text (mirage target, no image). Reverse swaps direction so that
# source is the mirage (no image) and target is the clean (with image) — tests
# whether axis-2 prompt-only patching is symmetric, i.e. does information
# transfer equally in both directions across the L17 site?
#
# Expected paper-grade outcome: if axis-2 is a symmetric pathway, reverse
# direction shows similar L11-L17 displacement magnitude. If asymmetric
# (forward >> reverse), this constrains the mechanism story (e.g. clean image
# context "drowns out" mirage prompt vs. mirage prompt being readily
# overridden by clean source).

#$ -l h_rt=24:0:0
#$ -l mem=64G
#$ -l gpu=1
#$ -wd /home/ucab352/Scratch/p79
#$ -N cellhprm_cls_rev
#$ -o /home/ucab352/Scratch/p79/logs/qsub_cellhprm_cls_rev.$JOB_ID.out
#$ -e /home/ucab352/Scratch/p79/logs/qsub_cellhprm_cls_rev.$JOB_ID.err
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

OUT_DIR="$REPO_DIR/results/mechanistic/stage3_cellhprompt_cls_rev_psom_myriad"
mkdir -p "$OUT_DIR"

echo "[$(date '+%H:%M:%S')] Cell H-prompt-cls-REVERSE: P-text (no image) → P-SoM (clean) × N=24"
echo "  Axis-2 patching asymmetry test"

python3 scripts/mechanistic/run_stage2b_continuation_pilot.py \
    --site classifieds \
    --n-tasks 24 \
    --step 2 \
    --max-new-tokens 50 \
    --source-mode phantom_som \
    --target-mode phantom_text \
    --reverse \
    --output-dir "$OUT_DIR" \
    --archived-run-dir "$REPO_DIR/results/mechanistic/archive_subset_b1_cls"

echo "[$(date '+%H:%M:%S')] Cell H-prompt-cls-REVERSE DONE"
ls -la "$OUT_DIR/"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv
