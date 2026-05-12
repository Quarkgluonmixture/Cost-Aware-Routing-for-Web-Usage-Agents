#!/bin/bash -l
# Cell H-prompt-cls-RAND: cls forward × strong × P-SoM → P-text × N=24 ×
# random-injection negative control (Exp 5 axis-2 content-specificity check).
#
# Companion to qsub_stage3_cellhprompt_cls.sh (real-source axis-2 patching).
# Replaces cached source hidden states with Gaussian noise matched to source
# variance per layer. /stress G3 critique demands content-specific negative
# control: if random-injection produces same L11-L17 displacement as real
# source, the axis-2 effect is non-specific (any perturbation triggers it,
# the prompt-family signal is not the cause).
#
# Expected paper-grade outcome: random-injection L11-L17 displacement ≈ 0
# (overlap→tgt ~1.00 across all layers, no mid-layer dip), vs real-source
# at 0.20-0.30 displacement. Ratio = content-specificity index.

#$ -l h_rt=24:0:0
#$ -l mem=64G
#$ -l gpu=1
#$ -wd /home/ucab352/Scratch/p79
#$ -N cellhprm_cls_rand
#$ -o /home/ucab352/Scratch/p79/logs/qsub_cellhprm_cls_rand.$JOB_ID.out
#$ -e /home/ucab352/Scratch/p79/logs/qsub_cellhprm_cls_rand.$JOB_ID.err
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

OUT_DIR="$REPO_DIR/results/mechanistic/stage3_cellhprompt_cls_fwd_ptext_rand_myriad"
mkdir -p "$OUT_DIR"

echo "[$(date '+%H:%M:%S')] Cell H-prompt-cls-RAND: cls fwd × strong × P-SoM → P-text × random-inject × N=24"
echo "  /stress G3 content-specificity control"

python3 scripts/mechanistic/run_stage2b_continuation_pilot.py \
    --site classifieds \
    --n-tasks 24 \
    --step 2 \
    --max-new-tokens 50 \
    --source-mode phantom_som \
    --target-mode phantom_text \
    --random-inject \
    --random-seed 42 \
    --output-dir "$OUT_DIR" \
    --archived-run-dir "$REPO_DIR/results/mechanistic/archive_subset_b1_cls"

echo "[$(date '+%H:%M:%S')] Cell H-prompt-cls-RAND DONE"
ls -la "$OUT_DIR/"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv
