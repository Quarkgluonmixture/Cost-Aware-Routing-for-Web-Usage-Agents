#!/bin/bash -l
# Cell H-d-cls: cls forward × strong × source=som × target=dom × N=24.
# Closes the 2x2 mechanism additivity test:
#   target ∈ {phantom_som (Cell A), phantom_text (Ht_cls), phantom_prompt (Hp_cls), dom (THIS)}
# Pre-registered falsifiable prediction (from §124 partial-additivity computation):
#   if 2x2 mechanism additive at L11 → predicted Δ_to_target ≈ +10.74 (= Ht_cls + Hp_cls − Cell A
#   = 9.04 + 5.62 − 3.92). Observed Δ outside this prediction → prompt × text interaction at
#   mid-layer (i.e., P-SoM is more than the sum of its axes).

#$ -l h_rt=24:0:0
#$ -l mem=64G
#$ -l gpu=1
#$ -wd /home/ucab352/Scratch/p79
#$ -N cellhd_cls
#$ -o /home/ucab352/Scratch/p79/logs/qsub_cellhd_cls.$JOB_ID.out
#$ -e /home/ucab352/Scratch/p79/logs/qsub_cellhd_cls.$JOB_ID.err
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

OUT_DIR="$REPO_DIR/results/mechanistic/stage3_cellhd_cls_fwd_dom_myriad"
mkdir -p "$OUT_DIR"

echo "[$(date '+%H:%M:%S')] Cell H-d-cls: cls fwd × strong × som → dom × N=24"

python3 scripts/mechanistic/run_stage2b_continuation_pilot.py \
    --site classifieds \
    --n-tasks 24 \
    --step 2 \
    --max-new-tokens 50 \
    --source-mode som \
    --target-mode dom \
    --output-dir "$OUT_DIR" \
    --archived-run-dir "$REPO_DIR/results/mechanistic/archive_subset_b1_cls"

echo "[$(date '+%H:%M:%S')] Cell H-d-cls DONE"
ls -la "$OUT_DIR/"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv
