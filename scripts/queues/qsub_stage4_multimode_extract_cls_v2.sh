#!/bin/bash -l
# Stage 4 Method 4.2 v2 fixed extract (cls strong-tier).
#
# v2 fixes from /codex-stress methodology audit 2026-05-12:
#   Bug 2 — build_som_marks now calls production p79.experiment.som._extract_text_marks
#           (previously lossy regex `^\[\d+\]\s+\w+` dropped labels + 71/72 marks on cls)
#   Bug 1 — --tier strong filter via manifest.json (previously lexicographic glob
#           contaminated 24-task selection with reverse-tier tasks: cls archive has
#           24 strong + 15 reverse mixed)
#   Bug 5 — --model-revision pinned to match Stage 2B / agent extraction
#   Repro — provenance.json sidecar with command, git SHA, formatter hash, task IDs
#
# Output: hidden_states_v2_fixed.npz (NOT hidden_states.npz; preserve legacy
# for comparison). 24 tasks × step 2 only (matching paper §5 N=24 strong-tier).

#$ -l h_rt=12:0:0
#$ -l mem=64G
#$ -l gpu=1
#$ -wd /home/ucab352/Scratch/p79
#$ -N stage4mm_cls_v2
#$ -o /home/ucab352/Scratch/p79/logs/qsub_stage4mm_cls_v2.$JOB_ID.out
#$ -e /home/ucab352/Scratch/p79/logs/qsub_stage4mm_cls_v2.$JOB_ID.err
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

if [ ! -d "$REPO_DIR/results/mechanistic/archive_subset_b1_cls" ]; then
  echo "FATAL: archive_subset_b1_cls missing"
  exit 1
fi

HF_REVISION="ebb281ec70b05090aa6165b016eac8ec08e71b17"
HF_SNAPSHOT_DIR="$HF_HOME/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/$HF_REVISION"
if [ ! -f "$HF_SNAPSHOT_DIR/config.json" ]; then
  echo "FATAL: HF model snapshot missing at $HF_SNAPSHOT_DIR"
  exit 1
fi

OUT_DIR="$REPO_DIR/results/mechanistic/stage4_multimode_b1_cls"
mkdir -p "$OUT_DIR"
OUT_NPZ="$OUT_DIR/hidden_states_v2_fixed.npz"

echo "[$(date '+%H:%M:%S')] Stage 4 v2 fixed: --tier strong, 24 tasks × step 2 × 6 modes = 144 fwd passes"

python3 scripts/mechanistic/run_stage4_multimode_extract.py \
    --site classifieds \
    --tier strong \
    --n-tasks 24 \
    --steps 2 \
    --archived-run-dir "$REPO_DIR/results/mechanistic/archive_subset_b1_cls" \
    --output "$OUT_NPZ" \
    --model-revision "$HF_REVISION" \
    --modes dom phantom_text phantom_prompt phantom_som som vision

# Sentinel for auto_pull Phase 0 done-check
touch "$OUT_DIR/pilot_summary.md"

echo "[$(date '+%H:%M:%S')] Stage 4 cls v2 fixed DONE"
ls -la "$OUT_DIR/"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv
