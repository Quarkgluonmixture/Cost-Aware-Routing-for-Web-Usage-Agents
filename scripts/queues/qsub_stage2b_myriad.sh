#!/bin/bash -l
# Mechanistic Stage 2B (forward direction) — 24 strong mirage task on Myriad.
#
# Submission:
#   ssh myriad
#   cd ~/Scratch/p79
#   qsub scripts/queues/qsub_stage2b_myriad.sh
#
# Monitoring:
#   qstat -u $USER
#   ssh myriad tail -f ~/Scratch/p79/logs/qsub_stage2b_b1_cls.*.out
#
# Prereqs (run once via myriad_bootstrap.sh):
#   - ~/Scratch/p79 git pulled at commit ≥ 1fefd39 (post-§115 protocols)
#   - ~/Scratch/p79/.venv with torch + transformers + p79
#   - HF model Qwen3-VL-4B-Instruct revision ebb281ec... cached in ~/.cache/huggingface
#   - results/mechanistic/archive_subset_b1_cls/ pulled (24 strong + 15 reverse subset)
#
# Output:
#   results/mechanistic/stage2b_curated_b1_cls_myriad/
#     ├── env_snapshot.json                       (auto, §114 Gap 1)
#     ├── run_manifest.json                       (auto, §114 Gap 3)
#     ├── patching_continuation_results.json      (full 24 task data)
#     ├── patching_continuation_curves.png        (4-panel layer-resolved plot)
#     └── pilot_summary.md                        (paper §5 quotable summary)

# ============================================================================
# SGE directives
# ============================================================================

#$ -l h_rt=36:0:0          # 36h wallclock (24 task × ~50 min/task with margin)
#$ -l mem=64G              # 64GB host memory (model needs ~12GB GPU + activations)
#$ -l gpu=1                # single GPU sufficient (Qwen3-VL-4B bf16 ~10GB)
#$ -ac allow=L             # L-type 4× A100 40GB (single GPU = 40GB; faster queue than V/U)
#$ -wd /home/ucab352/Scratch/p79
#$ -N stage2b_b1_cls
#$ -o /home/ucab352/Scratch/p79/logs/qsub_stage2b_b1_cls.$JOB_ID.out
#$ -e /home/ucab352/Scratch/p79/logs/qsub_stage2b_b1_cls.$JOB_ID.err
#$ -j n                    # separate stdout / stderr

# Ensure logs/ exists (qsub fails if -o / -e dirs missing)
mkdir -p /home/ucab352/Scratch/p79/logs

# ============================================================================
# Environment setup
# ============================================================================

set -euo pipefail

REPO_DIR="/home/ucab352/Scratch/p79"
cd "$REPO_DIR"

echo "[$(date '+%H:%M:%S')] Job $JOB_ID start on $(hostname)"
echo "[$(date '+%H:%M:%S')] GPU info:"
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv

# Load Myriad modules (varies by node — fail-soft)
# CRITICAL: gcc-libs/10.2.0 must load BEFORE python (provides modern libstdc++
# for torch C++ extensions; RHEL 7 system libstdc++ too old, breaks at import)
module load gcc-libs/10.2.0 2>/dev/null || true
module load python/3.11.4 2>/dev/null || module load python/3.11 2>/dev/null || module load python3 2>/dev/null || true
module load cuda/12.1 2>/dev/null || module load cuda/11.8 2>/dev/null || module load cuda 2>/dev/null || true

# Activate p79 venv
source "$REPO_DIR/.venv/bin/activate"

# Compute nodes are firewalled — force HF offline (use cache only)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# Tell HF where cache is
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

echo "[$(date '+%H:%M:%S')] Python: $(python3 --version)"
echo "[$(date '+%H:%M:%S')] Repo HEAD: $(git rev-parse --short HEAD)"
echo "[$(date '+%H:%M:%S')] Working dir: $(pwd)"

# ============================================================================
# Sanity check
# ============================================================================

if [ ! -f "$REPO_DIR/results/mechanistic/archive_subset_b1_cls/manifest.json" ]; then
  echo "FATAL: archive_subset_b1_cls/manifest.json missing. Run myriad_bootstrap.sh first."
  exit 1
fi

n_strong=$(python3 -c "import json; print(len(json.load(open('$REPO_DIR/results/mechanistic/archive_subset_b1_cls/manifest.json'))['strong']))")
echo "[$(date '+%H:%M:%S')] Dataset: $n_strong strong mirage candidates"

# ============================================================================
# Run Stage 2B forward direction (paper §5 mechanism scale-up)
# ============================================================================

OUT_DIR="$REPO_DIR/results/mechanistic/stage2b_curated_b1_cls_myriad"
mkdir -p "$OUT_DIR"

echo "[$(date '+%H:%M:%S')] Launching Stage 2B forward (24 task × all layers × 50 max_new_tokens)..."

python3 scripts/mechanistic/run_stage2b_continuation_pilot.py \
    --site classifieds \
    --n-tasks 24 \
    --step 2 \
    --max-new-tokens 50 \
    --source-mode som \
    --target-mode phantom_som \
    --output-dir "$OUT_DIR" \
    --archived-run-dir "$REPO_DIR/results/visualwebarena/phase1/B1_phantom_som_classifieds_20260428"

# ============================================================================
# Done
# ============================================================================

echo "[$(date '+%H:%M:%S')] Stage 2B forward DONE"
echo "[$(date '+%H:%M:%S')] Output: $OUT_DIR"
ls -la "$OUT_DIR/"

# Final GPU stats
echo "[$(date '+%H:%M:%S')] Final GPU memory:"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv
