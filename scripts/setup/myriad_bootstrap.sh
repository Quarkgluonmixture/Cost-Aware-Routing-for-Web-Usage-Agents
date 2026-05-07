#!/usr/bin/env bash
# Myriad one-shot bootstrap — clone p79 + venv + pip install + pre-download HF model.
#
# Run on Myriad LOGIN node (where internet is available; compute nodes are firewalled).
# Idempotent: re-running detects existing state and skips done steps.
#
# Usage (from quark or anywhere):
#   ssh myriad bash -s < scripts/setup/myriad_bootstrap.sh
#
# Or interactive on Myriad:
#   ssh myriad
#   cd ~/Scratch && wget https://raw.githubusercontent.com/<user>/<repo>/master/scripts/setup/myriad_bootstrap.sh
#   bash myriad_bootstrap.sh
#
# Estimated time: ~30-45 min (mostly pip install torch + HF model download).
#
# Result: ~/Scratch/p79 ready to qsub mechanistic Stage 2B/2C jobs.

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/Quarkgluonmixture/Cost-Aware-Routing-for-Web-Usage-Agents.git}"
WORKSPACE="${WORKSPACE:-$HOME/Scratch/p79}"
HF_MODEL="${HF_MODEL:-Qwen/Qwen3-VL-4B-Instruct}"
HF_REVISION="${HF_REVISION:-ebb281ec70b05090aa6165b016eac8ec08e71b17}"

log() { echo "[bootstrap $(date '+%H:%M:%S')] $*"; }
fail() { log "FAIL: $*"; exit 1; }

# ---------------------------------------------------------------------------
# Step 1: Sanity — must run on Myriad login (internet present)
# ---------------------------------------------------------------------------

log "=== Step 1: environment sanity ==="
hostname=$(hostname)
log "  Host: $hostname"
if [[ "$hostname" != login* ]]; then
  log "  WARN: hostname does not start with 'login*'. Are you sure this is Myriad login?"
fi

if ! curl -sS --max-time 5 https://huggingface.co/ -I &>/dev/null; then
  fail "No internet access. Bootstrap MUST run on login node (compute nodes are firewalled)."
fi
log "  Internet: OK"

# Myriad-specific gotcha 1: gcc-libs/10.2.0 must be loaded BEFORE python+torch
# import to provide modern libstdc++ (RHEL 7 system libstdc++ is too old, will
# cause GLIBCXX_3.4.X not found errors at torch C++ extension import).
log "=== Step 1a: module load gcc-libs/10.2.0 (Myriad RHEL 7 GLIBCXX fix) ==="
module load gcc-libs/10.2.0 2>/dev/null && log "  gcc-libs/10.2.0 loaded" \
  || log "  WARN: gcc-libs/10.2.0 not available; torch C++ ext may fail"

# Myriad-specific gotcha 2: ~/.cache must symlink to Scratch
# (Home quota typically 50 GB; one HF model = 10 GB; cache will explode Home).
log "=== Step 1b: ~/.cache → Scratch symlink (Home quota protection) ==="
mkdir -p "$HOME/Scratch/cache"
if [ -L "$HOME/.cache" ]; then
  log "  ~/.cache already symlinked: $(readlink "$HOME/.cache")"
elif [ -d "$HOME/.cache" ]; then
  # Move existing content to Scratch then symlink
  log "  ~/.cache is real dir — moving content to ~/Scratch/cache"
  if [ -n "$(ls -A "$HOME/.cache" 2>/dev/null)" ]; then
    mv "$HOME/.cache"/* "$HOME/Scratch/cache/" 2>/dev/null || true
    mv "$HOME/.cache"/.* "$HOME/Scratch/cache/" 2>/dev/null || true
  fi
  rmdir "$HOME/.cache" 2>/dev/null || rm -rf "$HOME/.cache"
  ln -s "$HOME/Scratch/cache" "$HOME/.cache"
  log "  ~/.cache → ~/Scratch/cache (content migrated)"
else
  ln -s "$HOME/Scratch/cache" "$HOME/.cache"
  log "  ~/.cache → ~/Scratch/cache (created)"
fi

# ---------------------------------------------------------------------------
# Step 2: Clone or update repo
# ---------------------------------------------------------------------------

log "=== Step 2: workspace ==="
mkdir -p "$WORKSPACE"
if [ -d "$WORKSPACE/.git" ]; then
  log "  Repo exists at $WORKSPACE — git pull"
  cd "$WORKSPACE"
  git pull --ff-only
else
  log "  Cloning $REPO_URL → $WORKSPACE"
  git clone "$REPO_URL" "$WORKSPACE"
  cd "$WORKSPACE"
fi
log "  HEAD: $(git rev-parse --short HEAD) ($(git log -1 --format=%s | head -c 60))"

# Verify archive_subset_b1_cls present (paper-grade dataset for Stage 2B/2C)
if [ ! -f "$WORKSPACE/results/mechanistic/archive_subset_b1_cls/manifest.json" ]; then
  fail "archive_subset_b1_cls/manifest.json missing. Repo state is inconsistent — git pull failed?"
fi
n_strong=$(python3 -c "import json; print(len(json.load(open('$WORKSPACE/results/mechanistic/archive_subset_b1_cls/manifest.json'))['strong']))")
n_reverse=$(python3 -c "import json; print(len(json.load(open('$WORKSPACE/results/mechanistic/archive_subset_b1_cls/manifest.json'))['reverse']))")
log "  Dataset: $n_strong strong + $n_reverse reverse mirage candidates"

# ---------------------------------------------------------------------------
# Step 3: Python module + venv
# ---------------------------------------------------------------------------

log "=== Step 3: Python venv ==="
# Try Myriad pre-built python module first
if module avail python 2>&1 | grep -qE "python/3\.(10|11|12)"; then
  log "  Loading Myriad python module..."
  module load python/3.11.4 2>/dev/null || module load python/3.11 2>/dev/null || module load python3 2>/dev/null || true
fi

if [ ! -d "$WORKSPACE/.venv" ]; then
  log "  Creating venv..."
  python3 -m venv "$WORKSPACE/.venv"
else
  log "  venv exists, skipping creation"
fi

source "$WORKSPACE/.venv/bin/activate"
pip install --upgrade pip --quiet
log "  Python: $(python3 --version), pip: $(pip --version | head -c 50)"

# ---------------------------------------------------------------------------
# Step 4: Install p79 + dependencies
# ---------------------------------------------------------------------------

log "=== Step 4: pip install p79 + torch ==="
# Detect CUDA module
CUDA_VERSION=""
if module avail cuda 2>&1 | grep -qE "cuda/12\.[0-9]"; then
  module load cuda/12.1 2>/dev/null || module load cuda 2>/dev/null || true
  CUDA_VERSION="12.1"
elif module avail cuda 2>&1 | grep -qE "cuda/11\.[0-9]"; then
  module load cuda/11.8 2>/dev/null || module load cuda 2>/dev/null || true
  CUDA_VERSION="11.8"
fi
log "  CUDA module: ${CUDA_VERSION:-not loaded — may need module load cuda manually}"

if ! python3 -c "import torch" &>/dev/null; then
  log "  Installing torch (Myriad: --only-binary=:all: avoids gcc 4.8.5 source build)..."
  # Myriad-specific: cc=gcc 4.8.5 (RHEL 7 default), c++=gcc 10.2.0 (gcc-libs module).
  # Mismatch breaks numpy meson build from source. Force binary wheels only.
  # Pin numpy<2 because numpy 2.x has narrower wheel coverage on RHEL 7.
  if [ "$CUDA_VERSION" = "12.1" ]; then
    pip install --quiet --only-binary=:all: "numpy<2" \
        torch torchvision --index-url https://download.pytorch.org/whl/cu121
  elif [ "$CUDA_VERSION" = "11.8" ]; then
    pip install --quiet --only-binary=:all: "numpy<2" \
        torch torchvision --index-url https://download.pytorch.org/whl/cu118
  else
    pip install --quiet --only-binary=:all: "numpy<2" torch torchvision
  fi
fi

torch_ver=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "MISSING")
torch_cuda=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "?")
log "  torch: $torch_ver, cuda available: $torch_cuda"

# Install p79 + mechanistic deps
if ! pip show p79 &>/dev/null; then
  log "  Installing p79 + analysis deps..."
  pip install --quiet -e ".[analysis]" 2>&1 | tail -3
fi

# ---------------------------------------------------------------------------
# Step 5: Pre-download HF model (revision-pinned, paper-grade lock)
# ---------------------------------------------------------------------------

log "=== Step 5: HF model pre-download ==="
log "  Model: $HF_MODEL @ revision $HF_REVISION (DGX baseline lock)"
python3 - <<PYEOF
import os
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
from huggingface_hub import snapshot_download
import time
print("  Downloading (10-15 GB, ~5-10 min)...", flush=True)
t0 = time.time()
path = snapshot_download(
    "$HF_MODEL",
    revision="$HF_REVISION",
    local_dir_use_symlinks=False,
)
print(f"  Cache path: {path}")
print(f"  Elapsed: {time.time()-t0:.1f}s")
PYEOF

# ---------------------------------------------------------------------------
# Step 6: Sanity — env_snapshot + provenance
# ---------------------------------------------------------------------------

log "=== Step 6: env_snapshot + provenance ==="
python3 scripts/provenance/snapshot_env.py "$WORKSPACE/results/provenance/env_myriad_${hostname}_baseline.json" 2>&1 | tail -2

eval_sha=$(python3 -c "
import json
d = json.load(open('$WORKSPACE/results/provenance/env_myriad_${hostname}_baseline.json'))
print(d.get('evaluator_code', {}).get('combined_sha256', 'missing')[:16])
" 2>/dev/null || echo "?")
log "  Evaluator SHA on Myriad: ${eval_sha}..."

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------

log ""
log "=== Bootstrap COMPLETE ==="
log "Workspace: $WORKSPACE"
log "Venv: $WORKSPACE/.venv (activate: source $WORKSPACE/.venv/bin/activate)"
log "Model cache: ~/.cache/huggingface (compute nodes can read this offline)"
log ""
log "Next: qsub Stage 2B forward + Stage 2C reverse jobs:"
log "  cd $WORKSPACE"
log "  qsub scripts/queues/qsub_stage2b_myriad.sh    # forward 24 task ~12-24h"
log "  qsub scripts/queues/qsub_stage2c_myriad.sh    # reverse 15 task ~8-12h"
log "  qstat -u \$USER   # check job status"
