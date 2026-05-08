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

# Pre-create logs/ — SGE redirects stdout/stderr BEFORE script execution.
# Without this, qsub jobs go to Eqw (Error queued waiting) state.
mkdir -p "$WORKSPACE/logs"
n_strong=$(python3 -c "import json; print(len(json.load(open('$WORKSPACE/results/mechanistic/archive_subset_b1_cls/manifest.json'))['strong']))")
n_reverse=$(python3 -c "import json; print(len(json.load(open('$WORKSPACE/results/mechanistic/archive_subset_b1_cls/manifest.json'))['reverse']))")
log "  Dataset: $n_strong strong + $n_reverse reverse mirage candidates"

# ---------------------------------------------------------------------------
# Step 3: Load Myriad pre-built pytorch module (avoids slow Lustre venv setup)
# ---------------------------------------------------------------------------

log "=== Step 3: Load pytorch module + setup PYTHONUSERBASE ==="
# Myriad pre-built pytorch/2.1.0/gpu auto-loads:
#   python/3.9.6-gnu-10.2.0 + cuda/11.8 + cudnn/9.2 + gcc-libs/10.2.0
# This avoids: (a) pip torch install ~3GB Lustre extraction, (b) numpy<2
# constraint failures, (c) GLIBCXX_3.4.X import errors.
module unload python python3 2>/dev/null || true
if ! module load pytorch/2.1.0/gpu 2>&1 | tail -2; then
  fail "module load pytorch/2.1.0/gpu failed. Check 'module avail pytorch'."
fi

log "  Loaded modules: $(module list 2>&1 | grep -E 'pytorch|python|cuda|cudnn' | tr -d '\n')"
log "  Python: $(python3 --version), at $(which python3)"

# Verify torch import works (login node will say cuda=False, that's fine)
torch_ver=$(python3 -c "import torch; print(torch.__version__)" 2>&1)
torch_cuda=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>&1)
log "  torch: $torch_ver, cuda: $torch_cuda"

# PYTHONUSERBASE = pip --user install dir. Default ~/.local fills Home quota;
# redirect to Scratch (lustre, plenty of space).
export PYTHONUSERBASE="$HOME/Scratch/python_user"
mkdir -p "$PYTHONUSERBASE"
log "  PYTHONUSERBASE: $PYTHONUSERBASE"

# Critical: PYTHONPATH must put user-site BEFORE module-bundled site-packages,
# because pytorch/2.1.0/gpu ships urllib3 v2.3.0 which is incompatible with
# RHEL 7 OpenSSL 1.0.2k. Without this, our pinned urllib3<2 in user-site is
# shadowed by the module's v2.3.0 and verify fails.
USER_SITE="$PYTHONUSERBASE/lib/python3.9/site-packages"
mkdir -p "$USER_SITE"
export PYTHONPATH="$USER_SITE:${PYTHONPATH:-}"
log "  PYTHONPATH prepended user-site: $USER_SITE"

# Persist both in .bashrc (idempotent: only add if not already there)
if ! grep -q "PYTHONUSERBASE.*Scratch/python_user" "$HOME/.bashrc" 2>/dev/null; then
  cat >> "$HOME/.bashrc" <<'BASHRC_EOF'

# Myriad p79 environment (added by myriad_bootstrap.sh)
export PYTHONUSERBASE=$HOME/Scratch/python_user
export PYTHONPATH=$PYTHONUSERBASE/lib/python3.9/site-packages:$PYTHONPATH
BASHRC_EOF
  log "  Added PYTHONUSERBASE + PYTHONPATH to ~/.bashrc"
fi

# ---------------------------------------------------------------------------
# Step 4: pip install --user (skip torch — already from module)
# ---------------------------------------------------------------------------

log "=== Step 4: pip install --user (transformers + p79 deps) ==="

# Install deps to PYTHONUSERBASE.
# Final working stack (笔记 §115b chronicle, 2026-05-08 night Myriad bootstrap session):
#   - Use module torch 2.1.0 (skip pip torch — saves 2GB Lustre extract).
#     Compatibility via sitecustomize.py adapter that wraps
#     torch._pytree._register_pytree_node → register_pytree_node and drops
#     unsupported kwargs (serialized_type_name etc).
#   - urllib3<2: RHEL 7 OpenSSL 1.0.2k incompatible with v2 (urllib3#2168)
#   - numpy<2: torch 2.1 binary compiled against NumPy 1.x (crashes on 2.0)
#   - transformers==4.57.6: Qwen3VLForConditionalGeneration first stable here
#     (4.55 still missing it; 4.49 lacks Qwen3-VL entirely)
#   - constraints.txt enforces pins across all transitive resolves

# Write constraints file (single source of truth)
cat > "$HOME/Scratch/myriad_constraints.txt" <<'CONST_EOF'
urllib3<2
numpy<2
CONST_EOF
log "  Wrote constraints: ~/Scratch/myriad_constraints.txt (urllib3<2, numpy<2)"

# Write sitecustomize.py compatibility shim (torch 2.1 ↔ transformers 4.50+)
cat > "$USER_SITE/sitecustomize.py" <<'SHIM_EOF'
"""Myriad workaround: torch 2.1 lacks register_pytree_node public API +
its private _register_pytree_node has narrower signature than 2.3+ public.
Wrap with adapter dropping unsupported kwargs."""
import sys
import builtins
import functools

_PATCHED = False

def _maybe_patch_torch():
    global _PATCHED
    if _PATCHED:
        return
    pt = sys.modules.get('torch.utils._pytree')
    if not pt or not hasattr(pt, '_register_pytree_node'):
        return
    if hasattr(pt, 'register_pytree_node') and getattr(
        pt.register_pytree_node, '_myriad_patched', False):
        _PATCHED = True
        return
    _orig = pt._register_pytree_node

    @functools.wraps(_orig)
    def register_pytree_node(*args, **kwargs):
        for k in ('serialized_type_name', 'to_dumpable_context',
                  'from_dumpable_context', 'flatten_with_keys_fn'):
            kwargs.pop(k, None)
        return _orig(*args, **kwargs)

    register_pytree_node._myriad_patched = True
    pt.register_pytree_node = register_pytree_node
    _PATCHED = True

_orig_import = builtins.__import__
def _patched_import(name, *args, **kwargs):
    mod = _orig_import(name, *args, **kwargs)
    if 'torch' in name and not _PATCHED:
        _maybe_patch_torch()
    return mod
builtins.__import__ = _patched_import
SHIM_EOF
log "  Wrote sitecustomize.py: torch 2.1 ↔ transformers 4.50+ compat shim"

# Pin urllib3 + numpy first (constraints satisfied for all later installs)
log "  Installing urllib3<2 + numpy<2 (RHEL 7 + torch 2.1 binary compat)..."
pip install --user --only-binary=:all: --progress-bar=on \
    --cache-dir=/tmp/pip_cache_$USER \
    "urllib3<2" "numpy<2" 2>&1 | tail -3

# Install transformers 4.57.6 (Qwen3-VL first stable) + ML stack
log "  Installing transformers 4.57.6 + ML deps (sklearn / scipy / matplotlib / pandas)..."
pip install --user --only-binary=:all: --progress-bar=on \
    --cache-dir=/tmp/pip_cache_$USER \
    -c "$HOME/Scratch/myriad_constraints.txt" \
    "transformers==4.57.6" accelerate qwen-vl-utils huggingface_hub \
    Pillow PyYAML scikit-learn scipy matplotlib pandas 2>&1 | tail -5

# Install p79 editable, --no-deps so pip doesn't try to re-resolve torch
log "  Installing p79 (editable, --no-deps)..."
pip install --user --no-deps -e . 2>&1 | tail -3

# Verify import chain
log "  Verifying p79 + torch + transformers import..."
python3 -c "
import sys, torch, transformers, p79
from p79.mechanistic.extract_hidden_states import HiddenStateExtractor
print(f'    Python: {sys.version.split()[0]}')
print(f'    torch:  {torch.__version__}')
print(f'    transformers: {transformers.__version__}')
print(f'    p79.mechanistic: import OK')
"

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
log "PyTorch: from module pytorch/2.1.0/gpu (auto-loads python/3.9.6 + cuda/11.8)"
log "User packages: \$PYTHONUSERBASE = $PYTHONUSERBASE"
log "Model cache: ~/.cache/huggingface → ~/Scratch/cache (compute nodes can read offline)"
log ""
log "Next: qsub Stage 2B forward + Stage 2C reverse jobs:"
log "  cd $WORKSPACE"
log "  qsub scripts/queues/qsub_stage2b_myriad.sh    # forward 24 task ~12-24h"
log "  qsub scripts/queues/qsub_stage2c_myriad.sh    # reverse 15 task ~8-12h"
log "  qstat -u \$USER   # check job status"
