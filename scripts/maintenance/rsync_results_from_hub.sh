#!/usr/bin/env bash
# Pull results from the hub host (default DGX).
# Default: Tier B (episodes/*.jsonl + summary), no artifacts.
# Set ARTIFACTS=1 to also pull artifacts (screenshots/SoM); useful when
# regenerating paper figures locally. Optional COND=<id> filters to one
# condition dir to keep download small.
#
# Usage:
#   bash scripts/maintenance/rsync_results_from_hub.sh
#   ARTIFACTS=1 COND=phase1_phantom_som_router_0 bash scripts/maintenance/rsync_results_from_hub.sh
#   HOST=jiaming@spark-9ea3 RUN=B0_phantom_classifieds_20260426 \
#     bash scripts/maintenance/rsync_results_from_hub.sh
#
# Env:
#   HOST       SSH host/alias (default: spark-9ea3)
#   HUB_PATH   source dir on hub (default: ~/workspace/.../results/)
#   DEST       local destination (default: <repo_root>/results/)
#   ARTIFACTS  set to 1 to include artifacts/
#   RUN        optional: only sync results/<benchmark>/<phase>/<RUN>/...
#   COND       optional: only sync condition <COND> within RUN
#   DRY        set to 1 for --dry-run

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HOST="${HOST:-spark-9ea3}"
HUB_PATH="${HUB_PATH:-/home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents/results/}"
DEST="${DEST:-${REPO_ROOT}/results/}"
DRY_FLAG=""
[[ "${DRY:-0}" == "1" ]] && DRY_FLAG="--dry-run"

INCLUDES=(
  --include='*/'
  --include='*.jsonl'
  --include='condition_summary_v2.json'
  --include='condition_meta.json'
  --include='run_meta.json'
  --include='analysis/**'
)
EXCLUDES=()

if [[ "${ARTIFACTS:-0}" == "1" ]]; then
  INCLUDES+=( --include='artifacts/**' )
  echo "[rsync←hub] including artifacts/"
else
  EXCLUDES+=( --exclude='artifacts/' )
fi
EXCLUDES+=( --exclude='*' )

# Optionally narrow scope with RUN and COND filters
SRC_PATH="$HUB_PATH"
if [[ -n "${RUN:-}" ]]; then
  # Find which benchmark/phase the RUN lives under by remote ls (best effort).
  # Easier: assume visualwebarena/phase1 unless RUN looks like webarena.
  if [[ "$RUN" == *_wa_* || "$RUN" == webarena_* ]]; then
    SRC_PATH="${HUB_PATH%/}/webarena/phase1/${RUN}/"
    DEST="${DEST%/}/webarena/phase1/${RUN}/"
  else
    SRC_PATH="${HUB_PATH%/}/visualwebarena/phase1/${RUN}/"
    DEST="${DEST%/}/visualwebarena/phase1/${RUN}/"
  fi
  if [[ -n "${COND:-}" ]]; then
    SRC_PATH="${SRC_PATH%/}/${COND}/"
    DEST="${DEST%/}/${COND}/"
  fi
  mkdir -p "$DEST"
fi

echo "[rsync←hub] $HOST:$SRC_PATH → $DEST"
[[ -n "$DRY_FLAG" ]] && echo "  (DRY RUN)"

rsync -avzh --prune-empty-dirs $DRY_FLAG \
  "${INCLUDES[@]}" \
  "${EXCLUDES[@]}" \
  "$HOST:$SRC_PATH" "$DEST"

echo "[done]"
