#!/usr/bin/env bash
# Push results JSONL + summary to the hub host (default DGX).
# Tier B (episodes/*.jsonl + condition/run_meta.json + analysis/*) only;
# Tier C (artifacts: screenshots, SoM图) is excluded — pull on demand via
# rsync_results_from_hub.sh ARTIFACTS=1 COND=<id>.
#
# Usage:
#   bash scripts/maintenance/rsync_results_to_hub.sh
#   HOST=jiaming@spark-9ea3 bash scripts/maintenance/rsync_results_to_hub.sh
#   HUB_PATH=/data/p79/results/ HOST=dgx bash scripts/maintenance/rsync_results_to_hub.sh
#   DRY=1 bash scripts/maintenance/rsync_results_to_hub.sh
#
# Env:
#   HOST       SSH host/alias (default: spark-9ea3)
#   HUB_PATH   destination dir on hub (default: ~/workspace/.../results/)
#   SOURCE     source dir locally (default: <repo_root>/results/)
#   DRY        set to 1 for --dry-run

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HOST="${HOST:-spark-9ea3}"
HUB_PATH="${HUB_PATH:-/home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents/results/}"
SOURCE="${SOURCE:-${REPO_ROOT}/results/}"
DRY_FLAG=""
[[ "${DRY:-0}" == "1" ]] && DRY_FLAG="--dry-run"

[[ -d "$SOURCE" ]] || { echo "ERROR: source $SOURCE not found" >&2; exit 1; }

echo "[rsync→hub] $SOURCE → $HOST:$HUB_PATH (Tier B only, no artifacts)"
[[ -n "$DRY_FLAG" ]] && echo "  (DRY RUN — no actual transfer)"

rsync -avzh --prune-empty-dirs $DRY_FLAG \
  --include='*/' \
  --include='*.jsonl' \
  --include='condition_summary_v2.json' \
  --include='condition_meta.json' \
  --include='run_meta.json' \
  --include='analysis/**' \
  --exclude='artifacts/' \
  --exclude='*' \
  "$SOURCE" "$HOST:$HUB_PATH"

echo "[done]"
