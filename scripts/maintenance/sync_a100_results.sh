#!/usr/bin/env bash
# sync_a100_results.sh — DGX-side cron rsync of A100 Phase 1a results.
#
# WHY: Phase 1a fire runs on condense-a100, results land in
# /mnt/scratch/results/visualwebarena/phase1/. P79 GLM Phase 2 cron
# (cells.base auto-update, PLAYBOOK refresh) lives on DGX and reads
# DGX-local results/. Without sync bridge, 18-day fire = DGX-side
# monitoring stack stale.
#
# WHAT: rsync over SSH chain (DGX → quark Tailscale → condense-a100 via
# Cisco VPN ProxyCommand). Pulls new run_dir + condition_summary_v2.json
# + episode summaries + analysis subdirs. After successful sync, if any
# new condition_summary_v2.json detected, trigger `make analysis FAST=1`
# (incremental).
#
# USAGE (cron, every 15 min):
#   */15 * * * * cd /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents && \
#     bash scripts/maintenance/sync_a100_results.sh >> logs/cron_sync_a100.log 2>&1
#
# Manual run:
#   bash scripts/maintenance/sync_a100_results.sh
#   bash scripts/maintenance/sync_a100_results.sh --dry-run
#   bash scripts/maintenance/sync_a100_results.sh --no-analysis
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

DRY_RUN=0
SKIP_ANALYSIS=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --no-analysis) SKIP_ANALYSIS=1 ;;
    *) echo "Unknown arg: $arg" >&2; exit 2 ;;
  esac
done

A100_HOST="condense-a100"
A100_RESULTS="/mnt/scratch/results/visualwebarena/phase1/"
DGX_RESULTS="${REPO_ROOT}/results/visualwebarena/phase1/"

mkdir -p "${DGX_RESULTS}"

ts() { date +'%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*"; }

# Pre-sync: snapshot which condition_summary_v2.json files we have locally
SUMMARY_BEFORE=$(find "${DGX_RESULTS}" -maxdepth 4 -name "condition_summary_v2.json" 2>/dev/null | sort)

# rsync flags:
#  -a  archive (preserve perms / times / symlinks)
#  -z  compress over network
#  --partial  keep partial files on transfer fail (resume next run)
#  --append-verify  resume large files via append + checksum verify
#  --delete-excluded  drop locally-cached files matching exclude patterns
#  --exclude artifacts/  large screenshot dir (sync only summaries first, artifacts opt-in)
#  --exclude episodes/*/step_*.png  (image-only excludes; keep step JSONLs)
log "rsync ${A100_HOST}:${A100_RESULTS} → ${DGX_RESULTS}"
RSYNC_OPTS=(-az --partial --append-verify --info=stats1)
if [[ "${DRY_RUN}" == "1" ]]; then
  RSYNC_OPTS+=(--dry-run)
  log "DRY-RUN mode (no actual transfer)"
fi

# Run rsync with SSH chain. SSH config alias condense-a100 resolves via quark
# ProxyCommand. ConnectTimeout matters because UCL Cisco VPN can drop.
if ! rsync "${RSYNC_OPTS[@]}" \
       -e "ssh -o ConnectTimeout=20 -o ServerAliveInterval=30" \
       "${A100_HOST}:${A100_RESULTS}" "${DGX_RESULTS}" 2>&1 | tee /tmp/rsync_a100_last.log; then
  log "✗ rsync failed (SSH chain or A100 unreachable). Retry on next cron run."
  exit 1
fi

# Post-sync: detect new condition_summary_v2.json
SUMMARY_AFTER=$(find "${DGX_RESULTS}" -maxdepth 4 -name "condition_summary_v2.json" 2>/dev/null | sort)
NEW_SUMMARIES=$(comm -13 <(echo "${SUMMARY_BEFORE}") <(echo "${SUMMARY_AFTER}") | grep -v '^$' || true)

if [[ -z "${NEW_SUMMARIES}" ]]; then
  log "No new condition_summary_v2.json this cycle. Skipping analysis."
  exit 0
fi

log "New condition_summary_v2.json this cycle:"
echo "${NEW_SUMMARIES}" | while read -r s; do log "  + ${s}"; done

if [[ "${SKIP_ANALYSIS}" == "1" ]]; then
  log "--no-analysis flag set; skipping analysis trigger."
  exit 0
fi

# Trigger incremental analysis (FAST=1 skips per-run rederive,
# only re-runs aggregators + figures + status updates).
log "Triggering 'make analysis FAST=1' (incremental)..."
if make analysis FAST=1 2>&1 | tail -20 | sed 's/^/    /'; then
  log "✓ analysis FAST=1 completed"
else
  log "✗ analysis FAST=1 returned non-zero (logged to console; cron continues)"
fi

# Update GLM Phase 2 frontmatter via cells autoupdater (if exists)
if [[ -f scripts/maintenance/glm/glm_cell_autoupdate.py ]]; then
  log "Triggering glm_cell_autoupdate (cells.base sync)..."
  .venv/bin/python3 scripts/maintenance/glm/glm_cell_autoupdate.py 2>&1 | tail -5 | sed 's/^/    /' || true
fi

log "Cycle complete."
