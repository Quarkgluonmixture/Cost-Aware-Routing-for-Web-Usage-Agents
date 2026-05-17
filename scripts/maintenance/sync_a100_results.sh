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

# B-859 (/stress A1.24 P0-5-AB*, 2026-05-17): single-host flock on sync
# script to prevent concurrent rsync overlap. A100 sync >15min (slow VPN
# or large artifacts) → next cron tick fires concurrent rsync → two rsync
# instances both write same ${DGX_RESULTS} + share /tmp/rsync_a100_last.log
# + run --delete-after on different file-list snapshots → can delete files
# the other instance just transferred OR leave partial+mtime mixed state.
# Per-host operator framing (user is sole A100+DGX operator) eliminates
# cross-operator drama but NOT cron-self-overlap (cron is automatic).
mkdir -p "${REPO_ROOT}/.locks"
SYNC_LOCK="${REPO_ROOT}/.locks/sync_a100_results.lock"
exec 9>"${SYNC_LOCK}"
if ! flock -n 9; then
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] [skip] another sync_a100 instance holds ${SYNC_LOCK}; skipping this tick." >&2
  exit 0
fi
echo "pid=$$ start=$(date +%s)" >&9 || true

ts() { date +'%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*"; }

# Pre-sync: snapshot which condition_summary_v2.json files we have locally
SUMMARY_BEFORE=$(find "${DGX_RESULTS}" -maxdepth 4 -name "condition_summary_v2.json" 2>/dev/null | sort)

# rsync flags:
#  -a  archive (preserve perms / times / symlinks)
#  -z  compress over network
#  --partial  keep partial files on transfer fail (resume next run)
#  --append-verify  resume large files via append + checksum verify
#  --delete-after  B-841 (A1.15b P0-1): propagate A100-side deletions to DGX
#    AFTER successful transfer. If operator runs clear_tasks.py on A100 for
#    paper-grade re-fire, stale finalized run must NOT persist on DGX mirror,
#    otherwise glm_cell_autoupdate.latest_match (mtime-based) picks the stale
#    finalized over the fresh in-flight → cells.base shows wrong state → user
#    decides re-launch based on corrupt signal. --delete-after (vs --delete)
#    only fires after rsync transfer completes successfully — safer if SSH
#    chain drops mid-sync (no data loss on transient failure).
#  --delete-excluded  drop locally-cached files matching exclude patterns
#  --exclude artifacts/  large screenshot dir (sync only summaries first, artifacts opt-in)
#  --exclude episodes/*/step_*.png  (image-only excludes; keep step JSONLs)
log "rsync ${A100_HOST}:${A100_RESULTS} → ${DGX_RESULTS}"
RSYNC_OPTS=(-az --partial --append-verify --delete-after --info=stats1)
if [[ "${DRY_RUN}" == "1" ]]; then
  RSYNC_OPTS+=(--dry-run)
  log "DRY-RUN mode (no actual transfer)"
fi

# Run rsync with SSH chain. SSH config alias condense-a100 resolves via quark
# ProxyCommand. ConnectTimeout matters because UCL Cisco VPN can drop.
#
# B-859: rsync log path now per-PID (was shared /tmp/rsync_a100_last.log
# → concurrent instances overwrote each other's diagnostic). Lock above
# prevents true concurrency, but if lock fails-open on platform without
# flock, per-PID log keeps forensic separate.
RSYNC_LOG="/tmp/rsync_a100_last.${$}.log"
if ! rsync "${RSYNC_OPTS[@]}" \
       -e "ssh -o ConnectTimeout=20 -o ServerAliveInterval=30" \
       "${A100_HOST}:${A100_RESULTS}" "${DGX_RESULTS}" 2>&1 | tee "${RSYNC_LOG}"; then
  log "✗ rsync failed (SSH chain or A100 unreachable). Retry on next cron run."
  exit 1
fi

# Post-sync: detect new condition_summary_v2.json
SUMMARY_AFTER=$(find "${DGX_RESULTS}" -maxdepth 4 -name "condition_summary_v2.json" 2>/dev/null | sort)
NEW_SUMMARIES=$(comm -13 <(echo "${SUMMARY_BEFORE}") <(echo "${SUMMARY_AFTER}") | grep -v '^$' || true)

# B-860 (/stress A1.24 P0-5-AB* sub-b, 2026-05-17): also detect DELETED
# summaries. Pre-fix: A100 clear_tasks → rsync --delete-after propagates
# deletion to DGX → NEW_SUMMARIES empty → exit 0 → no analysis refresh →
# Obsidian cells.base + GLM PLAYBOOK show stale SR from old finalized
# summary. With cron-vs-manual race window now: deletion event MUST
# trigger downstream state recompute, same as new-summary event.
DELETED_SUMMARIES=$(comm -23 <(echo "${SUMMARY_BEFORE}") <(echo "${SUMMARY_AFTER}") | grep -v '^$' || true)

if [[ -n "${DELETED_SUMMARIES}" ]]; then
  log "Deleted condition_summary_v2.json (propagated from A100 clear_tasks):"
  echo "${DELETED_SUMMARIES}" | while read -r s; do log "  - ${s}"; done
  # B-860: log to dedicated jsonl + ntfy high priority. Operator audit trail.
  mkdir -p "${REPO_ROOT}/logs/cron"
  DEL_LOG="${REPO_ROOT}/logs/cron/sync_a100_deletions.jsonl"
  while IFS= read -r del_path; do
    [[ -z "${del_path}" ]] && continue
    printf '{"ts":"%s","path":"%s","cron_pid":%s}\n' "$(ts)" "${del_path}" "$$" >> "${DEL_LOG}"
  done <<< "${DELETED_SUMMARIES}"
  # ntfy best-effort (single curl, 3s timeout)
  DEL_COUNT=$(echo "${DELETED_SUMMARIES}" | grep -c . || true)
  curl -s --max-time 3 \
    -H "Title: P79 sync_a100 deletion propagated" \
    -H "Priority: high" \
    -d "[$(ts)] ${DEL_COUNT} condition_summary_v2.json deleted from DGX mirror (A100 clear_tasks propagation). See logs/cron/sync_a100_deletions.jsonl." \
    "https://ntfy.sh/p79-claude" >/dev/null 2>&1 || true
fi

if [[ -z "${NEW_SUMMARIES}" && -z "${DELETED_SUMMARIES}" ]]; then
  log "No new or deleted condition_summary_v2.json this cycle. Skipping analysis."
  exit 0
fi

if [[ -n "${NEW_SUMMARIES}" ]]; then
  log "New condition_summary_v2.json this cycle:"
  echo "${NEW_SUMMARIES}" | while read -r s; do log "  + ${s}"; done
fi

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
