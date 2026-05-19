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
# B-1754 (2026-05-19 morning, Fire-3 LIVE follow-up): source corrected from
# stale `/mnt/scratch/results/...` (last A100-side mirror update 2026-05-15~16,
# B2_3mode + _archive_smoke_20260515 only — pre-dates Fire-3 attempt #6 LIVE
# 21:27Z 2026-05-18) → live `/home/ubuntu/workspace/p79/results/...` where
# Fire-3 R19740 + queue_phase1_paper_grade.sh writes. Pre-fix cron synced
# stale /mnt/scratch so DGX mirror never saw R19740 fresh data → quark:8765
# Gallery view always 3+ days behind. Post-fix: 15-min auto-refresh from
# live LIVE Fire-3 source.
A100_RESULTS="/home/ubuntu/workspace/p79/results/visualwebarena/phase1/"
DGX_RESULTS="${REPO_ROOT}/results/visualwebarena/phase1/"

# B-1755 (2026-05-19 morning, Fire-3 LIVE follow-up + user gallery URL ask):
# additional top-level result dirs that watchdog auto-refreshes via the
# 5-step GALLERY pipeline (per_run + aggregate + combined + unified +
# phase1_paper_grade per experiment_watchdog.py:_regenerate_aggregate_gallery).
# Pre-fix cron only synced visualwebarena/phase1/ subtree → top-level
# results/phase1_paper_grade/ + results/B0_3mode/ + results/B0_unified/
# (all watchdog outputs) never reached DGX → quark:8765 mirror could not
# serve them → user had no aggregated Phase 1 paper-grade gallery URL.
# Post-fix: same 15-min cron syncs these too (separate rsync per dir, no
# --delete-after on top-level — defensive, watchdog regen can re-create
# from A100 anyway).
A100_TOPLEVEL_DIRS=(
  "phase1_paper_grade"  # 5th GALLERY step output (cross-baseline aggregate)
  "B0_3mode"            # 4th GALLERY step output (combined VWA+WA per baseline)
  "B1_3mode"
  "B2_3mode"
  "B0_unified"          # 5th-deprecated GALLERY step output (per-baseline unified)
  "B1_unified"
  "B2_unified"
)

mkdir -p "${DGX_RESULTS}"

# B-877 (/stress A1.24 P0-5-AB*, 2026-05-17): single-host flock on sync
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
# B-877: rsync log path now per-PID (was shared /tmp/rsync_a100_last.log
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

# B-1755 (2026-05-19 morning): rsync top-level watchdog-output dirs too.
# Separate rsync per dir for fault isolation (one dir's failure doesn't
# abort the others). NO --delete on top-level (defensive: watchdog regen
# can re-create from A100 source, but locally-cached gallery.html should
# not be silently nuked if A100 source dir doesn't exist yet for a given
# baseline).
DGX_RESULTS_ROOT="${REPO_ROOT}/results/"
A100_RESULTS_ROOT="/home/ubuntu/workspace/p79/results/"
# B-1761 (2026-05-19): exclude VWA submodule build/docs artifacts pulled in
# via `-L` symlink-deref on A100's `_vwa → external/visualwebarena/`. The
# `environment_docker/` subtree alone is ~32 GB (docker build context) and
# is NOT needed by gallery rendering (gallery only needs `coco_images/` +
# `config_files/` for intent images + task configs, ~15 MB total). Pre-fix
# the foreground sync test PID 2368843 spent ~43 min pulling 32 GB of docker
# context to DGX. `--delete-excluded` prunes the locally-cached 32 GB
# on next sync since they now match exclude patterns.
TOPLEVEL_EXCLUDES=(
  --exclude='_vwa/environment_docker/'
  --exclude='_vwa/docs/'
  --delete-excluded
)
for d in "${A100_TOPLEVEL_DIRS[@]}"; do
  src="${A100_RESULTS_ROOT}${d}/"
  dst="${DGX_RESULTS_ROOT}${d}/"
  mkdir -p "${dst}"
  if ssh -o ConnectTimeout=10 "${A100_HOST}" "test -d ${src}" 2>/dev/null; then
    log "rsync top-level ${A100_HOST}:${src} → ${dst}"
    if ! rsync -az --partial --append-verify --info=stats1 -L \
           "${TOPLEVEL_EXCLUDES[@]}" \
           -e "ssh -o ConnectTimeout=20 -o ServerAliveInterval=30" \
           "${A100_HOST}:${src}" "${dst}" 2>&1 | tee -a "${RSYNC_LOG}"; then
      log "  ✗ top-level rsync ${d} failed (non-fatal; continuing)"
    else
      log "  ✓ ${d} synced"
    fi
  else
    log "  - skipping ${d} (not on A100 yet — watchdog generates lazily)"
  fi
done

# Post-sync: detect new condition_summary_v2.json
SUMMARY_AFTER=$(find "${DGX_RESULTS}" -maxdepth 4 -name "condition_summary_v2.json" 2>/dev/null | sort)
NEW_SUMMARIES=$(comm -13 <(echo "${SUMMARY_BEFORE}") <(echo "${SUMMARY_AFTER}") | grep -v '^$' || true)

# B-878 (/stress A1.24 P0-5-AB* sub-b, 2026-05-17): also detect DELETED
# summaries. Pre-fix: A100 clear_tasks → rsync --delete-after propagates
# deletion to DGX → NEW_SUMMARIES empty → exit 0 → no analysis refresh →
# Obsidian cells.base + GLM PLAYBOOK show stale SR from old finalized
# summary. With cron-vs-manual race window now: deletion event MUST
# trigger downstream state recompute, same as new-summary event.
DELETED_SUMMARIES=$(comm -23 <(echo "${SUMMARY_BEFORE}") <(echo "${SUMMARY_AFTER}") | grep -v '^$' || true)

if [[ -n "${DELETED_SUMMARIES}" ]]; then
  log "Deleted condition_summary_v2.json (propagated from A100 clear_tasks):"
  echo "${DELETED_SUMMARIES}" | while read -r s; do log "  - ${s}"; done
  # B-878: log to dedicated jsonl + ntfy high priority. Operator audit trail.
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
