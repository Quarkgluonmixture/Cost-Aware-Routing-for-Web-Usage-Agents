#!/usr/bin/env bash
# orchestrate_reddit_boundaries.sh — 4-day unattended boundary orchestrator (v2, 2026-07-08).
#
# v1 (2026-07-07) probed the proxy ONCE at the B1-som boundary; outage#5 hit exactly
# there (00:19Z 07-08, 3x503) so B0 was skipped with no later retry — violating the
# "earliest-window B0 insertion" decision (user 2026-07-04). v2 changes:
#   - try_b0() at EVERY boundary: proxy UP + B0 psom/pprompt not yet complete → insert.
#   - all conditions launch with FORCE_NEW=0 + RESET auto (existing run dir → resume
#     RESET_BEFORE=0 [B-1882 glob-resume, paper-grade clean on reddit per
#     PROTOCOL_NOTE_03]; no dir → fresh RESET_BEFORE=1). No FORCE_NEW=1 relaunch can
#     ever discard an aborted run's episodes.
#   - idempotent queue: condition with eps>=205 is skipped, so v2 can be killed and
#     re-armed at any time and resumes where the filesystem says we are.
#   - B0 abort → retry at later boundaries, max 2 attempts per condition.
# Fail-safe unchanged: B1/B2 abort or launch anomaly → ntfy + stop in safe state.
set -u

REPO="/home/ubuntu/workspace/p79"
PHASE_DIR="${REPO}/results/visualwebarena/phase1"
TOPIC="${NTFY_TOPIC:-p79-claude}"
SCORED=205
COND_WAIT_MAX_ITERS=1950   # x 120s = 65h per-condition ceiling
cd "${REPO}" || exit 1

log() { echo "[$(date '+%F %T')] $*"; }
ntfy() { curl -s -d "$1" "ntfy.sh/${TOPIC}" >/dev/null 2>&1 || true; }

runner_count() { pgrep -cf 'run_experiment\.[p]y' || true; }

dir_exists() { ls -dt "${PHASE_DIR}/$1_"[0-9]* >/dev/null 2>&1; }

eps_count() {
  local d
  d="$(ls -dt "${PHASE_DIR}/$1_"[0-9]* 2>/dev/null | head -1 || true)"
  [ -z "$d" ] && { echo 0; return; }
  find "$d" -path '*/episodes/*_summary_v2.json' 2>/dev/null | wc -l
}

wait_runner_done() {
  local i
  for i in $(seq 1 "${COND_WAIT_MAX_ITERS}"); do
    [ "$(runner_count)" = "0" ] && return 0
    sleep 120
  done
  return 1
}

do_bind() {
  .venv/bin/python3 scripts/analysis/validate_fire_manifest.py --populate --apply \
    && log "bind OK" || { log "bind FAILED (non-blocking)"; ntfy "⚠️ orch: bind failed after $1 — data intact, re-run --populate --apply manually"; }
}

proxy_up() {
  local ok=0 i
  for i in 1 2 3; do
    .venv/bin/python3 scripts/maintenance/probe_proxy_alive.py >/dev/null 2>&1 && ok=$((ok+1))
    sleep 20
  done
  [ "$ok" -ge 2 ]
}

# run_condition <label> <cfg_prefix> <queue_script> <queue_args...>
# FORCE_NEW=0 always; RESET_BEFORE auto (resume if a run dir exists).
# returns: 0 complete+bound / 1 abort (eps<205) / 2 launch failure / 3 timeout
run_condition() {
  local label="$1" prefix="$2"; shift 2
  local reset=1
  dir_exists "$prefix" && reset=0
  if [ "$(runner_count)" != "0" ]; then
    log "$label: pre-check FAILED (runner already present)"; ntfy "🔴 orch: $label pre-check found a live runner — STOPPING"; return 2
  fi
  log "$label: launching (RESET_BEFORE=$reset resume=$((1-reset))): $*"
  if ! env FORCE_NEW=0 RESET_BEFORE=$reset bash "$@" >> "${REPO}/logs/orch_queue_${label}.log" 2>&1; then
    log "$label: queue script exited non-zero"; ntfy "🔴 orch: $label queue launch FAILED (see logs/orch_queue_${label}.log)"; return 2
  fi
  sleep 120
  if [ "$(runner_count)" = "0" ]; then
    log "$label: runner not up 120s after queue OK"; ntfy "🔴 orch: $label runner did not start — STOPPING"; return 2
  fi
  ntfy "▶️ orch: $label launched$( [ $reset -eq 0 ] && echo ' (RESUME)' ), runner up"
  if ! wait_runner_done; then
    log "$label: 65h ceiling hit"; ntfy "🔴 orch: $label exceeded 65h ceiling — orchestrator exiting (runner left running)"; return 3
  fi
  local n; n="$(eps_count "$prefix")"
  if [ "$n" -ge "$SCORED" ]; then
    log "$label: COMPLETE ($n/$SCORED)"; do_bind "$label"; ntfy "✅ orch: $label COMPLETE $n/$SCORED + bind"; return 0
  fi
  log "$label: ABORT ($n/$SCORED)"; ntfy "🟠 orch: $label ABORT at $n/$SCORED"; return 1
}

# ---- B0 insertion, retried at every boundary while incomplete ----
B0_PSOM_TRIES=0
B0_PPROMPT_TRIES=0
try_b0() {
  local psom_eps pprompt_eps
  psom_eps="$(eps_count B0_phantom_som_reddit)"
  pprompt_eps="$(eps_count B0_phantom_prompt_reddit)"
  [ "$psom_eps" -ge "$SCORED" ] && [ "$pprompt_eps" -ge "$SCORED" ] && return 0
  if ! proxy_up; then
    log "try_b0: proxy DOWN (psom=$psom_eps pprompt=$pprompt_eps) — deferring to next boundary"
    return 0
  fi
  if [ "$psom_eps" -lt "$SCORED" ] && [ "$B0_PSOM_TRIES" -lt 2 ]; then
    B0_PSOM_TRIES=$((B0_PSOM_TRIES+1))
    ntfy "🟢 orch: proxy UP → B0 psom insert (at $psom_eps/205, attempt $B0_PSOM_TRIES/2)"
    run_condition "B0_psom" "B0_phantom_som_reddit" scripts/queues/queue_phantom_som.sh B0 reddit
    local rc=$?
    [ "$rc" -ge 2 ] && { ntfy "🔴 orch: B0 psom launch/timeout failure (rc=$rc) — STOPPING"; exit 1; }
    [ "$rc" -eq 1 ] && { ntfy "🟠 orch: B0 psom aborted (attempt $B0_PSOM_TRIES/2) — will retry at a later boundary"; return 0; }
    ntfy "ℹ️ orch: B0 psom done — task149 registry classify deferred to operator return"
    psom_eps="$(eps_count B0_phantom_som_reddit)"
  fi
  if [ "$psom_eps" -ge "$SCORED" ] && [ "$pprompt_eps" -lt "$SCORED" ] && [ "$B0_PPROMPT_TRIES" -lt 2 ]; then
    proxy_up || return 0
    B0_PPROMPT_TRIES=$((B0_PPROMPT_TRIES+1))
    ntfy "🟢 orch: proxy UP → B0 pprompt insert (at $pprompt_eps/205, attempt $B0_PPROMPT_TRIES/2)"
    run_condition "B0_pprompt" "B0_phantom_prompt_reddit" scripts/queues/queue_phantom_prompt.sh B0 reddit
    local rc2=$?
    [ "$rc2" -ge 2 ] && { ntfy "🔴 orch: B0 pprompt launch/timeout failure (rc=$rc2) — STOPPING"; exit 1; }
    [ "$rc2" -eq 1 ] && ntfy "🟠 orch: B0 pprompt aborted (attempt $B0_PPROMPT_TRIES/2) — will retry at a later boundary"
  fi
  return 0
}

# run_seq <label> <cfg_prefix> <queue_script> <args...> — idempotent + B0-first
run_seq() {
  local label="$1" prefix="$2"; shift 2
  try_b0
  local n; n="$(eps_count "$prefix")"
  if [ "$n" -ge "$SCORED" ]; then
    log "$label: already complete ($n/$SCORED) — skip"
    return 0
  fi
  run_condition "$label" "$prefix" "$@"
  local rc=$?
  if [ "$rc" -ne 0 ]; then
    ntfy "🔴 orch: $label failed (rc=$rc, proxy-immune baseline should not abort) — STOPPING queue"
    exit 1
  fi
}

log "=== orchestrator v2 up (pid $$) ==="
ntfy "🤖 orch v2 armed: per-boundary B0 insertion (psom→pprompt when proxy UP, ≤2 tries each) + idempotent B1/B2 queue"

# ---- Phase 0: adopt whatever is in flight ----
if [ "$(runner_count)" != "0" ]; then
  log "phase0: waiting on in-flight runner"
  wait_runner_done || { ntfy "🔴 orch: in-flight runner exceeded 65h ceiling — exiting"; exit 1; }
  do_bind "in-flight"
fi

run_seq "B1_vision"  "B1_vision_reddit"          scripts/queues/queue_baseline.sh B1 vision reddit
run_seq "B1_ptext"   "B1_phantom_text_reddit"    scripts/queues/queue_phantom_text.sh B1 reddit
run_seq "B1_psom"    "B1_phantom_som_reddit"     scripts/queues/queue_phantom_som.sh B1 reddit
run_seq "B1_pprompt" "B1_phantom_prompt_reddit"  scripts/queues/queue_phantom_prompt.sh B1 reddit
run_seq "B2_dom"     "B2_dom_reddit"             scripts/queues/queue_baseline.sh B2 dom reddit
run_seq "B2_som"     "B2_som_reddit"             scripts/queues/queue_baseline.sh B2 som reddit
run_seq "B2_vision"  "B2_vision_reddit"          scripts/queues/queue_baseline.sh B2 vision reddit
run_seq "B2_ptext"   "B2_phantom_text_reddit"    scripts/queues/queue_phantom_text.sh B2 reddit
run_seq "B2_psom"    "B2_phantom_som_reddit"     scripts/queues/queue_phantom_som.sh B2 reddit
run_seq "B2_pprompt" "B2_phantom_prompt_reddit"  scripts/queues/queue_phantom_prompt.sh B2 reddit
try_b0

ntfy "🏁 orch v2: reddit queue drained (B0 state: psom=$(eps_count B0_phantom_som_reddit)/205 pprompt=$(eps_count B0_phantom_prompt_reddit)/205)"
log "=== orchestrator v2 done ==="
