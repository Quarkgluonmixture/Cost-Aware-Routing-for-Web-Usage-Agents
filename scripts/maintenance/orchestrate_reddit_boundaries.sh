#!/usr/bin/env bash
# orchestrate_reddit_boundaries.sh — 4-day unattended boundary orchestrator (2026-07-07).
#
# Context: operator away 07-08→07-11 (user decision 2026-07-07, all-A answers).
# Runs ON the A100 (setsid). Replaces the manual "monitor fires → operator acts"
# boundary loop that caused the 15h (07-01) and 35h (07-04→06) fire stalls.
#
# Queue (sequential, one condition at a time — same-site XOR hard rule):
#   0. wait current B1 som reddit → bind
#   1. probe proxy → UP:  B0 psom RESUME (FORCE_NEW=0 RESET_BEFORE=0, R28173 146/205)
#                          → B0 pprompt (fresh)     [any B0 abort → skip rest of B0]
#                    DOWN: skip B0 entirely
#   2. B1 vision → ptext → psom → pprompt   (fresh, RESET_BEFORE=1; abort → STOP)
#   3. B2 dom → som → vision → ptext → psom → pprompt (same policy)
#
# Per-boundary: bind via validate_fire_manifest --populate --apply + ntfy.
# Fail-safe: any unexpected state → ntfy + exit (stop in a known-good state).
# B0-abort policy: skip to B1 chain (user-confirmed 2026-07-07).
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

# eps_count <run_dir_prefix> — v2 episode summaries in NEWEST matching run dir
eps_count() {
  local d
  d="$(ls -dt "${PHASE_DIR}/$1_"[0-9]* 2>/dev/null | head -1 || true)"
  [ -z "$d" ] && { echo 0; return; }
  find "$d" -path '*/episodes/*_summary_v2.json' 2>/dev/null | wc -l
}

# wait_runner_done — poll until no runner (returns 1 on 65h ceiling)
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
    && log "bind OK" || { log "bind FAILED (non-blocking, re-run on return)"; ntfy "⚠️ orch: bind failed after $1 — data intact, re-run --populate --apply manually"; }
}

proxy_up() {
  local ok=0 i
  for i in 1 2 3; do
    .venv/bin/python3 scripts/maintenance/probe_proxy_alive.py >/dev/null 2>&1 && ok=$((ok+1))
    sleep 20
  done
  [ "$ok" -ge 2 ]
}

# run_condition <label> <cfg_prefix> <cmd...>
# returns: 0 complete+bound / 1 abort (eps<205) / 2 launch failure / 3 timeout
run_condition() {
  local label="$1" prefix="$2"; shift 2
  if [ "$(runner_count)" != "0" ]; then
    log "$label: pre-check FAILED (runner already present)"; ntfy "🔴 orch: $label pre-check found a live runner — STOPPING"; return 2
  fi
  log "$label: launching: $*"
  if ! "$@" >> "${REPO}/logs/orch_queue_${label}.log" 2>&1; then
    log "$label: queue script exited non-zero"; ntfy "🔴 orch: $label queue launch FAILED (see logs/orch_queue_${label}.log)"; return 2
  fi
  sleep 120
  if [ "$(runner_count)" = "0" ]; then
    log "$label: runner not up 120s after queue OK"; ntfy "🔴 orch: $label runner did not start — STOPPING"; return 2
  fi
  ntfy "▶️ orch: $label launched, runner up"
  if ! wait_runner_done; then
    log "$label: 65h ceiling hit"; ntfy "🔴 orch: $label exceeded 65h ceiling, runner still alive — orchestrator exiting (runner left running)"; return 3
  fi
  local n; n="$(eps_count "$prefix")"
  if [ "$n" -ge "$SCORED" ]; then
    log "$label: COMPLETE ($n/$SCORED)"; do_bind "$label"; ntfy "✅ orch: $label COMPLETE $n/$SCORED + bind"; return 0
  fi
  log "$label: ABORT ($n/$SCORED)"; ntfy "🟠 orch: $label ABORT at $n/$SCORED (likely proxy/env) — applying skip policy"; return 1
}

log "=== orchestrator up (pid $$) ==="
ntfy "🤖 orch: 4-day boundary orchestrator armed on A100 (queue: B1som-wait → B0 psom-resume+pprompt if proxy UP → B1 vision/ptext/psom/pprompt → B2 ×6)"

# ---- Phase 0: wait for the in-flight B1 som reddit ----
if [ "$(runner_count)" != "0" ]; then
  log "phase0: waiting on in-flight B1 som"
  wait_runner_done || { ntfy "🔴 orch: B1 som exceeded 65h ceiling — exiting"; exit 1; }
fi
n="$(eps_count B1_som_reddit)"
if [ "$n" -lt "$SCORED" ]; then
  ntfy "🔴 orch: B1 som ended at $n/$SCORED (<205, non-proxy abort suspected) — STOPPING for manual triage"
  exit 1
fi
do_bind "B1_som"
ntfy "✅ orch: B1 som COMPLETE $n/$SCORED + bind — probing proxy for B0 slot"

# ---- Phase 1: B0 insertion (proxy-gated) ----
if proxy_up; then
  ntfy "🟢 orch: proxy UP → inserting B0 psom RESUME (146/205)"
  if run_condition "B0_psom_resume" "B0_phantom_som_reddit" \
      env FORCE_NEW=0 RESET_BEFORE=0 bash scripts/queues/queue_phantom_som.sh B0 reddit; then
    ntfy "ℹ️ orch: B0 psom done — task149 rerun landed, registry classify deferred to operator return"
    run_condition "B0_pprompt" "B0_phantom_prompt_reddit" \
      env FORCE_NEW=1 RESET_BEFORE=1 bash scripts/queues/queue_phantom_prompt.sh B0 reddit \
      || ntfy "🟠 orch: B0 pprompt did not complete — continuing to B1 chain (B0-skip policy)"
  else
    rc=$?
    [ "$rc" -ge 2 ] && { ntfy "🔴 orch: B0 psom launch/timeout failure (rc=$rc) — STOPPING"; exit 1; }
    ntfy "🟠 orch: B0 psom aborted — skipping B0 pprompt, continuing to B1 chain (B0-skip policy)"
  fi
else
  ntfy "🔘 orch: proxy DOWN at boundary → skipping B0, straight to B1 chain"
fi

# ---- Phase 2+3: B1 remaining 4 + B2 all 6 (fresh, abort → STOP) ----
run_seq() {
  local label="$1" prefix="$2"; shift 2
  run_condition "$label" "$prefix" "$@"
  local rc=$?
  if [ "$rc" -ne 0 ]; then
    ntfy "🔴 orch: $label failed (rc=$rc, non-proxy-immune baseline should not abort) — STOPPING queue here"
    exit 1
  fi
}
run_seq "B1_vision"  "B1_vision_reddit"          env FORCE_NEW=1 RESET_BEFORE=1 bash scripts/queues/queue_baseline.sh B1 vision reddit
run_seq "B1_ptext"   "B1_phantom_text_reddit"    env FORCE_NEW=1 RESET_BEFORE=1 bash scripts/queues/queue_phantom_text.sh B1 reddit
run_seq "B1_psom"    "B1_phantom_som_reddit"     env FORCE_NEW=1 RESET_BEFORE=1 bash scripts/queues/queue_phantom_som.sh B1 reddit
run_seq "B1_pprompt" "B1_phantom_prompt_reddit"  env FORCE_NEW=1 RESET_BEFORE=1 bash scripts/queues/queue_phantom_prompt.sh B1 reddit
run_seq "B2_dom"     "B2_dom_reddit"             env FORCE_NEW=1 RESET_BEFORE=1 bash scripts/queues/queue_baseline.sh B2 dom reddit
run_seq "B2_som"     "B2_som_reddit"             env FORCE_NEW=1 RESET_BEFORE=1 bash scripts/queues/queue_baseline.sh B2 som reddit
run_seq "B2_vision"  "B2_vision_reddit"          env FORCE_NEW=1 RESET_BEFORE=1 bash scripts/queues/queue_baseline.sh B2 vision reddit
run_seq "B2_ptext"   "B2_phantom_text_reddit"    env FORCE_NEW=1 RESET_BEFORE=1 bash scripts/queues/queue_phantom_text.sh B2 reddit
run_seq "B2_psom"    "B2_phantom_som_reddit"     env FORCE_NEW=1 RESET_BEFORE=1 bash scripts/queues/queue_phantom_som.sh B2 reddit
run_seq "B2_pprompt" "B2_phantom_prompt_reddit"  env FORCE_NEW=1 RESET_BEFORE=1 bash scripts/queues/queue_phantom_prompt.sh B2 reddit

ntfy "🏁 orch: FULL reddit queue drained (B0 psom+pprompt / B1 ×6 / B2 ×6 as launched) — nothing left to run"
log "=== orchestrator done ==="
