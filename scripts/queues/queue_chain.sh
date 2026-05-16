#!/usr/bin/env bash
# queue_chain.sh — Sequentially launch a list of queue commands, waiting for
# each runner to complete before launching the next. Useful for chaining cells
# that share a single GPU instance (B1 4B local) or any paper-grade sequence.
#
# Each queued command goes through queue_baseline.sh / queue_phantom_som.sh /
# queue_phantom_text.sh which already handle reset+auth_refresh+watchdog
# launch+idempotent skip. This chain ALWAYS exports RESET_BEFORE=1 by default
# (paper-grade — every cell starts from a fresh post-reset site state); pass
# --no-reset to disable (rare, e.g. resume-only chain).
# Note: queue_phantom_dom.sh exists as a back-compat symlink to queue_phantom_text.sh.
#
# Usage:
#   nohup bash scripts/queues/queue_chain.sh [--no-reset] \
#     "<cmd1>" "<cmd2>" ... \
#     > logs/queue_chain_<label>.log 2>&1 &
#
# Each <cmd> is a queue script invocation, relative to scripts/queues/:
#   "queue_phantom_som.sh B1 classifieds"
#   "queue_phantom_text.sh B1 reddit"
#   "queue_baseline.sh B0 dom shopping"
#   "queue_baseline.sh B0 som shopping wa"
#
# The chain auto-detects an already-running cell (queue scripts are idempotent;
# RESET is skipped when a runner is already attached). For the FIRST queued
# cell — if it's already running, chain just waits for completion and proceeds
# to the next.
#
# Examples:
#   # B1 phantom 4-cell chain (cls already running):
#   nohup bash scripts/queues/queue_chain.sh \
#     "queue_phantom_som.sh B1 classifieds" \
#     "queue_phantom_som.sh B1 reddit" \
#     "queue_phantom_text.sh B1 classifieds" \
#     "queue_phantom_text.sh B1 reddit" \
#     > logs/queue_chain_b1_phantom.log 2>&1 &
#
#   # B0 phantom shopping pair (after B0 dom shopping done):
#   nohup bash scripts/queues/queue_chain.sh \
#     "queue_phantom_som.sh B0 shopping" \
#     "queue_phantom_text.sh B0 shopping" \
#     > logs/queue_chain_b0_phantom_shop.log 2>&1 &

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_DIR}"

log() { echo "[chain $(date '+%H:%M:%S')] $*"; }

# ---------- arg parsing ----------
RESET_FLAG=1
if [[ "${1:-}" == "--no-reset" ]]; then
  RESET_FLAG=0
  shift
fi

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 [--no-reset] <queue_command_1> [<queue_command_2> ...]" >&2
  echo "  Each command: 'queue_<name>.sh <args>' (relative to scripts/queues/)" >&2
  echo "  See header for examples." >&2
  exit 2
fi

# ---------- helpers ----------
# wait_for_runner_done blocks until the runner exits cleanly OR aborts the chain
# if the watchdog dies mid-run (A1.13 P0-4, 2026-05-16).
#
# Watchdog liveness invariant: queue_baseline.sh:288-296 declares watchdog FATAL
# at launch ("paper-grade launch requires watchdog (auth refresh + auto-clean)").
# Pre-fix wait loop only watched runner PID — watchdog death mid-run was silent,
# losing reactive auth refresh + idle alerts + auto-clean for the rest of the run
# (multi-day chain weekend runs at highest risk). Q2 A decision: abort chain on
# watchdog death — paper-grade > compute reclaim. User triggers manual restart
# after addressing watchdog root cause (typical: ntfy curl SIGPIPE, OOM, glm
# config bug, NPE on bad state JSON).
wait_for_runner_done() {
  local pattern="$1"
  local label="$2"
  local elapsed=0
  while pgrep -f "run_experiment.py.*${pattern}" > /dev/null; do
    sleep 60
    elapsed=$((elapsed + 60))
    # A1.13 P0-4: watchdog liveness check
    if ! pgrep -f "experiment_watchdog.*${pattern}" > /dev/null; then
      log "  [FATAL] ${label}: watchdog died mid-run (runner still alive after ${elapsed}s)"
      log "  paper-grade contamination risk: no reactive auth refresh + no auto-clean + no idle alert"
      log "  Q2 A decision (A1.13 audit 2026-05-16): abort chain, kill runner, notify user"
      pkill -f "run_experiment.py.*${pattern}" 2>/dev/null || true
      if command -v curl > /dev/null; then
        curl -d "queue_chain ABORT (${label}): watchdog died after ${elapsed}s; runner killed. Restart after root cause." \
          "ntfy.sh/${NTFY_TOPIC:-p79-exp-dgx-spark}" 2>/dev/null || true
      fi
      exit 1
    fi
    if (( elapsed % 1800 == 0 )); then
      log "  ${label}: still running (${elapsed}s elapsed; watchdog alive)..."
      pgrep -af "run_experiment.py.*${pattern}" | head -1 | sed 's/^/    /'
    fi
  done
  log "  ${label}: runner done"
}

# ---------- chain ----------
log "=================================================="
log "queue_chain — $# cells (RESET_BEFORE=${RESET_FLAG})"
for arg in "$@"; do log "  - $arg"; done
log "=================================================="

idx=0
for cmd in "$@"; do
  idx=$((idx + 1))
  log ""
  log "------ [${idx}/$#] ${cmd} ------"

  # Validate the script exists (cmd is "queue_xxx.sh args...")
  script_name="${cmd%% *}"
  if [[ ! -f "${SCRIPT_DIR}/${script_name}" ]]; then
    log "  [error] script not found: ${SCRIPT_DIR}/${script_name}"
    log "  aborting chain"
    exit 1
  fi

  # ---- Same-site collision check (paper-grade hard rule §106) ----
  # Parse <baseline> + <site> from the queue command args.
  # queue_baseline.sh format: <baseline> <mode> <site> [benchmark]
  # queue_phantom_*.sh format: <baseline> <site> [benchmark]
  # Hard rule: a site's docker container + login account is shared, so only
  # ONE baseline (of B0 / B1 / B2) may run on a site at a time. Generalised
  # from the original B0-vs-B1-only check — with Gemma3-VL (B2) the pairwise
  # single-"other_baseline" logic silently missed the third baseline.
  cmd_args=( ${cmd} )
  this_baseline="${cmd_args[1]:-}"  # B0 / B1 / B2
  if [[ "${script_name}" == queue_baseline.sh ]]; then
    this_site="${cmd_args[3]:-}"    # 4th token (script bash mode site)
  else
    this_site="${cmd_args[2]:-}"    # 3rd token (script bash site)
  fi
  if [[ -n "${this_baseline}" && -n "${this_site}" ]]; then
    # Cross-baseline collision (different baseline on same site)
    for other_baseline in B0 B1 B2; do
      [[ "${other_baseline}" == "${this_baseline}" ]] && continue
      if pgrep -f "run_experiment.*${other_baseline}_.*_${this_site}_" > /dev/null 2>&1; then
        log "  [collision] ${other_baseline} runner already active on site=${this_site}"
        log "  paper-grade hard rule: only one baseline may run on a site at a time"
        log "  waiting for ${other_baseline} ${this_site} to finish before launching ${this_baseline}..."
        while pgrep -f "run_experiment.*${other_baseline}_.*_${this_site}_" > /dev/null 2>&1; do
          sleep 60
        done
        log "  ${other_baseline} ${this_site} finished; proceeding with ${this_baseline}"
      fi
    done

    # B-129 fix 2026-05-15 (codex Mode B P1-3): same-baseline + same-site
    # mode collision check (e.g. B0_dom_reddit + B0_som_reddit launched
    # concurrently outside the master gate). Same reddit user account =
    # RESET_BEFORE between modes wipes the other's session. Master gate
    # `queue_phase1_paper_grade.sh` already blocks any active run before
    # launching new chains, so this check defends against manual bypass.
    # Skip our own PID to avoid self-match (we're not yet calling
    # run_experiment from this script — only orchestrating queue scripts).
    if pgrep -f "run_experiment.*${this_baseline}_.*_${this_site}_" > /dev/null 2>&1; then
      log "  [collision] same-baseline ${this_baseline} runner already active on site=${this_site} (different mode)"
      log "  paper-grade hard rule: same baseline same site = shared docker user account + RESET_BEFORE race"
      log "  waiting for existing ${this_baseline} ${this_site} run to finish before launching new mode..."
      while pgrep -f "run_experiment.*${this_baseline}_.*_${this_site}_" > /dev/null 2>&1; do
        sleep 60
      done
      log "  same-baseline ${this_baseline} ${this_site} finished; proceeding"
    fi
  fi

  # Launch via the queue script (idempotent — picks up existing or fresh+reset).
  # FORCE_NEW propagated explicitly (codex stress v6 C1) — paper-grade master chain
  # exports FORCE_NEW=1 so each cell gets a fresh timestamped run_id, never resumes
  # a pre-fix archived dir.
  #
  # B-301 (A1.17 P1-4, codex OOB unique): pre-fix `out=$(... || true)` discarded
  # queue script rc; reset/auth gate failure was hidden because run_id had been
  # printed before reset → chain proceeded to wait_for_runner_done finding no
  # runner → declared "done" instantly → fell through to silent sentinel check.
  # New behavior: capture rc explicitly; nonzero rc + no run_id printed → fatal;
  # nonzero rc + run_id printed (idempotent-skip case where queue script already
  # had complete data) → continue (legacy path).
  set +e
  out=$(FORCE_NEW="${FORCE_NEW:-0}" RESET_BEFORE="${RESET_FLAG}" bash "${SCRIPT_DIR}/${script_name}" \
        ${cmd#${script_name} } 2>&1)
  queue_rc=$?
  set -e
  echo "$out" | sed 's/^/    /'

  # Extract run_id + condition_id from queue script output
  run_id=$(echo "$out" | grep -oP 'run_id=\K\S+' | tail -1)
  cond_id=$(echo "$out" | grep -oP 'condition=\K\S+' | tail -1)

  # B-301 P1-4: nonzero queue rc + no run_id minted = reset/auth FATAL or
  # arg-parse error. Surface + abort. Nonzero rc + run_id minted = legacy
  # idempotent-skip with stale run_dir; allow through but warn.
  if [[ "${queue_rc}" != "0" ]]; then
    if [[ -z "${run_id}" ]]; then
      log "  [FATAL] queue script rc=${queue_rc}, no run_id minted — reset/auth/arg error"
      log "  full output above; aborting chain"
      exit 1
    fi
    log "  [warn] queue script rc=${queue_rc} but run_id=${run_id} minted (idempotent-skip path?)"
  fi

  if [[ -z "$run_id" ]]; then
    log "  [error] could not extract run_id from queue script output, aborting"
    exit 1
  fi
  if [[ -z "$cond_id" ]]; then
    log "  [error] could not extract condition id from queue script output, aborting"
    exit 1
  fi
  log "  watching run_id=${run_id} condition=${cond_id}"

  wait_for_runner_done "$run_id" "[${idx}/$#] $cmd"

  # ---- C3 completion sentinel (codex stress v6, 2026-05-14; A1.13 P0-3 upgraded 2026-05-16) ----
  # Runner process gone != success. A mid-run crash also makes pgrep empty.
  # File-presence alone was insufficient: same-second FORCE_NEW collision, mid-write
  # SIGKILL, or stale prior-run summary can all pass a `-f` check yet contain
  # 0-byte / truncated JSON / wrong-condition data. A1.13 P0-3 (3-AI overlap):
  # validate (a) file non-empty, (b) JSON parsable, (c) condition_id matches,
  # (d) total_tasks > 0 before accepting cell completion.
  # B-302 (A1.17 P0-4, codex OOB unique LAUNCH BLOCKER): pre-fix sentinel queried
  # `total_tasks / num_tasks / scored_task_count` — none exist in actual
  # condition_summary_v2.json schema (verified empirically 2026-05-16 on 5 sample
  # summaries: top-level has `episodes: int` and `condition_id`, NOT total_tasks).
  # Pre-fix every cell completion failed validation → chain aborted after cell 1.
  # New: use `episodes` field (the canonical count) + compare against expected_n
  # per site (sources of truth: launch.sh:67-70 SITE_N table); accept ≥90%
  # completion as valid (allows interrupt+resume partial cells), reject below.
  declare -A SITE_EXPECTED_N=(
    [classifieds]=234 [reddit]=210 [shopping]=466
    [wa_shopping]=192 [wa_shopping_admin]=182 [wa_reddit]=106
  )
  # Extract site from cond_id (formats like `phase1_dom_router_0` won't have site;
  # but full run_id pattern is e.g. `B0_dom_classifieds_20260516_...`)
  expected_n=0
  for site_key in classifieds reddit shopping wa_shopping wa_shopping_admin wa_reddit; do
    if [[ "${run_id}" == *"_${site_key}_"* ]]; then
      expected_n="${SITE_EXPECTED_N[${site_key}]}"; break
    fi
  done

  summary_found=""
  for base in results/visualwebarena/phase1 results/webarena/phase1; do
    cand="${REPO_DIR}/${base}/${run_id}/${cond_id}/condition_summary_v2.json"
    if [[ -s "${cand}" ]]; then
      if python3 -c "
import json, sys
try:
    d = json.load(open('${cand}'))
except Exception as e:
    print(f'invalid JSON: {e}', file=sys.stderr); sys.exit(1)
cid = d.get('condition_id', '')
if cid and cid != '${cond_id}':
    print(f'condition_id mismatch: got {cid!r}, expected ${cond_id!r}', file=sys.stderr); sys.exit(2)
# B-302 (A1.17 P0-4): canonical field is 'episodes' (int count); legacy
# fallbacks kept for forward-compat if schema ever rev'd.
ep = d.get('episodes', d.get('total_tasks', d.get('num_tasks', d.get('scored_task_count', 0))))
if not isinstance(ep, int) or ep <= 0:
    print(f'episodes invalid: {ep!r}', file=sys.stderr); sys.exit(3)
expected = ${expected_n:-0}
if expected > 0 and ep < expected * 0.9:
    print(f'partial completion {ep}/{expected} = {ep/expected:.1%} < 90% threshold', file=sys.stderr); sys.exit(4)
sys.exit(0)
" 2>/tmp/queue_chain_sentinel_err; then
        summary_found="${cand}"; break
      else
        err="$(cat /tmp/queue_chain_sentinel_err 2>/dev/null || true)"
        log "  [error] ${cand} present but FAILED validation: ${err}"
        log "  treating as missing summary — paper-grade aborts on invalid content"
      fi
    fi
  done
  if [[ -z "${summary_found}" ]]; then
    log "  [error] ${run_id}/${cond_id} — no valid condition_summary_v2.json"
    log "  failure modes: runner crash mid-write / FORCE_NEW same-second collision /"
    log "                 schema-version mismatch / stale prior-run dir / disk full"
    log "  aborting chain to prevent silent partial-data advancement"
    if command -v curl > /dev/null; then
      curl -d "queue_chain ABORT: ${run_id}/${cond_id} sentinel validation failed" \
        "ntfy.sh/${NTFY_TOPIC:-p79-exp-dgx-spark}" 2>/dev/null || true
    fi
    exit 1
  fi
  log "  ${run_id}: completion sentinel OK (${summary_found})"
done

log ""
log "=================================================="
log "queue_chain done — $# cells complete"
log "=================================================="

# ntfy notify
if command -v curl > /dev/null; then
  curl -d "queue_chain done: $# cells (${*})" \
    "ntfy.sh/${NTFY_TOPIC:-p79-exp-dgx-spark}" 2>/dev/null || true
fi
