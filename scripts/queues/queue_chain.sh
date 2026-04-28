#!/usr/bin/env bash
# queue_chain.sh — Sequentially launch a list of queue commands, waiting for
# each runner to complete before launching the next. Useful for chaining cells
# that share a single GPU instance (B1 4B local) or any paper-grade sequence.
#
# Each queued command goes through queue_baseline.sh / queue_phantom_som.sh /
# queue_phantom_dom.sh which already handle reset+auth_refresh+watchdog
# launch+idempotent skip. This chain ALWAYS exports RESET_BEFORE=1 by default
# (paper-grade — every cell starts from a fresh post-reset site state); pass
# --no-reset to disable (rare, e.g. resume-only chain).
#
# Usage:
#   nohup bash scripts/queues/queue_chain.sh [--no-reset] \
#     "<cmd1>" "<cmd2>" ... \
#     > logs/queue_chain_<label>.log 2>&1 &
#
# Each <cmd> is a queue script invocation, relative to scripts/queues/:
#   "queue_phantom_som.sh B1 classifieds"
#   "queue_phantom_dom.sh B1 reddit"
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
#     "queue_phantom_dom.sh B1 classifieds" \
#     "queue_phantom_dom.sh B1 reddit" \
#     > logs/queue_chain_b1_phantom.log 2>&1 &
#
#   # B0 phantom shopping pair (after B0 dom shopping done):
#   nohup bash scripts/queues/queue_chain.sh \
#     "queue_phantom_som.sh B0 shopping" \
#     "queue_phantom_dom.sh B0 shopping" \
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
wait_for_runner_done() {
  local pattern="$1"
  local label="$2"
  local elapsed=0
  while pgrep -f "run_experiment.py.*${pattern}" > /dev/null; do
    sleep 60
    elapsed=$((elapsed + 60))
    if (( elapsed % 1800 == 0 )); then
      log "  ${label}: still running (${elapsed}s elapsed)..."
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

  # Launch via the queue script (idempotent — picks up existing or fresh+reset)
  out=$(RESET_BEFORE="${RESET_FLAG}" bash "${SCRIPT_DIR}/${script_name}" \
        ${cmd#${script_name} } 2>&1 || true)
  echo "$out" | sed 's/^/    /'

  # Extract run_id from queue script output
  run_id=$(echo "$out" | grep -oP 'run_id=\K\S+' | tail -1)
  if [[ -z "$run_id" ]]; then
    log "  [error] could not extract run_id from queue script output, aborting"
    exit 1
  fi
  log "  watching run_id=${run_id}"

  wait_for_runner_done "$run_id" "[${idx}/$#] $cmd"
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
