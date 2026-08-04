#!/usr/bin/env bash
# Keep the canonical mechanistic sweep alive across silent deaths. — 2026-07-27
#
# WHY: the 2026-07-24 launch died at 11:51 mid-cell after finishing 1 of 24 —
# no traceback, no exit line, no OOM message in the log, just a stop. On a
# shared GPU box that is the expected failure mode (an outside job arrives, the
# kernel reaps yours), and the sweep script itself has no restart: it runs the
# cell loop once and exits. Three days of the deadline window were lost to a
# failure nobody was watching for.
#
# The sweep is already resumable by design — a cell whose output dir holds
# `pilot_summary.md` is skipped — so a restart is safe and cheap. All this
# supervisor adds is noticing.
#
# Liveness is PID-based (`kill -0` on a recorded pid), never `pgrep -f` on a
# pattern that appears in this file: a pattern check would match this
# supervisor's own command line and report the sweep as alive forever.
# (CLAUDE.md done-monitor rule; the 2026-05-09 codex monitor burned 7h on it.)
#
# Usage:
#   setsid nohup bash scripts/queues/supervise_mechanistic_canonical.sh \
#     > logs/mechanistic_canonical/supervisor.log 2>&1 < /dev/null &
# Env:
#   DEADLINE        passed through to the sweep (default = sweep's own 2026-08-01)
#   POLL_SECONDS    liveness poll interval (default 300)
#   MAX_RESTARTS    give up after this many restarts (default 40)
#   MIN_UPTIME_SEC  a run that dies faster than this counts as a hard failure
#                   (default 600) — protects against restart-storming a run that
#                   cannot start at all, e.g. GPU fully occupied
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1

OUT_ROOT="$REPO/results/mechanistic/canonical"
LOGDIR="$REPO/logs/mechanistic_canonical"
PIDFILE="$LOGDIR/.sweep.pid"
NTFY_TOPIC="${NTFY_TOPIC:-p79-claude}"
POLL_SECONDS="${POLL_SECONDS:-300}"
MAX_RESTARTS="${MAX_RESTARTS:-40}"
MIN_UPTIME_SEC="${MIN_UPTIME_SEC:-600}"
mkdir -p "$LOGDIR"

log() { echo "[$(date '+%F %H:%M:%S')] $*"; }
notify() {
  curl -s -m 20 -H "Title: mechanistic sweep" -d "$1" \
    "https://ntfy.sh/${NTFY_TOPIC}" > /dev/null 2>&1 || true
}

n_done() { ls -d "$OUT_ROOT"/*/ 2>/dev/null | while read -r d; do
             [ -f "$d/pilot_summary.md" ] && echo x; done | wc -l; }

restarts=0
consecutive_fast_deaths=0

log "supervisor up — poll ${POLL_SECONDS}s, max ${MAX_RESTARTS} restarts, cells done=$(n_done)/24"

while true; do
  if [ -f "$OUT_ROOT/.SWEEP_DONE" ]; then
    log "sweep reports DONE (marker present) — cells done=$(n_done)/24; supervisor exiting"
    notify "mechanistic sweep finished: $(n_done)/24 cells (supervisor exit, ${restarts} restarts)"
    exit 0
  fi

  # Alive? PID-based only.
  if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null; then
    sleep "$POLL_SECONDS"
    continue
  fi

  if [ "$restarts" -ge "$MAX_RESTARTS" ]; then
    log "restart budget exhausted (${MAX_RESTARTS}) — giving up at $(n_done)/24"
    notify "mechanistic sweep GAVE UP after ${MAX_RESTARTS} restarts at $(n_done)/24 cells"
    exit 1
  fi

  before=$(n_done)
  started=$(date +%s)
  ts=$(date +%Y%m%d_%H%M%S)
  restarts=$((restarts + 1))
  log "sweep not running — restart #${restarts} (cells done=${before}/24)"

  setsid nohup bash scripts/queues/queue_mechanistic_canonical.sh \
    > "$LOGDIR/sweep_supervised_${ts}.log" 2>&1 < /dev/null &
  echo $! > "$PIDFILE"
  sleep 30
  # The launcher backgrounds nothing itself, so the recorded pid is the sweep.
  if ! kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    elapsed=$(( $(date +%s) - started ))
    consecutive_fast_deaths=$((consecutive_fast_deaths + 1))
    log "restart #${restarts} died in ${elapsed}s (consecutive fast deaths: ${consecutive_fast_deaths})"
    tail -5 "$LOGDIR/sweep_supervised_${ts}.log" 2>/dev/null | sed 's/^/    /'
    if [ "$consecutive_fast_deaths" -ge 3 ]; then
      log "3 consecutive sub-${MIN_UPTIME_SEC}s deaths — this is not a transient reap; stopping"
      notify "mechanistic sweep: 3 immediate failures in a row, supervisor stopping at ${before}/24. Check GPU availability."
      exit 1
    fi
    sleep 600
    continue
  fi
  consecutive_fast_deaths=0
  notify "mechanistic sweep restarted (#${restarts}) at ${before}/24 cells"
  sleep "$POLL_SECONDS"
done
