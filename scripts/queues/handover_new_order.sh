#!/usr/bin/env bash
# One-shot cutover to the reordered / deadline-free queue. — 2026-07-30
#
# WHY: queue_mechanistic_canonical.sh was replaced on disk (new cell order, no
# DEADLINE), but the queue process running right now (pgid 38603) loaded the OLD
# file into memory and still holds its deleted inode. It would keep the old order
# and would still hit the 2026-08-01 deadline. Replacing the file cannot change a
# process that already read it — only a restart can, and the supervisor restarts
# the sweep automatically whenever it finds the recorded pid dead.
#
# So: wait for the in-flight cell (p2_psom_ptext_red, 20h in, 19/24 tasks) to
# finish, then kill the old queue so the supervisor picks up the new file.
# Killing it any earlier would throw that cell away.
#
# The kill targets the PROCESS GROUP (-PGID): at the moment the summary appears
# the queue has already launched the next cell, and killing only the bash parent
# would orphan that python — the supervisor would then start a SECOND queue that
# re-launches the same cell into the same output dir. Two writers, corrupt cell.
# The 15s poll bounds the loss: a cell that young is still loading the model.
# The supervisor (pgid 38587) is a different group and is not touched.
set -u
REPO=/home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
MARKER="$REPO/results/mechanistic/canonical/p2_psom_ptext_red/pilot_summary.md"
PGID=38603
DEADLINE_TS=$(( $(date +%s) + 12*3600 ))   # give up rather than linger forever

log() { echo "[$(date "+%F %H:%M:%S")] $*"; }
log "handover watcher up — waiting for $(basename "$(dirname "$MARKER")") to finish (pgid $PGID)"

while [ ! -f "$MARKER" ]; do
  if ! kill -0 "-$PGID" 2>/dev/null; then
    log "queue pgid $PGID already gone — supervisor will restart on the new file by itself; nothing to do"
    exit 0
  fi
  if [ "$(date +%s)" -ge "$DEADLINE_TS" ]; then
    log "12h elapsed with no summary — giving up, cutover NOT performed (cell may be stuck; check manually)"
    exit 1
  fi
  sleep 15
done

log "p2_psom_ptext_red finished — cutting over"
sleep 2
kill -TERM "-$PGID" 2>/dev/null
sleep 5
if kill -0 "-$PGID" 2>/dev/null; then
  log "TERM did not take, sending KILL"
  kill -KILL "-$PGID" 2>/dev/null
  sleep 3
fi
if kill -0 "-$PGID" 2>/dev/null; then
  log "ERROR: pgid $PGID still alive after KILL — cutover FAILED, investigate"
  exit 1
fi
log "old queue down. supervisor polls every 300s and will relaunch from the new script."
log "expected next cell: p2_taskshuf_cls (new order, no deadline)"
