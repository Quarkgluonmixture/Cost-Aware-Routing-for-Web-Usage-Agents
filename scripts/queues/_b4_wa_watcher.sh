#!/usr/bin/env bash
# Waits for the 2026-08-09 shopping chain to finish, then runs the B4/WA-shop launcher.
#
# TWO conditions, both required (CLAUDE.md done-monitor tiers). PID alone is not enough:
# if the chain dies before its 4th segment (B1 phantom_som shopping) the PID also vanishes,
# and the launcher would fire on an incomplete chain. The Tier-1 file marker is the 4th
# segment's own completion sentinel.
set -uo pipefail
REPO="/home/ubuntu/workspace/p79"
cd "$REPO" || exit 1
CHAIN_PID="${1:?chain pid required}"
NTFY="${NTFY_TOPIC:-p79-exp-dgx-spark}"
LOG="logs/b4_wa_watcher_$(date -u +%Y%m%d_%H%M%S).log"

say() { echo "[watcher $(date -u '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
say "armed: waiting on chain pid ${CHAIN_PID} + phantom_som sentinel"

# 6-day ceiling: the chain's own ETA is ~08-16, so anything past that is a stall to look at.
for i in $(seq 1 1728); do   # 1728 x 300s = 6 days
  pid_gone=0; sentinel=0
  kill -0 "$CHAIN_PID" 2>/dev/null || pid_gone=1
  for f in results/visualwebarena/phase1/B1_phantom_som_shopping_*/*/condition_summary_v2.json; do
    [ -s "$f" ] && sentinel=1 && break
  done
  if [ "$pid_gone" = 1 ] && [ "$sentinel" = 1 ]; then
    say "both conditions met -> launching"
    curl -s -m 20 -H "Title: chain done, launching B4 smoke" \
      -d "chain pid gone + phantom_som sentinel present" "https://ntfy.sh/${NTFY}" >/dev/null 2>&1 || true
    bash "$REPO/scripts/queues/_launch_b4_smoke_and_wa_shop.sh" >> "$LOG" 2>&1
    say "launcher exited rc=$?"
    exit 0
  fi
  if [ "$pid_gone" = 1 ] && [ "$sentinel" = 0 ]; then
    say "chain pid gone but NO phantom_som sentinel -- chain likely aborted. Holding."
    curl -s -m 20 -H "Title: chain gone, sentinel MISSING" -H "Priority: high" \
      -d "watcher is holding; B4 launch NOT fired. Inspect the 4th chain segment." \
      "https://ntfy.sh/${NTFY}" >/dev/null 2>&1 || true
    exit 2
  fi
  sleep 300
done
say "6-day ceiling reached without both conditions -- giving up"
curl -s -m 20 -H "Title: b4 watcher timed out" -d "6 days elapsed" \
  "https://ntfy.sh/${NTFY}" >/dev/null 2>&1 || true
exit 1
