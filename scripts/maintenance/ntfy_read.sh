#!/usr/bin/env bash
# ntfy_read.sh — pull P79 ntfy notifications so Claude / operator can READ them
# without manual copy-paste (the monitors only push; nothing pulled them back).
#
# ntfy.sh is a pub/sub: the topic name is the read+write credential. The same
# topic the watchdogs `curl -d` INTO can be polled OUT of. ntfy.sh caches
# messages ~12h on the public instance; `poll=1` returns the cache and closes
# the connection immediately (NOT streaming → safe for a one-shot command).
#
# Reads MULTIPLE active topics by default and MERGES by time, so no channel is
# missed. Active topics (verified 2026-05-22):
#   - p79-exp-dgx-spark  experiment lifecycle + alerts (watchdog / paper_grade /
#                        fire6 / health / GLM crons / canary monitors)  [tag: exp]
#   - p79-claude         delete / cleanup ops (sync_a100 --delete-after
#                        propagation, clear_tasks)                      [tag: cln]
# p79-jiaming is RETIRED (0 traffic, not in repo/cron) — not polled.
#
# Usage:
#   bash scripts/maintenance/ntfy_read.sh               # last 12h, BOTH topics, all
#   bash scripts/maintenance/ntfy_read.sh 1h            # last 1h
#   bash scripts/maintenance/ntfy_read.sh 6h alerts     # alerts only (excl routine progress)
#   SINCE=30m bash ...                                  # env form
#   TOPIC=p79-claude bash ...                           # single-topic override
#   TOPICS="a b c" bash ...                             # custom topic list
set -uo pipefail
SINCE="${1:-${SINCE:-12h}}"
FILTER="${2:-all}"                                          # all | alerts
TOPICS="${TOPIC:-${TOPICS:-p79-exp-dgx-spark p79-claude}}"  # TOPIC= forces single; else multi

{ for t in $TOPICS; do curl -s -m 20 "https://ntfy.sh/${t}/json?poll=1&since=${SINCE}"; done; } \
 | FILTER="$FILTER" TOPICS="$TOPICS" SINCE="$SINCE" python3 -c '
import sys, json, os
from datetime import datetime, timezone

flt = os.environ.get("FILTER", "all")
# anomaly markers — high-priority / failure-class. NOT bare "fail" (routine
# "P79 Status" bodies say "task N: fail" every tick → excluded by title).
ALERT_KW = ("ALERT", "🔴", "⚠️", "FATAL", "ABORT", "GONE", "RUNNER_GONE",
            "health alert", "OOM", "exhaust", "cron fail", "stall",
            "delet", "propagat")          # delet → deleted/deletion (p79-claude)
TAG = {"p79-exp-dgx-spark": "exp", "p79-claude": "cln"}

rows = []  # (epoch, line) — collect across topics then sort by time
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        d = json.loads(line)
    except Exception:
        continue
    if d.get("event") != "message":
        continue
    title = d.get("title", "-") or "-"
    msg = (d.get("message", "") or "").replace("\n", " ")
    if flt == "alerts":
        if title == "P79 Status":            # routine progress — never an alert
            continue
        prio = d.get("priority", 3)          # ntfy: high=4 urgent=5 default=3
        if prio < 4 and not any(k in (title + " " + msg) for k in ALERT_KW):
            continue
    topic = d.get("topic", "?")
    tag = TAG.get(topic, topic[:8])
    epoch = d.get("time", 0)
    t = datetime.fromtimestamp(epoch, timezone.utc).strftime("%m-%d %H:%M")
    rows.append((epoch, f"{t}Z {tag:>3}  [{title}]  {msg[:170]}"))

rows.sort(key=lambda r: r[0])                  # merge multiple topics by time
since = os.environ.get("SINCE")
topics = os.environ.get("TOPICS")
hdr = f"ntfy [{topics}] | since={since} | filter={flt} | {len(rows)} msgs"
print(hdr)
print("-" * min(len(hdr), 100))
print("\n".join(r[1] for r in rows) if rows else "(none)")
'
