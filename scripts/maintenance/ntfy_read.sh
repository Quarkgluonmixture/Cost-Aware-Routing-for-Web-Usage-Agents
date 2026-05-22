#!/usr/bin/env bash
# ntfy_read.sh — pull P79 ntfy notifications so Claude / operator can READ them
# without manual copy-paste (the monitors only push; nothing pulled them back).
#
# ntfy.sh is a pub/sub: the topic name is the read+write credential. The same
# topic the watchdogs `curl -d` INTO can be polled OUT of. ntfy.sh caches
# messages ~12h on the public instance; `poll=1` returns the cache and closes
# the connection immediately (NOT streaming → safe for a one-shot command).
#
# Usage:
#   bash scripts/maintenance/ntfy_read.sh                 # last 12h, all msgs
#   bash scripts/maintenance/ntfy_read.sh 1h              # last 1h
#   bash scripts/maintenance/ntfy_read.sh 6h alerts       # last 6h, ALERT/🔴/fail only
#   SINCE=30m bash scripts/maintenance/ntfy_read.sh       # env form
#   TOPIC=other-topic bash scripts/maintenance/ntfy_read.sh
set -uo pipefail
SINCE="${1:-${SINCE:-12h}}"
FILTER="${2:-all}"                       # all | alerts
TOPIC="${TOPIC:-p79-exp-dgx-spark}"

curl -s -m 20 "https://ntfy.sh/${TOPIC}/json?poll=1&since=${SINCE}" \
 | FILTER="$FILTER" TOPIC="$TOPIC" SINCE="$SINCE" python3 -c '
import sys, json, os
from datetime import datetime, timezone

flt = os.environ.get("FILTER", "all")
# anomaly markers — high-priority / failure-class notifications worth isolating.
# NOTE: do NOT use bare "fail" — routine "P79 Status" progress bodies say
# "task N: fail" every tick; we exclude that title outright and match only
# title-level / strong message markers.
ALERT_KW = ("ALERT", "🔴", "⚠️", "FATAL", "ABORT", "GONE", "RUNNER_GONE",
            "health alert", "OOM", "exhaust", "cron fail", "stall")

rows = []
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
        if not any(k in (title + " " + msg) for k in ALERT_KW):
            continue
    t = datetime.fromtimestamp(d["time"], timezone.utc).strftime("%m-%d %H:%M")
    rows.append(f"{t}Z  [{title}]  {msg[:180]}")

topic = os.environ.get("TOPIC")
since = os.environ.get("SINCE")
hdr = f"ntfy {topic} | since={since} | filter={flt} | {len(rows)} msgs"
print(hdr)
print("-" * min(len(hdr), 100))
print("\n".join(rows) if rows else "(none)")
'
