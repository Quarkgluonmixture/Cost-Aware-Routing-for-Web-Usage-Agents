#!/bin/bash
# notify_on_fail.sh — run a command, send ntfy alert if it fails.
#
# Usage:
#   notify_on_fail.sh "<title>" -- <command> [args...]
#
# Sends ntfy alert (priority=high) on non-zero exit code, including last
# 500 chars of combined stdout+stderr. Exit code is preserved.
#
# Topic: $NTFY_TOPIC (env) or default p79-exp-dgx-spark.
#
# Example crontab entry:
#   */10 * * * * cd $REPO && bash scripts/maintenance/notify_on_fail.sh \
#       "glm-update-cells" -- make glm-update-cells APPLY=1 \
#       >> logs/cron/glm_update_cells.log 2>&1

set -u

TOPIC="${NTFY_TOPIC:-p79-exp-dgx-spark}"

if [ "$#" -lt 3 ] || [ "$2" != "--" ]; then
  echo "Usage: notify_on_fail.sh \"<title>\" -- <command> [args...]" >&2
  exit 64
fi

TITLE="$1"
shift 2  # drop title + "--"

LOG=$(mktemp /tmp/notify_on_fail.XXXXXX)
"$@" > "$LOG" 2>&1
RC=$?

if [ "$RC" -ne 0 ]; then
  # B-851 (A1.15b Chunk γ P2-4): backtick escape was dead code. curl `-d`
  # passes body as POST data without shell interpretation; backticks need
  # no escaping. Pre-fix `sed 's/\`/\\\`/g'` was no-op + risk of breaking
  # genuine backtick content. Just use raw tail output.
  TAIL_OUT=$(tail -c 500 "$LOG")
  BODY="❌ P79 cron failed: $TITLE
Exit: $RC
$(date)
---
$TAIL_OUT"

  curl -s --max-time 10 \
    -H "Title: ⚠️ P79 cron fail: $TITLE" \
    -H "Priority: high" \
    -H "Tags: warning,gear" \
    -d "$BODY" \
    "https://ntfy.sh/$TOPIC" > /dev/null 2>&1 || true
fi

cat "$LOG"
rm -f "$LOG"
exit "$RC"
