#!/usr/bin/env bash
# paper_grade_check.sh — scheduled paper-grade integrity check of the A100 Phase-1a fire.
#
# DGX cron (crontab.txt). SSHes the A100, runs the deterministic validators where the
# data lives, ntfy's a one-line verdict + logs full output. NO Claude tokens.
#   - paper_grade_check.py    : per-condition (episodes==scored / parse_err / noise /
#                               som·vision IMAGE PRESENCE [B-1832/1835 regression class] /
#                               B0 cost coverage) + in-progress run health
#   - validate_fire_manifest.py : ghost / over-complete / == binding (B-1825/B-1834)
#
# Verdict ntfy'd every run (✅ ok / 🔴 issues). Tune cadence in crontab.txt.
# (set 2026-05-22, user request — watch paper-grade quality while away)
set -uo pipefail

TOPIC="${NTFY_TOPIC:-p79-exp-dgx-spark}"
A100="condense-a100"
REMOTE="/home/ubuntu/workspace/p79"
TS="$(date '+%Y-%m-%d %H:%M')"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LOG="${REPO_ROOT}/logs/cron/paper_grade_check.log"
mkdir -p "$(dirname "$LOG")"

# Host guard (2026-07-03): running this wrapper ON the A100 itself used to
# `ssh condense-a100` (alias only resolvable from DGX) → rc=255 → pushed a
# false "could NOT reach A100" alert (2 real incidents 2026-07-03 08:35Z).
# On the A100, run the validators locally instead of ssh-ing to ourselves.
if [ "$(hostname)" = "a100-jiaming-test" ]; then
  OUT="$(cd "${REMOTE}" && \
    .venv/bin/python3 scripts/maintenance/paper_grade_check.py 2>&1; \
    echo '===MANIFEST==='; \
    .venv/bin/python3 scripts/analysis/validate_fire_manifest.py 2>&1 | tail -3)"
  SSH_RC=$?
else
  OUT="$(timeout 180 ssh "$A100" "cd ${REMOTE} && \
    .venv/bin/python3 scripts/maintenance/paper_grade_check.py 2>&1; \
    echo '===MANIFEST==='; \
    .venv/bin/python3 scripts/analysis/validate_fire_manifest.py 2>&1 | tail -3" 2>&1)"
  SSH_RC=$?
fi

{
  echo "===================================================================="
  echo "[$TS] paper-grade check (ssh_rc=$SSH_RC)"
  echo "$OUT"
} >> "$LOG"

VERDICT="$(printf '%s\n' "$OUT" | grep -m1 'VERDICT:')"

if [ "$SSH_RC" -ne 0 ] && [ -z "$VERDICT" ]; then
  curl -s -d "⚠️ PAPER-GRADE [$TS]: check could NOT reach A100 (ssh_rc=$SSH_RC) — verify chain / cert" "ntfy.sh/${TOPIC}" >/dev/null 2>&1
  exit 0
fi

# issue signals from EITHER validator
if printf '%s\n' "$OUT" | grep -qiE 'VERDICT: ISSUES|VERDICT: FAIL|\[FAIL\]|GHOST run|OVER-COMPLETE|regression'; then
  DETAIL="$(printf '%s\n' "$OUT" | grep -iE 'ISSUE:|GHOST|OVER-COMPLETE|\[FAIL\]' | head -3 | tr '\n' ' ')"
  MSG="🔴 PAPER-GRADE [$TS]: ${VERDICT} | ${DETAIL}"
  PRIO="high"
else
  MSG="✅ PAPER-GRADE [$TS]: ${VERDICT:-clean}"
  PRIO="default"
fi

# Edge-triggered push (2026-06-11, user request): pre-fix this re-pushed an
# identical verdict every 6h (stable ISSUES list = spam; fire death/progress
# are owned by fire6_monitor / watchdog channels). Push ONLY when the verdict
# signature changes; full output still lands in $LOG every run (next_steps §0
# restricted-session live source). Signature strips the timestamp and rolling
# counters (ep=/img=) so episode progress alone doesn't re-trigger, while
# completed_ok / ISSUE-set / errflood / inprog-run changes do.
STATE="${REPO_ROOT}/logs/cron/paper_grade_check.last_pushed"
SIG="$(printf '%s' "$MSG" | sed -E 's/\[[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9:]+\]//; s/ep=[0-9]+//g; s/img=[0-9]+//g')"
if [ "$SIG" != "$(cat "$STATE" 2>/dev/null)" ]; then
  curl -s -H "Priority: ${PRIO}" -d "$MSG" "ntfy.sh/${TOPIC}" >/dev/null 2>&1
  printf '%s' "$SIG" > "$STATE"
fi
