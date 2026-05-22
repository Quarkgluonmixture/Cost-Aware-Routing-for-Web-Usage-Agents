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

OUT="$(timeout 180 ssh "$A100" "cd ${REMOTE} && \
  .venv/bin/python3 scripts/maintenance/paper_grade_check.py 2>&1; \
  echo '===MANIFEST==='; \
  .venv/bin/python3 scripts/analysis/validate_fire_manifest.py 2>&1 | tail -3" 2>&1)"
SSH_RC=$?

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
  curl -s -H "Priority: high" -d "🔴 PAPER-GRADE [$TS]: ${VERDICT} | ${DETAIL}" "ntfy.sh/${TOPIC}" >/dev/null 2>&1
else
  curl -s -d "✅ PAPER-GRADE [$TS]: ${VERDICT:-clean}" "ntfy.sh/${TOPIC}" >/dev/null 2>&1
fi
