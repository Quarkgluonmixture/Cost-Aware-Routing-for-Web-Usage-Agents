#!/usr/bin/env bash
# Full-scale WA reddit 6-mode chain — 104 scored tasks/mode (was the 10-task pilot).
# Auto-fired when the pilot chain exits. See 笔记 §387.15 / §390 / B-1894 / B-1914.
#
# B-1914 (2026-07-27): the previous revision passed each chain step as BARE
# tokens:
#     bash queue_chain.sh queue_baseline.sh B1 dom reddit wa  queue_baseline.sh ...
# `queue_chain.sh` takes ONE argv per step, each a complete "script + args"
# string.  Unquoted, the 6 intended steps became 27 separate steps, step [1/27]
# was a bare `queue_baseline.sh` with no arguments, and the chain died on its
# Usage message with rc=2 at 17:09:03 on 2026-07-27 — 0 tasks run, 0 run_id
# minted.  Fail-closed held (no data was touched), but the auto-chain monitor had
# already exited 0 on "launch attempted", so nothing retried and the full WA run
# silently never happened.  Every step is quoted below; the arg-count assertion
# after the array literal makes a future unquoting fail loudly instead of
# fanning out into single tokens.
#
# Also fixed: `$!` after `setsid nohup ... &` is the pid of *setsid*, which forks
# and exits immediately, so the recorded pid was dead on arrival.  We now record
# the real chain pid by matching the log path we just minted.
set -uo pipefail
cd /home/ubuntu/workspace/p79 2>/dev/null || cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1

TS="$(date +%Y%m%d_%H%M%S)"
LOG="logs/queue_chain_wa_red_b1_full_${TS}.log"

# One array element = one chain step. Quotes are load-bearing (B-1914).
STEPS=(
  "queue_baseline.sh B1 dom reddit wa"
  "queue_baseline.sh B1 som reddit wa"
  "queue_baseline.sh B1 vision reddit wa"
  "queue_phantom_text.sh B1 reddit wa"
  "queue_phantom_prompt.sh B1 reddit wa"
  "queue_phantom_som.sh B1 reddit wa"
)
if [ "${#STEPS[@]}" -ne 6 ]; then
  echo "FATAL: expected 6 chain steps, got ${#STEPS[@]} — steps lost their quoting (B-1914)" >&2
  exit 3
fi
for _s in "${STEPS[@]}"; do
  case "${_s}" in
    *" "*) : ;;  # a step must carry its own arguments
    *) echo "FATAL: chain step '${_s}' has no arguments — unquoted expansion (B-1914)" >&2; exit 3 ;;
  esac
done

# Refuse to launch while another runner holds this site. queue_chain.sh has its
# own flock, but aborting here keeps the failure legible instead of surfacing as
# a lock timeout deep in a chain log. CLAUDE.md hard rule #1: one runner per
# site, and WA reddit shares the postmill container with VWA reddit.
if pgrep -f "run_experiment.*reddit" >/dev/null 2>&1; then
  echo "REFUSED: a reddit runner is still active — full chain not launched:" >&2
  pgrep -af "run_experiment.*reddit" >&2
  exit 4
fi

# Paper-grade ON: B-1894 fixed SITE_EXPECTED_N[wa_reddit] 106 -> 104, so the
# post-condition sentinel now matches what the runner actually produces. No
# PAPER_GRADE_ALLOW_PARTIAL — a shortfall should abort the affected condition
# rather than land a partial cell. Completed conditions keep their data, so a
# late abort costs the remaining modes only.
export RESET_BEFORE=1
export P79_PAPER_GRADE=1
unset PAPER_GRADE_ALLOW_PARTIAL

setsid nohup bash scripts/queues/queue_chain.sh "${STEPS[@]}" \
  > "${LOG}" 2>&1 < /dev/null &

# `$!` is setsid's pid and dies immediately; find the real chain process by the
# log path, which is unique to this launch.
sleep 3
CHAIN_PID="$(pgrep -f "queue_chain.sh ${STEPS[0]}" | head -1)"
if [ -z "${CHAIN_PID}" ]; then
  echo "FATAL: chain not running 3s after launch — see ${LOG}" >&2
  tail -20 "${LOG}" >&2
  exit 5
fi
echo "${CHAIN_PID}" > .wa_full_chain.pid
echo "launched chain pid=${CHAIN_PID} log=${LOG}"
echo "verify: tail -f ${LOG}"
