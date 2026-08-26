#!/usr/bin/env bash
# _b5_reddit_chain.sh — B5 (GPT-5.6 terra) × reddit × {dom, som, vision}
#
# Intent (declared BEFORE fire, §469.7):
#   docs/checkpoints/pre_run/b5_reddit_chain_launch_intent_20260826.md
#
# Buys NAACL attack surface #2 (cross-site) for B5, the one backbone whose every
# observation is from a single site. One arm per side (text | combined | visual).
#
# WHY A LIVE QUOTA GATE AND NOT A RECORDED-COST CEILING
#   The previous chains halted on cumulative recorded cost. That number is
#   tokens x the price written in the config, and on 2026-08-26 a registry sweep
#   found every API config off the live price — terra by +25% (0.002/0.012 in
#   the configs vs 0.0025/0.015 live). A ceiling that under-reads real spend by
#   25% is precisely the instrument that lets a chain run past an empty pool.
#   So the gate probes the actual remaining quota before each cell instead.
#   The configs are deliberately NOT re-priced: A1/A2/Phase C all bill at
#   0.002/0.012 and changing it mid-study breaks the cost column's comparability.
#   Consequence is disclosed in the intent file: every B5 dollar figure in this
#   project under-states real spend by 25%; recompute from tokens when publishing.
#
# Usage (on the A100, where the fire lives):
#   nohup setsid bash scripts/queues/_b5_reddit_chain.sh > /dev/null 2>&1 &
#   DRY_RUN=1 bash scripts/queues/_b5_reddit_chain.sh      # print plan, fire nothing
#   SKIP_WAIT=1 ...                                        # do not wait for Phase C
set -uo pipefail

REPO="${P79_REPO:-/home/ubuntu/workspace/p79}"
cd "$REPO" || { echo "no repo at $REPO"; exit 1; }

NTFY="${NTFY_TOPIC:-p79-exp-dgx-spark}"
TS="$(date -u +%Y%m%d_%H%M%S)"
LOG="logs/b5_reddit_chain_${TS}.log"
STATE="logs/.b5_reddit_chain_${TS}.state"
RED_N=205   # COLLECTION denominator, not the scored one. AMENDMENT_08 dropped the
            # reddit *scoring* denominator 205 -> 203 and said in the same breath
            # that the *collection* denominator stays 205 "so the B-1834 exact
            # episode-count check ... [is] unaffected". Writing 203 here put the
            # scoring number into the collection check: on 2026-08-26 03:10 UTC it
            # halted the reframe chain on `episodes=205 != expected=203` AFTER all
            # three Phase B cells had completed correctly, costing Phase C its
            # launch and the host 5.5 idle hours. The data was never at fault.
QUOTA_FLOOR="${QUOTA_FLOOR_USD:-60}"
DEADLINE_UTC="${DEADLINE_UTC:-2026-09-04}"

# Phase C's last cell — what this chain waits on before touching the host.
WAIT_GLOB="${WAIT_GLOB:-B5_phantom_som_classifieds_2*}"
WAIT_COND="${WAIT_COND:-phase1_phantom_som_router_0}"
WAIT_N="${WAIT_N:-224}"

say()  { echo "[b5red $(date -u '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
push() { curl -s -m 20 -H "Title: $1" -d "$2" "https://ntfy.sh/${NTFY}" >/dev/null 2>&1 || true; }
mark() { echo "$(date -u +%FT%TZ) $*" >> "$STATE"; }
halt() {
  say "HALT: $*"; mark "HALT $*"
  push "b5-reddit chain HALT" "$*"
  exit 1
}

_deadline_ok() { [[ "$(date -u +%F)" < "$DEADLINE_UTC" ]]; }

# ---- completeness: same C3 triple as the reframe chain ----------------------
_cell_complete() {  # <run_glob> <cond_id> <expected_n>
  local g="$1" cond="$2" exp="$3" f
  for f in ${REPO}/results/visualwebarena/phase1/${g}/${cond}/condition_summary_v2.json; do
    [ -s "$f" ] || continue
    EXPECTED_CID="$cond" EXPECTED_N="$exp" SUMMARY_PATH="$f" python3 -c "
import json, os, sys
d = json.load(open(os.environ['SUMMARY_PATH']))
cid = d.get('condition_id','')
if cid and cid != os.environ['EXPECTED_CID']: sys.exit(2)
ep = d.get('episodes', d.get('total_tasks', d.get('num_tasks', d.get('scored_task_count', 0))))
if not isinstance(ep,int) or ep <= 0: sys.exit(3)
if ep != int(os.environ['EXPECTED_N']): sys.exit(4)
sys.exit(0)
" 2>>"$LOG" && return 0
  done
  return 1
}

# ---- the gate that matters: live remaining quota ----------------------------
_quota() {
  .venv/bin/python3 scripts/maintenance/proxy_budget_watch.py --once 2>/dev/null \
    | python3 -c "
import json,sys
try:
    d=json.load(sys.stdin)
    r=d.get('remaining')
    print(f'{float(r):.2f}' if isinstance(r,(int,float)) else 'ERR')
except Exception: print('ERR')
"
}

_quota_gate() {  # <cell label>
  local q; q="$(_quota)"
  if [ "$q" = "ERR" ] || [ -z "$q" ]; then
    # A probe that cannot read the balance is not permission to spend.
    halt "quota probe failed before ${1} — refusing to start a paid cell blind"
  fi
  say "  live quota before ${1}: \$${q} (floor \$${QUOTA_FLOOR})"
  awk -v a="$q" -v b="$QUOTA_FLOOR" 'BEGIN{exit !(a < b)}' \
    && halt "quota \$${q} below floor \$${QUOTA_FLOOR} before ${1} — top up and restart"
  mark "QUOTA_OK ${1} ${q}"
}

# ---- host-global lease: a real runner, not a waiting chain ------------------
_host_free() {
  ! pgrep -f "run_experiment\.py" > /dev/null 2>&1
}

# ---- one cell ---------------------------------------------------------------
run_cell() {  # <mode> <cond_id>
  local mode="$1" cond="$2" glob="B5_${1}_reddit_2*"
  local label="B5-red-${mode}"

  if _cell_complete "$glob" "$cond" "$RED_N"; then
    say "${label}: already complete — skipping"; mark "SKIP ${label}"; return 0
  fi

  _deadline_ok || halt "past DEADLINE_UTC=${DEADLINE_UTC} before ${label}"
  _quota_gate "$label"

  local waited=0
  until _host_free; do
    [ "$waited" -eq 0 ] && say "${label}: waiting for the host to go free"
    sleep 300; waited=$((waited+300))
    [ "$waited" -gt 259200 ] && halt "${label}: host busy for 3 days"
    _deadline_ok || halt "past deadline while waiting for host (${label})"
  done

  say "${label}: launching (reset + queue_baseline)"
  mark "LAUNCH ${label}"
  push "b5-reddit ${label} start" "quota-gated launch; n=${RED_N}"

  RESET_BEFORE=1 bash scripts/queues/queue_baseline.sh B5 "$mode" reddit >> "$LOG" 2>&1

  # WAIT FOR THE RUNNER TO APPEAR FIRST — do not go straight to waiting for it
  # to exit. A site reset + auth gate + model cold start takes minutes, not
  # seconds (Phase C on 2026-08-26 held the container lock for well over 7 min
  # before its first runner). A bare `while pgrep` after a 60s sleep evaluates
  # false during that window, falls through to the completeness check, finds no
  # episodes, and halts a cell that was launching perfectly well.
  local appear=0
  until pgrep -f "run_experiment.*B5_${mode}_reddit" > /dev/null 2>&1; do
    sleep 30; appear=$((appear+30))
    if [ "$appear" -gt 2400 ]; then
      halt "${label}: no runner 40 min after launch — reset or auth gate is stuck, see ${LOG}"
    fi
    _deadline_ok || halt "past deadline while ${label} was starting"
  done
  say "  ${label}: runner up after ${appear}s (reset + cold start)"

  local spun=0
  while pgrep -f "run_experiment.*B5_${mode}_reddit" > /dev/null 2>&1; do
    sleep 300; spun=$((spun+300))
    [ "$spun" -gt 432000 ] && halt "${label}: runner alive past 5 days"
    _deadline_ok || halt "past deadline while ${label} was running"
  done

  if _cell_complete "$glob" "$cond" "$RED_N"; then
    say "${label}: ✓ complete (${RED_N})"; mark "CELL OK ${label}"
    push "b5-reddit ${label} done" "complete at n=${RED_N}; quota now \$$(_quota)"
  else
    halt "${label}: did NOT reach ${RED_N} episodes"
  fi
}

# ============================ plan ==========================================
CELLS="dom|phase1_dom_router_0
som|phase1_som_router_0
vision|phase1_vision_router_0"

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "PLAN (dry run) — B5 × reddit"
  echo "  wait for : ${WAIT_GLOB} / ${WAIT_COND} / ${WAIT_N}"
  echo "  cells    : dom → som → vision   (n=${RED_N} each)"
  echo "  quota    : probe before each cell, floor \$${QUOTA_FLOOR}"
  echo "  deadline : ${DEADLINE_UTC}"
  echo "  live quota now: \$$(_quota)"
  exit 0
fi

say "b5-reddit chain armed — floor \$${QUOTA_FLOOR} deadline ${DEADLINE_UTC}"
mark "ARMED"

# ---- upstream liveness: never wait silently on something that already died --
#
# Learned the hard way on 2026-08-26: the reframe chain HALTed at 03:10 UTC on a
# wrong episode-count gate, and nothing noticed for 5.5 hours because the only
# thing watching it was a human. A downstream chain that waits on an upstream
# it cannot see is a chain that waits until its own deadline. Two independent
# detectors, because each misses what the other catches:
#
#   1. HALT marker in the upstream state file — a clean, deliberate stop.
#      File-marker based (CLAUDE.md Tier 1), so no pgrep self-match hazard.
#   2. Stall — no Phase C episode has landed in STALL_HOURS. Catches the deaths
#      that leave no marker at all (kill -9, host reboot, OOM).
#
STALL_HOURS="${STALL_HOURS:-3}"

_upstream_halted() {
  local st; st="$(ls -t "${REPO}"/logs/.reframe_chain_*.state 2>/dev/null | head -1)"
  [ -n "$st" ] && grep -q " HALT " "$st"
}

# Newest Phase C episode mtime, as epoch seconds. Empty if none yet.
_phase_c_progress() {
  find "${REPO}"/results/visualwebarena/phase1/B5_*_classifieds_2*/*/episodes \
       -name '*_summary_v2.json' -newermt "-${STALL_HOURS} hours" 2>/dev/null | head -1
}

if [ "${SKIP_WAIT:-0}" != "1" ]; then
  say "waiting for Phase C last cell (${WAIT_GLOB}/${WAIT_COND} @ ${WAIT_N})"
  say "  upstream watch: HALT marker + ${STALL_HOURS}h stall detector"
  _grace=0
  until _cell_complete "$WAIT_GLOB" "$WAIT_COND" "$WAIT_N"; do
    _deadline_ok || halt "past deadline while waiting for Phase C"

    if _upstream_halted; then
      halt "upstream reframe chain has HALTED — this chain would wait forever. \
Fix the upstream halt, then restart this chain (it is idempotent: completed cells are skipped)."
    fi

    if [ -z "$(_phase_c_progress)" ]; then
      # Grace period: reset + auth + cold start legitimately produce no episode
      # for a while at each cell boundary. Only a sustained gap is a stall.
      _grace=$((_grace+1))
      if [ "$_grace" -ge 2 ]; then
        halt "no Phase C episode in ${STALL_HOURS}h across two consecutive checks — upstream is stalled, not slow. \
Check the A100 before restarting."
      fi
      say "  no Phase C episode in ${STALL_HOURS}h (grace ${_grace}/2)"
    else
      _grace=0
    fi
    sleep 600
  done
  say "Phase C complete — proceeding"
  mark "PHASE_C_DONE"
  push "b5-reddit chain proceeding" "Phase C finished; starting reddit cells"
fi

while IFS='|' read -r mode cond; do
  [ -n "$mode" ] || continue
  run_cell "$mode" "$cond"
done <<< "$CELLS"

say "ALL CELLS COMPLETE — quota now \$$(_quota)"
mark "ALL COMPLETE"
push "b5-reddit chain COMPLETE" "3 cells verified. quota \$$(_quota). Next: sync + analysis + register in the evidence inventory."
