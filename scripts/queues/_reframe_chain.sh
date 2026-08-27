#!/usr/bin/env bash
# _reframe_chain.sh — unattended multi-phase chain for the reframe evidence
# (declared in docs/checkpoints/pre_run/reframe_chain_launch_intent_20260819.md).
#
# Runs, in order, with a gate between every phase and nobody watching:
#   0.  B5 x cls x dom, 1-task SMOKE                       ~$1     gate: must produce a step
#   A.  B5 x cls x dom  x2 (the second is the REPLICATE)   ~$64    breaks the MoE<->serving confound
#   B.  B0 x red x {P-text, P-prompt, P-SoM}               ~$66    second site for the noise envelope
#   C.  B5 x cls x {som, vision, P-text, P-prompt, P-SoM}  ~$160   attack surface #3, trimmable
#
# WHY A NEW SCRIPT RATHER THAN queue_chain.sh DIRECTLY. queue_chain runs cells
# back-to-back but has no notion of a gate: a cell that lands empty is followed by the
# next one regardless. Here cell 0 exists precisely to stop the chain, and the site
# changes twice (cls -> red -> cls), which the host-global lease has to be re-taken for.
#
# WHAT IS DELIBERATELY COPIED FROM _b1_floor_watcher.sh (v2, post three-lineage audit).
# Every one of these was a bug in that script's v1; they are not stylistic choices:
#   - PID identity by /proc starttime, not the number (B-1975: pid_max churn wraps in
#     50-129 h on this host, and a bare pid cannot tell "alive" from "reused")
#   - completeness by the queue_chain C3 triple (JSON parses / condition_id matches /
#     episodes EXACTLY equals expected), not `[ -s file ]` — B-1974 found a 3191-byte
#     summary carrying `episodes: 0` that `-s` happily passed
#   - `export FORCE_NEW=1` is LOAD-BEARING (B-1916/§451.5): queue_chain reads
#     `FORCE_NEW="${FORCE_NEW:-0}"`, so without it the replicate resume-globs onto the
#     canonical run — losing the second observation AND polluting the first, while
#     looking like it ran
#   - RESET_BEFORE is NOT exported: queue_chain sets RESET_FLAG=1 itself and overrides
#     the leaf env, so exporting it here would be declaration drift
#   - child acknowledgement: after launching, wait and confirm it is still alive rather
#     than reading `$!` and declaring success (B-1976)
#
# Usage (on the A100, after the cell-5 runner finishes or with WAIT_RUN set):
#   setsid nohup bash scripts/queues/_reframe_chain.sh > logs/reframe_chain.log 2>&1 &
#
# Env:
#   WAIT_RUN / WAIT_COND / WAIT_N  wait for this condition to complete before starting
#   START_AT                       skip ahead: smoke | A | B | C   (default smoke)
#   COST_CEILING_USD               halt above this cumulative cost (default 400)
#   DEADLINE_UTC                   halt after this date (default 2026-09-06)
#   DRY_RUN=1                      print the plan and exit
set -uo pipefail
REPO="${P79_REPO:-/home/ubuntu/workspace/p79}"
cd "$REPO" || { echo "repo not found: $REPO" >&2; exit 1; }

NTFY="${NTFY_TOPIC:-p79-exp-dgx-spark}"
TS="$(date -u +%Y%m%d_%H%M%S)"
CHAIN_EPOCH="$(date +%s)"
LOG="logs/reframe_chain_${TS}.log"
STATE="logs/.reframe_chain_${TS}.state"
COST_CEILING="${COST_CEILING_USD:-400}"
DEADLINE_UTC="${DEADLINE_UTC:-2026-09-06}"
START_AT="${START_AT:-smoke}"
CLS_N=224
RED_N=205   # COLLECTION denominator, not the scored one. AMENDMENT_08 dropped the
            # reddit *scoring* denominator 205 -> 203 and said in the same breath
            # that the *collection* denominator stays 205 "so the B-1834 exact
            # episode-count check ... [is] unaffected". Writing 203 here put the
            # scoring number into the collection check: on 2026-08-26 03:10 UTC it
            # halted the reframe chain on `episodes=205 != expected=203` AFTER all
            # three Phase B cells had completed correctly, costing Phase C its
            # launch and the host 5.5 idle hours. The data was never at fault.

say()  { echo "[reframe $(date -u '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
push() { curl -s -m 20 -H "Title: $1" -d "$2" "https://ntfy.sh/${NTFY}" >/dev/null 2>&1 || true; }
mark() { echo "$(date -u +%FT%TZ) $*" >> "$STATE"; }

halt() {  # <reason>
  say "HALT: $*"
  mark "HALT $*"
  push "reframe chain HALTED" "$*  (log: ${LOG})"
  exit 1
}

_deadline_ok() {
  [[ "$(date -u +%F)" < "$DEADLINE_UTC" ]] || return 1
}

# ---- completeness: the queue_chain C3 triple (see header) -------------------
_condition_complete() {  # <run_id> <cond_id> <expected_n>
  local run_id="$1" cond_id="$2" exp_n="$3" f base
  for base in results/visualwebarena/phase1 results/webarena/phase1; do
    f="${REPO}/${base}/${run_id}/${cond_id}/condition_summary_v2.json"
    [ -s "$f" ] || continue
    EXPECTED_CID="$cond_id" EXPECTED_N="$exp_n" SUMMARY_PATH="$f" python3 -c "
import json, os, sys
try: d = json.load(open(os.environ['SUMMARY_PATH']))
except Exception as e: print(f'invalid JSON: {e}', file=sys.stderr); sys.exit(1)
cid = d.get('condition_id', '')
if cid and cid != os.environ['EXPECTED_CID']:
    print(f'condition_id mismatch: {cid!r}', file=sys.stderr); sys.exit(2)
ep = d.get('episodes', d.get('total_tasks', d.get('num_tasks', d.get('scored_task_count', 0))))
if not isinstance(ep, int) or ep <= 0:
    print(f'episodes invalid: {ep!r}', file=sys.stderr); sys.exit(3)
n = int(os.environ['EXPECTED_N'])
if ep != n:
    print(f'episodes={ep} != expected={n}', file=sys.stderr); sys.exit(4)
sys.exit(0)
" 2>>"$LOG" && return 0
  done
  return 1
}

# ---- cumulative cost across everything this chain produced ------------------
# Read from the artefacts, not accumulated in a variable: a variable resets if the
# script is restarted, and the ceiling is meant to survive that.
_chain_cost() {
  # Sum from the artefacts, bounded below by this chain's own start epoch, so a
  # restart cannot reset the ceiling and pre-existing runs cannot inflate it.
  # Deliberately NOT a running variable: the ceiling has to survive the script dying.
  python3 - "$REPO" "$CHAIN_EPOCH" <<'PY' 2>/dev/null || echo 0
import json, sys, glob, os
repo, since = sys.argv[1], float(sys.argv[2])
tot = 0.0
for pat in ("B5_*", "B0_*reddit*"):
    for f in glob.glob(os.path.join(repo, "results", "*", "phase1", pat, "*", "condition_summary_v2.json")):
        try:
            if os.path.getmtime(f) < since:
                continue
            d = json.load(open(f))
        except Exception:
            continue
        for k in ("cost_total_usd", "total_cost_usd", "cost_usd", "canonical_action_cost_usd"):
            v = d.get(k)
            if isinstance(v, (int, float)):
                tot += float(v)
                break
print(f"{tot:.2f}")
PY
}

_cost_gate() {
  local c; c="$(_chain_cost)"
  say "cumulative recorded cost so far: \$${c} (ceiling \$${COST_CEILING})"
  awk -v a="$c" -v b="$COST_CEILING" 'BEGIN{exit !(a > b)}' && \
    halt "cost ceiling exceeded: \$${c} > \$${COST_CEILING}"
  return 0
}

# ---- one phase = one queue_chain invocation, then verify every cell ---------
# `cells_spec` is newline-separated "<queue cmd>|<run glob>|<cond id>|<expected n>".
run_phase() {  # <label> <site> <cells_spec>
  local label="$1" site="$2" spec="$3"
  _deadline_ok || halt "past DEADLINE_UTC=${DEADLINE_UTC}"
  _cost_gate

  # host-global lease: queue_chain itself never calls this (it takes only the
  # per-container lock and releases it each cell) — see _b1_floor_watcher B-1973.
  # shellcheck disable=SC1091
  source "${REPO}/scripts/queues/_lib_paper_grade_gates.sh" 2>/dev/null || true
  if declare -F assert_no_other_site_chain_running >/dev/null; then
    assert_no_other_site_chain_running "$site" "reframe-${label}" \
      || halt "another site chain is running; refusing to launch phase ${label}"
    say "phase ${label}: host lease OK (site=${site})"
  else
    say "phase ${label}: WARN assert_no_other_site_chain_running unavailable"
  fi

  local cmds=() line
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    cmds+=("${line%%|*}")
  done <<< "$spec"

  say "phase ${label}: launching ${#cmds[@]} cell(s) on ${site}"
  mark "PHASE ${label} START ${#cmds[@]} cells"

  export FORCE_NEW=1   # LOAD-BEARING — see header
  local clog="logs/queue_chain_reframe_${label}_$(date -u +%Y%m%d_%H%M%S).log"
  setsid nohup bash scripts/queues/queue_chain.sh "${cmds[@]}" > "$clog" 2>&1 < /dev/null &
  local pid=$!
  local st; st="$(sed -e 's/^.*) //' "/proc/$pid/stat" 2>/dev/null | awk '{print $20}')"
  sleep 60
  if ! kill -0 "$pid" 2>/dev/null; then
    halt "phase ${label} child died within 60s — see ${clog}"
  fi
  say "phase ${label}: child ${pid} alive after 60s (chain log ${clog})"
  push "reframe ${label} started" "${#cmds[@]} cells on ${site}"

  # wait for the child, identity-checked so a recycled pid cannot end the wait early
  while :; do
    local now; now="$(sed -e 's/^.*) //' "/proc/$pid/stat" 2>/dev/null | awk '{print $20}')"
    [ -n "$now" ] && [ "$now" = "$st" ] || break
    _deadline_ok || halt "past DEADLINE_UTC while phase ${label} was running"
    sleep 300
  done
  say "phase ${label}: chain process exited; verifying cells"

  # verify EVERY declared cell landed complete — the chain exiting proves nothing
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    local glob cond n rid
    glob="$(echo "$line" | cut -d'|' -f2)"
    cond="$(echo "$line" | cut -d'|' -f3)"
    n="$(echo "$line" | cut -d'|' -f4)"
    rid="$(ls -dt "${REPO}"/results/*/phase1/${glob} 2>/dev/null | head -1)"
    [ -n "$rid" ] || halt "phase ${label}: no run dir matching ${glob}"
    rid="$(basename "$rid")"
    if _condition_complete "$rid" "$cond" "$n"; then
      say "  ✓ ${rid}/${cond} complete (${n})"
      mark "CELL OK ${rid} ${cond} ${n}"
    else
      halt "phase ${label}: ${rid}/${cond} did NOT reach ${n} episodes"
    fi
  done <<< "$spec"

  say "phase ${label}: all cells verified"
  push "reframe ${label} done" "all cells verified; cost so far \$$(_chain_cost)"
}

# ============================ plan =========================================
PHASE_A="queue_baseline.sh B5 dom classifieds|B5_dom_classifieds_2*|phase1_dom_router_0|${CLS_N}"
PHASE_A2="queue_baseline.sh B5 dom classifieds|B5_dom_classifieds_2*|phase1_dom_router_0|${CLS_N}"
PHASE_B="queue_phantom_text.sh B0 reddit|B0_phantom_text_reddit_2*|phase1_phantom_text_router_0|${RED_N}
queue_phantom_prompt.sh B0 reddit|B0_phantom_prompt_reddit_2*|phase1_phantom_prompt_router_0|${RED_N}
queue_phantom_som.sh B0 reddit|B0_phantom_som_reddit_2*|phase1_phantom_som_router_0|${RED_N}"
PHASE_C="queue_baseline.sh B5 som classifieds|B5_som_classifieds_2*|phase1_som_router_0|${CLS_N}
queue_baseline.sh B5 vision classifieds|B5_vision_classifieds_2*|phase1_vision_router_0|${CLS_N}
queue_phantom_text.sh B5 classifieds|B5_phantom_text_classifieds_2*|phase1_phantom_text_router_0|${CLS_N}
queue_phantom_prompt.sh B5 classifieds|B5_phantom_prompt_classifieds_2*|phase1_phantom_prompt_router_0|${CLS_N}
queue_phantom_som.sh B5 classifieds|B5_phantom_som_classifieds_2*|phase1_phantom_som_router_0|${CLS_N}"

# B-1993 restart (2026-08-27). A halted phase is resumed by re-declaring ONLY the
# cells that still owe data. Leaving a finished cell in the spec is not merely
# paying for it twice: `export FORCE_NEW=1` mints a fresh run_id for every cell,
# and the verify step below then reads that new run via `ls -dt | head -1` while
# ignoring the good one — so a re-run that lands worse turns a passing phase into
# a halt. Default stays the full phase; override only to resume.
PHASE_C="${PHASE_C_CELLS:-$PHASE_C}"

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "PLAN (dry run)"
  echo "  wait  : ${WAIT_RUN:-<none>} / ${WAIT_COND:-} / ${WAIT_N:-}"
  echo "  start : ${START_AT}"
  echo "  ceiling \$${COST_CEILING}  deadline ${DEADLINE_UTC}"
  for p in smoke A A2 B C; do echo "  phase ${p}"; done
  exit 0
fi

say "reframe chain armed — start=${START_AT} ceiling=\$${COST_CEILING} deadline=${DEADLINE_UTC}"
mark "ARMED start=${START_AT}"

# ---- 0. wait for whatever is currently on the site --------------------------
if [ -n "${WAIT_RUN:-}" ]; then
  WAIT_COND="${WAIT_COND:-phase1_dom_router_0}"; WAIT_N="${WAIT_N:-${CLS_N}}"
  say "waiting for ${WAIT_RUN}/${WAIT_COND} to reach ${WAIT_N}"
  until _condition_complete "$WAIT_RUN" "$WAIT_COND" "$WAIT_N"; do
    _deadline_ok || halt "past deadline while waiting for ${WAIT_RUN}"
    sleep 300
  done
  say "${WAIT_RUN} complete — proceeding"
  mark "WAIT_RUN complete ${WAIT_RUN}"
fi

# ---- 1. smoke gate ----------------------------------------------------------
if [ "$START_AT" = "smoke" ]; then
  say "cell 0: B5 dom cls 1-task smoke"
  export FORCE_NEW=1
  SMOKE_CONFIG=configs/exp_v2_B5_dom_classifieds_smoke.yaml RESET_BEFORE=1 \
    bash scripts/queues/queue_baseline.sh B5 dom classifieds >> "$LOG" 2>&1
  # the smoke launches a runner in the background; wait for it to stop
  sleep 30
  # 1-task smoke; 40 min is generous (a cls episode averages ~2.3 min at B0 throughput).
  # Bounded so a done-condition mistake stalls loudly instead of silently forever.
  _sw=0
  while pgrep -f "run_experiment.*B5_dom_classifieds_smoke" > /dev/null; do
    sleep 30; _sw=$((_sw+30))
    [ "$_sw" -gt 2400 ] && halt "smoke runner still alive after 40 min — investigate before spending"
  done
  SMOKE_DIR="$(ls -dt "${REPO}"/results/visualwebarena/phase1/B5_dom_classifieds_smoke_* 2>/dev/null | head -1)"
  [ -n "$SMOKE_DIR" ] || halt "smoke produced no run directory"
  python3 - "$SMOKE_DIR" <<'PY' >> "$LOG" 2>&1 || halt "SMOKE GATE FAILED — see $LOG; no paid cell was launched"
import json, sys, glob, os
d = sys.argv[1]
fs = glob.glob(os.path.join(d, "*", "episodes", "*_summary_v2.json"))
assert fs, f"no episode summary under {d}"
s = json.load(open(fs[0]))
steps = s.get("agent_action_step_count") or 0
err = s.get("error")
print(f"smoke: task={s.get('task_id')} steps={steps} error={err!r} success={s.get('success')}")
assert steps > 0, f"agent produced 0 steps (error={err!r})"
assert not (isinstance(err, str) and "400" in err), f"HTTP 400 from proxy: {err!r}"
print("SMOKE GATE PASS")
PY
  say "cell 0: SMOKE GATE PASS"
  mark "SMOKE PASS"
  push "reframe smoke PASS" "B5 produced steps on a real episode; proceeding to paid cells"
  START_AT="A"
fi

# ---- 2. phases --------------------------------------------------------------
case "$START_AT" in
  A)  run_phase A  classifieds "$PHASE_A"
      run_phase A2 classifieds "$PHASE_A2"
      run_phase B  reddit      "$PHASE_B"
      run_phase C  classifieds "$PHASE_C" ;;
  A2) run_phase A2 classifieds "$PHASE_A2"
      run_phase B  reddit      "$PHASE_B"
      run_phase C  classifieds "$PHASE_C" ;;
  B)  run_phase B  reddit      "$PHASE_B"
      run_phase C  classifieds "$PHASE_C" ;;
  C)  run_phase C  classifieds "$PHASE_C" ;;
  *)  halt "unknown START_AT=${START_AT}" ;;
esac

say "ALL PHASES COMPLETE — cost \$$(_chain_cost)"
mark "ALL COMPLETE"
push "reframe chain COMPLETE" "all phases verified. cost \$$(_chain_cost). Next: sync + register A1/A2 in CLEAN_PAIRS."
