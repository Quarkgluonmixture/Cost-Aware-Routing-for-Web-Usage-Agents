#!/usr/bin/env bash
# _reframe_bootstrap.sh — deploy-then-arm, unattended, on the fire host.
#
# Runs on the A100. Waits for the in-flight cell to finish, brings the repo to the
# intended commit, VERIFIES it there, and only then arms the reframe chain.
#
# WHY DEPLOY IS A SEPARATE SCRIPT FROM THE CHAIN. The chain lives in the repo, so it
# cannot be the thing that updates the repo. This script is self-contained and is
# copied over by scp, not pulled.
#
# WHY IT WAITS. §296: changing fire code mid-fire is how you get a run whose first
# half and second half were produced by different code. Cell 5 (B1 dom cls) is still
# producing; nothing is touched until its condition summary is complete.
#
# WHY IT VERIFIES ON THE HOST RATHER THAN TRUSTING THE DEV MACHINE. §469.4 — a fix
# that is green on the DGX and absent on the A100 has happened before, and cost a
# whole chain launch. §469.5 is the sharper version: the two hosts run different
# Python (3.12 vs 3.10), and a 3.12-only f-string reaches the A100 through a file
# copy and fails there as a SyntaxError — which `validate_fire_manifest` then turns
# into "every replicate is a ghost". So: compile on the host, run the tests on the
# host, and refuse to arm if either fails.
#
# ON DISCARDING THE A100 WORKING TREE. Verified 2026-08-19 before writing this:
# A100's 84 modified files are all older-or-equal to the target commit —
# `proxy_api_agent.py` diffs against the target by exactly the B-1990 change and
# nothing else, and the untracked-by-diff files (e.g. Makefile) are unmodified copies
# of the old commit. Nothing on that host is unique. It is stashed rather than reset
# anyway, because "verified" and "irreversible" should not be combined.
set -uo pipefail
REPO="${P79_REPO:-/home/ubuntu/workspace/p79}"
BRANCH="${TARGET_BRANCH:-fix/b1878-reddit-reference-image}"
TARGET_SHA="${TARGET_SHA:-}"
WAIT_RUN="${WAIT_RUN:-}"
WAIT_COND="${WAIT_COND:-phase1_dom_router_0}"
WAIT_N="${WAIT_N:-224}"
NTFY="${NTFY_TOPIC:-p79-exp-dgx-spark}"
TS="$(date -u +%Y%m%d_%H%M%S)"
LOG="${REPO}/logs/reframe_bootstrap_${TS}.log"

cd "$REPO" || { echo "no repo at $REPO" >&2; exit 1; }
say()  { echo "[bootstrap $(date -u '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
push() { curl -s -m 20 -H "Title: $1" -d "$2" "https://ntfy.sh/${NTFY}" >/dev/null 2>&1 || true; }
die()  { say "ABORT: $*"; push "reframe bootstrap ABORTED" "$*"; exit 1; }

_complete() {  # <run_id> <cond> <n>
  local f="${REPO}/results/visualwebarena/phase1/$1/$2/condition_summary_v2.json"
  [ -s "$f" ] || return 1
  EXPECTED_CID="$2" EXPECTED_N="$3" SUMMARY_PATH="$f" python3 -c "
import json, os, sys
d = json.load(open(os.environ['SUMMARY_PATH']))
cid = d.get('condition_id', '')
if cid and cid != os.environ['EXPECTED_CID']: sys.exit(2)
ep = d.get('episodes', d.get('total_tasks', d.get('num_tasks', d.get('scored_task_count', 0))))
sys.exit(0 if isinstance(ep, int) and ep == int(os.environ['EXPECTED_N']) else 4)
" 2>/dev/null
}

say "bootstrap armed — target ${BRANCH}${TARGET_SHA:+ @ ${TARGET_SHA}}"

# ---- 1. wait for the in-flight cell ---------------------------------------
if [ -n "$WAIT_RUN" ]; then
  say "waiting for ${WAIT_RUN}/${WAIT_COND} to reach ${WAIT_N} episodes"
  DEADLINE=$(( $(date +%s) + 3*24*3600 ))
  until _complete "$WAIT_RUN" "$WAIT_COND" "$WAIT_N"; do
    [ "$(date +%s)" -lt "$DEADLINE" ] || die "3-day deadline waiting for ${WAIT_RUN}"
    # if the runner is gone but the summary never completed, stop — do not deploy
    # onto a half-finished cell and pretend the wait succeeded
    if ! pgrep -f "run_experiment.*${WAIT_RUN}" >/dev/null 2>&1; then
      sleep 120   # allow the summary write to land after the process exits
      _complete "$WAIT_RUN" "$WAIT_COND" "$WAIT_N" && break
      die "runner for ${WAIT_RUN} exited but the condition never reached ${WAIT_N} — needs a human"
    fi
    sleep 300
  done
  say "${WAIT_RUN} complete"
fi

# nothing else may be running when we touch the code
pgrep -f "run_experiment.py" >/dev/null 2>&1 && die "a runner is still active; refusing to change fire code"

# ---- 2. deploy -------------------------------------------------------------
# B-1988 (2026-08-20). The stash below is NOT redundant for every file. Some tracked
# files are WRITTEN ON THIS HOST by the watchdog and the runner — the manifest the
# auto-bind fills in, and the quarantine registry the paper-grade abort appends to —
# and this host has no git credentials (§471.8), so those writes live only in the
# working tree until a human pulls them to the DGX. `git stash push -u` sweeps them
# into a stash where nothing looks for them: `quarantine_registry.py query` reports
# zero events while the events sit in stash@{N}, so the ntfy that says "manual review"
# points at an empty queue.
# Measured the day this was written: 9 events lost that way, 2026-08-16 → 08-20,
# including three human classifications with written rationale (the B4 protocol-wall
# smokes and the B-1980 shape-vs-loss guard). The fire_manifest's 6 shopping bindings
# survived the same sweep only because the watchdog happened to rebind them a minute
# later. So: copy them out of the repo FIRST, where a later sync can still find them,
# and say so loudly. Copy rather than halt — halting would block every deploy that
# follows an auto-bind, which is most of them.
_HOST_AUTHORED=(
  "docs/checkpoints/pre_run/fire_manifest.json"
  "docs/checkpoints/quarantine_registry.jsonl"
)
_PENDING_DIR="/home/ubuntu/_a100_pending/${TS}"
_preserved=()
for _f in "${_HOST_AUTHORED[@]}"; do
  if [ -n "$(git status --porcelain -- "$_f" 2>/dev/null)" ]; then
    mkdir -p "${_PENDING_DIR}/$(dirname "$_f")"
    cp -a "$_f" "${_PENDING_DIR}/$_f" 2>/dev/null && _preserved+=("$_f")
  fi
done
if [ ${#_preserved[@]} -gt 0 ]; then
  say "PRESERVING ${#_preserved[@]} host-authored file(s) to ${_PENDING_DIR} before stashing:"
  for _f in "${_preserved[@]}"; do say "    ${_f}"; done
  push "P79 A100 有未同步的本机产物" \
    "部署前已复制 ${#_preserved[@]} 个文件到 ${_PENDING_DIR} (A100 推不了 git, stash 会把它们埋掉)。请 sync 到 DGX 并提交。"
fi

say "stashing local working tree (verified redundant, kept as a way back)"
git stash push -u -m "pre-reframe-bootstrap ${TS}" >>"$LOG" 2>&1 || say "  (nothing to stash)"
git fetch origin "$BRANCH" >>"$LOG" 2>&1 || die "git fetch failed"
git checkout -B "$BRANCH" "origin/${BRANCH}" >>"$LOG" 2>&1 || die "git checkout failed"
NOW_SHA="$(git rev-parse --short HEAD)"
say "now at ${NOW_SHA} on ${BRANCH}"
# B-1983 (2026-08-20). This used to be a STRING comparison of two `--short` SHAs, and
# `--short` has no fixed length: git picks the shortest unambiguous prefix, which depends
# on how many objects the repo holds. The DGX abbreviates to 7, this host to 8 — so a pin
# handed over from the dev machine could NEVER match here, whatever the commit was. The
# 2026-08-19 03:32Z abort reported "expected a449abb, got defc809b — someone pushed in
# between"; a push had indeed happened, which made the message look like a complete
# diagnosis and hid the fact that the check was unconditionally broken underneath it.
# Resolve both sides to full object names instead; that is length-agnostic and also
# accepts a full SHA, a tag, or any other rev the operator passes.
if [ -n "$TARGET_SHA" ]; then
  WANT_SHA="$(git rev-parse --verify --quiet "${TARGET_SHA}^{commit}" || true)"
  HAVE_SHA="$(git rev-parse --verify HEAD)"
  [ -n "$WANT_SHA" ] || die "TARGET_SHA '${TARGET_SHA}' does not resolve to a commit here"
  if [ "$WANT_SHA" != "$HAVE_SHA" ]; then
    die "expected ${TARGET_SHA} (${WANT_SHA}), got ${NOW_SHA} (${HAVE_SHA}) — someone pushed in between"
  fi
  say "  SHA pin OK (${WANT_SHA})"
fi

# ---- 3. verify ON THIS HOST (§469.4 / §469.5) ------------------------------
say "compiling changed python on host python $(python3 --version 2>&1)"
python3 -m py_compile p79/agents/proxy_api_agent.py \
                      scripts/analysis/aggregate_noise_floor_inventory.py \
                      scripts/maintenance/probe_model_five_gates.py \
  >>"$LOG" 2>&1 || die "py_compile FAILED on host python (this is the §469.5 failure mode)"
say "  py_compile OK"

say "syntax-checking the queue scripts"
for s in scripts/queues/queue_baseline.sh scripts/queues/queue_phantom_*.sh \
         scripts/queues/queue_chain.sh scripts/queues/_reframe_chain.sh; do
  bash -n "$s" >>"$LOG" 2>&1 || die "bash -n failed: $s"
done
say "  bash -n OK"

say "running the B-1990 invariants + the agent/action suite on host"
if [ -x .venv/bin/python3 ]; then PY=.venv/bin/python3; else PY=python3; fi
$PY -m pytest tests/test_b1990_response_format_road.py -q >>"$LOG" 2>&1 \
  || die "B-1990 invariant tests FAILED on host"
$PY -m pytest tests/ -q -k "proxy or agent or action" >>"$LOG" 2>&1 \
  || die "agent/action suite FAILED on host"
say "  tests OK"

say "confirming every config the chain needs is present"
for c in exp_v2_B5_dom_classifieds_smoke exp_v2_B5_dom_classifieds \
         exp_v2_B0_phantom_text_reddit exp_v2_B0_phantom_prompt_reddit exp_v2_B0_phantom_som_reddit \
         exp_v2_B5_som_classifieds exp_v2_B5_vision_classifieds \
         exp_v2_B5_phantom_text_classifieds exp_v2_B5_phantom_prompt_classifieds \
         exp_v2_B5_phantom_som_classifieds; do
  [ -f "configs/${c}.yaml" ] || die "missing config: ${c}.yaml"
done
say "  10/10 configs present"

# ---- 4. arm ----------------------------------------------------------------
say "verification complete — arming the reframe chain"
push "reframe bootstrap OK" "deployed ${NOW_SHA}, host verify passed, arming chain"
CHAINLOG="${REPO}/logs/reframe_chain_driver_${TS}.log"
setsid nohup env P79_REPO="$REPO" NTFY_TOPIC="$NTFY" \
  bash scripts/queues/_reframe_chain.sh > "$CHAINLOG" 2>&1 < /dev/null &
CPID=$!
sleep 60
kill -0 "$CPID" 2>/dev/null || die "chain died within 60s of arming — see ${CHAINLOG}"
say "chain running as pid ${CPID}; log ${CHAINLOG}"
push "reframe chain ARMED" "pid ${CPID}. smoke gate first; no paid cell runs unless it passes."
