#!/usr/bin/env bash
# queue_chain.sh — Sequentially launch a list of queue commands, waiting for
# each runner to complete before launching the next. Useful for chaining cells
# that share a single GPU instance (B1 4B local) or any paper-grade sequence.
#
# Each queued command goes through queue_baseline.sh / queue_phantom_som.sh /
# queue_phantom_text.sh which already handle reset+auth_refresh+watchdog
# launch+idempotent skip. This chain ALWAYS exports RESET_BEFORE=1 by default
# (paper-grade — every cell starts from a fresh post-reset site state); pass
# --no-reset to disable (rare, e.g. resume-only chain).
# Note: queue_phantom_dom.sh exists as a back-compat symlink to queue_phantom_text.sh.
#
# Usage:
#   nohup bash scripts/queues/queue_chain.sh [--no-reset] \
#     "<cmd1>" "<cmd2>" ... \
#     > logs/queue_chain_<label>.log 2>&1 &
#
# Each <cmd> is a queue script invocation, relative to scripts/queues/:
#   "queue_phantom_som.sh B1 classifieds"
#   "queue_phantom_text.sh B1 reddit"
#   "queue_baseline.sh B0 dom shopping"
#   "queue_baseline.sh B0 som shopping wa"
#
# The chain auto-detects an already-running cell (queue scripts are idempotent;
# RESET is skipped when a runner is already attached). For the FIRST queued
# cell — if it's already running, chain just waits for completion and proceeds
# to the next.
#
# Examples:
#   # B1 phantom 4-cell chain (cls already running):
#   nohup bash scripts/queues/queue_chain.sh \
#     "queue_phantom_som.sh B1 classifieds" \
#     "queue_phantom_som.sh B1 reddit" \
#     "queue_phantom_text.sh B1 classifieds" \
#     "queue_phantom_text.sh B1 reddit" \
#     > logs/queue_chain_b1_phantom.log 2>&1 &
#
#   # B0 phantom shopping pair (after B0 dom shopping done):
#   nohup bash scripts/queues/queue_chain.sh \
#     "queue_phantom_som.sh B0 shopping" \
#     "queue_phantom_text.sh B0 shopping" \
#     > logs/queue_chain_b0_phantom_shop.log 2>&1 &

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_DIR}"

log() { echo "[chain $(date '+%H:%M:%S')] $*"; }

# ---------- arg parsing ----------
RESET_FLAG=1
if [[ "${1:-}" == "--no-reset" ]]; then
  RESET_FLAG=0
  shift
fi

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 [--no-reset] <queue_command_1> [<queue_command_2> ...]" >&2
  echo "  Each command: 'queue_<name>.sh <args>' (relative to scripts/queues/)" >&2
  echo "  See header for examples." >&2
  exit 2
fi

# ---------- helpers ----------
# wait_for_runner_done blocks until the runner exits cleanly OR aborts the chain
# if the watchdog dies mid-run (A1.13 P0-4, 2026-05-16).
#
# Watchdog liveness invariant: queue_baseline.sh:288-296 declares watchdog FATAL
# at launch ("paper-grade launch requires watchdog (auth refresh + auto-clean)").
# Pre-fix wait loop only watched runner PID — watchdog death mid-run was silent,
# losing reactive auth refresh + idle alerts + auto-clean for the rest of the run
# (multi-day chain weekend runs at highest risk). Q2 A decision: abort chain on
# watchdog death — paper-grade > compute reclaim. User triggers manual restart
# after addressing watchdog root cause (typical: ntfy curl SIGPIPE, OOM, glm
# config bug, NPE on bad state JSON).
wait_for_runner_done() {
  local pattern="$1"
  local label="$2"
  local elapsed=0
  while pgrep -f "run_experiment.py.*${pattern}" > /dev/null; do
    sleep 60
    elapsed=$((elapsed + 60))
    # A1.13 P0-4: watchdog liveness check
    if ! pgrep -f "experiment_watchdog.*${pattern}" > /dev/null; then
      log "  [FATAL] ${label}: watchdog died mid-run (runner still alive after ${elapsed}s)"
      log "  paper-grade contamination risk: no reactive auth refresh + no auto-clean + no idle alert"
      log "  Q2 A decision (A1.13 audit 2026-05-16): abort chain, kill runner, notify user"
      pkill -f "run_experiment.py.*${pattern}" 2>/dev/null || true
      if command -v curl > /dev/null; then
        curl -d "queue_chain ABORT (${label}): watchdog died after ${elapsed}s; runner killed. Restart after root cause." \
          "ntfy.sh/${NTFY_TOPIC:-p79-exp-dgx-spark}" 2>/dev/null || true
      fi
      exit 1
    fi
    if (( elapsed % 1800 == 0 )); then
      log "  ${label}: still running (${elapsed}s elapsed; watchdog alive)..."
      pgrep -af "run_experiment.py.*${pattern}" | head -1 | sed 's/^/    /'
    fi
  done
  log "  ${label}: runner done"
}

# ---------- chain ----------
log "=================================================="
log "queue_chain — $# cells (RESET_BEFORE=${RESET_FLAG})"
for arg in "$@"; do log "  - $arg"; done
log "=================================================="

idx=0
for cmd in "$@"; do
  idx=$((idx + 1))
  log ""
  log "------ [${idx}/$#] ${cmd} ------"

  # Validate the script exists (cmd is "queue_xxx.sh args...")
  script_name="${cmd%% *}"
  if [[ ! -f "${SCRIPT_DIR}/${script_name}" ]]; then
    log "  [error] script not found: ${SCRIPT_DIR}/${script_name}"
    log "  aborting chain"
    exit 1
  fi

  # ---- Same-site collision check (paper-grade hard rule §106) ----
  # Parse <baseline> + <site> from the queue command args.
  # queue_baseline.sh format: <baseline> <mode> <site> [benchmark]
  # queue_phantom_*.sh format: <baseline> <site> [benchmark]
  # Hard rule: a site's docker container + login account is shared, so only
  # ONE baseline (of B0 / B1 / B2) may run on a site at a time. Generalised
  # from the original B0-vs-B1-only check — with Gemma3-VL (B2) the pairwise
  # single-"other_baseline" logic silently missed the third baseline.
  cmd_args=( ${cmd} )
  this_baseline="${cmd_args[1]:-}"  # B0 / B1 / B2
  if [[ "${script_name}" == queue_baseline.sh ]]; then
    this_site="${cmd_args[3]:-}"            # 4th token (script bash mode site)
    this_benchmark="${cmd_args[4]:-vwa}"    # 5th token (default vwa)
  else
    this_site="${cmd_args[2]:-}"            # 3rd token (script bash site)
    this_benchmark="${cmd_args[3]:-vwa}"    # 4th token (default vwa)
  fi

  # B-646 (A1.13 P1-7 codex F4 OOB, 2026-05-17): per-(site,benchmark) flock
  # acquired before collision check, held for entire iteration (collision check
  # + reset/auth + launch + wait + sentinel). Pre-fix pgrep-based collision
  # detection is TOCTOU — two chains fired in tight window (master orchestrator
  # + manual retry, or 2 manual chains) both see empty pgrep, both reset_and_auth,
  # both launch runner attached to same docker user account → session race +
  # cross-mode contamination. flock-nb defends:
  #   • single chain: acquires + holds → other chains see lock + FATAL exit;
  #   • lock is per (site, benchmark) so VWA shopping + WA shopping_admin can
  #     run concurrently (no docker container collision);
  #   • lock auto-releases when fd 9 closes (at end of iteration `exec 9>&-` or
  #     subshell exit).
  # Stale lock handling: rm `${LOCK_DIR}/p79_${site}_${benchmark}.lock` to
  # force-release (lock file is empty marker; presence + flock state matter).
  if [[ -n "${this_site}" && -n "${this_benchmark}" ]]; then
    LOCK_DIR="${REPO_DIR}/.locks"
    mkdir -p "${LOCK_DIR}" 2>/dev/null || true
    LOCK_FILE="${LOCK_DIR}/p79_${this_site}_${this_benchmark}.lock"
    exec 9>"${LOCK_FILE}"
    if ! flock -n 9; then
      log "  [FATAL] another paper-grade chain holds lock for site=${this_site} benchmark=${this_benchmark}"
      log "  lock file: ${LOCK_FILE}"
      log "  if lock is stale (prior chain crashed), 'rm ${LOCK_FILE}' to force-release before retry"
      if command -v curl > /dev/null; then
        curl -d "queue_chain ABORT (${this_baseline} ${this_site} ${this_benchmark}): another chain holds site lock; possible double-fire" \
          "ntfy.sh/${NTFY_TOPIC:-p79-exp-dgx-spark}" 2>/dev/null || true
      fi
      exit 1
    fi
    log "  [lock] acquired ${LOCK_FILE} (held until iteration end)"
  fi
  # B-637 (A1.13 P1-1 Claude + gemini G10 2-AI, 2026-05-17): regex anchor.
  # Pre-fix pgrep `_${this_site}_` substring overlap problems:
  #   (a) `_shopping_` substring-matched `_shopping_admin_` → VWA shopping
  #       waited for WA shopping_admin to finish (unrelated docker stack).
  #   (b) VWA `_shopping_` substring-matched WA `_wa_shopping_` AND vice
  #       versa → cross-benchmark wait loop even though docker stacks separate.
  # Fix: build benchmark-aware site pattern that anchors on the 8-digit date
  # token after the site name (which is structural — mint_run_id always inserts
  # YYYYMMDD right after CFG_NAME). For VWA: `_${this_site}_[0-9]{8}_` plus
  # `grep -v _wa_` exclusion to suppress WA processes. For WA: prefix `_wa_`
  # required. Empirical:
  #   `_shopping_[0-9]{8}_` does NOT match `B0_dom_shopping_admin_20260517_X`
  #   (because `_shopping_` is followed by `admin_`, not 8 digits).
  if [[ "${this_benchmark}" == "wa" ]]; then
    this_site_pattern="_wa_${this_site}_[0-9]{8}_"
  else
    # VWA pattern: match `_<site>_<date>_` but exclude `_wa_<site>_<date>_`.
    # Done with grep -v at call site since pgrep ERE lacks negative lookbehind.
    this_site_pattern="_${this_site}_[0-9]{8}_"
  fi
  # Helper: returns 0 if any process matches site pattern for given baseline,
  # 1 otherwise. Excludes WA processes when this_benchmark=vwa.
  _collision_match() {
    local baseline_pat="$1"
    local out
    out="$(pgrep -af "run_experiment.*${baseline_pat}.*${this_site_pattern}" 2>/dev/null || true)"
    if [[ -z "${out}" ]]; then
      return 1
    fi
    if [[ "${this_benchmark}" == "vwa" ]]; then
      # Exclude WA processes captured by substring overlap.
      out="$(echo "${out}" | grep -v "_wa_" || true)"
      [[ -z "${out}" ]] && return 1
    fi
    return 0
  }

  if [[ -n "${this_baseline}" && -n "${this_site}" ]]; then
    # Cross-baseline collision (different baseline on same site)
    for other_baseline in B0 B1 B2; do
      [[ "${other_baseline}" == "${this_baseline}" ]] && continue
      if _collision_match "${other_baseline}_"; then
        log "  [collision] ${other_baseline} runner already active on site=${this_site} (benchmark=${this_benchmark})"
        log "  paper-grade hard rule: only one baseline may run on a site at a time"
        log "  waiting for ${other_baseline} ${this_site} to finish before launching ${this_baseline}..."
        while _collision_match "${other_baseline}_"; do
          sleep 60
        done
        log "  ${other_baseline} ${this_site} finished; proceeding with ${this_baseline}"
      fi
    done

    # B-129 fix 2026-05-15 (codex Mode B P1-3): same-baseline + same-site
    # mode collision check (e.g. B0_dom_reddit + B0_som_reddit launched
    # concurrently outside the master gate). Same reddit user account =
    # RESET_BEFORE between modes wipes the other's session. Master gate
    # `queue_phase1_paper_grade.sh` already blocks any active run before
    # launching new chains, so this check defends against manual bypass.
    if _collision_match "${this_baseline}_"; then
      log "  [collision] same-baseline ${this_baseline} runner already active on site=${this_site} (different mode, benchmark=${this_benchmark})"
      log "  paper-grade hard rule: same baseline same site = shared docker user account + RESET_BEFORE race"
      log "  waiting for existing ${this_baseline} ${this_site} run to finish before launching new mode..."
      while _collision_match "${this_baseline}_"; do
        sleep 60
      done
      log "  same-baseline ${this_baseline} ${this_site} finished; proceeding"
    fi
  fi

  # Launch via the queue script (idempotent — picks up existing or fresh+reset).
  # FORCE_NEW propagated explicitly (codex stress v6 C1) — paper-grade master chain
  # exports FORCE_NEW=1 so each cell gets a fresh timestamped run_id, never resumes
  # a pre-fix archived dir.
  #
  # B-301 (A1.17 P1-4, codex OOB unique): pre-fix `out=$(... || true)` discarded
  # queue script rc; reset/auth gate failure was hidden because run_id had been
  # printed before reset → chain proceeded to wait_for_runner_done finding no
  # runner → declared "done" instantly → fell through to silent sentinel check.
  # New behavior: capture rc explicitly; nonzero rc + no run_id printed → fatal;
  # nonzero rc + run_id printed (idempotent-skip case where queue script already
  # had complete data) → continue (legacy path).
  #
  # B-633 (A1.13 P0-4 Claude OOB unique, 2026-05-17): pre-fix `set +e; ...;
  # set -e` bracket — script top is `set -uo pipefail` (NO -e), so set +e is
  # no-op and the trailing set -e *activates* errexit for the rest of chain.
  # Combined with pipefail, `run_id=$(... | grep -oP '...' | tail -1)` would
  # silently exit on grep-no-match (queue script failure printing no run_id=
  # line). Explicit FATAL log at "[FATAL] queue script rc=" was UNREACHABLE
  # dead code. Empirically verified: `set -uo pipefail; set +e; set -e;
  # x=$(echo nope | grep something | tail -1)` exits script with rc=1.
  # Fix: no set -e flip; rely on `|| true` defense on grep extracts so the
  # downstream `[[ -z "${run_id}" ]]` FATAL block runs as designed.
  out=$(FORCE_NEW="${FORCE_NEW:-0}" RESET_BEFORE="${RESET_FLAG}" bash "${SCRIPT_DIR}/${script_name}" \
        ${cmd#${script_name} } 2>&1)
  queue_rc=$?
  echo "$out" | sed 's/^/    /'

  # Extract run_id + condition_id from queue script output.
  # `|| true` defense (B-633 P0-4): without it, grep-no-match + pipefail + -e
  # would silently exit before reaching the explicit empty-string FATAL below.
  run_id=$(echo "$out" | grep -oP 'run_id=\K\S+' | tail -1 || true)
  cond_id=$(echo "$out" | grep -oP 'condition=\K\S+' | tail -1 || true)

  # B-301 P1-4: nonzero queue rc + no run_id minted = reset/auth FATAL or
  # arg-parse error. Surface + abort. Nonzero rc + run_id minted = legacy
  # idempotent-skip with stale run_dir; allow through but warn.
  if [[ "${queue_rc}" != "0" ]]; then
    if [[ -z "${run_id}" ]]; then
      log "  [FATAL] queue script rc=${queue_rc}, no run_id minted — reset/auth/arg error"
      log "  full output above; aborting chain"
      exit 1
    fi
    log "  [warn] queue script rc=${queue_rc} but run_id=${run_id} minted (idempotent-skip path?)"
  fi

  if [[ -z "$run_id" ]]; then
    log "  [error] could not extract run_id from queue script output, aborting"
    exit 1
  fi
  if [[ -z "$cond_id" ]]; then
    log "  [error] could not extract condition id from queue script output, aborting"
    exit 1
  fi
  log "  watching run_id=${run_id} condition=${cond_id}"

  wait_for_runner_done "$run_id" "[${idx}/$#] $cmd"

  # ---- C3 completion sentinel ----
  # History: codex stress v6 (2026-05-14) introduced JSON validity check; A1.13
  # (2026-05-16) added condition_id field check + episodes field semantics;
  # A1.13 fix-scope batch (2026-05-17) hardens 3 ways:
  #
  #   B-631 (P0-2 codex F2 + gemini G1 2-AI OOB): site-key lookup. Pre-fix loop
  #     `classifieds reddit shopping wa_shopping...` substring-match break-first
  #     made `B0_dom_wa_shopping_<ts>` hit `shopping` (substring) before
  #     `wa_shopping` → wrong expected_n (466 VWA vs 192 WA). Fix: parse
  #     benchmark + site STRUCTURALLY from run_id pattern (longer-prefix-first
  #     OR explicit wa_ prefix check), no substring fallthrough.
  #
  #   B-632 (P0-3 Claude OOB): bash `${cond_id!r}` parameter expansion is
  #     invalid (bash treats `${var@op}` w/ op `r` as bad substitution exit 1).
  #     Pre-fix python heredoc f-string `expected ${cond_id!r}` would bash-parse
  #     fail on the cid != cond_id branch. Fix: export EXPECTED_CID env var,
  #     reference via os.environ in python (no bash `!r` parsing).
  #
  #   B-635 (P1-6 codex F5 OOB): pre-fix `ep < expected * 0.9` accepted 90%
  #     partial cells as paper-grade complete. User directive 2026-05-17
  #     ("应该百分百 paper grade"): require exact match. `PAPER_GRADE_ALLOW_PARTIAL=1`
  #     env override for explicit pilot/dirty mode. SITE_EXPECTED_N values
  #     switched to post-exclusion scored_task_count (cls=224 / red=205 /
  #     shop=435 per memory `reference_fp_architecture_2026-05-14`); WA stays
  #     at pre-exclusion since WA has no N/A taxonomy (per prereg).
  declare -A SITE_EXPECTED_N=(
    [classifieds]=224 [reddit]=205 [shopping]=435
    [wa_shopping]=192 [wa_shopping_admin]=182 [wa_reddit]=106
  )

  # B-631 P0-2: structural site-key parse, NOT substring loop. Benchmark
  # determined by `_wa_` substring; site is the token after benchmark prefix.
  expected_n=0
  parsed_site=""
  if [[ "${run_id}" == *_wa_shopping_admin_* ]]; then parsed_site="wa_shopping_admin"
  elif [[ "${run_id}" == *_wa_shopping_* ]];        then parsed_site="wa_shopping"
  elif [[ "${run_id}" == *_wa_reddit_* ]];          then parsed_site="wa_reddit"
  elif [[ "${run_id}" == *_classifieds_* ]];        then parsed_site="classifieds"
  elif [[ "${run_id}" == *_reddit_* ]];             then parsed_site="reddit"
  elif [[ "${run_id}" == *_shopping_* ]];           then parsed_site="shopping"
  fi
  if [[ -n "${parsed_site}" ]]; then
    expected_n="${SITE_EXPECTED_N[${parsed_site}]}"
  fi

  # B-632 P0-3: export EXPECTED_CID for safe python f-string repr (no bash `!r`).
  export EXPECTED_CID="${cond_id}"
  export EXPECTED_N="${expected_n}"

  # B-675 (/stress A1.14 Chunk a P0-4, gemini Mode C F3 unique OOB, 2026-05-17):
  # mktemp per-iteration replaces hardcoded /tmp/queue_chain_sentinel_err. Pre-fix
  # parallel cls + red queue_chain instances (spawned by `queue_phase1_paper_grade.sh`
  # launch_chain) raced on the same /tmp file → one chain's FAIL stderr overwrote
  # by the other's success → debug impossible, audit trail corrupted.
  summary_found=""
  sentinel_err="$(mktemp -t queue_chain_sentinel.XXXXXX 2>/dev/null || echo /tmp/queue_chain_sentinel_$$_${RANDOM}_err)"
  # shellcheck disable=SC2064
  trap "rm -f '${sentinel_err}' 2>/dev/null" RETURN EXIT
  for base in results/visualwebarena/phase1 results/webarena/phase1; do
    cand="${REPO_DIR}/${base}/${run_id}/${cond_id}/condition_summary_v2.json"
    if [[ -s "${cand}" ]]; then
      if EXPECTED_CID="${cond_id}" EXPECTED_N="${expected_n}" SUMMARY_PATH="${cand}" python3 -c "
import json, os, sys
expected_cid = os.environ.get('EXPECTED_CID', '')
try:
    expected_n = int(os.environ.get('EXPECTED_N', '0'))
except ValueError:
    expected_n = 0
summary_path = os.environ.get('SUMMARY_PATH', '')
try:
    d = json.load(open(summary_path))
except Exception as e:
    print(f'invalid JSON: {e}', file=sys.stderr); sys.exit(1)
cid = d.get('condition_id', '')
# B-632 P0-3: safe f-string repr now that values come via env, not bash heredoc interp.
if cid and cid != expected_cid:
    print(f'condition_id mismatch: got {cid!r}, expected {expected_cid!r}', file=sys.stderr); sys.exit(2)
# Canonical field is 'episodes' (int count); legacy fallbacks kept for forward-compat.
ep = d.get('episodes', d.get('total_tasks', d.get('num_tasks', d.get('scored_task_count', 0))))
if not isinstance(ep, int) or ep <= 0:
    print(f'episodes invalid: {ep!r}', file=sys.stderr); sys.exit(3)
# B-635 P1-6: exact-match by default (user directive 2026-05-17 '应该百分百 paper grade').
# PAPER_GRADE_ALLOW_PARTIAL=1 env enables explicit pilot/dirty fallback (warn + advance).
if expected_n > 0 and ep != expected_n:
    if os.environ.get('PAPER_GRADE_ALLOW_PARTIAL') == '1':
        print(f'WARN partial cell {ep}/{expected_n} = {100*ep/expected_n:.1f}% (PAPER_GRADE_ALLOW_PARTIAL=1; degraded mode)', file=sys.stderr)
    else:
        print(f'FATAL episodes={ep} != expected={expected_n} ({100*ep/expected_n:.1f}%); abort. Set PAPER_GRADE_ALLOW_PARTIAL=1 for explicit dirty/pilot mode.', file=sys.stderr); sys.exit(4)
sys.exit(0)
" 2>"${sentinel_err}"; then
        summary_found="${cand}"; break
      else
        err="$(cat "${sentinel_err}" 2>/dev/null || true)"
        log "  [error] ${cand} present but FAILED validation: ${err}"
        log "  treating as missing summary — paper-grade aborts on invalid content"
      fi
    fi
  done
  rm -f "${sentinel_err}" 2>/dev/null || true
  if [[ -z "${summary_found}" ]]; then
    log "  [error] ${run_id}/${cond_id} — no valid condition_summary_v2.json"
    log "  failure modes: runner crash mid-write / FORCE_NEW same-second collision /"
    log "                 schema-version mismatch / stale prior-run dir / disk full"
    log "  aborting chain to prevent silent partial-data advancement"
    if command -v curl > /dev/null; then
      curl -d "queue_chain ABORT: ${run_id}/${cond_id} sentinel validation failed" \
        "ntfy.sh/${NTFY_TOPIC:-p79-exp-dgx-spark}" 2>/dev/null || true
    fi
    exit 1
  fi
  log "  ${run_id}: completion sentinel OK (${summary_found})"

  # B-646 (A1.13 P1-7): release per-(site,benchmark) flock for next iteration.
  # `exec 9>&-` closes fd 9 → kernel releases the advisory lock automatically.
  if [[ -n "${this_site:-}" && -n "${this_benchmark:-}" ]]; then
    exec 9>&-
    log "  [lock] released ${LOCK_FILE:-(unset)}"
  fi
done

log ""
log "=================================================="
log "queue_chain done — $# cells complete"
log "=================================================="

# ntfy notify
if command -v curl > /dev/null; then
  curl -d "queue_chain done: $# cells (${*})" \
    "ntfy.sh/${NTFY_TOPIC:-p79-exp-dgx-spark}" 2>/dev/null || true
fi
