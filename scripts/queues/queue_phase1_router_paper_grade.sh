#!/usr/bin/env bash
# queue_phase1_router_paper_grade.sh — Pass-2 orchestrator for Phase 1a learned router.
# v7 walk-back 2026-05-16 (paper-1 §6 learned-only per user Q3 confirmation).
#
# Scope: 6 operational conditions = 2 sites (cls, red) × 3 models (B0, B1, B2)
# × 1 learned router per cell. Statistical analysis: same 6 (site, model) cells
# as Pass-1 baseline; paired comparison per-task vs baseline-pass oracle.
#
# Gating: Pass-1 baseline pass MUST be done first (paper §1 hook data + per-task
# oracle labels for LR train fold). Pass-2 cannot launch until Pass-1 condition
# summaries complete.
#
# Pre-launch gates (mirror queue_phase1_paper_grade.sh):
#   - preregistration.md status=locked (incl. v7 walk-back H9+H11 DEFER amendment)
#   - env_snapshot + VWA snapshot baselines
#   - VWA reachability (preflight_v2.sh)
#   - GPU+model load smoke
#   - No conflicting active runs
#   - All 6 router config files present
#   + Pass-1 done: per-cell condition_summary_v2.json from baseline pass
#   + LR runtime integration in p79/experiment/runner/ (TODO gate)
#   + LR model artifacts in results/phantom_paper/l1_router/ (TODO gate)
#
# Usage:
#   bash scripts/queues/queue_phase1_router_paper_grade.sh dry-run
#   bash scripts/queues/queue_phase1_router_paper_grade.sh launch        # 6 conditions, cls+red
#   bash scripts/queues/queue_phase1_router_paper_grade.sh launch cls    # 3 conditions, cls only
#   bash scripts/queues/queue_phase1_router_paper_grade.sh launch red    # 3 conditions, red only
#
# Pass-2 conditions (6 total):
#   cls chain (3 conditions, B0 → B1 → B2 sequential):
#     - queue_router_learned.sh B0 classifieds
#     - queue_router_learned.sh B1 classifieds
#     - queue_router_learned.sh B2 classifieds
#   red chain (3 conditions, B0 → B1 → B2 sequential):
#     - queue_router_learned.sh B0 reddit
#     - queue_router_learned.sh B1 reddit
#     - queue_router_learned.sh B2 reddit
#
# Chain dependency: cls and red can run in parallel (different sites). Within each
# chain B0 → B1 → B2 sequential (same-site one-baseline rule). Pass-2 itself cannot
# run in parallel with Pass-1 on same site.
#
# ETA estimates (A100 40GB; Pass-2 = 1 condition/cell vs Pass-1 = 6 conditions/cell,
# 6x fewer conditions means ~6x shorter than Pass-1 router pass):
#   cls chain (3 conditions): B0 (~4h) → B1 (~8h) → B2 (~8h) = 20h ≈ 1 day
#   red chain (3 conditions): B0 (~3.5h) → B1 (~7h) → B2 (~7h) = 17.5h ≈ 0.7 days
#   Total Pass-2 wallclock with 2 parallel chains = ~1 day
#
# Sentinel files (A2.8 P0-5-B* B-1557 cond_id alignment 2026-05-18; pre-fix doc cited
# legacy static "phase1_learned_router" but runner emits per-cell + per-backend pattern):
#   results/visualwebarena/phase1/<run_id>/phase1_learned_router_{backend_id}_{site}/condition_summary_v2.json
#   backend_id = "api_strong" (B0) | "local_4b" (B1) | "local_gemma" (B2)
#   site       = "classifieds" | "reddit"
# Producer: p79/experiment/conditions.py:339-356 (single source of truth).

set -euo pipefail
# B-879 (/stress A1.24 P0-6-B*, 2026-05-17): -e fail-fast added — pre-fix
# `-uo pipefail` would let partial-error orchestrator pass through to chain
# launch (e.g. silent typo in build_cls_router_chain), masking failures
# until cell-level downstream. Baseline orchestrator queue_phase1_paper_grade.sh
# already uses -euo (default Bash strict mode).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_DIR}"

# B-879 P0-6-B*: source shared paper-grade gates lib (parity baseline
# orchestrator). Provides: init_paper_grade_env (loads vwa_env_remote.sh →
# preflight Gate 4 reachability), acquire_site_lock / release_site_lock,
# load_proxy_api_key, mint_run_id, reset_and_auth_gate, assert_a100_url_locality,
# assert_no_cross_mode_collision. Pre-fix: router script had ZERO lib
# integration → wrong-host launches, missing env wires, no orchestrator
# flock → Pass-2 fire fragile.
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_lib_paper_grade_gates.sh"
init_paper_grade_env "${REPO_DIR}"

# B-395 (/stress A1.1 v8 3-AI overlap P0-1, 2026-05-16): paper_grade env wire.
# See `queue_phase1_paper_grade.sh` for full rationale — B-340 GLM hard-block
# reachability gate. Pass-2 router fire is part of paper-grade scope so
# same flag must propagate.
export P79_PAPER_GRADE=1

MODE="${1:-dry-run}"
SITE_FILTER="${2:-all}"

log() { echo "[router-phase1 $(date '+%H:%M:%S')] $*"; }
fail() { log "FAIL: $*"; exit 1; }

# B-879 P0-6-B*: A100 host + URL locality enforcement (parity baseline
# orchestrator queue_phase1_paper_grade.sh).
# B-1406 (/stress A2.7 P1-4-AB* 2026-05-18): local definition retired,
# canonical `require_paper_grade_host` lives in `_lib_paper_grade_gates.sh`
# (already sourced above at L72). Mode A F1 + Mode B F5 caught the sibling-
# propagation regex permissive match + duplicate-def attack vectors.

# ---------------------------------------------------------------------------
# Pre-launch gates
# ---------------------------------------------------------------------------

check_gates() {
  local errors=0

  # B-879 P0-6-B*: orchestrator flock + host check FIRST (parity baseline
  # queue_phase1_paper_grade.sh:L580-595). Pre-fix: two concurrent Pass-2
  # invocations could both pass Gate 6 and both fire chains → bypass
  # queue_chain.sh per-site flock (which fires AFTER chain start). Now:
  # orchestrator-level lock closes the race window.
  ORCH_LOCK_DIR="${REPO_DIR}/.locks"
  mkdir -p "${ORCH_LOCK_DIR}"
  ORCH_LOCK="${ORCH_LOCK_DIR}/phase1_router_orchestrator.lock"
  exec {ORCH_FD}>"${ORCH_LOCK}"
  if ! flock -n -x "${ORCH_FD}"; then
    stale_pid="$(cat "${ORCH_LOCK}" 2>/dev/null || echo unknown)"
    fail "Another paper-grade router orchestrator instance holds ${ORCH_LOCK} (pid ${stale_pid}). Wait for it to complete or kill stale lock holder."
  fi
  echo "$$" > "${ORCH_LOCK}"
  # Lock auto-releases on shell exit (FD closes); trap rm for clean cleanup.
  trap "rm -f '${ORCH_LOCK}' 2>/dev/null; exec {ORCH_FD}>&-; exit" EXIT INT TERM
  log "router orchestrator lock acquired: ${ORCH_LOCK} (pid $$)"

  # B-879 P0-6-B*: host + URL locality enforcement BEFORE gates run.
  require_paper_grade_host

  log "=== Gate 1: preregistration.md threshold lock + v7 walk-back ==="
  if ! grep -q "^status: locked" docs/checkpoints/pre_run/preregistration.md 2>/dev/null; then
    log "  FAIL: preregistration.md status is not 'locked'."
    errors=$((errors+1))
  elif ! grep -q "2026-05-16 v7" docs/checkpoints/pre_run/preregistration.md 2>/dev/null; then
    log "  FAIL: preregistration.md missing v7 walk-back amendment (H9+H11 DEFER)."
    log "        Append Appendix A 2026-05-16 v7 entry before Pass-2 launch."
    errors=$((errors+1))
  else
    log "  OK (status=locked + v7 amendment present)"
  fi

  log "=== Gate 2: Pass-1 baseline completion ==="
  # Heuristic: Pass-1 baseline runs leave per-cell condition_summary_v2.json.
  # We require at least 6 baseline runs done (6 cells × 1 sentinel each is too tight
  # given per-mode conditions; check at least one per-cell run done).
  local baseline_done=0
  for baseline in B0 B1 B2; do
    for site in classifieds reddit; do
      if ls results/visualwebarena/phase1/${baseline}_*_${site}_*/phase1_*_router_0/condition_summary_v2.json &>/dev/null 2>&1; then
        baseline_done=$((baseline_done+1))
      fi
    done
  done
  if [ "$baseline_done" -lt 6 ]; then
    log "  WARN: Only $baseline_done/6 cells have Pass-1 baseline summaries on disk."
    log "        Pass-2 router train fold needs Pass-1 per-task oracle labels."
    if [ "${ALLOW_PARTIAL_BASELINE:-0}" != "1" ]; then
      log "  FAIL: Set ALLOW_PARTIAL_BASELINE=1 to bypass (router LR will train on partial data)."
      errors=$((errors+1))
    else
      log "  WARN bypass: ALLOW_PARTIAL_BASELINE=1 — proceeding with partial baseline."
    fi
  else
    log "  OK ($baseline_done/6 cells with baseline summaries)"
  fi

  log "=== Gate 3: LR runtime integration in runner ==="
  if ! grep -rqE "(observation_mode.*==.*\"?learned\"?|obs_mode.*==.*\"?learned\"?|_dispatch_lr|lr_model_path)" \
      "${REPO_DIR}/p79/experiment/runner/" 2>/dev/null; then
    log "  FAIL: LR runtime integration NOT wired in p79/experiment/runner/."
    log "        Runner must accept observation_mode=\"learned\" and dispatch through LR predictor."
    log "        See proposals_v7.md §3 + scripts/analysis/train_l1_router.py (TODO)."
    if [ "${REQUIRE_LR_RUNTIME:-1}" == "1" ]; then
      errors=$((errors+1))
    else
      log "  WARN bypass: REQUIRE_LR_RUNTIME=0 — proceeding for scaffolding."
    fi
  else
    log "  OK (LR dispatch found in runner)"
  fi

  log "=== Gate 4: LR fold-aware artifact bundle (A2.8 P0-4-AB* B-1558) ==="
  # A2.8 P0-4-AB* B-1558 (/stress 2026-05-18 codex Mode B + Claude Mode A 2-AI OOB):
  # Pre-A2.8 gate checked only legacy single-pickle path (`{baseline}_{site}_lr.pkl`)
  # which does NOT exercise the fold-aware artifact runtime path. Paper-grade Pass-2
  # router fire requires 5 LR fold pickles + 5 TF-IDF vectorizers + 5 selected_idx
  # masks + 1 fold_assignment + 1 cell_meta per cell × 6 cells = 102 paths total.
  # Gate now checks the fold-aware bundle; legacy single-pickle remains a back-compat
  # smoke artifact (verified separately if present, but NOT a paper-grade gate).
  local missing_fold=0
  local missing_legacy=0
  local n_folds=5  # train_l1_router.py N_FOLDS_OUTER constant
  for baseline in B0 B1 B2; do
    for site in classifieds reddit; do
      cell_id="${baseline}_${site}"
      # Per-cell fold-aware bundle (15 path checks per cell × 6 cells = 90 paths)
      for k in 0 1 2 3 4; do
        for suffix in "_lr_fold${k}.pkl" "_vectorizer_fold${k}.pkl"; do
          path="results/phantom_paper/l1_router/${cell_id}${suffix}"
          [[ ! -f "$path" ]] && { log "  Missing fold-aware: $path"; missing_fold=$((missing_fold+1)); }
        done
        sidx_path="results/phantom_paper/l1_router/selected_idx_fold${k}.json"
        [[ ! -f "$sidx_path" ]] && { log "  Missing fold-aware: $sidx_path"; missing_fold=$((missing_fold+1)); }
      done
      # Per-cell meta (2 paths per cell × 6 cells = 12 paths)
      for suffix in "_fold_assignment.json" "_lr_meta.json"; do
        path="results/phantom_paper/l1_router/${cell_id}${suffix}"
        [[ ! -f "$path" ]] && { log "  Missing fold-aware: $path"; missing_fold=$((missing_fold+1)); }
      done
      # Legacy single-pickle back-compat smoke (NOT paper-grade gate; informational)
      legacy_path="results/phantom_paper/l1_router/${cell_id}_lr.pkl"
      [[ ! -f "$legacy_path" ]] && missing_legacy=$((missing_legacy+1))
    done
  done
  if [ "$missing_fold" -gt 0 ]; then
    log "  FAIL: $missing_fold/102 fold-aware artifact paths missing."
    log "        Run scripts/analysis/extract_50_features.py + train_l1_router_with_mi.py"
    log "        + train_l1_router.py per cell post-Pass-1 (A2.5 Chunk A+B substrate)."
    if [ "${ALLOW_NO_LR_MODEL:-0}" != "1" ]; then
      log "  FAIL: Set ALLOW_NO_LR_MODEL=1 to bypass for scaffolding (paper-grade fire BLOCKED)."
      errors=$((errors+1))
    fi
  else
    log "  OK (all 102 fold-aware artifact paths present for 6 cells × 17 paths/cell)"
  fi
  if [ "$missing_legacy" -gt 0 ]; then
    log "  INFO: $missing_legacy/6 legacy single-pickle smoke artifacts missing (NOT a paper-grade gate)."
  fi

  log "=== Gate 5: env_snapshot + VWA snapshot ==="
  if ! ls results/provenance/env_*_baseline.json &>/dev/null; then
    log "  FAIL: No env_*_baseline.json"
    errors=$((errors+1))
  else
    log "  OK"
  fi

  log "=== Gate 6: VWA reachability ==="
  if [ -f scripts/preflight_v2.sh ]; then
    # B-793 (/stress A1.9 cold-start P1-9): --paper-grade flag for evaluator
    # init probe — surface B-544 init-fail at preflight not at batch start.
    preflight_out=$(bash scripts/preflight_v2.sh --paper-grade 2>&1)
    preflight_rc=$?
    echo "$preflight_out" | tail -8 | sed 's/^/    /'
    if [ "$preflight_rc" -ne 0 ]; then
      log "  FAIL: preflight rc=$preflight_rc"
      errors=$((errors+1))
    else
      log "  OK"
    fi
  fi

  log "=== Gate 7: GPU + CUDA ==="
  cuda_ok=$(.venv/bin/python3 -c "import torch; print('YES' if torch.cuda.is_available() else 'NO')" 2>/dev/null)
  if [ "$cuda_ok" = "YES" ]; then
    log "  OK ($(.venv/bin/python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null))"
  else
    log "  FAIL: CUDA not available"
    errors=$((errors+1))
  fi

  log "=== Gate 8: No conflicting active runs ==="
  # B-879 P0-6-B*: removed `ALLOW_ACTIVE_RUNS=1` escape hatch. Pass-2 router
  # fire is paper-grade scope — any active runner on same site = cross-baseline
  # contamination per CLAUDE.md hard rule "same site only one baseline at a
  # time". Was: bypass flag allowed warn-and-proceed; now: hard fail.
  active=$(pgrep -f "run_experiment.*--config" | wc -l)
  if [ "$active" -gt 0 ]; then
    pgrep -af "run_experiment.*--config" | sed 's/^/    /'
    log "  FAIL: $active active run(s); Pass-2 cannot run parallel to Pass-1 on same site (CLAUDE.md hard rule)."
    errors=$((errors+1))
  else
    log "  OK"
  fi

  log "=== Gate 9: All 6 router configs exist ==="
  local missing_cfg=0
  for baseline in B0 B1 B2; do
    for site in classifieds reddit; do
      cfg="configs/exp_v2_${baseline}_router_learned_${site}.yaml"
      if [[ ! -f "$cfg" ]]; then
        log "  Missing: $cfg"
        missing_cfg=$((missing_cfg+1))
      fi
    done
  done
  if [ "$missing_cfg" -gt 0 ]; then
    log "  FAIL: $missing_cfg/6 configs missing"
    errors=$((errors+1))
  else
    log "  OK (all 6 configs present)"
  fi

  if [ "$errors" -gt 0 ]; then
    fail "$errors gate(s) failed; abort."
  fi
  log "All gates passed (or warnings only)."
}

# ---------------------------------------------------------------------------
# Chain definitions (v7 Pass-2: 3 cond/chain, B0 → B1 → B2 sequential)
# ---------------------------------------------------------------------------

build_cls_router_chain() {
  cat <<EOF
queue_router_learned.sh B0 classifieds
queue_router_learned.sh B1 classifieds
queue_router_learned.sh B2 classifieds
EOF
}

build_red_router_chain() {
  cat <<EOF
queue_router_learned.sh B0 reddit
queue_router_learned.sh B1 reddit
queue_router_learned.sh B2 reddit
EOF
}

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

dry_run() {
  log "DRY RUN — no launches will occur."
  log ""
  log "=== Phase 1a Pass-2 (learned router, v7 walk-back) ==="
  log ""
  log "Cls router chain (3 conditions, B0 → B1 → B2):"
  build_cls_router_chain | sed 's/^/  /'
  log ""
  log "Red router chain (3 conditions, B0 → B1 → B2):"
  build_red_router_chain | sed 's/^/  /'
  log ""
  log "Pass-2 total: 6 operational conditions across 6 statistical cells."
  log "Phase 1a total = 36 baseline (Pass-1) + 6 router (Pass-2) = 42 conditions."
  log ""
  log "ETA: ~1 day wallclock if cls + red chains parallel (each chain B0 → B1 → B2 sequential)."
  log ""
  log "Pre-launch TODOs (Gates 3+4 likely currently failing):"
  log "  - LR runtime integration in p79/experiment/runner/main.py (observation_mode=\"learned\" dispatch)"
  log "  - scripts/analysis/train_l1_router.py — train per-cell LR from Pass-1 oracle labels"
  log "  - 6× LR pickle artifacts in results/phantom_paper/l1_router/"
}

launch_chain() {
  local label=$1
  local builder=$2
  # B-879 P0-6-B*: timestamped log filenames (parity baseline orchestrator).
  # Pre-fix: static `logs/queue_phase1_router_<label>.log` → re-fire would
  # overwrite forensic. Now: per-invocation timestamp + `.latest.log` symlink
  # for live tail. Same convention as baseline queue_phase1_paper_grade.sh.
  local ts; ts="$(date +%Y%m%d_%H%M%S)"
  local logfile="logs/queue_phase1_router_${label}_${ts}.log"
  local latest_log="logs/queue_phase1_router_${label}.latest.log"
  mkdir -p logs

  local args=()
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    args+=("$line")
  done < <($builder)

  log "Launching $label router chain (${#args[@]} cells) → $logfile"
  FORCE_NEW=1 RESET_BEFORE=1 nohup bash scripts/queues/queue_chain.sh "${args[@]}" \
    > "$logfile" 2>&1 &
  local pid=$!
  # Update .latest.log symlink for live tail (rm -f tolerant if already-gone)
  rm -f "$latest_log" 2>/dev/null || true
  ln -s "$(basename "$logfile")" "$latest_log" 2>/dev/null || true
  log "  PID $pid, log $logfile (live tail: $latest_log)"
  echo "$pid" > "logs/queue_phase1_router_${label}_${ts}.pid"
  echo "$pid" > "logs/queue_phase1_router_${label}.pid"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

case "$MODE" in
  dry-run)
    dry_run
    ;;
  launch)
    check_gates
    case "$SITE_FILTER" in
      all)
        launch_chain "cls" build_cls_router_chain
        launch_chain "red" build_red_router_chain
        ;;
      cls)  launch_chain "cls" build_cls_router_chain ;;
      red)  launch_chain "red" build_red_router_chain ;;
      *) fail "Unknown site filter: $SITE_FILTER (expected: all|cls|red)" ;;
    esac
    log ""
    log "Pass-2 router launched (6 conditions, cls + red × B0+B1+B2 × 1 learned router). Monitor:"
    log "  - PIDs: cat logs/queue_phase1_router_*.pid"
    log "  - Logs: tail -f logs/queue_phase1_router_*.log"
    log "  - Cells: open Obsidian Bases view 'cells.base'"
    log "  - Active: make active"
    log ""
    log "Post-completion paper §6 analysis:"
    log "  python3 scripts/analysis/aggregate_pareto_router.py  # (TODO)"
    log "  python3 scripts/analysis/loco_cv_l1_router.py        # Phase 1a LOCO main number"
    ;;
  *)
    fail "Unknown mode: $MODE (expected: dry-run | launch)"
    ;;
esac
