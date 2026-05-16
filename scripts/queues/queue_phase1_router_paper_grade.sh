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
# Sentinel files:
#   results/visualwebarena/phase1/<run_id>/<condition_id=phase1_learned_router>/condition_summary_v2.json

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_DIR}"

MODE="${1:-dry-run}"
SITE_FILTER="${2:-all}"

log() { echo "[router-phase1 $(date '+%H:%M:%S')] $*"; }
fail() { log "FAIL: $*"; exit 1; }

# ---------------------------------------------------------------------------
# Pre-launch gates
# ---------------------------------------------------------------------------

check_gates() {
  local errors=0

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

  log "=== Gate 4: LR model artifacts ==="
  local missing_lr=0
  for baseline in B0 B1 B2; do
    for site in classifieds reddit; do
      lr_path="results/phantom_paper/l1_router/${baseline}_${site}_lr.pkl"
      if [[ ! -f "$lr_path" ]]; then
        log "  Missing: $lr_path"
        missing_lr=$((missing_lr+1))
      fi
    done
  done
  if [ "$missing_lr" -gt 0 ]; then
    log "  WARN: $missing_lr/6 LR models missing."
    log "        Run scripts/analysis/train_l1_router.py per cell post-Pass-1."
    if [ "${ALLOW_NO_LR_MODEL:-0}" != "1" ]; then
      log "  FAIL: Set ALLOW_NO_LR_MODEL=1 to bypass for scaffolding."
      errors=$((errors+1))
    fi
  else
    log "  OK (all 6 LR models present)"
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
    preflight_out=$(bash scripts/preflight_v2.sh 2>&1)
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
  active=$(pgrep -f "run_experiment.*--config" | wc -l)
  if [ "$active" -gt 0 ]; then
    pgrep -af "run_experiment.*--config" | sed 's/^/    /'
    if [ "${ALLOW_ACTIVE_RUNS:-0}" == "1" ]; then
      log "  WARN: $active active runs but ALLOW_ACTIVE_RUNS=1 — proceeding."
    else
      log "  FAIL: $active active run(s); Pass-2 cannot run parallel to Pass-1 on same site."
      errors=$((errors+1))
    fi
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
  local logfile="logs/queue_phase1_router_${label}.log"
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
  log "  PID $pid, log $logfile"
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
