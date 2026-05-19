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

  log "=== Gate 2: Pass-1 baseline completion (36-mode exact, B-1587) ==="
  # B-1587 (/stress A1.24 post-fire P1-7-B codex Mode B F5 OOB, 2026-05-18):
  # gate strengthened from cell-coverage heuristic (6/6 cells × at-least-one
  # mode summary) to mode-exact (6 cells × 6 modes = 36 condition summaries).
  # Pre-fix glob accepted ANY mode summary per cell → LR router could train on
  # partial Pass-1 oracle labels (e.g. cls B0 has dom done but no som/vision/
  # phantom_*) → silent class-balance / feature-coverage contamination of H10
  # PRIMARY estimand. Phantom queue scripts use canonical `phase1_<MODE>_router_0`
  # condition_id (dom/som/vision + phantom_text/phantom_som/phantom_prompt) so
  # one glob covers all 6 modes per cell.
  local baseline_done=0          # full-cell count (6/6 distinct canonical modes)
  local total_mode_summaries=0   # global tally across all cells × modes
  local cell_status=""           # diagnostic string for log
  # P0-4-B (/stress Phase 0 unified bug list 2026-05-19, codex unique): pre-fix
  # `ls -d ...*/condition_summary_v2.json | wc -l` counted DUPLICATE re-fires +
  # archive stale runs, not 6 distinct canonical modes. Archive 2 stale dom
  # runs + 1 som run → count=3 but mode coverage = {dom, som} only → LR oracle
  # trained on partial mode coverage → paper §6 H10 Pareto silently invalid.
  # Post-fix: enumerate 6 canonical mode condition_ids per cell, count exact
  # distinct-mode matches (each mode must have ≥1 paper-grade summary).
  # Canonical mode → condition_id mapping (per `conditions.py` + queue scripts):
  #   dom         → phase1_dom_router_0
  #   som         → phase1_som_router_0
  #   vision      → phase1_vision_router_0
  #   phantom_text  → phase1_phantom_text_router_0  (P-text)
  #   phantom_prompt → phase1_phantom_prompt_router_0 (P-prompt)
  #   phantom_som  → phase1_phantom_som_router_0   (P-SoM)
  local _canonical_cond_ids=(
    "phase1_dom_router_0"
    "phase1_som_router_0"
    "phase1_vision_router_0"
    "phase1_phantom_text_router_0"
    "phase1_phantom_prompt_router_0"
    "phase1_phantom_som_router_0"
  )
  for baseline in B0 B1 B2; do
    for site in classifieds reddit; do
      local cell_mode_count=0
      local cell_modes_present=""
      for cond_id in "${_canonical_cond_ids[@]}"; do
        # Glob expands to 0..N run dirs containing this condition_id's summary.
        # Each canonical mode counts once IF ≥1 paper-grade summary exists.
        if compgen -G "results/visualwebarena/phase1/${baseline}_*_${site}_*/${cond_id}/condition_summary_v2.json" > /dev/null 2>&1; then
          cell_mode_count=$((cell_mode_count + 1))
          cell_modes_present="${cell_modes_present}${cond_id##phase1_},"
        fi
      done
      total_mode_summaries=$((total_mode_summaries + cell_mode_count))
      if [ "$cell_mode_count" -ge 6 ]; then
        baseline_done=$((baseline_done + 1))
      fi
      cell_status="${cell_status}${baseline}_${site}=${cell_mode_count}/6 "
    done
  done
  log "  cell coverage: ${cell_status}"
  log "  total mode summaries: ${total_mode_summaries}/36"
  if [ "$baseline_done" -lt 6 ] || [ "$total_mode_summaries" -lt 36 ]; then
    log "  WARN: ${baseline_done}/6 cells fully complete, ${total_mode_summaries}/36 total mode summaries."
    log "        Pass-2 router train fold needs Pass-1 per-task oracle labels across ALL 6 modes per cell."
    # B-1644 (/stress A2.10 P1-7-A 2026-05-18): paper-grade parity with B-879
    # P0-6-B* ALLOW_ACTIVE_RUNS removal. Pass-2 router fire is paper-grade
    # scope; training Pass-2 LR on partial Pass-1 oracle labels = underspecified
    # class balance + feature coverage = silent contamination of H10 PRIMARY
    # estimand. Bypass HARD-BLOCKED when P79_PAPER_GRADE=1 (the standing env
    # mode for queue_phase1_router_paper_grade.sh per L83 export above).
    if [ "${P79_PAPER_GRADE:-0}" = "1" ]; then
      log "  FAIL: ALLOW_PARTIAL_BASELINE bypass DISALLOWED in paper-grade scope (B-1587)"
      log "        (mirrors B-879 P0-6-B* ALLOW_ACTIVE_RUNS removal pattern)."
      log "        Wait for ALL 36 Pass-1 mode summaries before Pass-2 fire."
      errors=$((errors+1))
    elif [ "${ALLOW_PARTIAL_BASELINE:-0}" != "1" ]; then
      log "  FAIL: Set ALLOW_PARTIAL_BASELINE=1 to bypass (router LR will train on partial data)."
      errors=$((errors+1))
    else
      log "  WARN bypass: ALLOW_PARTIAL_BASELINE=1 — proceeding with partial baseline."
    fi
  else
    log "  OK ($baseline_done/6 cells × 6 modes = $total_mode_summaries/36 mode summaries complete)"
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

  log "=== Gate 4: LR fold-aware artifact bundle (A2.8 P0-4-AB* B-1558 + A2.10 P0-3-B B-1642) ==="
  # A2.8 P0-4-AB* B-1558 (/stress 2026-05-18): pre-A2.8 gate checked only legacy
  # single-pickle path. Now: gate exercises fold-aware artifact runtime path.
  #
  # A2.10 P0-3-B B-1642 (/stress 2026-05-18 Claude Mode A solo, user Q4=B): vectorizer
  # naming corrected. Pre-fix gate expected `{cell_id}_vectorizer_fold{k}.pkl` (per-cell
  # convention) but loader `learned_router.py:load_vectorizer_fold` AND trainer
  # `train_l1_router_with_mi.py:354` both write/read `vectorizer_fold{k}.pkl` (shared
  # across cells, per fold). Gate-vs-runtime contract drift: trainer wrote shared name,
  # loader read shared name, but gate expected per-cell name → gate would always FAIL
  # on a correctly-trained bundle, forcing `ALLOW_NO_LR_MODEL=1` bypass and masking
  # the real artifact contract. Per user-confirmed final E'' router design 2026-05-18:
  # vectorizer + selected_idx are SHARED fold-local across cells (fit on pooled
  # train-fold tasks from all 6 cells) — one vocab + one MI-top-18 mask per fold.
  # Only LR heads + fold_assignment + cell_meta are per-cell. Correct path count:
  #   - 10 shared-fold-local (5 folds × 2: vectorizer_fold{k}.pkl + selected_idx_fold{k}.json)
  #   - 12 per-cell meta (6 cells × 2: {cell_id}_fold_assignment.json + {cell_id}_lr_meta.json)
  #   - 30 per-cell × per-fold LR (6 cells × 5: {cell_id}_lr_fold{k}.pkl)
  #   = 52 paths total (NOT 102 as pre-fix gate falsely claimed)
  local missing_fold=0
  local missing_legacy=0
  local total_expected=0

  # Shared-across-cells fold-local feature machinery (10 paths)
  for k in 0 1 2 3 4; do
    for fn in "vectorizer_fold${k}.pkl" "selected_idx_fold${k}.json"; do
      path="results/phantom_paper/l1_router/${fn}"
      total_expected=$((total_expected+1))
      [[ ! -f "$path" ]] && { log "  Missing shared-fold: $path"; missing_fold=$((missing_fold+1)); }
    done
  done

  # Per-cell artifacts (12 meta + 30 LR-heads = 42 paths)
  for baseline in B0 B1 B2; do
    for site in classifieds reddit; do
      cell_id="${baseline}_${site}"
      # Per-cell meta (2 per cell)
      for suffix in "_fold_assignment.json" "_lr_meta.json"; do
        path="results/phantom_paper/l1_router/${cell_id}${suffix}"
        total_expected=$((total_expected+1))
        [[ ! -f "$path" ]] && { log "  Missing per-cell meta: $path"; missing_fold=$((missing_fold+1)); }
      done
      # Per-cell × per-fold LR head (5 per cell)
      for k in 0 1 2 3 4; do
        path="results/phantom_paper/l1_router/${cell_id}_lr_fold${k}.pkl"
        total_expected=$((total_expected+1))
        [[ ! -f "$path" ]] && { log "  Missing per-cell LR: $path"; missing_fold=$((missing_fold+1)); }
      done
      # Legacy single-pickle back-compat smoke (NOT paper-grade gate; informational)
      legacy_path="results/phantom_paper/l1_router/${cell_id}_lr.pkl"
      [[ ! -f "$legacy_path" ]] && missing_legacy=$((missing_legacy+1))
    done
  done

  if [ "$missing_fold" -gt 0 ]; then
    log "  FAIL: ${missing_fold}/${total_expected} fold-aware artifact paths missing."
    log "        Run scripts/analysis/extract_50_features.py + train_l1_router_with_mi.py"
    log "        + train_l1_router.py per cell post-Pass-1 (A2.5 Chunk A+B substrate)."
    if [ "${ALLOW_NO_LR_MODEL:-0}" != "1" ]; then
      log "  FAIL: Set ALLOW_NO_LR_MODEL=1 to bypass for scaffolding (paper-grade fire BLOCKED)."
      errors=$((errors+1))
    fi
  else
    log "  OK (all ${total_expected} fold-aware paths present: 10 shared-fold + 12 per-cell meta + 30 per-cell LR-heads)"
    # B-1643 (/stress A2.10 P1-5-A 2026-05-18): when existence check passes,
    # run Python validate-each-pickle preflight — catches corrupt pickle /
    # numpy version mismatch / sklearn version drift / partial-write at gate
    # time rather than at first runtime task (where it would fall through
    # to LearnedRouterArtifactError per B-1640, killing the cell loudly but
    # late). Existence check alone is overconfident — paths exist ≠ pickle
    # loadable. ~5-10s preflight cost; catches the corruption failure mode
    # invisible to `-f` check. See `scripts/queues/_lib_lr_artifact_validate.py`.
    local validate_script="${REPO_DIR}/scripts/queues/_lib_lr_artifact_validate.py"
    if [ -f "$validate_script" ]; then
      log "  Running validate-each-pickle preflight..."
      if .venv/bin/python3 "$validate_script" 2>&1 | sed 's/^/    /'; then
        log "  OK (all artifacts load cleanly via runtime path)"
      else
        log "  FAIL: artifact validate-each-pickle preflight FAILED"
        if [ "${ALLOW_CORRUPT_LR_ARTIFACT:-0}" != "1" ]; then
          errors=$((errors+1))
        else
          log "  WARN bypass: ALLOW_CORRUPT_LR_ARTIFACT=1 — proceeding (paper-grade BLOCKED)"
        fi
      fi
    else
      log "  WARN: validate-each-pickle preflight script not found at $validate_script — skipped"
    fi
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
    #
    # R2-P1-7-B* (/stress Phase 0 post-fix Mode B codex F4 OOB, 2026-05-19):
    # parity with Pass-1 baseline orchestrator preflight pattern
    # (queue_phase1_paper_grade.sh:289). Pre-fix Pass-2 router missed
    # (a) `STRICT_PORTS=1 --strict-ports` defense-in-depth (Gate 4 hardening
    # B-680), (b) `|| preflight_rc=$?` set -e safety. Post-fix: explicit
    # parity so Pass-2 router preflight has same hardening as Pass-1 baseline.
    preflight_rc=0
    preflight_out=$(STRICT_PORTS=1 bash scripts/preflight_v2.sh --strict-ports --paper-grade 2>&1) || preflight_rc=$?
    echo "$preflight_out" | tail -8 | sed 's/^/    /'
    if [ "$preflight_rc" -ne 0 ]; then
      log "  FAIL: preflight rc=$preflight_rc — paper-grade Pass-2 fire requires all checks pass"
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
  # B-1586 (/stress A1.24 post-fire P1-6-B, 2026-05-18): `|| true` + anchored
  # pgrep pattern mirroring baseline B-677 + B-709 fix. Pre-fix
  # `pgrep -f "run_experiment.*--config" | wc -l` no-match → pipefail exit
  # under `set -euo pipefail` → clean-host (zero active runs) Pass-2 launch
  # self-aborted at gate before any real failure. Sibling-propagation gap
  # codex Mode B F4 caught.
  active=$(pgrep -fa "(python|\.venv/bin/python3?)[a-zA-Z0-9_./-]* .*run_experiment\.py.*--config" 2>/dev/null | wc -l || true)
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
  # P0-2-B* (/stress Phase 0 unified bug list 2026-05-19): done-sentinel parity
  # with baseline orchestrator (`queue_phase1_paper_grade.sh:launch_chain`).
  # Inner queue_chain.sh exit code is captured + written to .done file so
  # cls→red sequential cascade reads sentinel for actual rc, not `kill -0`
  # liveness alone. See Fire-3 cascade Fire-3 2026-05-19 00:53/00:54 RCA.
  local donefile="logs/queue_phase1_router_${label}_${ts}.done"
  local latest_done="logs/queue_phase1_router_${label}.latest.done"
  mkdir -p logs

  local args=()
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    args+=("$line")
  done < <($builder)

  log "Launching $label router chain (${#args[@]} cells) → $logfile"
  # P0-2-B* sentinel wrapper (parity baseline orchestrator).
  FORCE_NEW=1 RESET_BEFORE=1 nohup bash -c "
    bash scripts/queues/queue_chain.sh \"\$@\"
    _rc=\$?
    printf 'rc=%d ts=%s label=%s pid=%d\n' \"\$_rc\" \"\$(date -u +%FT%TZ)\" '$label' \"\$\$\" > '$donefile'
    exit \$_rc
  " _ "${args[@]}" > "$logfile" 2>&1 &
  local pid=$!
  # Update .latest.log + .latest.done symlinks for live tail + sentinel read.
  rm -f "$latest_log" 2>/dev/null || true
  ln -s "$(basename "$logfile")" "$latest_log" 2>/dev/null || true
  rm -f "$latest_done" 2>/dev/null || true
  ln -s "$(basename "$donefile")" "$latest_done" 2>/dev/null || true
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
        # P0-3-B (/stress Phase 0 unified bug list 2026-05-19, codex unique):
        # Pre-fix Pass-2 router default fired cls+red PARALLEL. Violates
        # CLAUDE.md hard rule #3 "paper-grade fire 同物理 host 同时只能跑一条
        # site chain (cls XOR red XOR shop)" — Pass-1 already sequential via
        # baseline orchestrator (B-1663) to avoid A100 docker bridge +
        # Postgres + Redis + B0 proxy quota contention. Pass-2 router fire
        # has same substrate contention surface; default must also be
        # sequential. PHASE1A_PARALLEL=1 opt-in dev mode preserved (NOT
        # paper-grade).
        if [[ "${PHASE1A_PARALLEL:-0}" != "1" ]]; then
          log "Sequential paper-grade Pass-2 (P0-3-B): cls → red after cls completion."
          launch_chain "cls" build_cls_router_chain
          _cls_pid_file="logs/queue_phase1_router_cls.pid"
          if [[ ! -f "$_cls_pid_file" ]]; then
            fail "Pass-2 cls pid file missing: $_cls_pid_file — launch_chain failed?"
          fi
          _cls_pid=$(cat "$_cls_pid_file")
          # P0-2-B* sentinel-based wait (parity baseline orchestrator).
          log "Waiting for Pass-2 cls router chain pid=${_cls_pid} (max 8h, smaller than baseline 24h since router cells are 1 condition each)..."
          _wait_elapsed=0
          while kill -0 "$_cls_pid" 2>/dev/null && (( _wait_elapsed < 28800 )); do
            sleep 60
            _wait_elapsed=$((_wait_elapsed + 60))
            if (( _wait_elapsed % 1800 == 0 )); then
              log "  Pass-2 cls chain pid=${_cls_pid} still running (${_wait_elapsed}s elapsed)"
            fi
          done
          if kill -0 "$_cls_pid" 2>/dev/null; then
            fail "Pass-2 cls chain pid=${_cls_pid} alive after 8h max-wait — investigate manually"
          fi
          sleep 2
          _cls_done_sentinel="logs/queue_phase1_router_cls.latest.done"
          _cls_rc=1
          if [[ -f "$_cls_done_sentinel" ]]; then
            _cls_rc=$(grep -oE 'rc=-?[0-9]+' "$_cls_done_sentinel" | head -1 | cut -d= -f2)
            _cls_rc="${_cls_rc:-1}"
            log "  Pass-2 cls chain done sentinel: $(cat "$_cls_done_sentinel" 2>/dev/null | head -1)"
          else
            log "  Pass-2 cls chain done sentinel ABSENT — treating as rc=1"
          fi
          if (( _cls_rc != 0 )); then
            fail "Pass-2 cls chain pid=${_cls_pid} exited rc=${_cls_rc} — paper-grade cascade halt (P0-2-B* + P0-3-B): NOT launching red Pass-2 chain. Investigate cls failure; manually re-fire after fix."
          fi
          log "Pass-2 cls chain pid=${_cls_pid} done rc=${_cls_rc}; launching red Pass-2 chain"
          launch_chain "red" build_red_router_chain
        else
          log "PHASE1A_PARALLEL=1 set — DEV MODE Pass-2 parallel fire (NOT paper-grade per CLAUDE.md hard rule #3)"
          launch_chain "cls" build_cls_router_chain
          launch_chain "red" build_red_router_chain
        fi
        ;;
      cls)
        # R2-P0-1-B* (/stress Phase 0 post-fix Mode B codex F1 OOB, 2026-05-19):
        # Pass-2 router single-site launch refuses if another site chain alive
        # (parity with Pass-1 baseline orchestrator). PHASE1A_PARALLEL=1
        # dev opt-in preserved via lib helper internal check.
        assert_no_other_site_chain_running "cls" "queue_phase1_router"
        launch_chain "cls" build_cls_router_chain
        ;;
      red)
        assert_no_other_site_chain_running "red" "queue_phase1_router"
        launch_chain "red" build_red_router_chain
        ;;
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
