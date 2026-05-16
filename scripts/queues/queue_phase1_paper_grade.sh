#!/usr/bin/env bash
# queue_phase1_paper_grade.sh — Master orchestrator for Phase 1 paper-grade rerun.
# (Renamed 2026-05-13 from queue_16cell_paper_grade.sh; old name reflected prior
# 16-cell phantom-only scope that codex stress audit identified as incomplete.)
#
# Scope (revised 2026-05-14 — Gemma3-VL added as 3rd baseline, advisor 笔记 §138):
#   Phase 1a (THIS SCRIPT default): 36 operational conditions = 2 sites (cls, red)
#     × 3 models (B0, B1, B2) × 6 modes (DOM, SoM, Vision, P-text, P-prompt, P-SoM).
#     Statistical analysis: 6 (site, model) cells, pooled meta + TOST.
#     Target: workshop submission. (Was 24 conditions / 4 cells pre-2026-05-14;
#     B2 = Gemma3-VL google/gemma-3-4b-it, cross-family control.)
#   Phase 1b (deferred, requires explicit 'launch phase1b shop'): 18 additional
#     conditions = shop × 3 models × 6 modes. Feeds main paper R3→R1 framing
#     decision post-workshop submission.
#
# **Hard rule: Same site, one baseline only (B0 / B1 / B2)**. queue_chain wraps
# reset+watchdog+idempotent and enforces the 3-way same-site collision check.
# Splits into 2 parallel chains (cls / red) for Phase 1a, each chain internally sequential.
#
# Pre-launch gates (must all pass):
#   - Advisor email reply received → preregistration.md status `draft` → `locked`
#   - A100 SSH connectivity verified ('ssh condense-a100 nvidia-smi' returns OK)
#   - VWA stack running on chosen host (DGX→quark Tailscale OR A100 self-host)
#   - env_snapshot baseline committed (results/provenance/env_<host>_baseline.json)
#   - VWA snapshot baseline committed (results/provenance/vwa_<host>_baseline.json)
#
# Usage:
#   bash scripts/queues/queue_phase1_paper_grade.sh dry-run            # preview, no launch
#   bash scripts/queues/queue_phase1_paper_grade.sh launch             # Phase 1a (cls+red, 36 conditions)
#   bash scripts/queues/queue_phase1_paper_grade.sh launch cls         # only classifieds Phase 1a chain (18 conditions)
#   bash scripts/queues/queue_phase1_paper_grade.sh launch red         # only reddit Phase 1a chain (18 conditions)
#   bash scripts/queues/queue_phase1_paper_grade.sh launch phase1b     # Phase 1b shop chain (18 conditions, deferred to post-workshop)
#
# Phase 1a conditions (36 total):
#   cls chain (18 conditions, B0 → B1 → B2 sequential):
#     - B0 cls × {DOM, SoM, Vision, P-text, P-SoM, P-prompt}  (6)
#     - B1 cls × {DOM, SoM, Vision, P-text, P-SoM, P-prompt}  (6)
#     - B2 cls × {DOM, SoM, Vision, P-text, P-SoM, P-prompt}  (6)
#   red chain (18 conditions, B0 → B1 → B2 sequential):
#     - B0 red × {DOM, SoM, Vision, P-text, P-SoM, P-prompt}  (6)
#     - B1 red × {DOM, SoM, Vision, P-text, P-SoM, P-prompt}  (6)
#     - B2 red × {DOM, SoM, Vision, P-text, P-SoM, P-prompt}  (6)
#
# Phase 1b conditions (18 total, deferred main-paper expansion):
#     - B0 shop × {DOM, SoM, Vision, P-text, P-SoM, P-prompt}  (6)
#     - B1 shop × {DOM, SoM, Vision, P-text, P-SoM, P-prompt}  (6)
#     - B2 shop × {DOM, SoM, Vision, P-text, P-SoM, P-prompt}  (6)
#
# Chain dependency:
#   cls and red can run in parallel (different sites = no resource contention beyond A100 GPU).
#   Within each chain B0 → B1 → B2 sequential (same-site baselines share the site
#   docker container + user account login).
#   Phase 1b shop launched separately after workshop submission to avoid Magento FPC bug
#   surface co-occurring with Phase 1a critical path.
#
# ETA estimates (A100 40GB; B2 ≈ B1 throughput, both ~4B local):
#   cls chain (18 conditions): B0 (~24h) → B1 (~48h) → B2 (~48h) = 120h ≈ 5 days
#   red chain (18 conditions): B0 (~20h) → B1 (~40h) → B2 (~40h) = 100h ≈ 4 days
#   Total Phase 1a wallclock with 2 parallel chains = max(120, 100) ≈ 5 days
#   Phase 1b shop chain (18 conditions): B0 (~32h) → B1 (~64h) → B2 (~64h) = 160h ≈ 6.5 days (deferred)
#
# Sentinel files (used by chain to detect completion):
#   results/visualwebarena/phase1/<run_id>/<condition_id>/condition_summary_v2.json

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_DIR}"

# B-395 (/stress A1.1 v8 3-AI overlap P0-1, 2026-05-16): export P79_PAPER_GRADE
# so `p79/experiment/config.py:normalize_config` sets top-level
# `cfg["paper_grade"] = True` → runner propagates to backend cfg → B0
# `ApiProxyBackend` forwards to agent → B-340 GLM hard-block at
# `proxy_api_agent.py:179-186` raises if any yaml still has
# `use_glm_fallback: true`. Defense-in-depth: (a) all B0 paper-1 yamls
# explicit `use_glm_fallback: false` (B-396), (b) this env wire makes
# B-340 hard-block reachable, (c) B-340 RuntimeError fail-fast at init.
export P79_PAPER_GRADE=1

MODE="${1:-dry-run}"
SITE_FILTER="${2:-all}"

log() { echo "[phase1 $(date '+%H:%M:%S')] $*"; }
fail() { log "FAIL: $*"; exit 1; }

# ---------------------------------------------------------------------------
# Pre-launch gates
# ---------------------------------------------------------------------------

check_gates() {
  local errors=0

  log "=== Gate 1: preregistration.md threshold lock ==="
  if grep -q "K_h1.*TBD\|K_h3.*TBD\|TOST.*TBD" docs/checkpoints/pre_run/preregistration.md 2>/dev/null; then
    log "  FAIL: preregistration.md still has TBD threshold values."
    log "        Wait for advisor email reply, then update preregistration.md."
    errors=$((errors+1))
  elif ! grep -q "^status: locked" docs/checkpoints/pre_run/preregistration.md 2>/dev/null; then
    # Gate 1b added 2026-05-13 (codex audit HIGH-1): launch_checklist.md
    # requires prereg `status: locked` before paper-grade rerun. Previously
    # only TBD threshold was checked; status=draft could pass.
    log "  FAIL: preregistration.md status is not 'locked' (still draft / pending advisor)."
    log "        Once advisor signs, flip 'status: draft' → 'status: locked' in"
    log "        docs/checkpoints/pre_run/preregistration.md before paper-grade launch."
    errors=$((errors+1))
  else
    log "  OK"
  fi

  log "=== Gate 2: env_snapshot baseline committed ==="
  if ! ls results/provenance/env_*_baseline.json &>/dev/null; then
    log "  FAIL: No env_*_baseline.json found in results/provenance/"
    log "        Run: python3 scripts/provenance/snapshot_env.py results/provenance/env_<host>_baseline.json"
    errors=$((errors+1))
  else
    log "  OK ($(ls results/provenance/env_*_baseline.json | head -3 | tr '\n' ' '))"
  fi

  log "=== Gate 3: VWA snapshot baseline committed ==="
  if ! ls results/provenance/vwa_*.json &>/dev/null; then
    log "  WARN: No vwa_*.json found. Recommend bash scripts/provenance/snapshot_vwa.sh"
  else
    log "  OK ($(ls results/provenance/vwa_*.json | head -3 | tr '\n' ' '))"
  fi

  # Gate 4 — BLOCKING (codex stress v6 C2): preflight exit code now captured.
  # Strict ports for actual paper-grade fire (--no-strict-ports dropped).
  log "=== Gate 4: VWA reachability ==="
  if [ -f scripts/preflight_v2.sh ]; then
    preflight_out=$(bash scripts/preflight_v2.sh 2>&1)
    preflight_rc=$?
    echo "$preflight_out" | tail -8 | sed 's/^/    /'
    if [ "$preflight_rc" -ne 0 ]; then
      log "  FAIL: preflight_v2.sh exited rc=$preflight_rc — paper-grade fire requires all preflight checks pass"
      errors=$((errors+1))
    else
      log "  OK"
    fi
  else
    log "  FAIL: scripts/preflight_v2.sh not found"
    errors=$((errors+1))
  fi

  # Gate 5 — BLOCKING (codex stress v6 C2 sibling): CUDA availability now gates launch.
  log "=== Gate 5: GPU + model load smoke ==="
  if command -v .venv/bin/python3 &>/dev/null; then
    cuda_ok=$(.venv/bin/python3 -c "import torch; print('YES' if torch.cuda.is_available() else 'NO')" 2>/dev/null)
    if [ "$cuda_ok" = "YES" ]; then
      log "  OK ($(.venv/bin/python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null))"
    else
      log "  FAIL: CUDA not available — paper-grade B1 local inference requires GPU"
      errors=$((errors+1))
    fi
  else
    log "  FAIL: .venv/bin/python3 not found"
    errors=$((errors+1))
  fi

  # Gate 6 — BLOCKING (codex stress v6 C6): active-run detection now fatal.
  # RESET_BEFORE=1 in launch_chain would reset site state under any active
  # same-site runner. Set ALLOW_ACTIVE_RUNS=1 only after verifying no collision.
  log "=== Gate 6: No conflicting active runs ==="
  active=$(pgrep -f "run_experiment.*--config" | wc -l)
  log "  Active run_experiment processes: $active"
  if [ "$active" -gt 0 ]; then
    pgrep -af "run_experiment.*--config" | sed 's/^/    /'
    if [ "${ALLOW_ACTIVE_RUNS:-0}" == "1" ]; then
      log "  WARN: $active existing run(s) detected but ALLOW_ACTIVE_RUNS=1 — proceeding."
      log "        You are responsible for verifying no same-site B0+B1 collision."
    else
      log "  FAIL: $active existing run_experiment process(es) detected."
      log "        Phase 1a fire with RESET_BEFORE=1 would reset site state under any active same-site runner."
      log "        Stop conflicting runs, OR set ALLOW_ACTIVE_RUNS=1 if you verified no site collision."
      errors=$((errors+1))
    fi
  else
    log "  OK (no active runs)"
  fi

  # Gate 7 — BLOCKING (codex stress v6 C9): every chain config must exist on disk.
  # Prevents advertising a chain that fails mid-run on a missing config.
  log "=== Gate 7: All chain configs exist ==="
  local missing_cfg=0
  local builders_to_check
  case "${SITE_FILTER}" in
    all|cls|red) builders_to_check="build_cls_chain build_red_chain" ;;
    phase1b)     builders_to_check="build_shop_chain" ;;
    *)           builders_to_check="build_cls_chain build_red_chain" ;;
  esac
  for builder in ${builders_to_check}; do
    while IFS= read -r cmd; do
      [ -z "$cmd" ] && continue
      cfg_path="$(config_for_cmd "$cmd")"
      if [ -n "$cfg_path" ] && [ ! -f "$cfg_path" ]; then
        log "  FAIL: missing config $cfg_path  (for: $cmd)"
        missing_cfg=$((missing_cfg+1))
      fi
    done < <($builder)
  done
  if [ "$missing_cfg" -gt 0 ]; then
    log "  $missing_cfg chain config(s) missing — cannot launch"
    errors=$((errors+1))
  else
    log "  OK (all chain configs present)"
  fi

  if [ "$errors" -gt 0 ]; then
    fail "$errors gate(s) failed; abort. Fix and re-run."
  fi
  log "All gates passed (or warnings only)."
}

# ---------------------------------------------------------------------------
# Chain definitions
# ---------------------------------------------------------------------------

# Map a chain command to its config file path. Mirrors the CFG_NAME convention
# in queue_baseline.sh / queue_phantom_*.sh (codex stress v6 C9).
config_for_cmd() {
  local cmd="$1"
  local parts=( $cmd )
  local script="${parts[0]}"
  local baseline="${parts[1]}"
  case "$script" in
    queue_baseline.sh)        # <baseline> <mode> <site>
      echo "configs/exp_v2_${baseline}_${parts[2]}_${parts[3]}.yaml" ;;
    queue_phantom_som.sh)     # <baseline> <site>
      echo "configs/exp_v2_${baseline}_phantom_${parts[2]}.yaml" ;;
    queue_phantom_text.sh)    # <baseline> <site>
      echo "configs/exp_v2_${baseline}_phantom_text_${parts[2]}.yaml" ;;
    queue_phantom_prompt.sh)  # <baseline> <site>
      echo "configs/exp_v2_${baseline}_phantom_prompt_${parts[2]}.yaml" ;;
    *) echo "" ;;
  esac
}

build_cls_chain() {
  # Phase 1a classifieds: 6 modes per model, B0 → B1 → B2 sequential = 18 conditions
  cat <<EOF
queue_baseline.sh B0 dom classifieds
queue_baseline.sh B0 som classifieds
queue_baseline.sh B0 vision classifieds
queue_phantom_text.sh B0 classifieds
queue_phantom_som.sh B0 classifieds
queue_phantom_prompt.sh B0 classifieds
queue_baseline.sh B1 dom classifieds
queue_baseline.sh B1 som classifieds
queue_baseline.sh B1 vision classifieds
queue_phantom_text.sh B1 classifieds
queue_phantom_som.sh B1 classifieds
queue_phantom_prompt.sh B1 classifieds
queue_baseline.sh B2 dom classifieds
queue_baseline.sh B2 som classifieds
queue_baseline.sh B2 vision classifieds
queue_phantom_text.sh B2 classifieds
queue_phantom_som.sh B2 classifieds
queue_phantom_prompt.sh B2 classifieds
EOF
}

build_red_chain() {
  # Phase 1a reddit: 6 modes per model, B0 → B1 → B2 sequential = 18 conditions
  cat <<EOF
queue_baseline.sh B0 dom reddit
queue_baseline.sh B0 som reddit
queue_baseline.sh B0 vision reddit
queue_phantom_text.sh B0 reddit
queue_phantom_som.sh B0 reddit
queue_phantom_prompt.sh B0 reddit
queue_baseline.sh B1 dom reddit
queue_baseline.sh B1 som reddit
queue_baseline.sh B1 vision reddit
queue_phantom_text.sh B1 reddit
queue_phantom_som.sh B1 reddit
queue_phantom_prompt.sh B1 reddit
queue_baseline.sh B2 dom reddit
queue_baseline.sh B2 som reddit
queue_baseline.sh B2 vision reddit
queue_phantom_text.sh B2 reddit
queue_phantom_som.sh B2 reddit
queue_phantom_prompt.sh B2 reddit
EOF
}

build_shop_chain() {
  # Phase 1b deferred: shop × 6 modes per model, B0 → B1 → B2 sequential = 18 conditions
  # B2 (Gemma3-VL) included for baseline consistency — a baseline applies to all
  # sites; Phase 1b's deferral is about launch TIMING, not model scope.
  # NOT launched as part of default `launch` (which is Phase 1a cls + red).
  # Launch via explicit `launch phase1b` after workshop submission.
  cat <<EOF
queue_baseline.sh B0 dom shopping
queue_baseline.sh B0 som shopping
queue_baseline.sh B0 vision shopping
queue_phantom_text.sh B0 shopping
queue_phantom_som.sh B0 shopping
queue_phantom_prompt.sh B0 shopping
queue_baseline.sh B1 dom shopping
queue_baseline.sh B1 som shopping
queue_baseline.sh B1 vision shopping
queue_phantom_text.sh B1 shopping
queue_phantom_som.sh B1 shopping
queue_phantom_prompt.sh B1 shopping
queue_baseline.sh B2 dom shopping
queue_baseline.sh B2 som shopping
queue_baseline.sh B2 vision shopping
queue_phantom_text.sh B2 shopping
queue_phantom_som.sh B2 shopping
queue_phantom_prompt.sh B2 shopping
EOF
}

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

dry_run() {
  log "DRY RUN — no launches will occur."
  log ""
  log "=== Phase 1a (default, workshop-target) ==="
  log ""
  log "Cls chain (18 conditions, 6 modes × B0+B1+B2):"
  build_cls_chain | sed 's/^/  /'
  log ""
  log "Red chain (18 conditions, 6 modes × B0+B1+B2):"
  build_red_chain | sed 's/^/  /'
  log ""
  log "Phase 1a total: 36 operational conditions across 6 statistical cells (= (site, model) tuples)."
  log ""
  log "=== Phase 1b (deferred, main paper expansion) ==="
  log ""
  log "Shop chain (18 conditions, 6 modes × B0+B1+B2):"
  build_shop_chain | sed 's/^/  /'
  log ""
  log "Phase 1b total: 18 conditions (launch separately via 'launch phase1b shop' post-workshop)."
  log ""
  log "Run with 'launch' for Phase 1a default, or 'launch phase1b shop' for shop expansion."
}

launch_chain() {
  local label=$1
  local builder=$2
  local logfile="logs/queue_phase1_${label}.log"
  mkdir -p logs

  # Convert chain commands to space-quoted args
  local args=()
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    args+=("$line")
  done < <($builder)

  log "Launching $label chain (${#args[@]} cells) → $logfile"
  # FORCE_NEW=1: paper-grade fresh rerun — each cell gets a timestamped run_id,
  # never resumes a pre-fix archived dir (codex stress v6 C1, 2026-05-14).
  # RESET_BEFORE=1: each condition resets site state for fair ablation.
  FORCE_NEW=1 RESET_BEFORE=1 nohup bash scripts/queues/queue_chain.sh "${args[@]}" \
    > "$logfile" 2>&1 &
  local pid=$!
  log "  PID $pid, log $logfile"
  echo "$pid" > "logs/queue_phase1_${label}.pid"
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
        # Default = Phase 1a (cls + red only). Phase 1b shop requires explicit launch.
        launch_chain "cls" build_cls_chain
        launch_chain "red" build_red_chain
        ;;
      cls)  launch_chain "cls" build_cls_chain ;;
      red)  launch_chain "red" build_red_chain ;;
      shop)
        log "WARN: 'launch shop' requested directly. shop is Phase 1b (main-paper expansion)."
        log "      Default Phase 1a does NOT include shop. Proceeding only if you confirm."
        log "      Use 'launch phase1b' to launch shop explicitly as Phase 1b."
        fail "Use 'launch phase1b' for shop chain (Phase 1b main-paper expansion)."
        ;;
      phase1b)
        log "=== Phase 1b launch (main-paper shop expansion) ==="
        launch_chain "shop" build_shop_chain
        ;;
      *) fail "Unknown site filter: $SITE_FILTER (expected: all|cls|red|phase1b)" ;;
    esac
    log ""
    log "Phase 1a rerun launched (36 conditions, cls + red × B0+B1+B2 × 6 modes). Monitor:"
    log "  - PIDs: cat logs/queue_phase1_*.pid"
    log "  - Logs: tail -f logs/queue_phase1_*.log"
    log "  - Cells: open Obsidian Bases view 'cells.base' (cron 10min refresh)"
    log "  - Active: make active"
    log ""
    log "Post-completion analysis:"
    log "  make analysis              # full pipeline"
    log "  # Step 1: produce per_task_sr.csv (B-122 producer, 2026-05-15):"
    log "  python3 scripts/analysis/generate_per_task_sr.py \\"
    log "      --run-manifest results/phantom_paper/run_manifest.yaml \\"
    log "      --out results/phantom_paper/per_task_sr.csv"
    log "  # Step 2: decision test (B-120 6-cell K=6, 2026-05-15):"
    log "  python3 scripts/analysis/preregistration_decision_test.py \\"
    log "      --per-task-csv results/phantom_paper/per_task_sr.csv \\"
    log "      --primary-gate drop_one_pooled_meta_superiority \\"
    log "      --TOST-delta-pp 1.0 --H1-magnitude-pp 1.0 \\"
    log "      --transparency-K_h1 4 --transparency-K_h3 4 \\"
    log "      --out results/phantom_paper/preregistration_test_results.json"
    ;;
  *)
    fail "Unknown mode: $MODE (expected: dry-run | launch)"
    ;;
esac
