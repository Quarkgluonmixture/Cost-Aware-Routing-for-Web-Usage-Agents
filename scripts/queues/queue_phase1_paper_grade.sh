#!/usr/bin/env bash
# queue_phase1_paper_grade.sh — Master orchestrator for Phase 1 paper-grade rerun.
# (Renamed 2026-05-13 from queue_16cell_paper_grade.sh; old name reflected prior
# 16-cell phantom-only scope that codex stress audit identified as incomplete.)
#
# Scope (revised 2026-05-13 post codex stress audit):
#   Phase 1a (THIS SCRIPT default): 24 operational conditions = 2 sites (cls, red)
#     × 2 models (B0, B1) × 6 modes (DOM, SoM, Vision, P-text, P-prompt, P-SoM).
#     Statistical analysis: 4 (site, model) cells, pooled DerSimonian-Laird meta + TOST.
#     Target: workshop submission. Replaces prior 16-cell phantom-only scope which
#     lacked DOM/SoM/Vision baseline rerun (codex Flaw 1).
#   Phase 1b (deferred, requires explicit 'launch phase1b shop'): 12 additional
#     conditions = shop × 2 models × 6 modes. Feeds main paper R3→R1 framing
#     decision post-workshop submission.
#
# **Hard rule: Same site, B0 XOR B1 only**. queue_chain wraps reset+watchdog+idempotent.
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
#   bash scripts/queues/queue_phase1_paper_grade.sh launch             # Phase 1a (cls+red, 24 conditions)
#   bash scripts/queues/queue_phase1_paper_grade.sh launch cls         # only classifieds Phase 1a chain (12 conditions)
#   bash scripts/queues/queue_phase1_paper_grade.sh launch red         # only reddit Phase 1a chain (12 conditions)
#   bash scripts/queues/queue_phase1_paper_grade.sh launch phase1b     # Phase 1b shop chain (12 conditions, deferred to post-workshop)
#
# Phase 1a conditions (24 total):
#   cls chain (12 conditions, B0 → B1 sequential):
#     - B0 cls × {DOM, SoM, Vision, P-text, P-SoM, P-prompt}  (6)
#     - B1 cls × {DOM, SoM, Vision, P-text, P-SoM, P-prompt}  (6)
#   red chain (12 conditions, B0 → B1 sequential):
#     - B0 red × {DOM, SoM, Vision, P-text, P-SoM, P-prompt}  (6)
#     - B1 red × {DOM, SoM, Vision, P-text, P-SoM, P-prompt}  (6)
#
# Phase 1b conditions (12 total, deferred main-paper expansion):
#     - B0 shop × {DOM, SoM, Vision, P-text, P-SoM, P-prompt}  (6)
#     - B1 shop × {DOM, SoM, Vision, P-text, P-SoM, P-prompt}  (6)
#
# Chain dependency:
#   cls and red can run in parallel (different sites = no resource contention beyond A100 GPU).
#   Within each chain B0 → B1 sequential (same-site B0/B1 share user account login).
#   Phase 1b shop launched separately after workshop submission to avoid Magento FPC bug
#   surface co-occurring with Phase 1a critical path.
#
# ETA estimates (A100 40GB, post-advisor lock):
#   cls chain (12 conditions): B0 (~24h) → B1 (~48h) = 72h ≈ 3 days
#   red chain (12 conditions): B0 (~20h) → B1 (~40h) = 60h ≈ 2.5 days
#   Total Phase 1a wallclock with 2 parallel chains = max(72, 60) ≈ 3 days
#   Phase 1b shop chain (12 conditions): B0 (~32h) → B1 (~64h) = 96h ≈ 4 days (deferred)
#
# Sentinel files (used by chain to detect completion):
#   results/visualwebarena/phase1/<run_id>/<condition_id>/condition_summary_v2.json

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_DIR}"

MODE="${1:-dry-run}"
SITE_FILTER="${2:-all}"

log() { echo "[16cell $(date '+%H:%M:%S')] $*"; }
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

  log "=== Gate 4: VWA reachability ==="
  if [ -f scripts/preflight_v2.sh ]; then
    bash scripts/preflight_v2.sh --no-strict-ports 2>&1 | tail -5 | sed 's/^/    /'
  else
    log "  WARN: scripts/preflight_v2.sh not found"
  fi

  log "=== Gate 5: GPU + model load smoke ==="
  if command -v .venv/bin/python3 &>/dev/null; then
    .venv/bin/python3 -c "import torch; print(f'  CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" 2>&1 | sed 's/^/  /'
  fi

  log "=== Gate 6: No conflicting active runs ==="
  active=$(pgrep -f "run_experiment.*--config" | wc -l)
  log "  Active run_experiment processes: $active"
  if [ "$active" -gt 0 ]; then
    log "  WARN: Existing runs detected. Verify no same-site B0+B1 conflict before launch."
    pgrep -af "run_experiment.*--config" | sed 's/^/    /'
  fi

  if [ "$errors" -gt 0 ]; then
    fail "$errors gate(s) failed; abort. Fix and re-run."
  fi
  log "All gates passed (or warnings only)."
}

# ---------------------------------------------------------------------------
# Chain definitions
# ---------------------------------------------------------------------------

build_cls_chain() {
  # Phase 1a classifieds: 6 modes per model, B0 → B1 sequential = 12 conditions
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
EOF
}

build_red_chain() {
  # Phase 1a reddit: 6 modes per model, B0 → B1 sequential = 12 conditions
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
EOF
}

build_shop_chain() {
  # Phase 1b deferred: shop × 6 modes per model, B0 → B1 sequential = 12 conditions
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
  log "Cls chain (12 conditions, 6 modes × B0+B1):"
  build_cls_chain | sed 's/^/  /'
  log ""
  log "Red chain (12 conditions, 6 modes × B0+B1):"
  build_red_chain | sed 's/^/  /'
  log ""
  log "Phase 1a total: 24 operational conditions across 4 statistical cells (= (site, model) tuples)."
  log ""
  log "=== Phase 1b (deferred, main paper expansion) ==="
  log ""
  log "Shop chain (12 conditions, 6 modes × B0+B1):"
  build_shop_chain | sed 's/^/  /'
  log ""
  log "Phase 1b total: 12 conditions (launch separately via 'launch phase1b shop' post-workshop)."
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
  RESET_BEFORE=1 nohup bash scripts/queues/queue_chain.sh "${args[@]}" \
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
    log "Phase 1a rerun launched (24 conditions, cls + red × B0+B1 × 6 modes). Monitor:"
    log "  - PIDs: cat logs/queue_phase1_*.pid"
    log "  - Logs: tail -f logs/queue_phase1_*.log"
    log "  - Cells: open Obsidian Bases view 'cells.base' (cron 10min refresh)"
    log "  - Active: make active"
    log ""
    log "Post-completion analysis:"
    log "  make analysis              # full pipeline"
    log "  python3 scripts/analysis/preregistration_decision_test.py \\"
    log "      --per-task-csv results/phantom_paper/per_task_sr.csv \\"
    log "      --primary-gate drop_one_pooled_meta_superiority \\"
    log "      --TOST-delta-pp 1.0 --H1-magnitude-pp 1.0 \\"
    log "      --transparency-K_h1 3 --transparency-K_h3 3 \\"
    log "      --out results/phantom_paper/preregistration_test_results.json"
    ;;
  *)
    fail "Unknown mode: $MODE (expected: dry-run | launch)"
    ;;
esac
