#!/usr/bin/env bash
# queue_16cell_paper_grade.sh — Master orchestrator for the 16-cell post-advisor-sync
# rerun scope. Spec: B0×{cls,red}×3 phantom + B1×{cls,red}×3 phantom + B0 shop×2 + B1 shop×2.
#
# **Hard rule: Same site, B0 XOR B1 only**. queue_chain wraps reset+watchdog+idempotent.
# Splits into 3 parallel chains (cls / red / shop), each chain is internally sequential.
#
# Pre-launch gates (must all pass):
#   - Advisor email reply received → K_h1 / K_h3 / TOST_delta locked in preregistration.md
#   - A100 SSH connectivity verified ('ssh condense-a100 nvidia-smi' returns OK)
#   - VWA stack running on chosen host (DGX→quark Tailscale OR A100 self-host)
#   - env_snapshot baseline committed (results/provenance/env_<host>_baseline.json)
#   - VWA snapshot baseline committed (results/provenance/vwa_<host>_baseline.json)
#
# Usage:
#   bash scripts/queues/queue_16cell_paper_grade.sh dry-run             # preview, no launch
#   bash scripts/queues/queue_16cell_paper_grade.sh launch              # actual launch (3 parallel chains)
#   bash scripts/queues/queue_16cell_paper_grade.sh launch cls          # only classifieds chain
#   bash scripts/queues/queue_16cell_paper_grade.sh launch red          # only reddit chain
#   bash scripts/queues/queue_16cell_paper_grade.sh launch shop         # only shopping chain
#
# Cells (16 total — confirmed post-5/5 sync, see preregistration.md):
#   - B0 cls × {P-text, P-SoM, P-prompt}  (3)
#   - B0 red × {P-text, P-SoM, P-prompt}  (3)
#   - B1 cls × {P-text, P-SoM, P-prompt}  (3)
#   - B1 red × {P-text, P-SoM, P-prompt}  (3)
#   - B0 shop × {P-text, P-SoM}  (2)  — shopping skip P-prompt (advisor-confirmed scope cut)
#   - B1 shop × {P-text, P-SoM}  (2)
#
# Chain dependency:
#   cls and red can run in parallel (different sites = no resource contention beyond A100 GPU).
#   shop runs after cls + red because B0/B1 shop has historical bug surface (Magento FPC).
#   Within each chain B0 → B1 sequential (same-site B0/B1 share user account login).
#
# ETA estimates (A100 40GB, post-advisor lock):
#   cls chain: B0 (~12h) → B1 (~24h) = 36h
#   red chain: B0 (~10h) → B1 (~20h) = 30h
#   shop chain: B0 (~16h) → B1 (~32h) = 48h
#   Total wallclock with 3 parallel chains = max(36, 30, 48) = ~48h ≈ 2 days
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
  cat <<EOF
queue_phantom_text.sh B0 classifieds
queue_phantom_som.sh B0 classifieds
queue_phantom_prompt.sh B0 classifieds
queue_phantom_text.sh B1 classifieds
queue_phantom_som.sh B1 classifieds
queue_phantom_prompt.sh B1 classifieds
EOF
}

build_red_chain() {
  cat <<EOF
queue_phantom_text.sh B0 reddit
queue_phantom_som.sh B0 reddit
queue_phantom_prompt.sh B0 reddit
queue_phantom_text.sh B1 reddit
queue_phantom_som.sh B1 reddit
queue_phantom_prompt.sh B1 reddit
EOF
}

build_shop_chain() {
  cat <<EOF
queue_phantom_text.sh B0 shopping
queue_phantom_som.sh B0 shopping
queue_phantom_text.sh B1 shopping
queue_phantom_som.sh B1 shopping
EOF
}

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

dry_run() {
  log "DRY RUN — no launches will occur."
  log ""
  log "Cls chain (6 cells):"
  build_cls_chain | sed 's/^/  /'
  log ""
  log "Red chain (6 cells):"
  build_red_chain | sed 's/^/  /'
  log ""
  log "Shop chain (4 cells):"
  build_shop_chain | sed 's/^/  /'
  log ""
  log "Total: 16 cells across 3 chains."
  log ""
  log "Run with 'launch [site]' to actually launch."
}

launch_chain() {
  local label=$1
  local builder=$2
  local logfile="logs/queue_16cell_${label}.log"
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
  echo "$pid" > "logs/queue_16cell_${label}.pid"
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
        launch_chain "cls" build_cls_chain
        launch_chain "red" build_red_chain
        launch_chain "shop" build_shop_chain
        ;;
      cls)  launch_chain "cls" build_cls_chain ;;
      red)  launch_chain "red" build_red_chain ;;
      shop) launch_chain "shop" build_shop_chain ;;
      *) fail "Unknown site filter: $SITE_FILTER (expected: all|cls|red|shop)" ;;
    esac
    log ""
    log "16-cell rerun launched. Monitor:"
    log "  - PIDs: cat logs/queue_16cell_*.pid"
    log "  - Logs: tail -f logs/queue_16cell_*.log"
    log "  - Cells: open Obsidian Bases view 'cells.base' (cron 10min refresh)"
    log "  - Active: make active"
    log ""
    log "Post-completion analysis:"
    log "  make analysis              # full pipeline"
    log "  python3 scripts/analysis/preregistration_decision_test.py \\"
    log "      --cells-csv results/phantom_paper/cells_aggregated.csv \\"
    log "      --K_h1 \$(cat docs/checkpoints/pre_run/preregistration.md | grep K_h1 | head -1) \\"
    log "      --out results/phantom_paper/preregistration_test_results.json"
    ;;
  *)
    fail "Unknown mode: $MODE (expected: dry-run | launch)"
    ;;
esac
