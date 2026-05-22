#!/usr/bin/env bash
# diag_autorun.sh — Tier-1 deterministic failure-pattern scan across completed
# conditions. The AUTO-TRIGGER layer for the diagnose skill's Tier-1.
#
# Tier-1 (diag_pattern_match.py) is 0-token + ~0.23s/run (224ep) + O(1-episode)
# memory, so it is safe to run on every completed condition automatically (cron
# or post-condition hook). Tier-2/3 (Claude sub-agent deep-dive + digest) stay
# MANUAL (`/diagnose <run>`) because they spend quota — see SKILL.md Tier-2
# budget cap.
#
# For each completed condition (has condition_summary_v2.json) lacking a fresh
# diag.json, runs diag_pattern_match and writes <run>/diag.json. Idempotent:
# skips runs whose diag.json already exists (DIAG_FORCE=1 to re-run). Skips smoke.
#
# Usage:
#   bash scripts/maintenance/diag_autorun.sh                          # default phase dir
#   bash scripts/maintenance/diag_autorun.sh results/visualwebarena/phase1
#   DIAG_FORCE=1 bash scripts/maintenance/diag_autorun.sh            # re-run all
#
# Cron (DGX / A100, every 30 min — pairs with the analysis sync cadence):
#   */30 * * * * cd <repo> && bash scripts/maintenance/diag_autorun.sh >> logs/cron/diag_autorun.log 2>&1
set -uo pipefail
cd "$(dirname "$0")/../.." 2>/dev/null || true
PY="${PYTHON_BIN:-.venv/bin/python3}"
PHASE_DIR="${1:-results/visualwebarena/phase1}"

[ -d "$PHASE_DIR" ] || { echo "[diag_autorun] phase dir not found: $PHASE_DIR"; exit 1; }

n=0; skipped=0; failed=0
for cond_dir in "$PHASE_DIR"/*/; do
  run="${cond_dir%/}"
  case "$run" in *smoke*) continue;; esac
  # completed = at least one condition_summary_v2.json under the run
  find "$run" -maxdepth 2 -name condition_summary_v2.json 2>/dev/null | grep -q . || continue
  out="$run/diag.json"
  if [ -s "$out" ] && [ -z "${DIAG_FORCE:-}" ]; then skipped=$((skipped+1)); continue; fi
  if "$PY" scripts/analysis/diag_pattern_match.py --run-dir "$run" --output "$out" >/dev/null 2>&1; then
    n=$((n+1)); echo "[diag] wrote $(basename "$run")/diag.json"
  else
    failed=$((failed+1)); echo "[diag] FAILED $(basename "$run") (schema incompat? check manually)"
  fi
done
echo "[diag_autorun] $n new, $skipped already-done, $failed failed. Tier-1 only (0 token); Tier-2/3 = manual /diagnose."
