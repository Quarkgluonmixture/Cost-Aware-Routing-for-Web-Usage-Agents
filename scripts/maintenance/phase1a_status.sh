#!/usr/bin/env bash
# phase1a_status.sh — list Phase 1a 36 conditions, report complete / partial / pending.
#
# WHY: After fire interrupt, need to know which conditions are done before
# selective re-launch. FORCE_NEW=1 paper-grade fresh = each re-launch
# creates new run_id, would re-fire completed conditions wastefully.
# This script + phase1a_relaunch_missing.sh enable resume.
#
# WHAT: For each (baseline, mode, site) in 36-condition manifest:
#   - find latest run_id matching B{baseline}_{mode}_{site}_*
#   - check condition_summary_v2.json exists AND episodes >= scored_task_count
#   - emit table: STATUS | BASELINE | MODE | SITE | RUN_ID | EPISODES | EXPECTED
#
# Scored task count (post §139.8 N/A exclusion):
#   classifieds: 224 (234 nominal - 10 N/A)
#   reddit:      208 (210 nominal - 2 N/A; codex count was 15 total across cls+red)
#
# USAGE:
#   bash scripts/maintenance/phase1a_status.sh                       # text table
#   bash scripts/maintenance/phase1a_status.sh --json                # machine-readable
#   bash scripts/maintenance/phase1a_status.sh --missing-only        # only NOT-complete rows
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

JSON=0
MISSING_ONLY=0
# CRITICAL: only count runs at-or-after this date as paper-grade. Pre-2026-05-16
# runs are pre-cross-system-audit-fixes (BUG-1..16) and pre-N/A-exclusion-422
# scored count. Default = 2026-05-16 fire date (post all 16-bug fix commits).
# Override: --min-date YYYYMMDD or env MIN_RUN_DATE=20260516.
MIN_RUN_DATE="${MIN_RUN_DATE:-20260516}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --json) JSON=1; shift ;;
    --missing-only) MISSING_ONLY=1; shift ;;
    --min-date) MIN_RUN_DATE="$2"; shift 2 ;;
    --min-date=*) MIN_RUN_DATE="${1#--min-date=}"; shift ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

# Phase 1a manifest = 3 baselines × 6 modes × 2 sites
BASELINES=(B0 B1 B2)
MODES=(dom som vision phantom_text phantom_som phantom_prompt)
SITES=(classifieds reddit)

declare -A SCORED_COUNT
SCORED_COUNT[classifieds]=224
SCORED_COUNT[reddit]=208

RESULTS_ROOT="results/visualwebarena/phase1"

# Per-mode → condition_id mapping (P79 internal labels).
# Post B-261 fix (2026-05-16, A1.7): phantom_text is canonical; phantom_dom
# legacy alias retired from conditions.py + yaml. Archive run dirs named
# phase1_phantom_dom_router_0/ stay historical; new fires write phantom_text.
# Lookup: dom/som/vision → phase1_{mode}_router_0
#         phantom_text → phase1_phantom_text_router_0  (B-261: canonical)
#         phantom_som  → phase1_phantom_som_router_0
#         phantom_prompt → phase1_phantom_prompt_router_0
condition_id_for_mode() {
  local m=$1
  echo "phase1_${m}_router_0"
}

count_episodes() {
  local summary=$1
  python3 -c "
import json,sys
try:
    d = json.load(open('${summary}'))
    print(int(d.get('episodes', 0)))
except: print(0)
" 2>/dev/null
}

# Collect rows
ROWS=()
for site in "${SITES[@]}"; do
  for bl in "${BASELINES[@]}"; do
    for mode in "${MODES[@]}"; do
      condition=$(condition_id_for_mode "$mode")
      expected=${SCORED_COUNT[$site]}

      # Find latest paper-grade run dir matching pattern (post MIN_RUN_DATE).
      # Run_id format: B{X}_{mode}_{site}_{YYYYMMDD}[_{HHMMSS}], so extract the
      # YYYYMMDD segment and filter. Pre-MIN_RUN_DATE runs = pre-cross-system-
      # audit-fix data = NOT paper-grade for the post-fix fire.
      pattern="${RESULTS_ROOT}/${bl}_${mode}_${site}_*"
      latest_run=""
      for candidate in $(ls -dt ${pattern} 2>/dev/null); do
        # Extract YYYYMMDD from candidate basename
        cand_date=$(basename "${candidate}" | grep -oE '[0-9]{8}' | head -1)
        if [[ -n "${cand_date}" && "${cand_date}" -ge "${MIN_RUN_DATE}" ]]; then
          latest_run="${candidate}"
          break
        fi
      done

      if [[ -z "${latest_run}" ]]; then
        ROWS+=("PENDING|${bl}|${mode}|${site}|—|0|${expected}")
        continue
      fi

      run_id=$(basename "${latest_run}")
      summary="${latest_run}/${condition}/condition_summary_v2.json"

      if [[ ! -f "${summary}" ]]; then
        ROWS+=("PARTIAL|${bl}|${mode}|${site}|${run_id}|0|${expected}")
        continue
      fi

      episodes=$(count_episodes "${summary}")
      if [[ "${episodes}" -ge "${expected}" ]]; then
        ROWS+=("COMPLETE|${bl}|${mode}|${site}|${run_id}|${episodes}|${expected}")
      else
        ROWS+=("PARTIAL|${bl}|${mode}|${site}|${run_id}|${episodes}|${expected}")
      fi
    done
  done
done

# Emit
if [[ "${JSON}" == "1" ]]; then
  echo "["
  first=1
  for row in "${ROWS[@]}"; do
    IFS='|' read -r status bl mode site run_id eps exp <<< "${row}"
    [[ "${MISSING_ONLY}" == "1" && "${status}" == "COMPLETE" ]] && continue
    [[ "${first}" == "1" ]] || echo "  ,"
    first=0
    cat <<EOF
  {"status":"${status}","baseline":"${bl}","mode":"${mode}","site":"${site}","run_id":"${run_id}","episodes":${eps},"expected":${exp}}
EOF
  done
  echo "]"
  exit 0
fi

# Text table
printf "%-9s %-3s %-15s %-12s %-50s %-8s %-8s\n" "STATUS" "BL" "MODE" "SITE" "RUN_ID" "EPISODES" "EXPECTED"
printf "%-9s %-3s %-15s %-12s %-50s %-8s %-8s\n" "------" "--" "----" "----" "------" "--------" "--------"
n_complete=0; n_partial=0; n_pending=0
for row in "${ROWS[@]}"; do
  IFS='|' read -r status bl mode site run_id eps exp <<< "${row}"
  case "${status}" in
    COMPLETE) n_complete=$((n_complete+1)); [[ "${MISSING_ONLY}" == "1" ]] && continue ;;
    PARTIAL)  n_partial=$((n_partial+1)) ;;
    PENDING)  n_pending=$((n_pending+1)) ;;
  esac
  printf "%-9s %-3s %-15s %-12s %-50s %-8s %-8s\n" "${status}" "${bl}" "${mode}" "${site}" "${run_id}" "${eps}" "${exp}"
done
echo
total=${#ROWS[@]}
echo "Summary: ${n_complete} complete / ${n_partial} partial / ${n_pending} pending / ${total} total"
if [[ "${n_partial}" -gt 0 || "${n_pending}" -gt 0 ]]; then
  echo "Re-launch missing via: bash scripts/maintenance/phase1a_relaunch_missing.sh"
fi
