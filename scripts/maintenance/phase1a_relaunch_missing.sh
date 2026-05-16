#!/usr/bin/env bash
# phase1a_relaunch_missing.sh — re-launch only PARTIAL + PENDING conditions
# after a Phase 1a fire interrupt.
#
# Uses phase1a_status.sh --json --missing-only to derive which conditions
# need re-launch. For each:
#   - PARTIAL → FORCE_NEW=0 (resume-by-glob, picks up existing run_dir
#     and continues from last unfinished episode via runtime.resume=true)
#   - PENDING → FORCE_NEW=1 (fresh timestamped run_id, paper-grade clean)
#
# Groups by site to enable 2-chain parallel (cls + red), mirroring
# queue_phase1_paper_grade.sh launch architecture.
#
# USAGE:
#   bash scripts/maintenance/phase1a_relaunch_missing.sh dry-run
#   bash scripts/maintenance/phase1a_relaunch_missing.sh launch
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

MODE=${1:-dry-run}
case "$MODE" in
  dry-run|launch) ;;
  *) echo "Usage: $0 [dry-run|launch]"; exit 2 ;;
esac

# Mode → queue script
queue_script_for_mode() {
  local m=$1
  case "$m" in
    dom|som|vision) echo "queue_baseline.sh" ;;
    phantom_text)   echo "queue_phantom_text.sh" ;;
    phantom_som)    echo "queue_phantom_som.sh" ;;
    phantom_prompt) echo "queue_phantom_prompt.sh" ;;
  esac
}

# Build queue invocation line per condition
queue_cmd_for_condition() {
  local bl=$1 mode=$2 site=$3
  local qs
  qs=$(queue_script_for_mode "$mode")
  if [[ "$mode" == "dom" || "$mode" == "som" || "$mode" == "vision" ]]; then
    echo "${qs} ${bl} ${mode} ${site}"
  else
    echo "${qs} ${bl} ${site}"
  fi
}

# Parse status JSON
STATUS_JSON=$(bash "${SCRIPT_DIR}/phase1a_status.sh" --json --missing-only)
MISSING_ROWS=$(echo "${STATUS_JSON}" | python3 -c "
import json, sys
data = json.load(sys.stdin)
for row in data:
    print(f\"{row['status']}|{row['baseline']}|{row['mode']}|{row['site']}\")
")

if [[ -z "${MISSING_ROWS}" ]]; then
  echo "✓ No missing conditions. Phase 1a complete."
  exit 0
fi

# Split into cls and red chains
declare -a CLS_CHAIN RED_CHAIN
declare -a CLS_RESUME RED_RESUME  # parallel array — 1 if PARTIAL (resume), 0 if PENDING (fresh)
while IFS='|' read -r status bl mode site; do
  [[ -z "$status" ]] && continue
  cmd=$(queue_cmd_for_condition "$bl" "$mode" "$site")
  is_resume=0
  [[ "$status" == "PARTIAL" ]] && is_resume=1
  case "$site" in
    classifieds) CLS_CHAIN+=("$cmd"); CLS_RESUME+=("$is_resume") ;;
    reddit)      RED_CHAIN+=("$cmd"); RED_RESUME+=("$is_resume") ;;
  esac
done <<< "${MISSING_ROWS}"

echo "=== Phase 1a relaunch plan ==="
echo
echo "cls chain (${#CLS_CHAIN[@]} conditions):"
for i in "${!CLS_CHAIN[@]}"; do
  tag="FRESH"; [[ "${CLS_RESUME[$i]}" == "1" ]] && tag="RESUME"
  printf "  [%s] %s\n" "$tag" "${CLS_CHAIN[$i]}"
done
echo
echo "red chain (${#RED_CHAIN[@]} conditions):"
for i in "${!RED_CHAIN[@]}"; do
  tag="FRESH"; [[ "${RED_RESUME[$i]}" == "1" ]] && tag="RESUME"
  printf "  [%s] %s\n" "$tag" "${RED_CHAIN[$i]}"
done

if [[ "${MODE}" == "dry-run" ]]; then
  echo
  echo "Run with 'launch' to actually fire."
  exit 0
fi

# Launch each chain in background
launch_chain() {
  local label=$1
  shift
  local resume_arr_name=$1
  shift
  local cmds=("$@")
  [[ "${#cmds[@]}" -eq 0 ]] && { echo "(${label} chain empty)"; return 0; }

  local logfile="logs/queue_phase1a_relaunch_${label}.log"
  mkdir -p logs

  # Build queue_chain invocation. We must respect per-condition FORCE_NEW.
  # queue_chain takes one FORCE_NEW for whole chain. Compromise: if any
  # condition is RESUME, set FORCE_NEW=0 for whole chain; queue_chain
  # idempotent skip + per-condition resume-by-glob handle the rest.
  # (Fresh conditions with no prior run_dir get a new timestamp anyway.)
  local force_new=1
  local resume_arr
  declare -n resume_arr=$resume_arr_name
  for v in "${resume_arr[@]}"; do
    [[ "$v" == "1" ]] && { force_new=0; break; }
  done

  echo "Launching ${label} chain (${#cmds[@]} cells, FORCE_NEW=${force_new}) → ${logfile}"
  FORCE_NEW=${force_new} RESET_BEFORE=1 nohup bash scripts/queues/queue_chain.sh "${cmds[@]}" \
    > "${logfile}" 2>&1 &
  local pid=$!
  echo "  PID ${pid}"
  echo "${pid}" > "logs/queue_phase1a_relaunch_${label}.pid"
}

launch_chain "cls" CLS_RESUME "${CLS_CHAIN[@]}"
launch_chain "red" RED_RESUME "${RED_CHAIN[@]}"

echo
echo "Phase 1a relaunch fired (${#CLS_CHAIN[@]} cls + ${#RED_CHAIN[@]} red = $((${#CLS_CHAIN[@]} + ${#RED_CHAIN[@]})) conditions). Monitor:"
echo "  - PIDs: cat logs/queue_phase1a_relaunch_*.pid"
echo "  - Logs: tail -f logs/queue_phase1a_relaunch_*.log"
echo "  - Status: bash scripts/maintenance/phase1a_status.sh"
