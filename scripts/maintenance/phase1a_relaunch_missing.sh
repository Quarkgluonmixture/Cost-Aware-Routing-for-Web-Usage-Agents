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

# B-1823 (Fire-6 /stress P0-1-B*, 2026-05-21): DEPRECATED for paper-grade runs.
# This tool calls queue_chain.sh directly (bypassing the orchestrator lock/gates
# at queue_phase1_paper_grade.sh:763-780 + Gate 8 quarantine :424-451) and then
# launches up to FOUR background chains back-to-back (cls_fresh/red_fresh/
# cls_resume/red_resume) — violating the single-site-chain hard rule (shared
# docker user account → session race + cross-condition contamination), and it can
# re-run completed conditions (it also depended on phase1a_status.sh's pre-fix
# reddit=208 count → a valid 205-ep reddit run was mis-flagged PARTIAL → rerun).
# Paper-grade resume now lives in the orchestrator (same preflight/Gate8/quarantine,
# sequential cls→red, manifest-bound done-skip, no parallel chains):
#     RESUME_MISSING=1 bash scripts/queues/queue_phase1_paper_grade.sh launch
if [[ "${P79_PAPER_GRADE:-1}" == "1" ]]; then
  echo "[phase1a_relaunch_missing][FATAL] DEPRECATED for paper-grade (B-1823)." >&2
  echo "[phase1a_relaunch_missing][FATAL] reason: bypasses orchestrator gates + launches 4 parallel chains (single-site-rule violation) + can rerun completed data." >&2
  echo "[phase1a_relaunch_missing][FATAL] use instead: RESUME_MISSING=1 bash scripts/queues/queue_phase1_paper_grade.sh launch" >&2
  echo "[phase1a_relaunch_missing][FATAL] (dev/non-paper-grade override only: P79_PAPER_GRADE=0)" >&2
  exit 1
fi

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

# Launch each chain — split by resume-vs-fresh to fix B-303 (A1.17 P0-5 A+C)
# and B-304 (A1.17 P1-5-B codex OOB).
#
# B-303 (FORCE_NEW leakage): pre-fix bundled all conditions in a single
# queue_chain with chain-level FORCE_NEW = 0 if ANY condition was PARTIAL.
# This let PENDING (fresh) conditions go through resume-by-glob and possibly
# inherit stale partial dirs. Now split: PENDING conditions → fresh chain with
# FORCE_NEW=1; PARTIAL conditions → resume chain with FORCE_NEW=0.
#
# B-304 (resume+reset trajectory discontinuity P1-5-B α'): resume chains
# additionally run with RESET_BEFORE=0 — preserves trajectory continuity per
# 3-AI brainstorm Tier 1 stack decision. Fresh chains keep RESET_BEFORE=1.
# Paper §3 disclosure: "PARTIAL cells resumed without additional reset;
# trajectory continuity preserved; fresh cells reset before launch."
launch_chain_homogeneous() {
  local label=$1
  local force_new=$2     # 1 = fresh, 0 = resume
  local reset_before=$3  # 1 = reset, 0 = no-reset (resume case)
  shift 3
  local cmds=("$@")
  [[ "${#cmds[@]}" -eq 0 ]] && { echo "(${label} chain empty)"; return 0; }

  local logfile="logs/queue_phase1a_relaunch_${label}.log"
  mkdir -p logs

  echo "Launching ${label} chain (${#cmds[@]} cells, FORCE_NEW=${force_new}, RESET_BEFORE=${reset_before}) → ${logfile}"
  FORCE_NEW=${force_new} RESET_BEFORE=${reset_before} nohup bash scripts/queues/queue_chain.sh "${cmds[@]}" \
    > "${logfile}" 2>&1 &
  local pid=$!
  echo "  PID ${pid}"
  echo "${pid}" > "logs/queue_phase1a_relaunch_${label}.pid"
}

# Split each site chain into PENDING (fresh) + PARTIAL (resume) subgroups
split_by_resume() {
  local cmds_var=$1
  local resume_var=$2
  local fresh_out=$3
  local resume_out=$4
  declare -n cmds=$cmds_var
  declare -n resume_arr=$resume_var
  declare -n fresh_list=$fresh_out
  declare -n resume_list=$resume_out
  fresh_list=()
  resume_list=()
  for i in "${!cmds[@]}"; do
    if [[ "${resume_arr[$i]}" == "1" ]]; then
      resume_list+=("${cmds[$i]}")
    else
      fresh_list+=("${cmds[$i]}")
    fi
  done
}

declare -a CLS_FRESH CLS_RESUMES RED_FRESH RED_RESUMES
split_by_resume CLS_CHAIN CLS_RESUME CLS_FRESH CLS_RESUMES
split_by_resume RED_CHAIN RED_RESUME RED_FRESH RED_RESUMES

# Fresh chains: FORCE_NEW=1 + RESET_BEFORE=1 (paper-grade clean launch)
launch_chain_homogeneous "cls_fresh" 1 1 "${CLS_FRESH[@]}"
launch_chain_homogeneous "red_fresh" 1 1 "${RED_FRESH[@]}"
# Resume chains: FORCE_NEW=0 + RESET_BEFORE=0 (trajectory continuity preserved)
launch_chain_homogeneous "cls_resume" 0 0 "${CLS_RESUMES[@]}"
launch_chain_homogeneous "red_resume" 0 0 "${RED_RESUMES[@]}"

echo
echo "Phase 1a relaunch fired (${#CLS_CHAIN[@]} cls + ${#RED_CHAIN[@]} red = $((${#CLS_CHAIN[@]} + ${#RED_CHAIN[@]})) conditions, split into up to 4 sub-chains). Monitor:"
echo "  - PIDs: cat logs/queue_phase1a_relaunch_*.pid"
echo "  - Logs: tail -f logs/queue_phase1a_relaunch_*.log"
echo "  - Status: bash scripts/maintenance/phase1a_status.sh"
echo "  - Split logic (A1.17 B-303+B-304): cls_fresh/red_fresh = FORCE_NEW=1 RESET=1; cls_resume/red_resume = FORCE_NEW=0 RESET=0 (trajectory continuity)"
