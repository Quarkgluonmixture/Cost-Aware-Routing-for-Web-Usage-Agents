#!/usr/bin/env bash
# queue_phantom_pair.sh — Sequential phantom_som + phantom_dom on 一个 site，每 condition 间自动 reset.
#
# Mirrors queue_b0_with_reset.sh pattern (reset → run → wait → reset → run).
# Designed to produce paper-grade ablation pair (phantom_som vs phantom_dom)
# from identical post-reset starting state.
#
# 用法:
#   bash scripts/queues/queue_phantom_pair.sh <baseline> <site> [order] [benchmark]
#     baseline: B0 | B1
#     site: classifieds | reddit | shopping | shopping_admin (后者仅 wa)
#     order: som,dom (default) | dom,som | som | dom
#     benchmark: vwa (default) | wa
#
# 例:
#   nohup bash scripts/queues/queue_phantom_pair.sh B0 reddit dom,som > logs/chain_B0_reddit.log 2>&1 &
#   nohup bash scripts/queues/queue_phantom_pair.sh B1 classifieds som,dom > logs/chain_B1_cls.log 2>&1 &
#
# B0/B1 分配不同 site 即可 parallel；同 site 必须 sequential。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_DIR}"

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <baseline:B0|B1> <site> [order=som,dom] [benchmark=vwa|wa]" >&2
  exit 2
fi

BASELINE="$1"; SITE="$2"
ORDER="${3:-som,dom}"
BENCHMARK="${4:-vwa}"

# Validate
[[ "${BASELINE}" =~ ^B[01]$ ]] || { echo "Invalid baseline: ${BASELINE}" >&2; exit 2; }
[[ "${BENCHMARK}" =~ ^(vwa|wa)$ ]] || { echo "Invalid benchmark: ${BENCHMARK}" >&2; exit 2; }

# Source reset utility
source "${REPO_DIR}/scripts/maintenance/reset_vwa_sites.sh"

# Phase dir based on benchmark
if [[ "${BENCHMARK}" == "wa" ]]; then
  PHASE_DIR="${REPO_DIR}/results/webarena/phase1"
else
  PHASE_DIR="${REPO_DIR}/results/visualwebarena/phase1"
fi

log() {
  echo "[chain $(date '+%H:%M:%S')] $*"
}

# Build CFG_NAME per (baseline, mode, site, benchmark)
build_cfg_name() {
  local baseline=$1 mode=$2 site=$3 benchmark=$4
  local name="${baseline}_phantom"
  [[ "$mode" == "dom" ]] && name="${baseline}_phantom_dom"
  [[ "$benchmark" == "wa" ]] && name="${name}_wa"
  echo "${name}_${site}"
}

# Find run_dir for a given config name
find_run_dir() {
  local cfg=$1
  ls -dt "${PHASE_DIR}/${cfg}_"* 2>/dev/null | head -1 || true
}

# Check if condition is complete (condition_summary_v2.json exists)
condition_complete() {
  local run_dir=$1 mode=$2
  [[ -f "${run_dir}/phase1_phantom_${mode}_router_0/condition_summary_v2.json" ]]
}

# Wait until a condition is complete
wait_for_completion() {
  local run_dir=$1 mode=$2
  local elapsed=0
  log "waiting for ${run_dir}/phase1_phantom_${mode}_router_0 to complete..."
  while ! condition_complete "$run_dir" "$mode"; do
    sleep 60
    elapsed=$((elapsed + 60))
    if (( elapsed % 600 == 0 )); then
      # 每 10 min progress report
      local n=$(ls "${run_dir}/phase1_phantom_${mode}_router_0/episodes/"*_summary_v2.json 2>/dev/null | wc -l)
      log "  still waiting... (${elapsed}s elapsed, episodes=${n})"
    fi
  done
  log "  ${mode} complete!"
}

# Main loop
log "=================================================="
log "${BASELINE} phantom pair on ${BENCHMARK}/${SITE}, order=${ORDER}"
log "=================================================="

IFS=',' read -ra MODES <<< "${ORDER}"
for mode in "${MODES[@]}"; do
  log ""
  log "====== STEP: ${BASELINE}_phantom_${mode} on ${BENCHMARK}/${SITE} ======"

  cfg_name=$(build_cfg_name "$BASELINE" "$mode" "$SITE" "$BENCHMARK")
  run_dir=$(find_run_dir "$cfg_name")

  if [[ -n "$run_dir" ]] && condition_complete "$run_dir" "$mode"; then
    log "  [skip] ${mode} already complete (run_dir=${run_dir})"
    continue
  fi

  # If a runner is already running for this run_dir, just wait
  if [[ -n "$run_dir" ]] && pgrep -f "run_experiment.py.*$(basename ${run_dir})" > /dev/null; then
    log "  [resume-wait] ${mode} runner already running, waiting for completion..."
    wait_for_completion "$run_dir" "$mode"
    continue
  fi

  # Reset site (only for VWA; WA reset uses different mechanism if any)
  if [[ "$BENCHMARK" == "vwa" ]]; then
    log "  resetting ${SITE} before ${mode}..."
    reset_vwa_sites "$SITE" "chain_${BASELINE}_phantom_${mode}_${SITE}" || log "  [warn] reset failed (continuing)"
    sleep 15
  else
    log "  [info] WA benchmark — no site reset"
  fi

  # Launch via queue_phantom.sh (it handles env setup + watchdog)
  log "  launching ${BASELINE}_phantom_${mode} ${SITE}..."
  bash "${REPO_DIR}/scripts/queues/queue_phantom.sh" \
    "$BASELINE" "$mode" "$SITE" "$BENCHMARK" 2>&1 | sed 's/^/  /'

  # Wait for it to complete
  run_dir=$(find_run_dir "$cfg_name")
  if [[ -z "$run_dir" ]]; then
    log "  [error] could not find run_dir for ${cfg_name}, exiting" >&2
    exit 1
  fi

  wait_for_completion "$run_dir" "$mode"
done

log ""
log "=================================================="
log "${BASELINE} phantom pair on ${BENCHMARK}/${SITE} — ALL DONE"
log "=================================================="
