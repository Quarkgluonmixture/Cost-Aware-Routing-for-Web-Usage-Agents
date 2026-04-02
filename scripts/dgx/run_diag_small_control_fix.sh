#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID_PREFIX="${DIAG_RUN_ID_PREFIX:-diag_small_control_fix}"
LOG_PATH_DEFAULT="${REPO_DIR}/logs/${RUN_ID_PREFIX}_${STAMP}.log"

export BASELINE_CONFIG="${DIAG_CONFIG:-${REPO_DIR}/configs/exp_v2_qwen3vl4b_diagnostic_not_for_main_baseline.yaml}"
export BASELINE_RUN_ID_PREFIX="${RUN_ID_PREFIX}"
export BASELINE_LOG_PATH="${DIAG_LOG_PATH:-${LOG_PATH_DEFAULT}}"

echo "[diag] config=${BASELINE_CONFIG}"
echo "[diag] run_id_prefix=${BASELINE_RUN_ID_PREFIX}"
echo "[diag] log_path=${BASELINE_LOG_PATH}"

exec bash "${REPO_DIR}/scripts/dgx/run_qwen3vl4b_baseline.sh"
