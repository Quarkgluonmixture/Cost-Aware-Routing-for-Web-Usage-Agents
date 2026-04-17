#!/usr/bin/env bash
# run_scroll_comparison.sh — Scroll error cross-validation
#
# Two experiments:
#   claude    — Claude Sonnet 4.6 on Bedrock proxy (free, 20 tasks)
#   dashscope — Qwen3-VL-235B-instruct on DashScope official (free quota, 10 tasks)
#
# Usage:
#   bash scripts/dgx/run_scroll_comparison.sh claude
#   bash scripts/dgx/run_scroll_comparison.sh dashscope
#   bash scripts/dgx/run_scroll_comparison.sh all
set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_DIR}"

MODE="${1:-all}"
DATE="$(date +%Y%m%d)"
LOG_DIR="${REPO_DIR}/logs"
RESULTS_BASE="${REPO_DIR}/results/visualwebarena/phase1"
mkdir -p "${LOG_DIR}"

# ---------- Python ----------
if [[ -x "${REPO_DIR}/.venv/bin/python" ]]; then
  PYTHON_BIN="${REPO_DIR}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
else
  echo "[scroll-cmp][error] No Python interpreter found" >&2; exit 127
fi

# ---------- VWA env ----------
if [[ -f "${REPO_DIR}/scripts/vwa_env_remote.sh" ]]; then
  source "${REPO_DIR}/scripts/vwa_env_remote.sh" || true
elif [[ -f "${REPO_DIR}/scripts/vwa_env.sh" ]]; then
  source "${REPO_DIR}/scripts/vwa_env.sh" || true
fi
export OPENAI_API_KEY="${OPENAI_API_KEY:-DUMMY_P79_NON_LLM_EVAL}"
export PYTORCH_NVML_BASED_CUDA_CHECK=1
export CUDA_MPS_PIPE_DIRECTORY=""
export CUDA_MPS_LOG_DIRECTORY=""
export P79_DISABLE_STALE_CLEANUP="${P79_DISABLE_STALE_CLEANUP:-1}"

# ---------- API keys ----------
load_proxy_key() {
  if [[ -z "${PROXY_API_KEY:-}" ]]; then
    local auth_file="${REPO_DIR}/.auth/qwen_api"
    if [[ -f "${auth_file}" ]]; then
      local key
      key="$(grep -m1 '^rp_' "${auth_file}" | tr -d '[:space:]')"
      if [[ -n "${key}" ]]; then
        export PROXY_API_KEY="${key}"
        echo "[scroll-cmp] Loaded PROXY_API_KEY" >&2
      else
        echo "[scroll-cmp][error] No rp_ key in ${auth_file}" >&2; return 1
      fi
    else
      echo "[scroll-cmp][error] ${auth_file} not found" >&2; return 1
    fi
  fi
}

load_dashscope_key() {
  if [[ -z "${DASHSCOPE_API_KEY:-}" ]]; then
    local auth_file="${REPO_DIR}/.auth/qwen_api_official"
    if [[ -f "${auth_file}" ]]; then
      local key
      key="$(grep -m1 '^sk-' "${auth_file}" | tr -d '[:space:]')"
      if [[ -n "${key}" ]]; then
        export DASHSCOPE_API_KEY="${key}"
        echo "[scroll-cmp] Loaded DASHSCOPE_API_KEY" >&2
      else
        echo "[scroll-cmp][error] No sk- key in ${auth_file}" >&2; return 1
      fi
    else
      echo "[scroll-cmp][error] ${auth_file} not found" >&2; return 1
    fi
  fi
}

# ---------- Run function ----------
run_experiment() {
  local label="$1" config="$2" run_id="$3"
  local log_path="${LOG_DIR}/scroll_cmp_${label}_${run_id}.log"

  echo "[scroll-cmp] === ${label} === run_id=${run_id}" >&2
  echo "[scroll-cmp] config: ${config}" >&2
  echo "[scroll-cmp] log:    ${log_path}" >&2

  "${PYTHON_BIN}" scripts/run_experiment.py \
    --config "${config}" \
    --run_id "${run_id}" \
    --log_path "${log_path}" \
    2>&1 | tee -a "${log_path}"

  echo "[scroll-cmp] === ${label} done ===" >&2
}

# ---------- Main ----------
if [[ "${MODE}" == "claude" ]] || [[ "${MODE}" == "all" ]]; then
  load_proxy_key
  RUN_ID="scroll_test_claude_${DATE}"
  run_experiment "claude" \
    "${REPO_DIR}/configs/exp_v2_scroll_test_claude.yaml" \
    "${RUN_ID}"
fi

if [[ "${MODE}" == "dashscope" ]] || [[ "${MODE}" == "all" ]]; then
  load_dashscope_key
  RUN_ID="scroll_test_dashscope_${DATE}"
  run_experiment "dashscope" \
    "${REPO_DIR}/configs/exp_v2_scroll_test_dashscope.yaml" \
    "${RUN_ID}"
fi

echo "[scroll-cmp] All requested experiments complete." >&2
