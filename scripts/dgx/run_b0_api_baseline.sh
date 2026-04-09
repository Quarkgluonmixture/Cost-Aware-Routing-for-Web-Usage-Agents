#!/usr/bin/env bash
# run_b0_api_baseline.sh — Run B0 strong upper bound using Qwen3-VL-235B via proxy API
#
# B0 purpose: confirm task solvability and establish performance ceiling.
# Unlike B1 (local Qwen3-VL-4B), B0 uses the API model — no local GPU inference.
#
# API key source: .auth/qwen_api (single-line raw key)
#   Expected format: one line containing the proxy API key (rp_...)
#   Export PROXY_API_KEY manually to override.
#
# Usage:
#   bash scripts/dgx/run_b0_api_baseline.sh
#   B0_RUN_ID=my_b0_run bash scripts/dgx/run_b0_api_baseline.sh
#   B0_CONFIG=configs/exp_v2_B0_baseline.yaml bash scripts/dgx/run_b0_api_baseline.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ---------- API key loading ----------
# Load API key from .auth/qwen_api unless already set.
# Used as PROXY_API_KEY for the custom proxy API (also kept as QWEN_API_KEY for compat).
AUTH_FILE="${REPO_DIR}/.auth/qwen_api"
if [[ -z "${PROXY_API_KEY:-}" ]]; then
  if [[ -f "${AUTH_FILE}" ]]; then
    raw_key="$(grep -m1 '^rp_' "${AUTH_FILE}" | tr -d '[:space:]')"
    if [[ -n "${raw_key}" ]]; then
      export PROXY_API_KEY="${raw_key}"
      export QWEN_API_KEY="${raw_key}"
      export DASHSCOPE_API_KEY="${raw_key}"
      echo "[b0] Loaded PROXY_API_KEY from ${AUTH_FILE}" >&2
    else
      echo "[b0][error] ${AUTH_FILE} exists but is empty." >&2
      exit 1
    fi
  else
    echo "[b0][error] ${AUTH_FILE} not found and PROXY_API_KEY not set." >&2
    echo "[b0][error] Either create ${AUTH_FILE} with your API key (rp_...) or:" >&2
    echo "[b0][error]   export PROXY_API_KEY=rp_..." >&2
    exit 1
  fi
else
  echo "[b0] PROXY_API_KEY already set in environment." >&2
fi

# ---------- Config and output paths ----------
CONFIG_PATH="${B0_CONFIG:-${REPO_DIR}/configs/exp_v2_B0_baseline.yaml}"
LOG_DIR="${REPO_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_PATH="${LOG_DIR}/b0_api_$(date +%F_%H%M%S).log"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[b0][error] Config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

# ---------- VWA site environment ----------
# Browser environment is still required even for API model (VWA task execution).
if [[ -n "${VWA_ENV_FILE:-}" ]]; then
  if [[ -f "${VWA_ENV_FILE}" ]]; then
    # shellcheck disable=SC1090
    source "${VWA_ENV_FILE}" || true
  else
    echo "[b0][warn] VWA_ENV_FILE does not exist: ${VWA_ENV_FILE}" >&2
  fi
elif [[ -f "${REPO_DIR}/scripts/vwa_env_remote.sh" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_DIR}/scripts/vwa_env_remote.sh" || true
elif [[ -f "${REPO_DIR}/scripts/vwa_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_DIR}/scripts/vwa_env.sh" || true
fi

# VWA evaluators may import OpenAI modules even for non-LLM-graded tasks.
export OPENAI_API_KEY="${OPENAI_API_KEY:-DUMMY_P79_NON_LLM_EVAL}"

# ---------- DGX environment (minimal — no local model) ----------
export PYTORCH_NVML_BASED_CUDA_CHECK=1
export CUDA_MPS_PIPE_DIRECTORY=""
export CUDA_MPS_LOG_DIRECTORY=""

# ---------- Python interpreter ----------
if [[ -x "${REPO_DIR}/.venv/bin/python" ]]; then
  PYTHON_BIN="${REPO_DIR}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
else
  echo "[b0][error] No python interpreter found (.venv/bin/python, python3, python)" >&2
  exit 127
fi

# ---------- Watchdog settings ----------
B0_WATCHDOG="${B0_WATCHDOG:-1}"
NTFY_TOPIC="${NTFY_TOPIC:-p79-exp-dgx-spark}"
WATCHDOG_POLL_SECS="${WATCHDOG_POLL_SECS:-30}"
WATCHDOG_IDLE_ALERT_MINS="${WATCHDOG_IDLE_ALERT_MINS:-20}"
WATCHDOG_GLM_CONFIG="${REPO_DIR}/.auth/glm"
WATCHDOG_PID=""

# ---------- Run ID ----------
RUN_ID="${B0_RUN_ID:-B0_api_strong_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${REPO_DIR}/results/visualwebarena/phase1/${RUN_ID}"

echo "[b0] run_id:  ${RUN_ID}" >&2
echo "[b0] config:  ${CONFIG_PATH}" >&2
echo "[b0] log:     ${LOG_PATH}" >&2
echo "[b0] model:   qwen3-vl-235b-a22b (via proxy API)" >&2
echo "[b0] watchdog: ${B0_WATCHDOG}" >&2

cd "${REPO_DIR}"

# ---------- Cleanup handler ----------
cleanup() {
  if [[ -n "${WATCHDOG_PID}" ]] && kill -0 "${WATCHDOG_PID}" 2>/dev/null; then
    echo "[b0] stopping watchdog pid=${WATCHDOG_PID}" >&2
    kill "${WATCHDOG_PID}" 2>/dev/null || true
    sleep 1
    kill -9 "${WATCHDOG_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# ---------- Start watchdog (optional) ----------
if [[ "${B0_WATCHDOG}" == "1" ]]; then
  WATCHDOG_SCRIPT="${REPO_DIR}/scripts/experiment_watchdog.py"
  if [[ -f "${WATCHDOG_SCRIPT}" ]]; then
    mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"
    WATCHDOG_LOG="${LOG_DIR}/experiment_watchdog_b0_${RUN_ID}.log"
    WATCHDOG_STATE="${LOG_DIR}/experiment_watchdog_b0_${RUN_ID}.state.json"

    watchdog_cmd=(
      "${PYTHON_BIN}" -u "${WATCHDOG_SCRIPT}"
      --run-dir "${OUTPUT_DIR}"
      --poll-secs "${WATCHDOG_POLL_SECS}"
      --idle-alert-mins "${WATCHDOG_IDLE_ALERT_MINS}"
      --ntfy-topic "${NTFY_TOPIC}"
      --state-file "${WATCHDOG_STATE}"
    )
    if [[ -f "${WATCHDOG_GLM_CONFIG}" ]]; then
      watchdog_cmd+=(--glm-config "${WATCHDOG_GLM_CONFIG}")
      watchdog_cmd+=(--digest-dir "${OUTPUT_DIR}/analysis/digest")
    fi

    nohup "${watchdog_cmd[@]}" > "${WATCHDOG_LOG}" 2>&1 < /dev/null &
    WATCHDOG_PID=$!
    sleep 1
    if kill -0 "${WATCHDOG_PID}" 2>/dev/null; then
      echo "[b0] watchdog started: pid=${WATCHDOG_PID} log=${WATCHDOG_LOG}" >&2
    else
      echo "[b0][warn] watchdog failed to start. log=${WATCHDOG_LOG}" >&2
      WATCHDOG_PID=""
    fi
  else
    echo "[b0][warn] watchdog script not found: ${WATCHDOG_SCRIPT}" >&2
  fi
fi

# ---------- Run experiment ----------
set +e
"${PYTHON_BIN}" scripts/run_experiment.py \
  --config "${CONFIG_PATH}" \
  --run_id "${RUN_ID}" \
  --log_path "${LOG_PATH}" \
  2>&1 | tee -a "${LOG_PATH}"
rc=${PIPESTATUS[0]}
set -e

echo "[b0] exit code: ${rc}" >&2
exit "${rc}"
