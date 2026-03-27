#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_PATH="${REPO_DIR}/configs/exp_v2_qwen3vl4b_baseline.yaml"
LOG_DIR="${REPO_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_PATH_DEFAULT="${LOG_DIR}/baseline_qwen3vl4b_$(date +%F_%H%M%S).log"
LOG_PATH="${BASELINE_LOG_PATH:-${LOG_PATH_DEFAULT}}"

cd "${REPO_DIR}"

# DGX Spark quirks: avoid CUDA probe / MPS related hangs.
export CUDA_MPS_PIPE_DIRECTORY=""
export CUDA_MPS_LOG_DIRECTORY=""
export PYTORCH_NVML_BASED_CUDA_CHECK=1

# Best-effort conda activation for a reproducible env.
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)" || true
  if conda env list 2>/dev/null | awk '{print $1}' | grep -qx "p79_ai"; then
    conda activate p79_ai || true
  fi
fi

# Best-effort VWA environment loading.
if [[ -n "${VWA_ENV_FILE:-}" ]]; then
  if [[ -f "${VWA_ENV_FILE}" ]]; then
    # shellcheck disable=SC1090
    source "${VWA_ENV_FILE}" || true
  else
    echo "VWA_ENV_FILE does not exist: ${VWA_ENV_FILE}" >&2
  fi
elif [[ -f "${REPO_DIR}/scripts/vwa_env_remote.sh" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_DIR}/scripts/vwa_env_remote.sh" || true
elif [[ -f "${REPO_DIR}/scripts/vwa_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_DIR}/scripts/vwa_env.sh" || true
fi

# VisualWebArena may import OpenAI provider modules during evaluator setup even
# when current tasks do not require LLM-based judging.
export OPENAI_API_KEY="${OPENAI_API_KEY:-DUMMY_P79_NON_LLM_EVAL}"

if command -v x86_64-conda-linux-gnu-gcc >/dev/null 2>&1; then
  export CC
  CC="$(command -v x86_64-conda-linux-gnu-gcc)"
fi

if [[ -x "${REPO_DIR}/.venv/bin/python" ]]; then
  PYTHON_BIN="${REPO_DIR}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
else
  echo "No python interpreter found (.venv/bin/python, python3, python)" >&2
  exit 127
fi

echo "[baseline] log file: ${LOG_PATH}" >&2

set +e
"${PYTHON_BIN}" scripts/run_experiment.py --config "${CONFIG_PATH}" 2>&1 | tee -a "${LOG_PATH}"
rc=${PIPESTATUS[0]}
set -e
exit "${rc}"
