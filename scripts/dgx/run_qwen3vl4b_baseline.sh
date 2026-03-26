#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_PATH="${REPO_DIR}/configs/exp_v2_qwen3vl4b_baseline.yaml"

cd "${REPO_DIR}"

# DGX Spark quirks: avoid CUDA probe / MPS related hangs.
export CUDA_MPS_PIPE_DIRECTORY=""
export CUDA_MPS_LOG_DIRECTORY=""
export PYTORCH_NVML_BASED_CUDA_CHECK=1

# Best-effort conda activation for a reproducible env.
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)" || true
  conda activate p79_ai || true
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

exec "${PYTHON_BIN}" scripts/run_experiment.py --config "${CONFIG_PATH}"
