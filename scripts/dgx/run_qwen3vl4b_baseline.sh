#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_PATH="${BASELINE_CONFIG:-${REPO_DIR}/configs/exp_v2_qwen3vl4b_baseline.yaml}"
LOG_DIR="${REPO_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_PATH_DEFAULT="${LOG_DIR}/baseline_qwen3vl4b_$(date +%F_%H%M%S).log"
LOG_PATH="${BASELINE_LOG_PATH:-${LOG_PATH_DEFAULT}}"
SITE_HEALTH_TIMEOUT="${BASELINE_SITE_HEALTH_TIMEOUT:-6}"
RUN_ID=""
if [[ -n "${BASELINE_RUN_ID:-}" ]]; then
  RUN_ID="${BASELINE_RUN_ID}"
elif [[ -n "${BASELINE_RUN_ID_PREFIX:-}" ]]; then
  RUN_ID="${BASELINE_RUN_ID_PREFIX}_$(date +%Y%m%d_%H%M%S)"
fi

site_env_var_for() {
  case "${1:-}" in
    shopping) printf "SHOPPING\n" ;;
    reddit) printf "REDDIT\n" ;;
    wikipedia) printf "WIKIPEDIA\n" ;;
    classifieds) printf "CLASSIFIEDS\n" ;;
    *)
      return 1
      ;;
  esac
}

extract_host_from_url() {
  local url="${1:-}"
  local host_port=""
  if [[ -z "${url}" ]]; then
    return 1
  fi
  host_port="${url#*://}"
  host_port="${host_port%%/*}"
  host_port="${host_port%%:*}"
  if [[ -z "${host_port}" ]]; then
    return 1
  fi
  printf "%s\n" "${host_port}"
  return 0
}

is_local_hostname() {
  local host="${1:-}"
  if [[ -z "${host}" ]]; then
    return 1
  fi
  case "${host}" in
    localhost|127.0.0.1|::1)
      return 0
      ;;
  esac
  if hostname -I 2>/dev/null | tr ' ' '\n' | grep -Fxq "${host}"; then
    return 0
  fi
  return 1
}

is_site_http_healthy() {
  local url="${1:-}"
  local code=""
  if [[ -z "${url}" ]]; then
    return 1
  fi
  if ! command -v curl >/dev/null 2>&1; then
    echo "[baseline][warn] curl not found; skip endpoint health check for ${url}" >&2
    return 0
  fi

  code="$(curl -sS -o /dev/null -w "%{http_code}" -m "${SITE_HEALTH_TIMEOUT}" "${url}" || true)"
  echo "[baseline] endpoint health url=${url} http=${code:-000}" >&2
  case "${code}" in
    000|502|503|504)
      return 1
      ;;
    *)
      return 0
      ;;
  esac
}

validate_shopping_redirect_host() {
  local shopping_url="${1:-}"
  local shopping_host="${2:-}"
  local location=""
  local location_host=""

  if [[ -z "${shopping_url}" ]] || [[ -z "${shopping_host}" ]]; then
    return 0
  fi
  if ! command -v curl >/dev/null 2>&1; then
    return 0
  fi

  location="$(curl -sS -I -m "${SITE_HEALTH_TIMEOUT}" "${shopping_url}" | awk 'BEGIN{IGNORECASE=1} /^Location:/ {print $2; exit}' | tr -d '\r' || true)"
  if [[ -z "${location}" ]]; then
    return 0
  fi
  location_host="$(extract_host_from_url "${location}" || true)"
  if [[ -z "${location_host}" ]]; then
    return 0
  fi
  if is_local_hostname "${location_host}" && ! is_local_hostname "${shopping_host}"; then
    echo "[baseline][error] shopping endpoint redirects to local host: ${location}" >&2
    return 1
  fi
  return 0
}

validate_shopping_page_links() {
  local shopping_url="${1:-}"
  local shopping_host="${2:-}"
  local page_html=""

  if [[ -z "${shopping_url}" ]] || [[ -z "${shopping_host}" ]]; then
    return 0
  fi
  if is_local_hostname "${shopping_host}"; then
    return 0
  fi
  if ! command -v curl >/dev/null 2>&1; then
    return 0
  fi

  page_html="$(curl -sS -L -m "${SITE_HEALTH_TIMEOUT}" "${shopping_url}" || true)"
  if grep -Eqi '(https?:)?//(localhost|127\.0\.0\.1)(:7770)?/' <<< "${page_html}"; then
    echo "[baseline][error] shopping homepage still contains localhost links (e.g. http://localhost:7770/...)." >&2
    echo "[baseline][error] Clicking those links on DGX will produce 502." >&2
    echo "[baseline][error] On the docker host machine, re-run: bash scripts/start_vwa_docker.sh --sites shopping --hostname <PUBLIC_HOST_IP>" >&2
    return 1
  fi
  return 0
}

detect_include_sites_from_config() {
  local line=""
  local inner=""
  line="$(awk '/^[[:space:]]*include_sites:[[:space:]]*\[/{print; exit}' "${CONFIG_PATH}" || true)"
  if [[ -z "${line}" ]]; then
    return 1
  fi
  inner="${line#*[}"
  inner="${inner%]*}"
  printf "%s\n" "${inner}" | tr ',' '\n' | sed -E 's/["'"'"'[:space:]]//g' | awk 'NF>0'
  return 0
}

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

if [[ "${BASELINE_REQUIRE_SITE_UP:-1}" != "0" ]]; then
  mapfile -t INCLUDE_SITES < <(detect_include_sites_from_config || true)
  if [[ "${#INCLUDE_SITES[@]}" -eq 0 ]]; then
    INCLUDE_SITES=(shopping reddit wikipedia classifieds)
    echo "[baseline][warn] failed to parse include_sites from ${CONFIG_PATH}; checking all known sites." >&2
  fi

  for site in "${INCLUDE_SITES[@]}"; do
    site_var="$(site_env_var_for "${site}" || true)"
    if [[ -z "${site_var}" ]]; then
      continue
    fi
    site_endpoint="${!site_var:-}"
    if [[ -z "${site_endpoint}" ]]; then
      echo "[baseline][error] ${site_var} is unset for site=${site}" >&2
      echo "[baseline][error] Provide VWA_ENV_FILE=scripts/vwa_env_remote.sh or export ${site_var}=http://<host>:<port>" >&2
      exit 3
    fi
    if ! is_site_http_healthy "${site_endpoint}"; then
      echo "[baseline][error] selected endpoint is unhealthy: ${site_var}=${site_endpoint}" >&2
      echo "[baseline][error] On DGX + remote docker, avoid localhost and use remote host ports." >&2
      exit 3
    fi
    if [[ "${site}" == "shopping" ]]; then
      shopping_host="$(extract_host_from_url "${site_endpoint}" || true)"
      if ! validate_shopping_redirect_host "${site_endpoint}" "${shopping_host}"; then
        echo "[baseline][error] shopping base_url looks incorrect; aborting to avoid 502 runs." >&2
        exit 3
      fi
      if ! validate_shopping_page_links "${site_endpoint}" "${shopping_host}"; then
        echo "[baseline][error] shopping page links still point to localhost; aborting to avoid 502 runs." >&2
        exit 3
      fi
    fi
  done
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
if [[ -n "${RUN_ID}" ]]; then
  echo "[baseline] run_id: ${RUN_ID}" >&2
fi

set +e
cmd=("${PYTHON_BIN}" scripts/run_experiment.py --config "${CONFIG_PATH}" --log_path "${LOG_PATH}")
if [[ -n "${RUN_ID}" ]]; then
  cmd+=(--run_id "${RUN_ID}")
fi
"${cmd[@]}" 2>&1 | tee -a "${LOG_PATH}"
rc=${PIPESTATUS[0]}
set -e
exit "${rc}"
