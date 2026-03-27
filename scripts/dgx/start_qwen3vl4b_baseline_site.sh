#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BASE_CONFIG="${REPO_DIR}/configs/exp_v2_qwen3vl4b_baseline.yaml"

SITE="${BASELINE_SITE:-shopping}"
MAX_STEPS="${BASELINE_MAX_STEPS:-20}"
SITE_HEALTH_TIMEOUT="${BASELINE_SITE_HEALTH_TIMEOUT:-6}"

usage() {
  cat <<USAGE
Usage: $(basename "$0") [--site <shopping|reddit|wikipedia|classifieds>] [--max-steps <N>]

Environment overrides:
  BASELINE_SITE        default site (default: shopping)
  BASELINE_MAX_STEPS   default max steps (default: 20)
  BASELINE_LOG_PATH    explicit log path (default: logs/baseline_qwen3vl4b_<site>_<timestamp>.log)
  BASELINE_FIX_SHOPPING_BASEURL  auto-fix shopping Magento base_url before run (default: 1)
  BASELINE_PREFER_LOCAL_SHOPPING prefer localhost shopping endpoint for --site shopping (default: 0)
  BASELINE_SHOPPING_URL explicit shopping endpoint override for --site shopping
  BASELINE_REDDIT_URL explicit reddit endpoint override for --site reddit
  BASELINE_WIKIPEDIA_URL explicit wikipedia endpoint override for --site wikipedia
  BASELINE_CLASSIFIEDS_URL explicit classifieds endpoint override for --site classifieds
  BASELINE_REQUIRE_SITE_UP fail fast when selected site endpoint is unreachable/502 (default: 1)
  BASELINE_SITE_HEALTH_TIMEOUT curl timeout seconds for site health check (default: 6)
  BASELINE_DRY_RUN      validate env and print launch config without starting run (default: 0)
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --site)
      SITE="$2"
      shift 2
      ;;
    --max-steps)
      MAX_STEPS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "${SITE}" in
  shopping|reddit|wikipedia|classifieds) ;;
  *)
    echo "Unsupported --site: ${SITE}" >&2
    exit 2
    ;;
esac

if ! [[ "${MAX_STEPS}" =~ ^[0-9]+$ ]]; then
  echo "--max-steps must be an integer, got: ${MAX_STEPS}" >&2
  exit 2
fi

if [[ ! -f "${BASE_CONFIG}" ]]; then
  echo "Missing baseline config: ${BASE_CONFIG}" >&2
  exit 1
fi

cd "${REPO_DIR}"
mkdir -p logs

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
  echo "[baseline] endpoint health url=${url} http=${code:-000}"

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
    echo "[baseline][error] This usually causes DGX-side 502/blank pages. Fix Magento base_url on the docker host first." >&2
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

if [[ "${SITE}" == "shopping" ]]; then
  if [[ -n "${BASELINE_SHOPPING_URL:-}" ]]; then
    export SHOPPING="${BASELINE_SHOPPING_URL}"
  elif [[ -z "${SHOPPING:-}" ]] && [[ "${BASELINE_PREFER_LOCAL_SHOPPING:-0}" == "1" ]]; then
    export SHOPPING="http://localhost:7770"
  fi
  echo "[baseline] shopping endpoint=${SHOPPING:-<unset>}"
fi

SITE_ENV_VAR="$(site_env_var_for "${SITE}")"
SITE_UPPER="$(echo "${SITE}" | tr '[:lower:]' '[:upper:]')"
SITE_URL_OVERRIDE_VAR="BASELINE_${SITE_UPPER}_URL"
SITE_URL_OVERRIDE="${!SITE_URL_OVERRIDE_VAR:-}"
if [[ -n "${SITE_URL_OVERRIDE}" ]]; then
  export "${SITE_ENV_VAR}=${SITE_URL_OVERRIDE}"
  echo "[baseline] ${SITE_ENV_VAR} overridden by ${SITE_URL_OVERRIDE_VAR}=${SITE_URL_OVERRIDE}"
fi

if [[ "${SITE}" == "shopping" ]] && [[ "${BASELINE_FIX_SHOPPING_BASEURL:-1}" != "0" ]]; then
  SHOPPING_HOST="$(extract_host_from_url "${SHOPPING:-}" || true)"
  if [[ -n "${SHOPPING_HOST}" ]]; then
    if is_local_hostname "${SHOPPING_HOST}"; then
      echo "[baseline] ensure shopping base_url host=${SHOPPING_HOST}"
      if ! bash "${REPO_DIR}/scripts/start_vwa_docker.sh" --sites shopping --hostname "${SHOPPING_HOST}" >/dev/null 2>&1; then
        echo "[baseline][warn] failed to auto-fix shopping base_url; run may still hit 502 if page links point to localhost:7770" >&2
      fi
    else
      echo "[baseline][warn] shopping host ${SHOPPING_HOST} is remote from this machine; local base_url auto-fix skipped" >&2
    fi
  fi
fi

SITE_ENDPOINT="${!SITE_ENV_VAR:-}"
if [[ -z "${SITE_ENDPOINT}" ]]; then
  echo "[baseline][error] ${SITE_ENV_VAR} is unset for --site ${SITE}" >&2
  echo "[baseline][error] Provide VWA_ENV_FILE=scripts/vwa_env_remote.sh or export ${SITE_ENV_VAR}=http://<host>:<port>" >&2
  exit 2
fi

if [[ "${BASELINE_REQUIRE_SITE_UP:-1}" != "0" ]]; then
  if ! is_site_http_healthy "${SITE_ENDPOINT}"; then
    echo "[baseline][error] selected endpoint is unhealthy: ${SITE_ENV_VAR}=${SITE_ENDPOINT}" >&2
    echo "[baseline][error] On DGX + remote docker, avoid localhost and use remote host ports (e.g. 100.95.81.103:7770)." >&2
    exit 3
  fi
fi

if [[ "${SITE}" == "shopping" ]]; then
  SHOPPING_HOST="$(extract_host_from_url "${SITE_ENDPOINT}" || true)"
  if ! validate_shopping_redirect_host "${SITE_ENDPOINT}" "${SHOPPING_HOST}"; then
    exit 3
  fi
  if ! validate_shopping_page_links "${SITE_ENDPOINT}" "${SHOPPING_HOST}"; then
    exit 3
  fi
fi

if [[ -x "${REPO_DIR}/.venv/bin/python" ]]; then
  PYTHON_BIN="${REPO_DIR}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
else
  echo "No python found (.venv/bin/python or python3)" >&2
  exit 127
fi

TMP_CONFIG="/tmp/exp_v2_qwen3vl4b_${SITE}.yaml"
cp "${BASE_CONFIG}" "${TMP_CONFIG}"
sed -i -E "s/include_sites:[[:space:]]*\[[^]]*\]/include_sites: [\"${SITE}\"]/" "${TMP_CONFIG}"
sed -i -E "s/name:[[:space:]]*\"qwen3vl4b_baseline_phase2\"/name: \"qwen3vl4b_baseline_phase2_${SITE}\"/" "${TMP_CONFIG}"

STAMP="$(date +%F_%H%M%S)"
DEFAULT_LOG="${REPO_DIR}/logs/baseline_qwen3vl4b_${SITE}_${STAMP}.log"
LOG_PATH="${BASELINE_LOG_PATH:-${DEFAULT_LOG}}"

if [[ "${BASELINE_DRY_RUN:-0}" == "1" ]]; then
  echo "[baseline] dry-run mode enabled; no process launched."
  echo "site=${SITE}"
  echo "config=${TMP_CONFIG}"
  echo "max_steps=${MAX_STEPS}"
  echo "endpoint_var=${SITE_ENV_VAR}"
  echo "endpoint=${SITE_ENDPOINT}"
  echo "log=${LOG_PATH}"
  exit 0
fi

nohup "${PYTHON_BIN}" scripts/run_experiment.py \
  --config "${TMP_CONFIG}" \
  --max_steps "${MAX_STEPS}" \
  > "${LOG_PATH}" 2>&1 < /dev/null &
PID=$!

echo "site=${SITE}"
echo "pid=${PID}"
echo "config=${TMP_CONFIG}"
echo "log=${LOG_PATH}"
echo "watch: tail -f ${LOG_PATH}"
