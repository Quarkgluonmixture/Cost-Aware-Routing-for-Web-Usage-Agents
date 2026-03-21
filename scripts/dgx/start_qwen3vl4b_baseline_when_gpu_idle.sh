#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
STATE_DIR="${REPO_DIR}/.autostart"

CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-30}"
IDLE_UTIL_THRESHOLD="${IDLE_UTIL_THRESHOLD:-5}"
IDLE_MEM_THRESHOLD_MIB="${IDLE_MEM_THRESHOLD_MIB:-1024}"

WATCH_PID_FILE="${STATE_DIR}/qwen3vl4b_watch.pid"
RUN_PID_FILE="${STATE_DIR}/qwen3vl4b_run.pid"
LAUNCH_MARK_FILE="${STATE_DIR}/qwen3vl4b_launched.ok"
RUN_LOG_FILE="${STATE_DIR}/qwen3vl4b_run.log"
WATCH_LOG_FILE="${STATE_DIR}/qwen3vl4b_watch.log"
RUN_EXIT_FILE="${STATE_DIR}/qwen3vl4b_run.exit_code"
RUN_DONE_OK_FILE="${STATE_DIR}/qwen3vl4b_run.done.ok"
RUN_DONE_FAIL_FILE="${STATE_DIR}/qwen3vl4b_run.done.fail"

mkdir -p "${STATE_DIR}"

# Avoid duplicate watchers.
if [[ -f "${WATCH_PID_FILE}" ]]; then
  old_pid="$(cat "${WATCH_PID_FILE}" 2>/dev/null || true)"
  if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
    exit 0
  fi
fi
echo "$$" > "${WATCH_PID_FILE}"
trap 'rm -f "${WATCH_PID_FILE}"' EXIT

gpu_is_idle() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 1
  fi

  local util_raw mem_raw util mem procs
  util_raw="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -n1 | tr -d '[:space:]')"
  mem_raw="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n1 | tr -d '[:space:]')"
  procs="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | awk 'NF>0' | wc -l | tr -d '[:space:]')"

  util="$(echo "${util_raw}" | tr -cd '0-9')"
  mem="$(echo "${mem_raw}" | tr -cd '0-9')"

  [[ -n "${procs}" ]] || return 1
  [[ -n "${util}" ]] || util="100"

  if [[ "${procs}" -ne 0 ]]; then
    return 1
  fi

  if [[ "${util}" -gt "${IDLE_UTIL_THRESHOLD}" ]]; then
    return 1
  fi

  # Some drivers report memory.used as N/A; in that case we skip memory threshold check.
  if [[ -n "${mem}" ]] && [[ "${mem}" -gt "${IDLE_MEM_THRESHOLD_MIB}" ]]; then
    return 1
  fi

  return 0
}

{
  echo "[$(date '+%F %T')] watcher started (check=${CHECK_INTERVAL_SECONDS}s util<=${IDLE_UTIL_THRESHOLD} mem<=${IDLE_MEM_THRESHOLD_MIB}MiB)"
} >> "${WATCH_LOG_FILE}"

while true; do
  if [[ -f "${LAUNCH_MARK_FILE}" ]]; then
    exit 0
  fi

  if gpu_is_idle; then
    {
      echo "[$(date '+%F %T')] gpu idle detected; launching baseline"
    } >> "${WATCH_LOG_FILE}"

    rm -f "${RUN_EXIT_FILE}" "${RUN_DONE_OK_FILE}" "${RUN_DONE_FAIL_FILE}"
    nohup bash -lc "
      set +e
      bash '${REPO_DIR}/scripts/dgx/run_qwen3vl4b_baseline.sh' >> '${RUN_LOG_FILE}' 2>&1
      code=\$?
      echo \"\${code}\" > '${RUN_EXIT_FILE}'
      if [[ \${code} -eq 0 ]]; then
        touch '${RUN_DONE_OK_FILE}'
      else
        touch '${RUN_DONE_FAIL_FILE}'
      fi
      exit 0
    " >/dev/null 2>&1 < /dev/null &
    echo "$!" > "${RUN_PID_FILE}"
    touch "${LAUNCH_MARK_FILE}"

    {
      echo "[$(date '+%F %T')] baseline pid=$(cat "${RUN_PID_FILE}")"
    } >> "${WATCH_LOG_FILE}"
    exit 0
  fi

  sleep "${CHECK_INTERVAL_SECONDS}"
done
