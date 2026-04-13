#!/usr/bin/env bash
# restart_watchdog.sh — Hot-restart experiment_watchdog.py instance(s)
#
# Finds running watchdog processes (experiment_watchdog.py or legacy
# monitor_glm_sidecar.py), kills them, and relaunches with
# experiment_watchdog.py using the same --run-dir/--condition/--ntfy-topic.
# State files are preserved — the watchdog resumes from where it left off.
#
# Usage:
#   bash scripts/dgx/restart_watchdog.sh [--dry-run] [--append-args "<flags>"]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_DIR}"

PYTHON_BIN="${REPO_DIR}/.venv/bin/python"
WATCHDOG_SCRIPT="scripts/experiment_watchdog.py"
DRY_RUN=0
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  bash scripts/dgx/restart_watchdog.sh [options]

Options:
  --dry-run              Show what would be done without actually restarting.
  --append-args <args>   Extra args appended to the restarted watchdog cmdline.
                         Quoted string of space-separated flags, e.g.:
                         --append-args "--window-size 30"
  -h, --help             Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --append-args) IFS=' ' read -ra EXTRA_ARGS <<< "$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

if [[ ! -f "${WATCHDOG_SCRIPT}" ]]; then
  log "ERROR: ${WATCHDOG_SCRIPT} not found in ${REPO_DIR}"
  exit 1
fi

# Collect running watchdog PIDs (match both current and legacy names)
mapfile -t watchdog_pids < <(
  ps -eo pid=,args= | awk '/experiment_watchdog\.py|monitor_glm_sidecar\.py/ && !/awk/ {print $1}'
)

if [[ ${#watchdog_pids[@]} -eq 0 ]]; then
  log "No running watchdog processes found."
  exit 0
fi

log "Found ${#watchdog_pids[@]} watchdog(s): ${watchdog_pids[*]}"

for pid in "${watchdog_pids[@]}"; do
  if [[ ! -r "/proc/${pid}/cmdline" ]]; then
    log "WARNING: Cannot read /proc/${pid}/cmdline for PID ${pid}, skipping."
    continue
  fi

  # Read NUL-separated args into array
  mapfile -d '' args < "/proc/${pid}/cmdline"
  while [[ ${#args[@]} -gt 0 && -z "${args[-1]}" ]]; do
    unset 'args[-1]'
  done

  # Extract key arguments to pass to the new watchdog
  run_dir=""
  condition=""
  ntfy_topic=""
  state_file=""
  poll_secs=""
  idle_alert_mins=""
  glm_config=""
  digest_dir=""
  notify_completion=0
  for ((i = 0; i < ${#args[@]} - 1; i++)); do
    case "${args[$i]}" in
      --run-dir)       run_dir="${args[$((i + 1))]}" ;;
      --condition)     condition="${args[$((i + 1))]}" ;;
      --ntfy-topic)    ntfy_topic="${args[$((i + 1))]}" ;;
      --state-file)    state_file="${args[$((i + 1))]}" ;;
      --poll-secs)     poll_secs="${args[$((i + 1))]}" ;;
      --idle-alert-mins) idle_alert_mins="${args[$((i + 1))]}" ;;
      --glm-config)    glm_config="${args[$((i + 1))]}" ;;
      --digest-dir)    digest_dir="${args[$((i + 1))]}" ;;
      --notify-completion) notify_completion=1 ;;
    esac
  done
  if [[ "${args[-1]}" == "--notify-completion" ]]; then
    notify_completion=1
  fi

  if [[ -z "${run_dir}" ]]; then
    log "WARNING: PID ${pid} has no --run-dir, skipping."
    continue
  fi

  # Build new command
  new_args=(-u "${WATCHDOG_SCRIPT}" --run-dir "${run_dir}")
  [[ -n "${condition}" ]]      && new_args+=(--condition "${condition}")
  [[ -n "${ntfy_topic}" ]]     && new_args+=(--ntfy-topic "${ntfy_topic}")
  [[ -n "${poll_secs}" ]]      && new_args+=(--poll-secs "${poll_secs}")
  [[ -n "${state_file}" ]]     && new_args+=(--state-file "${state_file}")
  [[ -n "${idle_alert_mins}" ]] && new_args+=(--idle-alert-mins "${idle_alert_mins}")
  [[ -n "${glm_config}" ]]     && new_args+=(--glm-config "${glm_config}")
  [[ -n "${digest_dir}" ]]     && new_args+=(--digest-dir "${digest_dir}")
  [[ "${notify_completion}" -eq 1 ]] && new_args+=(--notify-completion)

  # Determine log file
  if [[ -n "${state_file}" ]]; then
    _sf_abs="$(cd "$(dirname "${state_file}")" 2>/dev/null && pwd)/$(basename "${state_file}")" 2>/dev/null \
      || _sf_abs="${REPO_DIR}/${state_file}"
    log_file="${_sf_abs%.state.json}.log"
  else
    run_id="$(basename "${run_dir}")"
    cond_suffix="${condition:-all}"
    log_file="${REPO_DIR}/logs/watchdog_${run_id}_${cond_suffix}.log"
  fi

  log "PID=${pid}  old_script=$(echo "${args[@]}" | grep -oP '\S+\.py' | head -1)"
  log "  old args: ${args[*]}"
  log "  new args: ${PYTHON_BIN} ${new_args[*]} ${EXTRA_ARGS[*]+"${EXTRA_ARGS[*]}"}"
  log "  log=${log_file}"

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    log "  [dry-run] Would kill ${pid} and restart."
    continue
  fi

  # Kill old watchdog
  log "  Killing PID ${pid}..."
  kill "${pid}" 2>/dev/null || true
  sleep 2
  if kill -0 "${pid}" 2>/dev/null; then
    kill -9 "${pid}" 2>/dev/null || true
    sleep 1
  fi

  # Append restart marker
  mkdir -p "$(dirname "${log_file}")"
  {
    echo ""
    echo "=== WATCHDOG HOT-RESTARTED at $(date '+%Y-%m-%d %H:%M:%S') ==="
  } >> "${log_file}" 2>/dev/null || true

  # Re-launch
  setsid nohup "${PYTHON_BIN}" "${new_args[@]}" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" \
    >> "${log_file}" 2>&1 < /dev/null &
  new_pid=$!

  sleep 2
  if kill -0 "${new_pid}" 2>/dev/null; then
    log "  Restarted: new_pid=${new_pid}  log=${log_file}"
  else
    log "  ERROR: Watchdog failed to stay alive. Tail of log:"
    tail -n 30 "${log_file}" || true
    exit 1
  fi
done

if [[ "${DRY_RUN}" -eq 1 ]]; then
  log "[dry-run] No changes made."
else
  log "Hot-restart complete."
fi
