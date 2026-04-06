#!/usr/bin/env bash
# restart_sidecar.sh — Hot-restart glm_diagnosis_sidecar.py instance(s)
#
# "热重启": reads each running sidecar's exact args from /proc/PID/cmdline,
# kills the old process, then re-launches with setsid+nohup using the same args.
# State files are NOT deleted — the sidecar resumes from where it left off.
# Output appends to the existing log with a restart marker.
#
# Usage:
#   bash scripts/dgx/restart_sidecar.sh [--dry-run]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_DIR}"

PYTHON_BIN="${REPO_DIR}/.venv/bin/python"
DRY_RUN=0
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  bash scripts/dgx/restart_sidecar.sh [options]

Options:
  --dry-run              Show what would be done without actually restarting.
  --append-args <args>   Extra args appended to the restarted sidecar cmdline.
                         Quoted string of space-separated flags, e.g.:
                         --append-args "--label classifieds"
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

# Collect running sidecar PIDs
mapfile -t sidecar_pids < <(
  ps -eo pid=,args= | awk '/glm_diagnosis_sidecar\.py/ && !/awk/ {print $1}'
)

if [[ ${#sidecar_pids[@]} -eq 0 ]]; then
  log "No running glm_diagnosis_sidecar.py processes found."
  exit 0
fi

log "Found ${#sidecar_pids[@]} sidecar(s): ${sidecar_pids[*]}"

for pid in "${sidecar_pids[@]}"; do
  if [[ ! -r "/proc/${pid}/cmdline" ]]; then
    log "WARNING: Cannot read /proc/${pid}/cmdline for PID ${pid}, skipping."
    continue
  fi

  # Read NUL-separated args into array
  mapfile -d '' args < "/proc/${pid}/cmdline"
  # Trim trailing empty element that mapfile may leave after final NUL
  while [[ ${#args[@]} -gt 0 && -z "${args[-1]}" ]]; do
    unset 'args[-1]'
  done

  # Resolve log file from --state-file arg (mirrors queue_b1_serial.sh naming)
  state_file=""
  run_dir=""
  for ((i = 0; i < ${#args[@]} - 1; i++)); do
    case "${args[$i]}" in
      --state-file) state_file="${args[$((i + 1))]}" ;;
      --run-dir)    run_dir="${args[$((i + 1))]}" ;;
    esac
  done

  if [[ -n "${state_file}" ]]; then
    # state_file may be absolute or relative; normalize then swap extension.
    _sf_abs="$(cd "$(dirname "${state_file}")" 2>/dev/null && pwd)/$(basename "${state_file}")" 2>/dev/null \
      || _sf_abs="${REPO_DIR}/${state_file}"
    log_file="${_sf_abs%.state.json}.log"
  else
    run_id="$(basename "${run_dir:-unknown}")"
    log_file="${REPO_DIR}/logs/live_reason_watch_${run_id}.log"
  fi

  log "PID=${pid}  log=${log_file}"
  log "  args: ${args[*]}"

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    log "  [dry-run] Would kill ${pid} and restart with setsid."
    continue
  fi

  # Kill old sidecar
  log "  Killing PID ${pid}..."
  kill "${pid}" 2>/dev/null || true
  sleep 2
  if kill -0 "${pid}" 2>/dev/null; then
    kill -9 "${pid}" 2>/dev/null || true
    sleep 1
  fi

  # Append restart marker to existing log
  {
    echo ""
    echo "=== SIDECAR HOT-RESTARTED at $(date '+%Y-%m-%d %H:%M:%S') ==="
  } >> "${log_file}" 2>/dev/null || true

  # Re-launch: args[0]=python, args[1]=-u, args[2]=script, args[3..]= flags
  # Replace python binary with venv python for consistency; keep -u and rest.
  setsid nohup "${PYTHON_BIN}" "${args[@]:1}" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" \
    >> "${log_file}" 2>&1 < /dev/null &
  new_pid=$!

  sleep 2
  if kill -0 "${new_pid}" 2>/dev/null; then
    log "  Restarted: new_pid=${new_pid}  log=${log_file}"
  else
    log "  ERROR: Sidecar failed to stay alive. Tail of log:"
    tail -n 30 "${log_file}" || true
    exit 1
  fi
done

if [[ "${DRY_RUN}" -eq 1 ]]; then
  log "[dry-run] No changes made."
else
  log "Hot-restart complete."
fi
