#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_DIR}"

RESULTS_BASE="${REPO_DIR}/results/visualwebarena/phase1"

RUN_ID_CLASSIFIEDS="${RUN_ID_CLASSIFIEDS:-}"
RUN_ID_REDDIT="${RUN_ID_REDDIT:-}"
RUN_ID_SHOPPING="${RUN_ID_SHOPPING:-}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/dgx/restart_queue_b1_serial.sh [options]

Options:
  --run-id-classifieds ID     Override classifieds run_id.
  --run-id-reddit ID          Override reddit run_id.
  --run-id-shopping ID        Override shopping run_id.
  -h, --help                  Show this help.

Notes:
  1) If run_ids are not provided, the script tries (in order):
     - current running queue process env
     - latest queue meta file in logs/
     - queue_b1_serial.sh defaults
  2) OpenAI key is loaded from .auth/openai_key (or OPENAI_API_KEY if already set).
EOF
}

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

restart_gallery_server() {
  local gallery_dir="$1"
  local gallery_port="$2"
  local gallery_log="logs/gallery_http_${gallery_port}.log"
  local python_bin="${PYTHON_BIN:-python3}"

  # Stop any previous gallery server on the same port.
  local gallery_pids
  gallery_pids="$(
    ps -eo pid=,args= | awk -v port="${gallery_port}" '
      {
        pid=$1
        if ($0 !~ /python/ || $0 !~ /http\.server/) next
        for (i = 2; i <= NF; i++) {
          if ($i == "http.server" && (i + 1) <= NF && $(i + 1) == port) {
            print pid
            break
          }
        }
      }
    '
  )"
  for p in ${gallery_pids}; do
    kill "${p}" 2>/dev/null || true
  done
  sleep 1

  : > "${gallery_log}"
  setsid "${python_bin}" -u -m http.server "${gallery_port}" \
    --directory "${gallery_dir}" > "${gallery_log}" 2>&1 < /dev/null &
  local new_gpid=$!
  sleep 1

  if ! kill -0 "${new_gpid}" 2>/dev/null; then
    log "Gallery server failed to stay alive (pid=${new_gpid})."
    log "  log=${gallery_log}"
    tail -n 30 "${gallery_log}" || true
    return 1
  fi

  if command -v curl >/dev/null 2>&1; then
    local code
    code="$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${gallery_port}/gallery.html" || true)"
    if [[ "${code}" != "200" ]]; then
      log "Gallery probe returned ${code} (expect 200): http://127.0.0.1:${gallery_port}/gallery.html"
    fi
  fi

  log "Gallery server started on port ${gallery_port} (dir=${gallery_dir}, pid=${new_gpid})"
  log "  log=${gallery_log}"
  return 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id-classifieds)
      RUN_ID_CLASSIFIEDS="${2:-}"
      shift 2
      ;;
    --run-id-reddit)
      RUN_ID_REDDIT="${2:-}"
      shift 2
      ;;
    --run-id-shopping)
      RUN_ID_SHOPPING="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

infer_run_ids_from_live_queue() {
  local qpid
  qpid="$(ps -ef | awk '/queue_b1_serial.sh/ && !/awk/ {print $2; exit}')"
  if [[ -z "${qpid}" ]] || [[ ! -r "/proc/${qpid}/environ" ]]; then
    return 1
  fi
  local line
  while IFS= read -r line; do
    case "${line}" in
      RUN_ID_CLASSIFIEDS=*) [[ -z "${RUN_ID_CLASSIFIEDS}" ]] && RUN_ID_CLASSIFIEDS="${line#RUN_ID_CLASSIFIEDS=}" ;;
      RUN_ID_REDDIT=*)      [[ -z "${RUN_ID_REDDIT}" ]] && RUN_ID_REDDIT="${line#RUN_ID_REDDIT=}" ;;
      RUN_ID_SHOPPING=*)    [[ -z "${RUN_ID_SHOPPING}" ]] && RUN_ID_SHOPPING="${line#RUN_ID_SHOPPING=}" ;;
    esac
  done < <(tr '\0' '\n' < "/proc/${qpid}/environ")
}

infer_run_ids_from_latest_meta() {
  local latest_meta
  latest_meta="$(ls -1t logs/queue_b1_serial_*.meta.txt 2>/dev/null | head -n 1 || true)"
  if [[ -z "${latest_meta}" ]] || [[ ! -f "${latest_meta}" ]]; then
    return 1
  fi
  [[ -z "${RUN_ID_CLASSIFIEDS}" ]] && RUN_ID_CLASSIFIEDS="$(awk -F= '/^RUN_ID_CLASSIFIEDS=/{print $2; exit}' "${latest_meta}" || true)"
  [[ -z "${RUN_ID_REDDIT}" ]]      && RUN_ID_REDDIT="$(awk -F= '/^RUN_ID_REDDIT=/{print $2; exit}' "${latest_meta}" || true)"
  [[ -z "${RUN_ID_SHOPPING}" ]]    && RUN_ID_SHOPPING="$(awk -F= '/^RUN_ID_SHOPPING=/{print $2; exit}' "${latest_meta}" || true)"
}

infer_run_ids_from_queue_defaults() {
  local queue_script="scripts/dgx/queue_b1_serial.sh"
  [[ -z "${RUN_ID_CLASSIFIEDS}" ]] && RUN_ID_CLASSIFIEDS="$(awk -F'"' '/^RUN_ID_CLASSIFIEDS=/{print $2; exit}' "${queue_script}" || true)"
  [[ -z "${RUN_ID_REDDIT}" ]]      && RUN_ID_REDDIT="$(awk -F'"' '/^RUN_ID_REDDIT=/{print $2; exit}' "${queue_script}" || true)"
  [[ -z "${RUN_ID_SHOPPING}" ]]    && RUN_ID_SHOPPING="$(awk -F'"' '/^RUN_ID_SHOPPING=/{print $2; exit}' "${queue_script}" || true)"
}

infer_run_ids_from_live_queue || true
infer_run_ids_from_latest_meta || true
infer_run_ids_from_queue_defaults || true

if [[ -z "${RUN_ID_CLASSIFIEDS}" || -z "${RUN_ID_REDDIT}" || -z "${RUN_ID_SHOPPING}" ]]; then
  echo "Failed to infer run_ids. Please pass --run-id-* explicitly." >&2
  exit 1
fi

log "Using run_ids:"
log "  classifieds=${RUN_ID_CLASSIFIEDS}"
log "  reddit=${RUN_ID_REDDIT}"
log "  shopping=${RUN_ID_SHOPPING}"

# Stop existing queue + experiment runners + sidecars + watchdogs.
q_pids="$(ps -eo pid=,args= | awk '/bash scripts\/dgx\/queue_b1_serial.sh/ && !/awk/ {print $1}')"
r_pids="$(ps -eo pid=,args= | awk '/scripts\/run_experiment.py/ && !/awk/ {print $1}')"
s_pids="$(ps -eo pid=,args= | awk '/scripts\/glm_diagnosis_sidecar.py/ && !/awk/ {print $1}')"
w_pids="$(ps -eo pid=,args= | awk '/scripts\/experiment_watchdog.py/ && !/awk/ {print $1}')"
g_pids="$(ps -eo pid=,args= | awk '/python.*http\.server/ && !/awk/ {print $1}')"

if [[ -n "${q_pids}${r_pids}${s_pids}${w_pids}${g_pids}" ]]; then
  log "Stopping existing queue/runner/sidecar/watchdog/gallery processes..."
  for p in ${q_pids} ${r_pids} ${s_pids} ${w_pids} ${g_pids}; do
    kill "${p}" 2>/dev/null || true
  done
  sleep 2
  for p in ${q_pids} ${r_pids} ${s_pids} ${w_pids} ${g_pids}; do
    if kill -0 "${p}" 2>/dev/null; then
      kill -9 "${p}" 2>/dev/null || true
    fi
  done
fi

OPENAI_KEY="${OPENAI_API_KEY:-}"
if [[ -z "${OPENAI_KEY}" && -f ".auth/openai_key" ]]; then
  OPENAI_KEY="$(tr -d '\r\n' < .auth/openai_key)"
fi
if [[ -z "${OPENAI_KEY}" ]]; then
  echo "Missing OpenAI key. Set OPENAI_API_KEY or create .auth/openai_key." >&2
  exit 1
fi

ts="$(date +%Y%m%d_%H%M%S)"
queue_log="logs/queue_b1_serial_${ts}.log"
queue_meta="logs/queue_b1_serial_${ts}.meta.txt"

# Start gallery HTTP server at phase-level so cross-site ../reddit paths work.
GALLERY_PORT="${GALLERY_PORT:-8765}"
gallery_dir="${RESULTS_BASE}"
if [[ -d "${gallery_dir}" ]]; then
  if ! restart_gallery_server "${gallery_dir}" "${GALLERY_PORT}"; then
    log "Gallery restart failed, queue will continue without gallery."
  fi
else
  log "Gallery dir not found (${gallery_dir}), skipping gallery server."
fi

log "Starting queue with setsid (sidecar=off, watchdog+digest=on)..."
setsid env \
  RUN_ID_CLASSIFIEDS="${RUN_ID_CLASSIFIEDS}" \
  RUN_ID_REDDIT="${RUN_ID_REDDIT}" \
  RUN_ID_SHOPPING="${RUN_ID_SHOPPING}" \
  OPENAI_API_KEY="${OPENAI_KEY}" \
  LIVE_REASON_WATCH_ENABLE=0 \
  WATCHDOG_ENABLE=1 \
  NTFY_MINIMAL_MODE=0 \
  bash scripts/dgx/queue_b1_serial.sh > "${queue_log}" 2>&1 < /dev/null &
new_qpid=$!

ln -sfn "$(basename "${queue_log}")" logs/latest_queue_b1_serial.log
cat > "${queue_meta}" <<EOF
started_at=$(date '+%Y-%m-%d %H:%M:%S %Z')
pid=${new_qpid}
log=${queue_log}
RUN_ID_CLASSIFIEDS=${RUN_ID_CLASSIFIEDS}
RUN_ID_REDDIT=${RUN_ID_REDDIT}
RUN_ID_SHOPPING=${RUN_ID_SHOPPING}
EOF

sleep 3
if ! kill -0 "${new_qpid}" 2>/dev/null; then
  echo "Queue failed to stay alive. Check log: ${queue_log}" >&2
  tail -n 60 "${queue_log}" || true
  exit 1
fi

log "Queue restarted successfully."
log "  pid=${new_qpid}"
log "  log=${queue_log}"
log "  meta=${queue_meta}"
log "Recent queue log:"
tail -n 20 "${queue_log}" || true
