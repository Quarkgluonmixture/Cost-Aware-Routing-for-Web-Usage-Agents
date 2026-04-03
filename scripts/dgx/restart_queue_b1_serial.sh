#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_DIR}"

RESULTS_BASE="${REPO_DIR}/results/visualwebarena/phase1"

CLEAN=0
RUN_ID_CLASSIFIEDS="${RUN_ID_CLASSIFIEDS:-}"
RUN_ID_REDDIT="${RUN_ID_REDDIT:-}"
RUN_ID_SHOPPING="${RUN_ID_SHOPPING:-}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/dgx/restart_queue_b1_serial.sh [options]

Options:
  --clean                     Delete previous outputs/logs for selected run_ids before restart.
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

while [[ $# -gt 0 ]]; do
  case "$1" in
    --clean)
      CLEAN=1
      shift
      ;;
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

# Stop existing queue + experiment runners.
q_pids="$(ps -eo pid=,args= | awk '/bash scripts\/dgx\/queue_b1_serial.sh/ && !/awk/ {print $1}')"
r_pids="$(ps -eo pid=,args= | awk '/scripts\/run_experiment.py/ && !/awk/ {print $1}')"

if [[ -n "${q_pids}${r_pids}" ]]; then
  log "Stopping existing queue/runner processes..."
  for p in ${q_pids} ${r_pids}; do
    kill "${p}" 2>/dev/null || true
  done
  sleep 2
  for p in ${q_pids} ${r_pids}; do
    if kill -0 "${p}" 2>/dev/null; then
      kill -9 "${p}" 2>/dev/null || true
    fi
  done
fi

if [[ "${CLEAN}" -eq 1 ]]; then
  log "--clean enabled: deleting previous outputs/logs for selected run_ids..."
  rm -rf \
    "${RESULTS_BASE}/${RUN_ID_CLASSIFIEDS}" \
    "${RESULTS_BASE}/${RUN_ID_REDDIT}" \
    "${RESULTS_BASE}/${RUN_ID_SHOPPING}"

  rm -f \
    "logs/B1_baseline_qwen3vl4b_classifieds_${RUN_ID_CLASSIFIEDS}.log" \
    "logs/B1_baseline_qwen3vl4b_reddit_${RUN_ID_REDDIT}.log" \
    "logs/B1_baseline_qwen3vl4b_shopping_${RUN_ID_SHOPPING}.log"

  rm -f logs/queue_b1_serial_*.log logs/queue_b1_serial_*.meta.txt
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

log "Starting queue with setsid..."
setsid env \
  RUN_ID_CLASSIFIEDS="${RUN_ID_CLASSIFIEDS}" \
  RUN_ID_REDDIT="${RUN_ID_REDDIT}" \
  RUN_ID_SHOPPING="${RUN_ID_SHOPPING}" \
  OPENAI_API_KEY="${OPENAI_KEY}" \
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
clean=${CLEAN}
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
