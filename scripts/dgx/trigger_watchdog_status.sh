#!/usr/bin/env bash
# trigger_watchdog_status.sh — ask running experiment_watchdog to run one immediate status cycle.
#
# Mechanism: send SIGUSR1 to watchdog process(es). The watchdog handles SIGUSR1
# and performs an on-demand status+pipeline run without waiting for report window.
#
# Usage:
#   bash scripts/dgx/trigger_watchdog_status.sh
#   bash scripts/dgx/trigger_watchdog_status.sh --run-dir /abs/path/to/run_dir
#   bash scripts/dgx/trigger_watchdog_status.sh --dry-run
set -euo pipefail

RUN_DIR_FILTER=""
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/dgx/trigger_watchdog_status.sh [options]

Options:
  --run-dir PATH   Trigger only watchdog whose cmdline contains this run-dir.
  --dry-run        Show matched watchdog processes without signaling.
  -h, --help       Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir)
      RUN_DIR_FILTER="${2:-}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
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

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

mapfile -t rows < <(
  ps -eo pid=,args= | awk '
    /experiment_watchdog\.py/ && !/awk/ {
      pid=$1
      $1=""
      sub(/^ /,"")
      print pid "\t" $0
    }
  '
)

if [[ ${#rows[@]} -eq 0 ]]; then
  log "No running experiment_watchdog.py process found."
  exit 0
fi

matched=0
for row in "${rows[@]}"; do
  pid="${row%%$'\t'*}"
  cmd="${row#*$'\t'}"
  if [[ -n "${RUN_DIR_FILTER}" && "${cmd}" != *"${RUN_DIR_FILTER}"* ]]; then
    continue
  fi
  matched=$((matched + 1))
  log "Matched watchdog pid=${pid}"
  log "  cmd=${cmd}"
  if [[ "${DRY_RUN}" -eq 0 ]]; then
    kill -USR1 "${pid}"
    log "  sent SIGUSR1"
  else
    log "  [dry-run] would send SIGUSR1"
  fi
done

if [[ "${matched}" -eq 0 ]]; then
  log "No watchdog matched filter: --run-dir ${RUN_DIR_FILTER}"
  exit 1
fi

if [[ "${DRY_RUN}" -eq 0 ]]; then
  log "Triggered ${matched} watchdog process(es)."
else
  log "[dry-run] matched ${matched} watchdog process(es)."
fi

