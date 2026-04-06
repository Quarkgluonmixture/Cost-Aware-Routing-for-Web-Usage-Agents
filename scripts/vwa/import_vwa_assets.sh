#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VWA_DIR="${PROJECT_DIR}/external/visualwebarena"
ENV_DIR="${VWA_DIR}/environment_docker"

IMPORTS_DIR="/home/jiaming/imports"
SITES="all"
HOSTNAME_VALUE="localhost"
RUN_START=1
RUN_PREFLIGHT=1
CHECK_ONLY=0

usage() {
  cat <<USAGE
Usage: bash scripts/vwa/import_vwa_assets.sh [options]

Import offline VWA assets from another machine:
  - shopping_final_0712.tar
  - postmill-populated-exposed-withimg.tar
  - wikipedia_en_all_maxi_2022-05.zim
  - classifieds_docker_compose.tar.gz

Options:
  --imports-dir <path>   Source directory for exported files (default: /home/jiaming/imports)
  --sites <list>         Comma-separated sites for startup (default: all)
  --hostname <value>     Hostname passed to start_vwa_docker.sh (default: localhost)
  --no-start             Only import assets; do not start services
  --no-preflight         Skip preflight check after startup
  --check-only           Validate file presence only; do not import/start
  -h, --help             Show this help

Example:
  bash scripts/vwa/import_vwa_assets.sh --imports-dir /home/jiaming/imports --sites all --hostname localhost
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --imports-dir)
      IMPORTS_DIR="${2:-/home/jiaming/imports}"
      shift 2
      ;;
    --sites)
      SITES="${2:-all}"
      shift 2
      ;;
    --hostname)
      HOSTNAME_VALUE="${2:-localhost}"
      shift 2
      ;;
    --no-start)
      RUN_START=0
      shift
      ;;
    --no-preflight)
      RUN_PREFLIGHT=0
      shift
      ;;
    --check-only)
      CHECK_ONLY=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

SHOPPING_TAR="${IMPORTS_DIR}/shopping_final_0712.tar"
REDDIT_TAR="${IMPORTS_DIR}/postmill-populated-exposed-withimg.tar"
WIKI_ZIM_SRC="${IMPORTS_DIR}/wikipedia_en_all_maxi_2022-05.zim"
CLASSIFIEDS_TAR="${IMPORTS_DIR}/classifieds_docker_compose.tar.gz"
WIKI_ZIM_DST="${ENV_DIR}/data/wikipedia_en_all_maxi_2022-05.zim"

check_file() {
  local file_path="$1"
  local label="$2"
  if [[ -f "${file_path}" ]]; then
    echo "[PASS] ${label}: ${file_path}"
    return 0
  fi
  echo "[FAIL] Missing ${label}: ${file_path}" >&2
  return 1
}

check_requirements() {
  local failed=0

  if ! command -v docker >/dev/null 2>&1; then
    echo "[FAIL] docker command not found" >&2
    failed=1
  fi
  if ! command -v tar >/dev/null 2>&1; then
    echo "[FAIL] tar command not found" >&2
    failed=1
  fi
  if [[ ! -d "${ENV_DIR}" ]]; then
    echo "[FAIL] VWA environment directory missing: ${ENV_DIR}" >&2
    failed=1
  fi

  check_file "${SHOPPING_TAR}" "shopping image tar" || failed=1
  check_file "${REDDIT_TAR}" "reddit image tar" || failed=1
  check_file "${WIKI_ZIM_SRC}" "wikipedia zim" || failed=1
  check_file "${CLASSIFIEDS_TAR}" "classifieds compose tar.gz" || failed=1

  if (( failed == 1 )); then
    return 1
  fi
  return 0
}

import_assets() {
  echo "== Importing docker images =="
  docker load -i "${SHOPPING_TAR}"
  docker load -i "${REDDIT_TAR}"

  echo "== Installing wikipedia ZIM =="
  mkdir -p "${ENV_DIR}/data"
  cp -f "${WIKI_ZIM_SRC}" "${WIKI_ZIM_DST}"

  echo "== Extracting classifieds compose =="
  tar -xzf "${CLASSIFIEDS_TAR}" -C "${ENV_DIR}"

  if [[ ! -d "${ENV_DIR}/classifieds_docker_compose" ]]; then
    echo "[FAIL] classifieds_docker_compose was not created under ${ENV_DIR}" >&2
    return 1
  fi
}

run_start_and_preflight() {
  if (( RUN_START == 1 )); then
    echo "== Starting VWA docker services =="
    bash "${SCRIPT_DIR}/start_vwa_docker.sh" --sites "${SITES}" --hostname "${HOSTNAME_VALUE}"
  fi

  if (( RUN_PREFLIGHT == 1 )); then
    echo "== Running preflight =="
    # shellcheck disable=SC1090
    source "${SCRIPT_DIR}/../vwa_env.sh"
    bash "${SCRIPT_DIR}/../preflight_v2.sh"
  fi
}

main() {
  echo "=== Import VWA Assets ==="
  echo "project_dir=${PROJECT_DIR}"
  echo "imports_dir=${IMPORTS_DIR}"
  echo "sites=${SITES}"
  echo "hostname=${HOSTNAME_VALUE}"

  check_requirements

  if (( CHECK_ONLY == 1 )); then
    echo "Check-only mode complete."
    exit 0
  fi

  import_assets
  run_start_and_preflight
  echo "Import flow complete."
}

main "$@"
