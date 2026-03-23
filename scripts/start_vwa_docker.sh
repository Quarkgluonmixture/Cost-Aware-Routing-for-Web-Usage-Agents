#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
VWA_DIR="${PROJECT_DIR}/external/visualwebarena"
ENV_DIR="${VWA_DIR}/environment_docker"

SITES="all"
AUTO_SETUP=0
HOSTNAME_VALUE="localhost"
CHECK_ONLY=0

usage() {
  cat <<USAGE
Usage: bash scripts/start_vwa_docker.sh [options]

Options:
  --sites <list>       Comma-separated sites: all|shopping|reddit|wikipedia|classifieds|homepage
  --auto-setup         Run scripts/setup_vwa.sh automatically when required assets are missing
  --hostname <value>   Hostname used to patch VWA templates (default: localhost)
  --check-only         Only validate prerequisites, do not start services
  -h, --help           Show this help

Examples:
  bash scripts/start_vwa_docker.sh --sites all
  bash scripts/start_vwa_docker.sh --sites shopping,reddit --auto-setup
  bash scripts/start_vwa_docker.sh --check-only
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sites)
      SITES="${2:-all}"
      shift 2
      ;;
    --auto-setup)
      AUTO_SETUP=1
      shift
      ;;
    --hostname)
      HOSTNAME_VALUE="${2:-localhost}"
      shift 2
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

contains_site() {
  local needle="$1"
  if [[ "${SITES}" == "all" ]]; then
    return 0
  fi
  IFS=',' read -r -a selected <<< "${SITES}"
  for item in "${selected[@]}"; do
    if [[ "$(echo "${item}" | xargs)" == "${needle}" ]]; then
      return 0
    fi
  done
  return 1
}

check_prerequisites() {
  local missing=0

  if ! command -v docker >/dev/null 2>&1; then
    echo "[MISSING] docker command not found" >&2
    missing=1
  fi

  if [[ ! -d "${VWA_DIR}" ]] || [[ -z "$(ls -A "${VWA_DIR}" 2>/dev/null || true)" ]]; then
    echo "[MISSING] external/visualwebarena repository" >&2
    missing=1
  fi

  if contains_site "shopping"; then
    if ! docker images --format '{{.Repository}}:{{.Tag}}' | grep -q '^shopping_final_0712:latest$'; then
      echo "[MISSING] shopping_final_0712:latest image" >&2
      missing=1
    fi
  fi

  if contains_site "reddit"; then
    if ! docker images --format '{{.Repository}}:{{.Tag}}' | grep -q '^postmill-populated-exposed-withimg:latest$'; then
      echo "[MISSING] postmill-populated-exposed-withimg:latest image" >&2
      missing=1
    fi
  fi

  if contains_site "wikipedia"; then
    if [[ ! -f "${ENV_DIR}/data/wikipedia_en_all_maxi_2022-05.zim" ]]; then
      echo "[MISSING] wikipedia ZIM file" >&2
      missing=1
    fi
  fi

  if contains_site "classifieds"; then
    if [[ ! -d "${ENV_DIR}/classifieds_docker_compose" ]]; then
      echo "[MISSING] classifieds_docker_compose directory" >&2
      missing=1
    fi
  fi

  if (( missing == 1 )); then
    if (( AUTO_SETUP == 1 )); then
      echo "Missing prerequisites detected, running setup..."
      local setup_target="all"
      if [[ "${SITES}" != "all" ]]; then
        # setup_vwa expects dataset names; homepage has no dataset package
        setup_target="${SITES//homepage/}" 
        setup_target="${setup_target//,,/,}"
        setup_target="${setup_target#,}"
        setup_target="${setup_target%,}"
        [[ -z "${setup_target}" ]] && setup_target="all"
      fi
      bash "${PROJECT_DIR}/scripts/setup_vwa.sh" --target-dataset "${setup_target}"
      return 0
    fi
    echo "Prerequisite check failed. Re-run with --auto-setup or run scripts/setup_vwa.sh first." >&2
    return 1
  fi

  echo "Prerequisite check passed."
  return 0
}

start_shopping() {
  echo "[START] shopping (http://${HOSTNAME_VALUE}:7770)"
  if docker ps --format '{{.Names}}' | grep -q '^shopping$'; then
    echo "shopping already running"
    return
  fi
  docker start shopping >/dev/null 2>&1 || docker run --name shopping -p 7770:80 -d shopping_final_0712 >/dev/null
  sleep 10
  docker exec shopping /var/www/magento2/bin/magento setup:store-config:set --base-url="http://${HOSTNAME_VALUE}:7770" >/dev/null 2>&1 || true
  docker exec shopping mysql -u magentouser -pMyPassword magentodb -e "UPDATE core_config_data SET value='http://${HOSTNAME_VALUE}:7770/' WHERE path='web/secure/base_url';" >/dev/null 2>&1 || true
  docker exec shopping /var/www/magento2/bin/magento cache:flush >/dev/null 2>&1 || true
}

start_reddit() {
  echo "[START] reddit/forum (http://${HOSTNAME_VALUE}:9999)"
  if docker ps --format '{{.Names}}' | grep -q '^forum$'; then
    echo "forum already running"
    return
  fi
  docker start forum >/dev/null 2>&1 || docker run --name forum -p 9999:80 -d postmill-populated-exposed-withimg >/dev/null
}

start_wikipedia() {
  echo "[START] wikipedia (http://${HOSTNAME_VALUE}:8888)"
  if docker ps --format '{{.Names}}' | grep -q '^wikipedia$'; then
    echo "wikipedia already running"
    return
  fi
  docker run -d --name wikipedia --volume="${ENV_DIR}/data/:/data" -p 8888:80 ghcr.io/kiwix/kiwix-serve:3.3.0 wikipedia_en_all_maxi_2022-05.zim >/dev/null
}

start_classifieds() {
  echo "[START] classifieds (http://${HOSTNAME_VALUE}:9980)"
  if docker ps --format '{{.Names}}' | grep -q '^classifieds$'; then
    echo "classifieds already running"
    return
  fi

  local compose_dir="${ENV_DIR}/classifieds_docker_compose"
  if [[ ! -d "${compose_dir}" ]]; then
    echo "Classifieds compose directory missing: ${compose_dir}" >&2
    return 1
  fi

  sed -i "s|<your-server-hostname>|${HOSTNAME_VALUE}|g" "${compose_dir}/docker-compose.yml"
  (cd "${compose_dir}" && docker compose up --build -d)
  sleep 15
  docker exec classifieds_db mysql -u root -ppassword osclass -e 'source docker-entrypoint-initdb.d/osclass_craigslist.sql' >/dev/null 2>&1 || true
}

start_homepage() {
  echo "[START] homepage (http://${HOSTNAME_VALUE}:4399)"
  perl -pi -e "s|<your-server-hostname>|${HOSTNAME_VALUE}|g" "${ENV_DIR}/webarena-homepage/templates/index.html"

  if pgrep -f 'flask run.*4399' >/dev/null 2>&1; then
    echo "homepage already running"
    return
  fi
  (cd "${ENV_DIR}/webarena-homepage" && nohup flask run --host=0.0.0.0 --port=4399 >/tmp/vwa_homepage.log 2>&1 &)
}

main() {
  echo "=== VWA Docker Startup ==="
  echo "project_dir=${PROJECT_DIR}"
  echo "sites=${SITES}"
  echo "hostname=${HOSTNAME_VALUE}"

  check_prerequisites

  if (( CHECK_ONLY == 1 )); then
    echo "Check-only mode complete."
    exit 0
  fi

  contains_site "shopping" && start_shopping
  contains_site "reddit" && start_reddit
  contains_site "wikipedia" && start_wikipedia
  contains_site "classifieds" && start_classifieds
  contains_site "homepage" && start_homepage

  echo ""
  echo "=== Running containers ==="
  docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' | grep -E 'shopping|forum|wikipedia|classifieds|db|redis|chrome|NAMES' || true
}

main "$@"
