#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VWA_DIR="${PROJECT_DIR}/external/visualwebarena"
ENV_DIR="${VWA_DIR}/environment_docker"

SITES="all"
AUTO_SETUP=0
HOSTNAME_VALUE="localhost"
CHECK_ONLY=0

usage() {
  cat <<USAGE
Usage: bash scripts/vwa/start_vwa_docker.sh [options]

Options:
  --sites <list>       Comma-separated sites: all|shopping|shopping_admin|reddit|wikipedia|classifieds|homepage
                       (shopping_admin shares the shopping container, adds host port 7780 → same Magento)
  --auto-setup         Run scripts/vwa/setup_vwa.sh automatically when required assets are missing
  --hostname <value>   Hostname used to patch VWA templates (default: localhost)
  --check-only         Only validate prerequisites, do not start services
  -h, --help           Show this help

Examples:
  bash scripts/vwa/start_vwa_docker.sh --sites all
  bash scripts/vwa/start_vwa_docker.sh --sites shopping,reddit --auto-setup
  bash scripts/vwa/start_vwa_docker.sh --check-only
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

find_running_container() {
  local name
  for name in "$@"; do
    if docker ps --format '{{.Names}}' | grep -q "^${name}$"; then
      echo "${name}"
      return 0
    fi
  done
  return 1
}

find_existing_container() {
  local name
  for name in "$@"; do
    if docker ps -a --format '{{.Names}}' | grep -q "^${name}$"; then
      echo "${name}"
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

  if contains_site "shopping" || contains_site "shopping_admin"; then
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
      bash "${PROJECT_DIR}/scripts/vwa/setup_vwa.sh" --target-dataset "${setup_target}"
      return 0
    fi
    echo "Prerequisite check failed. Re-run with --auto-setup or run scripts/vwa/setup_vwa.sh first." >&2
    return 1
  fi

  echo "Prerequisite check passed."
  return 0
}

start_shopping() {
  local want_admin=0
  contains_site "shopping_admin" && want_admin=1
  if (( want_admin == 1 )); then
    echo "[START] shopping (http://${HOSTNAME_VALUE}:7770) + shopping_admin (http://${HOSTNAME_VALUE}:7780 → same container)"
  else
    echo "[START] shopping (http://${HOSTNAME_VALUE}:7770)"
  fi
  local container_name=""
  container_name="$(find_running_container vwa-shopping shopping || true)"
  if [[ -n "${container_name}" ]]; then
    echo "${container_name} already running; reconfiguring base URL"
  else
    container_name="$(find_existing_container vwa-shopping shopping || true)"
    if [[ -n "${container_name}" ]]; then
      docker start "${container_name}" >/dev/null 2>&1
    else
      container_name="vwa-shopping"
      local port_args="-p 7770:80"
      (( want_admin == 1 )) && port_args="${port_args} -p 7780:80"
      docker run --name "${container_name}" ${port_args} -d shopping_final_0712 >/dev/null
    fi
    sleep 10
  fi
  docker exec "${container_name}" /var/www/magento2/bin/magento setup:store-config:set --base-url="http://${HOSTNAME_VALUE}:7770" >/dev/null 2>&1 || true
  docker exec "${container_name}" mysql -u magentouser -pMyPassword magentodb -e "UPDATE core_config_data SET value='http://${HOSTNAME_VALUE}:7770/' WHERE path IN ('web/unsecure/base_url', 'web/secure/base_url');" >/dev/null 2>&1 || true
  docker exec "${container_name}" /var/www/magento2/bin/magento cache:flush >/dev/null 2>&1 || true
}

start_reddit() {
  echo "[START] reddit/forum (http://${HOSTNAME_VALUE}:9999)"
  local container_name=""
  container_name="$(find_running_container vwa-reddit forum || true)"
  if [[ -n "${container_name}" ]]; then
    echo "${container_name} already running"
  else
    container_name="$(find_existing_container vwa-reddit forum || true)"
    if [[ -n "${container_name}" ]]; then
      docker start "${container_name}" >/dev/null 2>&1
    else
      docker run --name vwa-reddit -p 9999:80 -d postmill-populated-exposed-withimg >/dev/null
    fi
  fi
}

start_wikipedia() {
  echo "[START] wikipedia (http://${HOSTNAME_VALUE}:8888)"
  local container_name=""
  container_name="$(find_running_container vwa-wikipedia wikipedia || true)"
  if [[ -n "${container_name}" ]]; then
    echo "${container_name} already running"
  else
    container_name="$(find_existing_container vwa-wikipedia wikipedia || true)"
    if [[ -n "${container_name}" ]]; then
      docker start "${container_name}" >/dev/null 2>&1
    else
      docker run -d --name vwa-wikipedia --volume="${ENV_DIR}/data/:/data" -p 8888:80 ghcr.io/kiwix/kiwix-serve:3.3.0 wikipedia_en_all_maxi_2022-05.zim >/dev/null
    fi
  fi
}

start_classifieds() {
  echo "[START] classifieds (http://${HOSTNAME_VALUE}:9980)"
  local classifieds_running=0
  if docker ps --format '{{.Names}}' | grep -q '^classifieds$'; then
    classifieds_running=1
    echo "classifieds already running; reconfiguring compose hostname"
  fi

  local compose_dir="${ENV_DIR}/classifieds_docker_compose"
  if [[ ! -d "${compose_dir}" ]]; then
    echo "Classifieds compose directory missing: ${compose_dir}" >&2
    return 1
  fi

  sed -i "s|<your-server-hostname>|${HOSTNAME_VALUE}|g" "${compose_dir}/docker-compose.yml"
  sed -i -E "s|CLASSIFIEDS=http://[^:]+:9980/|CLASSIFIEDS=http://${HOSTNAME_VALUE}:9980/|g" "${compose_dir}/docker-compose.yml"
  (cd "${compose_dir}" && docker compose up --build -d)
  if (( classifieds_running == 0 )); then
    sleep 15
    docker exec classifieds_db mysql -u root -ppassword osclass -e 'source docker-entrypoint-initdb.d/osclass_craigslist.sql' >/dev/null 2>&1 || true
  fi
}

start_homepage() {
  echo "[START] homepage (http://${HOSTNAME_VALUE}:4399)"
  perl -pi -e "s|<your-server-hostname>|${HOSTNAME_VALUE}|g" "${ENV_DIR}/webarena-homepage/templates/index.html"
  perl -pi -e "s|localhost:9980|${HOSTNAME_VALUE}:9980|g; s|localhost:7770|${HOSTNAME_VALUE}:7770|g; s|localhost:9999|${HOSTNAME_VALUE}:9999|g; s|localhost:8888|${HOSTNAME_VALUE}:8888|g" "${ENV_DIR}/webarena-homepage/templates/index.html"

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

  { contains_site "shopping" || contains_site "shopping_admin"; } && start_shopping
  contains_site "reddit" && start_reddit
  contains_site "wikipedia" && start_wikipedia
  contains_site "classifieds" && start_classifieds
  contains_site "homepage" && start_homepage

  echo ""
  echo "=== Running containers ==="
  docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' | grep -E 'shopping|forum|wikipedia|classifieds|db|redis|chrome|NAMES' || true
}

main "$@"
