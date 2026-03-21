#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

STRICT_PORTS="${STRICT_PORTS:-1}"
EXIT_CODE=0

print_check() {
  local level="$1"
  local msg="$2"
  echo "[${level}] ${msg}"
}

fail() {
  print_check "FAIL" "$1"
  EXIT_CODE=1
}

warn() {
  print_check "WARN" "$1"
}

pass() {
  print_check "PASS" "$1"
}

resolve_python() {
  if [[ -x "${PROJECT_DIR}/.venv/bin/python" ]]; then
    echo "${PROJECT_DIR}/.venv/bin/python"
    return
  fi
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return
  fi
  if command -v python >/dev/null 2>&1; then
    command -v python
    return
  fi
  echo ""
}

check_python() {
  PYTHON_BIN="$(resolve_python)"
  if [[ -z "${PYTHON_BIN}" ]]; then
    fail "No Python interpreter found (.venv/bin/python, python3, python)."
    return
  fi

  if "${PYTHON_BIN}" --version >/dev/null 2>&1; then
    pass "Python available: ${PYTHON_BIN} ($(${PYTHON_BIN} --version 2>&1))"
  else
    fail "Python interpreter is not executable: ${PYTHON_BIN}"
  fi
}

check_env_vars() {
  local required=(DATASET SHOPPING REDDIT WIKIPEDIA CLASSIFIEDS HOMEPAGE CLASSIFIEDS_RESET_TOKEN)
  local missing=()
  for key in "${required[@]}"; do
    if [[ -z "${!key:-}" ]]; then
      missing+=("${key}")
    fi
  done

  if (( ${#missing[@]} > 0 )); then
    fail "Missing VWA environment variables: ${missing[*]}"
  else
    pass "VWA environment variables are set."
  fi
}

check_docker() {
  if ! command -v docker >/dev/null 2>&1; then
    fail "docker command not found"
    return
  fi

  if docker info >/dev/null 2>&1; then
    pass "Docker daemon is reachable"
  else
    fail "Docker daemon is not reachable"
  fi
}

check_site_ports() {
  local endpoints=(
    "shopping:http://localhost:7770"
    "reddit:http://localhost:9999"
    "wikipedia:http://localhost:8888"
    "classifieds:http://localhost:9980"
    "homepage:http://localhost:4399"
  )

  for item in "${endpoints[@]}"; do
    local name="${item%%:*}"
    local url="${item#*:}"

    if curl -fsS --max-time 3 "${url}" >/dev/null 2>&1; then
      pass "${name} endpoint reachable (${url})"
    else
      if [[ "${STRICT_PORTS}" == "1" ]]; then
        fail "${name} endpoint not reachable (${url})"
      else
        warn "${name} endpoint not reachable (${url})"
      fi
    fi
  done
}

check_python_modules() {
  if [[ -z "${PYTHON_BIN:-}" ]]; then
    fail "Skipping Python module checks because no Python was found"
    return
  fi

  local py_output
  if ! py_output="$(${PYTHON_BIN} - <<'PY' 2>&1
import importlib
mods = ["playwright", "browser_env", "p79"]
missing = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception:
        missing.append(m)
if missing:
    raise SystemExit("missing:" + ",".join(missing))
print("ok")
PY
)"; then
    fail "Python module check failed (${py_output})"
    return
  fi

  pass "Python modules available: playwright, browser_env, p79"
}

main() {
  echo "=== P79 Preflight v2 ==="
  echo "project_dir=${PROJECT_DIR}"

  check_python
  check_env_vars
  check_docker
  check_site_ports
  check_python_modules

  if (( EXIT_CODE == 0 )); then
    echo "Preflight completed successfully."
  else
    echo "Preflight found issues."
  fi

  exit ${EXIT_CODE}
}

main "$@"
