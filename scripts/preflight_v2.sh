#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

STRICT_PORTS="${STRICT_PORTS:-1}"
SITE_MODE="${SITE_MODE:-auto}"
CHECK_DOCKER="${CHECK_DOCKER:-auto}"
REQUIRE_CUDA="${REQUIRE_CUDA:-0}"
ALLOW_MISSING_EVALUATOR="${ALLOW_MISSING_EVALUATOR:-0}"
EXIT_CODE=0

usage() {
  cat <<USAGE
Usage: bash scripts/preflight_v2.sh [options]

Options:
  --strict-ports       Treat unreachable site ports as FAIL (default)
  --no-strict-ports    Treat unreachable site ports as WARN
  --site-mode <mode>   auto|local|remote (default: auto)
  --remote-sites       Equivalent to --site-mode remote
  --local-sites        Equivalent to --site-mode local
  --skip-docker        Skip docker daemon check
  --check-docker       Force docker daemon check
  --require-cuda       Fail if torch CUDA runtime is unavailable
  --allow-missing-evaluator
                      Downgrade evaluator import failures to WARN
  -h, --help           Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --strict-ports)
      STRICT_PORTS=1
      shift
      ;;
    --no-strict-ports)
      STRICT_PORTS=0
      shift
      ;;
    --site-mode)
      SITE_MODE="${2:-auto}"
      shift 2
      ;;
    --remote-sites)
      SITE_MODE="remote"
      shift
      ;;
    --local-sites)
      SITE_MODE="local"
      shift
      ;;
    --skip-docker)
      CHECK_DOCKER="never"
      shift
      ;;
    --check-docker)
      CHECK_DOCKER="always"
      shift
      ;;
    --require-cuda)
      REQUIRE_CUDA=1
      shift
      ;;
    --allow-missing-evaluator)
      ALLOW_MISSING_EVALUATOR=1
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

if [[ "${SITE_MODE}" != "auto" ]] && [[ "${SITE_MODE}" != "local" ]] && [[ "${SITE_MODE}" != "remote" ]]; then
  echo "Invalid --site-mode value: ${SITE_MODE} (expected: auto|local|remote)" >&2
  exit 2
fi

if [[ "${CHECK_DOCKER}" != "auto" ]] && [[ "${CHECK_DOCKER}" != "always" ]] && [[ "${CHECK_DOCKER}" != "never" ]]; then
  echo "Invalid docker check mode: ${CHECK_DOCKER} (expected: auto|always|never)" >&2
  exit 2
fi

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
  # Choose required vars based on DATASET
  local required
  if [[ "${DATASET:-visualwebarena}" == "webarena" ]]; then
    required=(DATASET SHOPPING SHOPPING_ADMIN REDDIT HOMEPAGE)
  else
    required=(DATASET SHOPPING REDDIT WIKIPEDIA CLASSIFIEDS HOMEPAGE CLASSIFIEDS_RESET_TOKEN)
  fi
  local missing=()
  for key in "${required[@]}"; do
    if [[ -z "${!key:-}" ]]; then
      missing+=("${key}")
    fi
  done

  local label="${DATASET:-visualwebarena}"
  if (( ${#missing[@]} > 0 )); then
    fail "Missing ${label} environment variables: ${missing[*]}"
  else
    pass "${label} environment variables are set."
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

extract_url_host() {
  local url="$1"
  local no_scheme="${url#*://}"
  local host_port="${no_scheme%%/*}"
  local host
  if [[ "${host_port}" == \[*\]* ]]; then
    host="${host_port#\[}"
    host="${host%%\]*}"
  else
    host="${host_port%%:*}"
  fi
  echo "${host}"
}

is_local_host() {
  local host="$1"
  [[ "${host}" == "localhost" || "${host}" == "127.0.0.1" || "${host}" == "::1" ]]
}

resolve_site_mode() {
  if [[ "${SITE_MODE}" != "auto" ]]; then
    RESOLVED_SITE_MODE="${SITE_MODE}"
    return
  fi

  local endpoints=("${SHOPPING:-}" "${REDDIT:-}" "${WIKIPEDIA:-}" "${CLASSIFIEDS:-}" "${HOMEPAGE:-}" "${SHOPPING_ADMIN:-}")
  local all_local=1
  local url
  for url in "${endpoints[@]}"; do
    if [[ -z "${url}" ]]; then
      continue
    fi
    if ! is_local_host "$(extract_url_host "${url}")"; then
      all_local=0
      break
    fi
  done

  if (( all_local == 1 )); then
    RESOLVED_SITE_MODE="local"
  else
    RESOLVED_SITE_MODE="remote"
  fi
}

check_site_endpoints() {
  local endpoints=(
    "shopping:${SHOPPING:-}"
    "reddit:${REDDIT:-}"
    "wikipedia:${WIKIPEDIA:-}"
    "classifieds:${CLASSIFIEDS:-}"
    "homepage:${HOMEPAGE:-}"
  )
  # Add shopping_admin if set
  if [[ -n "${SHOPPING_ADMIN:-}" ]]; then
    endpoints+=("shopping_admin:${SHOPPING_ADMIN}")
  fi

  for item in "${endpoints[@]}"; do
    local name="${item%%:*}"
    local url="${item#*:}"

    if [[ -z "${url}" ]]; then
      fail "${name} endpoint is empty"
      continue
    fi

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

check_playwright_browser() {
  if [[ -z "${PYTHON_BIN:-}" ]]; then
    fail "Skipping Playwright browser check because no Python was found"
    return
  fi

  local py_output
  if ! py_output="$(${PYTHON_BIN} - <<'PY' 2>&1
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    browser.close()
print("ok")
PY
)"; then
    fail "Playwright Chromium runtime check failed (${py_output})"
    return
  fi

  pass "Playwright Chromium runtime is available"
}

check_torch_cuda() {
  if [[ -z "${PYTHON_BIN:-}" ]]; then
    fail "Skipping torch CUDA check because no Python was found"
    return
  fi

  local py_output
  if ! py_output="$(${PYTHON_BIN} - <<'PY' 2>&1
import importlib

try:
    torch = importlib.import_module("torch")
except Exception as exc:
    print(f"import_error:{exc}")
    raise SystemExit(2)

cuda_built = bool(getattr(torch.backends.cuda, "is_built", lambda: False)())
cuda_available = bool(torch.cuda.is_available())
cuda_version = getattr(torch.version, "cuda", None)
print(f"built={cuda_built};available={cuda_available};version={cuda_version};torch={torch.__version__}")
if not cuda_available:
    raise SystemExit(3)
PY
)"; then
    if [[ "${REQUIRE_CUDA}" == "1" ]]; then
      fail "Torch CUDA runtime check failed (${py_output})"
    else
      warn "Torch CUDA runtime unavailable (${py_output})"
    fi
    return
  fi

  pass "Torch CUDA runtime is available (${py_output})"
}

check_vwa_evaluator_import() {
  if [[ -z "${PYTHON_BIN:-}" ]]; then
    fail "Skipping evaluator import check because no Python was found"
    return
  fi

  local py_output
  if ! py_output="$(${PYTHON_BIN} - <<'PY' 2>&1
import os
import sys

cwd = os.getcwd()
candidate = os.path.join(cwd, "external", "visualwebarena")
if os.path.isdir(candidate):
    sys.path.append(candidate)

# VisualWebArena provider imports may read OPENAI_API_KEY at module import time.
# Use a harmless placeholder for import-time checks.
os.environ.setdefault("OPENAI_API_KEY", "DUMMY_P79_PRECHECK")

from evaluation_harness import evaluator_router  # noqa: F401
print("ok")
PY
)"; then
    if [[ "${ALLOW_MISSING_EVALUATOR}" == "1" ]]; then
      warn "VWA evaluator import failed (${py_output})"
    else
      fail "VWA evaluator import failed (${py_output})"
    fi
    return
  fi

  pass "VWA evaluator import is available"
}

main() {
  echo "=== P79 Preflight v2 ==="
  echo "project_dir=${PROJECT_DIR}"

  check_python
  check_env_vars
  resolve_site_mode
  pass "Site mode resolved as: ${RESOLVED_SITE_MODE}"

  if [[ "${CHECK_DOCKER}" == "always" ]] || { [[ "${CHECK_DOCKER}" == "auto" ]] && [[ "${RESOLVED_SITE_MODE}" == "local" ]]; }; then
    check_docker
  else
    warn "Skipping docker daemon check (mode=${CHECK_DOCKER}, site_mode=${RESOLVED_SITE_MODE})"
  fi

  check_site_endpoints
  check_python_modules
  check_playwright_browser
  check_torch_cuda
  check_vwa_evaluator_import

  if (( EXIT_CODE == 0 )); then
    echo "Preflight completed successfully."
  else
    echo "Preflight found issues."
  fi

  exit ${EXIT_CODE}
}

main "$@"
