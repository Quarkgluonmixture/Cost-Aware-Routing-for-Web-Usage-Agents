#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

STRICT_PORTS="${STRICT_PORTS:-1}"
SITE_MODE="${SITE_MODE:-auto}"
CHECK_DOCKER="${CHECK_DOCKER:-auto}"
REQUIRE_CUDA="${REQUIRE_CUDA:-0}"
ALLOW_MISSING_EVALUATOR="${ALLOW_MISSING_EVALUATOR:-0}"
# B-793 (/stress A1.9 cold-start P1-9 root-cause fix, 2026-05-17): paper-grade
# evaluator probe — actually instantiates `VwaEvaluator(paper_grade=True)` to
# catch B-544 init-time fail-loud BEFORE a 36-condition batch launches. Pre-fix
# preflight only checked `evaluator_router` import; runtime init failure
# (OpenAI key flicker, transformers cache lock, evaluation_harness path race)
# crashed the batch on condition #1, losing 35 conditions of wallclock. Now:
# `--paper-grade` flag opts in; queue scripts (queue_baseline.sh /
# queue_chain.sh / queue_phase1_paper_grade.sh) MUST pass --paper-grade when
# `cfg.paper_grade=True` so init failures surface at preflight (10s) rather
# than batch start (hours-into-fire).
PAPER_GRADE_PREFLIGHT="${PAPER_GRADE_PREFLIGHT:-0}"
# B-703 (/stress A1.14 Chunk d P1-3 codex F4, 2026-05-17): `--sites csv` flag
# lets orchestrator filter site reachability check to actual chain scope. Pre-fix
# preflight checked all 5 VWA endpoints (shop/red/wiki/cls/homepage) → shop
# unreachable would block Phase 1a cls+red launch even though shop is Phase 1b
# deferred. Empty value = all sites (back-compat).
SITES_FILTER="${SITES_FILTER:-}"
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
  --sites <csv>        Comma-separated site filter for reachability check
                       (e.g. classifieds,reddit for Phase 1a; shopping,reddit
                       for Phase 1b). Empty = all sites (default).
  --skip-docker        Skip docker daemon check
  --check-docker       Force docker daemon check
  --require-cuda       Fail if torch CUDA runtime is unavailable
  --allow-missing-evaluator
                      Downgrade evaluator import failures to WARN
  --paper-grade       Run B-793 paper-grade evaluator probe — actually
                      instantiates VwaEvaluator(paper_grade=True) to catch
                      B-544 init-time fail-loud BEFORE batch launch. Queue
                      scripts MUST pass this flag for paper-grade runs.
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
    --sites)
      SITES_FILTER="${2:-}"
      shift 2
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
    --paper-grade)
      PAPER_GRADE_PREFLIGHT=1
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

  # B-703 (A1.14 Chunk d P1-3): apply --sites csv filter if provided.
  # Empty filter = check all (back-compat default).
  local filter_set=""
  if [[ -n "${SITES_FILTER}" ]]; then
    filter_set=",${SITES_FILTER},"
  fi

  for item in "${endpoints[@]}"; do
    local name="${item%%:*}"
    local url="${item#*:}"

    # B-703: skip if filter is set and this site isn't in it.
    if [[ -n "${filter_set}" && "${filter_set}" != *",${name},"* ]]; then
      pass "${name} endpoint (skipped by --sites filter)"
      continue
    fi

    if [[ -z "${url}" ]]; then
      fail "${name} endpoint is empty"
      continue
    fi

    # B-706 (A1.14 Chunk d P2-3 Claude unique): curl timeout 3s → 10s. Tailscale
    # cold connection or A100 docker stack first-request can take >3s; pre-fix
    # 3s caused false FAIL on legitimate paper-grade infra.
    if curl -fsS --max-time 10 "${url}" >/dev/null 2>&1; then
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

check_vwa_submodule_lock() {
  # A1.13 P1-3 fix (2026-05-16, 3-AI overlap): preflight now pins VWA submodule
  # to the p79-patches branch + commit f0c835b (B-91 LLM judge `pred=""` guard).
  # Pre-fix: `evaluator_router` import succeeded on any branch; reviewers'
  # OSF reproducibility run (default main branch) would silently miss B-91 fix
  # → visual_fp regression ~2-3pp on N/A tasks. Aligned with Makefile
  # `verify-version-locks` target.
  local vwa_dir="${PROJECT_DIR}/external/visualwebarena"
  if [[ ! -d "${vwa_dir}/.git" && ! -f "${vwa_dir}/.git" ]]; then
    if [[ "${ALLOW_MISSING_EVALUATOR}" == "1" ]]; then
      warn "VWA submodule dir not initialized: ${vwa_dir}"
    else
      fail "VWA submodule not initialized at ${vwa_dir} (run: git submodule update --init)"
    fi
    return
  fi
  # /stress A1.18 (2026-05-16): SHA bumped from f0c835b (pre-A1.18 B-91-only)
  # to eb5cbd8 (post-A1.18 full sweep: 15 findings B-254~B-268, see chronicle §159).
  # Bumped 2026-05-17 to 1c3a615 (A1.25 GRL Chunks 1+4: B-445~B-447 + B-535~B-540).
  # Bumped 2026-05-17 to 2f9b0b4 (A1.18-re Chunk 1: 11-fix substrate sweep
  # B-604+B-609~B-615+B-618+B-623+B-625, see chronicle §182).
  # f0c835b remains as the B-91 commit reference but is no longer the pinned HEAD.
  local expected_sha="ac33d2fcd9cec2fcbeddd56d0fa3da58b4c7e927"
  local expected_branch="p79-patches"
  local actual_sha actual_branch
  actual_sha="$(git -C "${vwa_dir}" rev-parse HEAD 2>/dev/null)"
  actual_branch="$(git -C "${vwa_dir}" rev-parse --abbrev-ref HEAD 2>/dev/null)"
  # B-682 (/stress A1.14 Chunk c P1-7 codex F7 unique OOB B, 2026-05-17):
  # SHA-first check; branch mismatch becomes WARN not FAIL. Pre-fix order
  # (branch first, then SHA) rejected reproducible detached-HEAD checkouts
  # at correct SHA — OSF reviewer cloning the submodule pin via
  # `git checkout <sha>` (canonical reproducibility workflow) would land
  # in detached HEAD with branch=`HEAD` and preflight FAILed. SHA is the
  # immutable evidence; branch is social metadata.
  #
  # B-683 (/stress A1.14 Chunk c P1-10 Claude unique, 2026-05-17): ancestor
  # fallback. Pre-fix hard-coded SHA required manual bump every submodule
  # advance; parallel session work on submodule (A1.18-re Chunk 2 / A1.25
  # GRL extension) would falsely FAIL even though the new HEAD contains
  # `expected_sha` as ancestor. `git merge-base --is-ancestor` allows
  # forward sync while guaranteeing the pinned commit's code is reachable.
  # `EXPECTED_SHA_STRICT=1` env reverts to exact-match for OSF-mode runs.
  if [[ "${actual_sha}" != "${expected_sha}" ]]; then
    if [[ "${EXPECTED_SHA_STRICT:-0}" == "1" ]]; then
      fail "VWA submodule SHA strict-mismatch: ${actual_sha} (expected exact ${expected_sha}; EXPECTED_SHA_STRICT=1 set)"
      return
    fi
    # Forward-sync ancestor fallback — actual HEAD must contain expected_sha as ancestor.
    if git -C "${vwa_dir}" merge-base --is-ancestor "${expected_sha}" HEAD 2>/dev/null; then
      warn "VWA submodule advanced past pin: actual ${actual_sha:0:8}, expected ${expected_sha:0:8} (ancestor verified — pinned code reachable, reproducibility intact)"
    else
      fail "VWA submodule SHA mismatch + NOT ancestor: actual ${actual_sha}, expected ${expected_sha} (forward-sync fallback rejected — paper-grade pin lost)"
      return
    fi
  fi
  # Branch is social metadata. Detached HEAD at correct SHA = canonical OSF
  # reproducibility checkout (silent pass). Other branches at correct SHA/ancestor
  # = WARN (likely working branch, not paper-grade FAIL).
  if [[ "${actual_branch}" != "${expected_branch}" && "${actual_branch}" != "HEAD" ]]; then
    warn "VWA submodule on branch '${actual_branch}' (expected '${expected_branch}'); SHA matches/ancestor so reproducibility intact"
  fi
  # Inline grep confirms the B-91 guard code body is actually present (not just
  # an empty file or refactor that dropped the guard); the SHA pin already
  # covers this but the grep gives an extra layer + better error message.
  local hf="${vwa_dir}/evaluation_harness/helper_functions.py"
  # B-91 guard form: `if not pred or not pred.strip():` followed by `return 0.0` on next line.
  # Verified upstream form 2026-05-16 (grep at f0c835b: helper_functions.py:589 + :634, 2 occurrences).
  #
  # B-707 (/stress A1.14 Chunk d P2-4 Claude unique, 2026-05-17): two refinements
  # to the brittleness vector:
  #   (a) Accept alternate guard idiom `if not pred:` (without `.strip()`) — upstream
  #       could refactor the strip check into a helper while keeping early-return.
  #   (b) `EXPECTED_B91_GUARDS` env override (default 2). OSF audit can set strict
  #       count; future upstream consolidation can drop to 1 via env without
  #       editing preflight.
  local guard_count_v1 guard_count_v2 guard_count
  guard_count_v1="$(grep -cP '^\s*if not pred or not pred\.strip\(\):' "${hf}" 2>/dev/null || echo 0)"
  guard_count_v2="$(grep -cP '^\s*if not pred(\.strip\(\))?:' "${hf}" 2>/dev/null || echo 0)"
  guard_count=$(( guard_count_v1 > guard_count_v2 ? guard_count_v1 : guard_count_v2 ))
  local expected_b91_guards="${EXPECTED_B91_GUARDS:-2}"
  if [[ "${guard_count}" -lt "${expected_b91_guards}" ]]; then
    fail "VWA submodule missing B-91 LLM judge guard in ${hf} (expected ≥${expected_b91_guards} occurrences, got ${guard_count}; v1='if not pred or not pred.strip():'=${guard_count_v1}, v2='if not pred[.strip()]?:'=${guard_count_v2}; set EXPECTED_B91_GUARDS=1 to allow upstream refactor)"
    return
  fi
  pass "VWA submodule locked at ${expected_branch}@${expected_sha:0:8} (B-91 guard count=${guard_count} ≥ ${expected_b91_guards})"
}

check_openai_api_key() {
  # B-679 (/stress A1.14 Chunk b P1-9 Claude unique OOB A, 2026-05-17):
  # paper-grade VWA evaluator runs LLM judge for N/A tasks via OpenAI API
  # (`external/visualwebarena/evaluation_harness/helper_functions.py:613+707`
  # call `generate_from_openai_chat_completion`). Pre-fix preflight `setdefault
  # OPENAI_API_KEY=DUMMY_P79_PRECHECK` only ensured evaluator IMPORT works;
  # the DUMMY placeholder masked the runtime requirement → paper-grade run
  # started, then crashed at first N/A task evaluation when real OpenAI call
  # rejected the dummy key. This explicit check fails preflight if the key
  # is unset OR contains DUMMY/PLACEHOLDER substring.
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    fail "OPENAI_API_KEY not set — paper-grade VWA LLM judge (helper_functions.py:613/707) will crash at first N/A task evaluation. Source ~/.openai_api_key (or .env) before launch."
    return
  fi
  if [[ "${OPENAI_API_KEY}" == *DUMMY* || "${OPENAI_API_KEY}" == *PLACEHOLDER* ]]; then
    fail "OPENAI_API_KEY is a placeholder ('${OPENAI_API_KEY:0:24}...') — paper-grade run requires real API key for VWA LLM judge"
    return
  fi
  if [[ "${#OPENAI_API_KEY}" -lt 20 ]]; then
    fail "OPENAI_API_KEY suspiciously short (${#OPENAI_API_KEY} chars; real sk-* keys are ≥48 chars) — verify it's the real key"
    return
  fi
  pass "OPENAI_API_KEY set (${#OPENAI_API_KEY} chars, real-looking)"
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

# B-793 (/stress A1.9 cold-start P1-9 root-cause fix, 2026-05-17): paper-grade
# evaluator probe. Runs `create_evaluator(env_cfg, paper_grade=True)` in a
# subprocess so init-time `EvaluatorUnavailableError` (OpenAI key broken,
# transformers cache lock, evaluation_harness path race) surfaces here
# instead of crashing the 36-condition batch on its first condition.
#
# Why subprocess (not in-line eval): create_evaluator touches global env state
# (OPENAI_API_KEY, sys.path append) — subprocess isolates side-effects from
# the preflight Python interpreter.
check_vwa_evaluator_paper_grade() {
  if [[ "${PAPER_GRADE_PREFLIGHT}" != "1" ]]; then
    return  # not requested; skip silently
  fi
  if [[ -z "${PYTHON_BIN:-}" ]]; then
    fail "B-793 paper-grade probe needs Python (not found)"
    return
  fi

  local py_output
  if ! py_output="$(${PYTHON_BIN} - <<'PY' 2>&1
import sys
try:
    from p79.experiment.environment import create_evaluator, EvaluatorUnavailableError
    e = create_evaluator(
        {"type": "vwa", "benchmark": "visualwebarena"},
        paper_grade=True,
    )
    # Inspect _available — paper_grade=True should have raised if init was
    # broken; reach this only on success. e is unused otherwise.
    del e
    print("ok")
except EvaluatorUnavailableError as exc:
    print(f"PAPER_GRADE_INIT_BROKEN: {exc!r}")
    sys.exit(2)
except Exception as exc:
    print(f"UNEXPECTED_PROBE_ERROR: {type(exc).__name__}: {exc!r}")
    sys.exit(3)
PY
)"; then
    fail "B-793 paper-grade evaluator probe FAILED (${py_output})"
    fail "  → 36-condition batch would crash on condition #1; fix env before fire."
    return
  fi
  pass "B-793 paper-grade evaluator probe PASSED (init-fail surface at preflight, not at fire)"
}

# B-884 (/stress A1.24 P1-3-B*, 2026-05-17): half-deleted run substrate gate.
# Pre-fix: clear_tasks crash / ^C between digest cleanup and condition_summary
# unlink leaves "digest cleaned + cond_summary Finalized + .cleaning marker
# present" zombie state. Re-fire would resume to half-deleted run + corrupt
# downstream aggregation. preflight check now scans paper-grade run dirs for
# (a) `.cleaning` marker files left by interrupted clear_tasks (B-890), and
# (b) `.in_progress` markers staler than 6 hours (likely orphaned by killed
# runner). (a) fatal; (b) warn-only (runner may still resume legitimately).
check_clear_tasks_recovery() {
  if [[ "${PAPER_GRADE_PREFLIGHT}" != "1" ]]; then
    return  # only enforce in --paper-grade
  fi
  print_check "B-884 clear_tasks recovery substrate (half-deleted state)"
  local results_root="${PROJECT_DIR}/results/visualwebarena/phase1"
  if [[ ! -d "${results_root}" ]]; then
    pass "B-884 no phase1 results dir yet — clean substrate"
    return
  fi
  # (a) .cleaning markers — clear_tasks crash/^C between digest + cond_summary
  local cleaning_markers
  cleaning_markers=$(find "${results_root}" -maxdepth 4 -name ".cleaning" -type f 2>/dev/null || true)
  if [[ -n "${cleaning_markers}" ]]; then
    echo "${cleaning_markers}" | while IFS= read -r m; do echo "    half-deleted marker: ${m}"; done
    fail "B-884 found .cleaning marker(s) → clear_tasks was interrupted mid-operation. Re-run clear_tasks to completion OR manually delete the marker after verifying state consistency."
    return
  fi
  # (b) stale .in_progress markers (>6h likely orphaned)
  local stale_markers
  stale_markers=$(find "${results_root}" -maxdepth 5 -name ".in_progress" -type f -mmin +360 2>/dev/null || true)
  if [[ -n "${stale_markers}" ]]; then
    echo "${stale_markers}" | while IFS= read -r m; do echo "    stale (>6h): ${m}"; done
    warn "B-884 found stale .in_progress marker(s) >6h old — likely orphaned by killed runner. Manual cleanup recommended (verify no live pid + rm marker)."
    # Not fatal — runner might restart and reuse; warn only.
  fi
  pass "B-884 no half-deleted run substrate detected"
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
  check_vwa_submodule_lock
  check_openai_api_key
  check_vwa_evaluator_import
  check_vwa_evaluator_paper_grade  # B-793: skipped unless --paper-grade
  check_clear_tasks_recovery       # B-884: skipped unless --paper-grade

  if (( EXIT_CODE == 0 )); then
    echo "Preflight completed successfully."
  else
    echo "Preflight found issues."
  fi

  exit ${EXIT_CODE}
}

main "$@"
