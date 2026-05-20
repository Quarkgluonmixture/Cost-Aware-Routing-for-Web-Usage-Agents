#!/usr/bin/env bash
# queue_diagnostic_replay.sh — Fire-6 RCA Stage C2 targeted diagnostic replay.
#
# Runs a SMALL, task-scoped, NON-CANONICAL reproduction of named tasks under the
# real paper-grade runtime (real agent, real evaluator, C1 evaluator-isolation +
# `_dump_eval_timeout_forensic` active) to PROVE C1 fixes the evaluator
# `program_html` Page.goto 30s timeout WITHOUT a full Fire. It also provides the
# matched-temporal-context reproduce that Gate 8 cross_fire_recurrence (Rule 2)
# requires to classify a recurrent task (e.g. cls 75).
#
# Why this is NOT a canonical fire (all enforced, not by convention):
#   - output → results/diagnostic_replay/<run_id>/ (config redirect; never
#     globbed by results/visualwebarena/phase1 aggregators)
#   - every episode stamped sr_excluded=True (load_episode_summary_strict
#     reject_sr_excluded=True firewall keeps them out of paper §1 SR)
#   - M1 quarantine fail-closed abort suppressed → all named tasks run + capture
#     per-task forensics (does not abort at task 4's first quarantine)
#   - Gate 8 bypass is the diagnostic-scoped override ONLY (4 fail-closed
#     guards), never a canonical Gate 8 disable
#
# Usage:
#   bash scripts/queues/queue_diagnostic_replay.sh <baseline> <mode> <site> <tasks>
#   e.g. bash scripts/queues/queue_diagnostic_replay.sh B0 dom classifieds 4,75
#
#   RESET_BEFORE=0 ...  → skip site reset (default 1)
#
# Requires config: configs/exp_v2_<baseline>_<mode>_<site>.yaml (reuses the
# canonical config for real B0 proxy + A100 endpoints + observation_mode).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_DIR}"

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 <baseline:B0|B1|B2> <mode:dom|som|vision> <site> <tasks>" >&2
  echo "  e.g. bash $0 B0 dom classifieds 4,75" >&2
  exit 2
fi

BASELINE="$1"; MODE="$2"; SITE="$3"; TASKS="$4"
BENCHMARK="vwa"

# ---------- validation ----------
if [[ "${BASELINE}" != "B0" && "${BASELINE}" != "B1" && "${BASELINE}" != "B2" ]]; then
  echo "[diag][error] Invalid baseline: ${BASELINE} (expected B0/B1/B2)" >&2; exit 2
fi
if [[ "${MODE}" != "dom" && "${MODE}" != "som" && "${MODE}" != "vision" ]]; then
  echo "[diag][error] Invalid mode: ${MODE} (expected dom/som/vision)" >&2; exit 2
fi
if [[ "${SITE}" != "classifieds" && "${SITE}" != "reddit" && "${SITE}" != "shopping" ]]; then
  echo "[diag][error] Invalid VWA site: ${SITE}" >&2; exit 2
fi
if [[ -z "${TASKS}" || ! "${TASKS}" =~ ^[0-9,-]+$ ]]; then
  echo "[diag][error] Invalid --tasks spec: '${TASKS}' (expected e.g. '4,75' or '0-9')" >&2; exit 2
fi

CONFIG="${REPO_DIR}/configs/exp_v2_${BASELINE}_${MODE}_${SITE}.yaml"
if [[ ! -f "${CONFIG}" ]]; then
  echo "[diag][error] Config not found: ${CONFIG}" >&2; exit 1
fi

PYTHON_BIN="${REPO_DIR}/.venv/bin/python3"
LOG_DIR="${REPO_DIR}/logs"
mkdir -p "${LOG_DIR}"

# ---------- diagnostic-replay env (BEFORE lib + runner) ----------
# P79_DIAGNOSTIC_REPLAY=1  → config.normalize sets cfg.diagnostic_replay=True
#                            (non-canonical output + sr_excluded + abort-suppress)
# QUARANTINE_DIAGNOSTIC_REPLAY=1 → arms the diagnostic-scoped Gate 8 override
#                            (still requires the other 3 guards to be passed
#                            explicitly below).
export P79_DIAGNOSTIC_REPLAY=1
export QUARANTINE_DIAGNOSTIC_REPLAY=1
# Diagnostic replay is NOT a canonical fire: do not arm the paper-grade
# fail-closed env (env_snapshot strictness etc.). C1 isolation + forensics are
# unconditional, so fidelity is preserved without paper-grade gating.
export P79_PAPER_GRADE=0

# ---------- shared paper-grade lib (env init + A100 locality + reset/auth) ----------
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_lib_paper_grade_gates.sh"
init_paper_grade_env "${REPO_DIR}"
assert_a100_url_locality

# Same-site collision rule still applies — a diagnostic replay touches the live
# shared docker site + user session, so it must not co-run with any other runner
# on this site (B0 XOR B1 XOR B2 + no concurrent canonical fire).
if ! acquire_site_lock "${SITE}" "${BENCHMARK}" "queue_diagnostic_replay"; then
  exit $?
fi
trap "release_site_lock" EXIT INT TERM

if [[ "${BASELINE}" == "B0" ]]; then
  load_proxy_api_key "${REPO_DIR}" "diag"
fi

# ---------- run_id + non-canonical run_dir ----------
RUN_ID="diag_${BASELINE}_${MODE}_${SITE}_$(date +%Y%m%d_%H%M%S)_$$"
RUN_DIR="${REPO_DIR}/results/diagnostic_replay/${RUN_ID}"
RUNNER_LOG="${LOG_DIR}/${RUN_ID}_runner.log"
echo "[diag] config=${CONFIG}"
echo "[diag] tasks=${TASKS} site=${SITE} baseline=${BASELINE} mode=${MODE}"
echo "[diag] run_dir=${RUN_DIR} (NON-CANONICAL)"

# ---------- site reset (clean initial state; cumulative-load is mid-run, not here) ----------
if [[ "${RESET_BEFORE:-1}" == "1" ]]; then
  reset_and_auth_gate --site "${SITE}" --repo "${REPO_DIR}" --python "${PYTHON_BIN}" \
    --log-prefix "diag" --reset-label "diag_${MODE}_${SITE}"
else
  echo "[diag] RESET_BEFORE=0 — skipping site reset (explicit)"
fi

# ---------- Gate 8 diagnostic-scoped override (HARD safety gate) ----------
# The wrapper REQUIRES the override to be granted: this proves all four guards
# (env + --diagnostic-replay + non-canonical --output-path + task-scoped) hold
# BEFORE we touch the substrate. If any guard fails, the canonical halt fires
# and we abort — never silently downgrade to a canonical run.
REGISTRY_CLI="${REPO_DIR}/scripts/maintenance/quarantine_registry.py"
echo "[diag] Gate 8 diagnostic-override preflight (site=${SITE} tasks=${TASKS})"
if ! "${PYTHON_BIN}" "${REGISTRY_CLI}" preflight \
      --site "${SITE}" --tasks "${TASKS}" \
      --diagnostic-replay --output-path "${RUN_DIR}"; then
  echo "[diag][FATAL] Gate 8 diagnostic override NOT granted — refusing to run." >&2
  echo "[diag][FATAL] (check QUARANTINE_DIAGNOSTIC_REPLAY=1 + non-canonical output + <=25 tasks)" >&2
  exit 1
fi

# ---------- run the diagnostic replay (foreground; exit code propagates) ----------
echo "[diag] launching diagnostic replay → ${RUNNER_LOG}"
export GCE_METADATA_IP=127.0.0.1 GCE_METADATA_TIMEOUT=1 GCE_METADATA_HOST=disabled.invalid
set +e
"${PYTHON_BIN}" scripts/run_experiment.py \
  --config "${CONFIG}" \
  --run_id "${RUN_ID}" \
  --diagnostic-replay --tasks "${TASKS}" \
  --log_path "${RUNNER_LOG}" \
  2>&1 | tee "${RUNNER_LOG}"
RC=${PIPESTATUS[0]}
set -e

echo
echo "[diag] diagnostic replay finished rc=${RC}"
echo "[diag]   episodes:  ${RUN_DIR}/*/episodes/*_summary_v2.json (sr_excluded=True)"
echo "[diag]   forensics: ${REPO_DIR}/logs/eval_timeout_forensic/*.json (if any eval timeout)"
echo "[diag]   to inspect eval-context: grep eval_context_mode/eval_goto_latency_ms in episode summaries"
exit "${RC}"
