#!/usr/bin/env bash
# queue_phantom_prompt.sh — Launch P-prompt (SoM prompt + AXTree text + no image).
#
# P-prompt (§105): SoM prompt + AXTree text + 无图. Symmetric counterpart of
# phantom_text (P-text; legacy mode value phantom_dom). Together they complete the diamond design — DOM splits
# into two single-axis-swap phantoms (P-text via axis 1; P-prompt via axis 2),
# both converging at P-SoM (both axes swapped). Lets paper measure axis 2
# (prompt) effect in BOTH AXTree-text and [SOM_MARKS]-text contexts, and the
# axis 1 (text) effect in BOTH DOM-prompt and SoM-prompt contexts.
#
# Usage:
#   bash scripts/queues/queue_phantom_prompt.sh <baseline> <site> [benchmark]
#   - baseline:  B0 | B1 | B2
#   - site:      classifieds | reddit | shopping (vwa)
#   - benchmark: vwa (默认) | wa (only B0 right now)
#
# Examples:
#   bash scripts/queues/queue_phantom_prompt.sh B0 reddit
#   bash scripts/queues/queue_phantom_prompt.sh B0 classifieds
#   RESET_BEFORE=1 bash scripts/queues/queue_phantom_prompt.sh B0 classifieds

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_DIR}"

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <baseline:B0|B1|B2> <site> [benchmark:vwa|wa]" >&2
  echo "  Example: bash $0 B0 classifieds" >&2
  echo "  RESET_BEFORE=1 bash $0 B0 classifieds     # reset shopping-style site before launch" >&2
  exit 2
fi

BASELINE="$1"; SITE="$2"
BENCHMARK="${3:-vwa}"

# Validation
if [[ "${BASELINE}" != "B0" && "${BASELINE}" != "B1" && "${BASELINE}" != "B2" ]]; then
  echo "Invalid baseline: ${BASELINE} (expected B0, B1 or B2)" >&2; exit 2
fi
if [[ "${BENCHMARK}" != "vwa" && "${BENCHMARK}" != "wa" ]]; then
  echo "Invalid benchmark: ${BENCHMARK} (expected vwa or wa)" >&2; exit 2
fi
if [[ "${BENCHMARK}" == "vwa" && "${SITE}" != "classifieds" && "${SITE}" != "reddit" && "${SITE}" != "shopping" ]]; then
  echo "Invalid VWA site: ${SITE}" >&2; exit 2
fi
if [[ "${BENCHMARK}" == "wa" && "${SITE}" != "reddit" && "${SITE}" != "shopping" && "${SITE}" != "shopping_admin" ]]; then
  echo "Invalid WA site: ${SITE}" >&2; exit 2
fi

# Build config name
# VWA: exp_v2_<baseline>_phantom_prompt_<site>.yaml
# WA:  exp_v2_<baseline>_phantom_prompt_wa_<site>.yaml
CFG_NAME="${BASELINE}_phantom_prompt"
[[ "${BENCHMARK}" == "wa" ]] && CFG_NAME="${CFG_NAME}_wa"
CFG_NAME="${CFG_NAME}_${SITE}"
CONFIG="${REPO_DIR}/configs/exp_v2_${CFG_NAME}.yaml"

if [[ ! -f "${CONFIG}" ]]; then
  echo "[phantom_prompt][error] Config not found: ${CONFIG}" >&2; exit 1
fi

COND_ID="phase1_phantom_prompt_router_0"

PYTHON_BIN="${REPO_DIR}/.venv/bin/python3"
LOG_DIR="${REPO_DIR}/logs"
mkdir -p "${LOG_DIR}"

# ---------- A1.13 lib (2026-05-16): shared paper-grade gates ----------
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_lib_paper_grade_gates.sh"
init_paper_grade_env "${REPO_DIR}"
assert_a100_url_locality

# ---------- B0 PROXY API key 加载 ----------
if [[ "${BASELINE}" == "B0" ]]; then
  load_proxy_api_key "${REPO_DIR}" "phantom_prompt"
fi

# ---------- 决定 run_id + run_dir ----------
if [[ "${BENCHMARK}" == "wa" ]]; then
  PHASE_DIR="${REPO_DIR}/results/webarena/phase1"
else
  PHASE_DIR="${REPO_DIR}/results/visualwebarena/phase1"
fi

mint_run_id "${CFG_NAME}" "${PHASE_DIR}" "phantom_prompt"
# B-581 (A1.13 P0-5): RUNNER_LOG uses RUN_ID (0-collision); TS_FULL removed.

RUN_DIR="${PHASE_DIR}/${RUN_ID}"
echo "[phantom_prompt] config=${CONFIG}"
echo "[phantom_prompt] run_dir=${RUN_DIR}"
echo "[phantom_prompt] condition=${COND_ID}"

# ---------- 检查 runner 是否已在跑 ----------
if pgrep -f "run_experiment.py.*${RUN_ID}" > /dev/null; then
  echo "[phantom_prompt] runner for ${RUN_ID} already running, skipping spawn"
  echo "[phantom_prompt] (RESET_BEFORE skipped — runner already attached to current site state)"
else
  # ---------- Optional: site reset before launch ----------
  # IMPORTANT: reset AFTER idempotent runner check (race fixed 2026-04-28 §104).
  # A1.13 P0-1 (2026-05-16) propagated B-224 hard-fail via reset_and_auth_gate.
  if [[ "${RESET_BEFORE:-0}" == "1" && "${BENCHMARK}" != "wa" ]]; then
    reset_and_auth_gate "${SITE}" "${REPO_DIR}" "${PYTHON_BIN}" "phantom_prompt" "phantom_prompt_${SITE}"
  elif [[ "${RESET_BEFORE:-0}" == "1" ]]; then
    echo "[phantom_prompt] RESET_BEFORE=1 but BENCHMARK=wa — WA reset+auth refresh uses different mechanism, skipping"
  fi

  RUNNER_LOG="${LOG_DIR}/${RUN_ID}_runner.log"
  echo "[phantom_prompt] launching runner → ${RUNNER_LOG}"
  # codex stress v6 C4: redirect runner stdout/stderr to RUNNER_LOG (was /dev/null).
  # Python logging goes to stderr — /dev/null discarded all phantom runner logs,
  # making mid-run crash debug impossible + paper-grade audit trail incomplete.
  setsid nohup "${PYTHON_BIN}" scripts/run_experiment.py \
    --config "${CONFIG}" \
    --run_id "${RUN_ID}" \
    --log_path "${RUNNER_LOG}" \
    > "${RUNNER_LOG}" 2>&1 < /dev/null &
  disown
  sleep 3
  if pgrep -f "run_experiment.py.*${RUN_ID}" > /dev/null; then
    echo "[phantom_prompt] runner pid=$(pgrep -f "run_experiment.py.*${RUN_ID}" | head -1)"
  else
    echo "[phantom_prompt][error] runner failed to start, see ${RUNNER_LOG}" >&2
    [[ -f "${RUNNER_LOG}" ]] && tail -20 "${RUNNER_LOG}" >&2
    exit 1
  fi
fi

# ---------- watchdog 启动 ----------
WD_STATE="${LOG_DIR}/exp_watchdog_${RUN_ID}_v2.state.json"
WD_LOG="${LOG_DIR}/exp_watchdog_${RUN_ID}_v2.log"
# Unified per-baseline aggregate gallery — all B0/B1 modes share one URL.
AGGREGATE_PREFIX="${BASELINE}_3mode"

# Runner PID for watchdog self-exit — watchdog auto-exits when this PID dies
# AND condition_summary_v2.json present. Prevents init-orphan idle loops.
RUNNER_PID=$(pgrep -f "run_experiment.py.*${RUN_ID}" | head -1)

if pgrep -f "experiment_watchdog.*${RUN_ID}" > /dev/null; then
  echo "[phantom_prompt] watchdog for ${RUN_ID} already running, skipping spawn"
else
  echo "[phantom_prompt] launching watchdog → ${WD_LOG} (runner pid=${RUNNER_PID:-unknown})"
  setsid nohup "${PYTHON_BIN}" -u scripts/maintenance/experiment_watchdog.py \
    --run-dir "${RUN_DIR}" \
    --condition "${COND_ID}" \
    --poll-secs 30 --idle-alert-mins 30 \
    --ntfy-topic p79-exp-dgx-spark \
    --state-file "${WD_STATE}" \
    --aggregate-prefix "${AGGREGATE_PREFIX}" \
    --glm-config .auth/glm \
    --digest-dir "${RUN_DIR}/analysis/digest" \
    ${RUNNER_PID:+--runner-pid "${RUNNER_PID}"} \
    >> "${WD_LOG}" 2>&1 < /dev/null &
  disown
  sleep 2
  if pgrep -f "experiment_watchdog.*${RUN_ID}" > /dev/null; then
    echo "[phantom_prompt] watchdog pid=$(pgrep -f "experiment_watchdog.*${RUN_ID}" | head -1)"
  else
    echo "[phantom_prompt][error] watchdog failed to start, see ${WD_LOG}" >&2
    exit 1
  fi
fi

echo ""
echo "[phantom_prompt] OK — ${CFG_NAME} (${BENCHMARK}/${SITE}) running"
echo "  runner log:   ${RUNNER_LOG:-<existing>}"
echo "  watchdog log: ${WD_LOG}"
