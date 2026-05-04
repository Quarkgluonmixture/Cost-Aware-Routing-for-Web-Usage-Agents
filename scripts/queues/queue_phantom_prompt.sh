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
#   - baseline:  B0 | B1
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
  echo "Usage: $0 <baseline:B0|B1> <site> [benchmark:vwa|wa]" >&2
  echo "  Example: bash $0 B0 classifieds" >&2
  echo "  RESET_BEFORE=1 bash $0 B0 classifieds     # reset shopping-style site before launch" >&2
  exit 2
fi

BASELINE="$1"; SITE="$2"
BENCHMARK="${3:-vwa}"

# Validation
if [[ "${BASELINE}" != "B0" && "${BASELINE}" != "B1" ]]; then
  echo "Invalid baseline: ${BASELINE} (expected B0 or B1)" >&2; exit 2
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

# ---------- DGX Spark CUDA workaround ----------
export PYTORCH_NVML_BASED_CUDA_CHECK=1
export CUDA_MPS_PIPE_DIRECTORY=""
export CUDA_MPS_LOG_DIRECTORY=""

# ---------- VWA 远程站点 env ----------
if [[ -f "${REPO_DIR}/scripts/vwa_env_remote.sh" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_DIR}/scripts/vwa_env_remote.sh"
fi

# ---------- WIKIPEDIA ZIM 版本 ----------
export WIKIPEDIA_ZIM_VERSION="${WIKIPEDIA_ZIM_VERSION:-wikipedia_en_all_maxi_2025-08}"

# ---------- B0 PROXY API key 加载 ----------
if [[ "${BASELINE}" == "B0" ]]; then
  if [[ -z "${PROXY_API_KEY:-}" ]]; then
    AUTH_FILE="${REPO_DIR}/.auth/qwen_api"
    if [[ -f "${AUTH_FILE}" ]]; then
      raw_key="$(grep -m1 '^rp_' "${AUTH_FILE}" | tr -d '[:space:]')"
      if [[ -n "${raw_key}" ]]; then
        export PROXY_API_KEY="${raw_key}"
        export QWEN_API_KEY="${raw_key}"
        export DASHSCOPE_API_KEY="${raw_key}"
        echo "[phantom_prompt] Loaded PROXY_API_KEY from ${AUTH_FILE}"
      else
        echo "[phantom_prompt][error] ${AUTH_FILE} 存在但无 rp_ key" >&2; exit 1
      fi
    else
      echo "[phantom_prompt][error] ${AUTH_FILE} 不存在，且 PROXY_API_KEY 未设置" >&2; exit 1
    fi
  fi
fi

# ---------- 决定 run_id + run_dir ----------
TS_DATE="$(date +%Y%m%d)"
TS_FULL="$(date +%Y%m%d_%H%M%S)"
if [[ "${BENCHMARK}" == "wa" ]]; then
  PHASE_DIR="${REPO_DIR}/results/webarena/phase1"
else
  PHASE_DIR="${REPO_DIR}/results/visualwebarena/phase1"
fi

EXISTING="$(ls -dt "${PHASE_DIR}/${CFG_NAME}_"[0-9]* 2>/dev/null | head -1 || true)"
if [[ -n "${EXISTING}" ]]; then
  RUN_ID="$(basename "${EXISTING}")"
  echo "[phantom_prompt] resuming existing run_id=${RUN_ID}"
else
  RUN_ID="${CFG_NAME}_${TS_DATE}"
  echo "[phantom_prompt] new run_id=${RUN_ID}"
fi

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
  # IMPORTANT: reset is AFTER the idempotent runner check — resetting while
  # a runner is attached destroys site state under it (race condition fixed
  # 2026-04-28 — see 实验笔记 §104).
  if [[ "${RESET_BEFORE:-0}" == "1" && "${BENCHMARK}" != "wa" ]]; then
    if [[ -f "${REPO_DIR}/scripts/maintenance/reset_vwa_sites.sh" ]]; then
      # shellcheck disable=SC1091
      source "${REPO_DIR}/scripts/maintenance/reset_vwa_sites.sh"
      echo "[phantom_prompt] RESET_BEFORE=1 → resetting site=${SITE}..."
      if reset_vwa_sites "${SITE}" "phantom_prompt_${SITE}"; then
        echo "[phantom_prompt] reset OK; sleeping 15s for site to settle..."
        sleep 15
        echo "[phantom_prompt] refreshing .auth/${SITE}_state.json post-reset..."
        if "${PYTHON_BIN}" -c "
import sys
sys.path.insert(0, '${REPO_DIR}')
from pathlib import Path
from p79.utils.auth_refresh import refresh_site_auth
sys.exit(0 if refresh_site_auth('${SITE}', Path('${REPO_DIR}/.auth')) else 1)
" 2>&1; then
          echo "[phantom_prompt] auth refresh OK — runner task=0 will be LOGGED IN"
        else
          echo "[phantom_prompt][warn] post-reset auth refresh failed; watchdog will retry reactively after streak=3" >&2
        fi
      else
        rc=$?
        echo "[phantom_prompt][error] reset failed (rc=${rc}); aborting to preserve paper-grade integrity." >&2
        echo "[phantom_prompt][error] To bypass reset (paper-grade dirty), explicitly set RESET_BEFORE=0." >&2
        exit 1
      fi
    else
      echo "[phantom_prompt][error] reset_vwa_sites.sh not found but RESET_BEFORE=1; aborting." >&2
      echo "[phantom_prompt][error] To bypass reset (paper-grade dirty), explicitly set RESET_BEFORE=0." >&2
      exit 1
    fi
  elif [[ "${RESET_BEFORE:-0}" == "1" ]]; then
    echo "[phantom_prompt] RESET_BEFORE=1 but BENCHMARK=wa — WA reset+auth refresh uses different mechanism, skipping"
  fi

  RUNNER_LOG="${LOG_DIR}/${CFG_NAME}_resume_${TS_FULL}.log"
  echo "[phantom_prompt] launching runner → ${RUNNER_LOG}"
  setsid nohup "${PYTHON_BIN}" scripts/run_experiment.py \
    --config "${CONFIG}" \
    --run_id "${RUN_ID}" \
    --log_path "${RUNNER_LOG}" \
    > /dev/null 2>&1 < /dev/null &
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
