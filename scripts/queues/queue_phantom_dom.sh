#!/usr/bin/env bash
# queue_phantom_dom.sh — Launch Phantom-DOM (DOM prompt + [SOM_MARKS] text + no image).
#
# Phantom-DOM (§103): DOM prompt + SoM marks 文本 + 无图（control for axis 2 prompt effect,
# vs Phantom-SoM which uses SoM prompt). 同样的 [SOM_MARKS] 文本，区别仅在 system prompt.
#
# 这个脚本统一处理:
#   - PROXY_API_KEY 从 .auth/qwen_api 加载 (B0 用)
#   - VWA 远程 host env 加载
#   - CUDA workaround env (DGX Spark sm_121)
#   - WIKIPEDIA ZIM 版本
#   - runner + watchdog 一起启动，已存在则跳过 (idempotent)
#   - RESET 在 idempotent check 之后执行 (race-safe per §104 audit)
#
# Usage:
#   bash scripts/queues/queue_phantom_dom.sh <baseline> <site> [benchmark]
#   - baseline:  B0 | B1
#   - site:      classifieds | reddit | shopping (vwa) | shopping_admin (wa-only)
#   - benchmark: vwa (默认) | wa
#
# Examples:
#   bash scripts/queues/queue_phantom_dom.sh B0 reddit                      # VWA reddit
#   bash scripts/queues/queue_phantom_dom.sh B1 shopping                    # VWA shopping
#   bash scripts/queues/queue_phantom_dom.sh B0 shopping_admin wa           # WA shopping_admin
#   RESET_BEFORE=1 bash scripts/queues/queue_phantom_dom.sh B0 shopping     # with reset

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_DIR}"

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <baseline:B0|B1> <site> [benchmark:vwa|wa]" >&2
  echo "  Example: bash $0 B0 shopping" >&2
  echo "  RESET_BEFORE=1 bash $0 B0 shopping     # reset shopping container before launch" >&2
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
# VWA: exp_v2_<baseline>_phantom_dom_<site>.yaml
# WA:  exp_v2_<baseline>_phantom_dom_wa_<site>.yaml
CFG_NAME="${BASELINE}_phantom_dom"
[[ "${BENCHMARK}" == "wa" ]] && CFG_NAME="${CFG_NAME}_wa"
CFG_NAME="${CFG_NAME}_${SITE}"
CONFIG="${REPO_DIR}/configs/exp_v2_${CFG_NAME}.yaml"

if [[ ! -f "${CONFIG}" ]]; then
  echo "[phantom_dom][error] Config not found: ${CONFIG}" >&2; exit 1
fi

COND_ID="phase1_phantom_dom_router_0"

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
        echo "[phantom_dom] Loaded PROXY_API_KEY from ${AUTH_FILE}"
      else
        echo "[phantom_dom][error] ${AUTH_FILE} 存在但无 rp_ key" >&2; exit 1
      fi
    else
      echo "[phantom_dom][error] ${AUTH_FILE} 不存在，且 PROXY_API_KEY 未设置" >&2; exit 1
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
  echo "[phantom_dom] resuming existing run_id=${RUN_ID}"
else
  RUN_ID="${CFG_NAME}_${TS_DATE}"
  echo "[phantom_dom] new run_id=${RUN_ID}"
fi

RUN_DIR="${PHASE_DIR}/${RUN_ID}"
echo "[phantom_dom] config=${CONFIG}"
echo "[phantom_dom] run_dir=${RUN_DIR}"
echo "[phantom_dom] condition=${COND_ID}"

# ---------- 检查 runner 是否已在跑 ----------
if pgrep -f "run_experiment.py.*${RUN_ID}" > /dev/null; then
  echo "[phantom_dom] runner for ${RUN_ID} already running, skipping spawn"
  echo "[phantom_dom] (RESET_BEFORE skipped — runner already attached to current site state)"
else
  # ---------- Optional: site reset before launch ----------
  # IMPORTANT: reset is AFTER the idempotent runner check — resetting while
  # a runner is attached destroys site state under it (race condition fixed
  # 2026-04-28 — see 实验笔记 §104).
  if [[ "${RESET_BEFORE:-0}" == "1" && "${BENCHMARK}" != "wa" ]]; then
    if [[ -f "${REPO_DIR}/scripts/maintenance/reset_vwa_sites.sh" ]]; then
      # shellcheck disable=SC1091
      source "${REPO_DIR}/scripts/maintenance/reset_vwa_sites.sh"
      echo "[phantom_dom] RESET_BEFORE=1 → resetting site=${SITE}..."
      if reset_vwa_sites "${SITE}" "phantom_dom_${SITE}"; then
        echo "[phantom_dom] reset OK; sleeping 15s for site to settle..."
        sleep 15
      else
        echo "[phantom_dom][warn] reset failed (rc=$?); continuing anyway" >&2
      fi
    else
      echo "[phantom_dom][warn] reset_vwa_sites.sh not found; skipping reset" >&2
    fi
  elif [[ "${RESET_BEFORE:-0}" == "1" ]]; then
    echo "[phantom_dom] RESET_BEFORE=1 but BENCHMARK=wa — WA reset uses different mechanism, skipping"
  fi

  RUNNER_LOG="${LOG_DIR}/${CFG_NAME}_resume_${TS_FULL}.log"
  echo "[phantom_dom] launching runner → ${RUNNER_LOG}"
  setsid nohup "${PYTHON_BIN}" scripts/run_experiment.py \
    --config "${CONFIG}" \
    --run_id "${RUN_ID}" \
    --log_path "${RUNNER_LOG}" \
    > /dev/null 2>&1 < /dev/null &
  disown
  sleep 3
  if pgrep -f "run_experiment.py.*${RUN_ID}" > /dev/null; then
    echo "[phantom_dom] runner pid=$(pgrep -f "run_experiment.py.*${RUN_ID}" | head -1)"
  else
    echo "[phantom_dom][error] runner failed to start, see ${RUNNER_LOG}" >&2
    [[ -f "${RUNNER_LOG}" ]] && tail -20 "${RUNNER_LOG}" >&2
    exit 1
  fi
fi

# ---------- watchdog 启动 ----------
WD_STATE="${LOG_DIR}/exp_watchdog_${RUN_ID}_v2.state.json"
WD_LOG="${LOG_DIR}/exp_watchdog_${RUN_ID}_v2.log"
AGGREGATE_PREFIX="${CFG_NAME%_${SITE}}"

if pgrep -f "experiment_watchdog.*${RUN_ID}" > /dev/null; then
  echo "[phantom_dom] watchdog for ${RUN_ID} already running, skipping spawn"
else
  echo "[phantom_dom] launching watchdog → ${WD_LOG}"
  setsid nohup "${PYTHON_BIN}" -u scripts/maintenance/experiment_watchdog.py \
    --run-dir "${RUN_DIR}" \
    --condition "${COND_ID}" \
    --poll-secs 30 --idle-alert-mins 30 \
    --ntfy-topic p79-exp-dgx-spark \
    --state-file "${WD_STATE}" \
    --aggregate-prefix "${AGGREGATE_PREFIX}" \
    --glm-config .auth/glm \
    --digest-dir "${RUN_DIR}/analysis/digest" \
    >> "${WD_LOG}" 2>&1 < /dev/null &
  disown
  sleep 2
  if pgrep -f "experiment_watchdog.*${RUN_ID}" > /dev/null; then
    echo "[phantom_dom] watchdog pid=$(pgrep -f "experiment_watchdog.*${RUN_ID}" | head -1)"
  else
    echo "[phantom_dom][error] watchdog failed to start, see ${WD_LOG}" >&2
    exit 1
  fi
fi

echo ""
echo "[phantom_dom] OK — ${CFG_NAME} (${BENCHMARK}/${SITE}) running"
echo "  runner log:   ${RUNNER_LOG:-<existing>}"
echo "  watchdog log: ${WD_LOG}"
