#!/usr/bin/env bash
# queue_phantom.sh — 启动 phantom 系列实验 (phantom_som + phantom_dom on VWA + WA) + 自动 watchdog
#
# Phantom-SoM (§102): SoM prompt + SoM marks 文本 + 无图（mirage effect）
# Phantom-DOM (§103): DOM prompt + SoM marks 文本 + 无图（ablation: prompt vs representation）
#
# 这个脚本统一处理:
#   - PROXY_API_KEY 从 .auth/qwen_api 加载 (B0 用)
#   - VWA 远程 host env 加载
#   - CUDA workaround env (DGX Spark sm_121)
#   - WIKIPEDIA ZIM 版本（已在 tasks.py 默认 2025-08）
#   - runner + watchdog 一起启动，已存在则跳过 (idempotent)
#
# 用法:
#   bash scripts/queues/queue_phantom.sh <baseline> <mode> <site> [benchmark]
#   - baseline: B0 | B1
#   - mode: som | dom (phantom_<mode>)
#   - site: classifieds | reddit | shopping | shopping_admin (后两个仅 wa)
#   - benchmark: vwa (默认) | wa
#
# 例:
#   bash scripts/queues/queue_phantom.sh B0 som reddit            # phantom_som VWA reddit
#   bash scripts/queues/queue_phantom.sh B0 dom reddit            # phantom_dom VWA reddit (ablation)
#   bash scripts/queues/queue_phantom.sh B0 som shopping wa       # phantom_som WA shopping
#   bash scripts/queues/queue_phantom.sh B0 dom shopping_admin wa # phantom_dom WA shopping_admin

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_DIR}"

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <baseline:B0|B1> <mode:som|dom> <site> [benchmark:vwa|wa]" >&2
  echo "  e.g. bash $0 B0 som reddit" >&2
  echo "       bash $0 B0 dom shopping wa" >&2
  exit 2
fi

BASELINE="$1"; MODE="$2"; SITE="$3"
BENCHMARK="${4:-vwa}"

# Validation
if [[ "${BASELINE}" != "B0" && "${BASELINE}" != "B1" ]]; then
  echo "Invalid baseline: ${BASELINE} (expected B0 or B1)" >&2; exit 2
fi
if [[ "${MODE}" != "som" && "${MODE}" != "dom" ]]; then
  echo "Invalid mode: ${MODE} (expected som or dom)" >&2; exit 2
fi
if [[ "${BENCHMARK}" != "vwa" && "${BENCHMARK}" != "wa" ]]; then
  echo "Invalid benchmark: ${BENCHMARK} (expected vwa or wa)" >&2; exit 2
fi
if [[ "${BENCHMARK}" == "vwa" && "${SITE}" != "classifieds" && "${SITE}" != "reddit" && "${SITE}" != "shopping" ]]; then
  echo "Invalid VWA site: ${SITE} (expected classifieds/reddit/shopping)" >&2; exit 2
fi
if [[ "${BENCHMARK}" == "wa" && "${SITE}" != "reddit" && "${SITE}" != "shopping" && "${SITE}" != "shopping_admin" ]]; then
  echo "Invalid WA site: ${SITE} (expected reddit/shopping/shopping_admin)" >&2; exit 2
fi

# Build config name
# VWA: exp_v2_<baseline>_phantom_<site>.yaml (som) / exp_v2_<baseline>_phantom_dom_<site>.yaml (dom)
# WA:  exp_v2_<baseline>_phantom_wa_<site>.yaml / exp_v2_<baseline>_phantom_dom_wa_<site>.yaml
CFG_NAME="${BASELINE}_phantom"
[[ "${MODE}" == "dom" ]] && CFG_NAME="${BASELINE}_phantom_dom"
[[ "${BENCHMARK}" == "wa" ]] && CFG_NAME="${CFG_NAME}_wa"
CFG_NAME="${CFG_NAME}_${SITE}"
CONFIG="${REPO_DIR}/configs/exp_v2_${CFG_NAME}.yaml"

if [[ ! -f "${CONFIG}" ]]; then
  echo "Config not found: ${CONFIG}" >&2; exit 1
fi

# Condition id pattern: phase1_phantom_som_router_0 or phase1_phantom_dom_router_0
COND_ID="phase1_phantom_${MODE}_router_0"

PYTHON_BIN="${REPO_DIR}/.venv/bin/python3"
LOG_DIR="${REPO_DIR}/logs"
mkdir -p "${LOG_DIR}"

# ---------- Optional: site reset before launch ----------
# Set RESET_BEFORE=1 to reset the target site (via reset_vwa_sites.sh) before
# starting the runner. Useful when previous condition on same site may have
# drifted state (e.g. phantom_som -> phantom_dom on shopping). Default: skip.
if [[ "${RESET_BEFORE:-0}" == "1" ]]; then
  if [[ -f "${REPO_DIR}/scripts/maintenance/reset_vwa_sites.sh" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_DIR}/scripts/maintenance/reset_vwa_sites.sh"
    echo "[phantom] RESET_BEFORE=1 → resetting site=${SITE}..."
    if reset_vwa_sites "${SITE}" "phantom_${MODE}_${SITE}"; then
      echo "[phantom] reset OK; sleeping 15s for site to settle..."
      sleep 15
    else
      echo "[phantom][warn] reset failed (rc=$?); continuing anyway" >&2
    fi
  else
    echo "[phantom][warn] reset_vwa_sites.sh not found; skipping reset" >&2
  fi
fi

# ---------- DGX Spark CUDA workaround ----------
export PYTORCH_NVML_BASED_CUDA_CHECK=1
export CUDA_MPS_PIPE_DIRECTORY=""
export CUDA_MPS_LOG_DIRECTORY=""

# ---------- VWA 远程站点 env ----------
if [[ -f "${REPO_DIR}/scripts/vwa_env_remote.sh" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_DIR}/scripts/vwa_env_remote.sh"
fi

# ---------- WIKIPEDIA ZIM 版本（§81）----------
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
        echo "[phantom] Loaded PROXY_API_KEY from ${AUTH_FILE}"
      else
        echo "[phantom][error] ${AUTH_FILE} 存在但无 rp_ key" >&2; exit 1
      fi
    else
      echo "[phantom][error] ${AUTH_FILE} 不存在，且 PROXY_API_KEY 未设置" >&2; exit 1
    fi
  fi
fi

# ---------- 决定 run_id + run_dir ----------
TS_DATE="$(date +%Y%m%d)"
TS_FULL="$(date +%Y%m%d_%H%M%S)"
# Results root differs by benchmark
if [[ "${BENCHMARK}" == "wa" ]]; then
  PHASE_DIR="${REPO_DIR}/results/webarena/phase1"
else
  PHASE_DIR="${REPO_DIR}/results/visualwebarena/phase1"
fi

# Check for existing run_id
EXISTING="$(ls -dt "${PHASE_DIR}/${CFG_NAME}_"* 2>/dev/null | head -1 || true)"
if [[ -n "${EXISTING}" ]]; then
  RUN_ID="$(basename "${EXISTING}")"
  echo "[phantom] resuming existing run_id=${RUN_ID}"
else
  # Special case for legacy B0_phantom_reddit (run_reddit_1777238854_ef9c4b)
  if [[ "${CFG_NAME}" == "B0_phantom_reddit" ]] \
     && [[ -d "${PHASE_DIR}/run_reddit_1777238854_ef9c4b" ]]; then
    RUN_ID="run_reddit_1777238854_ef9c4b"
    echo "[phantom] resuming legacy run_id=${RUN_ID}"
  else
    RUN_ID="${CFG_NAME}_${TS_DATE}"
    echo "[phantom] new run_id=${RUN_ID}"
  fi
fi

RUN_DIR="${PHASE_DIR}/${RUN_ID}"
echo "[phantom] config=${CONFIG}"
echo "[phantom] run_dir=${RUN_DIR}"
echo "[phantom] condition=${COND_ID}"

# ---------- 检查 runner 是否已在跑 ----------
if pgrep -f "run_experiment.py.*${RUN_ID}" > /dev/null; then
  echo "[phantom] runner for ${RUN_ID} already running, skipping spawn"
else
  RUNNER_LOG="${LOG_DIR}/${CFG_NAME}_resume_${TS_FULL}.log"
  echo "[phantom] launching runner → ${RUNNER_LOG}"
  setsid nohup "${PYTHON_BIN}" scripts/run_experiment.py \
    --config "${CONFIG}" \
    --run_id "${RUN_ID}" \
    --log_path "${RUNNER_LOG}" \
    > /dev/null 2>&1 < /dev/null &
  disown
  sleep 3
  if pgrep -f "run_experiment.py.*${RUN_ID}" > /dev/null; then
    echo "[phantom] runner pid=$(pgrep -f "run_experiment.py.*${RUN_ID}" | head -1)"
  else
    echo "[phantom][error] runner failed to start, see ${RUNNER_LOG}" >&2
    [[ -f "${RUNNER_LOG}" ]] && tail -20 "${RUNNER_LOG}" >&2
    exit 1
  fi
fi

# ---------- watchdog 启动 ----------
WD_STATE="${LOG_DIR}/exp_watchdog_${RUN_ID}_v2.state.json"
WD_LOG="${LOG_DIR}/exp_watchdog_${RUN_ID}_v2.log"
# aggregate prefix mirrors config name (with _wa_ kept distinct)
AGGREGATE_PREFIX="${CFG_NAME%_${SITE}}"  # strip trailing _<site>
# Edge case: prefix should be like "B0_phantom" or "B0_phantom_dom" or "B0_phantom_wa"
# strip _<site> works for cls/red/shop; for shopping_admin too works (suffix removed).

if pgrep -f "experiment_watchdog.*${RUN_ID}" > /dev/null; then
  echo "[phantom] watchdog for ${RUN_ID} already running, skipping spawn"
else
  echo "[phantom] launching watchdog → ${WD_LOG}"
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
    echo "[phantom] watchdog pid=$(pgrep -f "experiment_watchdog.*${RUN_ID}" | head -1)"
  else
    echo "[phantom][error] watchdog failed to start, see ${WD_LOG}" >&2
    exit 1
  fi
fi

echo ""
echo "[phantom] OK — ${CFG_NAME} (${BENCHMARK}/${SITE}) running"
echo "  runner log:   ${RUNNER_LOG:-<existing>}"
echo "  watchdog log: ${WD_LOG}"
