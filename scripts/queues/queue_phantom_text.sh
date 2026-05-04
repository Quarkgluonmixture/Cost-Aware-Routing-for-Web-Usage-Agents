#!/usr/bin/env bash
# queue_phantom_text.sh — Launch P-text (DOM prompt + [SOM_MARKS] text + no image).
#
# P-text (formerly Phantom-DOM, §103): DOM prompt + SoM marks 文本 + 无图（control for
# axis 2 prompt effect, vs Phantom-SoM which uses SoM prompt). 同样的 [SOM_MARKS] 文本，
# 区别仅在 system prompt.
#
# Naming:
#   - mode value: phantom_text (current canonical; phantom_dom kept as legacy alias in
#     agents+som.py for paper-grade run dirs already on disk)
#   - paper-facing label: P-text
#   - this script lives at scripts/queues/queue_phantom_text.sh
#   - back-compat symlink: scripts/queues/queue_phantom_dom.sh -> queue_phantom_text.sh
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
#   bash scripts/queues/queue_phantom_text.sh <baseline> <site> [benchmark]
#   - baseline:  B0 | B1
#   - site:      classifieds | reddit | shopping (vwa) | shopping_admin (wa-only)
#   - benchmark: vwa (默认) | wa
#
# Examples:
#   bash scripts/queues/queue_phantom_text.sh B0 reddit                      # VWA reddit
#   bash scripts/queues/queue_phantom_text.sh B1 shopping                    # VWA shopping
#   bash scripts/queues/queue_phantom_text.sh B0 shopping_admin wa           # WA shopping_admin
#   RESET_BEFORE=1 bash scripts/queues/queue_phantom_text.sh B0 shopping     # with reset

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

# Build config name. Prefer the new phantom_text YAML; fall back to phantom_dom
# YAML if the rename hasn't propagated yet (e.g. live run still on disk under
# the legacy name).
# VWA: exp_v2_<baseline>_phantom_text_<site>.yaml
# WA:  exp_v2_<baseline>_phantom_text_wa_<site>.yaml
CFG_BASE_NEW="${BASELINE}_phantom_text"
CFG_BASE_LEGACY="${BASELINE}_phantom_dom"
[[ "${BENCHMARK}" == "wa" ]] && CFG_BASE_NEW="${CFG_BASE_NEW}_wa" && CFG_BASE_LEGACY="${CFG_BASE_LEGACY}_wa"
CFG_NAME_NEW="${CFG_BASE_NEW}_${SITE}"
CFG_NAME_LEGACY="${CFG_BASE_LEGACY}_${SITE}"
CONFIG_NEW="${REPO_DIR}/configs/exp_v2_${CFG_NAME_NEW}.yaml"
CONFIG_LEGACY="${REPO_DIR}/configs/exp_v2_${CFG_NAME_LEGACY}.yaml"

if [[ -f "${CONFIG_NEW}" ]]; then
  CONFIG="${CONFIG_NEW}"
  CFG_NAME="${CFG_NAME_NEW}"
elif [[ -f "${CONFIG_LEGACY}" ]]; then
  CONFIG="${CONFIG_LEGACY}"
  CFG_NAME="${CFG_NAME_LEGACY}"
  echo "[phantom_text] using legacy phantom_dom config: ${CONFIG_LEGACY}"
else
  echo "[phantom_text][error] Config not found: ${CONFIG_NEW} or ${CONFIG_LEGACY}" >&2
  exit 1
fi

# condition_id stays as phantom_dom_router_0 for paper-grade compatibility —
# existing run dirs (B0_phantom_text_*) all contain phase1_phantom_dom_router_0/.
# Newly-introduced phantom_text YAMLs may declare phantom_text_router_0 instead;
# in that case the runner reads the condition_id from YAML and this constant is
# only used for watchdog targeting.
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
        echo "[phantom_text] Loaded PROXY_API_KEY from ${AUTH_FILE}"
      else
        echo "[phantom_text][error] ${AUTH_FILE} 存在但无 rp_ key" >&2; exit 1
      fi
    else
      echo "[phantom_text][error] ${AUTH_FILE} 不存在，且 PROXY_API_KEY 未设置" >&2; exit 1
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

# Resume detection: try the modern phantom_text run-dir prefix first, then fall
# back to legacy phantom_dom prefix (some live runs still use the old name).
RUN_PREFIX_NEW="${BASELINE}_phantom_text"
RUN_PREFIX_LEGACY="${BASELINE}_phantom_dom"
[[ "${BENCHMARK}" == "wa" ]] && RUN_PREFIX_NEW="${RUN_PREFIX_NEW}_wa" && RUN_PREFIX_LEGACY="${RUN_PREFIX_LEGACY}_wa"
RUN_PREFIX_NEW="${RUN_PREFIX_NEW}_${SITE}"
RUN_PREFIX_LEGACY="${RUN_PREFIX_LEGACY}_${SITE}"

EXISTING="$(ls -dt "${PHASE_DIR}/${RUN_PREFIX_NEW}_"[0-9]* 2>/dev/null | head -1 || true)"
if [[ -z "${EXISTING}" ]]; then
  EXISTING="$(ls -dt "${PHASE_DIR}/${RUN_PREFIX_LEGACY}_"[0-9]* 2>/dev/null | head -1 || true)"
fi

if [[ -n "${EXISTING}" ]]; then
  RUN_ID="$(basename "${EXISTING}")"
  echo "[phantom_text] resuming existing run_id=${RUN_ID}"
else
  # Fresh run: name follows the active config (phantom_text vs phantom_dom).
  RUN_ID="${CFG_NAME}_${TS_DATE}"
  echo "[phantom_text] new run_id=${RUN_ID}"
fi

RUN_DIR="${PHASE_DIR}/${RUN_ID}"
echo "[phantom_text] config=${CONFIG}"
echo "[phantom_text] run_dir=${RUN_DIR}"
echo "[phantom_text] condition=${COND_ID}"

# ---------- 检查 runner 是否已在跑 ----------
if pgrep -f "run_experiment.py.*${RUN_ID}" > /dev/null; then
  echo "[phantom_text] runner for ${RUN_ID} already running, skipping spawn"
  echo "[phantom_text] (RESET_BEFORE skipped — runner already attached to current site state)"
else
  # ---------- Optional: site reset before launch ----------
  # IMPORTANT: reset is AFTER the idempotent runner check — resetting while
  # a runner is attached destroys site state under it (race condition fixed
  # 2026-04-28 — see 实验笔记 §104).
  if [[ "${RESET_BEFORE:-0}" == "1" && "${BENCHMARK}" != "wa" ]]; then
    if [[ -f "${REPO_DIR}/scripts/maintenance/reset_vwa_sites.sh" ]]; then
      # shellcheck disable=SC1091
      source "${REPO_DIR}/scripts/maintenance/reset_vwa_sites.sh"
      echo "[phantom_text] RESET_BEFORE=1 → resetting site=${SITE}..."
      if reset_vwa_sites "${SITE}" "phantom_text_${SITE}"; then
        echo "[phantom_text] reset OK; sleeping 15s for site to settle..."
        sleep 15
        echo "[phantom_text] refreshing .auth/${SITE}_state.json post-reset..."
        if "${PYTHON_BIN}" -c "
import sys
sys.path.insert(0, '${REPO_DIR}')
from pathlib import Path
from p79.utils.auth_refresh import refresh_site_auth
sys.exit(0 if refresh_site_auth('${SITE}', Path('${REPO_DIR}/.auth')) else 1)
" 2>&1; then
          echo "[phantom_text] auth refresh OK — runner task=0 will be LOGGED IN"
        else
          echo "[phantom_text][warn] post-reset auth refresh failed; watchdog will retry reactively after streak=3" >&2
        fi
      else
        rc=$?
        echo "[phantom_text][error] reset failed (rc=${rc}); aborting to preserve paper-grade integrity." >&2
        echo "[phantom_text][error] To bypass reset (paper-grade dirty), explicitly set RESET_BEFORE=0." >&2
        exit 1
      fi
    else
      echo "[phantom_text][error] reset_vwa_sites.sh not found but RESET_BEFORE=1; aborting." >&2
      echo "[phantom_text][error] To bypass reset (paper-grade dirty), explicitly set RESET_BEFORE=0." >&2
      exit 1
    fi
  elif [[ "${RESET_BEFORE:-0}" == "1" ]]; then
    echo "[phantom_text] RESET_BEFORE=1 but BENCHMARK=wa — WA reset+auth refresh uses different mechanism, skipping"
  fi

  RUNNER_LOG="${LOG_DIR}/${CFG_NAME}_resume_${TS_FULL}.log"
  echo "[phantom_text] launching runner → ${RUNNER_LOG}"
  setsid nohup "${PYTHON_BIN}" scripts/run_experiment.py \
    --config "${CONFIG}" \
    --run_id "${RUN_ID}" \
    --log_path "${RUNNER_LOG}" \
    > /dev/null 2>&1 < /dev/null &
  disown
  sleep 3
  if pgrep -f "run_experiment.py.*${RUN_ID}" > /dev/null; then
    echo "[phantom_text] runner pid=$(pgrep -f "run_experiment.py.*${RUN_ID}" | head -1)"
  else
    echo "[phantom_text][error] runner failed to start, see ${RUNNER_LOG}" >&2
    [[ -f "${RUNNER_LOG}" ]] && tail -20 "${RUNNER_LOG}" >&2
    exit 1
  fi
fi

# ---------- watchdog 启动 ----------
WD_STATE="${LOG_DIR}/exp_watchdog_${RUN_ID}_v2.state.json"
WD_LOG="${LOG_DIR}/exp_watchdog_${RUN_ID}_v2.log"
# Unified per-baseline aggregate gallery — all B0/B1 modes share one URL
# (B0_3mode/ or B1_3mode/). gallery script's "baseline alias" semantics
# expands this to match all B0_*/B1_* runs (3mode + dom + phantom variants).
AGGREGATE_PREFIX="${BASELINE}_3mode"

# Runner PID for watchdog self-exit — watchdog auto-exits when this PID dies
# AND condition_summary_v2.json present. Prevents init-orphan idle loops.
RUNNER_PID=$(pgrep -f "run_experiment.py.*${RUN_ID}" | head -1)

if pgrep -f "experiment_watchdog.*${RUN_ID}" > /dev/null; then
  echo "[phantom_text] watchdog for ${RUN_ID} already running, skipping spawn"
else
  echo "[phantom_text] launching watchdog → ${WD_LOG} (runner pid=${RUNNER_PID:-unknown})"
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
    echo "[phantom_text] watchdog pid=$(pgrep -f "experiment_watchdog.*${RUN_ID}" | head -1)"
  else
    echo "[phantom_text][error] watchdog failed to start, see ${WD_LOG}" >&2
    exit 1
  fi
fi

echo ""
echo "[phantom_text] OK — ${CFG_NAME} (${BENCHMARK}/${SITE}) running"
echo "  runner log:   ${RUNNER_LOG:-<existing>}"
echo "  watchdog log: ${WD_LOG}"
