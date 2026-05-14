#!/usr/bin/env bash
# queue_baseline.sh — 启动 baseline 实验 (dom / som / vision) + 自动 watchdog
#
# Baseline modes (Phase 1 表征筛选):
#   dom    — viewport-only AXTree (no image)
#   som    — [SOM_MARKS] 文本 + 带框截图
#   vision — 裸截图 (no DOM/AXTree)
#
# 这个脚本统一处理:
#   - PROXY_API_KEY 从 .auth/qwen_api 加载 (B0 用)
#   - VWA 远程 host env 加载
#   - CUDA workaround env (DGX Spark sm_121)
#   - WIKIPEDIA ZIM 版本
#   - runner + watchdog 一起启动，已存在则跳过 (idempotent)
#   - RESET 在 idempotent check 之后执行 (防 race — 见笔记 §104 audit)
#
# 用法:
#   bash scripts/queues/queue_baseline.sh <baseline> <mode> <site> [benchmark]
#   - baseline:  B0 | B1
#   - mode:      dom | som | vision
#   - site:      classifieds | reddit | shopping | shopping_admin
#   - benchmark: vwa (默认) | wa
#
# 例:
#   bash scripts/queues/queue_baseline.sh B0 dom shopping            # B0 DOM-only VWA shopping
#   bash scripts/queues/queue_baseline.sh B1 som reddit              # B1 SoM VWA reddit
#   bash scripts/queues/queue_baseline.sh B0 vision shopping wa      # B0 vision WA shopping
#
# Reset:
#   RESET_BEFORE=1 bash ...  →  reset site (VWA only) AFTER idempotent check
#
# Required configs (must exist before launch):
#   VWA:  configs/exp_v2_<baseline>_<mode>_<site>.yaml
#   WA:   configs/exp_v2_<baseline>_<mode>_wa_<site>.yaml

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_DIR}"

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <baseline:B0|B1> <mode:dom|som|vision> <site> [benchmark:vwa|wa]" >&2
  echo "  e.g. bash $0 B0 dom shopping" >&2
  echo "       bash $0 B0 vision shopping wa" >&2
  exit 2
fi

BASELINE="$1"; MODE="$2"; SITE="$3"
BENCHMARK="${4:-vwa}"

# Validation
if [[ "${BASELINE}" != "B0" && "${BASELINE}" != "B1" ]]; then
  echo "Invalid baseline: ${BASELINE} (expected B0 or B1)" >&2; exit 2
fi
if [[ "${MODE}" != "dom" && "${MODE}" != "som" && "${MODE}" != "vision" ]]; then
  echo "Invalid mode: ${MODE} (expected dom/som/vision)" >&2; exit 2
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
# VWA: exp_v2_<baseline>_<mode>_<site>.yaml
# WA:  exp_v2_<baseline>_<mode>_wa_<site>.yaml
CFG_NAME="${BASELINE}_${MODE}_${SITE}"
[[ "${BENCHMARK}" == "wa" ]] && CFG_NAME="${BASELINE}_${MODE}_wa_${SITE}"
CONFIG="${REPO_DIR}/configs/exp_v2_${CFG_NAME}.yaml"

if [[ ! -f "${CONFIG}" ]]; then
  echo "[baseline][error] Config not found: ${CONFIG}" >&2
  echo "  Single-mode baseline config 必须先创建 (template: exp_v2_B0_dom_shopping.yaml)" >&2
  echo "  或参考 configs/exp_v2_<baseline>_3mode_<site>.yaml 调整 observation_mode 单 list" >&2
  exit 1
fi

# Condition id: phase1_<mode>_router_0
COND_ID="phase1_${MODE}_router_0"

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
        echo "[baseline] Loaded PROXY_API_KEY from ${AUTH_FILE}"
      else
        echo "[baseline][error] ${AUTH_FILE} 存在但无 rp_ key" >&2; exit 1
      fi
    else
      echo "[baseline][error] ${AUTH_FILE} 不存在，且 PROXY_API_KEY 未设置" >&2; exit 1
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

# FORCE_NEW=1 (paper-grade fresh rerun): always timestamped run_id, never resume-glob.
# Prevents silently reusing pre-fix archived run dirs (codex stress v6 C1, 2026-05-14).
# Crash recovery: omit FORCE_NEW to allow resume-by-glob of an in-progress run.
if [[ "${FORCE_NEW:-0}" == "1" ]]; then
  RUN_ID="${CFG_NAME}_${TS_FULL}"
  echo "[baseline] FORCE_NEW=1 → fresh timestamped run_id=${RUN_ID} (resume-glob skipped)"
else
  EXISTING="$(ls -dt "${PHASE_DIR}/${CFG_NAME}_"[0-9]* 2>/dev/null | head -1 || true)"
  if [[ -n "${EXISTING}" ]]; then
    RUN_ID="$(basename "${EXISTING}")"
    echo "[baseline] resuming existing run_id=${RUN_ID}"
  else
    RUN_ID="${CFG_NAME}_${TS_DATE}"
    echo "[baseline] new run_id=${RUN_ID}"
  fi
fi

RUN_DIR="${PHASE_DIR}/${RUN_ID}"
echo "[baseline] config=${CONFIG}"
echo "[baseline] run_dir=${RUN_DIR}"
echo "[baseline] condition=${COND_ID}"

# ---------- 检查 runner 是否已在跑 ----------
if pgrep -f "run_experiment.py.*${RUN_ID}" > /dev/null; then
  echo "[baseline] runner for ${RUN_ID} already running, skipping spawn"
  echo "[baseline] (RESET_BEFORE skipped — runner already attached to current site state)"
else
  # ---------- Optional: site reset before launch ----------
  # IMPORTANT: reset is AFTER the idempotent runner check — resetting while
  # a runner is attached destroys site state under it (race condition fixed
  # 2026-04-28 — see 实验笔记 §104).
  if [[ "${RESET_BEFORE:-0}" == "1" && "${BENCHMARK}" != "wa" ]]; then
    if [[ -f "${REPO_DIR}/scripts/maintenance/reset_vwa_sites.sh" ]]; then
      # shellcheck disable=SC1091
      source "${REPO_DIR}/scripts/maintenance/reset_vwa_sites.sh"
      echo "[baseline] RESET_BEFORE=1 → resetting site=${SITE}..."
      if reset_vwa_sites "${SITE}" "baseline_${MODE}_${SITE}"; then
        echo "[baseline] reset OK; sleeping 15s for site to settle..."
        sleep 15
        # Refresh .auth/<site>_state.json post-reset — server-side session was wiped,
        # so the runner's first task would otherwise hit NOT-LOGGED-IN (watchdog only
        # reactively refreshes after streak=3, costing 3 dirty episodes).
        echo "[baseline] refreshing .auth/${SITE}_state.json post-reset..."
        if "${PYTHON_BIN}" -c "
import sys
sys.path.insert(0, '${REPO_DIR}')
from pathlib import Path
from p79.utils.auth_refresh import refresh_site_auth
sys.exit(0 if refresh_site_auth('${SITE}', Path('${REPO_DIR}/.auth')) else 1)
" 2>&1; then
          echo "[baseline] auth refresh OK — runner task=0 will be LOGGED IN"
        else
          echo "[baseline][warn] post-reset auth refresh failed; watchdog will retry reactively after streak=3" >&2
        fi
      else
        rc=$?
        echo "[baseline][error] reset failed (rc=${rc}); aborting to preserve paper-grade integrity." >&2
        echo "[baseline][error] To bypass reset (paper-grade dirty), explicitly set RESET_BEFORE=0." >&2
        exit 1
      fi
    else
      echo "[baseline][error] reset_vwa_sites.sh not found but RESET_BEFORE=1; aborting." >&2
      echo "[baseline][error] To bypass reset (paper-grade dirty), explicitly set RESET_BEFORE=0." >&2
      exit 1
    fi
  elif [[ "${RESET_BEFORE:-0}" == "1" ]]; then
    echo "[baseline] RESET_BEFORE=1 but BENCHMARK=wa — WA reset+auth refresh uses different mechanism, skipping"
  fi

  RUNNER_LOG="${LOG_DIR}/${CFG_NAME}_runner_${TS_FULL}.log"
  echo "[baseline] launching runner → ${RUNNER_LOG}"
  setsid nohup "${PYTHON_BIN}" scripts/run_experiment.py \
    --config "${CONFIG}" \
    --run_id "${RUN_ID}" \
    --log_path "${RUNNER_LOG}" \
    > "${RUNNER_LOG}" 2>&1 < /dev/null &
  disown
  sleep 3
  if pgrep -f "run_experiment.py.*${RUN_ID}" > /dev/null; then
    echo "[baseline] runner pid=$(pgrep -f "run_experiment.py.*${RUN_ID}" | head -1)"
  else
    echo "[baseline][error] runner failed to start — check ${RUNNER_LOG}" >&2
    tail -20 "${RUNNER_LOG}" >&2
    exit 1
  fi
fi

# ---------- 启动 watchdog (idempotent) ----------
WATCHDOG_LOG="${LOG_DIR}/exp_watchdog_${RUN_ID}_v2.log"
WATCHDOG_STATE="${LOG_DIR}/exp_watchdog_${RUN_ID}_v2.state.json"
WATCHDOG_DIGEST="${RUN_DIR}/analysis/digest"

# Runner PID for watchdog self-exit — watchdog auto-exits when this PID dies
# AND condition_summary_v2.json present. Prevents init-orphan idle loops.
RUNNER_PID=$(pgrep -f "run_experiment.py.*${RUN_ID}" | head -1)

if pgrep -f "experiment_watchdog.*${RUN_ID}" > /dev/null; then
  echo "[baseline] watchdog for ${RUN_ID} already running, skipping spawn"
else
  echo "[baseline] launching watchdog → ${WATCHDOG_LOG} (runner pid=${RUNNER_PID:-unknown})"
  setsid nohup "${PYTHON_BIN}" -u scripts/maintenance/experiment_watchdog.py \
    --run-dir "${RUN_DIR}" \
    --condition "${COND_ID}" \
    --poll-secs 30 \
    --idle-alert-mins 30 \
    --ntfy-topic p79-exp-dgx-spark \
    --state-file "${WATCHDOG_STATE}" \
    --aggregate-prefix "${BASELINE}_3mode" \
    --glm-config .auth/glm \
    --digest-dir "${WATCHDOG_DIGEST}" \
    ${RUNNER_PID:+--runner-pid "${RUNNER_PID}"} \
    > "${WATCHDOG_LOG}" 2>&1 < /dev/null &
  disown
  sleep 2
  if pgrep -f "experiment_watchdog.*${RUN_ID}" > /dev/null; then
    echo "[baseline] watchdog pid=$(pgrep -f "experiment_watchdog.*${RUN_ID}" | head -1)"
  else
    # codex stress v6 C5: watchdog failure is now FATAL for paper-grade launch.
    # Without watchdog, mid-run auth drift / crashes produce silent missing data
    # (no reactive auth_refresh, no idle alert, no auto-clean). Combined with the
    # queue_chain completion sentinel (C3), a watchdog-less cell is paper-grade-dirty.
    echo "[baseline][error] watchdog failed to start — check ${WATCHDOG_LOG}" >&2
    echo "[baseline][error] aborting: paper-grade launch requires watchdog (auth refresh + auto-clean)." >&2
    exit 1
  fi
fi

echo
echo "[baseline] OK — ${BASELINE}_${MODE}_${SITE} (${BENCHMARK}/${SITE}) running"
echo "  run_id=${RUN_ID}"
echo "  runner log:   ${RUNNER_LOG:-<existing>}"
echo "  watchdog log: ${WATCHDOG_LOG}"
