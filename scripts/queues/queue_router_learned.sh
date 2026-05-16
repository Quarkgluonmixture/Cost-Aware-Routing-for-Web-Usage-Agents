#!/usr/bin/env bash
# queue_router_learned.sh — Pass-2 learned router launch wrapper (per-cell).
# v7 walk-back 2026-05-16: paper-1 §6 single-router learned-only design.
# Mirrors queue_baseline.sh structure; differs in:
#   - config name: exp_v2_<baseline>_router_learned_<site>.yaml (6 configs total)
#   - condition_id: phase1_learned_router (single per cell, vs per-mode in baseline)
#   - obs_mode sentinel: "learned" (runner dispatches through LR predictor at runtime)
#
# ⚠️ RUNTIME LR INTEGRATION TODO: runner must accept observation_mode="learned"
#    and route each task through a trained LR predictor. See proposals_v7.md §3
#    + pending TODO in p79/experiment/runner/main.py + scripts/analysis/train_l1_router.py
#    (LR training pipeline producer, separate session). Without this, launch will
#    fail when the agent tries to instantiate observation_mode="learned" → no
#    matching observation processor.
#
# Pre-launch gate: REQUIRE_LR_RUNTIME=1 (default; abort if obs_mode="learned" not
# yet wired in runner). To bypass gate for scaffolding/dry-run, set REQUIRE_LR_RUNTIME=0.
#
# 用法:
#   bash scripts/queues/queue_router_learned.sh <baseline:B0|B1|B2> <site:classifieds|reddit>
#
# Reset:
#   RESET_BEFORE=1 bash ... → reset site (VWA only) after idempotent check
#
# Required configs (per cell, must exist before launch):
#   configs/exp_v2_<baseline>_router_learned_<site>.yaml

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_DIR}"

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <baseline:B0|B1|B2> <site:classifieds|reddit>" >&2
  echo "  e.g. bash $0 B0 classifieds" >&2
  exit 2
fi

BASELINE="$1"; SITE="$2"

# Validation
if [[ "${BASELINE}" != "B0" && "${BASELINE}" != "B1" && "${BASELINE}" != "B2" ]]; then
  echo "Invalid baseline: ${BASELINE} (expected B0, B1 or B2)" >&2; exit 2
fi
if [[ "${SITE}" != "classifieds" && "${SITE}" != "reddit" ]]; then
  echo "Invalid VWA site for Pass-2 router: ${SITE} (expected classifieds or reddit; shop is Phase 1b)" >&2; exit 2
fi

# Build config name (per-cell, v7 walk-back)
CFG_NAME="${BASELINE}_router_learned_${SITE}"
CONFIG="${REPO_DIR}/configs/exp_v2_${CFG_NAME}.yaml"

if [[ ! -f "${CONFIG}" ]]; then
  echo "[router][error] Config not found: ${CONFIG}" >&2
  echo "  Run-1 v7 router configs were generated 2026-05-16; if missing, check git status." >&2
  exit 1
fi

# v7 TODO gate: LR runtime integration must be implemented before launch.
# Gate evaluates by grep'ing p79/experiment/runner/ for "learned" obs_mode handling.
if [[ "${REQUIRE_LR_RUNTIME:-1}" == "1" ]]; then
  if ! grep -rqE "(observation_mode.*==.*\"?learned\"?|obs_mode.*==.*\"?learned\"?|_dispatch_lr|lr_model_path)" \
      "${REPO_DIR}/p79/experiment/runner/" 2>/dev/null; then
    echo "[router][error] LR runtime integration NOT wired in p79/experiment/runner/." >&2
    echo "  Without runner support for observation_mode=\"learned\", launch will fail." >&2
    echo "  Either (a) implement LR dispatch in runner (see proposals_v7.md §3 TODO)" >&2
    echo "      or (b) set REQUIRE_LR_RUNTIME=0 to bypass for scaffolding/dry-run." >&2
    exit 1
  fi
fi

# Condition id: phase1_learned_router (single per cell, NOT per-mode)
COND_ID="phase1_learned_router"

PYTHON_BIN="${REPO_DIR}/.venv/bin/python3"
LOG_DIR="${REPO_DIR}/logs"
mkdir -p "${LOG_DIR}"

# ---------- DGX Spark / A100 CUDA env (mirror queue_baseline) ----------
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

# ---------- TZ ALIGN (BUG-6 fix, mirror queue_baseline) ----------
export QUARK_TZ="${QUARK_TZ:-Europe/London}"

# ---------- BUG-2 preflight: assert all site URLs are local on A100 ----------
if [[ "$(hostname)" == *condense* ]] || [[ -d /home/ubuntu/workspace/p79 ]]; then
  for _v in CLASSIFIEDS REDDIT SHOPPING WIKIPEDIA; do
    case "${!_v:-}" in
      *localhost*|*127.0.0.1*|"") ;;
      *) echo "✗ FATAL preflight: \$${_v}=${!_v} not local on A100 host; refusing launch" >&2; exit 2 ;;
    esac
  done
  unset _v
fi

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
        echo "[router] Loaded PROXY_API_KEY from ${AUTH_FILE}"
      else
        echo "[router][error] ${AUTH_FILE} 存在但无 rp_ key" >&2; exit 1
      fi
    else
      echo "[router][error] ${AUTH_FILE} 不存在，且 PROXY_API_KEY 未设置" >&2; exit 1
    fi
  fi
fi

# ---------- Pre-launch: LR model artifact must exist ----------
LR_MODEL_PATH="${REPO_DIR}/results/phantom_paper/l1_router/${BASELINE}_${SITE}_lr.pkl"
if [[ ! -f "${LR_MODEL_PATH}" ]]; then
  echo "[router][warn] LR model artifact not found: ${LR_MODEL_PATH}" >&2
  echo "  Pass-2 router requires LR trained on Pass-1 baseline per-task oracle labels." >&2
  echo "  Run: python3 scripts/analysis/train_l1_router.py --baseline ${BASELINE} --site ${SITE} --out ${LR_MODEL_PATH}" >&2
  echo "  (train_l1_router.py is TODO; pending separate session)" >&2
  if [[ "${ALLOW_NO_LR_MODEL:-0}" != "1" ]]; then
    echo "  Set ALLOW_NO_LR_MODEL=1 to bypass (router will fail at runtime)." >&2
    exit 1
  fi
fi

# ---------- 决定 run_id + run_dir ----------
TS_DATE="$(date +%Y%m%d)"
TS_FULL="$(date +%Y%m%d_%H%M%S)"
PHASE_DIR="${REPO_DIR}/results/visualwebarena/phase1"

if [[ "${FORCE_NEW:-0}" == "1" ]]; then
  RUN_ID="${CFG_NAME}_${TS_FULL}"
  echo "[router] FORCE_NEW=1 → fresh timestamped run_id=${RUN_ID}"
else
  EXISTING="$(ls -dt "${PHASE_DIR}/${CFG_NAME}_"[0-9]* 2>/dev/null | head -1 || true)"
  if [[ -n "${EXISTING}" ]]; then
    RUN_ID="$(basename "${EXISTING}")"
    echo "[router] resuming existing run_id=${RUN_ID}"
  else
    RUN_ID="${CFG_NAME}_${TS_DATE}"
    echo "[router] new run_id=${RUN_ID}"
  fi
fi

RUN_DIR="${PHASE_DIR}/${RUN_ID}"
echo "[router] config=${CONFIG}"
echo "[router] run_dir=${RUN_DIR}"
echo "[router] condition=${COND_ID}"
echo "[router] lr_model=${LR_MODEL_PATH}"

# ---------- 检查 runner 是否已在跑 ----------
if pgrep -f "run_experiment.py.*${RUN_ID}" > /dev/null; then
  echo "[router] runner for ${RUN_ID} already running, skipping spawn"
else
  # ---------- Optional: site reset before launch ----------
  if [[ "${RESET_BEFORE:-0}" == "1" ]]; then
    if [[ -f "${REPO_DIR}/scripts/maintenance/reset_vwa_sites.sh" ]]; then
      # shellcheck disable=SC1091
      source "${REPO_DIR}/scripts/maintenance/reset_vwa_sites.sh"
      echo "[router] RESET_BEFORE=1 → resetting site=${SITE}..."
      if reset_vwa_sites "${SITE}" "router_learned_${SITE}"; then
        echo "[router] reset OK; sleeping 15s for site to settle..."
        sleep 15
        echo "[router] refreshing .auth/${SITE}_state.json post-reset..."
        if "${PYTHON_BIN}" -c "
import sys
sys.path.insert(0, '${REPO_DIR}')
from pathlib import Path
from p79.utils.auth_refresh import refresh_site_auth
sys.exit(0 if refresh_site_auth('${SITE}', Path('${REPO_DIR}/.auth')) else 1)
" 2>&1; then
          echo "[router] auth refresh OK — runner task=0 will be LOGGED IN"
        else
          echo "[router][warn] post-reset auth refresh failed; watchdog will retry reactively" >&2
        fi
      else
        rc=$?
        echo "[router][error] reset failed (rc=${rc}); aborting." >&2
        exit 1
      fi
    fi
  fi

  RUNNER_LOG="${LOG_DIR}/${CFG_NAME}_runner_${TS_FULL}.log"
  echo "[router] launching runner → ${RUNNER_LOG}"
  setsid nohup "${PYTHON_BIN}" scripts/run_experiment.py \
    --config "${CONFIG}" \
    --run_id "${RUN_ID}" \
    --log_path "${RUNNER_LOG}" \
    > "${RUNNER_LOG}" 2>&1 < /dev/null &
  disown
  sleep 3
  if pgrep -f "run_experiment.py.*${RUN_ID}" > /dev/null; then
    echo "[router] runner pid=$(pgrep -f "run_experiment.py.*${RUN_ID}" | head -1)"
  else
    echo "[router][error] runner failed to start — check ${RUNNER_LOG}" >&2
    tail -20 "${RUNNER_LOG}" >&2
    exit 1
  fi
fi

# ---------- 启动 watchdog (idempotent) ----------
WATCHDOG_LOG="${LOG_DIR}/exp_watchdog_${RUN_ID}_v2.log"
WATCHDOG_STATE="${LOG_DIR}/exp_watchdog_${RUN_ID}_v2.state.json"
WATCHDOG_DIGEST="${RUN_DIR}/analysis/digest"

RUNNER_PID=$(pgrep -f "run_experiment.py.*${RUN_ID}" | head -1)

if pgrep -f "experiment_watchdog.*${RUN_ID}" > /dev/null; then
  echo "[router] watchdog for ${RUN_ID} already running, skipping spawn"
else
  echo "[router] launching watchdog → ${WATCHDOG_LOG} (runner pid=${RUNNER_PID:-unknown})"
  setsid nohup "${PYTHON_BIN}" -u scripts/maintenance/experiment_watchdog.py \
    --run-dir "${RUN_DIR}" \
    --condition "${COND_ID}" \
    --poll-secs 30 \
    --idle-alert-mins 30 \
    --ntfy-topic p79-exp-dgx-spark \
    --state-file "${WATCHDOG_STATE}" \
    --aggregate-prefix "${BASELINE}_router_learned" \
    --glm-config .auth/glm \
    --digest-dir "${WATCHDOG_DIGEST}" \
    ${RUNNER_PID:+--runner-pid "${RUNNER_PID}"} \
    > "${WATCHDOG_LOG}" 2>&1 < /dev/null &
  disown
  sleep 2
  if pgrep -f "experiment_watchdog.*${RUN_ID}" > /dev/null; then
    echo "[router] watchdog pid=$(pgrep -f "experiment_watchdog.*${RUN_ID}" | head -1)"
  else
    echo "[router][error] watchdog failed to start — paper-grade launch requires watchdog." >&2
    exit 1
  fi
fi

echo
echo "[router] OK — ${BASELINE}_router_learned_${SITE} running"
echo "  run_id=${RUN_ID}"
echo "  runner log:   ${RUNNER_LOG:-<existing>}"
echo "  watchdog log: ${WATCHDOG_LOG}"
