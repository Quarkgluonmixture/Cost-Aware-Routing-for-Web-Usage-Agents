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
#   - baseline:  B0 | B1 | B2
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
  echo "Usage: $0 <baseline:B0|B1|B2> <mode:dom|som|vision> <site> [benchmark:vwa|wa]" >&2
  echo "  e.g. bash $0 B0 dom shopping" >&2
  echo "       bash $0 B0 vision shopping wa" >&2
  exit 2
fi

BASELINE="$1"; MODE="$2"; SITE="$3"
BENCHMARK="${4:-vwa}"

# Validation
if [[ "${BASELINE}" != "B0" && "${BASELINE}" != "B1" && "${BASELINE}" != "B2" ]]; then
  echo "Invalid baseline: ${BASELINE} (expected B0, B1 or B2)" >&2; exit 2
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

# ---------- A1.13 lib (2026-05-16): shared paper-grade gates ----------
# Centralizes env init + A100 URL locality preflight + auth gate + RUN_ID mint
# across queue_baseline + queue_phantom_{som,text,prompt}. Prevents future
# sibling-propagation drift (P0-1 + P0-2 + P1-2 fixes).
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_lib_paper_grade_gates.sh"
init_paper_grade_env "${REPO_DIR}"
assert_a100_url_locality

# ---------- BUG-6 NOTE (2026-05-16 A1.13 audit P1-4-A): vestigial QUARK_TZ removed ----------
# Pre-2026-05-14: paper-grade fired DGX→quark, container TZ rendering crossed midnight
# boundary for reddit `must_include` date tasks. Original fix attempt: export `QUARK_TZ`
# on runner client. A1.13 audit (Claude OOB) showed client-side export does not influence
# docker container TZ → no-op cargo cult. Post-2026-05-14: paper-grade fires on A100
# self-hosted docker (memory `project_paper_grade_target_host`); A100 host + container
# both default UTC, no quark in loop. Residual cross-midnight relative-timestamp drift
# bounded to ~5/210 reddit tasks; disclosed in paper §限制 (not code-fixable here).

# ---------- B0 PROXY API key 加载 ----------
if [[ "${BASELINE}" == "B0" ]]; then
  load_proxy_api_key "${REPO_DIR}" "baseline"
fi

# ---------- 决定 run_id + run_dir ----------
if [[ "${BENCHMARK}" == "wa" ]]; then
  PHASE_DIR="${REPO_DIR}/results/webarena/phase1"
else
  PHASE_DIR="${REPO_DIR}/results/visualwebarena/phase1"
fi

mint_run_id "${CFG_NAME}" "${PHASE_DIR}" "baseline"
# TS_FULL retained for runner log naming (no longer used in RUN_ID since A1.13 P1-2).
TS_FULL="$(date +%Y%m%d_%H%M%S)"

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
  # 2026-04-28 — see 实验笔记 §104). reset_and_auth_gate (in lib) enforces
  # B-224 hard-fail (no soft-warn fallthrough).
  if [[ "${RESET_BEFORE:-0}" == "1" && "${BENCHMARK}" != "wa" ]]; then
    reset_and_auth_gate "${SITE}" "${REPO_DIR}" "${PYTHON_BIN}" "baseline" "baseline_${MODE}_${SITE}"
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
