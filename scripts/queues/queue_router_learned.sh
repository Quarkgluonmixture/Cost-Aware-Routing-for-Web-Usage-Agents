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

# Condition id: phase1_learned_router_<backend_id>_<site> (single per cell, NOT per-mode)
# A2.8 P0-5-B* B-1557 (/stress 2026-05-18 codex Mode B unique OOB): COND_ID must
# match runner-generated condition_id pattern at p79/experiment/conditions.py:339-356
# (`f"phase1_learned_router_{backend_id}_{site_hint}"`), otherwise watchdog monitors
# wrong cond dir while runner writes to new cond dir → silent monitoring gap +
# per-condition cleanup targets the wrong directory + paper-grade invariant broken.
# backend_id mapping (must match each YAML's `backends.default_backend`):
case "${BASELINE}" in
  B0) BACKEND_ID="api_strong" ;;     # configs/exp_v2_B0_router_learned_*.yaml backends.default_backend
  B1) BACKEND_ID="local_4b" ;;       # configs/exp_v2_B1_router_learned_*.yaml backends.default_backend
  B2) BACKEND_ID="local_gemma" ;;    # configs/exp_v2_B2_router_learned_*.yaml backends.default_backend
  *) echo "[router][error] Unknown BASELINE=${BASELINE}; expected B0|B1|B2" >&2; exit 1 ;;
esac
COND_ID="phase1_learned_router_${BACKEND_ID}_${SITE}"

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

# ---------- BUG-2 preflight: A100 paper-grade host + URL-locality enforcement ----------
# B-1406 (/stress A2.7 P1-4-AB* 2-AI overlap, Claude Mode A F1 + codex Mode B F5,
# 2026-05-18): canonical paper-grade host + URL-locality gate now lives in
# `_lib_paper_grade_gates.sh::require_paper_grade_host` (which itself calls
# `assert_a100_url_locality`). Pre-fix this script's hand-rolled
# `*condense* || -d /home/ubuntu/workspace/p79` predicate (a) missed
# `a100-jiaming-test` canonical hostname (no "condense" substring), (b) silently
# passed on empty URL (B-643 fixed in lib version), (c) accepted empty string in
# OK set, (d) diverged from orchestrator + Pass-2 router scripts. Sourcing the
# lib + calling `require_paper_grade_host` consolidates to single canonical
# implementation; orchestrator + Pass-2 router scripts use the same code path.
# Define `log` + `fail` if missing (queue_router_learned.sh doesn't always have
# them in scope before this gate fires).
log() { echo "[router_learned $(date '+%H:%M:%S')] $*"; }
fail() { log "FAIL: $*"; exit 1; }
source "${REPO_DIR}/scripts/queues/_lib_paper_grade_gates.sh"
init_paper_grade_env "${REPO_DIR}"
require_paper_grade_host
# P0-5-B* (/stress Phase 0 unified bug list 2026-05-19, codex unique OOB
# sibling-propagation gap): per-(site, benchmark) flock — port from
# queue_baseline.sh:102-105 + queue_phantom_*.sh siblings. Pre-fix Pass-2
# leaf bypassed site lock → manual leaf invocation during active baseline
# chain on same site → race → RESET wipes session state under detached
# runner. queue_chain.sh enforces at chain layer (P79_CHAIN_LOCK_HELD env)
# but documented manual leaf usage (per CLAUDE.md "leaf is supported entry")
# remained a bypass surface.
if ! acquire_site_lock "${SITE}" "vwa" "router_learned"; then
  exit $?
fi
trap "release_site_lock" EXIT INT TERM

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

# ---------- Pre-launch: fold-aware LR bundle must validate ----------
# S3 cross-AI P0-2-B* (2026-06-02): pre-fix gated on the DEPRECATED single-pickle
# (${BASELINE}_${SITE}_lr.pkl). The fold-aware runtime (predict_mode_fold_aware)
# needs the full bundle (5 fold vectorizers + selectors + 6 cells × {meta + 5 LR
# heads + fold_assignment}), NOT the legacy pickle — a stale single-pickle would
# pass the old check while every fold-aware artifact is missing (codex measured
# 52 validation failures on the May-16 on-disk state). Delegate to the canonical
# validator. Under P79_PAPER_GRADE=1 the ALLOW_NO_LR_MODEL bypass is hard-blocked
# (no stale-artifact paper-grade fire). The old "train_l1_router.py is TODO" note
# was stale — Stage 1→3 (extract_50_features → train_l1_router_with_mi →
# train_l1_router) is the canonical bundle producer.
LR_ARTIFACTS_DIR="${REPO_DIR}/results/phantom_paper/l1_router"
if ! "${PYTHON_BIN:-${REPO_DIR}/.venv/bin/python3}" "${REPO_DIR}/scripts/queues/_lib_lr_artifact_validate.py" --artifacts-dir "${LR_ARTIFACTS_DIR}"; then
  echo "[router][warn] fold-aware LR bundle validation FAILED (${LR_ARTIFACTS_DIR})." >&2
  echo "  Pass-2 router needs the fold-aware bundle; regenerate via Stage 1→3:" >&2
  echo "    extract_50_features.py → train_l1_router_with_mi.py → train_l1_router.py" >&2
  if [[ "${P79_PAPER_GRADE:-0}" == "1" ]]; then
    echo "  [FATAL] P79_PAPER_GRADE=1: ALLOW_NO_LR_MODEL bypass HARD-BLOCKED (no stale-artifact fire)." >&2
    exit 1
  elif [[ "${ALLOW_NO_LR_MODEL:-0}" != "1" ]]; then
    echo "  Set ALLOW_NO_LR_MODEL=1 to bypass (DEV ONLY; router will fail at runtime)." >&2
    exit 1
  fi
  echo "[router][warn] ALLOW_NO_LR_MODEL=1 (dev) — proceeding despite validation failure." >&2
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
echo "[router] lr_artifacts_dir=${LR_ARTIFACTS_DIR}"

# ---------- 检查 runner 是否已在跑 ----------
if pgrep -f "run_experiment.py.*${RUN_ID}" > /dev/null; then
  # P0-5-B* (/stress Phase 0 unified bug list 2026-05-19, codex unique OOB):
  # port B-756 "Dirty Cell Backdoor" hard-fail from queue_baseline.sh:149-154
  # + phantom siblings. Pre-fix Pass-2 leaf: a manually-launched runner
  # without RESET could be picked up here (idempotent skip), RESET_BEFORE
  # silently skipped, queue_chain accepts "dirty cell" as paper-grade
  # complete via sentinel. Gemini area-chair attack: protocol prioritized
  # non-interruption over initial-state integrity. Now: under (PG=1 AND
  # RESET_BEFORE=1) the contradiction is explicit → hard fail.
  if [[ "${P79_PAPER_GRADE:-0}" == "1" && "${RESET_BEFORE:-0}" == "1" ]]; then
    echo "[router_learned][FATAL] runner for ${RUN_ID} already running under (P79_PAPER_GRADE=1 + RESET_BEFORE=1)." >&2
    echo "[router_learned][FATAL] paper-grade requires fresh post-reset cell; idempotent skip would dissolve the reset gate (dirty cell backdoor)." >&2
    echo "[router_learned][FATAL] options: (a) 'pkill -f \"run_experiment.py.*${RUN_ID}\"' then re-run; (b) RESET_BEFORE=0 to explicit-resume; (c) P79_PAPER_GRADE=0 for explicit dirty/dev." >&2
    exit 1
  fi
  echo "[router] runner for ${RUN_ID} already running, skipping spawn"
  echo "[router] (RESET_BEFORE skipped — runner already attached to current site state)"
else
  # P0-5-B* (/stress Phase 0 unified bug list 2026-05-19, codex unique OOB):
  # port B-858 cross-mode collision check from queue_baseline.sh:165 +
  # phantom siblings. Pre-fix Pass-2 leaf: pgrep above matches by FULL
  # RUN_ID; a second manual leaf invocation with DIFFERENT mode (same
  # baseline+site, e.g. baseline B0 dom cls already running, then user
  # invokes router B0 cls) → different RUN_ID → bypass idempotent skip
  # + run reset_and_auth_gate → site wipe under detached baseline runner.
  # queue_chain.sh:248 _collision_match enforces at chain layer; this
  # propagates to standalone leaf entry per CLAUDE.md hard rule.
  assert_no_cross_mode_collision "${BASELINE}" "${SITE}" "vwa" "${RUN_ID}" "router_learned"

  # ---------- Optional: site reset before launch ----------
  # B-1585 (/stress A1.24 post-fire P1-5-B codex Mode B F3, 2026-05-18):
  # Pass-2 router leaf inherits the same hard-fail reset+auth contract as
  # baseline + phantom leaves via `reset_and_auth_gate` lib helper. Pre-fix
  # inline soft `refresh_site_auth` + warn-and-continue allowed post-reset
  # auth failure to advance to runner spawn → NOT-LOGGED-IN task=0 →
  # paper-grade contamination identical to B-1575 watchdog hypocrisy.
  # Lib helper enforces B-224 hard-fail + B-639 P79_PAPER_GRADE=1 bypass
  # block + B-745 site-aware timeout + B-864 SIGTERM trap (now with B-1583
  # corrected container names).
  if [[ "${RESET_BEFORE:-0}" == "1" ]]; then
    reset_and_auth_gate --site "${SITE}" --repo "${REPO_DIR}" --python "${PYTHON_BIN}" --log-prefix "router_learned" --reset-label "router_learned_${SITE}"
  fi

  RUNNER_LOG="${LOG_DIR}/${CFG_NAME}_runner_${TS_FULL}.log"
  echo "[router] launching runner → ${RUNNER_LOG}"
  # B-1824 (Fire-6 /stress P2-2): shared daemon spawn closes inherited lock fds 9/8/7.
  spawn_paper_grade_daemon 0 "${RUNNER_LOG}" -- \
    "${PYTHON_BIN}" scripts/run_experiment.py \
    --config "${CONFIG}" \
    --run_id "${RUN_ID}" \
    --log_path "${RUNNER_LOG}"
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

RUNNER_PID=$(pgrep -f "run_experiment.py.*${RUN_ID}" | head -1)

# B-1824 (Fire-6 /stress P1-1-AB*): watchdog-spawn flock parity with baseline +
# phantom leaves. router_learned called acquire_site_lock (L135) but NOT
# acquire_watchdog_lock → 2 watchdogs on the same RUN_ID could pass the pgrep
# TOCTOU window + race on the shared WD_STATE (B-907 class). Extends the existing
# site-lock trap (L138).
if ! acquire_watchdog_lock "${RUN_ID}" "router_learned"; then
  exit $?
fi
trap "release_watchdog_lock; release_site_lock" EXIT INT TERM

if pgrep -f "experiment_watchdog.*${RUN_ID}" > /dev/null; then
  echo "[router] watchdog for ${RUN_ID} already running, skipping spawn"
else
  echo "[router] launching watchdog → ${WATCHDOG_LOG} (runner pid=${RUNNER_PID:-unknown})"
  # B-1824 (Fire-6 /stress P2-2): shared daemon spawn closes inherited lock fds.
  spawn_paper_grade_daemon 0 "${WATCHDOG_LOG}" -- \
    "${PYTHON_BIN}" -u scripts/maintenance/experiment_watchdog.py \
    --run-dir "${RUN_DIR}" \
    --condition "${COND_ID}" \
    --poll-secs 30 \
    --idle-alert-mins "${EXP_WATCHDOG_IDLE_ALERT_MINS:-60}" \
    --ntfy-topic p79-exp-dgx-spark \
    --state-file "${WATCHDOG_STATE}" \
    --aggregate-prefix "${BASELINE}_router_learned" \
    ${RUNNER_PID:+--runner-pid "${RUNNER_PID}"}
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
