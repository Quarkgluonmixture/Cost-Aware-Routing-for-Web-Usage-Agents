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
#   - baseline:  B0 | B1 | B2
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
  echo "Usage: $0 <baseline:B0|B1|B2> <site> [benchmark:vwa|wa]" >&2
  echo "  Example: bash $0 B0 shopping" >&2
  echo "  RESET_BEFORE=1 bash $0 B0 shopping     # reset shopping container before launch" >&2
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

# Build config name. Post B-261 fix (2026-05-16, A1.7): phantom_text is canonical;
# legacy phantom_dom yaml retired. Fallback removed since conditions.py now raises
# ValueError on phantom_dom (fail-loud), and the legacy yaml has been deleted.
# VWA: exp_v2_<baseline>_phantom_text_<site>.yaml
# WA:  exp_v2_<baseline>_phantom_text_wa_<site>.yaml
CFG_NAME="${BASELINE}_phantom_text"
[[ "${BENCHMARK}" == "wa" ]] && CFG_NAME="${CFG_NAME}_wa"
CFG_NAME="${CFG_NAME}_${SITE}"
CONFIG="${REPO_DIR}/configs/exp_v2_${CFG_NAME}.yaml"

if [[ ! -f "${CONFIG}" ]]; then
  echo "[phantom_text][error] Config not found: ${CONFIG}" >&2
  exit 1
fi

# B-630 (A1.13 P0-1, 2026-05-17): COND_ID canonical sweep. User directive
# 2026-05-17 (/stress A1.13 fix-scope): "phantom_dom 只应该存在于 archive results
# 里面,其他地方都应该为 phantom_text". Pre-fix this constant was legacy
# `phase1_phantom_dom_router_0` for backward-compat with April-2026 historical
# run-dirs containing legacy subdirs. Post-fix: canonical only. Runner derives
# from conditions.py:172 `cid = f"phase1_{obs_mode}_router_0"` → for
# obs_mode="phantom_text" produces `phase1_phantom_text_router_0`. Watchdog
# `--condition` and chain sentinel both must target canonical or they miss
# the actual write target (silent watchdog idle alert + sentinel FATAL abort).
# Historical run-dirs (e.g., B0_phantom_text_classifieds_20260427) keep their
# legacy `phase1_phantom_dom_router_0/` subdirs as archive evidence;
# fresh fires write to `phase1_phantom_text_router_0/` per canonical schema.
COND_ID="phase1_phantom_text_router_0"

PYTHON_BIN="${REPO_DIR}/.venv/bin/python3"
LOG_DIR="${REPO_DIR}/logs"
mkdir -p "${LOG_DIR}"

# ---------- A1.13 lib (2026-05-16): shared paper-grade gates ----------
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_lib_paper_grade_gates.sh"
init_paper_grade_env "${REPO_DIR}"
assert_a100_url_locality
# B-704 (A1.14 Chunk d P1-4): per-(site, benchmark) flock at leaf entry.
if ! acquire_site_lock "${SITE}" "${BENCHMARK}" "queue_phantom_text"; then
  exit $?
fi
trap "release_site_lock" EXIT INT TERM

# ---------- B0 PROXY API key 加载 ----------
if [[ "${BASELINE}" == "B0" ]]; then
  load_proxy_api_key "${REPO_DIR}" "phantom_text"
fi

# ---------- 决定 run_id + run_dir ----------
if [[ "${BENCHMARK}" == "wa" ]]; then
  PHASE_DIR="${REPO_DIR}/results/webarena/phase1"
else
  PHASE_DIR="${REPO_DIR}/results/visualwebarena/phase1"
fi

# B-630 (A1.13 P0-1, 2026-05-17): legacy phantom_dom resume removed. User
# directive 2026-05-17: phantom_dom artifacts only exist in archive results
# (historical run-dirs created pre-rename 2026-05-14); they are NOT a valid
# resume target — the legacy `phase1_phantom_dom_router_0/` subdirs have
# frozen schema that the post-rename runner cannot extend (different cond_id
# write path). Force fresh-or-canonical-resume only via mint_run_id.
mint_run_id "${CFG_NAME}" "${PHASE_DIR}" "phantom_text"
# B-634 (A1.13 P0-5): RUNNER_LOG uses RUN_ID (0-collision); TS_FULL removed.

RUN_DIR="${PHASE_DIR}/${RUN_ID}"
echo "[phantom_text] config=${CONFIG}"
echo "[phantom_text] run_dir=${RUN_DIR}"
echo "[phantom_text] condition=${COND_ID}"

# ---------- 检查 runner 是否已在跑 ----------
if pgrep -f "run_experiment.py.*${RUN_ID}" > /dev/null; then
  # B-867 (/stress A1.23 P1-10 B*, 2026-05-17): dirty-cell backdoor FATAL —
  # sibling propagation of B-756 (queue_baseline.sh:139-153).
  if [[ "${P79_PAPER_GRADE:-0}" == "1" && "${RESET_BEFORE:-0}" == "1" ]]; then
    echo "[phantom_text][FATAL] runner for ${RUN_ID} already running under (P79_PAPER_GRADE=1 + RESET_BEFORE=1)." >&2
    echo "[phantom_text][FATAL] paper-grade requires fresh post-reset cell; idempotent skip would dissolve the reset gate." >&2
    echo "[phantom_text][FATAL] options: (a) pkill the existing runner; (b) RESET_BEFORE=0 explicit-resume; (c) P79_PAPER_GRADE=0 dirty/dev." >&2
    exit 1
  fi
  echo "[phantom_text] runner for ${RUN_ID} already running, skipping spawn"
  echo "[phantom_text] (RESET_BEFORE skipped — runner already attached to current site state)"
else
  # B-858 (/stress A1.23 P0-1 ABC* OOB, 2026-05-17): cross-mode collision check.
  # See queue_baseline.sh for full rationale.
  assert_no_cross_mode_collision "${BASELINE}" "${SITE}" "${BENCHMARK}" "${RUN_ID}" "phantom_text"

  # ---------- Optional: site reset before launch ----------
  # IMPORTANT: reset AFTER idempotent runner check (race fixed 2026-04-28 §104).
  # A1.13 P0-1 (2026-05-16) propagated B-224 hard-fail via reset_and_auth_gate.
  if [[ "${RESET_BEFORE:-0}" == "1" && "${BENCHMARK}" != "wa" ]]; then
    reset_and_auth_gate --site "${SITE}" --repo "${REPO_DIR}" --python "${PYTHON_BIN}" --log-prefix "phantom_text" --reset-label "phantom_text_${SITE}"
  elif [[ "${RESET_BEFORE:-0}" == "1" ]]; then
    # B-647 (A1.13 P1-4-BC fix, 2026-05-17): see queue_baseline.sh equivalent.
    echo "[phantom_text][error] BENCHMARK=wa + RESET_BEFORE=1 unsupported until reset_wa_sites.sh full impl lands (B-647)." >&2
    echo "[phantom_text][error] scripts/maintenance/reset_wa_sites.sh is scaffold only; fill per-site bodies + remove this guard for WA paper-grade." >&2
    exit 1
  fi

  RUNNER_LOG="${LOG_DIR}/${RUN_ID}_runner.log"
  echo "[phantom_text] launching runner → ${RUNNER_LOG}"
  # codex stress v6 C4: redirect runner stdout/stderr to RUNNER_LOG (was /dev/null).
  # Python logging goes to stderr — /dev/null discarded all phantom runner logs,
  # making mid-run crash debug impossible + paper-grade audit trail incomplete.
  # B-1824 (Fire-6 /stress P2-2): shared daemon spawn closes inherited lock fds 9/8/7.
  spawn_paper_grade_daemon 0 "${RUNNER_LOG}" -- \
    "${PYTHON_BIN}" scripts/run_experiment.py \
    --config "${CONFIG}" \
    --run_id "${RUN_ID}" \
    --log_path "${RUNNER_LOG}"
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
# B-648 (A1.13 P2-3 gemini G3, 2026-05-17): unified per-baseline aggregate
# gallery — alias to `B${n}_3mode/` URL for ANY baseline (B0/B1/B2/future via
# regex `^B[0-9]+_3mode$` in generate_gallery.py:1274 post-B-638). "3mode" is
# historical alias only — NOT literal 3-mode-subset restriction.
AGGREGATE_PREFIX="${BASELINE}_3mode"

# Runner PID for watchdog self-exit — watchdog auto-exits when this PID dies
# AND condition_summary_v2.json present. Prevents init-orphan idle loops.
RUNNER_PID=$(pgrep -f "run_experiment.py.*${RUN_ID}" | head -1)

# B-907 (/stress A2.2 P0-5-B* codex F1 OOB, 2026-05-17): per-RUN_ID flock —
# sibling propagation from queue_baseline.sh. See lib `acquire_watchdog_lock`
# header for full rationale.
if ! acquire_watchdog_lock "${RUN_ID}" "queue_phantom_text"; then
  exit $?
fi
trap "release_watchdog_lock; release_site_lock" EXIT INT TERM
if pgrep -f "experiment_watchdog.*${RUN_ID}" > /dev/null; then
  echo "[phantom_text] watchdog for ${RUN_ID} already running, skipping spawn"
else
  echo "[phantom_text] launching watchdog → ${WD_LOG} (runner pid=${RUNNER_PID:-unknown})"
  # B-1824 (see runner above): shared daemon spawn closes inherited lock fds.
  spawn_paper_grade_daemon 1 "${WD_LOG}" -- \
    "${PYTHON_BIN}" -u scripts/maintenance/experiment_watchdog.py \
    --run-dir "${RUN_DIR}" \
    --condition "${COND_ID}" \
    --poll-secs 30 --idle-alert-mins "${EXP_WATCHDOG_IDLE_ALERT_MINS:-60}" \
    --ntfy-topic "${NTFY_TOPIC:-p79-exp-dgx-spark}" \
    --state-file "${WD_STATE}" \
    --aggregate-prefix "${AGGREGATE_PREFIX}" \
    ${RUNNER_PID:+--runner-pid "${RUNNER_PID}"}
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
