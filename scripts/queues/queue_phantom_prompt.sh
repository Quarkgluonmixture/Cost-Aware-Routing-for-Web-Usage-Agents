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
  echo "Usage: $0 <baseline:B0|B1|B2|B4> <site> [benchmark:vwa|wa]" >&2
  echo "  Example: bash $0 B0 classifieds" >&2
  echo "  RESET_BEFORE=1 bash $0 B0 classifieds     # reset shopping-style site before launch" >&2
  exit 2
fi

BASELINE="$1"; SITE="$2"
BENCHMARK="${3:-vwa}"

# Validation
if [[ "${BASELINE}" != "B0" && "${BASELINE}" != "B1" && "${BASELINE}" != "B2" && "${BASELINE}" != "B4" ]]; then
  echo "Invalid baseline: ${BASELINE} (expected B0, B1, B2 or B4)" >&2; exit 2
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
# B-704 (A1.14 Chunk d P1-4): per-(site, benchmark) flock at leaf entry.
# B-1937 (2026-08-03): was `if ! acquire_site_lock; then exit $?; fi` — inside that
# branch `$?` is the status of `!`, i.e. ALWAYS 0, so a refused lock
# exited 0 and the caller read it as success. `|| exit $?` evaluates
# `$?` when exit runs, where it still holds the real rc (78 = contention).
acquire_site_lock "${SITE}" "${BENCHMARK}" "queue_phantom_prompt" || exit $?
trap "release_site_lock" EXIT INT TERM

# ---------- B0 PROXY API key 加载 ----------
# B4 (Claude Sonnet 5, 2026-08-13) shares B0's AWS proxy endpoint, so it needs the
# same key load. Gating on the baseline NAME rather than on the config's backend type
# is why adding a proxy-served baseline touches this line at all -- see 笔记 §461.
if [[ "${BASELINE}" == "B0" || "${BASELINE}" == "B4" ]]; then
  load_proxy_api_key "${REPO_DIR}" "phantom_prompt"
fi

# ---------- 决定 run_id + run_dir ----------
if [[ "${BENCHMARK}" == "wa" ]]; then
  PHASE_DIR="${REPO_DIR}/results/webarena/phase1"
else
  PHASE_DIR="${REPO_DIR}/results/visualwebarena/phase1"
fi

mint_run_id "${CFG_NAME}" "${PHASE_DIR}" "phantom_prompt"
# B-634 (A1.13 P0-5): RUNNER_LOG uses RUN_ID (0-collision); TS_FULL removed.

RUN_DIR="${PHASE_DIR}/${RUN_ID}"
echo "[phantom_prompt] config=${CONFIG}"
echo "[phantom_prompt] run_dir=${RUN_DIR}"
echo "[phantom_prompt] condition=${COND_ID}"

# ---------- 检查 runner 是否已在跑 ----------
if pgrep -f "run_experiment.py.*${RUN_ID}" > /dev/null; then
  # B-867 (/stress A1.23 P1-10 B*, 2026-05-17): dirty-cell backdoor FATAL —
  # sibling propagation of B-756 (queue_baseline.sh:139-153).
  if [[ "${P79_PAPER_GRADE:-0}" == "1" && "${RESET_BEFORE:-0}" == "1" ]]; then
    echo "[phantom_prompt][FATAL] runner for ${RUN_ID} already running under (P79_PAPER_GRADE=1 + RESET_BEFORE=1)." >&2
    echo "[phantom_prompt][FATAL] paper-grade requires fresh post-reset cell; idempotent skip would dissolve the reset gate." >&2
    echo "[phantom_prompt][FATAL] options: (a) pkill the existing runner; (b) RESET_BEFORE=0 explicit-resume; (c) P79_PAPER_GRADE=0 dirty/dev." >&2
    exit 1
  fi
  echo "[phantom_prompt] runner for ${RUN_ID} already running, skipping spawn"
  echo "[phantom_prompt] (RESET_BEFORE skipped — runner already attached to current site state)"
else
  # B-858 (/stress A1.23 P0-1 ABC* OOB, 2026-05-17): cross-mode collision check.
  # See queue_baseline.sh for full rationale.
  assert_no_cross_mode_collision "${BASELINE}" "${SITE}" "${BENCHMARK}" "${RUN_ID}" "phantom_prompt"

  # ---------- Optional: site reset before launch ----------
  # IMPORTANT: reset AFTER idempotent runner check (race fixed 2026-04-28 §104).
  # A1.13 P0-1 (2026-05-16) propagated B-224 hard-fail via reset_and_auth_gate.
  if [[ "${RESET_BEFORE:-0}" == "1" ]] && wa_reset_supported "${BENCHMARK}" "${SITE}"; then
    reset_and_auth_gate --site "${SITE}" --repo "${REPO_DIR}" --python "${PYTHON_BIN}" --log-prefix "phantom_prompt" --reset-label "phantom_prompt_${SITE}" --benchmark "${BENCHMARK}"
  elif [[ "${RESET_BEFORE:-0}" == "1" ]]; then
    # B-647 (A1.13 P1-4-BC fix, 2026-05-17) + B-1930 (2026-08-03, WA shopping
    # routes to the shared Magento container): see queue_baseline.sh equivalent.
    echo "[phantom_prompt][error] RESET_BEFORE=1 but no reset implementation for benchmark=${BENCHMARK} site=${SITE}." >&2
    echo "[phantom_prompt][error] Supported: all VWA sites; WA reddit / shopping / shopping_admin (B-1930, shared containers)." >&2
    exit 1
  else
    # B-1962 (/stress Mode B P0-2, 2026-08-06): RESET_BEFORE=0 is the
    # B-304-mandated way to resume a partial condition without splicing a
    # fresh-reset trajectory into a dirty one — but pre-fix it also skipped
    # the ONLY auth refresh on the launch path. A resume after an outage
    # longer than the ~1440s PHP session then ran its first episodes on dead
    # cookies (2026-08-05 shopping: tasks 346/347/348 fought the login page
    # for 22 minutes until the 5-episode refresh cadence caught up).
    # Auth-only = substrate restoration, no site-state mutation.
    auth_only_gate --site "${SITE}" --repo "${REPO_DIR}" \
      --python "${PYTHON_BIN}" --log-prefix "phantom_prompt" \
      --benchmark "${BENCHMARK}" || exit 1
  fi

  RUNNER_LOG="${LOG_DIR}/${RUN_ID}_runner.log"
  echo "[phantom_prompt] launching runner → ${RUNNER_LOG}"
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
# B-648 (A1.13 P2-3 gemini G3, 2026-05-17): unified per-baseline aggregate
# gallery — alias to `B${n}_3mode/` URL for ANY baseline (regex `^B[0-9]+_3mode$`
# in generate_gallery.py post-B-638). "3mode" is historical alias.
AGGREGATE_PREFIX="${BASELINE}_3mode"

# Runner PID for watchdog self-exit — watchdog auto-exits when this PID dies
# AND condition_summary_v2.json present. Prevents init-orphan idle loops.
RUNNER_PID=$(pgrep -f "run_experiment.py.*${RUN_ID}" | head -1)

# B-907 (/stress A2.2 P0-5-B* codex F1 OOB, 2026-05-17): per-RUN_ID flock —
# sibling propagation from queue_baseline.sh. See lib `acquire_watchdog_lock`
# header for full rationale.
# B-1937 (2026-08-03): was `if ! acquire_watchdog_lock; then exit $?; fi` — inside that
# branch `$?` is the status of `!`, i.e. ALWAYS 0, so a refused lock
# exited 0 and the caller read it as success. `|| exit $?` evaluates
# `$?` when exit runs, where it still holds the real rc (78 = contention).
acquire_watchdog_lock "${RUN_ID}" "queue_phantom_prompt" || exit $?
trap "release_watchdog_lock; release_site_lock" EXIT INT TERM
if pgrep -f "experiment_watchdog.*${RUN_ID}" > /dev/null; then
  echo "[phantom_prompt] watchdog for ${RUN_ID} already running, skipping spawn"
else
  echo "[phantom_prompt] launching watchdog → ${WD_LOG} (runner pid=${RUNNER_PID:-unknown})"
  # B-1824 (see runner above): shared daemon spawn closes inherited lock fds.
  spawn_paper_grade_daemon 1 "${WD_LOG}" -- \
    "${PYTHON_BIN}" -u scripts/maintenance/experiment_watchdog.py \
    --run-dir "${RUN_DIR}" \
    --condition "${COND_ID}" \
    --poll-secs 30 --idle-alert-mins "${EXP_WATCHDOG_IDLE_ALERT_MINS:-180}" \
    --ntfy-topic "${NTFY_TOPIC:-p79-exp-dgx-spark}" \
    --state-file "${WD_STATE}" \
    --aggregate-prefix "${AGGREGATE_PREFIX}" \
    ${RUNNER_PID:+--runner-pid "${RUNNER_PID}"}
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
