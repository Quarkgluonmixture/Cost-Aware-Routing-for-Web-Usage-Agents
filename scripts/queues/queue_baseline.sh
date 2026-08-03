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
  echo "Usage: $0 <baseline:B0|B1|B2|B3> <mode:dom|som|vision> <site> [benchmark:vwa|wa]" >&2
  echo "  e.g. bash $0 B0 dom shopping" >&2
  echo "       bash $0 B0 vision shopping wa" >&2
  exit 2
fi

BASELINE="$1"; MODE="$2"; SITE="$3"
BENCHMARK="${4:-vwa}"

# Validation
if [[ "${BASELINE}" != "B0" && "${BASELINE}" != "B1" && "${BASELINE}" != "B2" && "${BASELINE}" != "B3" ]]; then
  echo "Invalid baseline: ${BASELINE} (expected B0, B1, B2 or B3)" >&2; exit 2
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

# SMOKE_CONFIG override (pre-Fire-6 pipeline smoke WITHOUT touching production config).
# Set SMOKE_CONFIG=<repo-relative-or-absolute path> to run a dedicated NON-CANONICAL
# smoke config. cfg_name (→ the minted run_id) is derived from the smoke config
# basename so the smoke run dir is isolated from the canonical Fire run dirs. Smoke
# runs are non-canonical by naming convention — delete / ignore the smoke run dir
# after verification. Pairs with FORCE_NEW=1 (unique timestamped run_id).
if [[ -n "${SMOKE_CONFIG:-}" ]]; then
  [[ "${SMOKE_CONFIG}" = /* ]] && CONFIG="${SMOKE_CONFIG}" || CONFIG="${REPO_DIR}/${SMOKE_CONFIG}"
  CFG_NAME="$(basename "${CONFIG}" .yaml | sed 's/^exp_v2_//')"
  echo "[baseline] SMOKE_CONFIG override → config=${CONFIG} cfg_name=${CFG_NAME} (NON-CANONICAL smoke)"
fi

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
# B-704 (A1.14 Chunk d P1-4 codex F5 OOB B, 2026-05-17): per-(site, benchmark)
# flock at leaf entry. Skips if parent queue_chain holds the lock (via
# P79_CHAIN_LOCK_HELD env). Manual leaf invocation outside any chain →
# leaf acquires its own. trap release on EXIT/INT/TERM.
if ! acquire_site_lock "${SITE}" "${BENCHMARK}" "queue_baseline"; then
  exit $?
fi
trap "release_site_lock" EXIT INT TERM

# ---------- BUG-6 NOTE (2026-05-16 A1.13 audit P1-4-A): vestigial QUARK_TZ removed ----------
# Pre-2026-05-14: paper-grade fired DGX→quark, container TZ rendering crossed midnight
# boundary for reddit `must_include` date tasks. Original fix attempt: export `QUARK_TZ`
# on runner client. A1.13 audit (Claude OOB) showed client-side export does not influence
# docker container TZ → no-op cargo cult. Post-2026-05-14: paper-grade fires on A100
# self-hosted docker (memory `project_paper_grade_target_host`); A100 host + container
# both default UTC, no quark in loop. Residual cross-midnight relative-timestamp drift
# bounded to ~5/210 reddit tasks; disclosed in paper §限制 (not code-fixable here).

# ---------- TZ ALIGN (BUG-6 fix, 3-AI agree 2026-05-16) ----------
# A100 host = UTC, quark Windows host = GMT Standard Time (Europe/London).
# Postmill timestamps render in container TZ → reddit task `must_include:
# ["08-11-2023"]` evals break across midnight boundary. 1-2pp drift.
export QUARK_TZ="${QUARK_TZ:-Europe/London}"

# ---------- BUG-2 preflight: assert all site URLs are local on A100 ----------
# vwa_env_remote.sh may use ${VAR:-localhost} default-expansion; inherited
# shell env can override silently → A100 runner hits quark prod (worst-case
# 100% silent deployment substitution per codex CodexOnly-2).
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
  load_proxy_api_key "${REPO_DIR}" "baseline"
fi

# ---------- 决定 run_id + run_dir ----------
if [[ "${BENCHMARK}" == "wa" ]]; then
  PHASE_DIR="${REPO_DIR}/results/webarena/phase1"
else
  PHASE_DIR="${REPO_DIR}/results/visualwebarena/phase1"
fi

mint_run_id "${CFG_NAME}" "${PHASE_DIR}" "baseline"
# B-634 (A1.13 P0-5 gemini G4 OOB, 2026-05-17): RUNNER_LOG now uses RUN_ID
# (has PID+RANDOM+nanos suffix) → 0-collision risk when master orchestrator
# fires 2 chains in same second. Removed redundant TS_FULL recompute.

RUN_DIR="${PHASE_DIR}/${RUN_ID}"
echo "[baseline] config=${CONFIG}"
echo "[baseline] run_dir=${RUN_DIR}"
echo "[baseline] condition=${COND_ID}"

# ---------- 检查 runner 是否已在跑 ----------
if pgrep -f "run_experiment.py.*${RUN_ID}" > /dev/null; then
  # B-756 (/stress A1.17 cold-start P1-11 C, 2026-05-17): "Dirty Cell Backdoor"
  # FATAL under paper-grade. Pre-fix: a manually-launched runner without RESET
  # could be picked up here, RESET_BEFORE silently skipped, and `queue_chain` would
  # accept the dirty cell as "paper-grade complete" via the completion sentinel.
  # Gemini area-chair attack: protocol prioritized non-interruption over initial-state
  # integrity. Now: under (P79_PAPER_GRADE=1 AND RESET_BEFORE=1) the contradiction is
  # explicit — operator wanted reset but reset will be skipped → hard fail. Dev mode
  # (P79_PAPER_GRADE=0) keeps idempotent skip; legitimate resume (RESET_BEFORE=0 +
  # pre-existing runner) still works under PG=1.
  if [[ "${P79_PAPER_GRADE:-0}" == "1" && "${RESET_BEFORE:-0}" == "1" ]]; then
    echo "[baseline][FATAL] runner for ${RUN_ID} already running under (P79_PAPER_GRADE=1 + RESET_BEFORE=1)." >&2
    echo "[baseline][FATAL] paper-grade requires fresh post-reset cell; idempotent skip would dissolve the reset gate (dirty cell backdoor)." >&2
    echo "[baseline][FATAL] options: (a) 'pkill -f \"run_experiment.py.*${RUN_ID}\"' then re-run; (b) set RESET_BEFORE=0 to explicit-resume the pre-existing runner; (c) set P79_PAPER_GRADE=0 for explicit dirty/dev mode." >&2
    exit 1
  fi
  echo "[baseline] runner for ${RUN_ID} already running, skipping spawn"
  echo "[baseline] (RESET_BEFORE skipped — runner already attached to current site state)"
else
  # B-858 (/stress A1.23 P0-1 ABC* OOB, 2026-05-17): cross-mode collision check.
  # The pgrep above matches by FULL RUN_ID; a second manual leaf invocation
  # with a DIFFERENT mode (same baseline+site) has a different RUN_ID → would
  # bypass the idempotent skip + run reset_and_auth_gate → site wipe under
  # existing detached runner. queue_chain.sh:248 enforces this at chain layer;
  # this propagates to standalone leaf entry (CLAUDE.md documents leaf as
  # supported, so the check must live here too).
  assert_no_cross_mode_collision "${BASELINE}" "${SITE}" "${BENCHMARK}" "${RUN_ID}" "baseline"

  # ---------- Optional: site reset before launch ----------
  # IMPORTANT: reset is AFTER the idempotent runner check — resetting while
  # a runner is attached destroys site state under it (race condition fixed
  # 2026-04-28 — see 实验笔记 §104). reset_and_auth_gate (in lib) enforces
  # B-224 hard-fail (no soft-warn fallthrough).
  if [[ "${RESET_BEFORE:-0}" == "1" ]] && wa_reset_supported "${BENCHMARK}" "${SITE}"; then
    reset_and_auth_gate --site "${SITE}" --repo "${REPO_DIR}" --python "${PYTHON_BIN}" --log-prefix "baseline" --reset-label "baseline_${MODE}_${SITE}"
  elif [[ "${RESET_BEFORE:-0}" == "1" ]]; then
    # B-647 (A1.13 P1-4-BC codex F7 + gemini G6 fix, 2026-05-17): hard-fail
    # instead of silent skip. Pre-fix WA paths printed "skipping" + proceeded
    # with stale auth/cart → paper-grade contamination invisible. reset_wa_sites.sh
    # scaffold lands here (returns rc=78 "not implemented" until Phase 1b impl).
    # To bypass paper-grade gate intentionally (e.g., explicit dirty dev run):
    # set RESET_BEFORE=0 + accept watchdog-reactive-only auth refresh.
    echo "[baseline][error] BENCHMARK=wa + RESET_BEFORE=1 unsupported for site=${SITE} (B-647 remainder; reddit IS supported)." >&2
    echo "[baseline][error] WA reddit routes to the VWA postmill reset (shared container); WA shopping/shopping_admin still need a Magento DB restore, same gap as VWA shopping." >&2
    echo "[baseline][error] To proceed: (a) implement reset_wa_sites.sh per its header roadmap, OR (b) set RESET_BEFORE=0 for explicit dirty run." >&2
    exit 1
  fi

  RUNNER_LOG="${LOG_DIR}/${RUN_ID}_runner.log"
  echo "[baseline] launching runner → ${RUNNER_LOG}"
  # B-1664-fu1 (Smoke 6 post-mortem 2026-05-18, replaces original B-1664):
  # original B-1664 set NO_GCE_CHECK=true (NOT a real google-auth env var; bogus)
  # + GCE_METADATA_HOST=disabled.invalid (sets _METADATA_ROOT path-only). But the
  # GCE residency probe is `ping()` in google/auth/compute_engine/_metadata.py,
  # which uses `_METADATA_IP_ROOT` derived from `GCE_METADATA_IP` (default
  # 169.254.169.254). Empirical Smoke 6 (B0_dom_classifieds_20260518 17:02:24-30):
  # still 3-retry × 3s = 9s startup waste despite original env. Correct fix:
  # GCE_METADATA_IP=127.0.0.1 → ping target unreachable on loopback → instant
  # ECONNREFUSED; GCE_METADATA_TIMEOUT=1 → each retry 1s (verified 0.28s real).
  # 9s startup waste × 36 Pass-1 conditions ≈ 5.4 min saved + log noise cleared.
  export GCE_METADATA_IP=127.0.0.1
  export GCE_METADATA_TIMEOUT=1
  export GCE_METADATA_HOST=disabled.invalid  # retain (defense-in-depth, blocks path-based fallback)
  # B-1824 (Fire-6 /stress P2-2): shared daemon spawn closes inherited paper-grade
  # lock fds 9/8/7 (supersedes B-1822 per-leaf hand-written redirects — single
  # helper kills the sibling-propagation drift; ORCH_FD/10 closed at the
  # orchestrator→chain boundary). flock binds the OFD not the fd number → an
  # inheriting setsid daemon would keep the chain site-lock alive past the
  # condition boundary (false double-fire ABORT, Fire-6 21:21:12Z). See
  # spawn_paper_grade_daemon in _lib_paper_grade_gates.sh.
  spawn_paper_grade_daemon 0 "${RUNNER_LOG}" -- \
    "${PYTHON_BIN}" scripts/run_experiment.py \
    --config "${CONFIG}" \
    --run_id "${RUN_ID}" \
    --log_path "${RUNNER_LOG}"
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

# Runner PID for watchdog self-exit — watchdog auto-exits when this PID dies
# AND condition_summary_v2.json present. Prevents init-orphan idle loops.
RUNNER_PID=$(pgrep -f "run_experiment.py.*${RUN_ID}" | head -1)

# B-1702 (/stress A2.12 P0-3-B* OOB codex unique, 2026-05-18, user Q3=A):
# capture RUNNER process group ID. setsid above makes the runner its own
# process group leader (PGID = runner_pid in nominal case, but `ps -o pgid=`
# is robust to fork/exec chain). Watchdog reads PGID via os.getpgid() at
# SIGTERM time; we record here for chain-level cleanup + audit-trail.
RUNNER_PGID=""
if [[ -n "${RUNNER_PID}" ]]; then
  RUNNER_PGID="$(ps -o pgid= -p "${RUNNER_PID}" 2>/dev/null | tr -d ' ')"
  echo "[baseline] runner pid=${RUNNER_PID} PGID=${RUNNER_PGID:-unknown} (B-1702 process-group SIGTERM ready)"
fi

# B-907 (/stress A2.2 P0-5-B* codex F1 OOB, 2026-05-17): per-RUN_ID flock
# closes pgrep-TOCTOU window letting two queue leaves spawn 2 watchdogs same
# RUN_ID + shared WD_STATE mutual overwrite. Lock on fd 8 (held until script
# end via existing trap chain). Skip-acquire path: queue_chain.sh leaf already
# attached; lock contention with another watchdog process → rc=78 FATAL.
if ! acquire_watchdog_lock "${RUN_ID}" "queue_baseline"; then
  exit $?
fi
# Extend existing release trap (acquire_site_lock set EXIT INT TERM at line 105)
trap "release_watchdog_lock; release_site_lock" EXIT INT TERM
if pgrep -f "experiment_watchdog.*${RUN_ID}" > /dev/null; then
  echo "[baseline] watchdog for ${RUN_ID} already running, skipping spawn"
else
  echo "[baseline] launching watchdog → ${WATCHDOG_LOG} (runner pid=${RUNNER_PID:-unknown})"
  # B-1824 (see runner note above): shared daemon spawn closes inherited lock fds.
  # The watchdog is the empirical B-1822 culprit — it out-lives the runner, so its
  # inherited fd 9 is what held the chain site-lock OFD at 21:21:12Z.
  spawn_paper_grade_daemon 0 "${WATCHDOG_LOG}" -- \
    "${PYTHON_BIN}" -u scripts/maintenance/experiment_watchdog.py \
    --run-dir "${RUN_DIR}" \
    --condition "${COND_ID}" \
    --poll-secs 30 \
    --idle-alert-mins "${EXP_WATCHDOG_IDLE_ALERT_MINS:-180}" \
    --ntfy-topic "${NTFY_TOPIC:-p79-exp-dgx-spark}" \
    --state-file "${WATCHDOG_STATE}" \
    --aggregate-prefix "${BASELINE}_3mode" \
    ${RUNNER_PID:+--runner-pid "${RUNNER_PID}"}
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
