#!/usr/bin/env bash
# queue_pilot_t0.sh — DEPRECATED (B-266 fix 2026-05-16, A1.7)
#
# Pilot abandoned per user directive 2026-05-16 — pilot superseded by Phase 1a
# fresh fire on A100 self-host. Pilot config yamls (exp_v2_B0_dom_pilot_T0_*.yaml)
# deleted. This script refuses to fire to prevent accidental pilot re-launch.
# Archive pilot data on results/visualwebarena/phase1/B0_*_pilot_T0_* preserved.
echo "[queue_pilot_t0.sh] DEPRECATED (B-266, A1.7) — pilot yamls deleted 2026-05-16."
echo "                    Pilot superseded by Phase 1a fresh fire (Pass-1 36 cond + Pass-2 6 router)."
echo "                    Archive pilot results preserved at results/.../B0_*_pilot_T0_*; do not re-fire."
exit 2

# --- legacy launcher body (unreachable, kept for git blame trace) ---
# B-37 Phase A pilot launcher (T=0 + RNG seeding sanity gate)
#
# Runs B0 DOM pilot on subset (30 tasks/site) with new T=0 + torch seed code path.
# Compares pilot SR vs existing paper-grade DOM SR to decide:
#   PASS:     within ±5pp existing SR   → green-light Phase A full re-run
#   MARGINAL: -5pp to -15pp             → tune top_p / consider mild T (0.05)
#   FAIL:     < -15pp or mode collapse  → revert T=0 → 0.1, paper takes disclosure path
#
# Usage:
#   bash scripts/queues/queue_pilot_t0.sh <site>
#     site: classifieds | reddit | shopping
#
# All 3 sites:
#   for s in classifieds reddit shopping; do
#     bash scripts/queues/queue_pilot_t0.sh "$s"
#   done
#
# After pilot completes:
#   .venv/bin/python3 scripts/analysis/compare_pilot_t0_vs_paper_grade.py
#
# WARNING: pilot uses real B0 proxy API (~$5-10 cost across 90 ep × ~10 step avg)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_DIR}"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <site:classifieds|reddit|shopping>" >&2
    exit 2
fi

SITE="$1"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_ID="B0_dom_pilot_T0_${SITE}_${TS}"
CONFIG="configs/exp_v2_B0_dom_pilot_T0_${SITE}.yaml"

if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: missing config $CONFIG" >&2
    exit 2
fi

# Idempotent skip (don't re-run if dir exists with episodes)
RUN_DIR="results/visualwebarena/phase1/${RUN_ID}"
if [[ -d "$RUN_DIR" ]] && find "$RUN_DIR" -name "*_summary_v2.json" -type f -print -quit | grep -q .; then
    echo "[$(date +%H:%M:%S)] PILOT $RUN_ID already has episodes — skip (delete dir to force re-run)"
    exit 0
fi

# Concurrency guard: don't start if another runner is on same site
if pgrep -af "run_experiment.*${SITE}" | grep -v "queue_pilot_t0" | grep -v "$$" >/dev/null; then
    echo "[$(date +%H:%M:%S)] WARN: another runner on site=${SITE} active. Aborting per §107 hard rule (B0 XOR B1 + same site exclusivity)."
    pgrep -af "run_experiment.*${SITE}" | grep -v "queue_pilot_t0" | grep -v "$$" || true
    exit 3
fi

# Load PROXY_API_KEY from .auth/qwen_api (file is NOT shell format —
# uses prefix-grep extraction, same as scripts/queues/queue_baseline.sh)
if [[ -z "${PROXY_API_KEY:-}" ]]; then
    AUTH_FILE="${REPO_DIR}/.auth/qwen_api"
    if [[ -f "${AUTH_FILE}" ]]; then
        raw_key="$(grep -m1 '^rp_' "${AUTH_FILE}" | tr -d '[:space:]')"
        if [[ -n "${raw_key}" ]]; then
            export PROXY_API_KEY="${raw_key}"
            export QWEN_API_KEY="${raw_key}"
            export DASHSCOPE_API_KEY="${raw_key}"
            echo "[$(date +%H:%M:%S)] Loaded PROXY_API_KEY from ${AUTH_FILE}"
        else
            echo "[$(date +%H:%M:%S)] ERROR: ${AUTH_FILE} has no rp_ key" >&2
            exit 1
        fi
    else
        echo "[$(date +%H:%M:%S)] ERROR: ${AUTH_FILE} missing and PROXY_API_KEY unset" >&2
        exit 1
    fi
fi

# VWA remote endpoints
if [[ -f "${REPO_DIR}/scripts/vwa_env_remote.sh" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${REPO_DIR}/scripts/vwa_env_remote.sh"
    set +a
fi

# DGX Spark CUDA workaround env
export PYTORCH_NVML_BASED_CUDA_CHECK=1
export CUDA_MPS_PIPE_DIRECTORY=""
export CUDA_MPS_LOG_DIRECTORY=""

# Wikipedia ZIM (legacy compat) — /stress A1.18 P0-2 2026-05-16: localhost default;
# reproducers set WIKIPEDIA_BASE_URL or source scripts/vwa_env_remote.sh to override.
export WIKIPEDIA_BASE_URL="${WIKIPEDIA_BASE_URL:-http://localhost:8888}"

# Site reset BEFORE pilot launches (paper-grade hard rule)
if [[ "${RESET_BEFORE:-1}" == "1" ]]; then
    case "$SITE" in
        classifieds)
            # /stress A1.18 P0-2 (2026-05-16): env-driven base URL
            curl -s -m 30 "${CLASSIFIEDS:-http://localhost:9980}/setup_db.sh" >/dev/null 2>&1 || \
                echo "[$(date +%H:%M:%S)] WARN: classifieds reset failed (non-fatal)"
            ;;
        reddit)
            curl -s -m 30 "${REDDIT:-http://localhost:9999}/setup_db.sh" >/dev/null 2>&1 || \
                echo "[$(date +%H:%M:%S)] WARN: reddit reset failed (non-fatal)"
            ;;
        shopping)
            echo "[$(date +%H:%M:%S)] shopping reset is heavyweight — relying on existing state for pilot"
            ;;
    esac
    echo "[$(date +%H:%M:%S)] reset attempted for ${SITE}"
fi

LOG_PATH="logs/${RUN_ID}.log"
mkdir -p logs

echo "[$(date +%H:%M:%S)] Launching B-37 pilot: ${RUN_ID}"
echo "  config: $CONFIG"
echo "  log:    $LOG_PATH"
echo "  T=0 + top_p=1.0 + seed=42 propagated to RNG (per Cluster 4 patches)"

# B-1824 (Fire-6 /stress A-F7): close inherited paper-grade lock fds 9/8/7 so this
# setsid daemon doesn't keep a chain/leaf/watchdog lock OFD alive past a condition
# boundary. pilot_t0 does not source _lib_paper_grade_gates.sh, so the close is
# inline here rather than via spawn_paper_grade_daemon.
setsid nohup "${REPO_DIR}/.venv/bin/python3" scripts/run_experiment.py \
    --config "$CONFIG" \
    --run_id "$RUN_ID" \
    --log_path "$LOG_PATH" \
    > "$LOG_PATH" 2>&1 < /dev/null 9>&- 8>&- 7>&- &

PID=$!
echo "[$(date +%H:%M:%S)] runner PID=$PID launched"
echo "$PID" > "logs/${RUN_ID}.pid"

# Don't start watchdog for pilot — pilot is short (≤2h), fail fast preferred over auto-recovery
echo "[$(date +%H:%M:%S)] pilot running in background. Tail log with:"
echo "  tail -f $LOG_PATH"
echo ""
echo "After completion, compare:"
echo "  .venv/bin/python3 scripts/analysis/compare_pilot_t0_vs_paper_grade.py --site $SITE"
