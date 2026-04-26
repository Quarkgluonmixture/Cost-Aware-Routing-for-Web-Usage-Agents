#!/usr/bin/env bash
# wait_for_reddit_then_rederive.sh — wait for B1 reddit SoM run to finish,
# then rederive episode summaries with §97-audit fixed code.
#
# Designed to be launched once and left in background:
#   setsid nohup bash scripts/maintenance/wait_for_reddit_then_rederive.sh \
#     > logs/wait_reddit_followup.log 2>&1 < /dev/null &
#
# Behavior:
#   1. Polls for the Python runner PID matching b1_3mode_reddit_som every 5 min.
#   2. When the process exits, sleeps 30s for filesystem flush.
#   3. Runs scripts/maintenance/rederive_episode_summary.py on the B1 reddit run dir.
#   4. Touches logs/reddit_rederive_done.flag for the user to notice.
#   5. Does NOT auto-start B1 shopping queue — that requires user decision
#      about handling the partial SoM (331/466) under new max_marks=200.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_DIR}"

# Match the actual reddit SoM runner command pattern.
PROCESS_PATTERN="b1_3mode_reddit_som"
RUN_DIR="results/visualwebarena/phase1/B1_3mode_reddit_20260413"
POLL_INTERVAL=300  # 5 minutes
DONE_FLAG="logs/reddit_rederive_done.flag"

echo "[$(date -Iseconds)] wait_for_reddit_then_rederive: starting"
echo "  Watching for process: ${PROCESS_PATTERN}"
echo "  Run dir: ${RUN_DIR}"
echo "  Poll interval: ${POLL_INTERVAL}s"

# Step 1: wait for the runner process to exit.
while pgrep -f "${PROCESS_PATTERN}" > /dev/null; do
    n_episodes=$(ls "${RUN_DIR}/phase1_som_router_0/episodes/"*_summary_v2.json 2>/dev/null | wc -l)
    echo "[$(date -Iseconds)] still running, ${n_episodes}/210 SoM episodes done; sleeping ${POLL_INTERVAL}s"
    sleep "${POLL_INTERVAL}"
done

echo "[$(date -Iseconds)] reddit SoM process exited; sleeping 30s for fsync"
sleep 30

# Step 2: re-derive episode summaries using §97-audit fixed code.
echo "[$(date -Iseconds)] running rederive_episode_summary.py on ${RUN_DIR}"
.venv/bin/python3 scripts/maintenance/rederive_episode_summary.py --run-dir "${RUN_DIR}" 2>&1

# Step 3: signal rederive done.
mkdir -p logs
date -Iseconds > "${DONE_FLAG}"
echo "[$(date -Iseconds)] reddit rederive complete; flag=${DONE_FLAG}"

# Step 4: launch B1 shopping queue.
# B1 shopping was already cleaned (SoM 0/466, Vision 0/466, DOM 466/466 retained).
# Queue will resume DOM (skip already-done), run SoM clean from 0, then Vision from 0.
echo "[$(date -Iseconds)] launching B1 shopping queue (clean SoM + Vision)"
SHOPPING_LOG="logs/queue_b1_shopping_$(date +%Y%m%d_%H%M%S).log"
B1_SITE=shopping setsid nohup bash scripts/queues/queue_b1_with_reset.sh \
    > "${SHOPPING_LOG}" 2>&1 < /dev/null &
echo "[$(date -Iseconds)] B1 shopping queue launched (log: ${SHOPPING_LOG})"
date -Iseconds > logs/shopping_queue_started.flag
