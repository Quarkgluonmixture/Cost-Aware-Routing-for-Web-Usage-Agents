#!/usr/bin/env bash
# Auto-follow: fire the WA shopping B0 chain once the VWA one finishes.
#
# WHY THIS EXISTS: the two shop chains cannot overlap — VWA shopping, WA
# shopping and WA shopping_admin are all the SAME `vwa-shopping` container
# (7770 storefront + 7780 admin), so the container lock (B-1934) refuses the
# second one and CLAUDE.md hard rule #3 forbids it anyway. The VWA chain runs
# ~11.3 days. Without this, someone has to remember, eleven days later, to type
# one command. Precedent: `_launch_wa_full_reddit.sh` (B-1914) did the same for
# reddit, and its post-mortem is why every chain step below is quoted.
#
# Waits on the DONE SENTINEL, not on liveness: `logs/queue_phase1_shop.latest.done`
# carries `rc=N`, and a chain that aborted must NOT be followed by another one
# on the same container — the abort usually means the substrate is unhappy.
set -uo pipefail
cd /home/ubuntu/workspace/p79 2>/dev/null || cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1

DONE=logs/queue_phase1_shop.latest.done
NTFY="${NTFY_TOPIC:-p79-exp-dgx-spark}"
log() { echo "[wa-follow $(date '+%m-%d %H:%M:%S')] $*"; }

# The VWA chain must already be running, or we would fire into an empty slot and
# race whatever launches next.
CH=$(cat logs/queue_phase1_shop.latest.pid 2>/dev/null | tr -d '[:space:]')
if [ -z "${CH}" ] || ! kill -0 "${CH}" 2>/dev/null; then
  log "REFUSED: no live VWA shop chain to follow (pid='${CH:-none}')"
  exit 4
fi
START_DONE=$(stat -c %Y "${DONE}" 2>/dev/null || echo 0)
log "following chain pid=${CH}; waiting for a NEW ${DONE}"

# 20 days of headroom over the ~11.3-day estimate.
for _ in $(seq 1 28800); do
  if ! kill -0 "${CH}" 2>/dev/null; then
    NOW_DONE=$(stat -c %Y "${DONE}" 2>/dev/null || echo 0)
    [ "${NOW_DONE}" -gt "${START_DONE}" ] && break
    sleep 30
    continue
  fi
  sleep 60
done

RC=$(grep -oE 'rc=[0-9]+' "${DONE}" 2>/dev/null | head -1 | cut -d= -f2)
log "VWA chain finished, sentinel rc=${RC:-unknown}"
if [ "${RC:-1}" != "0" ]; then
  log "REFUSED to launch WA: VWA chain exited rc=${RC:-unknown}. A non-zero exit on"
  log "  this container usually means the substrate is unhappy; chaining another"
  log "  7 conditions onto it would compound the problem, not test it."
  curl -sf -d "WA shop follow-up REFUSED: VWA chain rc=${RC:-unknown}" "ntfy.sh/${NTFY}" >/dev/null 2>&1 || true
  exit 1
fi

# Storefront must actually be serving before we start resetting it again.
for _ in $(seq 1 60); do
  [ "$(curl -sS -o /dev/null --max-time 10 -w '%{http_code}' http://localhost:7770/ 2>/dev/null)" = "200" ] && break
  sleep 30
done
CODE=$(curl -sS -o /dev/null --max-time 10 -w '%{http_code}' http://localhost:7770/ 2>/dev/null || echo 000)
if [ "${CODE}" != "200" ]; then
  log "REFUSED: storefront ${CODE} after VWA chain — not launching WA onto a sick site"
  curl -sf -d "WA shop follow-up REFUSED: storefront ${CODE}" "ntfy.sh/${NTFY}" >/dev/null 2>&1 || true
  exit 1
fi

TS=$(date +%Y%m%d_%H%M%S)
log "launching WA shopping B0 chain"
setsid nohup bash -c "source scripts/vwa_env_remote.sh 2>/dev/null; exec bash scripts/queues/queue_phase1_paper_grade.sh launch wa_shop_b0" \
  > "logs/orchestrator_wa_shop_b0_${TS}.log" 2>&1 < /dev/null &
curl -sf -d "WA shop B0 chain launched (follows VWA rc=0): logs/orchestrator_wa_shop_b0_${TS}.log" "ntfy.sh/${NTFY}" >/dev/null 2>&1 || true
log "done → logs/orchestrator_wa_shop_b0_${TS}.log"
