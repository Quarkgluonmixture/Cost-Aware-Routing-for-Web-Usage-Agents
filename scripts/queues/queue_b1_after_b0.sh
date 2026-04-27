#!/usr/bin/env bash
# queue_b1_after_b0.sh — wait for all B0 phantom chains to complete, then sequentially run B1 phantom chains
#
# B1 GPU 单 instance，cls 和 red chain 必须 sequential（不能 parallel）
# B0 cls/red/shop chains 同时跑（已启动），等它们 done 后启 B1
#
# 用法: nohup bash scripts/queues/queue_b1_after_b0.sh > logs/queue_b1_after_b0.log 2>&1 &

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_DIR}"

log() {
  echo "[b1-after-b0 $(date '+%H:%M:%S')] $*"
}

log "=================================================="
log "Waiting for all B0 phantom chains to complete..."
log "=================================================="

# Wait for B0 chains to all finish (queue_phantom_pair.sh B0 ...)
elapsed=0
while pgrep -f "queue_phantom_pair.sh B0" > /dev/null; do
  sleep 120
  elapsed=$((elapsed + 120))
  if (( elapsed % 1800 == 0 )); then
    log "  still waiting (${elapsed}s elapsed)..."
    pgrep -af "queue_phantom_pair.sh B0" | head -3 | sed 's/^/    /'
  fi
done

log "All B0 chains complete. Starting B1 chains sequentially..."
log ""
log "=================================================="
log "B1 cls chain (phantom_som → reset → phantom_dom)"
log "=================================================="
bash "${REPO_DIR}/scripts/queues/queue_phantom_pair.sh" B1 classifieds som,dom vwa
b1_cls_rc=$?
log "B1 cls chain exit code: ${b1_cls_rc}"

log ""
log "=================================================="
log "B1 reddit chain (phantom_som → reset → phantom_dom)"
log "=================================================="
bash "${REPO_DIR}/scripts/queues/queue_phantom_pair.sh" B1 reddit som,dom vwa
b1_red_rc=$?
log "B1 red chain exit code: ${b1_red_rc}"

log ""
log "=================================================="
log "All B0 + B1 VWA paper-grade chains DONE"
log "B1 shopping cells remain: 待 Myriad GPU 上线"
log "=================================================="

# ntfy notify
if command -v curl > /dev/null && [[ -n "${NTFY_TOPIC:-p79-exp-dgx-spark}" ]]; then
  curl -d "B1 phantom chains complete on cls + red. B0 + B1 paper-grade clean ready (except B1 shop pending Myriad)." \
    "ntfy.sh/${NTFY_TOPIC:-p79-exp-dgx-spark}" 2>/dev/null
fi
