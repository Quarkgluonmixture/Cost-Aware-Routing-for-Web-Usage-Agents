#!/usr/bin/env bash
# Cron 版的 WA shopping 接续 —— `_launch_wa_shop_after_vwa.sh` 的无长驻替代。
#
# WHY: 原脚本用 `for _ in $(seq 1 28800); do sleep 60; done` 等 20 天。一个必须
# 连续存活 20 天的进程, 会死于 ssh 断连、机器重启、误杀、OOM —— 而它死掉是**静默**的:
# 十一天后 VWA chain 正常收尾, 没有任何东西接住 WA 的 7 个 condition, 也没人会收到通知。
# cron 每 10 分钟重新求值一次状态, 天然扛住上述全部情况。
#
# 状态机 (两个 flag 文件, 都在 logs/):
#   .wa_shop_follow.armed  首次见到活的 VWA chain 时写入 (记 pid + 当时的 sentinel mtime)
#   .wa_shop_follow.fired  已经启动过 WA chain, 之后所有调用直接退出 (幂等)
#
# 安全失败方向: 任何不确定都**不 fire**。宁可十一天后人工敲一条命令,
# 也不要在 substrate 异常时把 7 个 condition 追加上去。
set -uo pipefail
cd /home/ubuntu/workspace/p79 2>/dev/null || cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1

DONE=logs/queue_phase1_shop.latest.done
PIDF=logs/queue_phase1_shop.latest.pid
ARMED=logs/.wa_shop_follow.armed
FIRED=logs/.wa_shop_follow.fired
NTFY="${NTFY_TOPIC:-p79-exp-dgx-spark}"
log() { echo "[wa-follow $(date '+%m-%d %H:%M:%S')] $*"; }

[ -f "${FIRED}" ] && exit 0

CH=$(tr -d '[:space:]' < "${PIDF}" 2>/dev/null)

# ── 尚未 arm: 只有见到活的 VWA chain 才 arm ────────────────────────────────
# 不能无条件 arm —— 那样会把「上一轮早已结束的 chain」误当成要等的对象。
if [ ! -f "${ARMED}" ]; then
  if [ -n "${CH}" ] && kill -0 "${CH}" 2>/dev/null; then
    {
      echo "pid=${CH}"
      echo "done_mtime=$(stat -c %Y "${DONE}" 2>/dev/null || echo 0)"
      echo "armed_at=$(date -Is)"
    } > "${ARMED}"
    log "ARMED — following VWA shop chain pid=${CH}"
  fi
  exit 0
fi

ARM_PID=$(grep '^pid=' "${ARMED}" | head -1 | cut -d= -f2)
ARM_MTIME=$(grep '^done_mtime=' "${ARMED}" | head -1 | cut -d= -f2)

# VWA chain 还活着 → 继续等
kill -0 "${ARM_PID}" 2>/dev/null && exit 0

# ── chain 不在了: 必须看到**新的** done sentinel ───────────────────────────
# 被 SIGKILL 的 chain 写不出 sentinel → mtime 不变 → 永远不 fire。这是**有意的**:
# 非正常终止意味着 substrate 状态未知, 不该把下一条 chain 送进去。
NOW_MTIME=$(stat -c %Y "${DONE}" 2>/dev/null || echo 0)
if [ "${NOW_MTIME:-0}" -le "${ARM_MTIME:-0}" ]; then
  log "chain pid=${ARM_PID} 已消失但 sentinel 未更新 (可能被杀) — 不 fire, 等待人工裁定"
  exit 0
fi

RC=$(grep -oE 'rc=[0-9]+' "${DONE}" 2>/dev/null | head -1 | cut -d= -f2)
log "VWA chain finished, sentinel rc=${RC:-unknown}"
if [ "${RC:-1}" != "0" ]; then
  log "REFUSED: VWA chain exited rc=${RC:-unknown} — 在同一个容器上再追加 7 个 condition"
  log "  只会放大问题, 不会诊断问题。"
  curl -sf -d "WA shop follow-up REFUSED: VWA chain rc=${RC:-unknown}" "ntfy.sh/${NTFY}" >/dev/null 2>&1 || true
  touch "${FIRED}"   # 别每 10 分钟重复告警一次
  exit 1
fi

# storefront 必须真的在服务, 才轮得到我们再去 reset 它
CODE=$(curl -sS -o /dev/null --max-time 15 -w '%{http_code}' http://localhost:7770/ 2>/dev/null || echo 000)
if [ "${CODE}" != "200" ]; then
  # 单次 cron 调用不做 30 分钟重试 —— 下一个 tick (10min 后) 自然会重试,
  # 这正是 cron 相对长驻循环的好处: 重试逻辑由调度器承担, 不占进程。
  log "storefront ${CODE} — 本轮跳过, 10 分钟后重试"
  exit 0
fi

TS=$(date +%Y%m%d_%H%M%S)
touch "${FIRED}"     # 先落 flag 再启动: 宁可漏启动也不要重复启动两条 chain 抢同一容器
log "launching WA shopping B0 chain"
setsid nohup bash -c "source scripts/vwa_env_remote.sh 2>/dev/null; exec bash scripts/queues/queue_phase1_paper_grade.sh launch wa_shop_b0" \
  > "logs/orchestrator_wa_shop_b0_${TS}.log" 2>&1 < /dev/null &
curl -sf -d "WA shop B0 chain launched (follows VWA rc=0): logs/orchestrator_wa_shop_b0_${TS}.log" "ntfy.sh/${NTFY}" >/dev/null 2>&1 || true
log "done → logs/orchestrator_wa_shop_b0_${TS}.log"
