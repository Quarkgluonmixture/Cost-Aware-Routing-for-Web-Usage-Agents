#!/usr/bin/env bash
# Cron 版的 WA shopping 接续 —— `_launch_wa_shop_after_vwa.sh` 的无长驻替代。
#
# WHY: 原脚本用 `for _ in $(seq 1 28800); do sleep 60; done` 等 20 天。一个必须
# 连续存活 20 天的进程, 会死于 ssh 断连、机器重启、误杀、OOM —— 而它死掉是**静默**的:
# 十一天后 VWA chain 正常收尾, 没有任何东西接住 WA 的 7 个 condition, 也没人会收到通知。
# cron 每 10 分钟重新求值一次状态, 天然扛住上述全部情况。
#
# 状态机 (两个 flag 文件, 都在 logs/):
#   .wa_shop_follow.armed  首次见到活的 VWA chain 时写入 (pid + starttime 指纹 + sentinel mtime)
#   .wa_shop_follow.fired  已经启动过 WA chain, 之后所有调用直接退出 (幂等)
#
# 判据层级 (P0-1-A 修正 2026-08-04, /stress milestone):
#   **sentinel 是主判据, pid 只是辅助**。原实现把 `kill -0 $pid` 挡在 sentinel 检查
#   前面 —— pid 一旦误判「活着」, 即使 chain 已正常结束并写好 sentinel 也永远走不到 fire。
#   现在先看 sentinel 有没有更新 (更新了就不管 pid 说什么), 没更新才用 pid 区分
#   「还在跑」vs「被杀了」。这样即使 pid 指纹失效, sentinel 仍是一条独立触发路径。
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

# 进程身份指纹 = /proc/<pid>/stat 字段 22 (starttime, 进程创建时刻的 jiffies)。
# **pid 本身不足以标识进程**: A100 实测 pid_max=4194304, 消耗约 6.3 万/天,
# 而 VWA chain 要跑 11 天 (~70 万) —— PID 必然回绕。回绕后 `kill -0 <旧pid>`
# 会命中一个全新的无关进程, 于是 cron 永远认为 chain 还活着, WA 静默不启动。
# comm 字段在括号里且可能含空格 → 先剥到最后一个右括号之后再取字段;
# 剥完后 state 变成 $1 (原 $3), 所以 starttime (原 $22) = $20。
_proc_starttime() {
  sed -e 's/^.*) //' "/proc/$1/stat" 2>/dev/null | awk '{print $20}'
}

[ -f "${FIRED}" ] && exit 0

CH=$(tr -d '[:space:]' < "${PIDF}" 2>/dev/null)

# ── 尚未 arm: 只有见到活的 VWA chain 才 arm ────────────────────────────────
# 不能无条件 arm —— 那样会把「上一轮早已结束的 chain」误当成要等的对象。
if [ ! -f "${ARMED}" ]; then
  if [ -n "${CH}" ] && kill -0 "${CH}" 2>/dev/null; then
    {
      echo "pid=${CH}"
      echo "starttime=$(_proc_starttime "${CH}")"
      echo "done_mtime=$(stat -c %Y "${DONE}" 2>/dev/null || echo 0)"
      echo "armed_at=$(date -Is)"
    } > "${ARMED}"
    log "ARMED — following VWA shop chain pid=${CH} starttime=$(_proc_starttime "${CH}")"
  fi
  exit 0
fi

ARM_PID=$(grep '^pid=' "${ARMED}" | head -1 | cut -d= -f2)
ARM_ST=$(grep '^starttime=' "${ARMED}" | head -1 | cut -d= -f2)
ARM_MTIME=$(grep '^done_mtime=' "${ARMED}" | head -1 | cut -d= -f2)

# 旧格式 .armed (无 starttime 指纹) → 无法可靠判活, 丢弃重新 arm。
if [ -z "${ARM_ST}" ]; then
  log "armed 文件为旧格式 (无 starttime 指纹) — 丢弃, 下个 tick 重新 arm"
  rm -f "${ARMED}"
  exit 0
fi

# ── 主判据: sentinel 是否更新 ──────────────────────────────────────────────
# 放在 pid 检查**之前**: sentinel 更新 = chain 确实正常结束, 此时 pid 说什么都不重要
# (它可能已被回绕复用)。
NOW_MTIME=$(stat -c %Y "${DONE}" 2>/dev/null || echo 0)
if [ "${NOW_MTIME:-0}" -le "${ARM_MTIME:-0}" ]; then
  # sentinel 未更新 —— 用 pid 指纹区分「还在跑」vs「被杀了」
  NOW_ST=$(_proc_starttime "${ARM_PID}")
  if [ -n "${NOW_ST}" ] && [ "${NOW_ST}" = "${ARM_ST}" ]; then
    exit 0                     # 同一个进程还活着 → 继续等
  fi
  # 进程没了 (或 pid 已被别的进程复用): 被 SIGKILL 的 chain 写不出 sentinel。
  # **有意不 fire** —— 非正常终止意味着 substrate 状态未知。
  log "chain pid=${ARM_PID} 指纹不匹配/已消失, 且 sentinel 未更新 (可能被杀) — 不 fire, 等待人工裁定"
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
