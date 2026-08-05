#!/usr/bin/env bash
# B-1957 follow-on — 等手工 resume 的 dom 主臂跑完, 自动接上 shop_b0 的 cell 2-7。
#
# WHY 这个环节必须存在: shop_b0 是一条 7-cell chain, 但 2026-08-04 那次在 cell 1
# (dom) 的 320/435 处 abort 了。恢复它需要两种互斥的设置 ——
#   * dom 主臂要 RESET_BEFORE=0 (B-304: resume 时再 reset 会把前 320 个 episode
#     的脏轨迹和后 115 个的新鲜轨迹缝进同一个 condition_summary);
#   * 其余 6 个 cell 是全新的, 必须 RESET_BEFORE=1 (hard rule #2)。
# 而 `queue_chain.sh` 的 RESET_FLAG 是**整条 chain 一个值**, 满足不了两者;
# orchestrator 的 `launch_chain` 又把 FORCE_NEW=1 写死 (dom 会拿到新 run_id,
# 320 个 episode 白费); 唯一的 resume 开关 RESUME_MISSING=1 查 fire_manifest.json,
# 而其中 shopping 条目为 0, 对本站是空转的。
# 所以 dom 只能人工 resume, 而「人工 resume 完之后」到「其余 6 个 cell」之间
# 没有任何自动路径 —— 这个脚本就是那段路。
#
# 判据 (与 _cron_wa_shop_follow.sh 的分层相同, 但主判据换成数据本身):
#   手工 leaf resume **不写 chain sentinel**, 所以这里不能等 `.done`。
#   主判据 = dom 的 condition_summary_v2.json: episodes 达标 且 未 abort。
#   辅助 = runner 进程指纹, 只用来区分「还在跑」和「死了但数据不全」。
#
# 安全失败方向: 任何不确定都**不 fire**。数据不全而 runner 已消失 = 又一次
# abort 或被杀, 此时追加 6 个 condition 只会放大问题, 不会诊断问题。
set -uo pipefail
cd /home/ubuntu/workspace/p79 2>/dev/null || cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1

ARMED=logs/.shop_b0_tail_follow.armed
FIRED=logs/.shop_b0_tail_follow.fired
NTFY="${NTFY_TOPIC:-p79-exp-dgx-spark}"
COND_ID=phase1_dom_router_0
EXPECTED_N=435          # scored_task_count[shopping]; 与 queue_chain.sh 的表一致
log() { echo "[shop-tail $(date '+%m-%d %H:%M:%S')] $*"; }

# 见 _cron_wa_shop_follow.sh 的同名函数: pid 会在 11 天的 chain 里回绕,
# 单靠 pid 判活会命中一个无关的新进程。
_proc_starttime() {
  sed -e 's/^.*) //' "/proc/$1/stat" 2>/dev/null | awk '{print $20}'
}

# 方括号技巧: `[r]un_experiment` 这个字面量出现在本脚本自己的命令行里, 直接
# grep 'run_experiment' 会匹配到 cron 起的这个 shell 本身 (CLAUDE.md 记录的
# pgrep 自匹配坑, 本项目已踩三次)。
_live_dom_runner() {
  ps -eo pid=,args= 2>/dev/null | grep '[r]un_experiment\.py' | grep 'B0_dom_shopping' | head -1
}

[ -f "${FIRED}" ] && exit 0

# ── 尚未 arm: 只有见到活的 dom runner 才 arm ───────────────────────────────
# 不能无条件 arm: 那样会把一个早已结束的旧 run 当成要等的对象, 并在下一个 tick
# 直接 fire。
if [ ! -f "${ARMED}" ]; then
  LINE=$(_live_dom_runner)
  [ -z "${LINE}" ] && exit 0
  PID=$(echo "${LINE}" | awk '{print $1}')
  RUN_ID=$(echo "${LINE}" | grep -oE '\-\-run_id[= ]+[^ ]+' | head -1 | awk '{print $NF}' | tr -d '=')
  [ -z "${RUN_ID}" ] && { log "见到 dom runner pid=${PID} 但解析不出 run_id — 不 arm"; exit 0; }
  {
    echo "pid=${PID}"
    echo "starttime=$(_proc_starttime "${PID}")"
    echo "run_id=${RUN_ID}"
    echo "armed_at=$(date -Is)"
  } > "${ARMED}"
  log "ARMED — following dom resume pid=${PID} run_id=${RUN_ID}"
  exit 0
fi

ARM_PID=$(grep '^pid=' "${ARMED}" | head -1 | cut -d= -f2)
ARM_ST=$(grep '^starttime=' "${ARMED}" | head -1 | cut -d= -f2)
ARM_RUN=$(grep '^run_id=' "${ARMED}" | head -1 | cut -d= -f2)
[ -z "${ARM_RUN}" ] && { log "armed 文件缺 run_id — 丢弃重新 arm"; rm -f "${ARMED}"; exit 0; }

SUMMARY="results/visualwebarena/phase1/${ARM_RUN}/${COND_ID}/condition_summary_v2.json"

# ── 主判据: dom 的数据是否完整且干净 ──────────────────────────────────────
# 放在 pid 检查**之前**, 与 wa-follow 的 P0-1-A 分层一致: 数据齐了就不必关心
# pid 说什么 (它可能已被回绕复用)。
VERDICT=$(SUMMARY="${SUMMARY}" EXPECTED_N="${EXPECTED_N}" python3 - <<'PY' 2>/dev/null
import json, os, sys
p = os.environ["SUMMARY"]; want = int(os.environ["EXPECTED_N"])
try:
    d = json.load(open(p))
except Exception as exc:
    print(f"nosummary {exc.__class__.__name__}"); sys.exit(0)
eps = int(d.get("episodes", 0) or 0)
if d.get("condition_aborted"):
    print(f"aborted at_task={d.get('aborted_at_task')} reason={d.get('abort_reason')} eps={eps}")
elif eps >= want:
    print(f"complete eps={eps}")
else:
    print(f"partial eps={eps}/{want}")
PY
)
STATE=$(echo "${VERDICT}" | awk '{print $1}')

if [ "${STATE}" != "complete" ]; then
  NOW_ST=$(_proc_starttime "${ARM_PID}")
  if [ -n "${NOW_ST}" ] && [ "${NOW_ST}" = "${ARM_ST}" ]; then
    exit 0                      # 同一个 runner 还在跑 → 继续等 (安静, 不刷日志)
  fi
  # runner 没了而数据不全: 又一次 abort, 或被杀。**有意不 fire**。
  log "dom runner 已消失但数据未完成 (${VERDICT}) — 不 fire, 等待人工裁定"
  curl -sf -d "shop tail follow HELD: dom ${VERDICT} (run ${ARM_RUN})" "ntfy.sh/${NTFY}" >/dev/null 2>&1 || true
  touch "${FIRED}"              # 别每 10 分钟重复告警
  exit 1
fi

log "dom 主臂完成 (${VERDICT}, run=${ARM_RUN})"

# storefront 必须真的在服务, 才轮得到我们再去 reset 它
CODE=$(curl -sS -o /dev/null --max-time 15 -w '%{http_code}' http://localhost:7770/ 2>/dev/null || echo 000)
if [ "${CODE}" != "200" ]; then
  # 不在这里做重试循环 —— 10 分钟后的下一个 tick 就是重试, 重试逻辑归调度器。
  log "storefront ${CODE} — 本轮跳过, 10 分钟后重试"
  exit 0
fi

# 同 site 只能有一条 chain (hard rule #1 + /stress A2.11 P0-5)。dom runner 已确认
# 消失才走到这里, 但 chain/其它 runner 仍要再查一次。
if ps -eo args= 2>/dev/null | grep -q '[r]un_experiment\.py'; then
  log "仍有 runner 在跑 — 本轮跳过"
  exit 0
fi

TS=$(date +%Y%m%d_%H%M%S)
touch "${FIRED}"     # 先落 flag 再启动: 宁可漏启动也不要两条 chain 抢同一个 Magento
log "launching shop_b0_tail (cell 2-7)"
setsid nohup bash -c "source scripts/vwa_env_remote.sh 2>/dev/null; exec bash scripts/queues/queue_phase1_paper_grade.sh launch shop_b0_tail" \
  > "logs/orchestrator_shop_b0_tail_${TS}.log" 2>&1 < /dev/null &
curl -sf -d "shop_b0_tail chain launched (dom ${VERDICT}): logs/orchestrator_shop_b0_tail_${TS}.log" "ntfy.sh/${NTFY}" >/dev/null 2>&1 || true
log "done → logs/orchestrator_shop_b0_tail_${TS}.log"
