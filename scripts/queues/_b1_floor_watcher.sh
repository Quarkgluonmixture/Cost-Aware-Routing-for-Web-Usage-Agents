#!/usr/bin/env bash
# _b1_floor_watcher.sh — 接在正在跑的 B1 som classifieds replicate 后面, 把 B1 的
# same-condition rerun floor 从「不存在」补到「两站 × 六 mode」。
#
# WHY (2026-08-16, 笔记 §466 之后定):
#   08-13 四家 lineage 零预设 framing 收敛后, rerun floor 成了方法论中心 —— 标题那句
#   "the rows a router must learn from are the contested ones, which are the rows that
#   flip between identical reruns" 就架在它上面。但 `noise_floor_inventory §1` 里只有
#   4 个 pair: B0.cls.{dom,vision,som} (n=224) + B1.wa-red (n=50 的 pilot draw)。
#   两个缺口:
#     (a) DeepSeek 的最强反对 —— 完整地板只存在于**一个格** (B0×cls), "two-cell anecdote"
#     (b) 更要紧且此前没人提: 那三个 pair 全是 dom/som/vision, **没有一个 phantom 臂有
#         干净地板**, 而 paper 的 hero (drop-one oracle) 恰恰跑在 phantom 臂上。唯一的
#         P-SoM pair 是 restart partial (单向漂移 11.54%), 已被清单自己排除。
#   ⇒ 拿地板当尺子, 却没量过要量的那几条臂。
#
# WHY B1 AND NOT B2: B1/B2 都是本地模型, 复制跑 API 成本都是 $0。但 08-13 已裁定 B2 不排
#   —— 它每臂只有 1-8 个成功, 翻转太少, 测不出带。免费产能全给 B1。
#
# 判据 = 双条件 (§461.4 + CLAUDE.md done-monitor tiers)。PID 单独不够: 如果那条 chain 在
# 写完 condition_summary 之前就死了, PID 同样消失, 此时开火等于把地板建在半截数据上。
#   Tier 3: chain PID 消失
#   Tier 1: 该 replicate 自己的 condition_summary_v2.json 非空
#
# FORCE_NEW=1 是 LOAD-BEARING (B-1916 / §451.5): queue_chain.sh:394 读的是
# `FORCE_NEW="${FORCE_NEW:-0}"`, 不显式 export 的话 mint_run_id 会 resume-glob 到既有的
# canonical run —— 那样既拿不到第二次观测, 又污染了第一次, 而且看起来像跑过了。
set -uo pipefail
REPO="/home/ubuntu/workspace/p79"
cd "$REPO" || exit 1

CHAIN_PID="${1:?需要正在跑的 chain pid}"
WATCH_RUN="${2:?需要正在跑的 replicate run_id}"
NTFY="${NTFY_TOPIC:-p79-exp-dgx-spark}"
LOG="logs/b1_floor_watcher_$(date -u +%Y%m%d_%H%M%S).log"

say() { echo "[b1-floor $(date -u '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
push() { curl -s -m 20 -H "Title: $1" -d "$2" "https://ntfy.sh/${NTFY}" >/dev/null 2>&1 || true; }

say "armed: 等 chain pid ${CHAIN_PID} 退出 AND ${WATCH_RUN} 的 condition_summary_v2.json 落盘"

# 3 天上限: 那格 cls replicate 的 ETA 约 21h, 超出这么多说明卡住了, 该人看。
for _ in $(seq 1 864); do   # 864 x 300s = 3 days
  pid_gone=0; sentinel=0
  kill -0 "$CHAIN_PID" 2>/dev/null || pid_gone=1
  for f in results/visualwebarena/phase1/"${WATCH_RUN}"/*/condition_summary_v2.json; do
    [ -s "$f" ] && sentinel=1 && break
  done

  if [ "$pid_gone" = 1 ] && [ "$sentinel" = 1 ]; then
    say "双条件满足 -> 发车"
    break
  fi
  if [ "$pid_gone" = 1 ] && [ "$sentinel" = 0 ]; then
    say "chain pid 没了但 sentinel 缺失 —— 那格很可能 abort 了。**不开火**, 等人裁定。"
    push "B1 floor chain 未发车" "前一格 pid 消失但 condition_summary 缺失; 地板 chain 保持不动, 请查 ${WATCH_RUN}"
    exit 2
  fi
  sleep 300
done

if [ "${pid_gone:-0}" != 1 ] || [ "${sentinel:-0}" != 1 ]; then
  say "3 天上限到了双条件仍未满足 —— 放弃"
  push "B1 floor watcher 超时" "3 天未等到 ${WATCH_RUN} 收尾"
  exit 1
fi

# ---------- 发车 ----------------------------------------------------------------
# 顺序 = 先 classifieds 后 reddit。cls 先跑是因为它补齐的是与 B0×cls 逐 mode 对齐的
# 第二个完整格 —— 那是 "two-cell anecdote" 反对意见的直接解药, 先落地先能用。
# som 不在列表里: 正在跑的就是它。
# 一条 chain 内部是串行的, 所以任一时刻只有一个 site 活着, 满足 hard rule
# 「同一物理 host 同时只能跑一条 site chain」。
export FORCE_NEW=1
export RESET_BEFORE=1
TS=$(date -u +%Y%m%d_%H%M%S)

setsid nohup bash scripts/queues/queue_chain.sh \
  "queue_baseline.sh B1 dom classifieds" \
  "queue_baseline.sh B1 vision classifieds" \
  "queue_phantom_som.sh B1 classifieds" \
  "queue_phantom_text.sh B1 classifieds" \
  "queue_phantom_prompt.sh B1 classifieds" \
  "queue_baseline.sh B1 dom reddit" \
  "queue_baseline.sh B1 som reddit" \
  "queue_baseline.sh B1 vision reddit" \
  "queue_phantom_som.sh B1 reddit" \
  "queue_phantom_text.sh B1 reddit" \
  "queue_phantom_prompt.sh B1 reddit" \
  > "logs/queue_chain_b1_floor_${TS}.log" 2>&1 < /dev/null &

NEWPID=$!
say "B1 floor chain 已发车 (pid ${NEWPID}) -> logs/queue_chain_b1_floor_${TS}.log"
push "B1 floor chain 发车" \
  "11 格 same-condition replicate (cls 5 + red 6), 本地模型 \$0, ETA 约 13 天。补的是 phantom 臂第一次有干净地板 + 第二/第三个完整格。"
exit 0
