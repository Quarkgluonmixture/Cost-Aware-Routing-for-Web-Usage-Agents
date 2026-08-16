#!/usr/bin/env bash
# _b1_floor_watcher.sh — v2 (2026-08-16, /stress A+B+C 之后重写)
#
# 接在正在跑的 B1 som classifieds replicate 后面, 补 rerun floor 在 **phantom 臂**上的
# 空白。v1 于同日 arm 后被三家审计打穿, 本版逐条修掉; 差异见文件末尾 CHANGELOG。
#
# ── 为什么是这几格 (v1 排错了, 这是重排后的) ────────────────────────────────────
# 08-13 四家 lineage 收敛后 rerun floor 成了方法论中心, 但 `noise_floor_inventory §1`
# 只有 4 个 pair, 且三个全是 dom/som/vision ⇒ **没有一条 phantom 臂有干净地板**, 而
# drop-one hero 恰恰跑在 phantom 臂上。
#
# v1 选了"免费的 B1"去填。**那是错的**: 地板的可测性由 discordance d 决定, 而
# d ≈ n × SR × 0.59 (由三个已测 B0 pair 反推: 0.46/0.58/0.74)。逐格代入:
#
#     B1 cls phantom 臂 SR 6.25-7.59%  →  d ≈ 8.3-10.1   ← 低于 d<10 的可报门槛
#     B1 red 全部 6 臂  SR 2.46-7.39%  →  d ≈ 3.0-8.9    ← 全部低于门槛
#     B0 cls phantom 臂 SR 15.62-19.64% → d ≈ 20.7-26.1  ← 与已测三个 pair (27/32/29) 同级
#
# ⇒ "免费"和"有用"不是一回事。B0 cls 的三个 phantom 臂实测 $16.14/格 = **$48 买到
#   真有 power 的地板**, 而 B1 red 六格要 8 天 wall-clock 换六个不可报的数。
# 本版: B0 cls 三个 phantom 臂**先跑**(付费的先落地, 出问题第 2 天就知道而不是第 10 天),
#       之后接 B1 cls 五格(免费, vision d≈16.6 有 power, 其余作描述性证据不报 CI)。
# B0 **reddit** 的三个 phantom 臂 (d≈13.0-15.9, $66) **刻意不进本链** —— 等 cls 的结果
# 落地再决定值不值, 不预支。B1 red (d<10) 同理排除。
#
# ── 单站 ────────────────────────────────────────────────────────────────────
# 本链全部是 classifieds。v1 把 cls 和 red 混在一条链里, 于是无法向 host-global lease
# 声明一个 self_site。单站也让下面的 lease 有意义。
set -uo pipefail
REPO="/home/ubuntu/workspace/p79"
cd "$REPO" || exit 1

CHAIN_PID="${1:?需要正在跑的 chain pid}"
WATCH_RUN="${2:?需要正在跑的 replicate run_id}"
WATCH_COND="${3:-phase1_som_router_0}"
EXPECT_N="${4:-224}"                       # classifieds scored universe
NTFY="${NTFY_TOPIC:-p79-exp-dgx-spark}"
TS="$(date -u +%Y%m%d_%H%M%S)"
LOG="logs/b1_floor_watcher_${TS}.log"
STATE="logs/.b1_floor_${TS}"               # .started / .done / .failed 落在这个前缀下

say()  { echo "[b1-floor $(date -u '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
push() { curl -s -m 20 -H "Title: $1" -d "$2" "https://ntfy.sh/${NTFY}" >/dev/null 2>&1 || true; }

# ── PID identity (B-1975): 数字 pid 不足以标识进程 ───────────────────────────
# A100 实测 pid_max=4,194,304, churn 均值 9.04 PID/s / 峰值 22.97 PID/s ⇒ 绕回一圈
# 50.7-128.9h。本 watcher 的等待窗口远小于此, 但 v1 只存数字 pid 就没有任何办法
# 区分"原进程还活着"与"pid 被复用了"。starttime 是 /proc/<pid>/stat 字段 22;
# comm 在括号里且可能含空格 → 先剥到最后一个右括号之后, 剥完 state 变 $1, 故 22→20。
_starttime() { sed -e 's/^.*) //' "/proc/$1/stat" 2>/dev/null | awk '{print $20}'; }
_cmdline()   { tr '\0' ' ' < "/proc/$1/cmdline" 2>/dev/null; }

ARM_ST="$(_starttime "$CHAIN_PID")"
ARM_CMD="$(_cmdline "$CHAIN_PID")"
if [ -z "$ARM_ST" ]; then
  say "FATAL: pid ${CHAIN_PID} 在 arm 时就不存在"; push "B1 floor watcher 未 arm" "pid ${CHAIN_PID} 不存在"; exit 3
fi
case "$ARM_CMD" in
  *queue_chain.sh*) : ;;
  *) say "FATAL: pid ${CHAIN_PID} 的 cmdline 不是 queue_chain: ${ARM_CMD}"; exit 3 ;;
esac

_chain_alive() {
  local now_st; now_st="$(_starttime "$CHAIN_PID")"
  [ -n "$now_st" ] && [ "$now_st" = "$ARM_ST" ]
}

# ── 完成判据 (B-1974): 与 queue_chain.sh:522-560 的 C3 validator 同一套判据 ────
# v1 只做 `[ -s file ]`。实测反例: 今天 15:26 abort 的 B0_dom_wa_shopping 写出了
# **3191 字节、`episodes: 0`** 的 condition_summary_v2.json —— `-s` 会放行。
# 这里复刻 C3 的三条: JSON 可解析 / condition_id 匹配 / episodes 精确等于 expected。
# (刻意复刻而非 source: queue_chain 的 C3 是内联在循环里的 heredoc, 抽出来要动
#  fire-path 主文件; 复刻 + 指针注释是本轮更小的改动面。若将来抽公共函数, 这里换掉。)
_condition_complete() {  # <run_id> <cond_id> <expected_n>
  local run_id="$1" cond_id="$2" exp_n="$3" f
  for base in results/visualwebarena/phase1 results/webarena/phase1; do
    f="${REPO}/${base}/${run_id}/${cond_id}/condition_summary_v2.json"
    [ -s "$f" ] || continue
    EXPECTED_CID="$cond_id" EXPECTED_N="$exp_n" SUMMARY_PATH="$f" python3 -c "
import json, os, sys
try: d = json.load(open(os.environ['SUMMARY_PATH']))
except Exception as e: print(f'invalid JSON: {e}', file=sys.stderr); sys.exit(1)
cid = d.get('condition_id', '')
if cid and cid != os.environ['EXPECTED_CID']:
    print(f'condition_id mismatch: {cid!r}', file=sys.stderr); sys.exit(2)
ep = d.get('episodes', d.get('total_tasks', d.get('num_tasks', d.get('scored_task_count', 0))))
if not isinstance(ep, int) or ep <= 0:
    print(f'episodes invalid: {ep!r}', file=sys.stderr); sys.exit(3)
n = int(os.environ['EXPECTED_N'])
if ep != n:
    print(f'episodes={ep} != expected={n}', file=sys.stderr); sys.exit(4)
sys.exit(0)
" 2>>"$LOG" && return 0
  done
  return 1
}

say "armed v2: 等 pid ${CHAIN_PID} (starttime=${ARM_ST}) 退出 AND ${WATCH_RUN}/${WATCH_COND} 达 ${EXPECT_N} episode"

# ── 等待 ─────────────────────────────────────────────────────────────────────
# v1 的 864 次循环在最后一次检查后仍 sleep 300 才退出, 那 300 秒里完成的话会被误报
# timeout。本版把判据抽成函数, 循环结束后**再查一次**。
DEADLINE=$(( $(date +%s) + 3*24*3600 ))
fired=0
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  if ! _chain_alive; then
    if _condition_complete "$WATCH_RUN" "$WATCH_COND" "$EXPECT_N"; then fired=1; break; fi
    say "chain 已退出但 ${WATCH_RUN}/${WATCH_COND} 未达 ${EXPECT_N} episode —— **不开火**, 等人裁定"
    push "B1 floor chain 未发车" "前一格未完整收尾 (非 ${EXPECT_N} episode); 不自动发车, 请查 ${WATCH_RUN}"
    exit 2
  fi
  sleep 300
done
# 循环出口再查一次 (修 v1 的最后 300 秒盲区)
if [ "$fired" -eq 0 ]; then
  if ! _chain_alive && _condition_complete "$WATCH_RUN" "$WATCH_COND" "$EXPECT_N"; then
    fired=1
  else
    say "3 天上限到, 双条件未满足 —— 放弃"; push "B1 floor watcher 超时" "3 天未等到 ${WATCH_RUN} 收尾"; exit 1
  fi
fi
say "双条件满足 (pid 消失 + ${EXPECT_N}/${EXPECT_N} episode) -> 准备发车"

# ── host-global lease (B-1973) ───────────────────────────────────────────────
# v1 的注释断言"一条 chain 内部串行 ⇒ 满足同 host 单 site chain 硬规则"。错。
# queue_chain 只取 container/site lock 且每格后释放, **从不调用**
# `assert_no_other_site_chain_running` (`_lib_paper_grade_gates.sh:1071`; 调用者
# 只有两个 queue_phase1_*)。该 lib 的 line 127 自己写着 "each got the same thing
# wrong" —— 这是有案底的重复错误。
# 两个方向都要堵: (a) 发车前问一次有没有别的站在跑; (b) 写自己的 pidfile, 让将来的
# queue_phase1_* 看得见我。pidfile 用 trap 清理, 免得残留把未来的发车永久挡住。
# shellcheck disable=SC1091
source "${REPO}/scripts/queues/_lib_paper_grade_gates.sh" 2>/dev/null || true
if declare -F assert_no_other_site_chain_running >/dev/null; then
  if ! assert_no_other_site_chain_running cls "b1-floor"; then
    say "REFUSED: 另一条 site chain 在跑, 不发车"; push "B1 floor chain 未发车" "另有 site chain 活着"; exit 4
  fi
  say "host-global lease OK (无其它 site chain)"
else
  say "WARN: assert_no_other_site_chain_running 不可用, 跳过 lease 检查"
fi

PIDFILE="${REPO}/logs/queue_phase1_cls.latest.pid"
cleanup() { [ -n "${PIDFILE:-}" ] && rm -f "$PIDFILE" 2>/dev/null; }
trap cleanup EXIT

# ── 发车 ─────────────────────────────────────────────────────────────────────
# FORCE_NEW=1 是 LOAD-BEARING (B-1916/§451.5): queue_chain.sh:394 读的是
# `FORCE_NEW="${FORCE_NEW:-0}"`, 不显式 export 就会 resume-glob 到既有 canonical run
# —— 既拿不到第二次观测又污染第一次, 而且看起来像跑过了。
# `RESET_BEFORE` **不再 export**: queue_chain.sh:65 自设 RESET_FLAG=1 并在 :394 用
# `RESET_BEFORE="${RESET_FLAG}"` 覆盖叶子 env ⇒ 父层的 export 无效。v1 那句是
# declaration drift, 会让未来的操作者以为能从 watcher 控制 reset。已实测确认删除安全。
export FORCE_NEW=1
CTS=$(date -u +%Y%m%d_%H%M%S)
CHAINLOG="logs/queue_chain_b1_floor_${CTS}.log"

setsid nohup bash scripts/queues/queue_chain.sh \
  "queue_phantom_text.sh B0 classifieds" \
  "queue_phantom_prompt.sh B0 classifieds" \
  "queue_phantom_som.sh B0 classifieds" \
  "queue_baseline.sh B1 vision classifieds" \
  "queue_baseline.sh B1 dom classifieds" \
  "queue_phantom_som.sh B1 classifieds" \
  "queue_phantom_text.sh B1 classifieds" \
  "queue_phantom_prompt.sh B1 classifieds" \
  > "$CHAINLOG" 2>&1 < /dev/null &
NEWPID=$!
echo "$NEWPID" > "$PIDFILE"

# ── child acknowledgement (B-1976) ───────────────────────────────────────────
# v1 读了 `$!` 就推 ntfy 说"发车"然后 exit 0。child 若因 lock / config / reset / auth
# 秒死, 通知照发, 之后整条长链没有任何 done-monitor —— 正是"退出码 0, 做的不是你以为
# 的那件事"。本版等它 settle 再确认, 并落 .started / .done / .failed receipt。
sleep 60
NEW_ST="$(_starttime "$NEWPID")"
if [ -z "$NEW_ST" ]; then
  say "FATAL: chain (pid ${NEWPID}) 在 60s 内就死了 —— 未发车"
  tail -30 "$CHAINLOG" >> "$LOG" 2>/dev/null
  echo "chain died within 60s; see $CHAINLOG" > "${STATE}.failed"
  push "B1 floor chain 秒死" "pid ${NEWPID} 60s 内退出, 见 ${CHAINLOG}"
  exit 5
fi
{ echo "pid=${NEWPID}"; echo "starttime=${NEW_ST}"; echo "log=${CHAINLOG}"; echo "started_at=$(date -Is)"; } > "${STATE}.started"
say "chain 已确认存活 (pid ${NEWPID}, starttime ${NEW_ST}) -> ${CHAINLOG}"
push "B1 floor chain 发车" \
  "8 格 cls: B0×3 phantom 臂 (\$48, d≈21-26 有 power) + B1×5 (免费)。约 7 天。收后需 DGX 侧 sync_a100_results.sh + 注册 CLEAN_PAIRS。"

# ── 守到底 ───────────────────────────────────────────────────────────────────
# 本 watcher 现在**就是**这条长链的 done-monitor (v1 没有)。identity-checked 轮询,
# 不用裸 `kill -0`。
while :; do
  now_st="$(_starttime "$NEWPID")"
  [ -n "$now_st" ] && [ "$now_st" = "$NEW_ST" ] || break
  sleep 300
done

# 计数只认**本链自己 mint 的 run_id**, 从 chain log 里取。
# 不能用 `results/.../B0_phantom_text_classifieds_*` 这种 glob —— 它会先命中
# canonical 的那个旧 run (B0 是 20260526, B1 是 2026-06-0x), 于是即使新格一个没跑,
# 计数照样凑到 8/8。这正是本文件要防的那一类"看起来完成了"。
DONE_N=0; SEEN=0
while read -r rid cid; do
  [ -n "$rid" ] || continue
  SEEN=$((SEEN+1))
  _condition_complete "$rid" "$cid" "$EXPECT_N" && DONE_N=$((DONE_N+1))
done < <(grep -oE 'run_id=[A-Za-z0-9_.-]+ condition=[A-Za-z0-9_]+' "$CHAINLOG" 2>/dev/null \
         | sed -E 's/run_id=([^ ]+) condition=(.*)/\1 \2/' | sort -u)
say "chain log 报告 mint 了 ${SEEN} 个 condition; 其中 ${DONE_N} 个通过 exact-N 校验"

if [ "$DONE_N" -eq 8 ]; then
  { echo "cells_complete=8"; echo "log=${CHAINLOG}"; echo "ended_at=$(date -Is)"; } > "${STATE}.done"
  say "chain 收尾: 8/8 格完整"
  push "B1 floor chain 完成 8/8" "下一步在 DGX: bash scripts/maintenance/sync_a100_results.sh 然后把新 run 注册进 aggregate_noise_floor_inventory.py 的 CLEAN_PAIRS"
else
  { echo "cells_complete=${DONE_N}/8"; echo "log=${CHAINLOG}"; echo "ended_at=$(date -Is)"; } > "${STATE}.failed"
  say "chain 收尾: 只有 ${DONE_N}/8 格完整"
  push "B1 floor chain 部分完成 ${DONE_N}/8" "见 ${CHAINLOG}; 不要当完整证据用"
fi
exit 0

# ── CHANGELOG v1 → v2 (2026-08-16, /stress A+B+C) ────────────────────────────
# B-1972  chain list 重排: v1 的 11 格 (B1 cls 5 + B1 red 6) 里有 9 格 d<10 不可报;
#         换成 B0 cls 3 个 phantom 臂 (d≈21-26, $48) + B1 cls 5 格。B0 red / B1 red 移出。
# B-1973  加 host-global lease (assert_no_other_site_chain_running) + 自己的 pidfile;
#         v1 的注释把"chain 内部串行"误当成满足硬规则。
# B-1974  完成判据由 `[ -s ]` 换成 queue_chain C3 同款 (JSON + condition_id + exact-N);
#         实证反例 = 3191 字节 / episodes:0 的 abort summary。
# B-1975  kill -0 换成 starttime(/proc/pid/stat 字段 22) + cmdline 双核对。
# B-1976  加 .started/.done/.failed receipt + 60s settle 检查 + 守到链结束当 done-monitor。
# B-1977  修 3 天上限最后 300 秒盲区 (循环出口再查一次)。
# B-1978  删掉无效的 `export RESET_BEFORE=1` (queue_chain.sh:65+:394 会覆盖叶子 env)。
