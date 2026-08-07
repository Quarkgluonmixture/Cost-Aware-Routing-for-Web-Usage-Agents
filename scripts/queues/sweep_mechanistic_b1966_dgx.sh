#!/usr/bin/env bash
# B-1966 重跑 · DGX 本机串行 sweep（Sparks 的补充/替代）。
#
# 为什么有这个脚本
# ================
# Sparks 只有两个节点，组里其他人也要用，所以那边的 array 默认 `%1`（只占一个）。
# 2026-08-07 进一步把 pending 的 cell 全部转到 DGX，让 Sparks 彻底空出来。
#
# 与 Slurm 版的关系：cell 清单同一份（`_mechanistic_cells.sh`），输出同一个目录名
# （`canonical_b1966fix`），完成判据同一个（`pilot_summary.md`）。两边可以任意分工，
# 靠 skip 逻辑天然幂等 —— 只要不在同一台机器上同时跑同一个 cell。
#
# ⚠️ DGX 是共享 GPU。实测 ~5.22h/cell（有争抢）vs Sparks ~3.2h（独占）。
# 一次只跑一个 cell，不并发 —— 并发只会让两个都变慢，还挤占别人。
#
# 用法:
#   bash scripts/queues/sweep_mechanistic_b1966_dgx.sh 16-23     # 跑 index 16..23
#   bash scripts/queues/sweep_mechanistic_b1966_dgx.sh 18,20,22  # 跑指定几个
#   DRY_RUN=1 bash scripts/queues/sweep_mechanistic_b1966_dgx.sh 16-23
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
source "$REPO/scripts/queues/_mechanistic_cells.sh"

SPEC="${1:?用法: $0 <index 范围, 如 16-23 或 18,20,22>}"
OUT_ROOT="$REPO/results/mechanistic/canonical_b1966fix"
LOGDIR="$REPO/logs/mechanistic_b1966fix"
SCRIPT="$REPO/scripts/mechanistic/run_stage2b_continuation_pilot.py"
SUB_CLS="$REPO/results/mechanistic/archive_subset_b1_cls_canonical"
SUB_RED="$REPO/results/mechanistic/archive_subset_b1_red_canonical"
NTFY="${NTFY_TOPIC:-p79-exp-dgx-spark}"
mkdir -p "$OUT_ROOT" "$LOGDIR"

# 解析 "16-23" 或 "18,20,22"
declare -a IDXS=()
if [[ "$SPEC" == *-* ]]; then
  for ((i=${SPEC%%-*}; i<=${SPEC##*-}; i++)); do IDXS+=("$i"); done
else
  IFS=',' read -ra IDXS <<< "$SPEC"
fi

export PYTORCH_NVML_BASED_CUDA_CHECK=1
export CUDA_MPS_PIPE_DIRECTORY=""
export CUDA_MPS_LOG_DIRECTORY=""

echo "[$(date '+%F %H:%M:%S')] DGX sweep — ${#IDXS[@]} 个 cell: ${IDXS[*]}"
RAN=0; SKIPPED=0; FAILED=0
for IDX in "${IDXS[@]}"; do
  [ "$IDX" -lt "${#CELLS[@]}" ] || { echo "  index $IDX 越界，跳过"; continue; }
  entry="${CELLS[$IDX]}"
  NAME="${entry%%|*}"; rest="${entry#*|}"
  SITE="${rest%%|*}"; rest="${rest#*|}"
  SUBKEY="${rest%%|*}"; EXTRA="${rest#*|}"
  [ "$SUBKEY" = "CLS" ] && SUBSET="$SUB_CLS" || SUBSET="$SUB_RED"
  OUT="$OUT_ROOT/$NAME"

  if [ -f "$OUT/pilot_summary.md" ]; then
    echo "[$(date '+%H:%M:%S')] ($IDX) $NAME — 已完成，skip"
    SKIPPED=$((SKIPPED+1)); continue
  fi
  if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "  ($IDX) $NAME [$SITE] $EXTRA"; continue
  fi

  # 前置断言 — 缺数据立刻失败，别跑 5 小时才发现读不到东西
  n_png=$(find "$SUBSET" -path "*step_002*" -name screenshot_annotated.png 2>/dev/null | wc -l)
  if [ "$n_png" -lt 15 ]; then
    echo "[$(date '+%H:%M:%S')] ($IDX) $NAME FATAL: $SUBSET 的 step_002 截图只有 $n_png 个"
    FAILED=$((FAILED+1)); continue
  fi

  mkdir -p "$OUT"
  echo "[$(date '+%H:%M:%S')] ($IDX) $NAME [$SITE] 启动 — $EXTRA"
  START=$(date +%s)
  # shellcheck disable=SC2086
  .venv/bin/python3 "$SCRIPT" \
    --site "$SITE" --step 2 --max-new-tokens 50 \
    --output-dir "$OUT" --archived-run-dir "$SUBSET" \
    $EXTRA > "$LOGDIR/${NAME}.log" 2>&1
  RC=$?
  MIN=$(( ($(date +%s) - START) / 60 ))
  if [ $RC -eq 0 ] && [ -f "$OUT/pilot_summary.md" ]; then
    RAN=$((RAN+1))
    echo "[$(date '+%H:%M:%S')] ($IDX) $NAME DONE ${MIN}min"
  else
    # rc=0 但没有 pilot_summary.md 也算失败 —— 那是「跑完了但没产出」，
    # 与「跑挂了」在下游表现相同，不能靠退出码区分（今天踩过四次的那个坑）。
    FAILED=$((FAILED+1))
    echo "[$(date '+%H:%M:%S')] ($IDX) $NAME FAILED rc=$RC summary=$([ -f "$OUT/pilot_summary.md" ] && echo 有 || echo 无) after ${MIN}min"
    tail -20 "$LOGDIR/${NAME}.log"
  fi
done

DONE_N=$(ls "$OUT_ROOT"/*/pilot_summary.md 2>/dev/null | wc -l)
MSG="DGX sweep 结束: ran=$RAN skip=$SKIPPED fail=$FAILED — canonical_b1966fix 共 $DONE_N/24"
echo "[$(date '+%F %H:%M:%S')] $MSG"
command -v curl >/dev/null && curl -s -d "$MSG" "ntfy.sh/${NTFY}" >/dev/null 2>&1 || true
[ "$FAILED" -eq 0 ]
