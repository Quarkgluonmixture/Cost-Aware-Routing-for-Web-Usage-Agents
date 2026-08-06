#!/usr/bin/env bash
#SBATCH --job-name=p79-b1966
#SBATCH --array=0-23%2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=/clusterhome/jiaming/p79/logs/slurm/%x_%A_%a.out
#SBATCH --error=/clusterhome/jiaming/p79/logs/slurm/%x_%A_%a.out
#
# B-1966 重跑 — mechanistic canonical sweep, 在 Holistic AI Sparks 上跑。
#
# 为什么要重跑 24 个 cell
# =======================
# `run_stage2b_continuation_pilot.py` 的 source 侧原先**无条件**喂标注截图, 不看
# `--source-mode`。由于 `som` 与 `phantom_som` 共享 [SOM_MARKS] 文本 payload、
# system prompt 又逐字节相同 (设计如此: P-SoM ≡ SoM prompt 减去那张图), 那张图是
# 二者**唯一**的区别 —— 无条件喂图使两个 mode 完全不可区分。实证: `p4_som_ptext_*`
# 与 `p2_psom_ptext_*` 的 per_task 结果逐位相同, 24 个 cell 只有 22 个不同 payload。
# 危害不是"数字偏了", 而是"测的不是声称的对象": `phantom_som` 被实现成了 `som`。
# 修复见 B-1966 / `p79.experiment.som.mode_receives_page_image`。
#
# 输出为什么不覆盖旧目录
# ======================
# 旧的 `results/mechanistic/canonical/` 是 B-1966 的**证据**, 不删。新结果落在
# `canonical_b1966fix/`, 于是修复效果可以直接检验:
#     p2_psom_ptext_* 与 p4_som_ptext_* 修复前逐位相同, 修复后**必须不同**。
# 这条检验就是重跑自带的验收断言 (见文末 VERIFY 段)。
# 附带好处: 原 queue 的「pilot_summary.md 存在则 skip」逻辑不会把重跑全部跳过。
#
# 为什么是 array 而不是一个长 job
# ================================
# researcher QoS: 最多 2 GPU、单 job 最长 72h。24 cell × 5.22h(实测中位数) ≈ 125h
# 串行, 塞进一个 job 必然撞墙; 分成 12 轮 × 2 并发也要 ~63h, 仍然贴着 72h 上限,
# 一次抖动就全盘皆输。改成 24 个独立 job (`%2` 让 Slurm 只跑 2 个并发, 正好吃满
# GPU 配额): 单 job 只需 ~5-6h, 任何一个失败只损失那一个 cell, 重投即可。
#
# 用法
# ====
#     ssh sparks
#     cd /clusterhome/jiaming/p79
#     sbatch scripts/queues/sbatch_mechanistic_b1966_rerun.sh
#     squeue -u jiaming            # 看队列
#     scancel <jobid>              # 全部取消
#     scancel <jobid>_<n>          # 取消单个 cell
set -uo pipefail

REPO=/clusterhome/jiaming/p79
cd "$REPO" || exit 1

# cell 清单来自单一真相源 —— 与 DGX 的 queue_mechanistic_canonical.sh 同一份。
# 复制一份进来就是在给自己埋 B-1966 同形状的坑 (契约写两遍, 一处悄悄漂了)。
source "$REPO/scripts/queues/_mechanistic_cells.sh"

OUT_ROOT="$REPO/results/mechanistic/canonical_b1966fix"
LOGDIR="$REPO/logs/mechanistic_b1966fix"
SCRIPT="$REPO/scripts/mechanistic/run_stage2b_continuation_pilot.py"
SUB_CLS="$REPO/results/mechanistic/archive_subset_b1_cls_canonical"
SUB_RED="$REPO/results/mechanistic/archive_subset_b1_red_canonical"
mkdir -p "$OUT_ROOT" "$LOGDIR" "$REPO/logs/slurm"

IDX="${SLURM_ARRAY_TASK_ID:?必须经 sbatch --array 提交; 不要直接 bash 这个脚本}"
if [ "$IDX" -ge "${#CELLS[@]}" ]; then
  echo "array index $IDX 超出 cell 数 ${#CELLS[@]} — 无事可做"
  exit 0
fi

entry="${CELLS[$IDX]}"
NAME="${entry%%|*}"; rest="${entry#*|}"
SITE="${rest%%|*}"; rest="${rest#*|}"
SUBKEY="${rest%%|*}"; EXTRA="${rest#*|}"
[ "$SUBKEY" = "CLS" ] && SUBSET="$SUB_CLS" || SUBSET="$SUB_RED"
OUT="$OUT_ROOT/$NAME"

# GB10 (sm_121) 与 DGX 同架构 —— 同一组 workaround 适用。
export PYTORCH_NVML_BASED_CUDA_CHECK=1
export CUDA_MPS_PIPE_DIRECTORY=""
export CUDA_MPS_LOG_DIRECTORY=""

echo "=========================================================================="
echo "[$(date '+%F %H:%M:%S')] array=$IDX/${#CELLS[@]}  cell=$NAME"
echo "  node=$(hostname)  job=${SLURM_JOB_ID:-?}  site=$SITE"
echo "  extra=$EXTRA"
echo "  out=$OUT"
echo "  subset=$SUBSET"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
echo "=========================================================================="

# 幂等: 该 cell 已完成就跳过 (重投整个 array 时只补失败的那些)
if [ -f "$OUT/pilot_summary.md" ]; then
  echo "[$(date '+%H:%M:%S')] $NAME 已完成, skip"
  exit 0
fi

# 前置断言 — 缺数据要立刻失败, 而不是让 python 跑到一半才发现。
# 这批实验最贵的失败模式是"跑了 5 小时才发现读不到东西"。
[ -d "$SUBSET" ] || { echo "FATAL: archived-run-dir 不存在: $SUBSET"; exit 2; }
n_png=$(find "$SUBSET" -path "*step_002*" -name screenshot_annotated.png 2>/dev/null | wc -l)
[ "$n_png" -ge 15 ] || { echo "FATAL: $SUBSET 的 step_002 截图只有 $n_png 个 (需 ≥15)"; exit 2; }
echo "[preflight] subset OK — step_002 截图 $n_png 个"

mkdir -p "$OUT"
START=$(date +%s)
# shellcheck disable=SC2086
.venv/bin/python3 "$SCRIPT" \
  --site "$SITE" --step 2 --max-new-tokens 50 \
  --output-dir "$OUT" --archived-run-dir "$SUBSET" \
  $EXTRA > "$LOGDIR/${NAME}.log" 2>&1
RC=$?
MIN=$(( ($(date +%s) - START) / 60 ))

if [ $RC -eq 0 ]; then
  echo "[$(date '+%H:%M:%S')] $NAME DONE in ${MIN}min"
  tail -5 "$LOGDIR/${NAME}.log"
else
  echo "[$(date '+%H:%M:%S')] $NAME FAILED rc=$RC after ${MIN}min"
  tail -30 "$LOGDIR/${NAME}.log"
fi
exit $RC

# ── VERIFY (整个 array 跑完后在 DGX 或 Sparks 上手动跑) ──────────────────────
#
# 修复的验收断言: 修复前 p2_psom_ptext_* 与 p4_som_ptext_* 的 per_task 逐位相同,
# 修复后必须不同 (前者 source=phantom_som 无图, 后者 source=som 带图, 差 578 token)。
#
#   .venv/bin/python3 - <<'PY'
#   import json, hashlib
#   from pathlib import Path
#   R = Path("results/mechanistic/canonical_b1966fix")
#   def h(name):
#       f = R / name / "patching_continuation_results.json"
#       if not f.exists(): return None
#       return hashlib.md5(json.dumps(json.load(open(f))["per_task"], sort_keys=True).encode()).hexdigest()
#   for site in ("cls", "red"):
#       a, b = h(f"p2_psom_ptext_{site}"), h(f"p4_som_ptext_{site}")
#       print(site, "psom:", a, "som:", b,
#             "→", "❌ 仍然相同 (修复未生效)" if a and a == b else "✓ 已可区分")
#   PY
