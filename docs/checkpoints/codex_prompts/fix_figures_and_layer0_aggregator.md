# Codex prompt: Fix stale figures + add Layer 0a/0b aggregator + Layer 2 figure

## 用途

`paper_planning.md §3` 4-Layer Evidence Framework 已就位，但 figure 层有 5 个 gap：

1. ⚠️ **fig1ab_cascade_diamond**: 数字 stale (§103 N=48) + "Theoretical cell" 占位（应填 P-prompt）+ "Two-Knob" framing 已被 4-Layer/3-Axis 取代
2. ⚠️ **fig3d_cost_sr_frontier**: 漏 Phantom-SoM —— Layer 3d cost-SR Pareto 上 paper headline mode 缺席
3. ⚠️ **fig0d_taskpool_jaccard**: 标题"4-mode" + 模式列表 stale，没 Phantom-DOM (P-text)；docstring 还提"backup episode summaries"（早过时）
4. ⚠️ **Layer 2 micro 没 figure** —— `axis1_microbehavior.py` 只输出 .json + .md，没 visualization
5. ⚠️ **Layer 0a/0b 没 standalone aggregator** —— SR + FP per mode 数字散在 phantom_lift / fig5 等里，paper writeup 不好 cite

修复 **必须保留 paper-grade 数据准确性**，不要乱改采样/统计方法。

## 任务清单（按优先级）

### 0. 修 fig1ab_cascade_diamond.py（schematic, 全面 stale）

**当前问题**（三个）：
- a. **数字 stale**: cells 显示 `DOM search-loop 22.7%` / `Phantom-DOM 10.8%` / `Phantom-SoM 10.8%` —— 是 §103 N=48 anchor，**已被 N=210 全数据 superseded**（Layer 1c fig3 现在 51.9/49.5/35.7 reddit）
- b. **"Theoretical cell" 占位符**："AXTree with SoM prompt / not run in Phase 2.1" —— 这正好是新加的 **P-prompt** mode (`phantom_prompt`)。B0 P-prompt reddit 跑中 (~6h ETA)，cls 待启
- c. **Title + subtitle stale framing**：现在是 "Two-Knob Ablation" + "Text representation shapes exploration; prompt wording tunes commitment confidence" —— 这是 §103 N=48 旧 two-layer narrative, 已被 4-Layer Evidence + 3-Axis Cascade Diamond framework superseded（见 paper_planning §3）

**修法**：
- **数字**: 从 `docs/analysis/cross_sites/axis_effect_size.json` 读 N=210/234 live 数字。每个 cell 显示 (search_loop_pct, finish_rate_pct, n_steps_mean) 三个数字（替代旧的 search-loop + FP gap）。
- **第 4 cell (P-prompt)**: 检查 `results/visualwebarena/phase1/B0_phantom_prompt_reddit_*/phase1_phantom_prompt_router_0/episodes/` 是否存在 + 完整：
  - 如果完整 (≥ 200 episodes)：用 P-prompt 真数据填 cell
  - 如果在跑中 (< 200)：cell 标 "P-prompt (in progress: N/210)" + 灰色 placeholder
  - 如果不存在：cell 标 "P-prompt (queued)" + 灰色
  数字读取要 robust（缺失 graceful degrade）。
- **Title + framing**: 改为 "3-Axis Cascade Diamond Ablation"。subtitle 改为 "Each cell isolates one axis swap from DOM (text-payload × prompt × image-no); P-text and P-prompt are 'controlled mismatch' phantom modes". 不要再讲 "two-knob" / "prompt = commitment confidence only" —— 这两 claim 已被 N=210 数据 falsify。
- **Visual layout**: 保持 2×2 grid (text × prompt 是核心 2 axis)，加箭头标注 axis 1 (DOM↔P-text) 和 axis 2 (DOM↔P-prompt) 是 paper_planning §2 cascade 的两条独立路径。

**验证**: 跑完 fig4 显示 4 个 cell（DOM 顶左、P-text 底左、P-prompt 顶右 either real-data-or-placeholder、P-SoM 底右）+ 当前 N=210/234 live 数字。

### 1. 修 fig3d_cost_sr_frontier.py（Layer 3d）

**当前问题**：`scripts/analysis/figures/fig3d_cost_sr_frontier.py` 的 `ConditionSpec` 列表只有 4 mode（DOM / SoM / Vision / Phantom-DOM），完全没 Phantom-SoM。

**修法**：加 Phantom-SoM cls + Phantom-SoM reddit 两个 ConditionSpec，源路径：
```
RESULTS / "B0_phantom_classifieds_20260426/phase1_phantom_som_router_0"  # cls 234 ep
RESULTS / "B0_phantom_reddit_20260428/phase1_phantom_som_router_0"        # red 210 ep
```
保留 B0 / B1 baseline 部分不动。如果 B1 phantom 数据 ready 了也加进来（先 check 路径存在）。

**验证**：跑完 fig7 应该有 5 个 mode marker（颜色匹配 fig3 一致性: DOM #4c78a8, SoM #f58518, Vision #54a24b, Phantom-SoM #b279a2, Phantom-DOM #e45756）。Layer 3d cost-SR Pareto 应该显示 P-SoM 落在"DOM-cost / DOM-SoM 中间 SR"位置（4-fold drop-in property (a) cost ≈ DOM 的核心 visual evidence）。

### 2. 升级 fig0d_taskpool_jaccard.py → fig1_5mode_venn

**当前问题**：
- 文件名+标题 "4-mode"，但实际数据已是 5-mode (P-text added)
- COLORS dict 只 4 色（DOM/SoM/Vision/Phantom-SoM），漏 Phantom-DOM
- docstring："B0 Phantom-SoM uses pre-rederive backup episode summaries because the fresh runs are currently being regenerated" — 早就 stale，FRESH 数据 ready

**两个选项**（codex 选）：
- A. **重命名脚本**为 `fig1_5mode_venn.py`，输出 `fig1_5mode_venn.png`（同时**保留** `fig0d_taskpool_jaccard.py` 作 deprecated stub 重定向到新版，**或者** 直接删旧版 + grep 全 repo 替换 reference）。
- B. **保留文件名** `fig0d_taskpool_jaccard.py`（避免破 reference），但内部升级为 5-mode + 改 docstring。

**推荐 B**（最小 invasive；fig1 在 paper drafts / planning 里都用 `fig0d_taskpool_jaccard.png` 引用）。但 caption / title 里要改"4-mode"→"5-mode"避免误导。

**修法（B 路径）**：
- COLORS dict 加 `"Phantom-DOM": "#e45756"`
- 加载 P-text cls + reddit 数据：`B0_phantom_dom_classifieds_20260427` + `B0_phantom_dom_reddit_20260427`
- Venn diagram 5-mode 不可能用 2D 圆形 venn 表示（>3 set 几何不行）。**最佳做法**: 改成 **task-pool overlap matrix heatmap**（5×5 Jaccard heatmap）—— 数据上等价于 venn 但能容纳 5+ mode。
  - 或者用 UpSet plot（matplotlib 有第三方）—— 但加新 dep 不值得
  - **推荐**: 5×5 Jaccard heatmap with annotated counts（i ∩ j / i ∪ j） per cell
- docstring 重写：去掉 "backup summaries" 说法，标 "[Layer 0d] Outcome — task-pool Jaccard heatmap (5-mode)"

### 3. 新建 fig2_micro_divergence_heatmap.py（Layer 2 figure）

**目的**：把 `axis1_microbehavior.json` 的 micro 数字 visualize。Layer 2 现在只有 .md report，没 paper-friendly figure。

**新文件**：`scripts/analysis/figures/fig2_micro_divergence_heatmap.py`

**输出**：`results/phantom_paper/figures/fig2_micro_divergence_heatmap.png`

**内容**：2 site (reddit / cls) × 5 contrast (axis_1 / axis_2 / axis_3 / compound_DOM_to_PSoM / endpoint_DOM_to_SoM) heatmap 显示 URL-path Jaccard。
- 颜色: low Jaccard (= 高 divergence) 红, high Jaccard (= 低 divergence) 蓝
- annotate 每 cell: Jaccard + divergence pp
- 标题: "Layer 2 Micro Behavior — per-task URL-path overlap (lower = more decision divergence)"
- 数据 source: `docs/analysis/cross_sites/axis1_microbehavior.json` 的 `axis_contrasts.<site>.<contrast>.url_jaccard_mean`

如果觉得 Jaccard 一种太单薄，可以做 **2×N panel grid**：每 panel 一种 metric (URL Jaccard / target-hit diff / first-action divergence / keyword-repeat diff)，per metric 显示 5 contrast × 2 site 的 heatmap。但保持 figure 紧凑（不要超过 4 panel）。

加 docstring `"""[Layer 2 visualization] Micro behavior decision divergence heatmap."""`

### 4. 新建 aggregate_sr_fp_per_mode.py（Layer 0a + 0b standalone）

**目的**：现在 SR + FP per mode 散在 fig5 / phantom_lift / 各 ad-hoc 脚本里。需要一个 standalone aggregator 输出 `docs/analysis/cross_sites/sr_fp_per_mode.{json,md}`，paper 直接 cite "Layer 0a 数据见 sr_fp_per_mode.md"。

**新文件**：`scripts/analysis/aggregate_sr_fp_per_mode.py`

**输入**：所有 paper-grade B0 cls + reddit run dirs 的 `*_summary_v2.json`，路径同 axis_effect_size.py 的 STEP_DIRS（脚本要读的是 `.../episodes/<site>_task_*_summary_v2.json` 不是 `_steps_v2.jsonl`）。

**计算**：per (site, mode) 算：
```python
{
  "n_total": int,                   # task count
  "n_raw_success": int,             # success==True 数
  "n_adjusted_success": int,        # adjusted_success==True 数  
  "raw_sr_pct": float,              # 100 * n_raw_success / n_total
  "adjusted_sr_pct": float,         # 100 * n_adjusted_success / n_total
  "fp_count": int,                  # raw and not adjusted
  "fp_rate_pct": float,             # 100 * fp_count / n_total
  "fp_breakdown": {                 # 如果 summary_v2 里有 fp_reason 字段, 按 reason 分类计数
      "na_fp": int,
      "eval_fp": int,
      "visual_fp": int,
      ...
  }
}
```

**输出 JSON schema**:
```json
{
  "method": "aggregate raw/adjusted SR + FP from per-task summary_v2.json",
  "data_source": "paper-grade B0 5-mode runs (FRESH 04-29)",
  "cells": {
    "reddit/DOM":   {n_total: 210, raw_sr_pct: 11.43, adjusted_sr_pct: 9.52, fp_rate_pct: 1.90, ...},
    "reddit/SoM":   {...},
    ...
    "classifieds/DOM": {...},
    ...
  },
  "summary_table": [...]   // 长格式 per (site, mode) 一行
}
```

**Markdown 输出**: `docs/analysis/cross_sites/sr_fp_per_mode.md`，至少包含：
- 一张主表 site × mode × {n, raw_sr, adj_sr, fp_count, fp_rate}
- "FP rate ranking per site" — paper §3 Layer 0b "P-SoM red FP=0.48% lowest" finding 的 source

加 docstring `"""[Layer 0a + 0b] Outcome — aggregate SR + FP per mode."""`

### 5. 加进 Makefile

如果 codex layered refactor 已经把 `analyze-layer0` target 加进 Makefile（应该已经，看 `logs/codex_layered_refactor.run.log` 进度），追加这两个 invocations：

```makefile
analyze-layer0: ...existing... aggregate-sr-fp
analyze-layer2: ...existing... fig12-micro-heatmap

aggregate-sr-fp:
	$(PYTHON) scripts/analysis/aggregate_sr_fp_per_mode.py

fig12-micro-heatmap:
	$(PYTHON) scripts/analysis/figures/fig2_micro_divergence_heatmap.py
```

`make help` 也要更新（如果 codex refactor 已加 layered help）。

### 6. 验证

跑完所有改动后：
- `make analyze-layered` 应该跑通（所有 4 layer 不报错）
- `results/phantom_paper/figures/fig3d_cost_sr_frontier.png` 重新生成，含 5 mode marker
- `results/phantom_paper/figures/fig0d_taskpool_jaccard.png` 重新生成（5-mode heatmap）
- `results/phantom_paper/figures/fig2_micro_divergence_heatmap.png` 新建
- `docs/analysis/cross_sites/sr_fp_per_mode.{json,md}` 新建
- `docs/analysis/layered_evidence_status.md` 自动更新（如果 layered_status.py 已 wire 这两个新 source）

## token 预算

~50K（read 4 fig scripts + axis1_microbehavior.json + summary_v2 examples + write/edit 4 files）

## 不要做的事

- 不要重命名 `fig0d_taskpool_jaccard.png`（paper drafts 在 cite）—— 用 docstring + caption 区分新旧
- 不要改 fig7 现有 condition 路径，只 add Phantom-SoM
- 不要 commit
- 不要碰 codex layered refactor 已经加的 Layer 标签 docstring

## 触发命令（**等 codex layered refactor 完事后**）

```bash
~/.npm-global/bin/codex exec --skip-git-repo-check \
  -C /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents \
  -s danger-full-access \
  --output-last-message logs/codex_fix_figures.last.md \
  - < docs/checkpoints/codex_prompts/fix_figures_and_layer0_aggregator.md \
  > logs/codex_fix_figures.run.log 2>&1 &
```
