# Codex prompt: Refactor analysis pipeline → 4-Layer organization

## 用途

`paper_planning.md §3` 已重组为 **4-Layer Evidence + Mechanism Framework**（line 190-340），但 `scripts/analysis/` 21 个 py 文件 + 11 个 figures + Makefile `analyze*` targets 都没按 layer 组织。**目标**：让用户跑 `make analyze-layered` 一键产出 4-layer status report，每个 script 知道自己属于哪一层，下游 paper writeup 可以直接 cite "Layer 0d 数据" / "Layer 2a 数据"。

## 4-Layer Framework（来自 paper_planning §3）

```
Layer 0  Outcome           哪些 task 成功 / 哪个 mode 在哪个 task 上 win
  0a SR per mode          summary_v2.json (live)
  0b FP rate              raw_succ - adj_succ
  0c Routing oracle       phantom_lift.{md,csv}
  0d Task-pool Jaccard    phantom_lift Scenario C sentinel
  0e Per-category SR      fig0e_category_mode_heatmap
  0f Overlap depth        fig0f_overlap_stacked_bar
  0g Routing AUROC        auroc_cross_condition_summary
Layer 1  Macro Behavior    agent 平均怎么 act (action-type 频率)
  1a Tier 1 hook coarse   axis_effect_size.py compound contrast
  1b Tier 2a cascade      axis_effect_size.py cascade
  1c Strategy gradient    fig1c_strategy_gradient
Layer 2  Micro Behavior    per-step decision (URL/keyword/element)
  2a URL signature        axis1_microbehavior.py
  2b Target-hit           axis1_microbehavior.py
  2c Keyword reuse        axis1_microbehavior.py
  2d First-action         axis1_microbehavior.py
  2e Cross-site validity  axis1_microbehavior.py
Layer 3  Efficiency        cost / latency / carbon
  3a Token cost           condition_summary_v2.json
  3b Image embedding      run_summary_collect.json
  3c Latency              condition_summary_v2.json
  3d B0 vs B1 cost gap    fig3d_cost_sr_frontier
```

## 任务

实施 **minimal-invasive refactor**（不要移动现有文件 — 大量交叉引用会断），通过新增 layered view：

### 1. 创建 `scripts/analysis/README.md`

按 layer 列出每个 script + figure，每条带：
- Script/figure 路径
- 输入 source（哪个 summary_v2 / json / live data）
- 输出路径（json/md/png）
- 当前数字示例（B0 reddit + cls 各一个数字摘要）

### 2. Makefile 加 layered targets

不要动现有 `analyze` / `analyze-paper` targets（保留兼容）。加新 targets：

```makefile
# ---- Layered analysis (paper_planning §3 framework, paper-grade B0 only) ----

# Layer 0 — Outcome (SR / oracle / AUROC / task-pool / category)
analyze-layer0:
    $(MAKE) phantom-lift                    # 0c oracle + 0d Jaccard
    $(MAKE) routing-auroc                   # 0g AUROC
    $(PYTHON) scripts/analysis/figures/fig0e_category_mode_heatmap.py   # 0e
    $(PYTHON) scripts/analysis/figures/fig0f_overlap_stacked_bar.py     # 0f
    # 0a/0b: included via summary_v2.json read by phantom-lift / fig5

# Layer 1 — Macro Behavior (action-type frequencies, cascade)
analyze-layer1:
    $(PYTHON) scripts/analysis/axis_effect_size.py                     # 1a hook + 1b cascade
    $(PYTHON) scripts/analysis/figures/fig1c_strategy_gradient.py       # 1c

# Layer 2 — Micro Behavior (per-step decision quality)
analyze-layer2:
    $(PYTHON) scripts/analysis/axis1_microbehavior.py

# Layer 3 — Efficiency (cost / latency)
analyze-layer3:
    $(MAKE) summary-collect                 # 3a/3b/3c (consume condition_summary_v2.json)
    $(PYTHON) scripts/analysis/figures/fig3d_cost_sr_frontier.py        # 3d

# Run all 4 layers
analyze-layered: analyze-layer0 analyze-layer1 analyze-layer2 analyze-layer3
    $(PYTHON) scripts/analysis/layered_status.py

.PHONY: analyze-layer0 analyze-layer1 analyze-layer2 analyze-layer3 analyze-layered
```

也加进 `.PHONY` 那行（line 24-28）+ `make help` 输出。

### 3. 写 `scripts/analysis/layered_status.py`

读取所有 4 layer 已有 artifact，产出 `docs/analysis/layered_evidence_status.md`：
- 每 layer 一个 section
- 每个 sub-evidence (0a-0g, 1a-1c, 2a-2e, 3a-3d) 一行：当前数字 + source artifact 路径 + last-modified timestamp
- 失败 / 缺失的 sub-evidence 标 ⚠️
- 总结 paper claim → layer support 矩阵（参考 paper_planning §3 "Evidence chain" 表）

格式（约 300 行 markdown）：
```markdown
# 4-Layer Evidence Status (live snapshot)

Generated: 2026-04-29 14:00 UTC  
Source: `make analyze-layered`

## Layer 0 — Outcome

### 0a SR per mode (B0)
- reddit P-SoM adj **13.81%** | source: `results/visualwebarena/phase1/B0_phantom_reddit_*/episodes/*_summary_v2.json` (live)
- cls P-SoM adj **14.53%**
- last update: 2026-04-29 09:34
[...]

### 0c Routing oracle (3→5-mode lift)
- reddit: **+5.24pp** [2.38, 8.11] Wilcoxon p=0.0009 ✅
- cls: +4.70pp [2.14, 7.69] p=0.0009 ✅
- source: `results/phantom_paper/phantom_lift.{md,csv}`
- last update: 2026-04-28 18:00
[...]

## Layer 1 — Macro Behavior
[...]

## Paper Claim → Layer Support Matrix

| Claim | Layers cited | Verdict |
|---|---|---|
| C1 P-SoM independent routing arm | 0a, 0c, 0d, 0g, 1a, 2a | ✅ supported |
| C2 4-fold drop-in property | 3a, 3c, 0g, 0c | ✅ all 4 conditions met |
[...]
```

数字读取必须 **live** —— 直接读 json/md/csv，不要硬编码。如果某 artifact 不存在就标 ⚠️ 但不要 raise。

### 4. 给每个 layer-relevant script 头部 docstring 加 layer 标签

例如 `scripts/analysis/axis_effect_size.py` 头部加：
```python
"""[Layer 1a + 1b] Macro Behavior — axis-by-axis cascade ablation.

Outputs:
- docs/analysis/cross_sites/axis_effect_size.json  (machine-readable)
- docs/analysis/cross_sites/axis_effect_size_report.md  (paper-ready)

Tier 1 hook (1a): DOM↔P-SoM compound + DOM↔SoM endpoint (sanity)
Tier 2a cascade (1b): DOM→P-text→P-SoM→SoM 3-axis decomposition

See paper_planning.md §3 Layer 1 framework.
"""
```

加 layer 标签的 scripts (按现有 layer 映射):
- `axis_effect_size.py` → Layer 1a + 1b
- `axis1_microbehavior.py` → Layer 2a-2e
- `aggregate_phantom_lift.py` → Layer 0c + 0d
- `aggregate_routing_auroc.py` → Layer 0g
- `aggregate_cross_site.py` → Layer 3a-3c
- `collect_analysis_summary.py` → Layer 3 supporting (run-level metadata)
- `figures/fig1c_strategy_gradient.py` → Layer 1c
- `figures/fig0e_category_mode_heatmap.py` → Layer 0e
- `figures/fig3d_cost_sr_frontier.py` → Layer 3d
- `figures/fig0f_overlap_stacked_bar.py` → Layer 0f
- `figures/fig0c_phantom_lift_bars.py` → Layer 0c viz
- `figures/fig0g_routing_auroc_heatmap.py` → Layer 0g viz

不打 layer 标签 (诊断/单次/per-run): `analyze_experiment.py`, `analyze_*selflink*`, `analyze_search_over_browse.py`, `analyze_confidence_calibration.py`, `analyze_reason_diagnostics.py`, `analyze_cross_representation.py`, `analyze_noninteractive_*`, `compare_b0_b1.py`, `b0_vision_coordinate_errors.py`, `diag_pattern_match.py`, `validate_run.py` — 给它们的 docstring 头加一行说明"per-run diagnostic / not part of layered framework" 即可。

### 5. 验证

跑完 `make analyze-layered`：
- 不能 break 现有 `make analyze` / `make analyze-paper` (跑一遍确认)
- `docs/analysis/layered_evidence_status.md` 生成 + 可读
- 每个 layer 的所有 sub-evidence 有数字 (或明确标 ⚠️ missing)
- README.md 列出所有 21 scripts + 11 figures, layer 分组清晰

### 6. 不要做的事

- **不要移动** scripts/figures 到 layer 子目录 — 大量 absolute path reference 会断 (paper_planning, watchdog, queue scripts, README 等)
- **不要重命名** 输出 json/md/png 文件名 — fig5 / fig8 等已在 paper drafts 直接 cite
- **不要动** per-run analysis pipeline (`analyze`, `analyze-paper`)
- **不要 commit** — 让用户自己 commit

## 输出

- `scripts/analysis/README.md` (new)
- `Makefile` (edit, add layered targets)
- `scripts/analysis/layered_status.py` (new, ~300 lines)
- `docs/analysis/layered_evidence_status.md` (new, generated by layered_status.py)
- 12 scripts/figures 头部 docstring 加 Layer 标签 (in-place edit)

## token 预算

~80K (read 21 scripts + 11 figs headers + Makefile + paper_planning §3 + write 4 new files + edit 12 docstrings)

## 触发命令

```bash
~/.npm-global/bin/codex exec --skip-git-repo-check \
  -C /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents \
  -s danger-full-access \
  --output-last-message logs/codex_layered_refactor.last.md \
  - < docs/checkpoints/codex_prompts/refactor_layered_analysis.md \
  > logs/codex_layered_refactor.run.log 2>&1 &
```
