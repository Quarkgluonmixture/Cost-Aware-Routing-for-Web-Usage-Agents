# Codex prompt: Rename figures to layer-prefixed naming

## 用途

`paper_planning §3` 4-Layer Evidence Framework 已就位，但 figure 文件名 (`fig1_*` ... `fig12_*`) 是按时间顺序的 legacy 数字 prefix，没体现 layer 归属。Paper 写作 + reviewer 引用时 cross-reference 不直观。例如：reviewer 看到 "fig1ab_cascade_diamond" 不知道是 Layer 几，但看到 "fig1ab_cascade_diamond" 立刻明白是 Layer 1a + 1b。

**修改性质**：纯 rename + reference update，**不改任何 figure content / 数据 / 计算逻辑**。

## Naming map（必须完全按此映射）

| 当前文件名 | 新文件名 | Layer |
|---|---|---|
| `fig0d_taskpool_jaccard.png` | `fig0d_taskpool_jaccard.png` | 0d task-pool Jaccard heatmap |
| `fig0d_taskpool_jaccard.py` | `fig0d_taskpool_jaccard.py` | (script) |
| `fig0c_drop_one_oracle.png` | `fig0c_drop_one_oracle.png` | 0c routing oracle drop-one |
| `fig0c_drop_one_oracle.py` | `fig0c_drop_one_oracle.py` | (script) |
| `fig0c_drop_one_bootstrap_ci.csv` | `fig0c_drop_one_bootstrap_ci.csv` | (csv sidecar) |
| `fig1c_strategy_gradient.png` | `fig1c_strategy_gradient.png` | 1c macro strategy gradient |
| `fig1c_strategy_gradient.py` | `fig1c_strategy_gradient.py` | (script) |
| `fig1ab_cascade_diamond.png` | `fig1ab_cascade_diamond.png` | 1a hook + 1b cascade visualization |
| `fig1ab_cascade_diamond.py` | `fig1ab_cascade_diamond.py` | (script) |
| `fig0e_category_mode_heatmap.png` | `fig0e_category_mode_heatmap.png` | 0e per-category SR |
| `fig0e_category_mode_heatmap.py` | `fig0e_category_mode_heatmap.py` | (script) |
| `fig_capability_b0_b1.png` | `fig_capability_b0_b1.png` | cross-layer B0/B1 contrast |
| `fig_capability_b0_b1.py` | `fig_capability_b0_b1.py` | (script) |
| `fig3d_cost_sr_frontier.png` | `fig3d_cost_sr_frontier.png` | 3d cost-SR Pareto |
| `fig3d_cost_sr_frontier.py` | `fig3d_cost_sr_frontier.py` | (script) |
| `fig0f_overlap_stacked_bar.png` | `fig0f_overlap_stacked_bar.png` | 0f overlap depth |
| `fig0f_overlap_stacked_bar.py` | `fig0f_overlap_stacked_bar.py` | (script) |
| `fig3_regional_carbon.png` | `fig3_regional_carbon.png` | 3 regional carbon (B1) |
| `fig3_regional_carbon.py` | `fig3_regional_carbon.py` | (script — note current py name has extra `_carbon_` 前缀) |
| `fig0c_phantom_lift_bars.png` | `fig0c_phantom_lift_bars.png` | 0c routing lift bars (companion to fig0c_drop_one) |
| `fig0c_phantom_lift_bars.py` | `fig0c_phantom_lift_bars.py` | (script) |
| `fig0g_routing_auroc_heatmap.png` | `fig0g_routing_auroc_heatmap.png` | 0g routing AUROC |
| `fig0g_routing_auroc_heatmap.py` | `fig0g_routing_auroc_heatmap.py` | (script) |
| `fig2_micro_divergence_heatmap.png` | `fig2_micro_divergence_heatmap.png` | 2 micro decision divergence |
| `fig2_micro_divergence_heatmap.py` | `fig2_micro_divergence_heatmap.py` | (script) |

## 必须做的步骤

### 1. Rename script files (.py)

`git mv scripts/analysis/figures/fig{old}.py scripts/analysis/figures/fig{new}.py` 共 12 个 .py。如果不在 git 里就用 `mv`。

### 2. 改每个 script 内部的 OUT path

每个 script 头部有 `OUT = ROOT / "results/phantom_paper/figures/fig{old}.png"` —— 必须改为新文件名。例如：
```python
# fig0d_taskpool_jaccard.py
OUT = ROOT / "results/phantom_paper/figures/fig0d_taskpool_jaccard.png"
```

### 3. 删旧 PNG (or rename) + regenerate 新 PNG

最 robust 做法：
```bash
# 删旧 PNG
rm results/phantom_paper/figures/fig{old}.png  # all 12
# Rerun every script to produce fig{new}.png
make analyze-layered  # regenerates all
```

### 4. Update Makefile references

`Makefile` 里的 `analyze-layer{0,1,2,3}` 调用要更新（grep "fig{N}_" 找到）：
```makefile
# 例如 analyze-layer0:
$(PYTHON) scripts/analysis/figures/fig0e_category_mode_heatmap.py   # was fig5_*
$(PYTHON) scripts/analysis/figures/fig0f_overlap_stacked_bar.py     # was fig8_*
```

### 5. Update cross-reference 文档

Grep + replace 共 ~100 个 ref。涉及文件：
- `docs/checkpoints/paper_planning.md` (~30 refs)
- `docs/checkpoints/实验笔记.md` (~20 refs)
- `docs/checkpoints/next_steps.md` (~10 refs)
- `docs/analysis/cross_sites/*.md` (~10 refs, 包含 codex 之前写的 narrative)
- `docs/analysis/paper_drafts/section*.md` (~10 refs, 旧 prose 引用 fig name)
- `docs/analysis/layered_evidence_status.md` (~12 refs)
- `scripts/analysis/README.md` (~12 refs, codex 1 写的 layered inventory)
- `scripts/analysis/layered_status.py` (FIGURES dict 含 fig 路径)
- 其它 `.md` 散见 fig name

**所有 `figN_*.{png,py}` literal 都要替换为新名**。注意：
- 不要替换 PNG 之外的同名字符串（基本不会冲突，因为 fig name pattern 独特）
- LaTeX-style `\includegraphics{fig7_*}` 之类的也要改（如果 paper drafts 用 LaTeX）

### 6. Layered status report 更新

`scripts/analysis/layered_status.py` 的 `FIGURES` dict 路径需更新：
```python
FIGURES = {
    "fig0d": ROOT / "results/phantom_paper/figures/fig0d_taskpool_jaccard.png",
    "fig0c_drop_one": ROOT / "results/phantom_paper/figures/fig0c_drop_one_oracle.png",
    ...
}
```

跑 `python scripts/analysis/layered_status.py` 验证 status report 引用都对。

### 7. 验证

跑完所有改动：
- [ ] `ls results/phantom_paper/figures/` 显示全部新名 + 0 个旧名 (`fig{1-12}_*.png`)
- [ ] `python -m py_compile scripts/analysis/figures/fig*.py` 全过
- [ ] `make analyze-layered` 全跑通
- [ ] `grep -r "fig[0-9][_a-z]*\.png" docs/ scripts/ Makefile` 只显示新名（无 `fig0d_taskpool_jaccard` / `fig1ab_cascade_diamond` 等 stale name）
- [ ] `docs/analysis/layered_evidence_status.md` regenerated 后所有 figure ref 都指向新名

### 7.5. 顺便修 fig3d_cost_sr_frontier (即 fig7) 的 cost source

**问题**：当前 fig3d (fig7) 直接读 `condition_summary_v2.json` 的 `avg_total_cost_usd`，这个字段对 B1 是 artifact (用 B0 rate 算的，~$0.05/ep 假象)。fig 上现在显示 "Phantom-SoM/SoM cost 0.9-1.1×" 是 token-cost ratio in B0 only —— 但 paper Section 3d 真正想讲的是 B0 vs B1 deployment-class gap **~100×**（见 `docs/analysis/cross_sites/cost_per_mode.md`）。

**修法**：
- B0 cell 继续用 `avg_total_cost_usd` (real API token $)
- **B1 cell 改用 `cost_per_mode.json` 里的 `paper_cost_usd`** (即 `avg_total_energy_kwh × $0.12/kWh` electricity-equivalent)
- 加一个 prominent annotation 在 figure 顶部："B0 reports API token \\$; B1 reports electricity-equivalent \\$ (different cost classes)"
- 加 **deployment-class ratio** annotation: "B0/B1 ~100× deployment-class gap (reddit 98×, cls 105×)"
- x-axis 用 **log scale** 让 B0 (~$0.04) 和 B1 (~$0.0004) 都在同图清晰
- 保留现有 5-mode marker 颜色 + Phantom-SoM；不要删 SoM/Phantom-DOM cost ratio annotation（那是 Layer 3a intra-baseline ratio，paper 也用）

**Source code change**:
读 `docs/analysis/cross_sites/cost_per_mode.json` 来 lookup `cells[baseline][site][mode].paper_cost_usd`，rather than直接 read condition_summary_v2.json. 这样统一 source of truth，与 layered_evidence_status 的 Layer 3d 口径一致。

### 8. 不要做的事

- 不要改 figure 数据计算 / 颜色 / layout（除了 fig3d cost source 上面 7.5）
- 不要改 cost rate（cost_per_mode.json 的 $0.12/kWh 是 ground truth, 不要变）
- 不要 commit
- 不要碰 codex 1/2 之前加的 `[Layer N{a-g}]` docstring (保留)
- 不要重命名 condition_summary_v2.json / 其他 `.json` artifact

### 9. token 预算

~50K (read 12 scripts + ~10 doc files + Makefile + execute renames + regenerate figures + verify)

## 触发命令

```bash
~/.npm-global/bin/codex exec --skip-git-repo-check \
  -C /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents \
  -s danger-full-access \
  --output-last-message logs/codex_rename_figures.last.md \
  - < docs/checkpoints/codex_prompts/rename_figures_layered.md \
  > logs/codex_rename_figures.run.log 2>&1 &
```
