# Codex prompt: Per-task case-study mechanism narrative (Section 5 prose)

## 任务背景

`scripts/analysis/mechanism_per_task.py` 已经跑完，输出聚合 metrics 在
`docs/analysis/cross_sites/mechanism_per_task.{json,md}`。Validation pass=True，
覆盖 17 cells（B0×{cls,red}×5 modes + B1×{cls(4 含 P-SoM), red(3)}）。

聚合层（E1 click-target Jaccard / E2 trajectory boundary / E3 calibration / E4 action vocab）
告诉我们 **what shifts**。Tier 2 的下一步是给 paper Section 5 写 **why these shifts happen**
的 case-study prose——读 representative tasks 的 step JSONLs，描述发散瞬间到底发生了什么。

## 重要新数据 (2026-04-29)

| Cell | adj_SR | 结论 |
|---|---|---|
| B0 P-SoM cls | 14.53 % | DOM 17.95 %，跌 -3.42pp（4-fold drop-in 在 B0 上 hold） |
| **B1 P-SoM cls** | **7.69 %** | **B1 DOM 8.55 %，差 -0.86pp，binomial p=0.73 → 统计上 ≈ DOM** ✅ |
| B1 SoM cls | 13.68 % | 可见图带来 +5.99pp lift over P-SoM (Axis 3 image-only effect on B1) |
| B0 SoM cls | 21.79 %（参考）| Axis 3 image effect on B0 = +7.26pp |

**新观察**：
- **4-fold drop-in property (a) cost ≈ DOM 在 B0 + B1 上都 hold**（B1 P-SoM ≈ DOM cls）
- **Axis 3 image effect 量级 B0 ≈ B1**（cls：+7.26pp vs +5.99pp，scale 接近）
- → 暗示 Axis 1+2 (text payload + prompt) 的 contribution 是 **capability-tier
  invariant**，Axis 3 image 才是 capability-tier 的乘子。这一点 paper Section 5
  没写，case-study 必须确认 mechanism 一致。

## E2 case-study task IDs（已选定，对每个 axis × site）

来自 `mechanism_per_task.json` 的 `E2_trajectory_boundary`：

```
classifieds:
  DOM_vs_P-text         n_sym=21 cases=[63, 201, 98]
  DOM_vs_Phantom-SoM    n_sym=23 cases=[17, 201, 63]
  P-text_vs_Phantom-SoM n_sym=26 cases=[17, 103, 79]
  Phantom-SoM_vs_SoM    n_sym=42 cases=[14, 17, 49]

reddit:
  DOM_vs_P-text         n_sym=16 cases=[15, 81, 107]
  DOM_vs_Phantom-SoM    n_sym=23 cases=[7, 81, 15]
  P-text_vs_Phantom-SoM n_sym=15 cases=[7, 167, 26]
  Phantom-SoM_vs_SoM    n_sym=23 cases=[7, 0, 14]
```

注意 site×axis 之间有重叠 task ID（17 / 7 / 81 / 201）—— 这些是 multi-axis
sensitive tasks，case-study 中可以串起讲。

## STEP_DIRS（B0 + B1 paper-grade，最新命名）

```python
STEP_DIRS = {
    "B0": {
        "reddit": {
            "DOM":         "results/visualwebarena/phase1/B0_3mode_reddit_20260422/phase1_dom_router_0/episodes",
            "Vision":      "results/visualwebarena/phase1/B0_3mode_reddit_20260422/phase1_vision_router_0/episodes",
            "SoM":         "results/visualwebarena/phase1/B0_3mode_reddit_20260422/phase1_som_router_0/episodes",
            "Phantom-SoM": "results/visualwebarena/phase1/B0_phantom_som_reddit_20260428/phase1_phantom_som_router_0/episodes",
            "P-text":      "results/visualwebarena/phase1/B0_phantom_text_reddit_20260427/phase1_phantom_dom_router_0/episodes",
        },
        "classifieds": {
            "DOM":         "results/visualwebarena/phase1/B0_3mode_classifieds_20260413/phase1_dom_router_0/episodes",
            "Vision":      "results/visualwebarena/phase1/B0_3mode_classifieds_20260413/phase1_vision_router_0/episodes",
            "SoM":         "results/visualwebarena/phase1/B0_3mode_classifieds_20260413/phase1_som_router_0/episodes",
            "Phantom-SoM": "results/visualwebarena/phase1/B0_phantom_som_classifieds_20260426/phase1_phantom_som_router_0/episodes",
            "P-text":      "results/visualwebarena/phase1/B0_phantom_text_classifieds_20260427/phase1_phantom_dom_router_0/episodes",
        },
    },
    "B1": {
        "reddit": {
            "DOM":    "results/visualwebarena/phase1/B1_3mode_reddit_20260413/phase1_dom_router_0/episodes",
            "Vision": "results/visualwebarena/phase1/B1_3mode_reddit_20260413/phase1_vision_router_0/episodes",
            "SoM":    "results/visualwebarena/phase1/B1_3mode_reddit_20260413/phase1_som_router_0/episodes",
        },
        "classifieds": {
            "DOM":         "results/visualwebarena/phase1/B1_3mode_classifieds_20260413/phase1_dom_router_0/episodes",
            "Vision":      "results/visualwebarena/phase1/B1_3mode_classifieds_20260413/phase1_vision_router_0/episodes",
            "SoM":         "results/visualwebarena/phase1/B1_3mode_classifieds_20260413/phase1_som_router_0/episodes",
            "Phantom-SoM": "results/visualwebarena/phase1/B1_phantom_som_classifieds_20260428/phase1_phantom_som_router_0/episodes",
        },
    },
}
```

## 任务

为每个 (site, axis_contrast) 组合（共 8 个）选 1-2 个 representative task，做下面的事：

### 1. Pick + read

从 `case_study_task_ids` 里挑 1 个最有"教学意义"的 task：
- 优先 first_divergent_step ∈ [3, 10]（早不算早晚不算晚，看得到 mechanism）
- 优先 mode_a vs mode_b 行为差异**质性不同**（不同 click signature 不同 type 关键词）
- 跨 axis 出现的 task ID（17 / 7 / 81 / 201）优先，可串讲

读两侧 mode 的 `task_<id>_steps.jsonl` 全文（每 task 通常 5–30 step），归纳：
- task intent (从 `task_<id>_summary_v2.json` 的 intent 字段)
- mode_a 的轨迹 (key clicks / type / first_divergent_step 后的 plan)
- mode_b 的轨迹（同上）
- divergence root cause（一句话）

### 2. Write narrative paragraph

每个 case-study 一个段落（≤ 200 words），格式：

```
**Case A — {site} task #{id}: {axis label}**
Intent: {summarized}.
{mode_a} succeeded by {key trajectory}. {mode_b} failed at step {n}: {what went wrong}.
The divergence is {axis-specific mechanism}—e.g. "P-SoM clicked the listing link at
step 4 on {url_a} (server returns category 12), while DOM clicked the page-2 nav at
step 4 on {url_b} (server returns category index continuation). The task evaluator
checks listing_id, so the listing-link branch wins."
```

### 3. Cross-axis synthesis (1 段 ≤ 250w)

读完 8 段 case-study 后，**写 1 段 cross-axis synthesis**：哪些
mechanism 是 axis-shared，哪些 axis-specific？特别讨论：
- B0 vs B1 在同 task / 同 axis 上的轨迹相似度（用 B1 P-SoM cls 的新数据，对照 B0）
- 是否支撑 "Axis 3 image is the only capability-tier multiplier" 的假说
- 4-fold drop-in property (a)(b)(c)(d) 哪个 case 提供最强 evidence

## 输出

- `docs/analysis/cross_sites/mechanism_case_studies.md`（~1500 words）
  - 8 个 case study paragraphs
  - 1 cross-axis synthesis
  - 末尾 1 段 "paper Section 5 wiring": 8 个 case 分别 cite 在 paper 哪个 sub-section
- 不要修改 `mechanism_per_task.py`
- 不要 commit
- 如果发现某 task ID 的 step JSONL 不存在或 < 5 行（dirty / restart loss），跳过
  并记 "skipped task X (insufficient steps)"

## 不要做的事

- 不要重做 E1/E2/E3/E4 metrics（已聚合好）
- 不要写 paper Section 5 的最终 prose（这个 codex round 只产 case-study 草稿；
  Section 5 prose 由后续 codex round #13 整合）
- 不要碰 P-prompt 数据（reddit live 跑中，cls 未启动）

## token 预算

~50K (read 8 cases × 2 modes × ~20 step JSONL = ~320 file reads, write report)

## 验证

跑完检查：
- 8 个 case-study 段落齐全
- cross-axis synthesis 至少 200w
- 至少 3 个段落 cite specific URL 或 step content（不是 hand-wave）
- 至少 1 个段落涉及 B1 cell（B1 P-SoM cls 是 paper 新数据）

## 触发命令

```bash
~/.npm-global/bin/codex exec --skip-git-repo-check \
  -C /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents \
  -s danger-full-access \
  --output-last-message logs/codex_mechanism_case_studies.last.md \
  - < docs/checkpoints/codex_prompts/mechanism_per_task_case_studies.md \
  > logs/codex_mechanism_case_studies.run.log 2>&1 &
```
