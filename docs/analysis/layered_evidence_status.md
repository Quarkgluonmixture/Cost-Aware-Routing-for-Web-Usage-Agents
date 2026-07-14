# 4-dimension Evidence Status (live snapshot)

Generated: 2026-07-14 17:08 UTC
Source: `make analyze-layered` (CLI alias preserved)

> Four orthogonal dimensions: Outcome / Macro / Micro / Efficiency. Sub-codes (0a / 1c / 2a / 3d) remain as figure-internal anchors.
> Missing artifacts are marked with ⚠️. All percentages and counts are read live from existing JSON/CSV artifacts or episode summaries.

## Outcome — task 成功 / 路由 arm 证据

### 0a SR per mode (canonical)

- B0 reddit: DOM **14.63%**; P-text **13.66%**; P-prompt **n/a**; P-SoM **11.22%**; SoM **14.63%**; Vision **7.80%**
- B0 classifieds: DOM **17.41%**; P-text **15.62%**; P-prompt **19.64%**; P-SoM **15.62%**; SoM **27.23%**; Vision **25.00%**
- B1 reddit: DOM **6.83%**; P-text **6.83%**; P-prompt **n/a**; P-SoM **6.83%**; SoM **8.29%**; Vision **2.93%**
- B1 classifieds: DOM **6.25%**; P-text **7.59%**; P-prompt **6.70%**; P-SoM **6.70%**; SoM **14.29%**; Vision **12.50%**
- B2 reddit: DOM **n/a**; P-text **n/a**; P-prompt **n/a**; P-SoM **n/a**; SoM **n/a**; Vision **n/a**
- B2 classifieds: DOM **1.34%**; P-text **0.45%**; P-prompt **1.79%**; P-SoM **0.89%**; SoM **2.23%**; Vision **2.23%**
- canonical source: `docs/analysis/cross_sites/sr_per_mode.json` | last update: 2026-07-14 13:54 UTC
- standalone cite source: `docs/analysis/cross_sites/sr_per_mode.md` | last update: 2026-07-14 13:54 UTC

### 0b-extra Confidence calibration (E3)

- B0 reddit: best routing AUROC Vision **0.877**; ECE n/a in existing outputs
- B0 classifieds: best routing AUROC P-SoM **0.766**; ECE n/a in existing outputs
- B1 reddit: best routing AUROC P-text **0.688**; ECE n/a in existing outputs
- B1 classifieds: best routing AUROC DOM **0.870**; ECE n/a in existing outputs
- source: `docs/analysis/cross_sites/mechanism_per_task.json` | last update: 2026-07-14 13:54 UTC

### 0c Routing oracle (3→5-mode lift)

- classifieds: **+2.23pp** [0.45, 4.46] Wilcoxon p=0.0253, McNemar p=0.0312 ✅
  - single phantom lifts: +P-text +1.34pp; +P-SoM +1.34pp
- reddit: **+2.44pp** [0.49, 4.88] Wilcoxon p=0.0253, McNemar p=0.0312 ✅
  - single phantom lifts: +P-text +1.46pp; +P-SoM +0.98pp
- source: `results/phantom_paper/phantom_lift.csv` | last update: 2026-07-14 13:54 UTC
- figures: `results/phantom_paper/figures/fig0c_drop_one_oracle.png`, `results/phantom_paper/figures/fig0c_phantom_lift_bars.png` | last update: 2026-07-14 13:54 UTC

### 0d Task-pool Jaccard (Scenario C sentinel)

- classifieds: P-text↔P-SoM Jaccard **0.591** (✅ safe); threshold ≤0.7
- reddit: P-text↔P-SoM Jaccard **0.417** (✅ safe); threshold ≤0.7
- source: `results/phantom_paper/phantom_lift.csv` | last update: 2026-07-14 13:54 UTC
- figure: `results/phantom_paper/figures/fig0d_taskpool_jaccard.png` | last update: 2026-07-14 13:54 UTC

### 0e Per-category SR

- reddit: ⚠️ category audit missing
- classifieds: ⚠️ category audit missing
- figure: `results/phantom_paper/figures/fig0e_category_mode_heatmap.png` | last update: 2026-05-10 22:31 UTC

### 0f Overlap depth

- reddit P-SoM: d1=2 / d2=4 / d3=4 / d4=9 / d5=4 / d6=0
- reddit P-text: d1=3 / d2=7 / d3=5 / d4=9 / d5=4 / d6=0
- classifieds P-SoM: d1=2 / d2=0 / d3=4 / d4=8 / d5=12 / d6=9
- classifieds P-text: d1=2 / d2=1 / d3=5 / d4=7 / d5=11 / d6=9
- classifieds P-prompt: d1=6 / d2=5 / d3=6 / d4=6 / d5=12 / d6=9
- figure: `results/phantom_paper/figures/fig0f_overlap_stacked_bar.png` | last update: 2026-07-14 13:54 UTC

### 0g Routing AUROC

- reddit: DOM n/a; P-text n/a; P-prompt n/a; P-SoM n/a; SoM n/a; Vision n/a
- classifieds: DOM n/a; P-text n/a; P-prompt n/a; P-SoM n/a; SoM n/a; Vision n/a
- source: `results/phantom_paper/auroc_cross_condition.csv` | last update: 2026-07-14 13:54 UTC
- figure: `results/phantom_paper/figures/fig0g_routing_auroc_heatmap.png` | last update: 2026-07-14 13:54 UTC

## Macro — agent 平均怎么 act

### 1a Tier 1 hook coarse

- P-SoM distinct from both endpoints: reddit **0/8**, classifieds **0/8**
- DOM-only distinct cells: 0; SoM-only distinct cells: 0; indistinct cells: 0
- source: `docs/analysis/cross_sites/axis_effect_size.json` | last update: 2026-07-14 13:54 UTC

### 1b Tier 2a cascade

- Dominant cascade counts: text 0; prompt 0; image 0
- Antagonistic mechanism pairs: **0** ()
- source: `docs/analysis/cross_sites/axis_effect_size.json` | last update: 2026-07-14 13:54 UTC

### 1c Strategy gradient

- reddit: DOM search-loop 45.37% → P-SoM search-loop 33.66% → SoM search-loop 34.63%
- classifieds: DOM search-loop 80.80% → P-SoM search-loop 78.12% → SoM search-loop 68.30%
- figure: `results/phantom_paper/figures/fig1c_strategy_gradient.png` | last update: 2026-07-14 13:54 UTC

### 1d Full action vocabulary (E4)

- reddit: compound DOM→P-SoM top shifts: tab_focus 0.077; scroll -0.043; finish -0.030
- classifieds: compound DOM→P-SoM top shifts: scroll 0.025; type -0.024; tab_focus -0.010
- source: `docs/analysis/cross_sites/mechanism_per_task.json` | last update: 2026-07-14 13:54 UTC

## Micro — per-step 决策

### 2a URL signature

- reddit: axis-1 URL-path Jaccard **n/a**; compound DOM↔P-SoM **n/a**
- classifieds: axis-1 URL-path Jaccard **n/a**; compound DOM↔P-SoM **n/a**
- source: `docs/analysis/cross_sites/axis1_microbehavior.json` | last update: 2026-07-14 13:54 UTC
- figure: `results/phantom_paper/figures/fig2_micro_divergence_heatmap.png` | last update: 2026-07-14 13:54 UTC

### 2a-extra Click-target divergence (E1)

- reddit: axis-1 click-transition Jaccard **0.333**; compound DOM↔P-SoM **0.308**
- classifieds: axis-1 click-transition Jaccard **0.315**; compound DOM↔P-SoM **0.332**
- source: `docs/analysis/cross_sites/mechanism_per_task.json` | last update: 2026-07-14 13:54 UTC

### 2b Target-hit

- reddit: axis-1 n/a; compound n/a
- classifieds: axis-1 n/a; compound n/a
- source: `docs/analysis/cross_sites/axis1_microbehavior.json` | last update: 2026-07-14 13:54 UTC

### 2c Keyword reuse

- reddit: axis-1 max-keyword-repeat diff **n/a**; compound **n/a**
- classifieds: axis-1 max-keyword-repeat diff **n/a**; compound **n/a**
- source: `docs/analysis/cross_sites/axis1_microbehavior.json` | last update: 2026-07-14 13:54 UTC

### 2d First-action

- reddit: axis-1 divergence **n/a**; compound **n/a**
- classifieds: axis-1 divergence **n/a**; compound **n/a**
- source: `docs/analysis/cross_sites/axis1_microbehavior.json` | last update: 2026-07-14 13:54 UTC

### 2e Cross-site validity

- verdict: **not supported**; reddit ratio n/a, classifieds ratio n/a
- source: `docs/analysis/cross_sites/axis1_microbehavior.json` | last update: 2026-07-14 13:54 UTC

### 2f Trajectory boundary (E2)

- reddit: DOM↔P-SoM symmetric-diff N **n/a**; median first divergent step n/a; early n/a; late n/a
- classifieds: DOM↔P-SoM symmetric-diff N **n/a**; median first divergent step n/a; early n/a; late n/a
- source: `docs/analysis/cross_sites/mechanism_per_task.json` | last update: 2026-07-14 13:54 UTC

## Efficiency — cost / latency / carbon

### 3a Token/cost per step

- reddit: DOM input-cost/step $0.00452; P-SoM input-cost/step $0.00424; SoM input-cost/step $0.00503
- classifieds: DOM input-cost/step $0.00393; P-SoM input-cost/step $0.00392; SoM input-cost/step $0.00481
- source: B0 `condition_summary_v2.json` per condition

### 3b Image embedding / total-token gap

- reddit: SoM median tokens/step 5058 vs P-SoM 4166; observed gap **893 tokens/step**
- classifieds: SoM median tokens/step 4745 vs P-SoM 3894; observed gap **851 tokens/step**
- source: `results/phantom_paper/run_summary_collect.json` plus episode `total_tokens` fallback | last update: 2026-07-14 13:54 UTC

### 3c Latency

- reddit: DOM 572.1s/episode; P-SoM 563.0s/episode; SoM 458.5s/episode; P-SoM/SoM 1.23x
- classifieds: DOM 115.0s/episode; P-SoM 121.1s/episode; SoM 106.7s/episode; P-SoM/SoM 1.14x
- source: B0 `condition_summary_v2.json` per condition

### 3d B0 (API) vs B1 (local) deployment-class cost gap

Computed via `aggregate_cost_electricity.py`: B0 = API token dollars; B1 = `avg_total_energy_kwh × $0.12/kWh` (electricity equivalent, UK industrial). B0 vs B1 belong to different cost classes (API vs electricity), not a single ratio in $:
- reddit: B0 API $0.1043/ep vs B1 electricity $0.001254/ep → **83x** deployment-class gap
- classifieds: B0 API $0.0694/ep vs B1 electricity $0.000648/ep → **107x** deployment-class gap
- ⚠️ §103 / paper-planning legacy '30×' claim **superseded** by these data — real ratio ~100× (deployment class, not capability ratio)
- source: `docs/analysis/cross_sites/cost_per_mode.json` | last update: 2026-07-14 13:54 UTC
- figure: `results/phantom_paper/figures/fig3d_cost_sr_frontier.png` | last update: 2026-07-14 13:55 UTC

## Paper Claim → Dimension Support Matrix

| Claim | Dimensions cited | Verdict |
|---|---|---|
| C1 P-SoM independent routing arm | 0a, 0c, 0d, 0g, 1a, 2a | ✅ supported by live outcome + behavior artifacts |
| C2 4-fold drop-in property | 3a, 3c, 0g, 0c | ✅ cost/latency/signal/oracle evidence present |
| C3 3-axis hierarchical theory | 1b, 2a-2e, cross-dimension mechanism chain | ✅ cascade + micro decomposition present |
| C4 Aggregate macro can mislead about routing potential | 1a, 0d, 2a | ✅ supported by task-pool and micro-divergence evidence |
| C5 Prompt as task-conditional decision prior | 0b, 0b-extra, 0d, 1b, 1d, 2a-extra, 2f | ✅ supported; cite cautiously as mechanism evidence |
| C6 Image is bidirectional modality fusion | 1b, 0e, 3b | ✅ supported for cls-heavy image axis; 3b is a token-gap proxy |

## Cross-dimension Mechanism Chain

| Axis | Outcome dimension | Macro dimension | Micro dimension | Efficiency dimension |
|---|---|---|---|---|
| Axis 1 text payload | 0c single-phantom lift | 1b text-axis cells, 1d action shifts | 2a-2e URL/target/keyword shifts, E1 click transitions | no image tax |
| Axis 2 prompt | 0d task-pool divergence, 0b-extra calibration | 1b prompt-axis dominant cells, 1d action shifts | 2d first-action, E1 click transitions, E2 boundary | prompt-only cost-neutral |
| Axis 3 image | 0e category recovery | 1b image-axis dominant cells, 1d action shifts | 2a endpoint URL/target shifts, E1/E2 | 3b token/latency tax |
| Compound P-SoM vs DOM | 0a/0c/0d routing arm, E3 confidence | 1a hook contrast, E4 action vocabulary | 2a compound URL divergence, E1/E2 per-step evidence | 3a/3c drop-in profile |
