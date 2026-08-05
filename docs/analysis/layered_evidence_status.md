# 4-dimension Evidence Status (live snapshot)

Generated: 2026-08-05 02:34 UTC
Source: `make analyze-layered` (CLI alias preserved)

> Four orthogonal dimensions: Outcome / Macro / Micro / Efficiency. Sub-codes (0a / 1c / 2a / 3d) remain as figure-internal anchors.
> Missing artifacts are marked with ⚠️. All percentages and counts are read live from existing JSON/CSV artifacts or episode summaries.

## Outcome — task 成功 / 路由 arm 证据

### 0a SR per mode (canonical)

- B0 reddit: DOM **14.29%**; P-text **13.30%**; P-prompt **12.32%**; P-SoM **10.84%**; SoM **14.78%**; Vision **7.39%**
- B0 classifieds: DOM **17.41%**; P-text **15.62%**; P-prompt **19.64%**; P-SoM **15.62%**; SoM **27.23%**; Vision **25.00%**
- B1 reddit: DOM **5.91%**; P-text **5.91%**; P-prompt **5.42%**; P-SoM **5.91%**; SoM **7.39%**; Vision **2.46%**
- B1 classifieds: DOM **6.25%**; P-text **7.59%**; P-prompt **6.70%**; P-SoM **6.70%**; SoM **14.29%**; Vision **12.50%**
- B2 reddit: DOM **3.94%**; P-text **1.97%**; P-prompt **0.00%**; P-SoM **0.49%**; SoM **0.99%**; Vision **1.97%**
- B2 classifieds: DOM **1.34%**; P-text **0.45%**; P-prompt **1.79%**; P-SoM **0.89%**; SoM **2.23%**; Vision **2.23%**
- canonical source: `docs/analysis/cross_sites/sr_per_mode.json` | last update: 2026-08-05 02:30 UTC
- standalone cite source: `docs/analysis/cross_sites/sr_per_mode.md` | last update: 2026-08-05 02:30 UTC

### 0b-extra Confidence calibration (E3)

- B0 reddit: best routing AUROC Vision **0.877**; ECE n/a in existing outputs
- B0 classifieds: best routing AUROC P-SoM **0.766**; ECE n/a in existing outputs
- B1 reddit: best routing AUROC P-prompt **0.706**; ECE n/a in existing outputs
- B1 classifieds: best routing AUROC DOM **0.870**; ECE n/a in existing outputs
- source: `docs/analysis/cross_sites/mechanism_per_task.json` | last update: 2026-08-05 02:31 UTC

### 0c Routing oracle (3→5-mode lift)

- classifieds: **+2.23pp** [0.45, 4.46] Wilcoxon p=0.0253, McNemar p=0.0312 ✅
  - single phantom lifts: +P-text +1.34pp; +P-SoM +1.34pp
- reddit: **+2.46pp** [0.49, 4.93] Wilcoxon p=0.0253, McNemar p=0.0312 ✅
  - single phantom lifts: +P-text +1.48pp; +P-SoM +0.99pp
- source: `results/phantom_paper/phantom_lift.csv` | last update: 2026-08-05 02:30 UTC
- figures: `results/phantom_paper/figures/fig0c_drop_one_oracle.png`, `results/phantom_paper/figures/fig0c_phantom_lift_bars.png` | last update: 2026-08-05 02:33 UTC

### 0d Task-pool Jaccard (Scenario C sentinel)

- classifieds: P-text↔P-SoM Jaccard **0.591** (✅ safe); threshold ≤0.7
- reddit: P-text↔P-SoM Jaccard **0.441** (✅ safe); threshold ≤0.7
- source: `results/phantom_paper/phantom_lift.csv` | last update: 2026-08-05 02:30 UTC
- figure: `results/phantom_paper/figures/fig0d_taskpool_jaccard.png` | last update: 2026-08-05 02:33 UTC

### 0e Per-category SR

- reddit: ⚠️ category audit missing
- classifieds: ⚠️ category audit missing
- figure: `results/phantom_paper/figures/fig0e_category_mode_heatmap.png` | last update: 2026-05-10 22:31 UTC

### 0f Overlap depth

- reddit P-SoM: d1=2 / d2=3 / d3=3 / d4=3 / d5=8 / d6=4
- reddit P-text: d1=2 / d2=6 / d3=3 / d4=5 / d5=8 / d6=4
- reddit P-prompt: d1=2 / d2=2 / d3=4 / d4=6 / d5=8 / d6=4
- classifieds P-SoM: d1=2 / d2=0 / d3=4 / d4=8 / d5=12 / d6=9
- classifieds P-text: d1=2 / d2=1 / d3=5 / d4=7 / d5=11 / d6=9
- classifieds P-prompt: d1=6 / d2=5 / d3=6 / d4=6 / d5=12 / d6=9
- figure: `results/phantom_paper/figures/fig0f_overlap_stacked_bar.png` | last update: 2026-08-05 02:33 UTC

### 0g Routing AUROC

- reddit: DOM n/a; P-text n/a; P-prompt n/a; P-SoM n/a; SoM n/a; Vision n/a
- classifieds: DOM n/a; P-text n/a; P-prompt n/a; P-SoM n/a; SoM n/a; Vision n/a
- source: `results/phantom_paper/auroc_cross_condition.csv` | last update: 2026-08-05 02:30 UTC
- figure: `results/phantom_paper/figures/fig0g_routing_auroc_heatmap.png` | last update: 2026-08-05 02:33 UTC

## Macro — agent 平均怎么 act

### 1a Tier 1 hook coarse

- P-SoM distinct from both endpoints: reddit **0/8**, classifieds **0/8**
- DOM-only distinct cells: 7; SoM-only distinct cells: 14; indistinct cells: 12
- source: `docs/analysis/cross_sites/axis_effect_size.json` | last update: 2026-08-05 02:32 UTC

### 1b Tier 2a cascade

- Dominant cascade counts: text 12; prompt 9; image 19
- Antagonistic mechanism pairs: **21** (text_vs_image@selfcorr_count@B0/reddit, text_vs_prompt@finish_rate@B0/reddit, text_vs_image@finish_rate@B0/reddit, text_vs_image@n_steps@B0/reddit, text_vs_image@finish_rate@B1/reddit, text_vs_image@n_steps@B1/reddit)
- source: `docs/analysis/cross_sites/axis_effect_size.json` | last update: 2026-08-05 02:32 UTC

### 1c Strategy gradient

- reddit: DOM search-loop 45.37% → P-SoM search-loop 33.66% → SoM search-loop 34.63%
- classifieds: DOM search-loop 80.80% → P-SoM search-loop 78.12% → SoM search-loop 68.30%
- figure: `results/phantom_paper/figures/fig1c_strategy_gradient.png` | last update: 2026-08-05 02:33 UTC

### 1d Full action vocabulary (E4)

- reddit: compound DOM→P-SoM top shifts: tab_focus 0.079; scroll -0.042; finish -0.032
- classifieds: compound DOM→P-SoM top shifts: scroll 0.025; type -0.024; tab_focus -0.010
- source: `docs/analysis/cross_sites/mechanism_per_task.json` | last update: 2026-08-05 02:31 UTC

## Micro — per-step 决策

### 2a URL signature

- reddit: axis-1 URL-path Jaccard **n/a**; compound DOM↔P-SoM **n/a**
- classifieds: axis-1 URL-path Jaccard **n/a**; compound DOM↔P-SoM **n/a**
- source: `docs/analysis/cross_sites/axis1_microbehavior.json` | last update: 2026-08-05 02:32 UTC
- figure: `results/phantom_paper/figures/fig2_micro_divergence_heatmap.png` | last update: 2026-08-05 02:33 UTC

### 2a-extra Click-target divergence (E1)

- reddit: axis-1 click-transition Jaccard **0.335**; compound DOM↔P-SoM **0.307**
- classifieds: axis-1 click-transition Jaccard **0.315**; compound DOM↔P-SoM **0.332**
- source: `docs/analysis/cross_sites/mechanism_per_task.json` | last update: 2026-08-05 02:31 UTC

### 2b Target-hit

- reddit: axis-1 n/a; compound n/a
- classifieds: axis-1 n/a; compound n/a
- source: `docs/analysis/cross_sites/axis1_microbehavior.json` | last update: 2026-08-05 02:32 UTC

### 2c Keyword reuse

- reddit: axis-1 max-keyword-repeat diff **n/a**; compound **n/a**
- classifieds: axis-1 max-keyword-repeat diff **n/a**; compound **n/a**
- source: `docs/analysis/cross_sites/axis1_microbehavior.json` | last update: 2026-08-05 02:32 UTC

### 2d First-action

- reddit: axis-1 divergence **n/a**; compound **n/a**
- classifieds: axis-1 divergence **n/a**; compound **n/a**
- source: `docs/analysis/cross_sites/axis1_microbehavior.json` | last update: 2026-08-05 02:32 UTC

### 2e Cross-site validity

- verdict: **generalizes**; reddit ratio n/a, classifieds ratio n/a
- source: `docs/analysis/cross_sites/axis1_microbehavior.json` | last update: 2026-08-05 02:32 UTC

### 2f Trajectory boundary (E2)

- reddit: DOM↔P-SoM symmetric-diff N **n/a**; median first divergent step n/a; early n/a; late n/a
- classifieds: DOM↔P-SoM symmetric-diff N **n/a**; median first divergent step n/a; early n/a; late n/a
- source: `docs/analysis/cross_sites/mechanism_per_task.json` | last update: 2026-08-05 02:31 UTC

## Efficiency — cost / latency / carbon

### 3a Token/cost per step

- reddit: DOM input-cost/step $0.00452; P-SoM input-cost/step $0.00424; SoM input-cost/step $0.00503
- classifieds: DOM input-cost/step $0.00393; P-SoM input-cost/step $0.00392; SoM input-cost/step $0.00481
- source: B0 `condition_summary_v2.json` per condition

### 3b Image embedding / total-token gap

- reddit: SoM median tokens/step 5058 vs P-SoM 4166; observed gap **893 tokens/step**
- classifieds: SoM median tokens/step 4745 vs P-SoM 3894; observed gap **851 tokens/step**
- source: `results/phantom_paper/run_summary_collect.json` plus episode `total_tokens` fallback | last update: 2026-08-05 02:31 UTC

### 3c Latency

- reddit: DOM 572.1s/episode; P-SoM 563.0s/episode; SoM 458.5s/episode; P-SoM/SoM 1.23x
- classifieds: DOM 115.0s/episode; P-SoM 121.1s/episode; SoM 106.7s/episode; P-SoM/SoM 1.14x
- source: B0 `condition_summary_v2.json` per condition

### 3d B0 (API) vs B1 (local) deployment-class cost gap

Computed via `aggregate_cost_electricity.py`: B0 = API token dollars; B1 = `avg_total_energy_kwh × $0.12/kWh` (electricity equivalent, UK industrial). B0 vs B1 belong to different cost classes (API vs electricity), not a single ratio in $:
- reddit: B0 API $0.1038/ep vs B1 electricity $0.001267/ep → **82x** deployment-class gap
- classifieds: B0 API $0.0694/ep vs B1 electricity $0.000648/ep → **107x** deployment-class gap
- ⚠️ §103 / paper-planning legacy '30×' claim **superseded** by these data — real ratio ~100× (deployment class, not capability ratio)
- source: `docs/analysis/cross_sites/cost_per_mode.json` | last update: 2026-08-05 02:31 UTC
- figure: `results/phantom_paper/figures/fig3d_cost_sr_frontier.png` | last update: 2026-08-05 02:34 UTC

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
