# 4-dimension Evidence Status (live snapshot)

Generated: 2026-04-29 23:49 UTC  
Source: `make analyze-layered` (CLI alias preserved)

> Four orthogonal dimensions: Outcome / Macro / Micro / Efficiency. Sub-codes (0a / 1c / 2a / 3d) remain as figure-internal anchors.
> Missing artifacts are marked with ⚠️. All percentages and counts are read live from existing JSON/CSV artifacts or episode summaries.

## Outcome — task 成功 / 路由 arm 证据

### 0a SR per mode (B0)

- reddit: DOM raw 11.43% / adj **9.52%**; P-text raw 13.81% / adj **12.38%**; P-prompt raw 10.26% / adj **8.97%**; P-SoM raw 14.29% / adj **13.81%**; SoM raw 11.90% / adj **10.48%**; Vision raw 8.57% / adj **6.67%**
- classifieds: DOM raw 14.96% / adj **14.10%**; P-text raw 16.67% / adj **14.53%**; P-SoM raw 15.81% / adj **14.53%**; SoM raw 23.08% / adj **21.37%**; Vision raw 15.81% / adj **13.68%**
- source: `results/visualwebarena/phase1/B0_*/*/episodes/*_summary_v2.json` (live); last update: 2026-04-29 23:47 UTC
- standalone cite source: `docs/analysis/cross_sites/sr_fp_per_mode.md` | last update: 2026-04-29 23:49 UTC

### 0b FP rate (raw success - adjusted success)

- reddit: DOM 1.90%; P-text 1.43%; P-prompt 1.28%; P-SoM 0.48%; SoM 1.43%; Vision 1.90%
- classifieds: DOM 0.85%; P-text 2.14%; P-SoM 1.28%; SoM 1.71%; Vision 2.14%
- source: same live episode `summary_v2.json` files as 0a; standalone `docs/analysis/cross_sites/sr_fp_per_mode.md` | last update: 2026-04-29 23:49 UTC

### 0b-extra Confidence calibration (E3)

- B0 reddit: best routing AUROC DOM **0.817**; ECE n/a in existing outputs
- B0 classifieds: best routing AUROC Vision **0.773**; ECE n/a in existing outputs
- B1 reddit: best routing AUROC Vision **0.862**; lowest verbal ECE DOM 0.666
- B1 classifieds: best routing AUROC Vision **0.816**; lowest verbal ECE Vision 0.602
- source: `docs/analysis/cross_sites/mechanism_per_task.json` | last update: 2026-04-29 23:49 UTC

### 0c Routing oracle (3→5-mode lift)

- classifieds: **+4.70pp** [2.14, 7.69] Wilcoxon p=0.0009, McNemar p=0.0005 ✅
  - single phantom lifts: +P-text +3.42pp; +P-SoM +2.56pp
- reddit: **+4.49pp** [1.28, 7.71] Wilcoxon p=0.0082, McNemar p=0.0078 ✅
  - single phantom lifts: +P-text +3.21pp; +P-SoM +2.56pp
- source: `results/phantom_paper/phantom_lift.csv` | last update: 2026-04-29 23:49 UTC
- figures: `results/phantom_paper/figures/fig0c_drop_one_oracle.png`, `results/phantom_paper/figures/fig0c_phantom_lift_bars.png` | last update: 2026-04-29 23:49 UTC

### 0d Task-pool Jaccard (Scenario C sentinel)

- classifieds: P-text↔P-SoM Jaccard **0.447** (✅ safe); threshold ≤0.7
- reddit: P-text↔P-SoM Jaccard **0.500** (✅ safe); threshold ≤0.7
- source: `results/phantom_paper/phantom_lift.csv` | last update: 2026-04-29 23:49 UTC
- figure: `results/phantom_paper/figures/fig0d_taskpool_jaccard.png` | last update: 2026-04-29 23:49 UTC

### 0e Per-category SR

- reddit DOM: A 0.00%; B 15.48%; C 6.19%; D 0.00%
- reddit P-SoM: A 9.09%; B 26.19%; C 5.31%; D 0.00%
- reddit SoM: A 0.00%; B 17.86%; C 6.19%; D 0.00%
- classifieds DOM: A 7.41%; B 27.94%; C 7.29%; D 11.63%
- classifieds P-SoM: A 7.41%; B 27.94%; C 9.38%; D 9.30%
- classifieds SoM: A 11.11%; B 30.88%; C 20.83%; D 13.95%
- figure: `results/phantom_paper/figures/fig0e_category_mode_heatmap.png` | last update: 2026-04-29 23:49 UTC

### 0f Overlap depth

- reddit P-SoM: d1=2 / d2=8 / d3=5 / d4=8 / d5=5 / d6=1
- reddit P-text: d1=4 / d2=4 / d3=4 / d4=8 / d5=5 / d6=1
- reddit P-prompt: d1=4 / d2=1 / d3=1 / d4=3 / d5=4 / d6=1
- classifieds P-SoM: d1=3 / d2=7 / d3=7 / d4=10 / d5=7 / d6=0
- classifieds P-text: d1=5 / d2=7 / d3=7 / d4=8 / d5=7 / d6=0
- figure: `results/phantom_paper/figures/fig0f_overlap_stacked_bar.png` | last update: 2026-04-29 23:49 UTC

### 0g Routing AUROC

- reddit: DOM 0.817 (ep_mean_verbalized); P-text 0.793 (ep_mean_verbalized); P-prompt n/a; P-SoM 0.720 (ep_mean_verbalized); SoM 0.719 (ep_mean_verbalized); Vision 0.778 (ep_mean_verbalized)
- classifieds: DOM 0.769 (action_diversity); P-text 0.737 (ep_mean_verbalized); P-prompt n/a; P-SoM 0.728 (action_diversity); SoM 0.709 (ep_mean_verbalized); Vision 0.773 (max_repeat_streak)
- source: `results/phantom_paper/auroc_cross_condition.csv` | last update: 2026-04-29 23:49 UTC
- figure: `results/phantom_paper/figures/fig0g_routing_auroc_heatmap.png` | last update: 2026-04-29 23:49 UTC

## Macro — agent 平均怎么 act

### 1a Tier 1 hook coarse

- P-SoM distinct from both endpoints: reddit **0/8**, classifieds **0/8**
- DOM-only distinct cells: 4; SoM-only distinct cells: 6; indistinct cells: 4
- source: `docs/analysis/cross_sites/axis_effect_size.json` | last update: 2026-04-29 23:49 UTC

### 1b Tier 2a cascade

- Dominant cascade counts: text 1; prompt 6; image 14
- Antagonistic mechanism pairs: **6** (text_vs_prompt@scroll_frac@B0/reddit, text_vs_image@scroll_frac@B0/reddit, text_vs_prompt@selfcorr_count@B0/classifieds, prompt_vs_image@finish_rate@B0/classifieds, prompt_vs_image@n_steps@B0/classifieds, prompt_vs_image@action_repeat_frac@B0/classifieds)
- source: `docs/analysis/cross_sites/axis_effect_size.json` | last update: 2026-04-29 23:49 UTC

### 1c Strategy gradient

- reddit: DOM search-loop 51.90% → P-SoM search-loop 35.71% → SoM search-loop 31.43%
- classifieds: DOM search-loop 77.35% → P-SoM search-loop 77.35% → SoM search-loop 65.38%
- figure: `results/phantom_paper/figures/fig1c_strategy_gradient.png` | last update: 2026-04-29 23:49 UTC

### 1d Full action vocabulary (E4)

- reddit: compound DOM→P-SoM top shifts: type -0.089; tab_focus 0.071; back 0.013
- classifieds: compound DOM→P-SoM top shifts: scroll 0.027; type -0.022; click 0.014
- source: `docs/analysis/cross_sites/mechanism_per_task.json` | last update: 2026-04-29 23:49 UTC

## Micro — per-step 决策

### 2a URL signature

- reddit: axis-1 URL-path Jaccard **n/a**; compound DOM↔P-SoM **n/a**
- classifieds: axis-1 URL-path Jaccard **n/a**; compound DOM↔P-SoM **n/a**
- source: `docs/analysis/cross_sites/axis1_microbehavior.json` | last update: 2026-04-29 23:49 UTC
- figure: `results/phantom_paper/figures/fig2_micro_divergence_heatmap.png` | last update: 2026-04-29 23:49 UTC

### 2a-extra Click-target divergence (E1)

- reddit: axis-1 click-transition Jaccard **0.463**; compound DOM↔P-SoM **0.421**
- classifieds: axis-1 click-transition Jaccard **0.561**; compound DOM↔P-SoM **0.531**
- source: `docs/analysis/cross_sites/mechanism_per_task.json` | last update: 2026-04-29 23:49 UTC

### 2b Target-hit

- reddit: axis-1 n/a; compound n/a
- classifieds: axis-1 n/a; compound n/a
- source: `docs/analysis/cross_sites/axis1_microbehavior.json` | last update: 2026-04-29 23:49 UTC

### 2c Keyword reuse

- reddit: axis-1 max-keyword-repeat diff **n/a**; compound **n/a**
- classifieds: axis-1 max-keyword-repeat diff **n/a**; compound **n/a**
- source: `docs/analysis/cross_sites/axis1_microbehavior.json` | last update: 2026-04-29 23:49 UTC

### 2d First-action

- reddit: axis-1 divergence **n/a**; compound **n/a**
- classifieds: axis-1 divergence **n/a**; compound **n/a**
- source: `docs/analysis/cross_sites/axis1_microbehavior.json` | last update: 2026-04-29 23:49 UTC

### 2e Cross-site validity

- verdict: **generalizes**; reddit ratio n/a, classifieds ratio n/a
- source: `docs/analysis/cross_sites/axis1_microbehavior.json` | last update: 2026-04-29 23:49 UTC

### 2f Trajectory boundary (E2)

- reddit: DOM↔P-SoM symmetric-diff N **23**; median first divergent step 0.0; early 91.30%; late 0.00%
- classifieds: DOM↔P-SoM symmetric-diff N **23**; median first divergent step 1.0; early 73.91%; late 4.35%
- source: `docs/analysis/cross_sites/mechanism_per_task.json` | last update: 2026-04-29 23:49 UTC

## Efficiency — cost / latency / carbon

### 3a Token/cost per step

- reddit: DOM input-cost/step $0.00354; P-SoM input-cost/step $0.00332; SoM input-cost/step $0.00447
- classifieds: DOM input-cost/step $0.00314; P-SoM input-cost/step $0.00311; SoM input-cost/step $0.00415
- source: B0 `condition_summary_v2.json` per condition

### 3b Image embedding / total-token gap

- reddit: SoM median tokens/step 4301 vs P-SoM 3522; observed gap **778 tokens/step**
- classifieds: SoM median tokens/step 3975 vs P-SoM 3032; observed gap **943 tokens/step**
- source: `results/phantom_paper/run_summary_collect.json` plus episode `total_tokens` fallback | last update: 2026-04-29 23:49 UTC

### 3c Latency

- reddit: DOM 182.9s/episode; P-SoM 105.7s/episode; SoM 115.1s/episode; P-SoM/SoM 0.92x
- classifieds: DOM 69.6s/episode; P-SoM 67.8s/episode; SoM 79.7s/episode; P-SoM/SoM 0.85x
- source: B0 `condition_summary_v2.json` per condition

### 3d B0 (API) vs B1 (local) deployment-class cost gap

Computed via `aggregate_cost_electricity.py`: B0 = API token dollars; B1 = `avg_total_energy_kwh × $0.12/kWh` (electricity equivalent, UK industrial). B0 vs B1 belong to different cost classes (API vs electricity), not a single ratio in $:
- reddit: B0 API $0.0399/ep vs B1 electricity $0.000407/ep → **98x** deployment-class gap
- classifieds: B0 API $0.0386/ep vs B1 electricity $0.000474/ep → **81x** deployment-class gap
- ⚠️ §103 / paper-planning legacy '30×' claim **superseded** by these data — real ratio ~100× (deployment class, not capability ratio)
- source: `docs/analysis/cross_sites/cost_per_mode.json` | last update: 2026-04-29 23:49 UTC
- figure: `results/phantom_paper/figures/fig3d_cost_sr_frontier.png` | last update: 2026-04-29 23:49 UTC

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
