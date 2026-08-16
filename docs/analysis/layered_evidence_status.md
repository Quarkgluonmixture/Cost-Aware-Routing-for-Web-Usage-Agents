# 4-dimension Evidence Status (live snapshot)

Generated: 2026-08-16 16:24 UTC
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
- canonical source: `docs/analysis/cross_sites/sr_per_mode.json` | last update: 2026-08-16 15:30 UTC
- standalone cite source: `docs/analysis/cross_sites/sr_per_mode.md` | last update: 2026-08-16 15:30 UTC

### 0b-extra Confidence calibration (E3)

- B0 reddit: best routing AUROC Vision **0.877**; ECE n/a in existing outputs
- B0 classifieds: best routing AUROC P-SoM **0.766**; ECE n/a in existing outputs
- B1 reddit: best routing AUROC P-prompt **0.706**; ECE n/a in existing outputs
- B1 classifieds: best routing AUROC DOM **0.870**; ECE n/a in existing outputs
- source: `docs/analysis/cross_sites/mechanism_per_task.json` | last update: 2026-08-16 15:30 UTC

### 0c Routing oracle (3→5-mode lift)

- classifieds: **+2.23pp** [0.45, 4.46] Wilcoxon p=0.0253, McNemar p=0.0312 ✅
  - single phantom lifts: +P-text +1.34pp; +P-SoM +1.34pp
- reddit: **+2.46pp** [0.49, 4.93] Wilcoxon p=0.0253, McNemar p=0.0312 ✅
  - single phantom lifts: +P-text +1.48pp; +P-SoM +0.99pp
- source: `results/phantom_paper/phantom_lift.csv` | last update: 2026-08-16 15:30 UTC
- figures: `results/phantom_paper/figures/fig0c_drop_one_oracle.png`, `results/phantom_paper/figures/fig0c_phantom_lift_bars.png` | last update: 2026-08-16 15:31 UTC

### 0d Task-pool Jaccard (Scenario C sentinel)

- classifieds: P-text↔P-SoM Jaccard **0.591** (✅ safe); threshold ≤0.7
- reddit: P-text↔P-SoM Jaccard **0.441** (✅ safe); threshold ≤0.7
- source: `results/phantom_paper/phantom_lift.csv` | last update: 2026-08-16 15:30 UTC
- figure: `results/phantom_paper/figures/fig0d_taskpool_jaccard.png` | last update: 2026-08-16 15:31 UTC

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
- figure: `results/phantom_paper/figures/fig0f_overlap_stacked_bar.png` | last update: 2026-08-16 15:31 UTC

### 0g Routing AUROC

- reddit: DOM n/a; P-text n/a; P-prompt n/a; P-SoM n/a; SoM n/a; Vision n/a
- classifieds: DOM n/a; P-text n/a; P-prompt n/a; P-SoM n/a; SoM n/a; Vision n/a
- source: `results/phantom_paper/auroc_cross_condition.csv` | last update: 2026-08-16 15:30 UTC
- figure: `results/phantom_paper/figures/fig0g_routing_auroc_heatmap.png` | last update: 2026-08-16 15:31 UTC

## Macro — agent 平均怎么 act

### 1a Tier 1 hook coarse

- P-SoM distinct from both endpoints: reddit **0/8**, classifieds **0/8**
- DOM-only distinct cells: 7; SoM-only distinct cells: 14; indistinct cells: 12
- source: `docs/analysis/cross_sites/axis_effect_size.json` | last update: 2026-08-16 15:31 UTC

### 1b Tier 2a cascade

- Dominant cascade counts: text 12; prompt 9; image 19
- Antagonistic mechanism pairs: **21** (text_vs_image@selfcorr_count@B0/reddit, text_vs_prompt@finish_rate@B0/reddit, text_vs_image@finish_rate@B0/reddit, text_vs_image@n_steps@B0/reddit, text_vs_image@finish_rate@B1/reddit, text_vs_image@n_steps@B1/reddit)
- source: `docs/analysis/cross_sites/axis_effect_size.json` | last update: 2026-08-16 15:31 UTC

### 1c Strategy gradient

- reddit: DOM search-loop 45.37% → P-SoM search-loop 33.66% → SoM search-loop 34.63%
- classifieds: DOM search-loop 80.80% → P-SoM search-loop 78.12% → SoM search-loop 68.30%
- figure: `results/phantom_paper/figures/fig1c_strategy_gradient.png` | last update: 2026-08-16 15:31 UTC

### 1d Full action vocabulary (E4)

- reddit: compound DOM→P-SoM top shifts: tab_focus 0.079; scroll -0.042; finish -0.032
- classifieds: compound DOM→P-SoM top shifts: scroll 0.025; type -0.024; tab_focus -0.010
- source: `docs/analysis/cross_sites/mechanism_per_task.json` | last update: 2026-08-16 15:30 UTC

## Micro — per-step 决策

### 2a URL signature

- reddit: axis-1 URL-path Jaccard **n/a**; compound DOM↔P-SoM **n/a**
- classifieds: axis-1 URL-path Jaccard **n/a**; compound DOM↔P-SoM **n/a**
- source: `docs/analysis/cross_sites/axis1_microbehavior.json` | last update: 2026-08-16 15:31 UTC
- figure: `results/phantom_paper/figures/fig2_micro_divergence_heatmap.png` | last update: 2026-08-16 15:31 UTC

### 2a-extra Click-target divergence (E1)

- reddit: axis-1 click-transition Jaccard **0.335**; compound DOM↔P-SoM **0.307**
- classifieds: axis-1 click-transition Jaccard **0.315**; compound DOM↔P-SoM **0.332**
- source: `docs/analysis/cross_sites/mechanism_per_task.json` | last update: 2026-08-16 15:30 UTC

### 2b Target-hit

- reddit: axis-1 n/a; compound n/a
- classifieds: axis-1 n/a; compound n/a
- source: `docs/analysis/cross_sites/axis1_microbehavior.json` | last update: 2026-08-16 15:31 UTC

### 2c Keyword reuse

- reddit: axis-1 max-keyword-repeat diff **n/a**; compound **n/a**
- classifieds: axis-1 max-keyword-repeat diff **n/a**; compound **n/a**
- source: `docs/analysis/cross_sites/axis1_microbehavior.json` | last update: 2026-08-16 15:31 UTC

### 2d First-action

- reddit: axis-1 divergence **n/a**; compound **n/a**
- classifieds: axis-1 divergence **n/a**; compound **n/a**
- source: `docs/analysis/cross_sites/axis1_microbehavior.json` | last update: 2026-08-16 15:31 UTC

### 2e Cross-site validity

- verdict: **generalizes**; reddit ratio n/a, classifieds ratio n/a
- source: `docs/analysis/cross_sites/axis1_microbehavior.json` | last update: 2026-08-16 15:31 UTC

### 2f Trajectory boundary (E2)

- reddit: DOM↔P-SoM symmetric-diff N **n/a**; median first divergent step n/a; early n/a; late n/a
- classifieds: DOM↔P-SoM symmetric-diff N **n/a**; median first divergent step n/a; early n/a; late n/a
- source: `docs/analysis/cross_sites/mechanism_per_task.json` | last update: 2026-08-16 15:30 UTC

## Efficiency — cost / latency / carbon

### 3a Token/cost per step

- reddit: DOM input-cost/step $0.00452; P-SoM input-cost/step $0.00424; SoM input-cost/step $0.00503
- classifieds: DOM input-cost/step $0.00393; P-SoM input-cost/step $0.00392; SoM input-cost/step $0.00481
- source: B0 `condition_summary_v2.json` per condition

### 3b Image embedding / total-token gap

- reddit: SoM median tokens/step 5058 vs P-SoM 4166; observed gap **893 tokens/step**
- classifieds: SoM median tokens/step 4745 vs P-SoM 3894; observed gap **851 tokens/step**
- source: `results/phantom_paper/run_summary_collect.json` plus episode `total_tokens` fallback | last update: 2026-08-16 15:30 UTC

### 3c Latency

- reddit: DOM 572.1s/episode; P-SoM 563.0s/episode; SoM 458.5s/episode; P-SoM/SoM 1.23x
- classifieds: DOM 115.0s/episode; P-SoM 121.1s/episode; SoM 106.7s/episode; P-SoM/SoM 1.14x
- source: B0 `condition_summary_v2.json` per condition

### 3d B0 (API) vs B1 (local) deployment-class cost gap

Computed via `aggregate_cost_electricity.py`: B0 = API token dollars; B1 = `avg_total_energy_kwh × $0.12/kWh` (electricity equivalent, UK industrial). B0 vs B1 belong to different cost classes (API vs electricity), not a single ratio in $:
- reddit: B0 API $0.1038/ep vs B1 electricity $0.001267/ep → **82x** deployment-class gap
- classifieds: B0 API $0.0694/ep vs B1 electricity $0.000648/ep → **107x** deployment-class gap
- ⚠️ §103 / paper-planning legacy '30×' claim **superseded** by these data — real ratio ~100× (deployment class, not capability ratio)
- source: `docs/analysis/cross_sites/cost_per_mode.json` | last update: 2026-08-16 15:30 UTC
- figure: `results/phantom_paper/figures/fig3d_cost_sr_frontier.png` | last update: 2026-08-16 15:32 UTC

## Paper Claim → Evidence Index

> Rewritten **2026-08-13** against the REALM submission's actual contributions. The retired C1-C6 table is preserved at the bottom of this file, marked superseded, because it is cited from older notes.
> **No numbers here by design** (§450.8) — this is an index; each row names the artifact that owns the figure.

| Claim (as submitted) | Dimensions | Owning artifact | Verdict |
|---|---|---|---|
| (i) The six representations are genuinely complementary: each solves tasks the others do not, their failures differ structurally, and which one wins is a property of the deployment | 0a, 0c, 0d, 0e, 0f, 1a-1d, 2a-2f | `sr_per_mode.md`, `cross_mode_failure_signatures.md`, `phantom_lift.csv` | ✅ live artifacts present |
| (ii) Most apparent oracle headroom is rerun variance; the bound that survives the rerun control is the cost ceiling at unchanged success | 0c, 3a | `noise_floor_inventory.md` | ⚠️ **two gaps, both load-bearing** — (1) read §1b there: the observed band is a draw, not a bound; (2) the measured floors cover DOM/SoM/Vision in a single cell, so **no phantom arm has a clean same-condition floor**, and the phantom arms are the ones the drop-one hero runs on. Replicates in flight (`_b1_floor_watcher.sh`) |
| (iii) The benchmarks cannot produce routing supervision: labels arrive at the success rate, so the tested routing constructions land at or below trivial fixed policies | 0b, 0b-extra, 0g | `router_pooled_tier_learnability`, `confidence_cascade.md` | ✅ five constructions + two controls |
| (iv) **NEW** Which routing question is answerable is decided by label supply AND signal, and those fail separately | 0b-extra, 0g | `abstention_learnability.md`, `abstention_site_transfer.md`, `early_abort_B0_classifieds.md`, `retry_vs_switch_label_supply.md` | ✅ four questions, one works — see next table. The one that works has now been tested across sites, and it **splits**: ranking transfers, the operating point does not |
| (v) **NEW** A representation carries deployment properties orthogonal to success rate | 2g*, 3e*, 3f*, 3g* | `representation_deployment_profile.md`, `latency_decomposition.md`, `energy_carbon_audit.md`, `fusion_premium.md` | ✅ see deployment table below |

\* Sub-codes marked with an asterisk are artifact-owned rather than rendered in the four dimensions above: they are static products, not live snapshots.

## Routing question → label supply → signal → outcome

> The finding this table encodes: the circularity named in the draft's §7 has **two distinguishable failure modes** — a label that does not exist, and a label that exists with no signal behind it. Only the pre-flight question has both. Figures live in the artifacts named in the last column.

| Routing question | Label supply | Signal | Outcome | Artifact |
|---|---|---|---|---|
| Which mode per task | **starved** — most cells admit no classifier under the min-class rule | — | fails | `router_pooled_tier_learnability` |
| Retry the same arm or switch | adequate, but most of the decision set is preference-free (both actions fail together) | — | no gain over fixed | `retry_vs_switch_label_supply.md` |
| Abort at step k | **every episode has one** | **absent** — prefix-only AUROC sits at its own shuffle null | fails, and loses to truncate-at-k | `early_abort_B0_classifieds.md` |
| **Abstain before running** | **every task has one** | **present** | **works within a cell — the only held-out cost saving in the paper.** Across sites it **splits**: the ranking survives the site change, the threshold does not, and the direction that ranks best is the one that fails worst as a policy | `abstention_learnability.md`, `abstention_site_transfer.md` |

⚠️ The last row is the only place in this file where a *generalisation* claim is licensed at all, and it is licensed in one half only. Two sites is two points: what `abstention_site_transfer` shows is that ranking survived the one site change available and calibration did not — not that either behaviour would recur on a third site.

⚠️ The 0b-extra row above reports whole-episode confidence AUROC, which for any prefix decision is looking at the future. The prefix-only recomputation is in `early_abort_B0_classifieds.md`, and it agrees with the April 2026 literature review's own context line (token-level confidence non-discriminative on this setup) rather than with 0b-extra.

## Deployment properties a single-mode deployment cannot measure about itself

| # | Finding | Dimension | Artifact |
|---|---|---|---|
| D1 | Fusion does not earn its premium against a rerun of one arm | 0a, 0c | `fusion_premium.md`, `leakage_sensitivity.md` |
| D2 | Unstable element ids move which element is chosen; position-keyed payloads are unaffected | 2a, 2b | draft §4 (`latency`/`churn` audit) |
| D3 | The feature a practitioner reaches for first (does the task supply a reference image) does not help | 0g | `router_covariate_baseline` |
| D4 | Which representation wins does not transfer across task sets | 0a, 0d | `sr_per_mode.md`, registered Jaccard sentinel (0d) |
| D5 | Latency is not where a representation change acts: the model call is a minority of a step, and the share moves with the serving stack, not with model size | 3c | `latency_decomposition.md` |
| D6 | Abstention buys a held-out saving; its oracle is far larger and not reachable | 3a | `abstention_learnability.md` |
| D7 | Failure diagnosability differs sharply by representation — vision's failures are the least attributable | 2g* | `representation_deployment_profile.md` §1 |
| D8 | The per-step token tail differs by representation far more than the median does; screenshot-bearing payloads are the flat ones | 3e* | `representation_deployment_profile.md` §2 |
| D9 | Carbon is not an independent axis on this instrument — it tracks elapsed time | 3f* | `energy_carbon_audit.md` |
| D10 | A pre-flight abstention policy's **ranking** survives a site change while its **threshold** does not; AUROC cannot warn you, because the failure is base-rate drift and AUROC is rank-based | 0g, 3a | `abstention_site_transfer.md` §4 |
| D11 | The one interval that showed fusion significantly beaten anywhere rested on accumulated site state, not on behaviour; under the canonical leak policy every cell's interval includes zero | 0a, 0c | `leakage_sensitivity.md` §3 |

⚠️ **Provenance boundary on every B0 figure in this file (B-1970, 2026-08-16).** The AWS proxy changed its response shape between the last archived B0 run and 2026-08-16. The drift was representation-only and lost no information, but it establishes that the provider mutates without notice: **archived B0 data and any future B0 data are not on the same provider snapshot**, and any analysis that subtracts one from the other has to say so. → `master_bug_catalog` B-1970.

## ⚠️ SUPERSEDED — pre-REALM claim matrix and 3-axis mechanism chain

> Kept because older notes cite C1-C6 and the axis chain by name. **Do not read these as live support.** C1 is the claim the rerun floor demoted (§398.8/§406); C2 is the retired 4-fold drop-in hook; C3/C6 belong to the 3-axis mechanism framing shelved 2026-05-14 with §5. The dimension pointers in them remain accurate as pointers.

| Claim | Dimensions cited | Verdict (as of the retired framing) |
|---|---|---|
| C1 P-SoM independent routing arm | 0a, 0c, 0d, 0g, 1a, 2a | ⚠️ superseded |
| C2 4-fold drop-in property | 3a, 3c, 0g, 0c | ⚠️ superseded |
| C3 3-axis hierarchical theory | 1b, 2a-2e, axis chain | ⚠️ shelved with §5 |
| C4 Aggregate macro can mislead about routing potential | 1a, 0d, 2a | ↺ survives in contribution (i) |
| C5 Prompt as task-conditional decision prior | 0b, 0b-extra, 0d, 1b, 1d, 2a-extra, 2f | ⚠️ mechanism reading shelved; behavioural rows stand |
| C6 Image is bidirectional modality fusion | 1b, 0e, 3b | ⚠️ shelved with §5 |

| Axis (retired framing) | Outcome | Macro | Micro | Efficiency |
|---|---|---|---|---|
| Axis 1 text payload | 0c single-phantom lift | 1b, 1d | 2a-2e, E1 | no image tax |
| Axis 2 prompt | 0d, 0b-extra | 1b, 1d | 2d, E1, E2 | prompt-only cost-neutral |
| Axis 3 image | 0e category recovery | 1b, 1d | 2a, E1/E2 | 3b token/latency tax |
| Compound P-SoM vs DOM | 0a/0c/0d, E3 | 1a, E4 | 2a, E1/E2 | 3a/3c |

⚠️ Naming trap: `mechanism_per_task.json` holds E1-E4, which are **behavioural** metrics (click-target divergence, trajectory boundary, confidence calibration, action vocabulary), not mechanistic evidence. The mechanistic line (activation patching, layer probes) is shelved, and its linear-probe results were themselves ruled the wrong tool for this contrastive setup (§111.2).
