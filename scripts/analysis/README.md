# Analysis Pipeline Dimension-organized View

This directory keeps the historical script layout intact. The dimension-organized view maps
existing scripts and figures onto `docs/checkpoints/paper_planning.md` §3:
**4-dimension Evidence + Mechanism Framework** (Outcome / Macro / Micro / Efficiency,
four orthogonal dimensions). Sub-codes (0a / 1c / 2a / 3d) remain as figure-internal anchors.

CLI alias `make analyze-layered` and filename `layered_evidence_status.md` are retained
for backward compatibility.

Inventory note: the current repo has 63 top-level analysis scripts plus 26
figure scripts under `scripts/analysis/figures/`. This README's dimension
mapping table covers only the core paper-grade subset.

## One-command Status

```bash
make analyze-layered
```

Output:

- `docs/analysis/layered_evidence_status.md`

The status report reads live artifacts and marks missing evidence with `⚠️`.

## Outcome dimension

Evidence: which tasks succeed, which mode wins, oracle lift, task-pool
complementarity, category SR, overlap depth, and routing AUROC.

| Artifact | Sub-code | Input source | Output | Current B0 examples |
|---|---:|---|---|---|
| `scripts/analysis/aggregate_phantom_lift.py` | 0c, 0d | B0 3-mode + phantom episode `*_summary_v2.json` files | `results/phantom_paper/phantom_lift.csv`, `results/phantom_paper/phantom_lift.md` | reddit 3→5 lift +5.24pp, Jaccard 0.571; classifieds +4.70pp, Jaccard 0.447 |
| `scripts/analysis/aggregate_routing_auroc.py` | 0g | per-run `analysis/signals/combined/tables/cross_mode_auroc.csv` | `results/phantom_paper/auroc_cross_condition.csv`, `.md`, `_summary.md` | reddit P-SoM max AUROC 0.720; classifieds P-SoM 0.728 |
| `scripts/analysis/figures/fig0e_category_mode_heatmap.py` | 0e | audit JSON + episode `success` (canonical) | `results/phantom_paper/figures/fig0e_category_mode_heatmap.png` | reddit/cls category × mode SR heatmap |
| `scripts/analysis/figures/fig0f_overlap_stacked_bar.py` | 0f | B0 `success` (canonical) task sets | `results/phantom_paper/figures/fig0f_overlap_stacked_bar.png` | reddit P-SoM solve depth distribution; classifieds P-SoM/P-text overlap depth |
| `scripts/analysis/figures/fig0c_phantom_lift_bars.py` | 0c viz | `results/phantom_paper/phantom_lift.csv` | `results/phantom_paper/figures/fig0c_phantom_lift_bars.png` | visualizes reddit +5.24pp and classifieds +4.70pp 3→5 oracle lift |
| `scripts/analysis/figures/fig0g_routing_auroc_heatmap.py` | 0g viz | `results/phantom_paper/auroc_cross_condition.csv` | `results/phantom_paper/figures/fig0g_routing_auroc_heatmap.png` | visualizes B0/B1 AUROC by condition, mode, and signal |
| `scripts/analysis/figures/fig0d_taskpool_jaccard.py` | 0 supporting | live episode `success` (canonical) sets | `results/phantom_paper/figures/fig0d_taskpool_jaccard.png` | solve-pool overlap sketch for B0/B1 observation arms |
| `scripts/analysis/figures/fig0c_drop_one_oracle.py` | 0 supporting | live episode `success` (canonical) sets | `results/phantom_paper/figures/fig0c_drop_one_oracle.png` (figure) + `results/phantom_paper/fig0c_drop_one_bootstrap_ci.csv` (data sidecar) | drop-one oracle loss for B0/B1 mode pools |

> ⚠️ **Estimand-label (AMENDMENT_02 §3 / AMENDMENT_04, 2026-05-24)**: the `phantom_lift.csv`
> "+5.24pp / +4.70pp 3→5 lift" numbers in the rows above are the **4-mode ADD** estimand
> (`4psom_vs_3`, 3→{4,5}-mode incremental oracle lift) = **Appendix-D / H1-deploy exploratory
> sensitivity**, NOT the 6-mode-strict H1 drop-one PRIMARY gate. The canonical H1 hero =
> bootstrap-percentile FE pool from `aggregate_phase1_full_prereg_decision` (6-mode strict ≤
> 4-mode ADD by construction; H1-strict effect TBD post-Phase-1a). Note `fig0c_drop_one_oracle.py`
> (drop-one) ≠ `fig0c_phantom_lift_bars.py` (4-mode ADD bars).

Outcome live sources also include canonical `success` SR and FP rate from:

- `results/visualwebarena/phase1/B0_3mode_reddit_20260422/*/episodes/*_summary_v2.json`
- `results/visualwebarena/phase1/B0_3mode_classifieds_20260413/*/episodes/*_summary_v2.json`
- `results/visualwebarena/phase1/B0_phantom_som_reddit_20260428/*/episodes/*_summary_v2.json`
- `results/visualwebarena/phase1/B0_phantom_som_classifieds_20260426/*/episodes/*_summary_v2.json`
- `results/visualwebarena/phase1/B0_phantom_text_reddit_20260427/*/episodes/*_summary_v2.json`
- `results/visualwebarena/phase1/B0_phantom_text_classifieds_20260427/*/episodes/*_summary_v2.json`

## Macro dimension

Evidence: action-type frequencies and cascade decomposition.

| Artifact | Sub-code | Input source | Output | Current B0 examples |
|---|---:|---|---|---|
| `scripts/analysis/axis_effect_size.py` | 1a, 1b | B0 reddit/classifieds 5-mode step JSONL + summary JSON | `docs/analysis/cross_sites/axis_effect_size.json`, `_report.md` | hook: P-SoM distinct from both endpoints on 6 cells total; cascade: 6 antagonistic pairs |
| `scripts/analysis/figures/fig1c_strategy_gradient.py` | 1c | B0 5-mode step JSONL | `results/phantom_paper/figures/fig1c_strategy_gradient.png` | reddit DOM search-loop 51.9% → P-SoM 35.7% → SoM 31.4%; classifieds shows image-axis recovery |
| `scripts/analysis/figures/fig1ab_cascade_diamond.py` | macro/micro schematic | no live data; mechanism schematic | `results/phantom_paper/figures/fig1ab_cascade_diamond.png` | explains text/prompt two-knob design |

## Micro dimension

Evidence: per-step decision quality using mode-invariant anchors.

| Artifact | Sub-code | Input source | Output | Current B0 examples |
|---|---:|---|---|---|
| `scripts/analysis/axis1_microbehavior.py` | 2a-2e | B0 reddit/classifieds 5-mode step JSONL + task configs | `docs/analysis/cross_sites/axis1_microbehavior.json`, `_report.md` | reddit axis-1 URL Jaccard 0.573, target-hit +3.47pp, keyword repeat -0.633; classifieds axis-1 URL Jaccard 0.904, target-hit +2.33pp |

Micro sub-evidence mapping:

- 2a URL signature: `axis_contrasts.*.url_jaccard_mean`
- 2b target-hit: `axis_contrasts.*.target_hit_rate_diff_pct_pts`
- 2c keyword reuse: `axis_contrasts.*.max_keyword_repeat_diff`
- 2d first-action: `axis_contrasts.*.first_action_divergence_rate`
- 2e cross-site validity: `cross_site_validity`

## Efficiency dimension

Evidence: cost, latency, image-token gap, and B0/B1 frontier.

| Artifact | Sub-code | Input source | Output | Current B0 examples |
|---|---:|---|---|---|
| `scripts/analysis/aggregate_cross_site.py` | 3a-3c | per-condition `condition_summary_v2.json` | `results/phantom_paper/cross_site/cross_site_aggregation.csv`, `_summary.json`, plots | B0/B1 cross-site SR/cost/latency table |
| `scripts/analysis/collect_analysis_summary.py` | 3 supporting | run-level summaries, condition summaries, per-run analysis outputs | `results/phantom_paper/run_summary_collect.json` | consolidated run metadata for 8 paper-grade VWA runs |
| `scripts/analysis/figures/fig3d_cost_sr_frontier.py` | 3d | `cost_per_mode.json` paper cost + canonical `success` | `results/phantom_paper/figures/fig3d_cost_sr_frontier.png` | B0 API-token vs B1 electricity-equivalent cost/SR Pareto frontier |
| `scripts/analysis/figures/fig3_regional_carbon.py` | 3 supporting | B1 episode energy + region intensity constants | `results/phantom_paper/figures/fig3_regional_carbon.png` | regional carbon sensitivity for B1 measured energy |
| `scripts/analysis/layered_status.py` | status | all 4-dimension artifacts listed above | `docs/analysis/layered_evidence_status.md` | live markdown snapshot with timestamps and missing-artifact warnings |

Efficiency live sources:

- token/cost and latency: `results/visualwebarena/phase1/*/*/condition_summary_v2.json`
- total-token gap fallback: episode-level `total_tokens / steps`
- run-level metadata: `results/phantom_paper/run_summary_collect.json`

## Per-run Diagnostics (Not Dimension-organized Evidence)

These scripts are intentionally not part of `make analyze-layered`; they remain
single-run or ad-hoc diagnostic tools used by `make analyze`, `make
analyze-paper`, or manual debugging.

| Script | Input source | Output | Notes |
|---|---|---|---|
| `scripts/analysis/analyze_experiment.py` | one run dir | delegated CLI output | wrapper for `p79.cli.analyze_experiment` |
| `scripts/analysis/analyze_reason_diagnostics.py` | one run dir | reason diagnostic tables/plots | invoked by `make reason-diag` |
| `scripts/analysis/analyze_cross_representation.py` | one run dir + reason rows | cross-representation outputs | invoked by `make cross-rep` |
| `scripts/analysis/analyze_confidence_calibration.py` | one run dir | per-run signal calibration and AUROC artifacts | upstream source for Outcome 0g aggregation |
| `scripts/analysis/compare_b0_b1.py` | one B0 run + one B1 run | `results/visualwebarena/phase1/b0_vs_b1_<site>/` | per-site comparison |
| `scripts/analysis/validate_run.py` | one run dir | validation JSON/report | data integrity checks |
| `scripts/analysis/diag_pattern_match.py` | one run dir | pattern-hit JSON | batch hard-rule diagnostics |
| `scripts/analysis/analyze_comment_selflink_loop.py` | fixed B0/B1 reddit runs | printed/diagnostic tables | reddit self-link loop diagnosis |
| `scripts/analysis/analyze_comment_selflink_loop_v2.py` | fixed B0/B1 reddit runs | printed/diagnostic tables | self-link loop follow-up |
| `scripts/analysis/analyze_reddit_selflink_cycle.py` | fixed B0/B1 reddit DOM runs | printed/diagnostic tables | self-link cycle escape patterns |
| `scripts/analysis/analyze_search_over_browse.py` | fixed B0/B1 reddit runs | printed/diagnostic tables | search-over-browse diagnosis |
| `scripts/analysis/analyze_noninteractive_click_earlystop.py` | fixed B0/B1 classifieds SoM runs | printed/diagnostic tables | noninteractive click early-stop diagnosis |
| `scripts/analysis/b0_vision_coordinate_errors.py` | fixed B0 classifieds run | printed/diagnostic tables | B0 vision coordinate error analysis |

## Make Targets

```makefile
analyze-layer0   # Outcome dimension: 0c, 0d, 0e, 0f, 0g plus live 0a/0b sources
analyze-layer1   # Macro dimension: 1a, 1b, 1c
analyze-layer2   # Micro dimension: 2a-2e
analyze-layer3   # Efficiency dimension: 3a-3d
analyze-layered  # all dimensions + docs/analysis/layered_evidence_status.md
```

Target names retained as CLI aliases for backward compatibility; paper-facing
naming is "Outcome / Macro / Micro / Efficiency" (4 orthogonal dimensions).
Existing `analyze` and `analyze-paper` targets are unchanged.

## Script inventory note (2026-05-23 audit)

This README's dimension mapping table covers the core paper-grade script subset
only. The full repo contains 63 top-level analysis scripts and 26 figure scripts
(`scripts/analysis/figures/`). Un-listed scripts fall into three categories:

- **Mechanism scripts** (`stage2_*`, `stage4_*`, `mechanism_per_task.py`):
  target paper §5 (activation patching / layer probes / logit lens), which is
  暂搁 (frozen) since 2026-05-14. The sole exception is the E3 block in
  `mechanism_per_task.py`, which feeds `fig0b_extra_confidence_calibration.py`
  and is therefore pulled into the default `make analysis` pipeline.

- **Ad-hoc diagnostic scripts** (`analyze_comment_selflink_loop*.py`,
  `analyze_reddit_selflink_cycle.py`, `analyze_search_over_browse.py`,
  `b0_vision_coordinate_errors.py`, `analyze_noninteractive_click_earlystop.py`,
  etc.): one-time forensic runs for specific bug investigations; not part of any
  make target.

- **Routing / learned-router scripts** (`analyze_routing_*.py`,
  `preregistration_decision_test.py`, `power_analysis.py`, etc.): part of the
  Phase 1a statistical gate and Pass-2 router pipeline, documented separately in
  `docs/checkpoints/phase1_plan.md §C`.

For the complete script list use `ls scripts/analysis/*.py` and
`ls scripts/analysis/figures/*.py`.
