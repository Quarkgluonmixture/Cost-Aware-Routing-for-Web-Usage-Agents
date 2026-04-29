# Analysis Pipeline Layered View

This directory keeps the historical script layout intact. The layered view maps
existing scripts and figures onto `docs/checkpoints/paper_planning.md` §3:
**4-Layer Evidence + Mechanism Framework**.

Inventory note: the current repo has 20 top-level analysis scripts plus 11
figure scripts under `scripts/analysis/figures/` after adding
`layered_status.py`.

## One-command Status

```bash
make analyze-layered
```

Output:

- `docs/analysis/layered_evidence_status.md`

The status report reads live artifacts and marks missing evidence with `⚠️`.

## Layer 0 — Outcome

Evidence: which tasks succeed, which mode wins, oracle lift, task-pool
complementarity, category SR, overlap depth, and routing AUROC.

| Artifact | Layer | Input source | Output | Current B0 examples |
|---|---:|---|---|---|
| `scripts/analysis/aggregate_phantom_lift.py` | 0c, 0d | B0 3-mode + phantom episode `*_summary_v2.json` files | `results/phantom_paper/phantom_lift.csv`, `results/phantom_paper/phantom_lift.md` | reddit 3→5 lift +5.24pp, Jaccard 0.571; classifieds +4.70pp, Jaccard 0.447 |
| `scripts/analysis/aggregate_routing_auroc.py` | 0g | per-run `analysis/signals/combined/tables/cross_mode_auroc.csv` | `results/phantom_paper/auroc_cross_condition.csv`, `.md`, `_summary.md` | reddit P-SoM max AUROC 0.720; classifieds P-SoM 0.728 |
| `scripts/analysis/figures/fig0e_category_mode_heatmap.py` | 0e | audit JSON + episode `adjusted_success` | `results/phantom_paper/figures/fig0e_category_mode_heatmap.png` | reddit/cls category × mode adjusted SR heatmap |
| `scripts/analysis/figures/fig0f_overlap_stacked_bar.py` | 0f | B0 adjusted-success task sets | `results/phantom_paper/figures/fig0f_overlap_stacked_bar.png` | reddit P-SoM solve depth distribution; classifieds P-SoM/P-text overlap depth |
| `scripts/analysis/figures/fig0c_phantom_lift_bars.py` | 0c viz | `results/phantom_paper/phantom_lift.csv` | `results/phantom_paper/figures/fig0c_phantom_lift_bars.png` | visualizes reddit +5.24pp and classifieds +4.70pp 3→5 oracle lift |
| `scripts/analysis/figures/fig0g_routing_auroc_heatmap.py` | 0g viz | `results/phantom_paper/auroc_cross_condition.csv` | `results/phantom_paper/figures/fig0g_routing_auroc_heatmap.png` | visualizes B0/B1 AUROC by condition, mode, and signal |
| `scripts/analysis/figures/fig0d_taskpool_jaccard.py` | 0 supporting | live episode adjusted-success sets | `results/phantom_paper/figures/fig0d_taskpool_jaccard.png` | solve-pool overlap sketch for B0/B1 observation arms |
| `scripts/analysis/figures/fig0c_drop_one_oracle.py` | 0 supporting | live episode adjusted-success sets | `results/phantom_paper/figures/fig0c_drop_one_oracle.png` (figure) + `results/phantom_paper/fig0c_drop_one_bootstrap_ci.csv` (data sidecar) | drop-one oracle loss for B0/B1 mode pools |

Layer 0 live sources also include raw/adjusted SR and FP rate from:

- `results/visualwebarena/phase1/B0_3mode_reddit_20260422/*/episodes/*_summary_v2.json`
- `results/visualwebarena/phase1/B0_3mode_classifieds_20260413/*/episodes/*_summary_v2.json`
- `results/visualwebarena/phase1/B0_phantom_som_reddit_20260428/*/episodes/*_summary_v2.json`
- `results/visualwebarena/phase1/B0_phantom_som_classifieds_20260426/*/episodes/*_summary_v2.json`
- `results/visualwebarena/phase1/B0_phantom_text_reddit_20260427/*/episodes/*_summary_v2.json`
- `results/visualwebarena/phase1/B0_phantom_text_classifieds_20260427/*/episodes/*_summary_v2.json`

## Layer 1 — Macro Behavior

Evidence: action-type frequencies and cascade decomposition.

| Artifact | Layer | Input source | Output | Current B0 examples |
|---|---:|---|---|---|
| `scripts/analysis/axis_effect_size.py` | 1a, 1b | B0 reddit/classifieds 5-mode step JSONL + summary JSON | `docs/analysis/cross_sites/axis_effect_size.json`, `_report.md` | hook: P-SoM distinct from both endpoints on 4 cells total; cascade: 6 antagonistic pairs |
| `scripts/analysis/figures/fig1c_strategy_gradient.py` | 1c | B0 5-mode step JSONL | `results/phantom_paper/figures/fig1c_strategy_gradient.png` | reddit DOM search-loop 51.9% → P-SoM 35.7% → SoM 31.4%; classifieds shows image-axis recovery |
| `scripts/analysis/figures/fig1ab_cascade_diamond.py` | 1/2 schematic | no live data; mechanism schematic | `results/phantom_paper/figures/fig1ab_cascade_diamond.png` | explains text/prompt two-knob design |

## Layer 2 — Micro Behavior

Evidence: per-step decision quality using mode-invariant anchors.

| Artifact | Layer | Input source | Output | Current B0 examples |
|---|---:|---|---|---|
| `scripts/analysis/axis1_microbehavior.py` | 2a-2e | B0 reddit/classifieds 5-mode step JSONL + task configs | `docs/analysis/cross_sites/axis1_microbehavior.json`, `_report.md` | reddit axis-1 URL Jaccard 0.573, target-hit +3.47pp, keyword repeat -0.633; classifieds axis-1 URL Jaccard 0.904, target-hit +2.33pp |

Layer 2 sub-evidence mapping:

- 2a URL signature: `axis_contrasts.*.url_jaccard_mean`
- 2b target-hit: `axis_contrasts.*.target_hit_rate_diff_pct_pts`
- 2c keyword reuse: `axis_contrasts.*.max_keyword_repeat_diff`
- 2d first-action: `axis_contrasts.*.first_action_divergence_rate`
- 2e cross-site validity: `cross_site_validity`

## Layer 3 — Efficiency

Evidence: cost, latency, image-token gap, and B0/B1 frontier.

| Artifact | Layer | Input source | Output | Current B0 examples |
|---|---:|---|---|---|
| `scripts/analysis/aggregate_cross_site.py` | 3a-3c | per-condition `condition_summary_v2.json` | `results/phantom_paper/cross_site/cross_site_aggregation.csv`, `_summary.json`, plots | B0/B1 cross-site SR/cost/latency table |
| `scripts/analysis/collect_analysis_summary.py` | 3 supporting | run-level summaries, condition summaries, per-run analysis outputs | `results/phantom_paper/run_summary_collect.json` | consolidated run metadata for 8 paper-grade VWA runs |
| `scripts/analysis/figures/fig3d_cost_sr_frontier.py` | 3d | `cost_per_mode.json` paper cost + adjusted success | `results/phantom_paper/figures/fig3d_cost_sr_frontier.png` | B0 API-token vs B1 electricity-equivalent cost/SR Pareto frontier |
| `scripts/analysis/figures/fig3_regional_carbon.py` | 3 supporting | B1 episode energy + region intensity constants | `results/phantom_paper/figures/fig3_regional_carbon.png` | regional carbon sensitivity for B1 measured energy |
| `scripts/analysis/layered_status.py` | status | all 4-layer artifacts listed above | `docs/analysis/layered_evidence_status.md` | live markdown snapshot with timestamps and missing-artifact warnings |

Layer 3 live sources:

- token/cost and latency: `results/visualwebarena/phase1/*/*/condition_summary_v2.json`
- total-token gap fallback: episode-level `total_tokens / steps`
- run-level metadata: `results/phantom_paper/run_summary_collect.json`

## Per-run Diagnostics (Not Layered Evidence)

These scripts are intentionally not part of `make analyze-layered`; they remain
single-run or ad-hoc diagnostic tools used by `make analyze`, `make
analyze-paper`, or manual debugging.

| Script | Input source | Output | Notes |
|---|---|---|---|
| `scripts/analysis/analyze_experiment.py` | one run dir | delegated CLI output | wrapper for `p79.cli.analyze_experiment` |
| `scripts/analysis/analyze_reason_diagnostics.py` | one run dir | reason diagnostic tables/plots | invoked by `make reason-diag` |
| `scripts/analysis/analyze_cross_representation.py` | one run dir + reason rows | cross-representation outputs | invoked by `make cross-rep` |
| `scripts/analysis/analyze_confidence_calibration.py` | one run dir | per-run signal calibration and AUROC artifacts | upstream source for Layer 0g aggregation |
| `scripts/analysis/compare_b0_b1.py` | one B0 run + one B1 run | `results/visualwebarena/phase1/b0_vs_b1_<site>/` | per-site comparison |
| `scripts/analysis/validate_run.py` | one run dir | validation JSON/report | data integrity checks |
| `scripts/analysis/diag_pattern_match.py` | one run dir | pattern-hit JSON | batch hard-rule diagnostics |
| `scripts/analysis/analyze_comment_selflink_loop.py` | fixed B0/B1 reddit runs | printed/diagnostic tables | reddit self-link loop diagnosis |
| `scripts/analysis/analyze_comment_selflink_loop_v2.py` | fixed B0/B1 reddit runs | printed/diagnostic tables | self-link loop follow-up |
| `scripts/analysis/analyze_reddit_selflink_cycle.py` | fixed B0/B1 reddit DOM runs | printed/diagnostic tables | self-link cycle escape patterns |
| `scripts/analysis/analyze_search_over_browse.py` | fixed B0/B1 reddit runs | printed/diagnostic tables | search-over-browse diagnosis |
| `scripts/analysis/analyze_noninteractive_click_earlystop.py` | fixed B0/B1 classifieds SoM runs | printed/diagnostic tables | noninteractive click early-stop diagnosis |
| `scripts/analysis/b0_vision_coordinate_errors.py` | fixed B0 classifieds run | printed/diagnostic tables | B0 vision coordinate error analysis |
| `scripts/analysis/figures/fig_capability_b0_b1.py` | `docs/analysis/phantom_paper/disagreement_clusters.md` | `results/phantom_paper/figures/fig_capability_b0_b1.png` | B0/B1 capability contrast diagnostic |

## Make Targets

```makefile
analyze-layer0   # 0c, 0d, 0e, 0f, 0g plus live 0a/0b sources
analyze-layer1   # 1a, 1b, 1c
analyze-layer2   # 2a-2e
analyze-layer3   # 3a-3d
analyze-layered  # all layers + docs/analysis/layered_evidence_status.md
```

Existing `analyze` and `analyze-paper` targets are unchanged.
