# Cross-mode failure signatures — 36 landed Phase-1a conditions

Ruleset `11-intent-text-fallback` · 7649 episodes over 36 conditions (2 sites x 3 backbones x 6 modes).

Regenerate: `python3 scripts/analysis/aggregate_cross_mode_failure_signatures.py`


## Part A — signature frequency by mode (paper A §4.3)

Episode-level hit rate: share of episodes in which the signature fires at least once, pooled over the six (site, backbone) cells.

`vision` carries no element-id list, so id-space rules are structurally inapplicable there; the last column excludes it.


| rule | name | overall % | DOM | SoM | Vision | P-text | P-prompt | P-SoM | spread (all) | spread (text-bearing) |
|---|---|---|---|---|---|---|---|---|---|---|
| P31 | budget耗尽未完成 | 49.9 | 47.3 | 45.4 | 49.8 | 54.3 | 49.8 | 52.5 | 8.8 | 8.8 |
| P36 | WALK_FAIL_DEGENERATE | 44.1 | 55.8 | 40.6 | 20.7 | 46.9 | 53.5 | 47.2 | 35.1 | 15.2 |
| P5 | 感知缺失循环 | 43.6 | 44.3 | 38.7 | 55.6 | 36.1 | 46.0 | 40.7 | 19.5 | 9.9 |
| P45 | IDENTICAL_FAILED_ACTION_STREAK | 26.9 | 37.4 | 30.4 | 0.0 | 25.6 | 36.8 | 31.2 | 37.4 | 11.8 |
| P14 | URL 自环 | 26.3 | 24.0 | 23.3 | 36.9 | 20.2 | 27.8 | 25.5 | 16.7 | 7.6 |
| P43 | PAGE_EMBEDDED_VISUAL_NO_SCREENSHOT | 20.1 | 29.7 | 0.0 | 0.0 | 30.2 | 30.1 | 30.3 | 30.3 | 30.3 |
| P33 | 导航至裸图片URL幻觉 | 16.1 | 20.5 | 13.2 | 10.5 | 17.9 | 17.8 | 16.9 | 10.0 | 7.3 |
| P44 | HALLUCINATED_ELEMENT_REF | 13.6 | 20.9 | 8.4 | 0.0 | 16.0 | 27.4 | 8.8 | 27.4 | 19.0 |
| P12 | 从不翻页 | 12.7 | 12.6 | 16.5 | 9.3 | 10.9 | 15.1 | 11.7 | 7.2 | 5.6 |
| P25 | 跨站任务跳过其中一站 | 10.0 | 7.8 | 10.9 | 10.3 | 11.0 | 8.4 | 11.3 | 3.5 | 3.5 |
| P4 | 根节点误操作 | 7.4 | 0.4 | 10.5 | 0.0 | 13.9 | 0.5 | 19.4 | 19.4 | 19.0 |
| P18 | cheapest漏价格排序 | 6.1 | 6.6 | 6.2 | 6.2 | 5.5 | 6.2 | 6.0 | 1.0 | 1.0 |

**Top four signatures**: P31 (49.9%), P36 (44.1%), P5 (43.6%), P45 (26.9%).
Spread across the five text-bearing modes, **pooled over cells**: P31 8.8 pp, P36 15.2 pp, P5 9.9 pp, P45 11.8 pp.
Including `vision`: P31 8.8 pp, P36 35.1 pp, P5 19.5 pp, P45 37.4 pp.

⚠️ **The pooled spread is not a within-cell spread.** Pooling sums numerators and denominators over the six cells before the rate is formed, so cell × mode variation cancels. Per cell, over the same five text-bearing modes:

| rule | pooled spread | max within-cell | median within-cell | worst cell |
|---|---|---|---|---|
| P31 | 8.8 pp | **15.3 pp** | 12.8 pp | reddit·B0 |
| P36 | 15.2 pp | **31.0 pp** | 22.5 pp | reddit·B2 |
| P5 | 9.9 pp | **48.8 pp** | 15.6 pp | reddit·B2 |
| P45 | 11.8 pp | **48.3 pp** | 17.4 pp | reddit·B2 |

So "mode-invariant" is only defensible as *similar after pooling*. No claim that any individual cell shows mode-invariance is supported here.


## Part B — hallucinated element references (paper A §4.2)

Two denominators, because they disagree. **action-step** = share of click / type / select_option steps naming an absent id, which weights an episode by how many actions it took; **episode incidence** = share of episodes with at least one such step, which does not. A thirty-step deadlock on one invalid id moves the first a great deal and the second by one episode.

Restricted to the canonical SCORED task set.


🚨 **Rates are comparable only WITHIN an id namespace** (row 2 of each table below). The metric counts actions naming an id the dispatch map lacks, and that map is keyed by raw CDP nodeIds for the AXTree modes but re-keyed to sequential 1..K for the SoM-family modes (`runner/main.py:2853-2860`). Under sparse native ids almost any slip lands outside the valid set and is counted; under dense 1..K a *wrong* element choice usually still names a valid id and is NOT counted. **SoM is renumbered too**, so a SoM-vs-DOM ratio is exactly as cross-namespace as a P-SoM-vs-DOM one.


### By action-step

| cell | DOM | SoM | Vision | P-text | P-prompt | P-SoM |
| *id namespace* | *native* | *compact 1..K* | *native (no marks)* | *compact 1..K* | *native* | *compact 1..K* |
|---|---|---|---|---|---|---|
| classifieds·B0 | 1.508 | 0.298 | 0.000 | 0.506 | 2.104 | 0.139 |
| classifieds·B1 | 2.033 | 0.000 | 0.000 | 3.961 | 4.313 | 0.351 |
| classifieds·B2 | 5.312 | 2.264 | 0.000 | 8.331 | 14.689 | 2.469 |
| reddit·B0 | 0.384 | 0.076 | 0.000 | 0.214 | 0.349 | 0.036 |
| reddit·B1 | 2.925 | 0.452 | 0.000 | 3.011 | 3.894 | 0.121 |
| reddit·B2 | 18.215 | 8.878 | 0.000 | 8.777 | 33.226 | 7.842 |

### By episode incidence

| cell | DOM | SoM | Vision | P-text | P-prompt | P-SoM |
| *id namespace* | *native* | *compact 1..K* | *native (no marks)* | *compact 1..K* | *native* | *compact 1..K* |
|---|---|---|---|---|---|---|
| classifieds·B0 | 6.696 | 1.070 | 0.000 | 4.018 | 4.911 | 0.893 |
| classifieds·B1 | 8.929 | 0.000 | 0.000 | 20.089 | 14.732 | 3.125 |
| classifieds·B2 | 36.607 | 17.411 | 0.000 | 33.482 | 50.893 | 19.196 |
| reddit·B0 | 3.941 | 0.985 | 0.000 | 2.956 | 3.941 | 0.493 |
| reddit·B1 | 12.315 | 2.463 | 0.000 | 11.823 | 16.256 | 1.478 |
| reddit·B2 | 62.562 | 29.064 | 0.000 | 25.123 | 76.355 | 30.542 |

### The 2x2: which knob moves the rate

P-text = legend text under the DOM prompt; P-prompt = AXTree text under the SoM prompt. So the text knob is read at a fixed prompt family and vice versa.


🚨 **Only the two prompt-effect columns are comparable.** The metric counts action steps naming an id absent from the dispatch map, and that map is keyed by raw CDP nodeIds for the AXTree arms (sparse: median 7,839-18,729, max 691,695) but re-keyed to sequential 1..K for the legend arms (dense: median K = 15-17, max 176; `som.py` `build_som_text_from_obs_text`). Under sparse ids almost any slip is outside the valid set and is counted; under dense ids a *wrong* element choice usually still names a valid id and is NOT counted. So a legend-vs-AXTree difference mixes a behaviour change with a detector-sensitivity change. The text columns and the lowest/highest ranks below are printed for completeness and must not be read as effects.


**By action-step**

| cell | text @ DOM prompt | text @ SoM prompt | prompt @ AXTree | prompt @ legend | lowest | highest |
|---|---|---|---|---|---|---|
| classifieds·B0 | -1.003 | -1.965 | +0.595 | -0.367 | P-SoM | P-prompt |
| classifieds·B1 | +1.928 | -3.961 | +2.279 | -3.610 | P-SoM | P-prompt |
| classifieds·B2 | +3.019 | -12.221 | +9.377 | -5.863 | P-SoM | P-prompt |
| reddit·B0 | -0.170 | -0.313 | -0.035 | -0.178 | P-SoM | DOM |
| reddit·B1 | +0.086 | -3.773 | +0.970 | -2.889 | P-SoM | P-prompt |
| reddit·B2 | -9.438 | -25.383 | +15.010 | -0.935 | P-SoM | P-prompt |

**By episode incidence**

| cell | text @ DOM prompt | text @ SoM prompt | prompt @ AXTree | prompt @ legend | lowest | highest |
|---|---|---|---|---|---|---|
| classifieds·B0 | -2.679 | -4.018 | -1.786 | -3.125 | P-SoM | DOM |
| classifieds·B1 | +11.161 | -11.607 | +5.804 | -16.964 | P-SoM | P-text |
| classifieds·B2 | -3.125 | -31.696 | +14.286 | -14.286 | P-SoM | P-prompt |
| reddit·B0 | -0.985 | -3.448 | +0.000 | -2.463 | P-SoM | DOM |
| reddit·B1 | -0.493 | -14.778 | +3.941 | -10.345 | P-SoM | P-prompt |
| reddit·B2 | -37.438 | -45.813 | +13.793 | +5.419 | P-text | P-prompt |

### Which statements are comparable, and which survive both denominators

| statement | comparable? | by action-step | by episode incidence |
|---|---|---|---|
| SoM prompt LOWERS the rate when ids are native (so 6−n cells RAISE it) | **yes** | 1/6 | 1/6 |
| SoM prompt LOWERS the rate when ids are compact | **yes** | 6/6 | 5/6 |
| legend's reduction is larger under the SoM prompt than the DOM prompt | **NO — cross-namespace** | 6/6 | 6/6 |
| legend lowers the rate under the SoM prompt | **NO — cross-namespace** | 6/6 | 6/6 |
| legend lowers the rate under the DOM prompt | **NO — cross-namespace** | 3/6 | 5/6 |
| P-SoM is the lowest arm | **NO — cross-namespace** | 6/6 | 5/6 |
| P-prompt is the highest arm | **NO — cross-namespace** | 5/6 | 3/6 |

The paper states only the two comparable rows. Together they are an interaction with opposite signs: moving to the SoM prompt **raises** the rate when the text supplies native ids (5/6 cells by action-step, 5/6 by episode) and **lowers** it when the text supplies the 1..K legend (6/6 and 5/6). Both halves hold the id namespace fixed, so neither is a detector artefact, and the sign flip is what a prompt-text mismatch account predicts: a prompt announcing marks 1..K helps when the text has them and hurts when it does not.

The rows marked NOT comparable are printed so the asymmetry is visible, not as evidence. Note that they are also the rows that move most between the two denominators.
