# Cross-mode failure signatures — 36 landed Phase-1a conditions

Ruleset `8-reddit-p41p46-b1890fix` · 7686 episodes over 36 conditions (2 sites x 3 backbones x 6 modes).

Regenerate: `python3 scripts/analysis/aggregate_cross_mode_failure_signatures.py`


## Part A — signature frequency by mode (paper A §4.3)

Episode-level hit rate: share of episodes in which the signature fires at least once, pooled over the six (site, backbone) cells.

`vision` carries no element-id list, so id-space rules are structurally inapplicable there; the last column excludes it.


| rule | name | overall % | DOM | SoM | Vision | P-text | P-prompt | P-SoM | spread (all) | spread (text-bearing) |
|---|---|---|---|---|---|---|---|---|---|---|
| P31 | budget耗尽未完成 | 49.8 | 47.3 | 45.0 | 49.8 | 54.3 | 49.8 | 52.5 | 9.3 | 9.3 |
| P36 | WALK_FAIL_DEGENERATE | 48.8 | 59.6 | 45.9 | 20.7 | 52.2 | 57.5 | 57.1 | 38.9 | 13.7 |
| P5 | 感知缺失循环 | 43.5 | 44.3 | 38.1 | 55.6 | 36.1 | 46.0 | 40.7 | 19.5 | 9.9 |
| P14 | URL 自环 | 26.9 | 24.4 | 24.0 | 37.4 | 21.1 | 28.5 | 26.0 | 16.3 | 7.4 |
| P45 | IDENTICAL_FAILED_ACTION_STREAK | 26.8 | 37.4 | 30.0 | 0.0 | 25.6 | 36.8 | 31.2 | 37.4 | 11.8 |
| P43 | PAGE_EMBEDDED_VISUAL_NO_SCREENSHOT | 20.0 | 29.7 | 0.0 | 0.0 | 30.2 | 30.1 | 30.3 | 30.3 | 30.3 |
| P33 | 导航至裸图片URL幻觉 | 16.1 | 20.5 | 13.0 | 10.5 | 17.9 | 17.8 | 16.9 | 10.0 | 7.6 |
| P44 | HALLUCINATED_ELEMENT_REF | 13.5 | 20.9 | 8.1 | 0.0 | 16.0 | 27.4 | 8.8 | 27.4 | 19.3 |
| P12 | 从不翻页 | 12.6 | 12.6 | 16.2 | 9.3 | 10.9 | 15.1 | 11.7 | 6.9 | 5.4 |
| P25 | 跨站任务跳过其中一站 | 10.0 | 7.8 | 11.4 | 10.3 | 11.0 | 8.4 | 11.3 | 3.6 | 3.6 |
| P4 | 根节点误操作 | 7.4 | 0.4 | 10.3 | 0.0 | 13.9 | 0.5 | 19.4 | 19.4 | 19.0 |
| P18 | cheapest漏价格排序 | 6.1 | 6.6 | 6.2 | 6.2 | 5.5 | 6.2 | 6.0 | 1.0 | 1.0 |

**Top four signatures**: P31 (49.8%), P36 (48.8%), P5 (43.5%), P14 (26.9%).
Spread across the five text-bearing modes: P31 9.3 pp, P36 13.7 pp, P5 9.9 pp, P14 7.4 pp.
Including `vision`: P31 9.3 pp, P36 38.9 pp, P5 19.5 pp, P14 16.3 pp.

## Part B — hallucinated element references (paper A §4.2)

Two denominators, because they disagree. **action-step** = share of click / type / select_option steps naming an absent id, which weights an episode by how many actions it took; **episode incidence** = share of episodes with at least one such step, which does not. A thirty-step deadlock on one invalid id moves the first a great deal and the second by one episode.

Restricted to the canonical SCORED task set.


### By action-step

| cell | DOM | SoM | Vision | P-text | P-prompt | P-SoM |
|---|---|---|---|---|---|---|
| classifieds·B0 | 1.508 | 0.049 | 0.000 | 0.506 | 2.104 | 0.139 |
| classifieds·B1 | 2.033 | 0.000 | 0.000 | 3.961 | 4.313 | 0.351 |
| classifieds·B2 | 5.312 | 2.264 | 0.000 | 8.331 | 14.689 | 2.469 |
| reddit·B0 | 0.384 | 0.076 | 0.000 | 0.214 | 0.349 | 0.036 |
| reddit·B1 | 2.925 | 0.452 | 0.000 | 3.011 | 3.894 | 0.121 |
| reddit·B2 | 18.215 | 8.878 | 0.000 | 8.777 | 33.226 | 7.842 |

### By episode incidence

| cell | DOM | SoM | Vision | P-text | P-prompt | P-SoM |
|---|---|---|---|---|---|---|
| classifieds·B0 | 6.696 | 0.446 | 0.000 | 4.018 | 4.911 | 0.893 |
| classifieds·B1 | 8.929 | 0.000 | 0.000 | 20.089 | 14.732 | 3.125 |
| classifieds·B2 | 36.607 | 17.411 | 0.000 | 33.482 | 50.893 | 19.196 |
| reddit·B0 | 3.941 | 0.985 | 0.000 | 2.956 | 3.941 | 0.493 |
| reddit·B1 | 12.315 | 2.463 | 0.000 | 11.823 | 16.256 | 1.478 |
| reddit·B2 | 62.562 | 29.064 | 0.000 | 25.123 | 76.355 | 30.542 |

### The 2x2: which knob moves the rate

P-text = legend text under the DOM prompt; P-prompt = AXTree text under the SoM prompt. So the text knob is read at a fixed prompt family and vice versa.


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

### Which statements survive both denominators

| statement | by action-step | by episode incidence | quotable |
|---|---|---|---|
| legend's reduction is larger under the SoM prompt than the DOM prompt | 6/6 | 6/6 | **yes** |
| legend lowers the rate under the SoM prompt | 6/6 | 6/6 | **yes** |
| SoM prompt lowers the rate when the text is the legend | 6/6 | 5/6 | no |
| P-SoM is the lowest arm | 6/6 | 5/6 | no |
| P-prompt is the highest arm | 5/6 | 3/6 | no |
| legend lowers the rate under the DOM prompt | 3/6 | 5/6 | no |
| SoM prompt lowers the rate when the text is the AXTree | 1/6 | 1/6 | no |

Only the rows marked **yes** are stated in the paper. The interaction claim rests on the first row: the legend's effect on reference hallucination depends on which prompt it is paired with, in every cell under either denominator. The arms in which the prompt's advertised id scheme and the text's actual id scheme agree behave differently from the two mismatched arms.

Rows marked *no* are real under one denominator and not the other, which is why the second denominator is computed at all rather than assumed to agree.
