# Cross-mode failure signatures — 36 landed Phase-1a conditions

Ruleset `8-reddit-p41p46-b1890fix` · 7722 episodes over 36 conditions (2 sites x 3 backbones x 6 modes).

Regenerate: `python3 scripts/analysis/aggregate_cross_mode_failure_signatures.py`


## Part A — signature frequency by mode (paper A §4.3)

Episode-level hit rate: share of episodes in which the signature fires at least once, pooled over the six (site, backbone) cells.

`vision` carries no element-id list, so id-space rules are structurally inapplicable there; the last column excludes it.


| rule | name | overall % | DOM | SoM | Vision | P-text | P-prompt | P-SoM | spread (all) | spread (text-bearing) |
|---|---|---|---|---|---|---|---|---|---|---|
| P31 | budget耗尽未完成 | 49.6 | 47.2 | 44.8 | 49.7 | 54.1 | 49.7 | 52.3 | 9.2 | 9.2 |
| P36 | WALK_FAIL_DEGENERATE | 48.7 | 59.5 | 45.8 | 20.7 | 52.1 | 57.3 | 56.9 | 38.9 | 13.7 |
| P5 | 感知缺失循环 | 43.3 | 44.2 | 38.0 | 55.3 | 36.0 | 45.8 | 40.6 | 19.3 | 9.9 |
| P14 | URL 自环 | 26.8 | 24.2 | 23.9 | 37.2 | 21.0 | 28.4 | 25.9 | 16.2 | 7.4 |
| P45 | IDENTICAL_FAILED_ACTION_STREAK | 26.8 | 37.3 | 29.9 | 0.0 | 25.6 | 36.8 | 31.1 | 37.3 | 11.7 |
| P43 | PAGE_EMBEDDED_VISUAL_NO_SCREENSHOT | 20.0 | 29.7 | 0.0 | 0.0 | 30.1 | 30.0 | 30.1 | 30.1 | 30.1 |
| P33 | 导航至裸图片URL幻觉 | 16.1 | 20.4 | 13.1 | 10.5 | 17.8 | 17.7 | 16.9 | 9.9 | 7.4 |
| P44 | HALLUCINATED_ELEMENT_REF | 13.5 | 20.9 | 8.1 | 0.0 | 16.0 | 27.4 | 8.8 | 27.4 | 19.3 |
| P12 | 从不翻页 | 12.6 | 12.6 | 16.2 | 9.2 | 10.8 | 15.0 | 11.7 | 7.0 | 5.4 |
| P25 | 跨站任务跳过其中一站 | 10.2 | 8.0 | 11.6 | 10.5 | 11.2 | 8.6 | 11.5 | 3.6 | 3.6 |
| P4 | 根节点误操作 | 7.4 | 0.4 | 10.3 | 0.0 | 13.9 | 0.5 | 19.3 | 19.3 | 18.9 |
| P18 | cheapest漏价格排序 | 6.1 | 6.5 | 6.1 | 6.2 | 5.5 | 6.2 | 6.0 | 1.0 | 1.0 |

**Top four signatures**: P31 (49.6%), P36 (48.7%), P5 (43.3%), P14 (26.8%).
Spread across the five text-bearing modes: P31 9.2 pp, P36 13.7 pp, P5 9.9 pp, P14 7.4 pp.
Including `vision`: P31 9.2 pp, P36 38.9 pp, P5 19.3 pp, P14 16.2 pp.

## Part B — hallucinated element references (paper A §4.2)

Rate over action-steps (click / type / select_option). `all` = every episode (the estimand the original measurement used); `failed` = failed episodes only (what `check_p44` itself can see).


| cell | DOM | SoM | Vision | P-text | P-prompt | P-SoM |
|---|---|---|---|---|---|---|
| classifieds·B0 | 1.508 | 0.049 | 0.000 | 0.506 | 2.104 | 0.139 |
| classifieds·B1 | 2.033 | 0.000 | 0.000 | 3.961 | 4.313 | 0.351 |
| classifieds·B2 | 5.312 | 2.264 | 0.000 | 8.331 | 14.689 | 2.469 |
| reddit·B0 | 0.380 | 0.076 | 0.000 | 0.213 | 0.347 | 0.036 |
| reddit·B1 | 2.935 | 0.451 | 0.000 | 2.983 | 3.914 | 0.120 |
| reddit·B2 | 18.095 | 8.775 | 0.000 | 8.710 | 33.056 | 7.779 |

### The 2x2: which knob moves the rate

P-text = legend text under the DOM prompt; P-prompt = AXTree text under the SoM prompt. So the text knob is read at a fixed prompt family and vice versa.


| cell | text @ DOM prompt | text @ SoM prompt | prompt @ AXTree | prompt @ legend | lowest | highest |
|---|---|---|---|---|---|---|
| classifieds·B0 | -1.003 | -1.965 | +0.595 | -0.367 | P-SoM | P-prompt |
| classifieds·B1 | +1.928 | -3.961 | +2.279 | -3.610 | P-SoM | P-prompt |
| classifieds·B2 | +3.019 | -12.221 | +9.377 | -5.863 | P-SoM | P-prompt |
| reddit·B0 | -0.166 | -0.312 | -0.032 | -0.178 | P-SoM | DOM |
| reddit·B1 | +0.048 | -3.794 | +0.979 | -2.863 | P-SoM | P-prompt |
| reddit·B2 | -9.385 | -25.277 | +14.960 | -0.931 | P-SoM | P-prompt |

- P-SoM is the lowest-rate arm in **6/6** cells; P-prompt the highest in **5/6**.
- Substituting the legend for the AXTree lowers the rate in **6/6** cells under the SoM prompt but only **3/6** under the DOM prompt.
- Moving to the SoM prompt lowers the rate in **6/6** cells when the text is the legend, and in **1/6** when it is the AXTree.

The sign of each knob's effect depends on the other knob's setting, so the rate is not attributable to either knob as a main effect. The arms in which the prompt's advertised id scheme and the text's actual id scheme agree (DOM, P-SoM) behave differently from the two mismatched arms.
