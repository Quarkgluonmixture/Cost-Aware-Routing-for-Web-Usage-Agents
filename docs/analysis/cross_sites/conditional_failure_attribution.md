---
type: analysis
status: complete
created: 2026-08-02
purpose: when one channel uniquely solves a task, how the other channel failed on it
post_hoc_exploratory: true
scope_warning: TEXT is four arms and IMAGE is two, so the two sides' task counts are not comparable to each other. Only within-channel enrichment is read. Enrichment is a ratio of hit rates, not a test; no interval accompanies it.
producer: scripts/analysis/aggregate_conditional_failure_attribution.py
---

# Conditional failure attribution

Regenerate: `.venv/bin/python3 scripts/analysis/aggregate_conditional_failure_attribution.py`

The existing signature table is a marginal cut: which rules fire in which mode overall. This is the paired cut. Within a cell the six modes see one task set, so we can ask what the losing channel did on exactly the tasks the winning channel got and it did not. **Enrichment** is that signature's hit rate among the losing channel's failures on the disagreement set, over its hit rate across all its failures in the same cells. 1.0 means the channel failed there the way it fails everywhere.

## 1. Disagreement set sizes

| cell | only TEXT solves | only IMAGE solves |
|---|---|---|
| `cls_B0` | 21 | 33 |
| `red_B0` | 16 | 10 |
| `cls_B1` | 12 | 30 |
| `red_B1` | 7 | 4 |
| `cls_B2` | 6 | 10 |
| `red_B2` | 9 | 4 |
| **pooled** | **71** | **91** |

⚠️ TEXT is four arms against IMAGE's two, so a larger text-only count is partly arm count and must not be read as a larger effect.

## 2. Only the text channel solved it: how the IMAGE channel failed

Pooled over six cells. 142 losing-channel failure episodes on the disagreement set, against 2304 of that channel's failures overall.

| rule | name | on disagreement | baseline | enrichment | hits |
|---|---|---|---|---|---|
| `P17` | click-back振荡 | 7.0% | 4.5% | **1.56x** **←** | 10 |
| `P4` | 根节点误操作 | 6.3% | 5.7% | **1.11x** | 9 |
| `P36` | WALK_FAIL_DEGENERATE | 33.8% | 37.0% | **0.91x** | 48 |
| `P31` | budget耗尽未完成 | 43.0% | 52.7% | **0.82x** | 61 |
| `P14` | URL 自环 | 27.5% | 34.2% | **0.80x** | 39 |
| `P12` | 从不翻页 | 11.3% | 14.2% | **0.79x** | 16 |
| `P5` | 感知缺失循环 | 38.0% | 52.1% | **0.73x** | 54 |
| `P33` | 导航至裸图片URL幻觉 | 8.5% | 12.2% | **0.69x** | 12 |
| `P45` | IDENTICAL_FAILED_ACTION_STREAK | 10.6% | 16.7% | **0.63x** **←** | 15 |

## 3. Only the image channel solved it: how the TEXT channel failed

Pooled over six cells. 364 losing-channel failure episodes on the disagreement set, against 4737 of that channel's failures overall.

| rule | name | on disagreement | baseline | enrichment | hits |
|---|---|---|---|---|---|
| `P27` | 找不到即放弃 | 3.0% | 1.0% | **2.98x** **←** | 11 |
| `P17` | click-back振荡 | 16.8% | 7.6% | **2.20x** **←** | 61 |
| `P16` | 视觉图像内容DOM必败 | 6.9% | 3.1% | **2.18x** **←** | 25 |
| `P43` | PAGE_EMBEDDED_VISUAL_NO_SCREENSHOT | 53.8% | 32.5% | **1.66x** **←** | 196 |
| `P19` | url_match过早搜索页finish | 2.2% | 1.5% | **1.51x** **←** | 8 |
| `P30` | 到达正确item后离开 | 2.2% | 1.6% | **1.35x** | 8 |
| `P4` | 根节点误操作 | 9.3% | 9.2% | **1.01x** | 34 |
| `P33` | 导航至裸图片URL幻觉 | 17.9% | 18.8% | **0.95x** | 65 |
| `P10` | 跨步数值记忆失败 | 2.2% | 2.4% | **0.92x** | 8 |
| `P5` | 感知缺失循环 | 41.5% | 45.2% | **0.92x** | 151 |
| `P12` | 从不翻页 | 11.8% | 13.6% | **0.87x** | 43 |
| `P14` | URL 自环 | 22.8% | 27.0% | **0.84x** | 83 |
| `P45` | IDENTICAL_FAILED_ACTION_STREAK | 29.7% | 35.4% | **0.84x** | 108 |
| `P36` | WALK_FAIL_DEGENERATE | 44.2% | 61.2% | **0.72x** | 161 |
| `P6` | 视觉任务 DOM 必然失败 | 3.8% | 6.0% | **0.64x** **←** | 14 |
| `P44` | HALLUCINATED_ELEMENT_REF | 11.5% | 19.8% | **0.58x** **←** | 42 |
| `P18` | cheapest漏价格排序 | 3.6% | 6.5% | **0.55x** **←** | 13 |
| `P31` | budget耗尽未完成 | 26.1% | 55.1% | **0.47x** **←** | 95 |
| `P25` | 跨站任务跳过其中一站 | 2.7% | 10.4% | **0.26x** **←** | 10 |

## 4. Reading

A signature near 1.0x is the null and most rows sit there: the losing channel mostly fails on these tasks the way it fails on every task. Rows away from 1.0x are the ones that name a mechanism for the complementarity, and they are the only rows this analysis licenses anyone to cite. The reporting floor is 8 pooled hits; rules below it are omitted rather than shown at unstable ratios.
