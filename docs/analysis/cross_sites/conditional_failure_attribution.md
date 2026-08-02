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
| `wa_red_B1` | 15 | 4 |
| **pooled** | **86** | **95** |

⚠️ TEXT is four arms against IMAGE's two, so a larger text-only count is partly arm count and must not be read as a larger effect.

## 2. Only the text channel solved it: how the IMAGE channel failed

Pooled over six cells. 172 losing-channel failure episodes on the disagreement set, against 2488 of that channel's failures overall.

| rule | name | on disagreement | baseline | enrichment | hits |
|---|---|---|---|---|---|
| `P17` | click-back振荡 | 5.8% | 4.2% | **1.39x** | 10 |
| `P12` | 从不翻页 | 14.0% | 14.8% | **0.95x** | 24 |
| `P4` | 根节点误操作 | 5.2% | 5.5% | **0.94x** | 9 |
| `P31` | budget耗尽未完成 | 49.4% | 54.1% | **0.91x** | 85 |
| `P36` | WALK_FAIL_DEGENERATE | 31.4% | 36.3% | **0.87x** | 54 |
| `P14` | URL 自环 | 27.9% | 34.2% | **0.81x** | 48 |
| `P5` | 感知缺失循环 | 42.4% | 52.6% | **0.81x** | 73 |
| `P45` | IDENTICAL_FAILED_ACTION_STREAK | 12.8% | 17.1% | **0.75x** | 22 |
| `P33` | 导航至裸图片URL幻觉 | 7.6% | 11.7% | **0.65x** **←** | 13 |

## 3. Only the image channel solved it: how the TEXT channel failed

Pooled over six cells. 380 losing-channel failure episodes on the disagreement set, against 5090 of that channel's failures overall.

| rule | name | on disagreement | baseline | enrichment | hits |
|---|---|---|---|---|---|
| `P27` | 找不到即放弃 | 2.9% | 1.0% | **3.01x** **←** | 11 |
| `P17` | click-back振荡 | 16.1% | 7.1% | **2.26x** **←** | 61 |
| `P16` | 视觉图像内容DOM必败 | 6.6% | 2.9% | **2.25x** **←** | 25 |
| `P43` | PAGE_EMBEDDED_VISUAL_NO_SCREENSHOT | 51.6% | 30.3% | **1.70x** **←** | 196 |
| `P19` | url_match过早搜索页finish | 2.1% | 1.4% | **1.55x** **←** | 8 |
| `P30` | 到达正确item后离开 | 2.1% | 1.5% | **1.39x** | 8 |
| `P4` | 根节点误操作 | 9.2% | 9.0% | **1.02x** | 35 |
| `P33` | 导航至裸图片URL幻觉 | 17.4% | 17.9% | **0.97x** | 66 |
| `P10` | 跨步数值记忆失败 | 2.1% | 2.2% | **0.94x** | 8 |
| `P5` | 感知缺失循环 | 42.1% | 45.5% | **0.92x** | 160 |
| `P12` | 从不翻页 | 12.1% | 13.8% | **0.88x** | 46 |
| `P14` | URL 自环 | 23.7% | 27.2% | **0.87x** | 90 |
| `P45` | IDENTICAL_FAILED_ACTION_STREAK | 30.5% | 36.1% | **0.84x** | 116 |
| `P36` | WALK_FAIL_DEGENERATE | 45.8% | 62.1% | **0.74x** | 174 |
| `P6` | 视觉任务 DOM 必然失败 | 3.7% | 5.6% | **0.66x** **←** | 14 |
| `P44` | HALLUCINATED_ELEMENT_REF | 11.3% | 18.8% | **0.60x** **←** | 43 |
| `P18` | cheapest漏价格排序 | 3.4% | 6.1% | **0.56x** **←** | 13 |
| `P31` | budget耗尽未完成 | 28.4% | 56.7% | **0.50x** **←** | 108 |
| `P25` | 跨站任务跳过其中一站 | 2.6% | 9.7% | **0.27x** **←** | 10 |

## 4. Reading

A signature near 1.0x is the null and most rows sit there: the losing channel mostly fails on these tasks the way it fails on every task. Rows away from 1.0x are the ones that name a mechanism for the complementarity, and they are the only rows this analysis licenses anyone to cite. The reporting floor is 8 pooled hits; rules below it are omitted rather than shown at unstable ratios.
