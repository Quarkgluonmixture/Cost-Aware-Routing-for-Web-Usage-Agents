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
| `wa_red_B0` | 23 | 6 |
| **pooled** | **109** | **101** |

⚠️ TEXT is four arms against IMAGE's two, so a larger text-only count is partly arm count and must not be read as a larger effect.

## 2. Only the text channel solved it: how the IMAGE channel failed

Pooled over six cells. 218 losing-channel failure episodes on the disagreement set, against 2653 of that channel's failures overall.

| rule | name | on disagreement | baseline | enrichment | hits |
|---|---|---|---|---|---|
| `P49` | SUBMIT_PAGE_ANCHOR_MISCLICK | 3.7% | 1.0% | **3.61x** **←** | 8 |
| `P17` | click-back振荡 | 4.6% | 3.9% | **1.17x** | 10 |
| `P12` | 从不翻页 | 13.8% | 14.8% | **0.93x** | 30 |
| `P31` | budget耗尽未完成 | 49.5% | 54.2% | **0.91x** | 108 |
| `P36` | WALK_FAIL_DEGENERATE | 27.1% | 31.2% | **0.87x** | 59 |
| `P5` | 感知缺失循环 | 40.8% | 51.5% | **0.79x** | 89 |
| `P14` | URL 自环 | 25.7% | 32.5% | **0.79x** | 56 |
| `P45` | IDENTICAL_FAILED_ACTION_STREAK | 13.3% | 17.1% | **0.78x** | 29 |
| `P4` | 根节点误操作 | 4.1% | 5.3% | **0.78x** | 9 |
| `P33` | 导航至裸图片URL幻觉 | 7.3% | 11.5% | **0.64x** **←** | 16 |

## 3. Only the image channel solved it: how the TEXT channel failed

Pooled over six cells. 404 losing-channel failure episodes on the disagreement set, against 5388 of that channel's failures overall.

| rule | name | on disagreement | baseline | enrichment | hits |
|---|---|---|---|---|---|
| `P27` | 找不到即放弃 | 3.2% | 1.4% | **2.31x** **←** | 13 |
| `P17` | click-back振荡 | 15.1% | 6.7% | **2.25x** **←** | 61 |
| `P16` | 视觉图像内容DOM必败 | 6.2% | 2.8% | **2.24x** **←** | 25 |
| `P43` | PAGE_EMBEDDED_VISUAL_NO_SCREENSHOT | 48.5% | 29.3% | **1.65x** **←** | 196 |
| `P19` | url_match过早搜索页finish | 2.0% | 1.3% | **1.55x** **←** | 8 |
| `P30` | 到达正确item后离开 | 2.0% | 1.4% | **1.39x** | 8 |
| `P4` | 根节点误操作 | 8.7% | 8.6% | **1.01x** | 35 |
| `P12` | 从不翻页 | 14.4% | 14.6% | **0.99x** | 58 |
| `P5` | 感知缺失循环 | 43.8% | 45.5% | **0.96x** | 177 |
| `P33` | 导航至裸图片URL幻觉 | 16.3% | 17.2% | **0.95x** | 66 |
| `P10` | 跨步数值记忆失败 | 2.0% | 2.2% | **0.90x** | 8 |
| `P45` | IDENTICAL_FAILED_ACTION_STREAK | 32.9% | 36.5% | **0.90x** | 133 |
| `P14` | URL 自环 | 21.8% | 25.7% | **0.85x** | 88 |
| `P36` | WALK_FAIL_DEGENERATE | 41.6% | 53.5% | **0.78x** | 168 |
| `P6` | 视觉任务 DOM 必然失败 | 3.5% | 5.3% | **0.66x** **←** | 14 |
| `P44` | HALLUCINATED_ELEMENT_REF | 10.6% | 17.9% | **0.60x** **←** | 43 |
| `P18` | cheapest漏价格排序 | 3.2% | 5.8% | **0.56x** **←** | 13 |
| `P31` | budget耗尽未完成 | 30.0% | 55.8% | **0.54x** **←** | 121 |
| `P25` | 跨站任务跳过其中一站 | 2.5% | 9.2% | **0.27x** **←** | 10 |

## 4. Rules that cannot be compared across sites

Several P-rules carry site gates or are structurally inapplicable outside one site, so a 0.0% on one site is the gate and not a measurement. Verified firing rates over all episodes, which is the check that must precede any cross-site reading of a row above:

| rule | VWA cls | VWA red | WA red | comparable across sites? |
|---|---|---|---|---|
| `P6` visual-task-DOM-must-fail | 7.9% | **0.0%** | **0.0%** | **no** — gated off all reddit |
| `P16` visual-image-content | 3.9% | **0.0%** | **0.0%** | **no** — gated off all reddit |
| `P17` click-back oscillation | 11.5% | **0.0%** | **0.0%** | **no** — classifieds only |
| `P25` cross-site task skips a site | 5.1% | 15.9% | **0.0%** | no — WA has no cross-site tasks |
| `P43` page-embedded visual, no screenshot | 20.4% | 19.5% | **0.0%** | **yes** — ungated; the 0.0% is real |
| `P27` gives up when not found | 1.6% | 1.1% | 0.3% | yes |
| `P31` budget exhausted | 30.8% | 70.2% | 65.1% | yes |
| `P45` / `P36` / `P5` | 26-50% | 27-47% | 32-50% | yes |

⚠️ `P43` is ungated and its WA zero is a property of the task set: no WA reddit intent matches the visual-intent regex. **But P43 is a neutral (task x mode) label by its own definition, not a failure mechanism.** Its docstring records a controlled dom->som comparison on exactly this task set measuring +0.00 / +1.56 / +0.00 pp from restoring the screenshot. P43 therefore says WHERE the image channel's unique wins sit, and explicitly does not say the text channel failed *because* the screenshot was withheld.

## 5. Reading

A signature near 1.0x is the null and most rows sit there: the losing channel mostly fails on these tasks the way it fails on every task. Rows away from 1.0x are the ones that name a mechanism for the complementarity, and they are the only rows this analysis licenses anyone to cite. The reporting floor is 8 pooled hits; rules below it are omitted rather than shown at unstable ratios.

## 5. Is the text-wins side unexplained, or just outside the vocabulary?

The rule-based cut leaves that direction as a residual — nothing clears 1.5×. The objection writes itself: the ruleset was discovered on VisualWebArena, so it can only find VWA-shaped failures, and an absent signature may be a property of the vocabulary rather than of the world. These probes bypass the vocabulary entirely — each is computed from raw step fields, never from a rule hit — and ask the same question directly.

| candidate mechanism | on the disagreement set | that channel's baseline | enrichment |
|---|---|---|---|
| never searched | 0.387 | 0.468 | **0.83×** |
| ran out of budget (>=30 steps) | 0.535 | 0.658 | **0.81×** |
| action failure on over half the steps | 0.310 | 0.416 | **0.74×** |
| page unchanged on over half the steps | 0.331 | 0.431 | **0.77×** |
| finished in five steps or fewer | 0.162 | 0.141 | **1.15×** |
| any parse failure | 0.092 | 0.109 | **0.84×** |

n = 142 disagreement episodes against 2304 baseline failures. **The largest enrichment is 1.15×**, and most candidates sit *below* 1: on the tasks the text channel uniquely solves, the image channel's failures are less pathological than its failures elsewhere, not more. That sharpens the original wording — it is not merely that it fails the way it fails everywhere, it is that it fails **more blandly**: it did not arrive, rather than breaking somewhere nameable.

⚠️ Six candidates chosen by us, so this cannot show that no mechanism exists. What it does close is the specific objection that the residual is an artifact of a VWA-shaped rule vocabulary: these six do not use that vocabulary and find nothing either.
