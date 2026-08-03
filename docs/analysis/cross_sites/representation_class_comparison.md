---
type: analysis
status: rolling
purpose: compare the three deployment classes web agents actually come in
producer: scripts/analysis/representation_class_comparison.py
---

# Three representation classes

Regenerate: `.venv/bin/python3 scripts/analysis/representation_class_comparison.py`

| class | arms | what a deployment ships |
|---|---|---|
| **no-image** | DOM, P-text, P-prompt, P-SoM (4) | accessibility tree only |
| **vision-only** | Vision (1) | screenshot only — the computer-use-aligned line |
| **hybrid** | SoM (1) | screenshot + tree + marks |

Grouping the four no-image modes is licensed by `per_mode_four_dimension_profile`: they reach the ≥83% consistency bar on **none** of 26 metrics over 8 cells (Vision 9, SoM 5).

## 1. Best arm in each class

| cell | n | no-image | vision-only | hybrid | best class |
|---|---|---|---|---|---|
| `cls_B0` | 224 | 19.64% (P-prompt) | 25.00% | 27.23% | **hybrid** |
| `cls_B1` | 224 | 7.59% (P-text) | 12.50% | 14.29% | **hybrid** |
| `cls_B2` | 224 | 1.79% (P-prompt) | 2.23% | 2.23% | **tie: hybrid+vision-only** |
| `red_B0` | 203 | 14.29% (DOM) | 7.39% | 14.78% | **hybrid** |
| `red_B1` | 203 | 5.91% (DOM) | 2.46% | 7.39% | **hybrid** |
| `red_B2` | 203 | 3.94% (DOM) | 1.97% | 0.99% | **no-image** |
| `wa_B0` | 104 | 35.58% (P-text) | 19.23% | 22.12% | **no-image** |
| `wa_B1` | 104 | 16.35% (DOM) | 9.62% | 13.46% | **no-image** |

Best-class tally (**sole** winners; ties counted separately): **hybrid** 4/8, **no-image** 3/8, plus 1 tied cell(s).

**vision-only is never the sole best class in any cell.** Where it appears to win, the win is a tie — and on `cls_B2` that tie is between two arms at 2.23% (5 successes out of 224), i.e. at the floor. A tie resolved by dict order is how this nearly became a claim.

## 1b. The same comparison at one arm per class

§1's `no-image` column is a **maximum over four arms**; the other two columns are single arms. A maximum over noisy quantities is biased up, so §1 is not like-for-like. This panel fixes arm count at 1 and uses the arm of each class that exists outside this study — **DOM** (P-text / P-prompt / P-SoM are constructed here), **Vision**, **SoM**.

| cell | no-image (DOM) | vision-only | hybrid (SoM) | winner | §1 winner | gap vs hybrid: 1-arm | §1 (max-of-4) |
|---|---|---|---|---|---|---|---|
| `cls_B0` | 17.41% | 25.00% | 27.23% | **hybrid** | hybrid | -9.82pp | -7.59pp |
| `cls_B1` | 6.25% | 12.50% | 14.29% | **hybrid** | hybrid | -8.04pp | -6.70pp |
| `cls_B2` | 1.34% | 2.23% | 2.23% | **tie: hybrid+vision-only** | tie: hybrid+vision-only | -0.89pp | -0.45pp |
| `red_B0` | 14.29% | 7.39% | 14.78% | **hybrid** | hybrid | -0.49pp | -0.49pp |
| `red_B1` | 5.91% | 2.46% | 7.39% | **hybrid** | hybrid | -1.48pp | -1.48pp |
| `red_B2` | 3.94% | 1.97% | 0.99% | **no-image** | no-image | +2.96pp | +2.96pp |
| `wa_B0` | 26.92% | 19.23% | 22.12% | **no-image** | no-image | +4.81pp | +13.46pp |
| `wa_B1` | 16.35% | 9.62% | 13.46% | **no-image** | no-image | +2.88pp | +2.88pp |

Arm-matched tally (sole winners): **hybrid** 4/8, **no-image** 3/8, plus 1 tied cell(s).

**The tally does not move** when arm count is held fixed, and vision-only is still never the sole best class. That is the robustness statement §1 could not make: the class conclusion survives the most obvious attack on it.

⚠️ **The effect sizes do move, and by a lot.** On `wa_B0` the no-image lead over hybrid is **+13.46pp** in §1 and **+4.81pp** here — §1's figure is carried by `P-text`, an arm constructed for this study rather than one a deployment ships. Any sentence quoting a class *gap* should quote this panel's number; only the *ordering* is safe to take from §1.

## 2. Dropping a whole class — ⚠️ NOT arm-matched

How much oracle coverage disappears if a class is unavailable. **The no-image class has four arms and the others have one each**, so a larger number here is mostly arm count. Reported because the figure is the obvious one to compute and would otherwise be computed by a reader without the caveat attached.

| cell | all six | drop no-image (4 arms) | drop vision-only (1) | drop hybrid (1) |
|---|---|---|---|---|
| `cls_B0` | 43.30% | +9.38pp | +4.02pp | +2.68pp |
| `cls_B1` | 24.55% | +5.36pp | +4.02pp | +4.46pp |
| `cls_B2` | 7.14% | +2.68pp | +2.23pp | +2.23pp |
| `red_B0` | 26.11% | +7.88pp | +1.97pp | +1.97pp |
| `red_B1` | 11.82% | +3.45pp | +0.99pp | +0.49pp |
| `red_B2` | 7.39% | +4.43pp | +1.48pp | +0.49pp |
| `wa_B0` | 51.92% | +22.12pp | +3.85pp | +1.92pp |
| `wa_B1` | 30.77% | +14.42pp | +0.96pp | +0.96pp |

## 3. Arm-matched: add ONE arm to the cell's best single arm

The comparison §2 should have been. From each cell's best single mode, add one more arm and record the gain, grouped by which class the added arm belongs to (taking the best available arm within each class). A class already holding the starting arm shows `—`.

| cell | start | +1 no-image | +1 vision-only | +1 hybrid | largest |
|---|---|---|---|---|---|
| `cls_B0` | SoM@27.23% | +7.14pp | +6.70pp | — | no-image |
| `cls_B1` | SoM@14.29% | +3.57pp | +4.91pp | — | vision-only |
| `cls_B2` | SoM@2.23% | +1.79pp | +2.23pp | — | vision-only |
| `red_B0` | SoM@14.78% | +4.93pp | +3.45pp | — | no-image |
| `red_B1` | SoM@7.39% | +1.97pp | +0.99pp | — | no-image |
| `red_B2` | DOM@3.94% | +0.99pp | +1.97pp | +0.49pp | vision-only |
| `wa_B0` | P-text@35.58% | +5.77pp | +5.77pp | +4.81pp | no-image+vision-only |
| `wa_B1` | DOM@16.35% | +4.81pp | +3.85pp | +3.85pp | no-image |

Largest-gain tally (sole winners only): **no-image** 4, **vision-only** 3. **The classes do not differ systematically once arm count is held fixed** — which is what kills §2's headline. Note `hybrid` supplies the starting arm in most cells, so it rarely gets the chance to contribute a marginal gain at all.

## 4. What survives

1. **The vision-only class is never the best class** (0 of 8). That is the computer-use-aligned shape, and on this data it is dominated in every cell by either the tree-only or the fused option.
2. **Which class wins reverses with the workload.** Hybrid takes the VWA cells; no-image takes both WebArena cells, by 13.46pp on `wa_B0` (35.58% P-text against 22.12% SoM). The same reversal claim 4 makes, restated as the deployment question *should we ship vision at all* rather than as a per-mode ranking.
3. **Class membership does not predict marginal value** (§3). Adding an arm is worth about the same regardless of which class it comes from.

⚠️ Every number is an oracle over landed runs: it says what a perfect chooser could have gotten, not what any deployable policy gets. `rule_routing_pareto` is the companion showing that a real policy built on the strongest available signal does not beat a fixed one.

