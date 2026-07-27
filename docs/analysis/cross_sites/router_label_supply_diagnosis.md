# Why the which-mode router cannot be learned — label-supply diagnosis

> `post_hoc_exploratory=True` · `h10_eligible=False` — diagnosis of a negative result, never a gate.

**Claim.** The which-mode router is unlearnable because labels are produced only by solve events; the bottleneck is production RATE, not the hypothesis class and not the label definition.

## 1. Supply: labels exist only where something succeeded

| cell | scored universe | trainable labels | solvable | classes present |
|---|---|---|---|---|
| B0_classifieds | 224 | **97** | 43.3% | 6/6 |
| B0_reddit | 203 | **53** | 26.11% | 6/6 |
| B1_classifieds | 224 | **55** | 24.55% | 6/6 |
| B1_reddit | 203 | **24** | 11.82% | 6/6 |
| B2_classifieds | 224 | **16** | 7.14% | 5/6 |
| B2_reddit | 203 | **15** | 7.39% | 5/6 |

Across 6 cells the trainable-label count spans **15-97**; pooled total **260**. A label exists only where some mode succeeded. The spread is the whole argument: no re-slicing of the supervision changes how many solve events the benchmark produced.

## 2. Trainability: the min-class filter empties most cells

| cell | labels | classes | surviving min-class filter | trainable |
|---|---|---|---|---|
| B0_classifieds | 97 | 6 | 3 (dom, phantom_prompt, som) | yes |
| B0_reddit | 53 | 6 | 1 (dom) | **no** |
| B1_classifieds | 55 | 6 | 2 (dom, som) | yes |
| B1_reddit | 24 | 6 | 0 (—) | **no** |
| B2_classifieds | 16 | 5 | 0 (—) | **no** |
| B2_reddit | 15 | 5 | 0 (—) | **no** |

**4 of 6 cells have no trainable classifier at all.** A cell needs >=2 classes each surviving the N_MIN_CLASS_TRAIN=10 filter in a 5-fold split. Fewer than two leaves nothing to discriminate.

## 3. Pooling fixes supply and breaks identifiability

| site | tasks shared by 2+ cells | conflicting | conflict rate | Bayes ceiling (which-mode) | Bayes ceiling (cost tier) | tier agreement |
|---|---|---|---|---|---|---|
| classifieds | 54 | 31 | **57.41%** | 79.17% | **89.88%** | 68.52% |
| reddit | 25 | 14 | **56.0%** | 83.7% | **96.74%** | 88.0% |

conflict_rate is over tasks covered by >=2 cells. The two ceilings are the point: re-slicing the SAME features from 'which of six modes' down to 'image or text-only' raises the attainable ceiling without inventing a single new solve event — the only relabelling that buys anything.

## 4. How much of the supervision is a list literal

| cell | labels | multi-success | true cost tie (order decides) | order picked a pricier mode |
|---|---|---|---|---|
| B0_classifieds | 97 | 68 (70.1%) | **0 (0.0%)** | 53 (54.64%) |
| B0_reddit | 53 | 36 (67.92%) | **0 (0.0%)** | 23 (43.4%) |
| B1_classifieds | 55 | 29 (52.73%) | **0 (0.0%)** | 26 (47.27%) |
| B1_reddit | 24 | 17 (70.83%) | **0 (0.0%)** | 9 (37.5%) |
| B2_classifieds | 16 | 4 (25.0%) | **0 (0.0%)** | 2 (12.5%) |
| B2_reddit | 15 | 3 (20.0%) | **0 (0.0%)** | 2 (13.33%) |

`MODES = ['dom', 'phantom_som', 'phantom_text', 'phantom_prompt', 'som', 'vision']`

Three nested quantities, do not conflate them. `multi_success_pct` = the order was consulted at all (loose upper bound, NOT arbitrariness). `true_tie_pct` = several successful modes cost EXACTLY the same, so the MODES list literal alone picks the label — this is the literal tie-break arbitrariness 笔记 §383.4 reported at ~1/4. `order_wrong_pct` = no tie, but the order picked a strictly more expensive mode, i.e. MODES is not in ascending MEASURED cost here — a separate and worse defect than a tie-break, since the label is then not even 'cheapest successful'.

**The tie-break never fires.** `true_tie` is 0 in every cell, because `total_billed_cost_usd` is a continuous float and two modes costing *exactly* the same does not happen. So 'tie-break arbitrariness' is the wrong frame for this defect. What does happen is worse: on **12.5-54.64%** of labels the MODES order returns a mode that is strictly MORE expensive than another successful one. `MODES` is documented as "ascending prior cost" and used as a proxy for cheapest-successful; on those rows the assumption is false, so the label is not 'cheapest successful mode' at all — it is whatever the list literal reached first.

笔记 §383.4 reported ~1/4 of labels as order-decided (B0_cls 25/97) from a scratch script that no longer exists; that figure is not reproduced here and is not the same quantity — it is superseded by the two measured columns above rather than reconciled to.

