---
type: analysis
status: complete
created: 2026-08-02
purpose: whether the two obvious routing features point the way a practitioner would assume
post_hoc_exploratory: true
scope_warning: VWA only; WebArena ships no reference images and has no visual_difficulty annotation. Stratum sizes are small in the low-success cells.
producer: scripts/analysis/aggregate_routing_feature_diagnostics.py
---

# Two routing features, and what they actually predict

Regenerate: `.venv/bin/python3 scripts/analysis/aggregate_routing_feature_diagnostics.py`

**432 of 1281 scored VWA tasks (33.7%) ship a reference image.**

## 1. `has_reference_image`: the sign is backwards

The intuition is that a task shipping a picture should be routed to a mode that can see pictures. Solve rate by mode, split on the feature:

| cell | stratum | n | dom | som | vision | ptext | pprompt | psom |
|---|---|---|---|---|---|---|---|---|
| `cls_B0` | with ref image | 65 | 33.8 | 40.0 | 30.8 | 35.4 | 36.9 | 32.3 |
| `cls_B0` | without ref image | 159 | 10.7 | 22.0 | 22.6 | 7.5 | 12.6 | 8.8 |
| `cls_B1` | with ref image | 65 | 13.8 | 15.4 | 10.8 | 16.9 | 10.8 | 15.4 |
| `cls_B1` | without ref image | 159 | 3.1 | 13.8 | 13.2 | 3.8 | 5.0 | 3.1 |
| `cls_B2` | with ref image | 65 | 0.0 | 4.6 | 3.1 | 0.0 | 1.5 | 1.5 |
| `cls_B2` | without ref image | 159 | 1.9 | 1.3 | 1.9 | 0.6 | 1.9 | 0.6 |
| `red_B0` | with ref image | 79 | 25.3 | 26.6 | 11.4 | 22.8 | 21.5 | 20.3 |
| `red_B0` | without ref image | 124 | 7.3 | 7.3 | 4.8 | 7.3 | 6.5 | 4.8 |
| `red_B1` | with ref image | 79 | 11.4 | 13.9 | 5.1 | 12.7 | 10.1 | 11.4 |
| `red_B1` | without ref image | 124 | 2.4 | 3.2 | 0.8 | 1.6 | 2.4 | 2.4 |
| `red_B2` | with ref image | 79 | 8.9 | 2.5 | 0.0 | 3.8 | 0.0 | 1.3 |
| `red_B2` | without ref image | 124 | 0.8 | 0.0 | 3.2 | 0.8 | 0.0 | 0.0 |

Arm-count matched: on top of that stratum's strongest text mode, the best gain from adding **one** image-bearing arm against adding **one** other text arm.

| cell | with a reference image | without one | intuition holds? |
|---|---|---|---|
| `cls_B0` | 13.85 vs 9.23 (+4.62) | 17.61 vs 5.66 (+11.95) | **no** |
| `cls_B1` | 4.62 vs 4.62 (+0.00) | 11.95 vs 0.63 (+11.32) | **no** |
| `cls_B2` | 4.62 vs 0.00 (+4.62) | 1.89 vs 0.63 (+1.26) | yes |
| `red_B0` | 7.59 vs 6.33 (+1.27) | 4.84 vs 2.42 (+2.42) | **no** |
| `red_B1` | 3.80 vs 5.06 (-1.27) | 1.61 vs 0.81 (+0.81) | **no** |
| `red_B2` | 1.27 vs 1.27 (+0.00) | 3.23 vs 0.81 (+2.42) | **no** |

**The intuition holds in 1 of 6 cells.** In the two largest it is reversed: the image-bearing arm is worth more on the tasks that do *not* ship a reference image.

The mechanism is in the harness, not the data. `reference_images` is passed into `BackendStepContext` with no per-mode filter, so all six modes receive the task's own picture; what the image-free modes lack is the **page** screenshot. A task that ships its own reference image is therefore precisely the task that does not need one, and the feature measures the opposite of what a router needs.

## 2. `visual_difficulty`: the right feature, read and then dropped

`extract_50_features.py` reads this VWA-native annotation out of each task config and does not place it in the feature table. Present in `feature_names_numeric`: **False**. The table carries `['dom_complexity', 'text_length', 'tokens_input_text', 'intent_token_count', 'reasoning_difficulty']`.

Best image-bearing mode minus best text-only mode, by annotated visual difficulty:

| cell | easy (n) | medium (n) | hard (n) |
|---|---|---|---|
| `cls_B0` | +1.52 (66) | +6.49 (77) | +14.81 (81) |
| `cls_B1` | +7.58 (66) | +3.90 (77) | +8.64 (81) |
| `cls_B2` | +1.52 (66) | +3.90 (77) | -1.23 (81) |
| `red_B0` | -3.23 (93) | +2.82 (71) | -5.13 (39) |
| `red_B1` | +3.23 (93) | -1.41 (71) | -2.56 (39) |
| `red_B2` | -2.15 (93) | -2.82 (71) | +0.00 (39) |
| **mean** | **+1.41** | **+2.15** | **+2.42** |

The mean gap rises monotonically with annotated visual difficulty, which is the direction a router would want, **but it is carried by classifieds and reverses on reddit**. It is a better feature than `has_reference_image` and it is not a rescue: adding it changes no cell's supply arithmetic, since the constraint there is the number of labelled rows and not their separability.

## 3. What this does and does not show

It shows that the prior a practitioner would bring to this problem is wrong-signed, which matters because few-shot learning leans on priors: a hand-written rule of the obvious form would actively hurt, and an L1-regularised model fitted on 15 to 97 rows is unlikely to recover a counterintuitive coefficient. It does **not** show that better features would fix routing. The supply argument is about row counts and is untouched by anything on this page.
