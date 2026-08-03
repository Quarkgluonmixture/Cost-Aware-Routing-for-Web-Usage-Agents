---
type: analysis
status: complete
purpose: how much of a per-mode gap is carried by the scaffold's click delivery path
post_hoc_exploratory: true
scope_warning: this corrects no success rate. It identifies a mediator that a different harness would implement differently, i.e. an external-validity limit.
producer: scripts/analysis/dispatch_path_audit.py
---

# What actually delivered the click

Regenerate: `.venv/bin/python3 scripts/analysis/dispatch_path_audit.py`

`action_executed.dispatch_path` says how each action reached the browser. It is on 70% of steps and had one narrow consumer before 2026-08-03.

## 1. Three paths, five-fold different action success

| delivery path | actions | action success |
|---|---|---|
| element-id → locator | 8,857 | **88.9%** |
| other | 1,530 | **65.9%** |
| coordinate click | 2,138 | **38.6%** |
| element-id → framework fallback | 4,564 | **16.1%** |

An action that does not succeed still consumes a step from the budget, so a mode spending more of its actions on a weak path is spending its step budget at a discount — which is a mechanism by which representation and outcome are connected that has nothing to do with what the model saw.

## 2. The mix is not constant

| cell · mode | actions | locator | fallback | coordinate | action success |
|---|---|---|---|---|---|
| `B0·classifieds·DOM` | 266 | 77% | 22% | — | 77% |
| `B0·classifieds·SoM` | 199 | 93% | 6% | — | 83% |
| `B0·classifieds·Vision` | 265 | — | — | 100% | 60% |
| `B0·classifieds·P-text` | 255 | 96% | 3% | — | 95% |
| `B0·classifieds·P-prompt` | 260 | 83% | 16% | — | 87% |
| `B0·classifieds·P-SoM` | 269 | 92% | 7% | — | 89% |
| `B0·reddit·DOM` | 403 | 75% | 23% | — | 81% |
| `B0·reddit·SoM` | 408 | 85% | 15% | — | 85% |
| `B0·reddit·Vision` | 254 | — | — | 97% | 76% |
| `B0·reddit·P-text` | 441 | 84% | 11% | — | 84% |
| `B0·reddit·P-prompt` | 375 | 87% | 9% | — | 90% |
| `B0·reddit·P-SoM` | 433 | 90% | 9% | — | 88% |
| `B1·classifieds·DOM` | 417 | 47% | 52% | — | 55% |
| `B1·classifieds·SoM` | 306 | 60% | 36% | 1% | 61% |
| `B1·classifieds·Vision` | 268 | — | — | 100% | 16% |
| `B1·classifieds·P-text` | 356 | 58% | 38% | — | 67% |
| `B1·classifieds·P-prompt` | 317 | 60% | 38% | 1% | 56% |
| `B1·classifieds·P-SoM` | 356 | 67% | 32% | 1% | 64% |
| `B1·reddit·DOM` | 570 | 75% | 22% | — | 80% |
| `B1·reddit·SoM` | 615 | 58% | 41% | 0% | 58% |
| `B1·reddit·Vision` | 377 | — | — | 99% | 50% |
| `B1·reddit·P-text` | 550 | 73% | 16% | — | 83% |
| `B1·reddit·P-prompt` | 543 | 62% | 31% | 2% | 68% |
| `B1·reddit·P-SoM` | 608 | 59% | 39% | 1% | 61% |
| `B2·classifieds·DOM` | 685 | 38% | 49% | — | 36% |
| `B2·classifieds·SoM` | 642 | 55% | 31% | 1% | 38% |
| `B2·classifieds·Vision` | 480 | — | — | 80% | 26% |
| `B2·classifieds·P-text` | 534 | 36% | 40% | — | 49% |
| `B2·classifieds·P-prompt` | 642 | 35% | 40% | 3% | 36% |
| `B2·classifieds·P-SoM` | 718 | 33% | 39% | 6% | 42% |
| `B2·reddit·DOM` | 789 | 35% | 54% | — | 53% |
| `B2·reddit·SoM` | 804 | 63% | 24% | 1% | 67% |
| `B2·reddit·Vision` | 563 | — | — | 81% | 34% |
| `B2·reddit·P-text` | 748 | 61% | 20% | — | 79% |
| `B2·reddit·P-prompt` | 726 | 31% | 53% | 3% | 43% |
| `B2·reddit·P-SoM` | 647 | 62% | 21% | 3% | 76% |

Two structural facts in that table:

1. **`Vision` is on the coordinate path by construction.** It emits no element ids, so it cannot use the path that succeeds 89% of the time. Its action success is capped by whatever coordinate clicking achieves in this harness (39% overall). **This is not a confound to remove — it is part of what screenshot-only *is*** — but it does mean the Vision arm measures this scaffold's coordinate implementation as much as it measures the representation, and a harness with better grounding would report a different Vision.
2. **The fallback share rises as the backbone weakens**: mean B0 12% · B1 35% · B2 37% on the text arms. Falling back is downstream of the model — it emitted an id the locator could not resolve — so that part is a legitimate capability difference. What is **not** a capability difference is the 16% success of the fallback itself: that is this harness's fallback, and a better one would narrow every backbone gap that runs through it.

## 3. What this licenses

Nothing here changes a success rate, and no number elsewhere should be adjusted by it. What it establishes is that **the per-mode and per-backbone gaps reported in this project are partly mediated by two implementation choices** — how a coordinate click is issued, and what happens when an element id fails to resolve. Both are properties of this harness. A paper claiming a representation effect has to say so, because a reader's first alternative explanation for "screenshot-only does worst" is "their coordinate clicking is bad", and on this evidence that explanation is *partly correct*.

## 4. Raw dispatch names

Grouped above; listed here so the grouping can be checked rather than trusted.

| dispatch_path | actions |
|---|---|
| `element_id_locator_route` | 8,857 |
| `element_id_framework` | 4,564 |
| `coord_mouse_click` | 1,784 |
| `whitelisted` | 1,020 |
| `seq_unresolved_noop` | 222 |
| `coord_keyboard_fallback` | 170 |
| `coord_locator_route` | 169 |
| `offsite_blocked` | 136 |
| `relative` | 126 |
| `element_id` | 13 |
| `coord_true_oob_noop` | 13 |
| `id_based_escape_hatch` | 10 |
| `noop_serialize_fail` | 3 |
| `coord_mouse_hover` | 2 |
