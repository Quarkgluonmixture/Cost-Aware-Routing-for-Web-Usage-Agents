---
type: analysis
status: rolling
purpose: is a policy built on the 0-token visual-intent rule dominated by a fixed one
producer: scripts/analysis/rule_routing_pareto.py
---

# Rule routing on the (success, cost, latency) frontier

Regenerate: `.venv/bin/python3 scripts/analysis/rule_routing_pareto.py`

`visual_intent_routing` showed **where** the screenshot pays. This asks whether **routing on that** beats not routing. A router earns its keep only if no signal-free fixed policy dominates it — no worse on all three axes, strictly better on one.

The partition is a regex over the task intent: nothing is learned, so there is no train/test split and no in-sample optimism. Cost/latency are per-attempt cell means and are **within-cell comparable only**.

## `cls_B0` — flagged 71/224

| policy | SR | cost | latency (canonical) | on frontier? | dominated by |
|---|---|---|---|---|---|
| always-SoM | 27.23% | 0.07236 | 106.0s | **yes** | — |
| always-Vision | 25.00% | 0.06481 | 125.2s | **yes** | — |
| rule: flag→Vision else DOM ⭐ | 24.55% | 0.06809 | 117.6s | **yes** | — |
| rule: flag→SoM else P-text ⭐ | 24.11% | 0.07019 | 115.8s | **yes** | — |
| rule: flag→SoM else DOM ⭐ | 23.66% | 0.07049 | 111.5s | **yes** | — |
| always-P-prompt | 19.64% | 0.06853 | 107.8s | **yes** | — |
| always-DOM | 17.41% | 0.06962 | 114.1s | no | `always-P-prompt` |
| always-P-text | 15.62% | 0.06919 | 120.4s | no | `always-P-prompt`, `rule: flag→Vision else DOM` |
| always-P-SoM | 15.62% | 0.07206 | 117.9s | no | `always-DOM`, `always-P-prompt`, `rule: flag→Vision else DOM`, `rule: flag→SoM else DOM`, `rule: flag→SoM else P-text` |

## `cls_B1` — flagged 71/224

| policy | SR | cost | latency (canonical) | on frontier? | dominated by |
|---|---|---|---|---|---|
| always-SoM | 14.29% | 0.06028 | 262.0s | **yes** | — |
| always-Vision | 12.50% | 0.04316 | 269.8s | **yes** | — |
| rule: flag→Vision else DOM ⭐ | 11.61% | 0.05433 | 296.5s | no | `always-Vision` |
| rule: flag→SoM else P-text ⭐ | 11.16% | 0.05926 | 297.2s | no | `always-Vision`, `rule: flag→Vision else DOM` |
| rule: flag→SoM else DOM ⭐ | 10.27% | 0.05976 | 294.0s | no | `always-Vision` |
| always-P-text | 7.59% | 0.05879 | 313.5s | no | `always-Vision`, `rule: flag→Vision else DOM` |
| always-P-SoM | 6.70% | 0.05970 | 311.7s | no | `always-Vision`, `rule: flag→Vision else DOM`, `rule: flag→SoM else P-text` |
| always-P-prompt | 6.70% | 0.06304 | 301.4s | no | `always-SoM`, `always-Vision`, `rule: flag→Vision else DOM`, `rule: flag→SoM else DOM`, `rule: flag→SoM else P-text` |
| always-DOM | 6.25% | 0.05951 | 308.9s | no | `always-Vision`, `rule: flag→Vision else DOM`, `rule: flag→SoM else P-text` |

## `cls_B2` — flagged 71/224

| policy | SR | cost | latency (canonical) | on frontier? | dominated by |
|---|---|---|---|---|---|
| always-Vision | 2.23% | 0.07065 | 417.8s | **yes** | — |
| always-SoM | 2.23% | 0.09075 | 374.3s | **yes** | — |
| rule: flag→Vision else DOM ⭐ | 1.79% | 0.07483 | 407.2s | **yes** | — |
| always-P-prompt | 1.79% | 0.08453 | 396.4s | **yes** | — |
| always-DOM | 1.34% | 0.07676 | 402.3s | **yes** | — |
| rule: flag→SoM else P-text ⭐ | 1.34% | 0.07876 | 391.5s | **yes** | — |
| rule: flag→SoM else DOM ⭐ | 1.34% | 0.08120 | 393.4s | no | `rule: flag→SoM else P-text` |
| always-P-SoM | 0.89% | 0.08456 | 411.0s | no | `always-DOM`, `always-P-prompt`, `rule: flag→Vision else DOM`, `rule: flag→SoM else DOM`, `rule: flag→SoM else P-text` |
| always-P-text | 0.45% | 0.07320 | 399.4s | **yes** | — |

## `red_B0` — flagged 63/203

| policy | SR | cost | latency (canonical) | on frontier? | dominated by |
|---|---|---|---|---|---|
| always-SoM | 14.78% | 0.11045 | 451.6s | **yes** | — |
| always-DOM | 14.29% | 0.10147 | 552.5s | **yes** | — |
| rule: flag→SoM else DOM ⭐ | 14.29% | 0.10426 | 521.2s | **yes** | — |
| rule: flag→Vision else DOM ⭐ | 13.30% | 0.10041 | 510.9s | **yes** | — |
| always-P-text | 13.30% | 0.10577 | 562.1s | no | `always-DOM`, `rule: flag→Vision else DOM`, `rule: flag→SoM else DOM` |
| rule: flag→SoM else P-text ⭐ | 13.30% | 0.10722 | 527.8s | no | `rule: flag→Vision else DOM`, `rule: flag→SoM else DOM` |
| always-P-prompt | 12.32% | 0.10163 | 447.7s | **yes** | — |
| always-P-SoM | 10.84% | 0.10814 | 532.0s | no | `always-P-prompt`, `rule: flag→Vision else DOM`, `rule: flag→SoM else DOM`, `rule: flag→SoM else P-text` |
| always-Vision | 7.39% | 0.09807 | 418.5s | **yes** | — |

## `red_B1` — flagged 63/203

| policy | SR | cost | latency (canonical) | on frontier? | dominated by |
|---|---|---|---|---|---|
| always-SoM | 7.39% | 0.08000 | 609.1s | **yes** | — |
| rule: flag→SoM else P-text ⭐ | 6.90% | 0.07275 | 601.8s | **yes** | — |
| rule: flag→SoM else DOM ⭐ | 6.40% | 0.07538 | 604.6s | no | `rule: flag→SoM else P-text` |
| always-P-text | 5.91% | 0.06948 | 598.6s | **yes** | — |
| always-DOM | 5.91% | 0.07330 | 602.6s | no | `always-P-text`, `rule: flag→SoM else P-text` |
| always-P-SoM | 5.91% | 0.07480 | 616.6s | no | `always-DOM`, `always-P-text`, `rule: flag→SoM else P-text` |
| always-P-prompt | 5.42% | 0.07656 | 614.0s | no | `always-DOM`, `always-P-text`, `rule: flag→SoM else DOM`, `rule: flag→SoM else P-text` |
| rule: flag→Vision else DOM ⭐ | 4.93% | 0.06682 | 557.3s | **yes** | — |
| always-Vision | 2.46% | 0.05240 | 456.6s | **yes** | — |

## `red_B2` — flagged 63/203

| policy | SR | cost | latency (canonical) | on frontier? | dominated by |
|---|---|---|---|---|---|
| rule: flag→Vision else DOM ⭐ | 5.42% | 0.08658 | 632.7s | **yes** | — |
| always-DOM | 3.94% | 0.09479 | 669.9s | no | `rule: flag→Vision else DOM` |
| rule: flag→SoM else DOM ⭐ | 3.45% | 0.10001 | 655.4s | no | `rule: flag→Vision else DOM` |
| always-Vision | 1.97% | 0.06833 | 550.0s | **yes** | — |
| always-P-text | 1.97% | 0.08852 | 677.6s | no | `always-Vision`, `rule: flag→Vision else DOM` |
| rule: flag→SoM else P-text ⭐ | 1.48% | 0.09568 | 660.7s | no | `always-Vision`, `rule: flag→Vision else DOM` |
| always-SoM | 0.99% | 0.11160 | 623.0s | no | `always-Vision` |
| always-P-SoM | 0.49% | 0.09451 | 640.0s | no | `always-Vision`, `rule: flag→Vision else DOM` |
| always-P-prompt | 0.00% | 0.09940 | 599.3s | no | `always-Vision` |

## Verdict

**A rule policy survives on the frontier in 5 of 6 cells**: `cls_B0` (3 of 3); `cls_B2` (2 of 3); `red_B0` (2 of 3); `red_B1` (2 of 3); `red_B2` (1 of 3).

Surviving the frontier is a low bar: it means *nothing dominates*, not that the policy is preferable. Read it as "routing is not ruled out here" rather than "routing wins here". The cells where it is dominated are the informative ones — there, the signal is real (see `visual_intent_routing`) and routing on it still buys nothing, because the arm the rule sends work *to* is already the right arm to send everything to.

