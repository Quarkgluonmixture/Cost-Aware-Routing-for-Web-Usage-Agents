---
type: analysis
status: complete
post_hoc_exploratory: true
purpose: what a rerun flip looks like from the failing side, over every registered pair
producer: scripts/analysis/rerun_flip_failure_anatomy.py
---

# Rerun flips: how the failing run fails, and when the two runs part

Regenerate: `python scripts/analysis/rerun_flip_failure_anatomy.py`. Questions and reading thresholds were fixed in the producer docstring before computing.

24 registered same-condition pairs. Episodes excluded for a step-file / summary length mismatch or a missing file: 0 (Q3/Q4 only).

## Q1. Is the failure type stable across reruns?

Tasks failing in both runs: **3658**. Same paper bucket in both runs: **72.4%**; chance from the marginals 27.9%; **kappa = 0.62** → failure type is a property of (task, condition).

| pair | n | flip | both fail | same bucket | kappa |
|---|---|---|---|---|---|
| `B0.cls.dom` | 224 | 27 | 174 | 70% | 0.40 |
| `B0.cls.vision` | 224 | 32 | 153 | 60% | 0.36 |
| `B0.cls.som` | 224 | 29 | 146 | 79% | 0.63 |
| `B0.cls.ptext` | 224 | 23 | 179 | 74% | 0.51 |
| `B0.cls.pprompt` | 224 | 28 | 169 | 70% | 0.46 |
| `B0.cls.psom` | 224 | 27 | 177 | 74% | 0.53 |
| `B1.cls.vision` | 224 | 0 | 196 | 100% | 1.00 |
| `B1.cls.som` | 224 | 0 | 192 | 99% | 0.98 |
| `B1.cls.dom` | 224 | 7 | 206 | 83% | 0.76 |
| `B5.cls.dom` | 224 | 29 | 155 | 77% | 0.61 |
| `B0.red.ptext` | 203 | 15 | 172 | 64% | 0.49 |
| `B0.red.pprompt` | 203 | 23 | 169 | 60% | 0.40 |
| `B0.red.psom` | 203 | 21 | 167 | 68% | 0.54 |
| `B1.red.som` | 203 | 4 | 187 | 70% | 0.58 |
| `B1.red.dom` | 203 | 7 | 188 | 64% | 0.52 |
| `B0.red.som` | 203 | 17 | 167 | 61% | 0.42 |
| `B0.red.dom` | 203 | 20 | 167 | 57% | 0.34 |
| `B0.red.vision` | 203 | 10 | 183 | 68% | 0.47 |
| `B1.wared.dom` | 104 | 10 | 79 | 78% | 0.67 |
| `B1.wared.pprompt` | 104 | 9 | 82 | 71% | 0.58 |
| `B1.wared.ptext` | 104 | 6 | 82 | 72% | 0.60 |
| `B1.wared.som` | 104 | 7 | 88 | 72% | 0.59 |
| `B1.wared.psom` | 104 | 5 | 88 | 72% | 0.59 |
| `B1.wared.vision` | 104 | 2 | 92 | 63% | 0.45 |

**Post hoc strata** (added after the pooled value was seen; median of per-pair kappa):

| stratum | pairs | median kappa |
|---|---|---|
| B0 | 12 | 0.46 |
| pairs with >= 5 flips | 20 | 0.53 |
| B1 | 11 | 0.59 |
| B5 | 1 | 0.61 |

⚠️ The pooled kappa clears the 0.6 line partly because two B1·classifieds pairs have no flips and agree almost perfectly. Read the strata before quoting the pooled verdict.

## Q2. How do flips fail?

Bucket of the failing run on flipped tasks, against both-fail tasks (each run counted).

| paper bucket | flips (failing side) | both-fail |
|---|---|---|
| early-finish/wrong-commit | 204 (57%) | 3218 (44%) |
| max-steps-other | 49 (14%) | 1553 (21%) |
| visual-hijack/click-loop | 45 (13%) | 614 (8%) |
| search-loop | 39 (11%) | 1109 (15%) |
| element-misground | 17 (5%) | 755 (10%) |
| error/noise | 3 (1%) | 34 (0%) |
| missing-context | 1 (0%) | 33 (0%) |

## Q3. When do the two runs part?

First step at which (action type, url after the action, typed text) differ.

| group | tasks | already apart at step 0 | median first divergence |
|---|---|---|---|
| flip | 358 | 44% | 1.0 |
| both_success | 472 | 22% | 2.0 |
| both_fail | 3658 | 32% | 1.0 |

## Q4. How late is a flip decided?

Shared prefix as a fraction of the successful run's length, over 358 flips: median **0.04**; below 0.25: 76%; 0.75 or more: 2%.

⚠️ Paper buckets come from `analyze_reason_diagnostics.py` (a rule-based reason classifier), not from /diag rules or human reading. Action signatures ignore element ids (SoM ids are re-keyed per page), so two runs clicking different elements that lead to the same URL count as not yet diverged.
