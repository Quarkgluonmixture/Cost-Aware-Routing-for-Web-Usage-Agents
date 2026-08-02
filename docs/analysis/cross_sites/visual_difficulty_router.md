---
type: analysis
status: complete
created: 2026-08-02
purpose: fit the feature the diagnosis said was missing, and report whether it rescues the router
post_hoc_exploratory: true
scope_warning: VWA only — WebArena ships no visual_difficulty annotation. The label is the triage label, not the which-mode label; a feature can help one and not the other.
producer: scripts/analysis/aggregate_visual_difficulty_router.py
---

# Does `visual_difficulty` rescue the router?

Regenerate: `.venv/bin/python3 scripts/analysis/aggregate_visual_difficulty_router.py`

`routing_feature_diagnostics` showed the feature in the table has the wrong sign and that the VWA-native annotation a practitioner would want is read out of every task config and then dropped before the table is built. That was a diagnosis. EVIDENCE_LAYER_SUMMARY §6 listed the corresponding test as open, on the grounds that reporting a fitted failure beats arguing one. Here it is fitted.

Same triage label and same out-of-fold logistic regression as `router_triage_learnability`, same rows, same folds — one extra column.

| cell | n | positives | AUROC without | AUROC with | Δ |
|---|---|---|---|---|---|
| `cls_B0` | 224 | 97 | 0.6759 | 0.6695 | -0.0064 |
| `red_B0` | 203 | 53 | 0.6655 | 0.6670 | +0.0014 |
| `cls_B1` | 224 | 55 | 0.7170 | 0.7031 | -0.0139 |
| `red_B1` | 203 | 24 | 0.6849 | 0.7126 | +0.0277 |
| `cls_B2` | 224 | 16 | 0.6505 | 0.6424 | -0.0081 |
| `red_B2` | 203 | 15 | 0.4828 | 0.5289 | +0.0461 |

**Mean ΔAUROC = +0.0078** over 6 cells; it improves 3 of them.

That is inside the noise of a fold split on cells this size. **The feature does not rescue the router**, and the reason is the one the supply argument already gives: the constraint is the number of usable labelled rows, not their separability. A better feature cannot manufacture labels.

### What this does and does not settle

It closes the specific objection that the authors diagnosed a missing feature and never tried it. It does **not** show that no feature would help: the space of features is not enumerable, and this is one annotation on one benchmark. What makes the negative durable is not this fit — it is that the binding constraint is row count, which no feature changes.

⚠️ The label here is *triage* (solvable by anything), not *which mode*. A feature could in principle help the which-mode decision and not this one; that label is the one `router_label_supply_diagnosis` shows there are too few rows to fit at all.
