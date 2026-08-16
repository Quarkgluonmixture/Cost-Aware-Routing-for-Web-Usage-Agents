---
type: analysis
status: complete
purpose: whether 'should this task be attempted at all' is learnable where 'which representation' is not, and what abstaining would have saved
scope_warning: 6 VWA cells. The abstention label is N=1 per mode -- a task no mode solved on one draw is not proven unsolvable, and same-condition rerun discordance on these cells is 12-14%, so some labels flip. Savings are an accounting identity on the observed matrix, not a forecast.
producer: scripts/analysis/abstention_learnability.py
---

# Abstention: the routing question whose labels the benchmark does supply

Regenerate: `.venv/bin/python3 scripts/analysis/abstention_learnability.py`

## 1. Label supply, inverted

§6's which-mode label needs a task **some mode solved**, so supervision is produced at the success rate. The abstention label -- did *any* of the six modes solve it -- exists on every task, and the class that starves the which-mode router is this one's majority class.

| cell | n | universal-fail | solvable | abstention rows | which-mode rows | ratio |
|---|---:|---:|---:|---:|---:|---:|
| `B0_classifieds` | 224 | 127 (56.7%) | 97 | **224** | 97 | 2.3x |
| `B0_reddit` | 203 | 150 (73.9%) | 53 | **203** | 53 | 3.8x |
| `B1_classifieds` | 224 | 169 (75.4%) | 55 | **224** | 55 | 4.1x |
| `B1_reddit` | 203 | 179 (88.2%) | 24 | **203** | 24 | 8.5x |
| `B2_classifieds` | 224 | 208 (92.9%) | 16 | **224** | 16 | 14.0x |
| `B2_reddit` | 203 | 188 (92.6%) | 15 | **203** | 15 | 13.5x |

Under `min_class_n=10` the which-mode router admits no classifier in four of these six cells. The abstention label clears it in **all six** -- the smallest class is the solvable side of `B2`, and it is still above the floor.

## 2. Held-out learnability

Task-level 5-fold, folds from `router_pooled_tier_learnability.outer_fold_map` (SEED=42), fold-local standardisation + L2 LR -- the same CV and the same estimator §6 uses, so the two routers' evidence is comparable. The null is §6's own control: identical features and folds, training labels permuted.

| cell | AUROC (held-out) | label-shuffle null | gap |
|---|---:|---:|---:|
| `B0_classifieds` | **0.726** | 0.347 | +0.379 |
| `B0_reddit` | **0.780** | 0.476 | +0.304 |
| `B1_classifieds` | **0.732** | 0.478 | +0.254 |
| `B1_reddit` | **0.864** | 0.477 | +0.387 |
| `B2_classifieds` | **0.642** | 0.604 | +0.038 |
| `B2_reddit` | **0.615** | 0.462 | +0.153 |

Four of six cells carry a gap of +0.254 or more; the two `B2` cells do not (+0.038 and +0.153), which is consistent with the draft's own note that the B2 cells sit near the floor. Unlike §6's which-mode router, no cell lands **below** chance.

## 3. What abstaining would have saved

The denominator is the money §5's surviving bound still pays: that bound routes never-solved tasks to the cheapest arm, so the cost still incurred is the cheapest arm's, `min(cost_dom, cost_psom)` per task. Abstention does not pay it at all.

| cell | total | oracle (abstain every universal-fail) | **nested** 0 loss | ≤2% budget | ≤5% | ≤10% |
|---|---:|---:|---:|---:|---:|---:|
| `B0_classifieds` | $12.71 | **63.8%** | 2.3% (−0) | 11.8% (−3) | 15.1% (−3) | 26.5% (−9) |
| `B0_reddit` | $17.33 | **78.5%** | 6.2% (−1) | 6.2% (−1) | 22.8% (−1) | 36.8% (−5) |
| `B1_classifieds` | $10.78 | **79.3%** | 9.6% (−1) | 9.6% (−1) | 14.4% (−2) | 28.7% (−5) |
| `B1_reddit` | $12.52 | **90.3%** | 19.8% (−0) | 19.8% (−0) | 24.6% (−0) | 30.1% (−1) |
| `B2_classifieds` | $15.74 | **93.6%** | 6.0% (−1) | 6.0% (−1) | 6.0% (−1) | 12.1% (−1) |
| `B2_reddit` | $16.54 | **92.3%** | 14.5% (−4) | 14.5% (−4) | 14.5% (−4) | 22.5% (−4) |

Two readings matter and they differ by an order of magnitude.

**The oracle is huge and irrelevant.** Abstaining from every universal-fail task would cut 63.8-93.6% of the bill at zero success cost -- but that needs the outcome in advance, exactly the objection §5 raises against its own success-rate ceiling.

⚠️ **The columns above are NESTED**: the abstention threshold is chosen by an inner 5-fold on each outer training split and then applied unseen to the test split. An earlier version of this artifact swept the threshold over the out-of-fold predictions themselves and kept the best one, which selects the operating point with the test labels — held-out *prediction* is not a held-out *policy*. Those optimistic numbers survive in the JSON under `frontier_by_loss_allowance_ORACLE_SELECTED` for contrast only.

**The held-out policy is modest at zero loss and useful just past it.** Insisting on losing no solvable task confines the policy to its most confident handful (2.3-19.8% of the bill). A 5% loss allowance reaches 6.0-24.6%, which overlaps but does not clear the **≤30.6% (7 of 8 cells)** band §5 quotes as an *oracle* — so the honest reading is that a held-out policy buys a comparable order of saving, not that it beats the oracle.

⚠️ A nested threshold meets its budget on the inner folds, not necessarily on the outer test split: 2 of 6 cells overrun the 5% budget (`B2_classifieds`, `B2_reddit`). The realised loss is in the table's parenthesised counts; the column header is the budget bought, not the loss paid.

⚠️ **This is pre-flight but not free.** The features are step-0 observation statistics plus task-config text, so a decision needs the first page loaded and its accessibility tree built -- but **no model call**. What is saved is the API spend; what is paid is one page load. That asymmetry is why the numbers above are worth quoting, and it must be stated whenever they are.

⚠️ The label is N=1 per mode. Rerun discordance on these cells is 12-14%, so a fraction of the universal-fail set would flip on a second draw; a rerun-calibrated version needs the replicate arms and is not attempted here.
