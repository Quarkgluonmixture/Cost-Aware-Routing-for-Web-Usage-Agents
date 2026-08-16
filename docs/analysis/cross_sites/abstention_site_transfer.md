---
type: analysis
status: complete
purpose: does an abstention policy fitted on one site work on a site it has never seen
scope_warning: 6 VWA cells, 2 sites. The abstention label is N=1 per mode. AUROC is rank-based and blind to calibration drift; the operating-point table is not, and they disagree by construction. Read both.
producer: scripts/analysis/abstention_site_transfer.py
generated: 2026-08-16T16:20:49+00:00
---

# Abstention across sites: does the policy transfer?

Regenerate: `.venv/bin/python3 scripts/analysis/abstention_site_transfer.py`

`abstention_learnability` answers the *within-cell* question with task-level 5-fold. Every fold there shares the site's task distribution, DOM idiom and evaluator, so it cannot speak to generalisation. Here the feature set, label function and estimator are unchanged and only the split moves: **train on one site, test on the other**.

## 1. Matched-model transfer (model fixed, site swapped)

| train | test | n train | n test | base rate train | base rate test | **transfer AUROC** | shuffle null | within-cell AUROC (ceiling) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `B0_classifieds` | `B0_reddit` | 224 | 203 | 56.7% | 73.9% | **0.717** | 0.454 | 0.780 |
| `B0_reddit` | `B0_classifieds` | 203 | 224 | 73.9% | 56.7% | **0.677** | 0.487 | 0.726 |
| `B1_classifieds` | `B1_reddit` | 224 | 203 | 75.4% | 88.2% | **0.718** | 0.354 | 0.864 |
| `B1_reddit` | `B1_classifieds` | 203 | 224 | 88.2% | 75.4% | **0.738** | 0.484 | 0.732 |
| `B2_classifieds` | `B2_reddit` | 224 | 203 | 92.9% | 92.6% | **0.732** | 0.749 | 0.615 |
| `B2_reddit` | `B2_classifieds` | 203 | 224 | 92.6% | 92.9% | **0.627** | 0.529 | 0.642 |

## 2. Pooled transfer (all three models of one site → each cell of the other)

| train site | test | n train | n test | **transfer AUROC** | shuffle null | within-cell AUROC (ceiling) |
|---|---|---:|---:|---:|---:|---:|
| classifieds (3 cells) | `B0_reddit` | 672 | 203 | **0.770** | 0.494 | 0.780 |
| classifieds (3 cells) | `B1_reddit` | 672 | 203 | **0.723** | 0.439 | 0.864 |
| classifieds (3 cells) | `B2_reddit` | 672 | 203 | **0.745** | 0.433 | 0.615 |
| reddit (3 cells) | `B0_classifieds` | 609 | 224 | **0.711** | 0.508 | 0.726 |
| reddit (3 cells) | `B1_classifieds` | 609 | 224 | **0.711** | 0.514 | 0.732 |
| reddit (3 cells) | `B2_classifieds` | 609 | 224 | **0.638** | 0.654 | 0.642 |

## 3. Transferred operating point — threshold never sees the test site

The threshold is chosen by an inner 5-fold **inside the training site** at the stated solvable-loss budget, then applied unseen to the other site. This is the cross-site analogue of the nested column in `abstention_learnability` §3, and it is a held-out *policy*, not merely a held-out prediction (§465).

| train | test | budget | abstain rate | solvable lost | realised loss | saved |
|---|---|---:|---:|---:|---:|---:|
| `B0_classifieds` | `B0_reddit` | ≤0% | 3.9% | 0/53 | 0.0% | 5.8% |
| `B0_classifieds` | `B0_reddit` | ≤2% | 15.3% | 1/53 | 1.9% | 19.1% |
| `B0_classifieds` | `B0_reddit` | ≤5% | 20.7% | 2/53 | 3.8% | 25.2% |
| `B0_classifieds` | `B0_reddit` | ≤10% | 30.0% | 6/53 | 11.3% | 37.4% |
| `B0_reddit` | `B0_classifieds` | ≤0% | 3.1% | 0/97 | 0.0% | 3.3% |
| `B0_reddit` | `B0_classifieds` | ≤2% | 5.8% | 0/97 | 0.0% | 6.6% |
| `B0_reddit` | `B0_classifieds` | ≤5% | 7.6% | 3/97 | 3.1% | 8.7% |
| `B0_reddit` | `B0_classifieds` | ≤10% | 12.9% | 7/97 | 7.2% | 15.6% |
| `B1_classifieds` | `B1_reddit` | ≤0% | 11.3% | 0/24 | 0.0% | 11.0% |
| `B1_classifieds` | `B1_reddit` | ≤2% | 17.2% | 0/24 | 0.0% | 16.8% |
| `B1_classifieds` | `B1_reddit` | ≤5% | 22.7% | 0/24 | 0.0% | 19.9% |
| `B1_classifieds` | `B1_reddit` | ≤10% | 34.5% | 2/24 | 8.3% | 33.8% |
| `B1_reddit` | `B1_classifieds` | ≤0% | 58.9% | 20/55 | 36.4% | 65.6% |
| `B1_reddit` | `B1_classifieds` | ≤2% | 58.9% | 20/55 | 36.4% | 65.6% |
| `B1_reddit` | `B1_classifieds` | ≤5% | 70.5% | 24/55 | 43.6% | 77.5% |
| `B1_reddit` | `B1_classifieds` | ≤10% | 70.5% | 24/55 | 43.6% | 77.5% |
| `B2_classifieds` | `B2_reddit` | ≤0% | 14.3% | 1/15 | 6.7% | 11.3% |
| `B2_classifieds` | `B2_reddit` | ≤2% | 14.3% | 1/15 | 6.7% | 11.3% |
| `B2_classifieds` | `B2_reddit` | ≤5% | 14.3% | 1/15 | 6.7% | 11.3% |
| `B2_classifieds` | `B2_reddit` | ≤10% | 21.2% | 1/15 | 6.7% | 17.7% |
| `B2_reddit` | `B2_classifieds` | ≤0% | 1.3% | 0/16 | 0.0% | 1.7% |
| `B2_reddit` | `B2_classifieds` | ≤2% | 1.3% | 0/16 | 0.0% | 1.7% |
| `B2_reddit` | `B2_classifieds` | ≤5% | 1.3% | 0/16 | 0.0% | 1.7% |
| `B2_reddit` | `B2_classifieds` | ≤10% | 16.1% | 2/16 | 12.5% | 17.7% |

⚠️ **A budget met at home is not a budget met abroad.** The budget column is what the threshold bought on the training site; the realised-loss column is what it actually cost on the test site. Where the two diverge, the cause is base-rate drift (56.7%–92.9% across these cells), and it is exactly the failure mode within-cell CV cannot show.

⚠️ Cost coverage is the fraction of test-cell tasks with a cost figure in `per_task_sr.csv`; that product is on the leak-kept, n=205 reddit convention while the labels here come from the canonical n=203 universe (§462.1). The join is by task id, so surplus rows are simply unused, but the saved-dollar column inherits the cost side's convention.

## 4. What transfers, and what does not

**Ranking transfers.** 5 of 6 pooled transfers clear their own label-shuffle null. Against the ceiling — the same cell's *within-cell* held-out AUROC — the pooled transfer lands between -0.141 (`B1_reddit`) and +0.130 (`B2_reddit`). It is **positive** in 1 of 6 cells, i.e. training on the other site beat training on the cell's own tasks there.

**Where it does not transfer:** `B2_classifieds` — the shuffle null matches or beats the fitted score, so nothing was learned that survives the site change there. These are the cells the within-cell product already flags as sitting near the floor.

**The operating point does not, and the AUROC does not warn you.** 15 of 24 transferred thresholds kept the realised loss inside the budget they were bought at. The worst is `B1_reddit` → `B1_classifieds` at a ≤5% budget: it abstains on 70.5% of the test site and destroys 24/55 (43.6%) of its solvable tasks. That same direction has AUROC **0.738**, the highest of the 6 matched transfers and above its own within-cell ceiling of 0.732 — i.e. **the direction that looks best by ranking is the one that fails worst as a policy**.

The mechanism is base-rate drift, and it is one-directional: universal-fail runs 56.7%–92.9% across these cells, so a threshold calibrated where almost everything fails abstains on far too much where less does. AUROC is rank-based and cannot see it. This is the cross-site form of the distinction §465 drew within a cell: **a held-out prediction is not a held-out policy**, and transfer is where the gap between them is widest.

⚠️ Two sites is two points. Nothing here licenses a claim about transfer to a *third* site; what it licenses is that ranking survived the one site change available and calibration did not.
