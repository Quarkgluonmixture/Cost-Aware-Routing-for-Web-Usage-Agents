---
type: analysis
status: complete
purpose: does an abstention policy fitted on one site work on a site it has never seen
scope_warning: 6 VWA cells, 2 sites. The abstention label is N=1 per mode. AUROC is rank-based and blind to calibration drift; the operating-point table is not, and they disagree by construction. Read both.
producer: scripts/analysis/abstention_site_transfer.py
generated: 2026-08-16T17:46:40+00:00
---

# Abstention across sites: does the policy transfer?

Regenerate: `.venv/bin/python3 scripts/analysis/abstention_site_transfer.py`

`abstention_learnability` answers the *within-cell* question with task-level 5-fold. Every fold there shares the site's task distribution, DOM idiom and evaluator, so it cannot speak to generalisation. Here the feature set, label function and estimator are unchanged and only the split moves: **train on one site, test on the other**.

## 1. Matched-model transfer (model fixed, site swapped)

The null column is the 95th percentile of **200 label permutations**, not a single draw, and the permutation p is plus-one corrected. A one-draw null cannot support a verdict: the AUROC null's own SD on the sparsest cells here is ~0.08, which is larger than several of the gaps being judged.

| train | test | n test | base rate train → test | **transfer AUROC** | null p95 | null SD (emp / analytic) | perm p | within-cell (ceiling) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `B0_classifieds` | `B0_reddit` | 203 | 56.7% → 73.9% | **0.717** | 0.637 | 0.082 / 0.046 | 0.005 | 0.780 |
| `B0_reddit` | `B0_classifieds` | 224 | 73.9% → 56.7% | **0.677** | 0.647 | 0.089 / 0.039 | 0.010 | 0.726 |
| `B1_classifieds` | `B1_reddit` | 203 | 75.4% → 88.2% | **0.718** | 0.679 | 0.103 / 0.063 | 0.035 | 0.864 |
| `B1_reddit` | `B1_classifieds` | 224 | 88.2% → 75.4% | **0.738** | 0.701 | 0.102 / 0.045 | 0.005 | 0.732 |
| `B2_classifieds` | `B2_reddit` | 203 | 92.9% → 92.6% | **0.732** | 0.725 | 0.124 / 0.078 | 0.050 | 0.615 |
| `B2_reddit` | `B2_classifieds` | 224 | 92.6% → 92.9% | **0.627** | 0.731 | 0.122 / 0.075 | 0.313 | 0.642 |

## 2. ⚠️ The pooled protocol is WITHDRAWN

An earlier version of this artifact also pooled all three models of one site (≈672 rows) and tested on each cell of the other, reporting that pooling cleared the null in 5 of 6 and even beat one cell's own within-cell ceiling. **That comparison is withdrawn**, because the pooled design does not estimate the quantity it is tested against:

- **classifieds**, 3 cells, 224 shared tasks: the feature vector is **byte-identical across all models on 131 tasks (58.5%)**, the label conflicts across models on 94 (42.0%), and **56 tasks (25.0%) are both at once**.
- **reddit**, 3 cells, 203 shared tasks: the feature vector is **byte-identical across all models on 12 tasks (5.9%)**, the label conflicts across models on 51 (25.1%), and **3 tasks (1.5%) are both at once**.

The abstention label is per-(task, model) — *did this model's six modes solve it* — while the features are step-0 observation statistics and task config, which are largely model-invariant. Pooling therefore trains on same-x-different-y triples and can only recover *average-LLM solvability*; scoring it against a single model's cell is a target mismatch, not a transfer result. The numbers remain in the JSON under `pooled` for the record; nothing in this document rests on them.

## 3. Transferred operating point — threshold never sees the test site

The threshold is chosen by an inner 5-fold **inside the training site** at the stated solvable-loss budget, then applied unseen to the other site. This is the cross-site analogue of the nested column in `abstention_learnability` §3, and it is a held-out *policy*, not merely a held-out prediction (§465).

⚠️ **The percentage budget is quantised.** What the inner CV actually enforces is an integer: `floor(budget × solvable_train)`. On a small solvable set several percentage rows collapse onto the SAME integer and are therefore the same policy — the `budget→tasks` column below makes that visible instead of implying a sweep that does not exist.

| train | test | budget | **budget→tasks** | abstain rate | solvable lost | realised loss | saved |
|---|---|---:|---:|---:|---:|---:|---:|
| `B0_classifieds` | `B0_reddit` | ≤0% | **0 / 97** | 3.9% | 0/53 | 0.0% | 5.8% |
| `B0_classifieds` | `B0_reddit` | ≤2% | **1 / 97** | 14.8% | 1/53 | 1.9% | 19.0% |
| `B0_classifieds` | `B0_reddit` | ≤5% | **4 / 97** | 22.2% | 3/53 | 5.7% | 26.2% |
| `B0_classifieds` | `B0_reddit` | ≤10% | **9 / 97** | 31.5% | 7/53 | 13.2% | 38.5% |
| `B0_reddit` | `B0_classifieds` | ≤0% | **0 / 53** | 2.2% | 0/97 | 0.0% | 3.0% |
| `B0_reddit` | `B0_classifieds` | ≤2% | **1 / 53** | 3.1% | 0/97 | 0.0% | 3.3% |
| `B0_reddit` | `B0_classifieds` | ≤5% | **2 / 53** | 5.8% | 0/97 | 0.0% | 6.6% |
| `B0_reddit` | `B0_classifieds` | ≤10% | **5 / 53** | 12.5% | 7/97 | 7.2% | 15.0% |
| `B1_classifieds` | `B1_reddit` | ≤0% | **0 / 55** | 9.9% | 0/24 | 0.0% | 9.1% |
| `B1_classifieds` | `B1_reddit` | ≤2% | **1 / 55** | 11.3% | 0/24 | 0.0% | 11.0% |
| `B1_classifieds` | `B1_reddit` | ≤5% | **2 / 55** | 17.7% | 0/24 | 0.0% | 16.8% |
| `B1_classifieds` | `B1_reddit` | ≤10% | **5 / 55** | 30.0% | 1/24 | 4.2% | 28.7% |
| `B1_reddit` | `B1_classifieds` | ≤0% | **0 / 24** | 62.1% | 20/55 | 36.4% | 67.9% |
| `B1_reddit` | `B1_classifieds` | ≤2% | **0 / 24** | 62.1% | 20/55 | 36.4% | 67.9% |
| `B1_reddit` | `B1_classifieds` | ≤5% | **1 / 24** | 66.1% | 21/55 | 38.2% | 73.0% |
| `B1_reddit` | `B1_classifieds` | ≤10% | **2 / 24** | 70.5% | 24/55 | 43.6% | 77.5% |
| `B2_classifieds` | `B2_reddit` | ≤0% | **0 / 16** | 12.8% | 1/15 | 6.7% | 9.9% |
| `B2_classifieds` | `B2_reddit` | ≤2% | **0 / 16** | 12.8% | 1/15 | 6.7% | 9.9% |
| `B2_classifieds` | `B2_reddit` | ≤5% | **0 / 16** | 12.8% | 1/15 | 6.7% | 9.9% |
| `B2_classifieds` | `B2_reddit` | ≤10% | **1 / 16** | 14.3% | 1/15 | 6.7% | 11.3% |
| `B2_reddit` | `B2_classifieds` | ≤0% | **0 / 15** | 3.1% | 0/16 | 0.0% | 3.6% |
| `B2_reddit` | `B2_classifieds` | ≤2% | **0 / 15** | 3.1% | 0/16 | 0.0% | 3.6% |
| `B2_reddit` | `B2_classifieds` | ≤5% | **0 / 15** | 3.1% | 0/16 | 0.0% | 3.6% |
| `B2_reddit` | `B2_classifieds` | ≤10% | **1 / 15** | 6.2% | 0/16 | 0.0% | 6.7% |

⚠️ **A budget met at home is not a budget met abroad.** The budget column is what the threshold bought on the training site; the realised-loss column is what it actually cost on the test site. Where the two diverge, the cause is base-rate drift (56.7%–92.9% across these cells), and it is exactly the failure mode within-cell CV cannot show.

⚠️ Cost coverage is the fraction of test-cell tasks with a cost figure in `per_task_sr.csv`; that product is on the leak-kept, n=205 reddit convention while the labels here come from the canonical n=203 universe (§462.1). The join is by task id, so surplus rows are simply unused, but the saved-dollar column inherits the cost side's convention.

## 4. What transfers, and what does not

**Ranking transfers, on the cells that have the events to show it.** 5 of 6 matched transfers clear their permutation null at p≤0.05. The 1 that do not — `B2_reddit`→`B2_classifieds` (p=0.313) — are **indeterminate, not negative**: their test cells carry so few solvable tasks that the null's own spread (SD 0.075) is of the same order as any effect one could hope to see there.

**The operating point does not transfer, and the ranking does not warn you.** 15 of 24 transferred thresholds kept the realised loss inside the budget they were bought at. The worst is `B1_reddit` → `B1_classifieds` at ≤10%: it abstains on 70.5% of the test site and loses 24/55 (43.6%) of its solvable tasks.

⚠️ **What this is NOT.** An earlier version attributed the failure to *base-rate drift* and paired it with the observation that the highest-AUROC direction fails worst. Both were withdrawn against this table. Base-rate drift does not order the failures: the largest drop (-17.2pp) stays inside budget while a near-zero drop overruns it; and pairing the max-AUROC direction with the max-loss direction over 6 transfers is a 1-in-6 coincidence under the null, not a mechanism.

**What the table does support** is quantisation. Of the 4×6 nominal budget rows, **5 are duplicates of another row in the same direction** — the integer `floor(budget × solvable_train)` repeats when the solvable set is small. The budget axis is coarser than it looks, and a policy bought at one nominal budget can be the identical policy sold at another. That is the cross-site form of the distinction §465 drew within a cell: **a held-out prediction is not a held-out policy** — and here the policy is not even a continuum.

⚠️ Two sites is two points. Nothing here licenses a claim about transfer to a *third* site; what it licenses is that ranking survived the one site change available on the cells that had the events to test it, and calibration did not.
