---
type: analysis
status: complete
created: 2026-08-02
purpose: is run-to-run instability spread over the benchmark, or concentrated on the tasks a router learns from
post_hoc_exploratory: true
scope_warning: one cell (cls_B0), two of six arms replicated once each. Every figure is a LOWER bound on the flip rate; replicating more arms can only add flips.
producer: scripts/analysis/aggregate_label_instability.py
---

# Where the instability sits

Regenerate: `.venv/bin/python3 scripts/analysis/aggregate_label_instability.py`

Cell `cls_B0`, n = 224. Replicated arms: `B0.cls.dom`, `B0.cls.vision`. **49 of 224 tasks change outcome between the two runs of at least one replicated arm.**

| stratum | tasks | share of cell | flipped | flip rate | share of all flips | vs complement |
|---|---|---|---|---|---|---|
| which-mode label rows (any mode solved) | 97 | 43.3% | 47 | 48.5% | 95.9% | **16.5x** |
| **…of those, the arms DISAGREE (the choice matters)** | 88 | 39.3% | 45 | **51.1%** | 91.8% | **17.4x** |
| three-way channel decision is contested | 74 | 33.0% | 36 | 48.6% | 73.5% | **16.5x** |
| exactly one mode solved it (label unambiguous) | 29 | 12.9% | 15 | 51.7% | 30.6% | **17.6x** |
| COMPLEMENT: no mode solved, or all did | 136 | 60.7% | 4 | 2.9% | 8.2% | 1.00x |
| whole cell | 224 | 100.0% | 49 | 21.9% | 100.0% | **7.4x** |

## Reading

The tasks on which the arms disagree are the only rows a which-mode router can learn from, and they are **39.3%** of the cell. They carry **91.8%** of all observed flips. Their flip rate is **51.1%** against **2.9%** on the complement, an enrichment of **17.4x**.

This is not the same statement as 'the benchmark is noisy'. Aggregate success rate between these same two runs moves by under 2.3 points, which any reader would call reproducible. The per-task counterfactual labels that routing needs are not, and the gap between those two facts is the point: **instability concentrates precisely where the decision is contested**, so a router is fitted on the least stable subset of the benchmark by construction.

It also bounds the problem independently of sample size. More data does not repair a target that a rerun rewrites, so this obstruction is of a different kind from the supply and predictability results, which a larger or easier benchmark could move.

## Is the enrichment just arithmetic?

"Contested" means at least one arm solved the task and at least one did not, which is by definition a **mid-difficulty band**. A task with true per-run success rate *p* flips between two runs with probability *2p(1−p)*: maximal near 0.5, zero at either end. So the enrichment could be nothing but the complement being full of tasks nobody solves. Taking *k/6* — how many of the six modes solved it — as a difficulty proxy:

| set | n | observed flip rate | binomial floor *2p(1−p)* |
|---|---|---|---|
| contested | 88 | 51.14% | 37.25% |
| complement | 136 | 2.94% | 0.00% |

**The attack fails, and it fails in the unexpected direction.** The complement's predicted rate is exactly zero — *k*=0 and *k*=6 both give *2p(1−p)*=0 — so the arithmetic enrichment is **infinite**. The observed figure is therefore *deflated* by this mechanism, not inflated: the complement flips more than the model permits at all, including two of the nine tasks that every mode solved.

**But the same table limits the claim.** Inside the contested band the observed rate exceeds the floor by only **1.37×** (51.1% against 37.2%). Most of the 51% is the band being mid-difficulty; the excess above that floor is what is left for structure to carry. The honest sentence is that instability concentrates on contested tasks **and** that being contested is itself most of the reason.

| *k* solved | n | flipped | observed | floor |
|---|---|---|---|---|
| 0 | 127 | 2 | 1.57% | 0.00% |
| 1 | 29 | 15 | 51.72% | 27.78% |
| 2 | 25 | 6 | 24.00% | 44.44% |
| 3 | 12 | 10 | 83.33% | 50.00% |
| 4 | 9 | 5 | 55.56% | 44.44% |
| 5 | 13 | 9 | 69.23% | 27.78% |
| 6 | 9 | 2 | 22.22% | 0.00% |

⚠️ The proxy is crude: the six modes are different representations, not six draws from one model, so *k/6* estimates difficulty rather than *p*. The per-*k* rates are not monotone in the floor, which is itself evidence the proxy is imperfect.
