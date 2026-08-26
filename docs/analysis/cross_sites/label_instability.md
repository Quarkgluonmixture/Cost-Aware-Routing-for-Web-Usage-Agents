---
type: analysis
status: complete
created: 2026-08-02
purpose: is run-to-run instability spread over the benchmark, or concentrated on the tasks a router learns from
post_hoc_exploratory: true
scope_warning: one cell (cls_B0), 6 of 6 arms replicated once each. Every figure is a LOWER bound on the flip rate; replicating more arms can only add flips.
producer: scripts/analysis/aggregate_label_instability.py
---

# Where the instability sits

Regenerate: `.venv/bin/python3 scripts/analysis/aggregate_label_instability.py`

Cell `cls_B0`, n = 224. Replicated arms: `B0.cls.dom`, `B0.cls.vision`, `B0.cls.som`, `B0.cls.ptext`, `B0.cls.pprompt`, `B0.cls.psom`. **86 of 224 tasks change outcome between the two runs of at least one replicated arm.**

| stratum | tasks | share of cell | flipped | flip rate | share of all flips | vs complement |
|---|---|---|---|---|---|---|
| which-mode label rows (any mode solved) | 97 | 43.3% | 75 | 77.3% | 87.2% | **7.5x** |
| **…of those, the arms DISAGREE (the choice matters)** | 88 | 39.3% | 72 | **81.8%** | 83.7% | **7.9x** |
| three-way channel decision is contested | 74 | 33.0% | 60 | 81.1% | 69.8% | **7.9x** |
| exactly one mode solved it (label unambiguous) | 29 | 12.9% | 26 | 89.7% | 30.2% | **8.7x** |
| COMPLEMENT: no mode solved, or all did | 136 | 60.7% | 14 | 10.3% | 16.3% | 1.00x |
| whole cell | 224 | 100.0% | 86 | 38.4% | 100.0% | **3.7x** |

## Reading

The tasks on which the arms disagree are the only rows a which-mode router can learn from, and they are **39.3%** of the cell. They carry **83.7%** of all observed flips. Their flip rate is **81.8%** against **10.3%** on the complement, an enrichment of **7.9x**.

This is not the same statement as 'the benchmark is noisy'. Aggregate success rate between these same two runs moves by under 2.3 points, which any reader would call reproducible. The per-task counterfactual labels that routing needs are not, and the gap between those two facts is the point: **instability concentrates precisely where the decision is contested**, so a router is fitted on the least stable subset of the benchmark by construction.

It also bounds the problem independently of sample size. More data does not repair a target that a rerun rewrites, so this obstruction is of a different kind from the supply and predictability results, which a larger or easier benchmark could move.

## Is the enrichment just arithmetic?

"Contested" means at least one arm solved the task and at least one did not, which is by definition a **mid-difficulty band**. A task with true per-run success rate *p* flips between two runs with probability *2p(1−p)*: maximal near 0.5, zero at either end. So the enrichment could be nothing but the complement being full of tasks nobody solves. Taking *k/6* — how many of the six modes solved it — as a difficulty proxy:

| set | n | observed flip rate | binomial floor *2p(1−p)* |
|---|---|---|---|
| contested | 88 | 81.82% | 37.25% |
| complement | 136 | 10.29% | 0.00% |

**The attack fails, and it fails in the unexpected direction.** The complement's predicted rate is exactly zero — *k*=0 and *k*=6 both give *2p(1−p)*=0 — so the arithmetic enrichment is **infinite**. The observed figure is therefore *deflated* by this mechanism, not inflated: the complement flips more than the model permits at all, including two of the nine tasks that every mode solved.

**But the same table limits the claim.** Inside the contested band the observed rate exceeds the floor by only **2.20×** (81.8% against 37.2%). Most of the 51% is the band being mid-difficulty; the excess above that floor is what is left for structure to carry. The honest sentence is that instability concentrates on contested tasks **and** that being contested is itself most of the reason.

| *k* solved | n | flipped | observed | floor |
|---|---|---|---|---|
| 0 | 127 | 11 | 8.66% | 0.00% |
| 1 | 29 | 26 | 89.66% | 27.78% |
| 2 | 25 | 15 | 60.00% | 44.44% |
| 3 | 12 | 11 | 91.67% | 50.00% |
| 4 | 9 | 7 | 77.78% | 44.44% |
| 5 | 13 | 13 | 100.00% | 27.78% |
| 6 | 9 | 3 | 33.33% | 0.00% |

⚠️ The proxy is crude: the six modes are different representations, not six draws from one model, so *k/6* estimates difficulty rather than *p*. The per-*k* rates are not monotone in the floor, which is itself evidence the proxy is imperfect.

### …and is the proxy circular?

**Yes, and the control that used to answer this is no longer available.** every mode in MODES now carries a replicate; no un-replicated arm remains to build an independent difficulty proxy from. Replicated: `dom`, `pprompt`, `psom`, `ptext`, `som`, `vision`.

⚠️ the six-arm enrichment figure has lost its anti-circularity control and must not be quoted as a headline on its own. Candidate fix: leave-one-out proxy (five other arms per arm) — changes the estimand, needs an explicit decision.

This is a consequence of the replicate inventory becoming *complete*, not of a defect — the control was only ever possible while some arm lacked a replicate. It is recorded here rather than silently dropped.
