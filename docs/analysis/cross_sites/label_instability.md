---
type: analysis
status: complete
created: 2026-08-02
purpose: is run-to-run instability spread over the benchmark, or concentrated on the tasks a router learns from
post_hoc_exploratory: true
scope_warning: one cell (cls_B0), 3 of 6 arms replicated once each. Every figure is a LOWER bound on the flip rate; replicating more arms can only add flips.
producer: scripts/analysis/aggregate_label_instability.py
---

# Where the instability sits

Regenerate: `.venv/bin/python3 scripts/analysis/aggregate_label_instability.py`

Cell `cls_B0`, n = 224. Replicated arms: `B0.cls.dom`, `B0.cls.vision`, `B0.cls.som`. **67 of 224 tasks change outcome between the two runs of at least one replicated arm.**

| stratum | tasks | share of cell | flipped | flip rate | share of all flips | vs complement |
|---|---|---|---|---|---|---|
| which-mode label rows (any mode solved) | 97 | 43.3% | 61 | 62.9% | 91.0% | **10.7x** |
| **…of those, the arms DISAGREE (the choice matters)** | 88 | 39.3% | 59 | **67.0%** | 88.1% | **11.4x** |
| three-way channel decision is contested | 74 | 33.0% | 49 | 66.2% | 73.1% | **11.3x** |
| exactly one mode solved it (label unambiguous) | 29 | 12.9% | 23 | 79.3% | 34.3% | **13.5x** |
| COMPLEMENT: no mode solved, or all did | 136 | 60.7% | 8 | 5.9% | 11.9% | 1.00x |
| whole cell | 224 | 100.0% | 67 | 29.9% | 100.0% | **5.1x** |

## Reading

The tasks on which the arms disagree are the only rows a which-mode router can learn from, and they are **39.3%** of the cell. They carry **88.1%** of all observed flips. Their flip rate is **67.0%** against **5.9%** on the complement, an enrichment of **11.4x**.

This is not the same statement as 'the benchmark is noisy'. Aggregate success rate between these same two runs moves by under 2.3 points, which any reader would call reproducible. The per-task counterfactual labels that routing needs are not, and the gap between those two facts is the point: **instability concentrates precisely where the decision is contested**, so a router is fitted on the least stable subset of the benchmark by construction.

It also bounds the problem independently of sample size. More data does not repair a target that a rerun rewrites, so this obstruction is of a different kind from the supply and predictability results, which a larger or easier benchmark could move.

## Is the enrichment just arithmetic?

"Contested" means at least one arm solved the task and at least one did not, which is by definition a **mid-difficulty band**. A task with true per-run success rate *p* flips between two runs with probability *2p(1−p)*: maximal near 0.5, zero at either end. So the enrichment could be nothing but the complement being full of tasks nobody solves. Taking *k/6* — how many of the six modes solved it — as a difficulty proxy:

| set | n | observed flip rate | binomial floor *2p(1−p)* |
|---|---|---|---|
| contested | 88 | 67.05% | 37.25% |
| complement | 136 | 5.88% | 0.00% |

**The attack fails, and it fails in the unexpected direction.** The complement's predicted rate is exactly zero — *k*=0 and *k*=6 both give *2p(1−p)*=0 — so the arithmetic enrichment is **infinite**. The observed figure is therefore *deflated* by this mechanism, not inflated: the complement flips more than the model permits at all, including two of the nine tasks that every mode solved.

**But the same table limits the claim.** Inside the contested band the observed rate exceeds the floor by only **1.80×** (67.0% against 37.2%). Most of the 51% is the band being mid-difficulty; the excess above that floor is what is left for structure to carry. The honest sentence is that instability concentrates on contested tasks **and** that being contested is itself most of the reason.

| *k* solved | n | flipped | observed | floor |
|---|---|---|---|---|
| 0 | 127 | 6 | 4.72% | 0.00% |
| 1 | 29 | 23 | 79.31% | 27.78% |
| 2 | 25 | 10 | 40.00% | 44.44% |
| 3 | 12 | 10 | 83.33% | 50.00% |
| 4 | 9 | 7 | 77.78% | 44.44% |
| 5 | 13 | 9 | 69.23% | 27.78% |
| 6 | 9 | 2 | 22.22% | 0.00% |

⚠️ The proxy is crude: the six modes are different representations, not six draws from one model, so *k/6* estimates difficulty rather than *p*. The per-*k* rates are not monotone in the floor, which is itself evidence the proxy is imperfect.

### …and is the proxy circular?

Yes, partly. The flips are defined by rerunning **dom** and **som** and **vision**, and those same two arms enter the six-mode difficulty proxy. A task's solve status on them therefore decides *both* whether it counts as contested *and* whether it counts as flipped. Rebuilding the proxy from the other four arms breaks the loop:

| proxy | contested | complement | enrichment |
|---|---|---|---|
| all six modes (as claimed) | 67.05% (n=88) | 5.88% (n=136) | **11.40×** |
| the four not replicated | 67.65% (n=34) | 23.16% (n=190) | **2.92×** |

**Neither figure may be quoted alone.** The six-mode version is the correct operationalisation of the *claim* — a router chooses among all six arms, so "contested" has to be defined over all six — but as a *difficulty control* it reuses the arms that define the outcome. The four-arm version is not circular and still shows **2.92×** enrichment, with the complement rate rising from 5.88% to 23.16% because genuinely unstable tasks move out of the contested set. The honest sentence is that instability is enriched on contested tasks by somewhere between 2.9× and 11.4× depending on whether the definition of contested is allowed to see the replicated arms.
