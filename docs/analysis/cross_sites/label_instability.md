---
type: analysis
status: complete
created: 2026-08-02
purpose: is run-to-run instability spread over the benchmark, or concentrated on the tasks a router learns from
post_hoc_exploratory: true
scope_warning: primary cell cls_B0 (6 of 6 arms replicated once each); also red_B0, wared_B1 (all six arms replicated), tables in the last section. Every figure is a LOWER bound on the flip rate; replicating more arms can only add flips.
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

This is not the same statement as 'the benchmark is noisy'. Aggregate success rate between two runs of one arm moves by a few points (the per-arm band is in `noise_floor_inventory.md`), which most readers would call reproducible. The per-task counterfactual labels that routing needs are not, and the gap between those two facts is the point: **instability concentrates precisely where the decision is contested**, so a router is fitted on the least stable subset of the benchmark by construction.

It is also a different kind of obstruction from the supply and predictability results: more *tasks* do not repair a target that a rerun rewrites, but more *runs per task* can, by averaging the label (笔记 §505.7 puts the price at k≥3 runs on classifieds and k≥9 on reddit). It is a statement about single-draw labels, not a proof that the conditional expectation is unlearnable.

## Is the enrichment just arithmetic?

"Contested" means at least one arm solved the task and at least one did not, which is by definition a **mid-difficulty band**. A task with true per-run success rate *p* flips between two runs of one arm with probability *2p(1−p)*: maximal near 0.5, zero at either end. A task counts as flipped here if **any** of the 6 replicated arms flips, so the floor for the same event is *1 − (1 − 2p(1−p))^6* (arms taken as independent). So the enrichment could be nothing but the complement being full of tasks nobody solves. Taking *k/6* — how many of the six modes solved it — as a difficulty proxy:

| set | n | observed flip rate | floor *1−(1−2p(1−p))^6* |
|---|---|---|---|
| contested | 88 | 81.82% | 91.88% |
| complement | 136 | 10.29% | 0.00% |

**The attack fails, and it fails in the unexpected direction.** The complement's predicted rate is exactly zero — *k*=0 and *k*=6 both give a zero floor — so the arithmetic enrichment is **infinite**. The observed figure is therefore *deflated* by this mechanism, not inflated: the complement flips more than the model permits at all, including 3 of the 9 tasks that every mode solved.

**But the same table limits the claim.** Inside the contested band the observed rate is **0.89×** the floor (81.8% against 91.9%). At or below 1, this crude model needs no task×mode structure at all to produce the contested band's instability: being mid-difficulty is enough. The honest sentence is that instability concentrates on contested tasks **and** that being contested is itself the reason, as far as this proxy can tell. (Until 2026-10-07 this compared the six-arm union with a single-arm floor and reported a 2.20× excess on cls_B0; 笔记 §541.) A null that conditions on task and arm margins instead of a k/6 proxy is the stronger test.

| *k* solved | n | flipped | observed | floor |
|---|---|---|---|---|
| 0 | 127 | 11 | 8.66% | 0.00% |
| 1 | 29 | 26 | 89.66% | 85.81% |
| 2 | 25 | 15 | 60.00% | 97.06% |
| 3 | 12 | 11 | 91.67% | 98.44% |
| 4 | 9 | 7 | 77.78% | 97.06% |
| 5 | 13 | 13 | 100.00% | 85.81% |
| 6 | 9 | 3 | 33.33% | 0.00% |

⚠️ The proxy is crude: the six modes are different representations, not six draws from one model, so *k/6* estimates difficulty rather than *p*. The per-*k* rates are not monotone in the floor, which is itself evidence the proxy is imperfect.

### …and is the proxy circular?

**Yes, and the control that used to answer this is no longer available.** every mode in MODES now carries a replicate; no un-replicated arm remains to build an independent difficulty proxy from. Replicated: `dom`, `pprompt`, `psom`, `ptext`, `som`, `vision`.

⚠️ the six-arm enrichment figure has lost its anti-circularity control and must not be quoted as a headline on its own. Candidate fix: leave-one-out proxy (five other arms per arm) — changes the estimand, needs an explicit decision.

This is a consequence of the replicate inventory becoming *complete*, not of a defect — the control was only ever possible while some arm lacked a replicate. It is recorded here rather than silently dropped.

## Other cells with every arm replicated

Same computation, per cell (added 2026-10-07, 实验笔记 §537). The reading above is cls_B0's; these cells are read from their own tables.

### `red_B0` — n = 203, 54 tasks flip on at least one of 6 replicated arms

| stratum | tasks | share of cell | flipped | flip rate | share of all flips | vs complement |
|---|---|---|---|---|---|---|
| which-mode label rows (any mode solved) | 53 | 26.1% | 44 | 83.0% | 81.5% | 10.65x |
| …of those, the arms DISAGREE (the choice matters) | 49 | 24.1% | 42 | 85.7% | 77.8% | 11.00x |
| three-way channel decision is contested | 47 | 23.2% | 40 | 85.1% | 74.1% | 10.92x |
| exactly one mode solved it (label unambiguous) | 17 | 8.4% | 15 | 88.2% | 27.8% | 11.32x |
| COMPLEMENT: no mode solved, or all did | 154 | 75.9% | 12 | 7.8% | 22.2% | 1.00x |
| whole cell | 203 | 100.0% | 54 | 26.6% | 100.0% | 3.41x |

| set | n | observed flip rate | floor *1−(1−2p(1−p))^6* | observed / floor |
|---|---|---|---|---|
| contested | 49 | 85.71% | 91.46% | 0.94x |
| complement | 154 | 7.79% | 0.00% | — |

Leave-replicated-out control: unavailable here for the same reason as above (every arm of this cell is replicated).

### `wared_B1` — n = 104, 25 tasks flip on at least one of 6 replicated arms

| stratum | tasks | share of cell | flipped | flip rate | share of all flips | vs complement |
|---|---|---|---|---|---|---|
| which-mode label rows (any mode solved) | 32 | 30.8% | 20 | 62.5% | 80.0% | 9.63x |
| …of those, the arms DISAGREE (the choice matters) | 27 | 26.0% | 20 | 74.1% | 80.0% | 11.41x |
| three-way channel decision is contested | 27 | 26.0% | 20 | 74.1% | 80.0% | 11.41x |
| exactly one mode solved it (label unambiguous) | 14 | 13.5% | 12 | 85.7% | 48.0% | 13.20x |
| COMPLEMENT: no mode solved, or all did | 77 | 74.0% | 5 | 6.5% | 20.0% | 1.00x |
| whole cell | 104 | 100.0% | 25 | 24.0% | 100.0% | 3.70x |

| set | n | observed flip rate | floor *1−(1−2p(1−p))^6* | observed / floor |
|---|---|---|---|---|
| contested | 27 | 74.07% | 89.71% | 0.83x |
| complement | 77 | 6.49% | 0.00% | — |

Leave-replicated-out control: unavailable here for the same reason as above (every arm of this cell is replicated).

