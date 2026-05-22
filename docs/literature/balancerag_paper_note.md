# Paper Reading Note — *BalanceRAG: Joint Risk Calibration for Cascaded RAG*

**Source:** Jia, Ye, Jia, Qian, Wang, Chen, Tang, Yu, Wang, *BalanceRAG: Joint Risk Calibration for Cascaded Retrieval-Augmented Generation*, arXiv:2605.20084, 2026-05.
**Found via:** p79-lit-digest cron 2026-05-22 (Score 2, RELATION = DESIGN-INPUT).
**Use case for our project:** **paper-2 cascade router** threshold calibration (DEFERRED from paper-1 §6 per `paper_planning §8` v7 walk-back) + **paper-1-main §6.6 deferred cost-weighted objective** (`SR̂ - λ·Cost`). **NOT a paper-1 §6 current-claim cite** — paper-1 §6 is a single learned router with a single per-(cell,fold) τ, no cascade.

---

## 0. One-sentence summary

A 3-stage selective-prediction cascade (LLM-only → RAG fallback → abstain) where the two stage thresholds are calibrated **jointly** — each threshold *pair* is an operating point on a 2D lattice, and **sequential graphical testing** identifies jointly-safe points that control the **system-level error rate among accepted answers while maximizing coverage** — instead of tuning each stage's threshold independently.

## 1. Core problem

Per-stage independent threshold tuning does NOT control the *end-to-end* (system-level) error/cost of a cascade: each stage's local calibration ignores the conditional distribution induced by the upstream stage's accept/escalate decision. BalanceRAG treats the threshold *vector* as the calibration unit.

## 2. Mechanism (precise)

- **Not** a Lagrangian, **not** a learned gate. The calibration is a **2D lattice of threshold pairs** + **sequential graphical testing** (multiple-testing-aware) to find jointly-safe operating points at a prescribed risk level.
- Objective = **risk-controlled coverage maximization**: control the error rate *among accepted points* at a target, retain the maximum number of accepted-correct examples, reduce unnecessary retrieval calls vs always-on RAG.
- **Multi-risk extension**: bounds retrieval *usage* (a cost proxy) alongside the selection-conditioned risk → a two-objective (risk × cost) controlled cascade.
- Evaluated on 3 open-domain QA benchmarks × multiple LLM backbones; reported (qualitatively in abstract) to meet risk targets at higher coverage + fewer retrieval calls than always-on RAG. No per-number results on the abs page.

## 3. Mapping to P79

| BalanceRAG | P79 analogue |
|---|---|
| 3-stage cascade LLM→RAG→abstain | §1 token-monotonic cascade DOM→P-text→P-SoM→SoM (+ "fall back to phantom_som under low conf", B-998) — **paper-2 operational scope per §8** |
| binary correct / abstain selective prediction | K-class mode routing with cost-monotone ordering |
| 2D threshold lattice (2 stages) | (K−1)-dim escalation-threshold lattice for a K-stage cascade |
| sequential graphical testing | naturally applicable — the token-monotonic cascade *is* an ordered escalation, so the testing sequence is the cascade order |
| risk-controlled coverage maximization | reviewer-defensible alternative to §6.6's deferred λ-scalarized `SR̂ - λ·Cost`: set an SR/risk target, find jointly-safe escalation thresholds maximizing the fraction of tasks served by cheaper modes |

## 4. Why it does NOT touch paper-1 §6 (scope honesty)

Paper-1 §6 (v7 walk-back, `paper_planning §8`) = **single learned router**, one τ per (cell, fold) chosen by inner-CV maximizing SR. There is no multi-stage escalation in paper-1, so there is no threshold *vector* to jointly calibrate. Cascade composition + first-step trigger + B1→B0 escalation are all explicitly **paper-2** (`§8 "Cascade + escalation (DEFERRED paper-2)"`). BalanceRAG is parked there.

## 5. Disposition

- **Score 2 / DESIGN-INPUT**, parked against **paper-2 cascade** (primary) + **paper-1-main Phase-1b §6.6 cost-weighted objective** (secondary).
- Action when paper-2 cascade is built: replace ad-hoc per-stage τ with joint lattice calibration; cite as the principled risk-control method.
- Not added to `paper.bib` yet (paper-1 scope); add when paper-2 cascade prose is drafted.
