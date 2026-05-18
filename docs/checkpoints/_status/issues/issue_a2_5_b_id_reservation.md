---
name: a2_5_b_id_reservation
date_opened: 2026-05-18
status: active
owner: claude_session_a2_5
session_id: 2026-05-18_001948
description: A2.5 /stress workflow reserved B-994 through B-1006 for fix landing
b_ids_reserved: [B-994, B-995, B-996, B-997, B-998, B-999, B-1000, B-1001, B-1002, B-1003, B-1004, B-1005, B-1006]
next_available_for_other_sessions: B-1007+
---

# A2.5 B-### Reservation (2026-05-18 ~001948 BST)

Atomic reservation for A2.5 /stress fix landing. Parallel sessions should consume B-1007+ to avoid collision.

## Mapping (B-### → Chunk)

| B-### | Severity-Source | Fix description | Chunk |
|---|---|---|---|
| B-994 | P0-1-ABC* | Pass-1/Pass-2 task-ID leak (LOCO leak fix via within-cell 5-fold CV deployment) | B + D prose |
| B-995 | P0-2-AB* | Oracle label collapse + minority hallucination (no-success filter + min_class_n=10 + class_weight=None) | B |
| B-996 | P0-3-A* | Feature schema drift (Stage 1 50-cand extractor + Stage 2 fold-local TF-IDF + global pooled MI selection — scope EXPANDED) | A + B |
| B-997 | P0-4-A | LOCO substrate absent (5-fold within-cell CV trainer + 30 pickles + fold_assignment) | B |
| B-998 | P0-5-BC* | Cost-aware mismatch (predict_proba + cost-weighted decision rule + τ inner-CV tuning) | B + C |
| B-999 | P0-6-B* | H10 estimand primitive wrong (line 212 K-of-6 PRIMARY + line 625 FE pool → APPENDIX sensitivity) | D prereg |
| B-1000 | P0-7-C* | §1 prose rule-based contradiction (delete "two complementary routing approaches" promise) | D |
| B-1001 | P1-8-A | section6_router.md v0 placeholder | D |
| B-1002 | P1-9-A | H10 Pareto verdict producer (aggregate_h10_pareto.py NEW) | C |
| B-1003 | P1-10-A | paper_planning §8 v6 cascade stale → trim + cross-link §C2 | D |
| B-1004 | P1-11-B | Bootstrap CI row-unit → task-unit refactor | D |
| B-1005 | P1-12-B | Anchor-flicker fallback producer for Phase 1a | D |
| B-1006 | P1-13-C | Missing intelligent baselines (always-cheapest + decision stump + per-task lookup) | D |

## P2 deferred

| B-### | Severity-Source | Fix description | Disposition |
|---|---|---|---|
| B-1007 (proposed) | P2-14-C | §8 limitations 2-site disclaimer | Will land in Chunk D as part of B-1006/§8 prose work |

## User decisions (5 user-OOB-catches over /stress A2.5 cycle)

1. **#1**: "我不是有 6 extra cells 吗" — cell-count vs task-id distinction surfaced; led to Q1=C decision (within-cell 5-fold CV deployment, not GroupKFold-task-id cross-baseline)
2. **#2**: "pooled MI 用全数据会泄漏吗" — caught option (IV) leak; retracted, moved to (E') per-(cell, fold)
3. **#3**: "fold-local pooled MI" — initial refinement to (E')
4. **#4**: "global fold-local pooled MI" — canonical sklearn pattern refinement to (E''): 5 selectors per fold (cross-cell uniform within fold)
5. **#5 (GPT-relay)**: TF-IDF transductive + vectorizer state + cell-constant exclusion + scaler-in-pipeline + τ pre-lock → (E''') 4 methodology rigor fixes

Q4: K-of-6 PRIMARY + APPENDIX FE pool (3-AI unanimous, user implicit confirm via accept-all-推荐 sweep)

Q3 (B-998 cost-aware): user confirmed "title 不需要太改, 后面再说" → keep "Cost-Aware Routing" title, add cost-weighted decision rule + ECE/Brier calibration

(b) τ inner-fold-CV tuning ADDED 2026-05-18 — per-(cell, fold) τ_{C,k} tuned on inner 5-fold CV on train_C_k over candidate τ ∈ [0.3, 0.4, 0.5, 0.6, 0.7]

Paper-1/Paper-2 scope split: B-1000 and B-1007 NOT expanded (paper-2 scope per user "之后多site和其他的router是paper2的问题")
