---
name: proposals-v7-learned-only
description: v7 router design — paper-1 learned-only walk-back. v6 cascade L2 (cycle + phantom-verbose) deferred to paper-2; paper-1 §6 contribution simplified to single learned router (LR over phantom-augmented mode set) with Pareto non-dominance H10 gate. Phase 1a 72 → 42 conditions.
metadata:
  type: project
  scope: paper-1 §6 router architecture (LOCKED 2026-05-16 v7)
  status: LOCKED 2026-05-16 — user Q3 confirmation walked back cascade
  parent: docs/checkpoints/router/proposals_v6.md
  supersedes: proposals_v6 (cascade architecture deferred to paper-2)
  paper-2-stub: docs/checkpoints/router/proposals_v6.md (kept as cascade forward stub)
---

# v7 router design — Learned-only paper-1 §6 + cascade deferred to paper-2

> **LOCKED 2026-05-16 v7** after user Q3 confirmation. Paper-1 §6 contribution is **single learned router** (LR over phantom-augmented mode set) with Pareto non-dominance H10 gate. v6 cascade architecture (L1 + L2 cycle + L2 phantom-verbose) preserved as paper-2 forward stub. Phase 1a condition count drops 72 → **42** (6× wall-clock saving on router pass).

## Why v6 → v7 walk-back

After 2-round cross-AI pre-fire stress (codex + gemini) + 4 user Q&A decisions on v6, user proposed (Q3) reducing paper-1 §6 to learned-only. Evaluated honestly:

| Concern with cascade in paper-1 | Resolved by learned-only |
|---|---|
| P1 v3 rule-based decision tree degenerate on archive (dead-code branches fire 0%) | ✓ rule-based router deferred to paper-2 |
| L2 verbose AUROC anchor partial-trajectory category error (8/12 cells fail step-3 ≥ 0.65) | ✓ no L2 layer in paper-1 |
| L2 cascade fallback dead-end at phantom_som 14.29% red ceiling (gemini #5) | ✓ no cascade fallback in paper-1 |
| §1 hook (phantom 4-fold drop-in latency claim) vs §6 mechanism (L2 verbose overhead) coherence collapse (gemini P1-17) | ✓ §1 hook fully consistent with single learned router |
| Cascade rebrand vs FrugalGPT/RouteLLM literature (gemini #4) | ✓ learned router stays within standard literature framing |
| M3/M4 absorption advisor sync gate (P1-22) | ✓ M3/M4 + cascade fully deferred to paper-2, no sync needed |
| Phase 1a 72-cond wall-clock 2-4 weeks | ✓ reduced to 42-cond 1.5-2.5 weeks A100 |
| 3-test Holm correction over {H9, H10, H11} | ✓ singleton {H10}, no correction |

**Cost** (what paper-1 loses):
- "Rule-based vs learned" comparison narrative (but archive shows rule-based degenerate, so weak claim anyway)
- "Cascade router" name (but FrugalGPT/RouteLLM already saturated literature; rebrand risk catches up to paper anyway)
- Larger paper §6 figure (single Pareto point vs cascade trajectory)

Net: **paper-1 contribution shape sharper**, paper-2 contribution shape thicker (M3/M4 + cascade + mechanism). Cleaner two-paper division.

## v7 Architecture (paper-1 §6 lock)

```
task arrives
    ↓
[L_learned] Multinomial Logistic Regression
    features (8):
      - site (one-hot cls / red / shop)
      - capability_tier (one-hot B0 235B / B1 4B / B2 4B-cross-family)
      - has_reference_image (bool)
      - intent_color_regex (bool)
      - intent_compare_regex (bool)
      - intent_search_regex (bool)
      - intent_token_count (z-scored in-fold)
      - axtree_element_count (z-scored in-fold, from step-0 state_digest.dom_complexity)
    target: oracle_best_mode (multinomial 6-class with balanced class_weight)
    output: routed_mode ∈ {dom, som, vision, phantom_text, phantom_prompt, phantom_som}
    ↓
agent runs in routed_mode for full episode (no L2 cascade, no mid-episode switch)
    ↓
episode outcome (Cost / SR / Latency) recorded
```

## §1 H10 (paper-1 sole router hypothesis)

Test whether **R_learned traces a (Cost, SR) point not Pareto-dominated** by any single-mode baseline {DOM, SoM, Vision, Phantom_SoM, Phantom_Text}, evaluated per cell across 6 planned Phase 1a cells.

**Estimand**:
    For each cell i, define point (C_learned_i, SR_learned_i) and 5 baseline points {(C_m_i, SR_m_i) for m ∈ baselines}.
    R_learned is **Pareto-non-dominated** at cell i if NO baseline m satisfies BOTH C_m_i ≤ C_learned_i AND SR_m_i ≥ SR_learned_i with at least one strict.
    Per-cell paired bootstrap (1000 resample): compute fraction of bootstrap samples where R_learned non-dominated. Cell passes if ≥ 95%.
    Pooled FE across 6 cells: report fraction of cells passing + meta-pooled Pareto frontier with cell-stratified (Cost, SR) confidence regions.

**H10 PRIMARY GATE**: pooled non-dominance test
    H0: R_learned dominated by at least one baseline in pooled population at α=0.05.
    Reject H0 if ≥ 5/6 cells pass paired bootstrap non-dominance at 95%.

**Latency dominance secondary check**: for each cell where R_learned passes (Cost, SR) Pareto non-dominance, verify latency_learned_i ≤ 1.10 × min_m latency_m_i (best-single-mode latency × 1.10 ceiling).

## §2 Site-asymmetric viability — paper §6 main finding

Archive simulation (`l1_archive_simulation_2026-05-16.md`) shows: balanced class_weight LR achieves cls +2.56pp vs always_phantom_som but red -4.76pp. Expected Phase 1a 6-cell pattern:

| Cell | Expected paper §6 narrative |
|---|---|
| cls visual-rich (3 cells × B0/B1/B2) | Learned router achieves Pareto non-dominance over best-single-mode (+0.5 to +2pp after nested CV correction); cls task distribution provides mode-asymmetric structure |
| red text-dominated (3 cells × B0/B1/B2) | Learned router prediction distribution collapses toward majority class (≈ always_phantom_som); per-cell Pareto non-dominance held marginally (≈ baseline); no router contribution beyond default mode swap |

**Paper §6 prose template**:
> "We evaluate a learned router L over the **phantom-augmented mode set** {DOM, SoM, Vision, Phantom-Text, Phantom-Prompt, Phantom-SoM} using site, capability tier, and task-content features. We test Pareto non-dominance on (Cost, SR) plane against 5 single-mode baselines per cell. We find learned routing exhibits **site-asymmetric viability**: visual-rich sites (classifieds) achieve Pareto-non-dominance at +X.X pp SR over best-single-mode; text-dominated sites (reddit) collapse to majority-class prediction matching always-phantom-SoM baseline. **This site-asymmetric pattern is an empirical finding**, not a router failure: it demonstrates that phantom routing space contributes routing-actionable structure only when the underlying site distribution provides mode-asymmetric task slices."

This is paper-grade interesting: not "router works/doesn't work" but "router viable conditional on site task-distribution heterogeneity" — a substantive contribution to cost-aware routing literature.

## §3 Phase 1a Protocol (v7-locked)

**Total**: 42 conditions = 36 baseline + 6 learned router

### Baseline pass (Launch 1)

```bash
bash scripts/queues/queue_phase1_paper_grade.sh launch
```

Generates: 36 conditions = 6 cells × 6 modes (cls + red × B0/B1/B2 × {DOM, SoM, Vision, P-text, P-prompt, P-SoM})

Wall-clock: 1-2 weeks A100

Outputs: paper §1 hook empirical evidence (phantom space + 4-fold drop-in property + oracle ceiling lift)

### Router pass (Launch 2, after baseline land)

```bash
bash scripts/queues/queue_phase1_router_paper_grade.sh launch  # new queue script — TODO
# OR: queue with PHASE1_VARIANT=router PHASE1_ROUTER_KIND=learned env override
```

Generates: 6 conditions = 6 cells × 1 learned router (`obs_mode="learned"` sentinel; runner queries LR per task)

Wall-clock: 3-5 days A100

Outputs: paper §6 H10 Pareto non-dominance evidence + per-cell L predictions + cost/SR/latency aggregates

## §4 Phase 1a CV protocol (v7 update)

**Within-cell**: 5-fold site-stratified CV (preregistration §354 unchanged) on per-cell task distribution. Train LR on 4 folds, predict on 1; record predicted mode per task.

**Cross-cell (LOCO — Leave-One-Cell-Out)**: train LR on 5 cells (~1000 tasks), test on 6th cell (~200 tasks); repeat 6 times. Reports cross-cell generalization with paired bootstrap CI. **This is the paper §6 main number source** per Q4 user decision 2026-05-16.

**Archive sim (development sanity, not paper-grade)**: repeated stratified 5-fold × 10 repeats (50 train-test pairs) on archive cls+red B0 — supplementary table only, not main §6 claim.

## §5 v6 → v7 Δ (changes summary)

| Aspect | v6 | v7 |
|---|---|---|
| Routers in paper-1 §6 | L1 (LR) + L2 (cycle + phantom-verbose) cascade | **L_learned only** (single LR) |
| Paper-1 hypotheses | H9 + H10 + H11 (3-test Holm) | **H10 only** (singleton) |
| Phase 1a router conditions/cell | 6 per-mode initial_mode | **1 learned with obs_mode=learned sentinel** |
| Phase 1a total conditions | 72 (36 baseline + 36 router) | **42 (36 baseline + 6 router)** |
| Phase 1a wall-clock estimate | 2-4 weeks A100 | **1.5-2.5 weeks A100** |
| router.py infra | `safe_fallback_target` + `latch_after_fallback` (v6 cascade) | Same code retained but paper-1 unused; paper-2 forward use |
| conditions.py enum | `phase1.variant: baseline\|router\|both` | + `phase1.router_kind: learned\|cascade` (learned default) |
| Paper-2 deferred items | mechanism §5 + M3/M4 ablation + cross-family routing | + **L1+L2 cascade + H9 rule-based + H11 cascade composition** |
| M3/M4 advisor authorization | Required (cascade absorbed M3/M4 into paper-1) | **Not needed** (M3/M4 stay paper-2 as originally planned) |
| Title | "Phantom Routing Space and Cost-Aware Cascade Routing for VLM Web Agents" | **"Phantom Routing Space and Cost-Aware Routing for VLM Web Agents"** (drop "Cascade") |

## §6 Paper-1 contribution shape (final v7)

1. **Phenomenon**: Phantom routing space discovery (3 arms + 4-fold drop-in property cost / latency / signal / drop-one oracle) — paper §1 + §4
2. **Design**: Learned router over phantom-augmented mode set with site-asymmetric viability empirical finding — paper §6
3. **Statistical**: H10 Pareto non-dominance on (Cost, SR) + latency dominance secondary — paper §6 + Appendix

## §7 Paper-2 forward scope (post-paper-1 publication)

- M1-M4 module ablation (originally Phase 3, deferred)
- L1+L2 cascade router (H9 + H11 estimand specs retained in preregistration §C as forward stubs)
- Mechanism analysis (§5 patching / layer probe / logit lens / SAE — paper-1 mechanism deferred 2026-05-14)
- Cross-VLM-family routing transfer (advanced learned router with capability tier interactions)
- Step-level reactive verbose triggers (paper-2 expanded with proper closed-loop empirical validation)

## §8 Execution checklist (this round)

- [x] v7 spec landed (this file)
- [x] `p79/experiment/conditions.py` `phase1.router_kind: learned\|cascade` enum added; learned emits 1 cond/cell
- [x] preregistration §C: H9 + H11 marked paper-2 DEFERRED; H10 singleton family; Appendix A 2026-05-16 v7 entry
- [ ] `scripts/analysis/l1_archive_simulation.py` repeated stratified 5-fold × 10 repeats option (Q4 fix)
- [ ] `scripts/queues/queue_phase1_router_paper_grade.sh` new queue (or env override path)
- [ ] Chronicle §154 walk-back entry
- [ ] Phase 1a launch — baseline pass first (Launch 1), router pass after (Launch 2)
