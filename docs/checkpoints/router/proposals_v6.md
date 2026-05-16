---
name: proposals-v6-pareto-cascade
description: v6 router design — Pareto-cascade framework. Resolves v5 4 P0 (Cost-Aware title mismatch / L2 verbose category error / Phase 1a router protocol gap / site-conditional L1 honesty) via 4 locked decisions from 2-round cross-AI stress + user Q&A.
metadata:
  type: project
  scope: paper-1 §6 router architecture
  status: LOCKED 2026-05-16 — 4 fundamental decisions confirmed
  parent: docs/checkpoints/router/proposals_v5.md
  evidence:
    - docs/checkpoints/router/l2_partial_traj_auroc_2026-05-16.{md,json}
    - docs/checkpoints/codex_outputs/router_v5_FINAL_prefire_2026-05-16_110112.md
    - docs/checkpoints/gemini_outputs/router_v5_prefire_2026-05-16_110112.md
  supersedes: proposals_v5 (single-axis SR + universal L2 verbose + Phase 1a 36-cond ambiguity + Cost-Aware title mismatch)
---

# v6 router design — Pareto-cascade with phantom-only verbose layer

> **LOCKED 2026-05-16** after round-2 pre-fire cross-AI stress (codex + gemini) surfaced 4 P0 issues + user Q&A locked resolution path. v6 is the executable spec for Phase 1a 72-condition launch + paper §6 prose final.

## 4 Locked Decisions

### D1 — Pareto framing (resolves P0-8 Cost-Aware title vs SR estimand)

**Decision**: 2D Cost × SR primary Pareto; Latency dominance check secondary.

**Rationale**: gemini #1 + codex #1 双 AI 标 paper title "Cost-Aware Routing" 但 H9/H10 only-SR estimand = false advertising. Pareto dominance test (not Lagrangian) avoids δ_cost weight pinning and matches FrugalGPT (Chen 2023) standard cost-quality framing.

**New H9 / H10 / H11 formulation** (preregistration §C amendment in-progress):

- **H9** — Rule-based router R9 is **Pareto-non-dominated** on (Cost, SR) plane wrt baselines {DOM, SoM, Vision, Phantom_SoM, Phantom_Text}. Test: paired bootstrap on (ΔCost_i, ΔSR_i) per task; reject H0 if no single baseline dominates R9 in 95% of bootstrap samples per cell. Pooled FE meta across 6 cells.
- **H10** — Learned router R10 same Pareto test.
- **H11** (NEW) — Cascade R9+R10 Pareto-non-dominated by either component alone.
- **Latency check**: secondary table reports Latency ± 10% wrt each cell's best-single-mode-latency. Pareto claim conditional on latency not exceeding +10% threshold.

### D2 — L2 trigger phantom-only verbose + universal cycle (resolves P0-1 category error + L2 partial-AUROC empirical evidence)

**Decision**: Verbose-confidence trigger fires ONLY when current_mode ∈ {phantom_text, phantom_prompt, phantom_som}. Cycle-detection trigger (max_repeat_streak ≥ 3, url_revisit_count ≥ 4) universal across all modes.

**Empirical anchor** (`l2_partial_traj_auroc_2026-05-16.md`):

| Mode | k=5 partial AUROC (range across cls/red) | Viable (≥0.65) at k=5 |
|---|:---:|:---:|
| phantom_som | 0.654-0.690 | ✅ all cells |
| phantom_text | 0.644-0.716 | ✅ all cells |
| phantom_prompt | 0.772 (red only) | ✅ |
| **dom** | 0.644-0.722 | partial (1/2 cells) |
| **som** | 0.625-0.681 | partial (1/2 cells) |
| **vision** | 0.593-0.648 | ❌ 0/2 cells |

**Paper §5 mechanism sub-finding** (NEW, OOB insight surfaced from sanity sim): phantom modes' verbose signal saturates faster (k=5 AUROC matches full-episode within 0.02-0.05), suggesting truncated-input commits agent decisions earlier in episode. Hypothesis: phantom modes provide narrow decision space → fewer plausible actions → confidence aggregates faster.

### D3 — Phase 1a 72-condition sequential (resolves P0-7 execution-substrate gap)

**Decision**: Phase 1a expanded to 72 conditions = 36 baseline (no-router) + 36 router-variant (v6 cascade).

**Generation protocol** (`p79/experiment/conditions.py` update):
- 36 baseline: existing Phase 1a × `router_on=False` (unchanged)
- 36 router-variant: same (site × baseline × mode) tuples × `router_on=True` × router config `v6_pareto_cascade.yaml`

**Wall-clock budget**: 2-4 weeks DGX (vs 1-2 weeks original Phase 1a). Advisor budget approval required.

**Statistical comparison**:
- Per-task paired delta: `(ΔSR_t, ΔCost_t) = (SR_router(t) - max_m SR_baseline_m(t), Cost_router(t) - min_m Cost_baseline_m(t))`
- Per-cell Pareto bootstrap: 1000-resample on paired deltas; report 95% confidence region on (Cost, SR) plane
- Pooled FE meta across 6 cells with Agresti-Coull SE (P0-9 fix)

**Site/account contamination prevention**: same `reset_before=True` hard rule; conditions execute sequentially (cls + red × B0 XOR B1 XOR B2 × 12 conditions × 2 router-states). No parallel runs same site (project hard rule).

### D4 — Site-conditional L1 honest reframe (resolves P0-2 + P1-5 marketing spin)

**Decision**: L1 reframed as **single LR with site as feature** (NOT "site-conditional learned router"). LR may learn near-degenerate function on red (predict phantom_som majority); this is **disclosed as empirical observation**, not architectural design.

**Paper §6 prose template**:

> "Layer 1 is a multinomial logistic regression with features (site, capability_tier, has_reference_image, intent_color_regex, intent_compare_regex, intent_token_count, axtree_element_count) and oracle-best-mode target. Per-cell prediction distribution may collapse toward majority class on text-dominated sites (we observe red baseline 87% dom-winnable on archive; LR predicts dom on red with ~85% frequency, matching always-phantom_som default within ±1pp). On visual-rich sites (cls) the LR learns site-asymmetric mode preferences (Variant B archive +2.56pp over always_phantom_som). We report cell-stratified L1 SR; **L1 cell-level viability is an empirical question, not an architectural guarantee**."

## §1 Architecture diagram (v6)

```
task arrives
    ↓
[L1] Multinomial LR (site as feature)
    inputs: site, capability_tier, has_ref_image,
            intent_{color,compare,search,nav}_regex,
            intent_token_count, axtree_element_count
    output: initial_mode (cell-stratified expected accuracy);
            may degenerate to phantom_som on text-heavy cells
    ↓
agent runs in initial_mode (step 0..k-1)
    ↓
[L2 Cycle] Universal cycle detection (step ≥ 3)
    triggers: max_repeat_streak ≥ 3 OR url_revisit_count ≥ 4
    action: switch to phantom_som (safe fallback, latch); L2 disabled rest of episode
    ↓
[L2 Verbose] Phantom-mode-only verbose trigger (step ≥ 5)
    fires only if current_mode ∈ {phantom_text, phantom_prompt, phantom_som}
    AND step ≥ 5 AND prefix-5 mean verbalized < th_verb_per_mode
    action: same as cycle (phantom_som fallback, latch)
    ↓
agent continues in fallback_mode (single latch — no second switch)
```

## §2 Trade-offs honestly disclosed

| Trade | Cost | Benefit |
|---|---|---|
| Phantom-only verbose (vs universal) | DOM/SoM/Vision modes don't get verbose tail-rescue | Avoids category error on AUROC anchor; 5/5 phantom cells viable at k=5 |
| Cycle trigger universal | Cycle metric also episode-aggregate AUROC (≥0.7 in 7-8 cells full-episode); partial-traj AUROC not yet measured | Cycle is mechanistically observable at step level (max_repeat is just counting), so step-level use is at minimum operationally sound even if AUROC magnitude shifts |
| Step ≥ 5 verbose trigger | Agent burns 5 steps before tail rescue (sunk cost ~33-62% of typical 8-15 step episode) | Per-mode AUROC 0.65+ on phantom modes at k=5 |
| One-shot latch (no second switch) | After fallback, no further adaptation if phantom_som also fails | Prevents oscillation; clean statistical semantics |
| Phase 1a 72-cond | wall-clock ×2 vs 36-cond | paper §1 hook + §6 router both land at same time |

## §3 Open verification items (pre-Phase-1a launch)

1. **Cycle-trigger partial-traj AUROC** — confirm cycle signals (max_repeat / url_revisit) at step-3/5 retain AUROC ≥ 0.65. Need extending `l2_partial_traj_auroc.py` to behavioral signals (1-2h).
2. **per-mode th_verb_per_mode calibration** — compute bottom-decile prefix-5 verbalized threshold per phantom mode from archive. Need 30-min extension.
3. **Pareto bootstrap protocol** — preregister 1000-resample paired bootstrap on (ΔCost, ΔSR); per-cell 95% confidence region computation. Need writeup in preregistration §C amendment.

## §4 Preregistration §C amendment (in progress, separate file)

See `docs/checkpoints/pre_run/preregistration.md` Appendix A 2026-05-16 v6 entry. Key edits:
- §2 H9/H10 estimand: SR-only superiority → Pareto-dominance bootstrap
- §2 H11 NEW: cascade router Pareto-non-dominance vs L9 OR L10 alone
- §C router formula: spelled out cycle + phantom-only verbose 2-trigger spec
- Appendix A 2026-05-16 v6 entry: chronicle of amendment + 4 user-confirmed decisions

OSF DOI commit deferred until amendment landed + advisor sync.

## §5 What v6 does NOT yet address (advisor-sync agenda items)

1. **Paper title decision** (Cost-Aware Routing 保留 with Pareto framework, OR rename to "Phantom Space Pareto-Cascade Routing"). Pareto framing makes title defensible but advisor may prefer rename for clarity.
2. **M3/M4 absorption authorization** (P1-22) — v6 L2 cycle + L2 verbose layers correspond to original Phase 3 deferred M3 retry / M4 two-stage. Need advisor explicit sign-off.
3. **Phase 1a 72-cond budget approval** — DGX wall-clock 2x increase.
4. **L1 nested CV viability** — v6 L1 archive sim still shows +2.56pp on cls but with in-fold z-score + invalid-fallback fixes (P1-1/P1-2/P1-11), likely shrinks to +1-1.5pp. Acceptable if reframed honestly per D4.

## §6 Δ vs v5

| Aspect | v5 | v6 |
|---|---|---|
| Statistical target | SR superiority only | Pareto (Cost, SR) dominance + Latency secondary |
| H9/H10 hypothesis form | SR threshold (+1.0pp Holm) | Pareto bootstrap 95% region exclusion |
| L2 verbose scope | universal across modes | phantom-only |
| L2 cycle scope | mentioned | universal, primary trigger |
| L2 step trigger k | ≥3 (universal) | ≥3 cycle, ≥5 phantom verbose |
| Phase 1a scope | 36 cond ambiguous router protocol | 72 cond explicit (36 baseline + 36 router) |
| Paper title coherence | mismatch (Cost-Aware vs SR-only) | Pareto framework matches title |
| Site-conditional L1 | "site-conditional router" marketing | LR with site feature; cell-level viability empirical |
| Fallback semantics | "fallback target phantom_som" undefined in production router | Code: `safe_fallback_target` field + one-shot lockout latch (`router.py` update) |
| New papers contribution | mechanism cascade rebrand | mechanism: phantom modes uniquely router-friendly (cost + signal early-saturation) — paper §5 sub-finding |

## §7 Execution checklist (this round)

- [x] v6 spec landed (this file)
- [ ] Preregistration §C amendment + Appendix A 2026-05-16 v6 entry
- [ ] `p79/experiment/conditions.py` 72-cond support (Phase 1a router variants)
- [ ] `p79/experiment/router.py` `safe_fallback_target` + lockout state
- [ ] `scripts/analysis/aggregate_phase1_prereg_gate.py` Agresti-Coull SE (P0-9)
- [ ] `scripts/analysis/l1_archive_simulation.py` nested CV + invalid-fallback + in-fold z-score (P1-1/P1-2/P1-11)
- [ ] `scripts/analysis/aggregate_cost_electricity.py` B2 branch (P1-13)
- [ ] `scripts/analysis/aggregate_routing_auroc.py` B2 baseline parser (P1-9)
- [ ] `scripts/analysis/figures/fig0c_phantom_lift_bars.py` per-comparison universe annotation (P1-16)
- [ ] Cycle partial-traj AUROC sim extension (`l2_partial_traj_auroc.py` add behavioral cols)
- [ ] th_verb_per_mode calibration table
- [ ] Chronicle §153 append
- [ ] Advisor sync agenda items (D2-D4 finalize)
