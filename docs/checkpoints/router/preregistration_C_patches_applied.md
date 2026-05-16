---
type: preregistration-patch-record
status: applied-2026-05-16
purpose: audit-trail record of the 3 router-related patches applied directly to preregistration.md per user instruction 2026-05-16
applies-to: docs/checkpoints/pre_run/preregistration.md (already applied — see Appendix A 2026-05-16 entry)
---

# Preregistration §C patches — applied 2026-05-16

> **All 3 patches were applied directly to `docs/checkpoints/pre_run/preregistration.md`** per user instruction "直接改 preregistration". This file is the **audit-trail record** of what changed and why. Treat as historical reference, not a draft to apply.

## Patch C1 — H9/H10 estimand lock

**Where applied**: `preregistration.md §2` — inserted NEW H9 + H10 PRIMARY family blocks + H9+H10 family-wise correction + rationale block, AFTER `### H3 — Phantom space 2-axis empirical structural claim` and BEFORE `### EXPLORATORY family`. Plus updates to:
- §0 outstanding items: "(ii) paper-1 H9/H10 router estimand details" → marked ✅ LOCKED 2026-05-16
- §0 gating hypotheses list: "H1 + H2 + H3" → "H1 + H2 + H3 + H9 + H10"
- §6 lock H-list: "H1-H3 only" → "H1-H3 + H9-H10"
- Appendix A: new 2026-05-16 entry recording lock event

**Lock content** (canonical lives in preregistration.md §2):
- Estimand: θ_FE = Σ_i w_i × (SR_router_i − SR_best_single_mode_i) over 6 cells, inverse-variance weighted (mirror H1 §2.5 decision "3A" 2026-05-14 — fixed-effects, no τ², no DerSimonian-Laird)
- H9 PRIMARY gate: one-sided FE superiority test H0: θ_FE^(P1) ≤ +1.0pp at α=0.05
- H10 PRIMARY gate: one-sided FE superiority test H0: θ_FE^(P2) ≤ +1.0pp at α=0.05
- H9+H10 family-wise: Holm correction over {H9, H10} test pair, α_family = 0.05
- δ rationale: same as H1 (≈ 2 tasks in N=234, matches per-cell bootstrap SE); fine-calibration via archive G-4 noise SD with rule "raise to max(1.0pp, 2×SD) if observed SD > 0.5pp" — **NB: this rule is statistically suspect, see archive diagnostic verdict 2026-05-16 + paper_planning §19 open question**
- H10 DEFER condition: if archive G-1 entropy < log(2), H10 collapses to {H9} only

## Patch C2 — Anchor-flicker fallback

**Where applied**: `preregistration.md §4` — replaced `**Best-single-mode baseline (H7/H8 anchor)**` row in §4 locked-analysis-choices table with new wording.

**Lock content** (canonical lives in preregistration.md §4 row "Best-single-mode baseline"):
- Anchor = mode with highest mean `success` rate on train fold (split-stratified per §354)
- Anchor-flicker fallback: if archive G-2 gate shows Kendall τ across 100 × 5-fold resamples < 0.7 for any cell, switch that cell's anchor to majority-winner-across-resamples (mode that wins best-single position in ≥ 50% of 100 resamples)

## Patch C3 — Adjusted-SR retirement reflection

**Where applied**: `preregistration.md §4` — same row as C2, appended "Outcome column convention" note.

**Lock content** (canonical lives in preregistration.md §4 row "Best-single-mode baseline"):
- Where the preregistration historically says "adjusted-SR", read `success` (canonical post-§139.8 retirement, VWA submodule `p79-patches` commit `f0c835b`: N/A LLM-judge FP fixed at upstream + N/A tasks excluded at task-load)
- Router pipeline uses `success` from `condition_summary_v2.json` directly, no `compute_adjusted_success` post-hoc layer

## Application sequence (chronicle)

| Step | Action | When |
|---|---|---|
| 1 | router_proposals_v1.md drafted (stress artifact) | 2026-05-16 morning |
| 2 | 3-AI cross-stress (Claude /stress + codex /codex-stress + gemini /gemini-stress) → v2 land | 2026-05-16 mid |
| 3 | user-caught 3 OOB (P0-8 task.category leak / P0-9 hijack triaxis / P0-10 has_ref_image) → v3 land | 2026-05-16 afternoon |
| 4 | user instructs "直接改 preregistration" — C1+C2+C3 applied | 2026-05-16 afternoon |
| 5 | router_archive_diagnostic.py run → H10 DEFER triggered on reddit (entropy 0.606 < log(2)), δ rule flagged statistically suspect | 2026-05-16 afternoon |
| 6 | 实验笔记 §150 chronicle append | 2026-05-16 afternoon |
| 7 | user instructs "router files 进文件夹" → files moved to `docs/checkpoints/router/` | 2026-05-16 afternoon |
| 8 | user instructs "archive 测的 router 有问题" — methodology critique, v4 redesign pending | 2026-05-16 afternoon |

## Outstanding items (advisor sync needed)

- δ_h9 / δ_h10 calibration rule wording — current C1 patch says "raise δ to max(1.0pp, 2×SD)" but archive G-4 SD ≈ 2.2pp would push δ to 4.4pp (overly conservative); statistically this conflates effect-size floor with noise SE. Recommend: keep δ=1.0pp + explicit low-power disclosure
- H10 DEFER actually pre-data triggered (reddit entropy 0.606 < log(2)) — paper §6 router section now H9-only; advisor confirm whether to formally lock H10 → paper-2 deferred OR collect Phase 1a reddit data and re-check entropy
- Archive-derived locks methodologically questionable — see v4 redesign discussion (router_proposals_v4.md TBD)

## See also

- `docs/checkpoints/router/proposals_v3.md` — v3 design spec (P1 capability-blind + P2 test-leak-free)
- `docs/checkpoints/router/archive_diagnostic_2026-05-16.md` — diagnostic verdicts
- `docs/checkpoints/pre_run/preregistration.md §2 H9/H10` + §4 anchor row + Appendix A 2026-05-16 — canonical lock
- 实验笔记 §150 — chronicle
