---
type: issue
category: blocker
status: blocked
priority: high
action: 等 advisor sync lock + RunPod $200 批准 → onboarding → launch chain
created: 2026-05-03
---

# 14-cell phantom rerun (post-Phase-A bug fix)

Paper main analysis only uses Phase A post-fix data (commit ≥ `3c15cd7`). All currently archived B0/B1 phantom cells are pre-Phase-A and need rerun for paper-grade clean numbers.

## Scope (final cell list to confirm at advisor sync)

Tentative 14 cells (subject to advisor sync decision):
- B0 × {classifieds, reddit} × {phantom_text, phantom_som, phantom_prompt} = 6
- B1 × {classifieds, reddit} × {phantom_text, phantom_som, phantom_prompt} = 6
- B0 shopping × {phantom_text, phantom_som} = 2 (P-prompt deferred for shop?)
- = 14

Final scope confirmed at sync — could be 13 if B0 shop P-prompt deferred, or 16 if B1 shop included.

## Cost estimate

- 4090 dedicated: $0.6/h × ~87-145 h = **~$52-87 actual**
- + 30% buffer (crash/retry) = **~$70-115**
- + ad-hoc probe headroom (~$60: Q3 multi-call + Tier 5 evaluator + diamond shop + §5 ad-hoc) = **~$200 total ask**

## Wallclock

- DGX shared: ~3 weeks (GPU contention with seonglae)
- RunPod 4090 dedicated: **~1 week**
- 后者 unblocks paper writing 2 周

## Blocks

- Paper §4 fresh-data prose (codex #11)
- Paper §5 mechanism prose (codex #13)
- Framing decision rule R1-R5 evaluation
- Final paper hook commit
- All downstream paper writing

## Unblocked by

1. Advisor sync (`issue_advisor_sync_preregistration.md`): RunPod $200 budget approval + cell list final + framing buy-in
2. RunPod onboarding (`docs/reference/RUNPOD_ONBOARDING.md` 7-step playbook)
3. Pre-registration `status: locked` (`preregistration.md`)

## Post-rerun pipeline

`make analysis [FAST=1]` 一条命令 regen:
- `phantom_lift.md` (Holm/BH/Bonf/TOST + H3 structural)
- `meta_phantom_lift.md` (DerSimonian-Laird RE pooled + I²)
- 13 figures including new `fig_forest_drop_one.png` / `fig_meta_forest.png` (Hero+Ablation hierarchy) / `fig_phantom_structure_venn.png` (paper §1 centerpiece)

Then framing rule R1-R5 fires → paper hook locks → codex #11/#13 prose.

## Refs

- `docs/reference/RUNPOD_ONBOARDING.md` 7-step playbook
- `docs/checkpoints/preregistration.md` (data_lock_until)
- `docs/checkpoints/ADVISOR_SYNC.md` §3 RunPod budget
