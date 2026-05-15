---
type: issue
category: blocker
status: blocked
priority: high
action: 等 advisor email lock K_h1/K_h3/TOST + A100 SSH verify → launch 16-cell on A100 (~3-5d wallclock)
created: 2026-05-03
updated: 2026-05-06
---

# 16-cell phantom rerun (post-Phase-A bug fix)

> **Filename legacy** `issue_14cell_phantom_rerun.md` — scope updated 5/5 student decision: **16 cells** (was 14 pre-sync default). Filename retained to avoid breaking 4 cross-doc references (cell_b1_cls_pprompt / cell_b1_red_pprompt / issue_paper_grade_rerun_5cells / issue_b1_gpu_contention / PLAYBOOK §1).

Paper main analysis only uses Phase A post-fix data (commit ≥ `3c15cd7`). All currently archived B0/B1 phantom cells are pre-Phase-A and need rerun for paper-grade clean numbers.

## Scope (✅ student-decided 5/5 post-sync, advisor email witness pending)

**16 cells**:
- B0 × {classifieds, reddit} × {phantom_text, phantom_som, phantom_prompt} = 6
- B1 × {classifieds, reddit} × {phantom_text, phantom_som, phantom_prompt} = 6
- B0 shopping × {phantom_text, phantom_som} = 2
- B1 shopping × {phantom_text, phantom_som} = 2 (added 5/5 student decision for cross-capability shop coverage)

K_h1 threshold: 0.75 → ≥ 12/16 cells pass; K_h3: 0.67 → ≥ 11/16. See `preregistration.md` line 203.

## Compute path (5/6 update — A100 unblock)

| Path | Wallclock | Cost | Status |
|---|---|---|---|
| ⭐ **UCL Condenser A100 dedicated** (VM `a100-jiaming-test` @ `10.134.51.2`) | ~3-5 d (B1 4B ~10GB fits 40GB w/ headroom; cell-parallel feasible) | $0 (UCL allocation, NOT student-funded) | ✅ operational 2026-05-14 (PyTorch smoke test passed) |
| DGX shared (fallback) | ~3 weeks (seonglae sweep contention) | $0 | available |
| RunPod 4090 (deprecated by A100) | ~1 week | $70-115 | NOT NEEDED |

⚠️ GPU is A100-PCIE-**40GB**, not 80GB — earlier docs/issues saying "80GB 8× headroom" are wrong. B1 4B still fits comfortably. **VWA reach caveat**: A100 cannot directly reach quark VWA Docker — 16-cell rerun needs either VWA self-host on the VM or Tailscale-to-quark setup (still TODO). See memory `reference_compute_resources.md`.

**A100 unblocks paper writing 1-2 weeks** — see 笔记 §112.

## Blocks

- Paper §4 fresh-data prose (codex #11)
- Paper §5 mechanism prose (codex #13) [partially unblocked by Stage 2 mechanistic finding §111 — case study + asymmetry can be quoted from current data]
- Framing decision rule R1-R5 evaluation (data-conditional hook lock)
- Final paper hook commit
- All downstream paper writing
- OSF DOI upload (also gated on advisor email reply)

## Unblocked by

1. ~~**A100 SSH verify**~~ ✅ done 2026-05-14 — A100 operational (VM `a100-jiaming-test` @ `10.134.51.2`), compute blocker eliminated. Remaining: VWA reach from A100 (self-host or Tailscale).
2. **Advisor email reply** to `advisor_sync_5_5_followup.md` Q1-Q11 — locks threshold (K_h1=0.75 / K_h3=0.67 / TOST δ=1.0pp) per `preregistration.md`
3. Pre-registration `status: locked` flip — happens after (2)

## Post-rerun pipeline

`make analysis [FAST=1]` 一条命令 regen:
- `phantom_lift.md` (Holm/BH/Bonf/TOST + H3 structural)
- `meta_phantom_lift.md` (DerSimonian-Laird RE pooled + I²)
- 13 figures including `fig_forest_drop_one.png` / `fig_meta_forest.png` (Hero+Ablation hierarchy) / `fig_phantom_structure_venn.png` (paper §1 centerpiece)

Then framing rule R1-R5 fires → paper hook locks → codex #11/#13 prose.

## Refs

- `docs/checkpoints/pre_run/preregistration.md` (`data_lock_until: <pending 16-cell rerun completion>`)
- `docs/checkpoints/advisor_sync_5_5_followup.md` (Q1-Q11 邮件主体)
- `docs/checkpoints/advisor_sync_5_5_outcomes.md` §A.8 (compute path, Tier 0 A100)
- `docs/checkpoints/实验笔记.md §112` (A100 allocation chronicle)
- ~~`docs/reference/RUNPOD_ONBOARDING.md`~~ (deprecated by A100)
