---
type: issue
category: blocker
status: active
priority: high
action: 一次性 launch 5 cells (B0 dom/som/vision + B1 dom/som) on shopping site (post-A100 SSH verify)
updated: 2026-05-06
---

# Pending paper-grade rerun: B0/B1 × {DOM, SoM, Vision} × **shopping**

**Scope**: shopping site **baseline modes only** (DOM / SoM / Vision). NOT phantom modes.

bugs 累积修完后一次性 re-run (含 §105 swatch radio fix). 已 confirmed 影响: §105 swatch radio (DOM+SoM, Vision 不受). 触发: 用户 stop debug → 一次性 launch.

## Distinction from `issue_14cell_phantom_rerun.md`

This issue covers **5 baseline cells on shopping** triggered by §105 swatch radio fix (DOM+SoM bug).

`issue_14cell_phantom_rerun.md` covers **16 phantom cells** (filename legacy, scope updated 5/5 to 16) triggered by Phase A 4-cluster bug fix (commit `3c15cd7`). Different bugs, different cell scopes, **both now unblocked on UCL Condense A100 dedicated** (allocated 5/6, pending SSH verify, see 笔记 §112) — RunPod path deprecated.

Both can be cell-parallel on A100 (80GB VRAM, 8× B1 4B headroom).

## Compute path (5/6 update)

⭐ **UCL Condense A100 dedicated** (allocated 5/6) — primary. RunPod path deprecated.

## Refs

- `docs/checkpoints/_status/issues/issue_14cell_phantom_rerun.md` (16-cell phantom rerun)
- `docs/checkpoints/advisor_sync_5_5_outcomes.md §A.8` (compute path Tier 0 A100)
- `docs/checkpoints/实验笔记.md §112` (A100 allocation chronicle)
