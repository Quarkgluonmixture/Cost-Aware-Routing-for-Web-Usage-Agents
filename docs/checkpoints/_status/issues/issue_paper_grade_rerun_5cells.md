---
type: issue
category: blocker
status: active
priority: high
action: 一次性 launch 5 cells (B0 dom/som/vision + B1 dom/som) on shopping site
updated: 2026-05-03
---

# Pending paper-grade rerun: B0/B1 × {DOM, SoM, Vision} × **shopping**

**Scope**: shopping site **baseline modes only** (DOM / SoM / Vision). NOT phantom modes.

bugs 累积修完后一次性 re-run (含 §105 swatch radio fix). 已 confirmed 影响: §105 swatch radio (DOM+SoM, Vision 不受). 触发: 用户 stop debug → 一次性 launch.

## Distinction from `issue_14cell_phantom_rerun.md`

This issue covers **5 baseline cells on shopping** triggered by §105 swatch radio fix (DOM+SoM bug).

`issue_14cell_phantom_rerun.md` covers **14 phantom cells** triggered by Phase A 4-cluster bug fix (commit `3c15cd7`). Different bugs, different cell scopes, both blocked on resource availability (DGX GPU contention or RunPod budget).

Both can theoretically be merged into single RunPod onboarding session if scope permits — confirm at advisor sync.
