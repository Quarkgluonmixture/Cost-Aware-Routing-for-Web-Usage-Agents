---
type: task
status: blocked
priority: P0
horizon: next
order: 2
blocker: "Pass-1 complete + LR pipeline land"
eta: "2026-07-08 (D7 H10 verdict; 含 Stage 1→3 fold-aware bundle + 6 cond fire)"
detail: phase1_plan
created: 2026-05-22
updated: 2026-06-10
---

# Pass-2 learned router (6 cond) ⭐⭐

Post-Pass-1: LR training pipeline (Pass-1 outcomes → oracle label matrix → entropy
defer gate → per-cell LR heads → artifact smoke) → `queue_phase1_router_paper_grade.sh`.
(B-1671 launch-pass2 raises until LR pipeline lands.) 详 [[phase1_plan]] §B-router + §C.
