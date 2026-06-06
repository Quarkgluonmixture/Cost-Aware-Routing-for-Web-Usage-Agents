---
type: audit
ref: B-1869
title: locator walk_fail fallback reports action_success=True
status: deferred
priority: P2
effort: 2h
phase: post-fire
blocker: post Phase 1a fire complete (analysis-layer fire-immutability)
---

# B-1869 · locator `walk_fail` fallback reports `action_success=True`

`/diag B1 ptext` (R933, 笔记 §321) ground-truth: 视觉盲 agent emit 不可解析 `type [1]` →
`locator_route_meta={success:False, error:'walk_fail:no_input_within_walk'}` + `element_bbox=[0,0,10,10]`
退化 → 但 step 报 `action_success=True`。`action_success` 未 gate 于 `locator_route_meta.success`。

**Impact**: action_success over-count, **axis-2 amplified** (ptext P4=41 / vision=0, numbered-ref+视觉盲
产更多 unresolvable ref) → §306 "ptext action_success 77.3% < dom 79.4%" 是 conservative (真 P-text
劣势更大)。measurement-layer only; **不影响 SR/eval gate** (eval outcome-based)。

**Trigger**: post Phase 1a fire (NO mid-fire change — fire-immutability)。
**Acceptance**: 决定 (a) gate action_success on `locator_route_meta.success` 或 (b) emit `locator_unresolved`
flag 供 aggregator filter; cross-tab via `aggregate_locator_route_metrics.py` (已 count walk_fail per cond);
action-level paper number 加 walk_fail-fallback caveat 或 recompute。

Cross-link: master_bug_catalog B-1869 · digest `B1_phantom_text_classifieds_diag_digest.md` §3/§9 ·
freeze-step `WALK_FAIL_DEGENERATE` diag P-rule (笔记 §321 §7) · B-114 (sibling fallback-masks-failure)。
