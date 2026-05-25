---
type: audit
ref: B-1860
title: Qwen 0-1000 坐标契约
status: done
priority: P0
effort: "~1.5h (applied 2026-05-25)"
phase: done
blocker: ""
post_fire: "dom/som digest 重扫 (3-domsom → 3-domsom-b1860coord) + B0 vision /diag 验证"
---

# B-1860 · Qwen 0-1000 坐标契约 (APPLIED 2026-05-25)

**根因**: Qwen3-VL 原生 0-1000 坐标范式 vs P79 栈全假设 [0,1]/viewport-pixel
(prompt + `validate_action_detailed` + `vwa_wrapper.py:614/616` `/1280,720`) →
① 0-1000 标 normalized → validate reject (507 parse error = 13.6%)
② 0-1000 标 pixel → wrapper 错归一化 (104 misclick 藏 SR)。
铁证: y_max=972 > 720 viewport + `422 = 0.422×1000` (R3671 /diag, 笔记 §285)。

**影响**: vision SR 13.84% 非 clean (coordinate-scaffold 主导: 48% ep≥3pe / 99% fail /
cap 3-5 提前处刑), 不可跨 mode 比 SR / 不能只 disclosure。

**Fix** (救 format 层不救 grounding 层): 逐维度 contract ≤1.1→[0,1] / >1.1→/1000;
single-source `normalize_coordinate_pair` + wrapper /1000 映射 + validate[0,1000]+legacy[0,1]
+ prompt 改 0-1000 删 coordinate_type + cap 只在归一化后计数。@75b37bb (branch
fix-coordinate-contract-b1860); codex verify 4 fix-impl bug 补 @3ea598e (V-F1 true_oob
fail-closed no-op / V-F2 negative coord OOB / V-F3 dead_zone counter / V-F4a annotate import
normalizer); 40+5 test pass。

**Per-model probe**: B0 铁证 / B1 probe DONE (底部项 (728,920) 920>720 = 0-1000 同 B0) /
B2 Gemma = normalized [0,1] (model-agnostic G-F1 defused)。

**APPLIED 2026-05-25**: merge `d977006` fix→diag + amendment 05 witness (tag
`prereg-amendment-05-b1860-coord-contract-20260525` + OSF kv9sf, 披露 instruction-strictness
relaxed + HARKing) + B-1861 watchdog ntfy fail-safe 同期。cls chain 从 vision restart
(手动 queue_chain 16 cells; B0 dom R31194/som R9725 保留旧 coord-invariant 代码, vision 起
新代码; ⚠️ NOT queue_phase1 launch — 它 FORCE_NEW 重跑全部)。R3671/R1099/R19776 A100-archived。
WHY → 笔记 §285-288。
