---
type: audit
ref: B-1858
title: Run-to-run SR 方差 — clean 3rd som replicate
status: gate-blocking
priority: P0
effort: "(a) gate-blocking · (b)-(d) post-fire"
phase: gate
blocker: "需两健康 B0 som run (R2815 是 B-1848 wedge, 不能用)"
---

# Repro-replicate · Run-to-run SR 方差 (机制确定 + 量级 OPEN, 笔记 §282 / B-1858)

**机制 (实证)**: VWA element-ID 非确定 (`processors.py:532` CDP/树序号) → 同页面 obs
byte-diff (仅 `[id=N]` 变, content-diff 证结构稳) → temp=0 模型对 ID token 敏感 →
action churn → 轨迹分叉; 截图 = 视觉锚 (SOM step-0 action 90% > DOM 73%)。

**量级 OPEN — 无干净 replicate** (两 pair 都 confound): DOM (R31194 vs R9755)=regime
(fresh vs pre-B-1839-stale), 0.4pp 对称 = regime≈0 proxy 非纯 replicate; SOM (R9725 vs
R2815)=R2815 是 B-1848 wedge run, +7pp 非对称 (19/6 p=0.015)。

**codex gpt-5.5 cross-AI re-derive** (→ `codex_outputs/repro_directionality_causes_2026-05-24.md`):
完整 3 源 = ②ID-churn 主导(20/25) + ①provider-nondet 真实(4/25) + ③site-state 少量
(1/25 task145); 方向 = chance + post-hoc (template 15/5/1 p≈0.041)。**撤回**早前
"0.4pp clean floor / 对称 / gate robust"。

**gate 风险 OPEN**: som run-to-run 可能几 pp → 接近 phantom 1.7-3.3pp → 可能威胁 gate;
dom<1pp 不背书 som。

**Forward**:
- (a) 🔴 **clean 3rd som replicate** (两健康 B0 som) = **gate 前置** (估真 run-to-run std + 是否 directional/复现)
- (b) §4 disclosure 用 replicate 数非 R2815
- (c) post-fire replay 切 ①/② + positional-ID fix (B-1858 根除 ②)
- (d) phantom 落地测 step-0 分叉
