---
type: task
status: pending
priority: P1
horizon: next
order: 1
blocker: "✅ 已解除 2026-08-03 — WA-B0 chain 今早 06:24 跑完释放 A100; 且 `local_mimo` backend 此前只在 DGX 有、A100 缺失(这就是 B3_som_classifieds_20260803 崩在 `Unsupported backend type: local_mimo` 产出 0 episode 的根因), 今日 rsync 已到位"
eta: "floor pilot 08-04 起 (几小时, 10 task × 15 step; ⚠️ 建议抬回 25×30 与 B1/B2 对齐否则 floor 判读无可比基线); pilot 通过则 12 conditions fire (A100, ≈ 2-2.5 周)"
detail: docs/checkpoints/paper_planning §19 2026-07-16 row
created: 2026-07-16
updated: 2026-07-31
---

# B3 = MiMo-VL-7B 跨族扩展 (post-submission 第一优先)

学长拍板 2026-07-15 ("MiMo 先行, 之后其他模型") + user 排期确认 2026-07-16。

**目的**: 堵 "cross-family 证据 = 单个地板模型" 攻击 (自预测于周会 brief; B-1284 已压 R2)。
强 7B 若复现结构 gates → rebuttal/camera-ready 跨族翻身。副产品: MiMo 格子若可训 →
paper-2 router 获得第二族训练基质 (强模型→标签充足假说检验, 笔记 §373)。

**Scope**: cls + red × 6 modes = 12 conditions, A100 serial。不进 paper-1 prereg
(6 cells locked); extension/rebuttal 数据。

**排期**: ① 适配 (prompt/parse/部署, 参照 B2 适配史 — 当年数周, 有经验应更快, DGX 先行)
② 8 月上旬 fire (rebuttal 日历倒推: AAAI author feedback 大概率 10 月)。

**首个动作**: codex 任务书 — MiMo backend 探测 + B2 适配清单复用 (待派发)。
