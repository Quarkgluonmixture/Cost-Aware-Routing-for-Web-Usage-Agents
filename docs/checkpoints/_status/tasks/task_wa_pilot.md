---
type: task
status: active
priority: P2
horizon: next
order: 2
blocker: ""
eta: "✅ **B1×reddit×6mode 已齐** (2026-07-31 15:01 收尾, 6 格 × 104/104 ep, sentinel OK, noise_rate 全 0)。SR: dom/P-text/P-prompt 各 17 · som 14 · P-SoM 12 · vision 10。⛔ **B0×reddit chain 07-31 16:54 ABORT** (发车 36min 后; B-1834 episode-count gate 正确拦截 episodes=10 != expected=104)。根因 = **config 继承层不对称**: B1 六格 07-27 被改成继承 `exp_v2_wa_full_reddit_base.yaml` (full 104, 注释「pilot base kept intact」), 而 B0 六格未改, 仍继承 `exp_v2_wa_base.yaml` 的 10-task pilot 采样 (task_ids.reddit = [581,584,597,598,607,635,641,652,715,729])。**待裁定: B0 跑 10 (注册设计) 还是 104 (与 B1 可比, 覆盖度证据需要)**。⚠️ **Jaccard 注册预测的 estimand 我报错过, 已更正**: 注册 PRIMARY = 10-task × **5-mode**(vision 排除) mean-pairwise = **0.690**, 落在 ≤0.5 与 >0.7 之间**未给出判定**, 且样本仅 2-4 元集合 (单 task 翻转即从 1.0 掉到 0.5) 几无分辨力; 我初版报的 104 全量 × 6-mode (0.286-0.611) 是 exploratory 非注册量 → 见笔记 §405.6 更正表"
detail: preregistration.md §8.8 B-1296 registered prediction
created: 2026-07-16
updated: 2026-07-31
---

# WA 50 任务 pilot — 兑现注册预测 (插空任务)

**目的**: prereg §8.8 (B-1296) 注册了 WA 50-task pilot 的可证伪预测
(Jaccard ≤0.5 vs >0.7 区分性预测)。几天算力兑现一个 pre-registered falsifiable
prediction — 无论方向都是干净科学增量 + 信誉资产。

**Scope**: WebArena 50 任务 × 6 modes, B1 本地 (无 proxy 依赖, outage 免疫)。
适配面: WA 无 reference images / evaluator 差异 — 先探 harness 兼容性。

**排期**: 8 月插空 (B3 适配期或 fire 间隙); 与 B0 replicate 附录同为机会性任务。
