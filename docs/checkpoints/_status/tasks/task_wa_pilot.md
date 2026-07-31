---
type: task
status: active
priority: P2
horizon: next
order: 2
blocker: ""
eta: "✅ **B1×reddit×6mode 已齐** (2026-07-31 15:01 收尾, 6 格 × 104/104 ep, sentinel OK, noise_rate 全 0)。SR: dom/P-text/P-prompt 各 17 · som 14 · P-SoM 12 · vision 10。🔄 **B0×reddit×6mode 全量 104 重新发车 07-31 19:54 UTC** (run `B0_dom_wa_reddit_20260731_195425_..._R10765`, FORCE_NEW=1, 真 PID 3242478; 早期哨兵实证首个 task=**27** ≪ 581 ⇒ 确为全量)。**ETA ~4-5.5 天 (08-04~08-06)** — B0 走 proxy ~24s/step, task_27 跑满 30 步耗时 ~13min; 比 B1 (本地 GPU, 3.9 天) 慢。⚠️ 可能跨过 08-05, 但 WA 不在两篇稿关键路径上。

**前一次 ABORT 的根因 (已修)**: 07-31 16:54 chain abort (发车 36min, B-1834 episode-count gate 正确拦截 episodes=10 != expected=104)。**config 继承层不对称** — B1 六格 07-27 被改成继承 `exp_v2_wa_full_reddit_base.yaml` (full 104, 注释「pilot base kept intact」), B0 六格未改仍继承 `exp_v2_wa_base.yaml` 的 10-task pilot 采样 (task_ids.reddit=[581,584,597,598,607,635,641,652,715,729])。**user 2026-07-31 裁定跑 104** (覆盖度证据须与 B1 可比) → B0 六格已切 full base + 解析实证 B0/B1 一致 (`{__delete__: true}` sentinel 移除 reddit 键); 失败 run 改名 `_ABORTED_pilot10_config` 留证。**教训: 逐文件 diff 查不出跨层继承分叉 — 发车前要验 `load_experiment_config()` 的解析结果, 不是文件内容**。⚠️ **Jaccard 注册预测的 estimand 我报错过, 已更正**: 注册 PRIMARY = 10-task × **5-mode**(vision 排除) mean-pairwise = **0.690**, 落在 ≤0.5 与 >0.7 之间**未给出判定**, 且样本仅 2-4 元集合 (单 task 翻转即从 1.0 掉到 0.5) 几无分辨力; 我初版报的 104 全量 × 6-mode (0.286-0.611) 是 exploratory 非注册量 → 见笔记 §405.6 更正表"
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
