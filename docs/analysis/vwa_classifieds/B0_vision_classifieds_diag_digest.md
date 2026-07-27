# B0 vision classifieds — diag digest pointer (dual-run preserved)

> Run-to-run dual-digest preserved per user directive 2026-05-26 (笔记 §297-298 H1 sensitivity 一脉 — vision floor 实证 + paper Risk 6 增量证据). 默认名指针文件, **不含数字** — paper / cross-mode aggregator 拉数据 follow links 到具体 RXXXXX digest, 不直接读本文。


## 0v8. v8 freeze 补记（2026-07-27）— cls 行为**不是**字节不变

`RULESET_VERSION` 升至 **`8-reddit-p41p46-b1890fix`**。该批规则源自 **reddit** discover，但有两处**确实改变了 cls 行为**，
均已逐条定性核实（不是回归）：

1. **B-1890 修复**：`P35`/`P39` 原先 guard 在 `effective_mutating_action_count`，而该字段从未被 runner
   填充、恒为 0 → guard 是 **no-op**，规则比其 docstring 声称的更宽松。v8 改为从 step record 派生突变计数。
   抽查确认被移除的旧命中确实有 6–8 个突变步（即**旧命中是错的**）。
2. **P33 正则扩展**：加入 reddit 的 `/submission_images/` 路径。cls 侧因此 **+1 例**（cls task 233 —— 它的
   `sites` 只写 classifieds，但 intent 实际要求"the characters in the image **on Reddit**"，
   该 episode 真的访问了 `localhost:9999`，旧正则漏检）。

本 condition 的 v8 数字 —— **跨 condition / 跨站聚合请用这一组**：

| 指标 | v8 |
|---|---|
| SR | **25.00%** (56/224) |
| failed + hit | 116 |
| **failed NO-hit** | **52** |
| success + hit | 4 |

v8 新规则 failed 侧: 无；success 侧: 无。
（`P43` 在 cls 上大量命中属预期 —— 它标记"intent 需要视觉 + 该 mode 无页面截图"这一**中性组合**，
并非预测失败；§387.10 实测补上截图的增益 ≈0。）

全部 36 个 canonical condition 现处同一版本 → **cross-mode / cross-site 聚合解锁**。

---
## Canonical pointers

| Run | File | Date | SR | Ruleset | Status |
|---|---|---|---|---|---|
| **R32024 (current canonical, paper-grade Phase 1a fire-3)** | [[B0_vision_classifieds_R32024_diag_digest]] | 2026-05-26 | 24.8% (55/222) | `4-domsomvis-b1860coord` | **active** |
| **R24792 (archive, AMENDMENT_07 前; H1 sensitivity replicate)** | [[B0_vision_classifieds_R24792_diag_digest]] | 2026-05-25 | 24.1% (54/224) | `4-domsomvis-b1860coord` | archived |

## Run-to-run summary (one-line per axis)

- **SR**: Δ=+0.7pp (24.1→24.8) — **non-gating noise**, < dom Δ=+2.2pp (§298)
- **substrate**: parse_error 0.027%→**0.0%** (B-1860 持续 hold + R32024 更干净)
- **benchmark-FP (净)**: 0→0 (R24792 task 40/132 翻案 + R32024 task 192 修正; "vision 失败更纯" 稳态)
- **scaffold-bug**: 0→0 (substrate 干净)
- **success-hit causal rate** ⚠️: 0%→47% (run-to-run **大变** — vision homepage dropdown 真因果模式在 R32024 暴露)
- **vision-only stuck pattern 新发现 (R32024)**: homepage category dropdown 视觉点击系统性卡死 (task 52/100/101/103) → paper §3 efficiency 实证, dom mode 无此模式

## 拆解 (差异来源, AMENDMENT_07 影响 vision = 0)

- **B0 MoE 非确定性** (§242): 字节相同输入 stochastic argmax (主导)
- **per-condition docker fresh restart** (B-1839): cart/listing/comment 空
- **manifest 池微差**: 224→222 ep (2 ep 待 audit, 单一非阻塞)

详 [[B0_vision_classifieds_R32024_diag_digest]] §0 完整对照表 + paper implication 分析.

## Cross-link

- 同 condition 不同 mode: [[B0_dom_classifieds_diag_digest]] / [[B0_som_classifieds_diag_digest]]
- 笔记 chronicle 入口: 实验笔记 §297-§300 (run-to-run noise 拆解闭环)
- paper Risk 6: paper_planning Risk 6 + AMENDMENT_06/07 + phase1_plan §D4
