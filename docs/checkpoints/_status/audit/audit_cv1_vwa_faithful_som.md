---
type: audit
ref: CV-1
title: VWA-faithful SoM control arm (interactable-only marks)
status: open
priority: P2
effort: ~0.5d impl + B1 cls pilot
phase: paper-finalize
blocker: cls 被 Phase 1a fire 占用 (hard rule 同 site 单 baseline); 非 gate-blocking; P1→P2 降级 2026-06-05 (dom 对照拆掉 claim ②, 见 §316 follow-up)
---

# CV-1 · VWA-faithful SoM control arm

**WHY (实证背景 → [[实验笔记]] §316)**: 我们的 `som` 走 `parse_accessibility_tree`,**给全部 AXTree 节点编号**(含非互动 StaticText/heading) — 为 2x2 ablation 保 DOM/SoM/phantom 共享同一 id 空间。代价: 模型可以误点一个"编了号但点不动"的 mark,而 VWA 原生 `image_som` 把非互动元素标 `[]` 无 id (`processors.py:940`) → 结构性屏蔽。**实测 wasted-click 率 (cls, emit-id join SOM_MARKS role)**: 纯 StaticText 误点 B0 2.8% / B1 3.4%; **真正浪费 (非互动 role + 无导航 + locator 没爬到可点祖先) B0 2.9% / B1 14.6%** — model-differential, 弱模型 B1 主力是 heading 146 次 (99% no-op) + RootWebArea 23 次。

**三条 claim 受影响程度** (§316 + dom 对照 follow-up 2026-06-05): ① phantom 主线 (som vs phantom_som) **安全** — 同一编号方案差分抵消; ② **跨模型 confound 已基本拆掉 (→ 本 note P1→P2 降级根因)** — dom **也**全节点编号 (纯 StaticText 点击率 dom 3.6-3.9% ≈ som 2.8-3.4%, 对称), 项目内 B0/B1/Gemma 同管线比 confound 不偏; B1 顶绳子照样拿 **+7.07pp ×2.08** SoM 涨幅 (≈ B0 **+9.82pp ×1.56**) → SoM 收益 capability-robust, 浪费绳子没吃掉收益; ③ **唯一残留** = 绝对 SR 对标外部 canonical VWA (interactable-only) 偏低, limitation 脚注。

**定位 (2026-06-05 降级 P1→P2)**: 不再防 load-bearing 跨模型公平性 claim (已被 dom 对照拆掉); 现为 **绝对-SR-vs-文献保真 + 便宜 robustness check**。仍值得做 (把 limitation 脚注变 "已验证"), 但可安心排 paper-finalize 末尾。4-格 canonical SR (epsum vs 官方 condition_summary 224/224 验证): B0 dom 17.41%→som 27.23% · B1 dom 6.57%→som 13.64% (n=198/224, fire 在写, provisional + 单 run)。

**WHAT**: 加一个 `som_vwa_faithful` sentinel obs mode — SOM_MARKS 只保留 VWA `Interactable=True` 的 mark, 非互动 role 不编号 (匹配 canonical VWA image_som)。其余 (prompt / 图 / dispatch) 与 `som` 一致。在 **cls 重跑 B1** (最敏感弱模型)。

**HOW** (两条实现路径):
- **便宜 (role-allowlist, ~10-20 LOC)**: 在 `p79/experiment/som.py` `_extract_text_marks` / `build_som_text_from_obs_text` 后加一个 role 过滤 (allowlist = link/button/textbox/combobox/checkbox/radio/menuitem/searchbox/...), 仅 `som_vwa_faithful` mode 启用。近似 VWA 但非 DOM-clickability 精确复刻。
- **严格 (VWA Interactable flag)**: 走 VWA `get_page_bboxes` 的 `Interactable` 列 (`external/visualwebarena/browser_env/processors.py` image_som 路径) 作 mark 过滤源 — DOM-based, 精确匹配 canonical。
- ⚠️ 口径: 便宜路径的 14.6% 估计用 `target_tag=None + 无导航` 当 "VWA 会标 `[]`" 代理, 非 Interactable flag 精确复刻 — 若要 reviewer-proof 用严格路径。

**Acceptance (两种结局都赢)**:
- B1 SR **基本不变** → 直接证明浪费绳子对 SR 影响可忽略 (dom 对照已**间接**证 confound 对称抵消; control 给**直接**证据), 绝对-SR 偏低脚注可弱化/删。
- B1 SR **上升** → 量化变体代价, 报数字进 limitation, 显严谨。

**便宜兜底 (0 重跑)**: wasted-click 率本身当 metric 报 (本 session 表已算), 并证 phantom delta 在剔除 wasted-click step 前后稳定 — 挡 ② 大部分火力, 但不如 control 硬。

**Trigger**: Phase 1a fire 释放 cls 后 (hard rule 同 site 同时单 baseline)。先 10-task pilot 探 SR shift, 若 shift 显著再跑 full B1 cls。

**不阻塞**: phantom 主线 paper-grade 数据不受影响 (① 抵消); 这是跨模型/绝对 SR 的 submission-hardening control。
