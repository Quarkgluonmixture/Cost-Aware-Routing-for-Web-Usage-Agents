# /diag digest — B2 × `phantom_prompt` × reddit

*生成 2026-07-27（Tier-1 全量 + Tier-2 深挖）*

> **定位声明**：本 digest 是**单 condition** 的失败归因，不下 cross-mode / cross-model 结论。
> 跨 mode 定量比较须等 reddit 规则批（R1–R8 + H2）落地、`RULESET_VERSION` 升到 `8-reddit-*`
> 并全量重扫后再做（/diag skill「discover-then-freeze」硬纪律）。


## 1. Header

| 字段 | 值 |
|---|---|
| **Run** | `B2_phantom_prompt_reddit_20260723` |
| **Condition** | `phase1_phantom_prompt_router_0` |
| **Site / Mode / Model** | reddit / `phantom_prompt` / B2 = Gemma3-VL `google/gemma-3-4b-it` (local) |
| **Episodes** | 205 |
| **SR** | **0.49%** (1 success / 204 failed) |
| **ruleset_version** | `7-p6p16clsgate-b1860coord` |
| **Tier-1 三子集** | failed+hit 203 · **failed-NO-hit 1** · success+hit 0 |

## 2. Tier-1 规则分布（failed 侧）

| 规则 | 含义 | step-level hits | 命中 episode 数 |
|---|---|---|---|
| `P36` | WALK_FAIL_DEGENERATE | 1450 | 159 |
| `P5` | 感知缺失循环 | 328 | 173 |
| `P31` | budget 耗尽未完成 | 195 | 195 |
| `P14` | URL 自环 | 157 | 121 |
| `P12` | 从不翻页 | 68 | 68 |
| `P25` | 跨站任务跳过其中一站 | 31 | 31 |
| `P10` | 跨步数值记忆失败 | 2 | 2 |
| `P4` | 根节点误操作 | 1 | 1 |
| `P13` | 搜索代替浏览 | 1 | 1 |

**success 侧 fire 的规则**: 无（success 侧 0 命中）

**failed-NO-hit episode（deterministic 盲区）**: [64]

**success episode**: [160]


## 3. Tier-2 深挖

**覆盖范围**：6 ep（no-hit 1 + P36 审计 4 + 唯一 success 1）· 1 sonnet sub-agent

**三分类**：agent-limit 5 · **benchmark-FP 1（唯一的 success）** · scaffold-bug 0 · unclear 0

### P36 因果审计

**真死因**。4 个抽样 episode 全部同一死法：命中一次 walk_fail 后连续 27–30 步原样重复，占该 episode 全部预算的 90–100%。sub-agent 另做了全 205 集结构扫描：**160/205 集（78%）至少命中一次 walk_fail，20/205 集单集内 ≥20/30 步被同一失败点击霸占**，总计 1458 次 step 级 walk_fail（与 Tier-1 的 1450 吻合）。

### 具体发现

- ⭐ **SR=0.49% 是真实能力崩溃，不是测量故障** —— 关键证据是**跨 baseline 的严格单调梯度**：B0(235B) 12.68% → B1(Qwen3-4B) 6.34% → B2(Gemma3-4B) 0.49%。若是 harness/infra 故障，三个 baseline 应**同等程度**失灵，而不是随模型规模/家族精确分级。token/延迟/cost 记账均正常，无 error 字段、无 auth 失败痕迹。
- ⭐ **唯一那个 success（task 160）不可信** → B-1889。**本 cell 修正后真实 SR = 0/205 = 0.00%**。
- **P-prompt 的 SoM-prompt × AXTree-text 组合是设计固有、非实现 bug**：代码确认（`_shared_vl_utils.py::build_mode_prompt_dispatch_table` + `som.py::prepare_observation_for_mode`）phantom_prompt 明确路由到 SoM system prompt + AXTree 原生文本 + 无图，element_id 用的是与 dom 模式**完全相同**的原生 AXTree id（`mark_count=0`，未走 seq 映射）→ walk_fail 与「提示-观测错配」正交，不是 ID 体系混乱导致点了不存在的编号。可归因于该刻意错配的是两个**间接**效应：(a) 无图像通道 → 失去独立视觉线索去发现自己卡死；(b) SoM prompt 反复宣称「你会收到标注截图」而实际没有，可能侵蚀 grounding 校准 —— 所有卡死点击的 confidence 都标 0.95（虚假自信）。**建议作为跨家族鲁棒性差异的证据写进分析，不建议改 harness。**

### 为什么这个 cell 是 0.49%

见上：perseveration 是主因，phantom_prompt 的无图像通道 + prompt/观测刻意错配放大了它。

## 4. 🔁 Self-evolving — 提议规则

- 把 `P35(MUTATION_MISSING)` 泛化为 `PASSIVE_MUST_EXCLUDE_FP`（去掉 `agent_finished==True` 与 locator 白名单限定）—— 当前 P35 恰好漏掉 task 160 这类 sidebar 场景。⚠️ 实现时**不要**用 `effective_mutating_action_count` 做判据（B-1890：该字段恒为 0）
- `P36-fatal`：同一 (action_type, element_id) 的 walk_fail 连续占满几乎整个预算 → 与「偶发可自愈」型 walk_fail 区分开，对路由信号设计也有用

> 这些提议**尚未落码**。按 discover-then-freeze 纪律，reddit 六 mode × 三 model 的 discover 产物应合并成一批（R1–R8 + H2）后统一 bump `RULESET_VERSION` 到 `8-reddit-*` 并全量重扫，而不是逐条落码逐次重扫。

## 5. Actionable

- ⚠️ **本 cell 的 success 含 task 160（B-1889 benchmark-FP）**。若排除，SR 0.49% → 0.00%。排除与否属 prereg 级改动，**待 user / advisor 决策**，本 digest 不自行调整数字。
- 未发现需要开 B-number 的 scaffold-bug（本轮范围内）。

---

**Cross-link**: 笔记 §387.6 / §387.7 · master_bug_catalog B-1889 (task 160 passive-FP) / B-1890 (footprint 字段恒 0，勿用作判据) · `/tmp/diag_red/` Tier-1 原始扫描产物
