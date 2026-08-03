# 数据质量审计 — diag / Macro / Micro 的字段级信任基础

*2026-08-03 建立。范围：VWA 36 condition + WA 12 condition = **48 condition**，schema v2 step record。*

> **为什么做**：2026-08-03 的 /diag 验证轮暴露了一个反复出现的模式 —— **字段级信任假设从未被验证过**。
> 四个独立案例：`page_changed` 会假阳性 · `action_success` 在 walk_fail 时仍为 True ·
> `_finish_answer()` 只读 `answer` 不读 `thought` · 我自己的扫描漏排了 `raw_action`。
> 每一个都让某层结论悄悄失真而不报错。本文件是对**全部字段**做一次系统体检。
>
> 相关：[[wa_reddit/_benchmark_level_findings]] §B1-R / §V / §W · 笔记 §416

---

## 0. 三类质量缺陷（本文的分类骨架）

| 类型 | 定义 | 症状 | 本轮实例 |
|---|---|---|---|
| **① 死字段** | 全 48 condition 恒 `None` | 读它的代码是死路径 | `checklist` · GLM 四兄弟 · `*_retry` |
| **② 条件性失效** | 部分 condition 有值、部分恒 `None` | **把"没填"读成"没发生"** | vision 的 `locator_route_meta` · B1 的 `tool_call_*` |
| **③ 语义失真** | 字段有值但不表达它宣称的语义 | 值可信度崩塌，无任何报错 | `page_changed` 假阳性 |

**②最危险** —— 它在跨 condition 比较里制造假 0，而假 0 和真 0 在表格里长得一模一样。
对应项目既有规则 [[feedback_absence_of_evidence_vs_measured_zero]]：*没测到 ≠ 测出是零*。

---

## 1. 死字段（10 个，全 48 condition 恒 None）

```
checklist · control_intervention · dialog_meta_retry · retry_action_type
locator_route_meta_retry · select_option_meta_retry
glm_fallback_attempted · glm_fallback_used · glm_fallback_latency_ms · glm_original_fail_reason
```

- **GLM 四兄弟**：已知 zombie，B-991 retire 后保留供 archive 读兼容（CLAUDE.md 有记）。**符合预期。**
- **`*_retry` 三兄弟**（`locator_route_meta_retry` / `select_option_meta_retry` / `retry_action_type`）：
  说明 **retry 路径的元数据在这 48 个 canonical run 上从未落盘**。⚠️ 任何"retry 行为"分析若读这些字段，
  得到的 0 是死字段而非无 retry —— 需改读 `retry_count` / `retry_action_applied`。
- `checklist` / `control_intervention`：对应功能未启用。

---

## 2. 条件性失效（18 个字段）—— 最危险的一类

### 2.1 vision 的 locator 层结构性缺失（**影响 diag 的跨 mode 比较**）

```
vision:  action.coordinate=[182,34] · element_bbox=None · locator_route_meta=None
dom   :  action.element_id=6        · element_bbox=[194,0,86,52]
                                    · locator_route_meta={success,target_tag,error,action_kind}
```

**按 action_type 拆开后的精确填充率**（8 个 vision condition）：

| action_type | `locator_route_meta` 填充 |
|---|---|
| **click** | **0 / 914 ~ 0 / 2270 = 0.0%**（全部 8 个 condition） |
| **type** | 459/459 · 585/585 · 233/233 …（≈100%） |

vision 的 **type 走 locator 路径**（需定位输入框），**click 走纯坐标**，完全绕开 locator 层。
**这是架构使然，不是 bug** —— 但下游后果严重。

#### 受影响的 P-rule（假 0 矩阵）

| 规则 | 依赖字段 | vision 命中 | 非 vision 命中 | 判定 |
|---|---|---:|---:|---|
| `P4` 根节点误操作 | `element_bbox` | **0** | **604** | ❌ **完全假 0** |
| `P2` 容器节点误点 | `element_bbox` | **0** | 126 | ❌ **完全假 0** |
| `P36` WALK_FAIL | `locator_route_meta{,_primary,_retry}` | 少量 | 大量 | ⚠️ **分母不同** —— 只覆盖 type 步，click 步（占 2–20 倍）完全不检查 |
| `P1` 元素中心越界 | `element_bbox` | 10 | 0 | ✅ vision-specific 逻辑，反向正常 |

> ⚠️ **任何"vision 的失败模式与 dom/som 不同"的结论，只要涉及 P2/P4/P36，都可能是字段缺失造成的假象。**
> per-rule cross-mode 表里 vision 那一列的 P2/P4 = 0 **必须标注为不可比**，不能读成"vision 没这个问题"。

### 2.2 其余条件性失效字段

| 字段 | 全 None 的 condition 数 | 模式 |
|---|---:|---|
| `tool_call_valid` / `tool_call_emitted` / `tool_call_parse_path` / `text_fallback_used` | 30/48 | **只在 B0 填充**（proxy tool-calling 专有；B1/B2 本地推理无此路径） |
| `tool_call_fallback_reason` | 40/48 | 同上，且仅在实际 fallback 时填 |
| `text_parse_path` | 36/48 | 同上 |
| `network_retry_count` / `network_retry_wait_ms` | 32/48 | 仅部分 run |
| `dialog_meta` / `dialog_meta_primary` | 27/48 | 仅在实际弹出 dialog 的 run |
| `intervention_type` / `counted_as_agent_action` / `intervention_from_url` / `intervention_recovery_url` | 19/48 | 仅在触发干预的 run |
| `parse_failure_reason` | 15/48 | 仅在实际 parse 失败时填 |
| `element_bbox` | 8/48 | **全部是 vision**（见 2.1） |
| `select_option_meta{,_primary}` | 1/48 | `B1_vision_reddit` |

**跨模型分析的硬约束**：`tool_call_*` 四件套只在 B0 有值 → **任何用它们做 B0/B1/B2 对比的分析，
B1/B2 的 0 都是假 0**。

---

## 3. 语义失真：`page_changed` 假阳性

详见 [[wa_reddit/_benchmark_level_findings]] §W2 与 **B-1926**。摘要：

- volatile DOM 片段（疑似 CSRF nonce）顶起 `content_changed` + `form_value_changed`
  → `page_changed=True`，而 url / dom_complexity / scroll_y 全程不变
- **规模**：`page_changed` 恒 True 且 URL 全程不变的 episode = **VWA 148 + WA 13 = 161**
- **step 级疑似假阳性率**：`vision 9.02% > som 7.90% > pprompt 6.86% > dom 6.40% > psom 6.32% > ptext 5.48%`
- ⚠️ `visibility_gap_rate` **抓不到它** —— volatile 片段把 `agent_visible_changed` 一起顶起
  （task 651 上 30/30 全 True，`page_changed=True AND agent_visible_changed=False` 命中 0）
- **先例**：`p79/experiment/router.py:95-99` 注释已记同现象（VWA B2 task 103，触发 reason 是 `scroll_changed`）

---

## 4. 分层可信度结论

### 4.1 diag 层

| 项 | 可信度 | 说明 |
|---|---|---|
| Tier-1 命中**存在性** | ✅ 高 | deterministic、可复现 |
| Tier-1 per-rule **cross-mode 比较** | ⚠️ **P2/P4/P36 不可比** | vision 列是假 0 / 不同分母（§2.1） |
| Tier-1 per-rule 表作为**死因分布** | ❌ **不成立** | P36(51%) / P31(49.9%) 经因果验证均为 risk-marker |
| Tier-2 三分类归因 | ⚠️ **约 1/3 分歧率** | 独立盲复检一致率 5/8；分歧集中在 agent-limit ↔ benchmark-FP 边界 |
| Tier-2 覆盖率 | ⚠️ 19.4% | 383 个 `failed+hit` 中只验了 6 个 |

### 4.2 Micro 层（四维度框架）

依赖字段 `page_changed` / `agent_visible_changed` / `action_success` / `scroll_y`
在 **6/6 mode 上填充率均为 100%** → **无缺失型问题**，只有 `page_changed` 的失真型问题。

**敏感性分析**（把疑似假阳性 step 重算为 no-change）：

| 结论 | 稳健性 |
|---|---|
| `no_change_rate` 最高 = Vision | ✅ **6/6 cell 稳健**（污染方向与结论一致，修正只加强） |
| `no_change_rate` 最低端 | ⚠️ **1/6 cell 翻转**（B0/cls: som → phantom_text）→ 需脚注或改用区间 |

### 4.3 Macro 层

1a/1b/1c/1d 分别源自 `axis_effect_size.json` 与 `mechanism_per_task.json`，**不直接消费 `page_changed`**；
唯一相关的 `url_revisit_rate` 建在 URL 序列上。→ **受本轮发现影响小。**

### 4.4 router（paper §6）

`router.py:80-83` 的 `unchanged_streak` 直接由 `prev_page_changed` 驱动。
那 **161 个 episode 上 `page_unchanged_streak` 永不累积**，`trigger_distribution` 全空。
§105 先例已明写同类污染的下游清单：*router signal AUROC / wasted_cost / no_op_rate 都受污染，
paper §5/§6 数字需校*。**这是同类问题第二次出现。**

---

## 5. Actionable

| # | 事项 | 优先级 |
|---|---|---|
| Q1 | per-rule cross-mode 表给 vision 列的 P2/P4/P36 加"字段不可用"标注（不是 0） | **P0** |
| Q2 | digest §2 加解读警告：per-rule 表是**症状分布**不是死因分布 | **P0** |
| Q3 | router escalation 触发率按 §105 清单重算（161 episode 的 streak 永不累积） | **P0** |
| Q4 | `page_changed` 判定加权：要求 url/dom_complexity/text_similarity 至少一项佐证（B-1926） | P1 |
| Q5 | `_finish_answer()` 在 answer/text 皆空时 fallback 读 `thought`（影响 P22/P24/P27/P29/P46） | P1 |
| Q6 | 任何用 `tool_call_*` 的跨模型分析显式排除 B1/B2（假 0） | P1 |
| Q7 | "retry 行为"分析改读 `retry_count`/`retry_action_applied`，不读死字段 `*_retry` | P2 |

---

## 6. 尚未审计

- **377 个 `failed+hit` 的死因**（本轮只验 6 个）· **B1 那批 80 个 Tier-2 归因**（一个没验）·
  **VWA 36 份的 Tier-2 归因**（完全没碰）
- **Efficiency 与 Outcome 两个维度**对上述字段的依赖（本文只覆盖 diag / Macro / Micro）
- `state_digest` 各子字段的语义正确性（本文只查了填充率，未查 `dom_complexity` / `text_similarity`
  的计算是否符合其命名）
- **cost / latency / token 字段**的一致性（`cost_unit_basis` 有 `cost_total_mixed_unit_warn` 标志位，
  未查其触发率）
