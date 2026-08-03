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

### 3.1 精确根因（2026-08-03 补，`state_change.py`）

```python
similarity_threshold   = 0.95     # config.py:84 / exp_v2_base.yaml:105
_TEXT_TRUNCATION_LIMIT = 20000    # 超过则走 md5 二值判定 (差 1 字节 → similarity=0.0)
similarity = SequenceMatcher(None, text_before, text_after).ratio()
if similarity < similarity_threshold: changes.append("content_changed")
```

链条：**多个分散的 volatile 元素**（相对时间戳等）→ SequenceMatcher 对分散小改动极敏感
（task 651 `text_length` 只抖 4.7%，`similarity` 掉到 **0.42–0.75**）→ < 0.95 → `content_changed`
→ ⚠️ **`content_changed` ∈ `AGENT_VISIBLE_REASONS`** → `page_changed` 与 `agent_visible_changed`
**同时**被顶起。

**B-09 的两层拆分设计本身是正确的**（`RUNNER_INTERNAL_REASONS` = interactive_elements /
form_fields / dom_complexity / text_length / form_value 已被排除在 agent-visible 之外），
问题出在 `content_changed` 这一条**判定粒度太粗**，不区分实质变化与时间戳抖动。

全量佐证：`content_changed` 占有-reason step 的 **88.5%**（16 condition 抽样），是最主导的 reason。

### 3.2 附带发现：`dom_complexity` 命名误导

```python
"dom_complexity": text.count("\n") + 1      # state_change.py:102
```

**它不是 DOM 复杂度，是可见文本的行数。** 任何把它当"DOM 节点数 / 结构复杂度"解释的分析都是错的。
（本审计用它做"文本结构是否变化"的代理是成立的，但语义须说明。）

---

## 4. 分层可信度结论

### 4.0 P36 / P31 —— 两条最大规则的可信度（跨 benchmark 10 例因果验证）

| benchmark | P36 | P31 |
|---|---|---|
| WA reddit | **3/3 risk-marker** | **3/3 risk-marker** |
| VWA (cls + reddit) | **2/2 risk-marker** | **2/2 risk-marker** |

**10 例跨 benchmark 一致：两条都是 presence marker，不是 causal attributor。**
而它们分别覆盖 51–53% 和 49.9–54.3% 的 failed episode。

**P36 另有两项独立的可信度折扣**：

1. **vision 上分母不同**（§2.1）—— 只覆盖 type 步，click 步（占 2–20 倍）无 `locator_route_meta`
2. **16.3% 的 click 命中是无害的** —— `dispatch_id_based_type` 有独立的 DOM walk-up
   (`_JS_RESOLVE_INPUT`)，失败的 pre-focus click 从不阻塞随后针对同一 `element_id` 的 type。
   全量实测「click walk_fail 但同一 id 后被成功 type」：

   | mode | 无害 / walk_fail click | 占比 |
   |---|---:|---:|
   | phantom_som | 941 / 4576 | **20.6%** |
   | som | 567 / 4230 | 13.4% |
   | phantom_text | 342 / 3037 | 11.3% |
   | phantom_prompt | 419 / 4536 | 9.2% |
   | dom | 269 / 4368 | **6.2%** |
   | **合计** | **3862 / 23712** | **16.3%** |

   ⚠️ **跨 mode 不均**（psom 20.6% vs dom 6.2%）→ 这层噪声本身就有 mode 偏向。

**综合结论：P36 作为定量分析量的可信度低**（risk-marker + 分母不同 + 16.3% 有偏噪声）。
**建议**：收窄为"click walk_fail 且该 element_id 此后未被成功 type"，并在 vision 上标注 type-only 分母。

### 4.0b 🔴 P43 的「中性标签」定位在 classifieds 上不成立 —— §407.26(b) 需 retract

台账 **§407.26** [RETRACTED] 的 (b) 项写：

> P43 是它自己 docstring 写明的中性 (task × mode) 标签而非失败预测，**§387.10 在它的任务集上做过
> 控制实验，恢复截图测得 +0.00/+1.56/+0.00pp** —— 它定位效应，不解释效应

**本轮在 P43 命中的 task 子集上重做受控 dom→som 对比**（全 48 condition）：

| | P43 子集 n | dom SR | som SR | Δ |
|---|---:|---:|---:|---:|
| **B0 / classifieds** | 71 | 9.9% | 29.6% | **+19.72pp** |
| **B1 / classifieds** | 71 | 1.4% | 14.1% | **+12.68pp** |
| B2 / classifieds | 71 | 2.8% | 2.8% | +0.00pp（地板效应，SR 2.8% 无区分力） |
| B0 / reddit | 64 | 12.5% | 12.5% | +0.00pp |
| B1 / reddit | 64 | 6.2% | 7.8% | +1.56pp |
| B2 / reddit | 64 | 1.6% | 1.6% | +0.00pp |

**台账记录的 `+0.00/+1.56/+0.00` 正是 reddit 那一行。** §387.10 的受控实验只覆盖了 reddit 的
64 个任务，而 **P43 跨站触发**（不像 P6/P16 那样 gate 到 classifieds），**classifieds 的 71 个命中
从未被这样检验** —— 在那里补图带来 **+12.7 ~ +19.7pp**。

**结论**：
1. **P43 在 classifieds 上是 death-cause，不是中性标签**（两个独立 Tier-2 样本 + 全量受控对比一致）
2. §407.26(b) 犯的是 **scope over-generalization** —— 把单站受控实验的结论推广到跨站规则的全部命中
3. ⚠️ **这条错误被一个 `[RETRACTED]` 条目固化了** —— 台账里 RETRACTED 状态的条目仍在传播其内部论证

**对 paper 的正面意义**：P43 子集是一个**可事前识别、routing 能救**的任务集
（classifieds 上 dom→som +19.72pp）—— 这是 routing 价值的直接证据，比"某 mode 平均更好"强得多。
建议接入 §6 router 的证据链。

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

## 4.5 Outcome / Efficiency 两维度体检（2026-08-03 补）

### 4.5.1 B0 的 Efficiency 字段结构性缺失（条件性失效，②类）

| 子字段 | B0 | B1 | B2 |
|---|---:|---:|---:|
| `energy.{kwh, co2e_kg, power_watts}` | **0%** | 100% | 100% |
| `latency_ms.{generate, preprocessing}` | **0%** | 100% | 100% |
| `tokens.{input_image, input_text}` | **0%** | 100% | 100% |
| `tokens.thinking` | 0% | 0% | 0% | ← 死字段（①类） |

B0 走 API proxy，不本地推理 → 无能耗、无 preprocessing 分段、无图文 token 拆分。**架构使然。**

⚠️ **后果**：任何用 `tokens.input_image` / `input_text` 做**跨模型**比较的分析，**B0 的 0 是假 0**。
Efficiency 3b（Image embedding / total-token gap）若依赖图文拆分，B0 上不可用，只能用 `total_tokens`。

### 4.5.2 `cost_total_mixed_unit_warn` —— 警告为真，量级可忽略

```
触发率:  B0 58774/58774 = 100.00%   ·   B1 0%   ·   B2 0%
```

**含义**（`types.py:279-283` + B-1559）：B0 的 `total_billed_cost_usd` 混了两种单位 ——
API cost (`api_usd`) + 本地 scaffold cost (`electricity_usd_derived`，来自 `obs_prepare` /
`router_overhead`)，两者尺度差 ~1000×。B1/B2 单一 basis 故不 warn。

**实测混入量**（`obs_prepare + router_overhead` 占 `total`）：

| | 全体占比 | 单 condition 最高 |
|---|---:|---|
| B0 | **0.0039%** | `B0_som_wa_reddit` 0.0214% |
| B1 | 0.0067% | `B1_som_wa_reddit` 0.0357% |
| B2 | 0.0060% | `B2_som_reddit` 0.0284% |

→ **警告技术上正确，但对 §1 "cost ≈ DOM" 无实质影响**（混入 < 0.03%）。
B1/B2 的本地占比反而更高，只是单一 basis 不触发 warn。**建议 §8 一句 disclose，不需重算。**

（附带观察：`som` mode 的 `obs_prepare` 比其他 mode 高一个量级 —— 生成标注图的成本，符合预期。）

### 4.5.3 Outcome 维度

依赖 `success`（已知 benchmark-FP 问题，见 §B2/§B4 与 §402.7）与 `confidence`
（B0 只有 4/6 子字段填充，CLAUDE.md 已记：entropy=None per top-2 truncation）。
→ **0g Routing AUROC 若消费 entropy，B0 上不可用**；本次未展开验证，列入 §6 待办。

---

## 4.6 P0-3 完成：router escalation 触发率重算

`router.py:80-83` 的 `unchanged_streak` 由 `prev_page_changed` 驱动。用 §3 的修正口径
（`page_changed=True` 但 url/scroll_y 不变且 `text_similarity>0.95` → 视为 unchanged）重算
`streak≥2` 的触发：

| mode | episode | streak≥2 实测 | 修正后 | 漏检 | 漏检率 |
|---|---:|---:|---:|---:|---:|
| som | 1287 | 668 | 755 | +87 | **13.0%** |
| phantom_som | 1287 | 731 | 822 | +91 | 12.4% |
| phantom_text | 1287 | 695 | 773 | +78 | 11.2% |
| dom | 1287 | 744 | 817 | +73 | 9.8% |
| phantom_prompt | 1287 | 804 | 879 | +75 | 9.3% |
| vision | 1287 | 937 | 988 | +51 | **5.4%** |
| **合计** | **7722** | **4579** | **5034** | **+455** | **9.9%** |

**结论**：rule-based router 的 escalation 触发率被**系统性低估约 10%**，且**跨 mode 不均**
（som 13.0% vs vision 5.4%）。→ **paper §6 里凡涉及 escalation 触发率的跨 mode 比较都会被扭曲**，
需按此校正或加披露。

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
