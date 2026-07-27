# /diag digest — B2 × `dom` × reddit

*生成 2026-07-27（Tier-1 全量 + Tier-2 深挖）*

> **定位声明**：本 digest 是**单 condition** 的失败归因，不下 cross-mode / cross-model 结论。
> 跨 mode 定量比较须等 reddit 规则批（R1–R8 + H2）落地、`RULESET_VERSION` 升到 `8-reddit-*`
> 并全量重扫后再做（/diag skill「discover-then-freeze」硬纪律）。


## 1. Header

| 字段 | 值 |
|---|---|
| **Run** | `B2_dom_reddit_20260715` |
| **Condition** | `phase1_dom_router_0` |
| **Site / Mode / Model** | reddit / `dom` / B2 = Gemma3-VL `google/gemma-3-4b-it` (local) |
| **Episodes** | 205 |
| **SR** | **3.90%** (8 success / 197 failed) |
| **ruleset_version** | `7-p6p16clsgate-b1860coord` |
| **Tier-1 三子集** | failed+hit 194 · **failed-NO-hit 3** · success+hit 2 |

## 2. Tier-1 规则分布（failed 侧）

| 规则 | 含义 | step-level hits | 命中 episode 数 |
|---|---|---|---|
| `P36` | WALK_FAIL_DEGENERATE | 1670 | 159 |
| `P5` | 感知缺失循环 | 261 | 149 |
| `P31` | budget 耗尽未完成 | 183 | 183 |
| `P14` | URL 自环 | 117 | 92 |
| `P12` | 从不翻页 | 37 | 37 |
| `P25` | 跨站任务跳过其中一站 | 31 | 31 |
| `P4` | 根节点误操作 | 4 | 1 |
| `P10` | 跨步数值记忆失败 | 2 | 1 |
| `P13` | 搜索代替浏览 | 1 | 1 |

**success 侧 fire 的规则（presence-only 误报审计对象）**: `P39`×2

**failed-NO-hit episode（deterministic 盲区）**: [64, 101, 171]

**success episode**: [105, 107, 130, 138, 150, 178, 188, 189]


## 3. Tier-2 深挖

**覆盖范围**：7 ep（no-hit 全 3 + P36 因果审计 4）· 1 sonnet sub-agent

**三分类**：agent-limit 7 · scaffold-bug 0 · benchmark-FP 0 · unclear 0

### P36 因果审计

**真死因**。walk_fail 信号真实准确 —— 被点的 element_id 确实存在于 observation（有真实 union_bound，非幻觉 ID），但确实不是可操作祖先（StaticText / 用 click 操作纯文本 input）。`locator_dispatch.py` 的 6 层 walk-up + ARIA 白名单按设计工作。致命的是模型**完全不响应** prompt 里 8-step 历史窗口中连续出现的 FAILED 反馈。walk_fail 后 episode **从不恢复**：task 145/100/40 是 100% 死锁，逐字重复同一 `(action_type, element_id)` 达 28–29 次直到预算耗尽。

### 具体发现

- **task 145 / 100** — 对同一 StaticText 元素连续点击 29 / 28 次，`walk_fail:no_actionable_within_walk` 100% 复现，thought 逐字不变。
- **task 40** — 模型正确说出「要在搜索框搜 Pittsburgh」的计划，却连续 29 步用 `click` 而非 `type` 作用于搜索框；CLICK 解析器按设计不接受纯文本输入框 → 必然 walk_fail。计划对、动词错。
- **task 72** — DOM-only 无图，模型却臆造出具体视觉描述（「一只猫睡在沙发上」）并反复自我强化，写入文本框的也是幻觉内容。比「答不出」更危险的失败模式。
- **task 64 / 101 / 171（no-hit）** — 三种不同形态：世界知识错误+无谓导航 / 任务语义误解+过早放弃 / 精细定位失败+从未完成动作。共同点是「看起来在推进」（URL 或 DOM 有变化）从而躲过现有 no_progress 类触发器。

### 为什么这个 cell 是 3.90%

三层叠加，且规模效应明显（4B vs 235B）：(a) **无自我修正** —— 明知连续 FAILED 仍逐字重复直到 budget 耗尽（4/7 episode，30 步里 27–29 步纯浪费），这是 P36 密度达 B0 4.5× 的直接机制；(b) **动作空间语义混淆** —— 把 StaticText 当可点击、用 click 操作文本框；(c) **DOM-only 下的视觉幻觉**。B0/B1 在相同陷阱前也会失手一两步，但能靠更强的指令遵循跳出死循环，不会把整条预算耗在同一个死点。

## 4. 🔁 Self-evolving — 提议规则

- `IDENTICAL_FAILED_ACTION_STREAK>=3`（同一 action_type+element_id 连续失败 ≥3 次即判 P36）—— 不是抓新模式，而是把现有 P36 的判定时机从 ~27 次**大幅提前**，可将此类 episode 的浪费步数从 ~28 压到 ~3。
- `EMPTY_FINISH_ON_MUTATION_TASK`（finish.answer=='' 且 eval 为 program_html/mutation 类）
- `NO_SEARCH_ATTEMPTED_FOR_PAGE_IMAGE_QUERY`（eval_types 含 page_image_query 且全轨迹 type 次数=0）

> 这些提议**尚未落码**。按 discover-then-freeze 纪律，reddit 六 mode × 三 model 的 discover 产物应合并成一批（R1–R8 + H2）后统一 bump `RULESET_VERSION` 到 `8-reddit-*` 并全量重扫，而不是逐条落码逐次重扫。

## 5. Actionable

- 本 cell 的 success 不含 task 160（B-1889 不影响本 cell 的 SR）。
- 未发现需要开 B-number 的 scaffold-bug（本轮范围内）。

---

**Cross-link**: 笔记 §387.6 / §387.7 · master_bug_catalog B-1889 (task 160 passive-FP) / B-1890 (footprint 字段恒 0，勿用作判据) · `/tmp/diag_red/` Tier-1 原始扫描产物
