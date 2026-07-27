# /diag digest — B2 × `phantom_som` × reddit

*生成 2026-07-27（Tier-1 全量 + Tier-2 深挖）*

> **定位声明**：本 digest 是**单 condition** 的失败归因，不下 cross-mode / cross-model 结论。
> 跨 mode 定量比较须等 reddit 规则批（R1–R8 + H2）落地、`RULESET_VERSION` 升到 `8-reddit-*`
> 并全量重扫后再做（/diag skill「discover-then-freeze」硬纪律）。


## 1. Header

| 字段 | 值 |
|---|---|
| **Run** | `B2_phantom_som_reddit_20260722` |
| **Condition** | `phase1_phantom_som_router_0` |
| **Site / Mode / Model** | reddit / `phantom_som` / B2 = Gemma3-VL `google/gemma-3-4b-it` (local) |
| **Episodes** | 205 |
| **SR** | **1.46%** (3 success / 202 failed) |
| **ruleset_version** | `7-p6p16clsgate-b1860coord` |
| **Tier-1 三子集** | failed+hit 197 · **failed-NO-hit 5** · success+hit 1 |

## 2. Tier-1 规则分布（failed 侧）

| 规则 | 含义 | step-level hits | 命中 episode 数 |
|---|---|---|---|
| `P36` | WALK_FAIL_DEGENERATE | 1037 | 133 |
| `P4` | 根节点误操作 | 262 | 46 |
| `P31` | budget 耗尽未完成 | 192 | 192 |
| `P5` | 感知缺失循环 | 161 | 91 |
| `P14` | URL 自环 | 58 | 49 |
| `P25` | 跨站任务跳过其中一站 | 37 | 37 |
| `P12` | 从不翻页 | 27 | 27 |
| `P10` | 跨步数值记忆失败 | 3 | 3 |
| `P22` | 图上数字 dom 不可读 | 1 | 1 |

**success 侧 fire 的规则（presence-only 误报审计对象）**: `P25`×1

**failed-NO-hit episode（deterministic 盲区）**: [73, 89, 129, 130, 140]

**success episode**: [58, 160, 179]


## 3. Tier-2 深挖

**覆盖范围**：8 ep（no-hit 5 + P36 审计 2 + success 1）· 1 sonnet sub-agent

**三分类**：agent-limit 7 · scaffold-bug 0 · benchmark-FP 0 · unclear 1（task 58，成功但走捷径）

### P36 因果审计

判为**伴随症状而非独立根因** —— 与 B2_dom / B2_phantom_text 两位 sub-agent 的措辞有细微分歧，此处如实并陈：三方对**机制**的描述完全一致（walk_fail 只在 element_id 已在 `obs_nodes_info` 中找到、union_bound 存在之后才触发，从不代表幻觉引用；`_JS_RESOLVE_CLICK` 按设计不接受纯文本 `<input>`，task 181 在 step_13 换成 `type` 后立刻 `target_tag='INPUT'` 成功，直接实锤），分歧只在把 P36 称作「直接死因」还是「执行层症状」。**综合表述：P36 是失败的直接放大机制，根因是模型 perseveration。**

### 具体发现

- ✅ **[SOM_MARKS] 文本与可操作元素集一致，未发现错标** —— `som.py::build_som_text_from_obs_text` 每条 mark 直接取自 AXTree 行（仅去掉 `[N]` 前缀），role 信息（如 'textbox'）本就在文本里，模型有足够线索区分「该 type 还是该 click」。选错动词是纯推理问题。也未命中已知的 P33（点击图片 href 跳裸图页）。**P-SoM 作为 hero mode 在 scaffold 层是干净的** —— 对论文有利，但样本仅 8 例，建议在 B0/B1 同 mode 交叉核对后再写进正文。
- **task 181** — 前 13 步对搜索框（经 step_13 证实 `target_tag='INPUT'`）连续误用 click 而非 type；step_14 点了搜索结果跳出沙盒到真实站点 wfsb.com，后续在外站 DOM 上继续大量 walk_fail。
- **task 73** — 把「描述计划」当「执行计划」：step_0 的 thought 说 'I will search for...'，同一步直接输出 `finish`，episode 在 0 次真实导航后终止。
- **task 58（success，判 unclear）** — string_match 'Reki Kawahara' 精确匹配，判定本身无误；但 21 步全程只在 reddit 内打转，**从未访问任务要求的第二站点 wikipedia（localhost:8888）**，答案很可能来自模型参数知识而非页面取证。不影响 success 判定，但值得作为诚实性附注。

### 为什么这个 cell 是 1.46%

与 B2 其他 mode 同源（perseveration + 视觉语义映射弱）。P-SoM 特有的是**无图像通道**：task 89 即使成功导航到图片 URL 页面，`input_image` 仍为 0 token → 该任务在此 mode 下**结构性不可解**，不应记为 Gemma3 的能力弱点。

## 4. 🔁 Self-evolving — 提议规则

- `MULTI_SITE_TASK_SINGLE_SITE_GROUNDING`（task.sites >1 但轨迹 obs_url 只覆盖 1 个站点）→ 标记「疑似参数知识捷径成功」，**直接关系 SR 数字的诚实性，且不只影响 P-SoM**
- `PHANTOM_IMAGE_BLIND`（全 episode input_image tokens==0 且任务本质需要看图）→ 把结构性不可解的任务从「模型能力不足」里摘出单独统计
- `STUCK_REPEAT_VALID_CLICK`（同一元组连续 ≥3 步、locator success=true 但 page_changed=false）—— 与 P36 walk_fail 型循环互补

> 这些提议**尚未落码**。按 discover-then-freeze 纪律，reddit 六 mode × 三 model 的 discover 产物应合并成一批（R1–R8 + H2）后统一 bump `RULESET_VERSION` 到 `8-reddit-*` 并全量重扫，而不是逐条落码逐次重扫。

## 5. Actionable

- ⚠️ **本 cell 的 success 含 task 160（B-1889 benchmark-FP）**。若排除，SR 1.46% → 0.98%。排除与否属 prereg 级改动，**待 user / advisor 决策**，本 digest 不自行调整数字。
- 未发现需要开 B-number 的 scaffold-bug（本轮范围内）。

---

**Cross-link**: 笔记 §387.6 / §387.7 · master_bug_catalog B-1889 (task 160 passive-FP) / B-1890 (footprint 字段恒 0，勿用作判据) · `/tmp/diag_red/` Tier-1 原始扫描产物
