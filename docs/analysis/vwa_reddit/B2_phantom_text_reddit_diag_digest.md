# /diag digest — B2 × `phantom_text` × reddit

*生成 2026-07-27（Tier-1 全量 + Tier-2 深挖）*

> **定位声明**：本 digest 是**单 condition** 的失败归因，不下 cross-mode / cross-model 结论。
> 跨 mode 定量比较须等 reddit 规则批（R1–R8 + H2）落地、`RULESET_VERSION` 升到 `8-reddit-*`
> 并全量重扫后再做（/diag skill「discover-then-freeze」硬纪律）。


## 1. Header

| 字段 | 值 |
|---|---|
| **Run** | `B2_phantom_text_reddit_20260720` |
| **Condition** | `phase1_phantom_text_router_0` |
| **Site / Mode / Model** | reddit / `phantom_text` / B2 = Gemma3-VL `google/gemma-3-4b-it` (local) |
| **Episodes** | 205 |
| **SR** | **2.44%** (5 success / 200 failed) |
| **ruleset_version** | `7-p6p16clsgate-b1860coord` |
| **Tier-1 三子集** | failed+hit 197 · **failed-NO-hit 3** · success+hit 1 |

## 2. Tier-1 规则分布（failed 侧）

| 规则 | 含义 | step-level hits | 命中 episode 数 |
|---|---|---|---|
| `P36` | WALK_FAIL_DEGENERATE | 722 | 103 |
| `P31` | budget 耗尽未完成 | 188 | 188 |
| `P5` | 感知缺失循环 | 137 | 74 |
| `P4` | 根节点误操作 | 84 | 20 |
| `P14` | URL 自环 | 51 | 47 |
| `P25` | 跨站任务跳过其中一站 | 38 | 38 |
| `P12` | 从不翻页 | 19 | 19 |
| `P10` | 跨步数值记忆失败 | 7 | 5 |

**success 侧 fire 的规则（presence-only 误报审计对象）**: `P39`×1

**failed-NO-hit episode（deterministic 盲区）**: [64, 104, 179]

**success episode**: [77, 130, 138, 160, 200]


## 3. Tier-2 深挖

**覆盖范围**：6 ep（no-hit 3 + P36 因果审计 3）· 1 sonnet sub-agent

**三分类**：agent-limit 6 · scaffold-bug 0（但发现一个真实检测缺陷，见下）· benchmark-FP 0 · unclear 0

### P36 因果审计

**真死因**。103/12/205 三条轨迹在触发 P36 后被完全钉死：对着已证实点不动的 element_id 逐字重复 20–28 次，耗尽全部预算，从未尝试换元素 / 滚动 / go-back / 改 URL。但**背后机制是 agent-limit**：element_id 均在当步 mark_count 范围内（非幻觉出界 ID）；task 205 甚至**已经成功到达目标页面**却仍在重复「我需要导航过去」的过时推理。

### 具体发现

- 🐛 **发现一个真实 scaffold 缺陷（非本 episode 根因，但值得单独修）**：task 103 的 `state_change_reason_distribution` 显示 `scroll_changed:29` —— **纯滚动位移被计入 `page_changed=True`**，导致 `no_progress_streak` / loop-trigger 全程哑火（`trigger_distribution={}` 空）。建议：`state_change_reason` 集合若只含 `scroll_changed` 不应计入 page_changed。否则这类「滚动但零实质进展」的 case 会继续被系统性漏记，影响一切基于 trigger 计数的分析。
- **task 104** — 把 'Notifications' 链接误判成通往论坛的路径，此后连续 8 次原样重发同一 click，url 恒为 /notifications。
- **task 179** — 图片识别正确（Missouri）、search 也成功，但陷入两段死循环：对真实 `<a>` 标签连点 7 次 url 不变；在 /forums 与 /search 间横跳 15+ 步，从未点击指向目标 forum 的链接。
- **task 12** — 从未表现出对参考图的任何识别，直接搜字面词 'Comments'，落在 NYC 版面（非目标 Pittsburgh），随后连点 26 次。

### 为什么这个 cell 是 2.44%

与 B2 其他 mode 同源：perseveration + 内部状态不随观测更新。task 205 是最清晰的例证 —— 已在目标页 20+ 步仍逐字重复「需要导航到该页」。

## 4. 🔁 Self-evolving — 提议规则

- `IDENTICAL_ACTION_NO_STATE_CHANGE`（连续 ≥5 步 action_type+element_id 相同 且 url_before==url_after）—— 纯字段比较 0 token，命中本轮 4/6 episode
- 修复 scroll-only 状态变化被误判为 progress（见上，是给现有 trigger 逻辑打补丁而非新规则）
- `URL_DIVERSITY_COLLAPSE`（trailing 10 步内 distinct url ≤2 且未 done）—— 覆盖比逐字重复更隐蔽的「两态乒乓」

> 这些提议**尚未落码**。按 discover-then-freeze 纪律，reddit 六 mode × 三 model 的 discover 产物应合并成一批（R1–R8 + H2）后统一 bump `RULESET_VERSION` 到 `8-reddit-*` 并全量重扫，而不是逐条落码逐次重扫。

## 5. Actionable

- ⚠️ **本 cell 的 success 含 task 160（B-1889 benchmark-FP）**。若排除，SR 2.44% → 1.95%。排除与否属 prereg 级改动，**待 user / advisor 决策**，本 digest 不自行调整数字。
- 未发现需要开 B-number 的 scaffold-bug（本轮范围内）。

---

**Cross-link**: 笔记 §387.6 / §387.7 · master_bug_catalog B-1889 (task 160 passive-FP) / B-1890 (footprint 字段恒 0，勿用作判据) · `/tmp/diag_red/` Tier-1 原始扫描产物
