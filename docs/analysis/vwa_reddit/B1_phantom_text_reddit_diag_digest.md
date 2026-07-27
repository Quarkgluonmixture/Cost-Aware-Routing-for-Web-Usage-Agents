# /diag digest — B1 × `phantom_text` × reddit

*生成 2026-07-27（Tier-1 全量 + Tier-2 深挖）*

> **定位声明**：本 digest 是**单 condition** 的失败归因，其中的 per-rule 分布只描述它自己。
>
> ✅ **discover-then-freeze 已完成**（2026-07-27）：reddit 规则批 P41–P46 + B-1890 修复 + P33
> reddit 路径扩展已落码，`RULESET_VERSION` = `8-reddit-p41p46-b1890fix`，**全部 36 个 canonical
> condition（reddit 18 + cls 18）已在该版本下重扫**，版本一致性由
> `scripts/analysis/diag_rescan_all.py` 校验 → **cross-mode / cross-model 定量聚合现已解锁**。
>
> ⚠️ v7→v8 的 cls 行为**不是**字节不变，差异全部经过定性核实：`P35`/`P39` 的旧命中因
> B-1890 死字段修复而移除（抽查确认那些 episode 确实有 6–8 个突变步，旧命中是错的）；
> `P33` 在 cls 上 +1 例（cls task 233 的 intent 实际要求访问 reddit，旧正则漏检）。


## 1. Header

| 字段 | 值 |
|---|---|
| **Run** | `B1_phantom_text_reddit_20260710` |
| **Condition** | `phase1_phantom_text_router_0` |
| **Site / Mode / Model** | reddit / `phantom_text` / B1 = Qwen3-VL-4B (local) |
| **Episodes** | 205 |
| **SR** | **6.83%** (14 success / 191 failed) |
| **ruleset_version** | `8-reddit-p41p46-b1890fix` |
| **Tier-1 三子集** | failed+hit 181 · **failed-NO-hit 10** · success+hit 2 |

## 2. Tier-1 规则分布（failed 侧）

| 规则 | 含义 | step-level hits | 命中 episode 数 |
|---|---|---|---|
| `P36` | WALK_FAIL_DEGENERATE | 787 | 115 |
| `P4` | 根节点误操作 | 170 | 35 |
| `P31` | budget 耗尽未完成 | 154 | 154 |
| `P5` | 感知缺失循环 | 120 | 79 |
| `P44` | HALLUCINATED_ELEMENT_REF | 95 | 21 |
| `P45` | IDENTICAL_FAILED_ACTION_STREAK | 70 | 53 |
| `P43` | PAGE_EMBEDDED_VISUAL_NO_SCREENSHOT(中性标签) | 61 | 61 |
| `P14` | URL 自环 | 55 | 50 |
| `P33` | 导航至裸图片 URL 幻觉 | 48 | 48 |
| `P25` | 跨站任务跳过其中一站 | 31 | 31 |
| `P12` | 从不翻页 | 25 | 25 |
| `P10` | 跨步数值记忆失败 | 13 | 6 |
| `P46` | COMMENT_INTENT_NO_TYPE | 9 | 9 |
| `P13` | 搜索代替浏览 | 6 | 6 |
| `P27` | 找不到即放弃 | 1 | 1 |

**success 侧 fire 的规则（presence-only 误报审计对象）**: `P25`×1, `P42`×1, `P41`×1

**failed-NO-hit episode（deterministic 盲区）**: [38, 111, 125, 133, 138, 139, 172, 179, 191, 193]

**success episode**: [0, 36, 40, 42, 58, 129, 131, 157, 160, 171, 178, 188, 189, 200]


## 3. Tier-2 深挖

**覆盖范围**：7 ep（no-hit 5 + success 2）· 1 sonnet sub-agent + 全 run 扫描

**三分类**：agent-limit 5 · benchmark-FP 2 · scaffold-bug 0（但复核出一个真实检测缺陷，见下）

### 具体发现

- 🐛 **scroll-only 状态变化确实存在于 B1**：全 run 5226 步中 23 步的 `page_change_reasons` 恰好只含 `scroll_changed`，且全部仍记 `page_changed=True`。但**它不是这些 episode 的主因** —— 我复核后确立的真根因是 `action_success` 语义脱节（→ **B-1891**）：`no_progress_streak` 由 `prev_action_success` 驱动而非 `page_changed`，两个 trigger 是被**各自独立**压制的。
- **4/5 no-hit 是指令-观测错配**：任务显式或隐式要求看图，而 phantom_text 剥离页面截图。分两种子模式 —— 纯页面内嵌图（零信息，task 104 高置信度编造「0 kirbies」）vs 任务级参考图可见但页面帖子图不可见（task 133，能看懂参考图却无法在页面里核对是哪个帖子）。
- **task 104 的高置信度幻觉值得单独记**：模型**不知道自己看不见**，会把无信息状态包装成「已观察」的确定性陈述。

### 为什么这个 cell 是 6.83%

见上：指令-观测错配为主，叠加 perseveration。

## 4. 🔁 Self-evolving — 提议规则

- → 已落码为 **P43**（但按 §387.10 的受控对比结果改成了**中性标签**，不是 sub-agent 提议的「结构性不可解」）
- B-1891 的修复（`action_success` 语义）属 runner 层，未在本批规则内

> 这些提议**尚未落码**。按 discover-then-freeze 纪律，reddit 六 mode × 三 model 的 discover 产物应合并成一批（R1–R8 + H2）后统一 bump `RULESET_VERSION` 到 `8-reddit-*` 并全量重扫，而不是逐条落码逐次重扫。

## 5. Actionable

- ⚠️ **本 cell 的 success 含 task 160（B-1889 benchmark-FP）**。若排除，SR 6.83% → 6.34%。排除与否属 prereg 级改动，**待 user / advisor 决策**，本 digest 不自行调整数字。
- 未发现需要开 B-number 的 scaffold-bug（本轮范围内）。

---

**Cross-link**: 笔记 §387.6 / §387.7 · master_bug_catalog B-1889 (task 160 passive-FP) / B-1890 (footprint 字段恒 0，勿用作判据) · `/tmp/diag_red/` Tier-1 原始扫描产物
