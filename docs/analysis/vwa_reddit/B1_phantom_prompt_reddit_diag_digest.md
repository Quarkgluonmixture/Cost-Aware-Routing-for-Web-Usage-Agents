# /diag digest — B1 × `phantom_prompt` × reddit

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
| **Run** | `B1_phantom_prompt_reddit_20260713` |
| **Condition** | `phase1_phantom_prompt_router_0` |
| **Site / Mode / Model** | reddit / `phantom_prompt` / B1 = Qwen3-VL-4B (local) |
| **Episodes** | 205 |
| **SR** | **6.34%** (13 success / 192 failed) |
| **ruleset_version** | `8-reddit-p41p46-b1890fix` |
| **Tier-1 三子集** | failed+hit 183 · **failed-NO-hit 9** · success+hit 4 |

## 2. Tier-1 规则分布（failed 侧）

| 规则 | 含义 | step-level hits | 命中 episode 数 |
|---|---|---|---|
| `P36` | WALK_FAIL_DEGENERATE | 904 | 124 |
| `P31` | budget 耗尽未完成 | 137 | 137 |
| `P44` | HALLUCINATED_ELEMENT_REF | 137 | 31 |
| `P5` | 感知缺失循环 | 114 | 80 |
| `P45` | IDENTICAL_FAILED_ACTION_STREAK | 81 | 59 |
| `P14` | URL 自环 | 61 | 49 |
| `P43` | PAGE_EMBEDDED_VISUAL_NO_SCREENSHOT(中性标签) | 60 | 60 |
| `P33` | 导航至裸图片 URL 幻觉 | 45 | 45 |
| `P12` | 从不翻页 | 34 | 34 |
| `P25` | 跨站任务跳过其中一站 | 33 | 33 |
| `P46` | COMMENT_INTENT_NO_TYPE | 11 | 11 |
| `P10` | 跨步数值记忆失败 | 2 | 2 |
| `P4` | 根节点误操作 | 1 | 1 |
| `P27` | 找不到即放弃 | 1 | 1 |
| `P13` | 搜索代替浏览 | 1 | 1 |

**success 侧 fire 的规则（presence-only 误报审计对象）**: `P33`×2, `P25`×1, `P42`×1, `P41`×1

**failed-NO-hit episode（deterministic 盲区）**: [38, 67, 87, 118, 119, 132, 138, 173, 191]

**success episode**: [36, 40, 58, 92, 129, 131, 153, 160, 167, 171, 179, 189, 200]


## 3. Tier-2 深挖

**覆盖范围**：7 ep（no-hit 5 + success 2）· 1 sonnet sub-agent

**三分类**：agent-limit 5 · benchmark-FP 2 · scaffold-bug 0 · unclear 0

### P36 因果审计

B1 在同一构造下也有 perseveration（task 142 连续 12 步猜不同 element_id），但**恢复能力明显更强** —— 最终靠 scroll 找回评论框并完成提交动作。

### 具体发现

- ⭐ **同一 P-prompt 构造下 B1 与 B2 的差异，正是「SR 梯度反映能力而非构造缺陷」的直接证据**。代码层确认两者面对**完全相同**的构造（`mark_count=0`、element_id 用原生 AXTree id、SoM prompt 仍宣称会给标注截图但从不发图）。差异在应对：(a) **B1 校准更诚实** —— task 152 直接给 confidence=0.0 并拒答，不像 B2 那样固定虚高 0.95；(b) **B1 会恢复** —— task 142 在 12 次误点后自行脱困。
- **幻觉措辞要分两类，不能混为一谈**：真幻觉（task 152 逐字出现「no image is visible in **the provided screenshot**」，而该 mode 从未提供 screenshot）vs 术语混用（task 132/138 说「the image」实指**真实存在**的任务级参考图 —— 参考图所有 mode 都发，不算幻觉）。
- **比幻觉更危险的模式**：task 142 编造具体日期、task 58 编造「评论里写着」的假引用来源，且配 0.95–1.0 高置信度。与 B2 的核心风险同质，只是发生率低得多。

### 为什么这个 cell 是 6.34%

构造缺陷是共同的，能力决定了伤害大小 —— 这正是 B0 12.68% > B1 6.34% > B2 0.49% 梯度的解释。

## 4. 🔁 Self-evolving — 提议规则

- → 已落码为 **P43**（中性标签版）

> 这些提议**尚未落码**。按 discover-then-freeze 纪律，reddit 六 mode × 三 model 的 discover 产物应合并成一批（R1–R8 + H2）后统一 bump `RULESET_VERSION` 到 `8-reddit-*` 并全量重扫，而不是逐条落码逐次重扫。

## 5. Actionable

- ⚠️ **本 cell 的 success 含 task 160（B-1889 benchmark-FP）**。若排除，SR 6.34% → 5.85%。排除与否属 prereg 级改动，**待 user / advisor 决策**，本 digest 不自行调整数字。
- 未发现需要开 B-number 的 scaffold-bug（本轮范围内）。

---

**Cross-link**: 笔记 §387.6 / §387.7 · master_bug_catalog B-1889 (task 160 passive-FP) / B-1890 (footprint 字段恒 0，勿用作判据) · `/tmp/diag_red/` Tier-1 原始扫描产物
