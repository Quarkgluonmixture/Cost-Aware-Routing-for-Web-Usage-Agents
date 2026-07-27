# /diag digest — B2 × `dom` × reddit

*生成 2026-07-27（Tier-1 全量 + Tier-2 未深挖）*

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
| **Run** | `B2_dom_reddit_20260715` |
| **Condition** | `phase1_dom_router_0` |
| **Site / Mode / Model** | reddit / `dom` / B2 = Gemma3-VL `google/gemma-3-4b-it` (local) |
| **Episodes** | 205 |
| **SR** | **3.90%** (8 success / 197 failed) |
| **ruleset_version** | `8-reddit-p41p46-b1890fix` |
| **Tier-1 三子集** | failed+hit 195 · **failed-NO-hit 2** · success+hit 1 |

## 2. Tier-1 规则分布（failed 侧）

| 规则 | 含义 | step-level hits | 命中 episode 数 |
|---|---|---|---|
| `P36` | WALK_FAIL_DEGENERATE | 1670 | 159 |
| `P44` | HALLUCINATED_ELEMENT_REF | 873 | 122 |
| `P5` | 感知缺失循环 | 261 | 149 |
| `P45` | IDENTICAL_FAILED_ACTION_STREAK | 257 | 143 |
| `P31` | budget 耗尽未完成 | 183 | 183 |
| `P14` | URL 自环 | 117 | 92 |
| `P43` | PAGE_EMBEDDED_VISUAL_NO_SCREENSHOT(中性标签) | 63 | 63 |
| `P33` | 导航至裸图片 URL 幻觉 | 46 | 46 |
| `P12` | 从不翻页 | 37 | 37 |
| `P25` | 跨站任务跳过其中一站 | 31 | 31 |
| `P46` | COMMENT_INTENT_NO_TYPE | 5 | 5 |
| `P4` | 根节点误操作 | 4 | 1 |
| `P10` | 跨步数值记忆失败 | 2 | 1 |
| `P13` | 搜索代替浏览 | 1 | 1 |

**success 侧 fire 的规则（presence-only 误报审计对象）**: `P33`×1

**failed-NO-hit episode（deterministic 盲区）**: [64, 171]

**success episode**: [105, 107, 130, 138, 150, 178, 188, 189]


## 3. Tier-2 深挖

**本轮未做 Tier-2 深挖。**

依 /diag skill 的跨-condition 预算纪律，Tier-2 只投给 (a) SR 异常低 / (b) 新 site-mode / (c) no-hit 比例 >25% 的 condition。本 condition 的 SR 落在该 model 的常规区间、no-hit 比例为 1.0%（<25%），故本轮排在 B2 六条之后。

**待深挖子集已就绪**：failed-NO-hit 2 个（见 §2 列表）+ success-with-hits 1 个（presence-only 误报审计）。

⚠️ 因此本 digest 的三分类**不完整** —— 未深挖不等于「无 scaffold-bug / 无 benchmark-FP」，只代表本轮没有查。请勿据此下「pipeline 干净」结论。

## 4. 🔁 Self-evolving — 提议规则

待 Tier-2 深挖后补。

## 5. Actionable

- 本 cell 的 success 不含 task 160（B-1889 不影响本 cell 的 SR）。
- scaffold-bug 情况未知（Tier-2 未做）。

---

**Cross-link**: 笔记 §387.6 / §387.7 · master_bug_catalog B-1889 (task 160 passive-FP) / B-1890 (footprint 字段恒 0，勿用作判据) · `/tmp/diag_red/` Tier-1 原始扫描产物
