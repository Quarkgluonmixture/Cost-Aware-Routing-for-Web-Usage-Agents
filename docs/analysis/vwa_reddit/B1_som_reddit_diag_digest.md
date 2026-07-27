# /diag digest — B1 × `som` × reddit

*生成 2026-07-27（Tier-1 全量 + Tier-2 未深挖）*

> **定位声明**：本 digest 是**单 condition** 的失败归因，不下 cross-mode / cross-model 结论。
> 跨 mode 定量比较须等 reddit 规则批（R1–R8 + H2）落地、`RULESET_VERSION` 升到 `8-reddit-*`
> 并全量重扫后再做（/diag skill「discover-then-freeze」硬纪律）。


## 1. Header

| 字段 | 值 |
|---|---|
| **Run** | `B1_som_reddit_20260706` |
| **Condition** | `phase1_som_router_0` |
| **Site / Mode / Model** | reddit / `som` / B1 = Qwen3-VL-4B (local) |
| **Episodes** | 205 |
| **SR** | **8.29%** (17 success / 188 failed) |
| **ruleset_version** | `7-p6p16clsgate-b1860coord` |
| **Tier-1 三子集** | failed+hit 158 · **failed-NO-hit 30** · success+hit 2 |

## 2. Tier-1 规则分布（failed 侧）

| 规则 | 含义 | step-level hits | 命中 episode 数 |
|---|---|---|---|
| `P36` | WALK_FAIL_DEGENERATE | 850 | 103 |
| `P31` | budget 耗尽未完成 | 126 | 126 |
| `P5` | 感知缺失循环 | 113 | 77 |
| `P4` | 根节点误操作 | 106 | 22 |
| `P14` | URL 自环 | 60 | 49 |
| `P25` | 跨站任务跳过其中一站 | 38 | 38 |
| `P12` | 从不翻页 | 26 | 26 |
| `P27` | 找不到即放弃 | 5 | 5 |
| `P10` | 跨步数值记忆失败 | 2 | 2 |

**success 侧 fire 的规则（presence-only 误报审计对象）**: `P25`×1, `P27`×1

**failed-NO-hit episode（deterministic 盲区）**: [14, 22, 38, 60, 70, 78, 87, 89, 90, 91, 92, 93, 94, 95, 96, 97, 102, 103, 104, 125, 132, 134, 141, 156, 169, 175, 191, 192, 201, 203]

**success episode**: [0, 18, 36, 40, 42, 58, 77, 120, 129, 130, 131, 139, 160, 171, 188, 189, 200]


## 3. Tier-2 深挖

**本轮未做 Tier-2 深挖。**

依 /diag skill 的跨-condition 预算纪律，Tier-2 只投给 (a) SR 异常低 / (b) 新 site-mode / (c) no-hit 比例 >25% 的 condition。本 condition 的 SR 落在该 model 的常规区间、no-hit 比例为 14.6%（<25%），故本轮排在 B2 六条之后。

**待深挖子集已就绪**：failed-NO-hit 30 个（见 §2 列表）+ success-with-hits 2 个（presence-only 误报审计）。

⚠️ 因此本 digest 的三分类**不完整** —— 未深挖不等于「无 scaffold-bug / 无 benchmark-FP」，只代表本轮没有查。请勿据此下「pipeline 干净」结论。

## 4. 🔁 Self-evolving — 提议规则

待 Tier-2 深挖后补。

## 5. Actionable

- ⚠️ **本 cell 的 success 含 task 160（B-1889 benchmark-FP）**。若排除，SR 8.29% → 7.80%。排除与否属 prereg 级改动，**待 user / advisor 决策**，本 digest 不自行调整数字。
- scaffold-bug 情况未知（Tier-2 未做）。

---

**Cross-link**: 笔记 §387.6 / §387.7 · master_bug_catalog B-1889 (task 160 passive-FP) / B-1890 (footprint 字段恒 0，勿用作判据) · `/tmp/diag_red/` Tier-1 原始扫描产物
