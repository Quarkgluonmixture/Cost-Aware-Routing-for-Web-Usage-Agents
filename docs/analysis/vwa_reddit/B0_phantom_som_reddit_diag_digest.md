# /diag digest — B0 × `phantom_som` × reddit

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
| **Run** | `B0_phantom_som_reddit_20260701_223127_661875492_3649813_R28173` |
| **Condition** | `phase1_phantom_som_router_0` |
| **Site / Mode / Model** | reddit / `phantom_som` / B0 = Qwen3-VL-235B-A22B (proxy) |
| **Episodes** | 205 |
| **SR** | **11.22%** (23 success / 182 failed) |
| **ruleset_version** | `8-reddit-p41p46-b1890fix` |
| **Tier-1 三子集** | failed+hit 167 · **failed-NO-hit 15** · success+hit 7 |

## 2. Tier-1 规则分布（failed 侧）

| 规则 | 含义 | step-level hits | 命中 episode 数 |
|---|---|---|---|
| `P36` | WALK_FAIL_DEGENERATE | 346 | 81 |
| `P5` | 感知缺失循环 | 121 | 73 |
| `P31` | budget 耗尽未完成 | 112 | 112 |
| `P33` | 导航至裸图片 URL 幻觉 | 64 | 64 |
| `P14` | URL 自环 | 61 | 50 |
| `P43` | PAGE_EMBEDDED_VISUAL_NO_SCREENSHOT(中性标签) | 57 | 57 |
| `P4` | 根节点误操作 | 54 | 14 |
| `P45` | IDENTICAL_FAILED_ACTION_STREAK | 44 | 25 |
| `P25` | 跨站任务跳过其中一站 | 34 | 34 |
| `P12` | 从不翻页 | 30 | 30 |
| `P46` | COMMENT_INTENT_NO_TYPE | 4 | 4 |
| `P27` | 找不到即放弃 | 2 | 2 |
| `P13` | 搜索代替浏览 | 1 | 1 |
| `P10` | 跨步数值记忆失败 | 1 | 1 |

**success 侧 fire 的规则（presence-only 误报审计对象）**: `P33`×6, `P27`×1, `P41`×1

**failed-NO-hit episode（deterministic 盲区）**: [5, 20, 21, 32, 38, 134, 138, 144, 145, 171, 172, 181, 198, 202, 203]

**success episode**: [0, 11, 12, 18, 19, 36, 40, 42, 94, 99, 100, 107, 129, 130, 131, 142, 155, 157, 160, 178, 179, 188, 200]


## 3. Tier-2 深挖

**覆盖范围**：6 ep（no-hit 5 + success 1）· 1 sonnet sub-agent + 全 condition 0-token 结构扫描

**三分类**：agent-limit 5 · benchmark-FP 1 · scaffold-bug 0 · unclear 0

### P36 因果审计

见 B1_phantom_som_reddit 的两层核查表 —— B0 是其中的干净端（幻觉 0.04% / walk_fail 13.3%）。

### 具体发现

- ⭐ **[SOM_MARKS] 一致性必须分两层说，不能一句「一致」带过**（这条方法论提醒来自本 agent，很到位）：
-   · **存在性层：非常干净** —— 2796 个带 element_id 的 action 里仅 1 例越界（0.036%）。**这一层可以放心写进论文正文。**
-   · **可执行性层：不能说零** —— walk_fail 覆盖 88/205 episode、356/4669 step（7.6%），其中 304 步最终失败。建议写法：可写「[SOM_MARKS] 编号幻觉率 <0.1%」，但**不要**无保留地写「零列了点不动」。
- **task 82 / 202** — 「多目标任务提前收工」：eval 要求 8 个 / 11 个不同目标，agent 只碰了 1 个就 finish 并自称全部完成。
- **task 120** — 严格结构性不可解：start_url 本身就是裸图片、无参考图、DOM 无内容。

### 为什么这个 cell 是 11.22%

P-SoM 在 B0 上的失败以「参考图↔页面缩略图无法比对」和「多目标覆盖不全」为主，scaffold 层干净。

## 4. 🔁 Self-evolving — 提议规则

- `SINGLE_TARGET_FINISH_ON_MULTI_TARGET_TASK`（eval must_include 含 N>1 个实体但轨迹交互的 distinct target < N 即 finish）—— mode-agnostic，本批**未落码**，留待下一轮（需要实体抽取，非纯字段比较）

> 这些提议**尚未落码**。按 discover-then-freeze 纪律，reddit 六 mode × 三 model 的 discover 产物应合并成一批（R1–R8 + H2）后统一 bump `RULESET_VERSION` 到 `8-reddit-*` 并全量重扫，而不是逐条落码逐次重扫。

## 5. Actionable

- ⚠️ **本 cell 的 success 含 task 160（B-1889 benchmark-FP）**。若排除，SR 11.22% → 10.73%。排除与否属 prereg 级改动，**待 user / advisor 决策**，本 digest 不自行调整数字。
- 未发现需要开 B-number 的 scaffold-bug（本轮范围内）。

---

**Cross-link**: 笔记 §387.6 / §387.7 · master_bug_catalog B-1889 (task 160 passive-FP) / B-1890 (footprint 字段恒 0，勿用作判据) · `/tmp/diag_red/` Tier-1 原始扫描产物
