# /diag digest — B1 × `phantom_som` × reddit

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
| **Run** | `B1_phantom_som_reddit_20260711` |
| **Condition** | `phase1_phantom_som_router_0` |
| **Site / Mode / Model** | reddit / `phantom_som` / B1 = Qwen3-VL-4B (local) |
| **Episodes** | 205 |
| **SR** | **6.83%** (14 success / 191 failed) |
| **ruleset_version** | `8-reddit-p41p46-b1890fix` |
| **Tier-1 三子集** | failed+hit 181 · **failed-NO-hit 10** · success+hit 3 |

## 2. Tier-1 规则分布（failed 侧）

| 规则 | 含义 | step-level hits | 命中 episode 数 |
|---|---|---|---|
| `P36` | WALK_FAIL_DEGENERATE | 1156 | 137 |
| `P31` | budget 耗尽未完成 | 149 | 149 |
| `P5` | 感知缺失循环 | 146 | 101 |
| `P4` | 根节点误操作 | 144 | 31 |
| `P45` | IDENTICAL_FAILED_ACTION_STREAK | 111 | 83 |
| `P14` | URL 自环 | 88 | 73 |
| `P43` | PAGE_EMBEDDED_VISUAL_NO_SCREENSHOT(中性标签) | 60 | 60 |
| `P33` | 导航至裸图片 URL 幻觉 | 45 | 45 |
| `P25` | 跨站任务跳过其中一站 | 39 | 39 |
| `P12` | 从不翻页 | 31 | 31 |
| `P46` | COMMENT_INTENT_NO_TYPE | 12 | 12 |
| `P10` | 跨步数值记忆失败 | 6 | 1 |
| `P27` | 找不到即放弃 | 4 | 4 |
| `P44` | HALLUCINATED_ELEMENT_REF | 3 | 2 |

**success 侧 fire 的规则（presence-only 误报审计对象）**: `P25`×1, `P42`×1, `P33`×1, `P41`×1

**failed-NO-hit episode（deterministic 盲区）**: [2, 38, 74, 138, 139, 173, 174, 191, 195, 200]

**success episode**: [0, 36, 40, 42, 58, 77, 129, 130, 131, 160, 171, 178, 188, 189]


## 3. Tier-2 深挖

**覆盖范围**：7 ep（no-hit 5 + success 2）· 1 sonnet sub-agent（首次因 session limit 中断，已重放）+ 我的独立全量复算

**三分类**：agent-limit 5 · benchmark-FP 2 · scaffold-bug 0 · unclear 0

### P36 因果审计

见下方两层核查 —— 结论是 walk_fail **既非 P-SoM 特有也不随能力单调**。

### 具体发现

- ⭐ **[SOM_MARKS] 两层核查（这是 hero mode 能否宣称 scaffold 干净的关键证据）**。我在 sub-agent 数字基础上补了 dom/som 对照，结果比原报告强得多：
-   · **(a) 幻觉引用率**（引用了 observation 里不存在的 element_id）：P-SoM **B0 0.04% / B1 0.12% / B2 7.84%**；同 model 的 dom 是 **0.39% / 2.98% / 18.21%**。→ **dom 在每个模型上都最差，P-SoM 干净 5–25×**。机制：dom 用原生 AXTree id（median 7839–18729，max 691695），P-SoM 用紧凑编号 1..N（median 15–17，max 176）—— 抄 5-6 位稀疏整数 vs 2-3 位紧凑编号。→ 这条催生了 **P44**。
-   · **(b) walk_fail 率**：P-SoM B0 13.3% / B1 29.5% / B2 21.9%；dom 23.5% / 18.5% / 35.2%。**3 个模型里 2 个是 dom 更差**，且不随能力单调 → 在 (model, mode) 格间就是噪声，**不能写成「walk 可执行性随能力劣化」**。
- ⚠️ **同时修正了 4 个 sub-agent 的集体误判**：它们都断言 `obs_nodes_info missing union_bound`（幻觉引用分支）「一次都没出现」。在各自 6–8 个样本里成立，**总体上不成立**（B2 上 374 次 psom / 895 次 dom）。walk_fail 与幻觉引用是**并存**的两条分支。
- **task 19** — 点 [SOM_MARKS] 里的 img href 跳到 `/submission_images/*.jpg` 裸图页（reddit 版 P33），旧正则漏检。

### 为什么这个 cell 是 6.83%

P-SoM 的失败集中在「无页面截图 → 无法把参考图与页面缩略图做比对」（task 19/139），以及与其他 mode 共通的 perseveration。**scaffold 层在 element-引用维度不仅干净，而且优于 dom。**

## 4. 🔁 Self-evolving — 提议规则

- → 已落码为 **P44**（HALLUCINATED_ELEMENT_REF，此前零覆盖）与 **P33 reddit 路径扩展**

> 这些提议**尚未落码**。按 discover-then-freeze 纪律，reddit 六 mode × 三 model 的 discover 产物应合并成一批（R1–R8 + H2）后统一 bump `RULESET_VERSION` 到 `8-reddit-*` 并全量重扫，而不是逐条落码逐次重扫。

## 5. Actionable

- ⚠️ **本 cell 的 success 含 task 160（B-1889 benchmark-FP）**。若排除，SR 6.83% → 6.34%。排除与否属 prereg 级改动，**待 user / advisor 决策**，本 digest 不自行调整数字。
- 未发现需要开 B-number 的 scaffold-bug（本轮范围内）。

---

**Cross-link**: 笔记 §387.6 / §387.7 · master_bug_catalog B-1889 (task 160 passive-FP) / B-1890 (footprint 字段恒 0，勿用作判据) · `/tmp/diag_red/` Tier-1 原始扫描产物
