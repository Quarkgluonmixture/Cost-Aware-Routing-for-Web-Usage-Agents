# /diag digest — B1 × `vision` × reddit

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
| **Run** | `B1_vision_reddit_20260708_002122_732634080_205180_R16847` |
| **Condition** | `phase1_vision_router_0` |
| **Site / Mode / Model** | reddit / `vision` / B1 = Qwen3-VL-4B (local) |
| **Episodes** | 205 |
| **SR** | **2.93%** (6 success / 199 failed) |
| **ruleset_version** | `8-reddit-p41p46-b1890fix` |
| **Tier-1 三子集** | failed+hit 187 · **failed-NO-hit 12** · success+hit 2 |

## 2. Tier-1 规则分布（failed 侧）

| 规则 | 含义 | step-level hits | 命中 episode 数 |
|---|---|---|---|
| `P5` | 感知缺失循环 | 237 | 141 |
| `P31` | budget 耗尽未完成 | 143 | 143 |
| `P14` | URL 自环 | 138 | 105 |
| `P12` | 从不翻页 | 40 | 40 |
| `P33` | 导航至裸图片 URL 幻觉 | 33 | 33 |
| `P25` | 跨站任务跳过其中一站 | 27 | 27 |
| `P46` | COMMENT_INTENT_NO_TYPE | 16 | 16 |
| `P36` | WALK_FAIL_DEGENERATE | 12 | 5 |
| `P27` | 找不到即放弃 | 8 | 8 |

**success 侧 fire 的规则（presence-only 误报审计对象）**: `P33`×1, `P27`×1, `P41`×1

**failed-NO-hit episode（deterministic 盲区）**: [1, 2, 38, 87, 88, 133, 138, 143, 156, 171, 192, 196]

**success episode**: [40, 120, 131, 160, 161, 201]


## 3. Tier-2 深挖

**覆盖范围**：6 ep（no-hit 5 + success 1）· 1 sonnet sub-agent + 全 condition 扩展扫描

**三分类**：agent-limit 5 · benchmark-FP 1 · scaffold-bug 0 · unclear 0

### 具体发现

- ⭐ **主导失败是「动作模态错误」而非 grounding 或 perception**：要求发真实评论的任务里，模型从不用 `type`，一律用 `finish(answer=...)` 把答案当文字描述交上去。**task 103 的视觉判断完全正确**（'blue' 与 reference 字面一致）却仍判失败 —— 答案没写进评论框。→ 这条观察催生了 **P46**。
- ✅ **坐标映射无 scaffold bug**：全 condition 2386 个带坐标动作全量扫描，`x_regime`/`y_regime` 全为 `qwen_0_1000`，**0 例 `true_oob`、0 例 `malformed`**，仅 1 例 `dead_zone` 且仍 `recovered=true`。问题在「点哪」不在「点到哪去了」。
- **submission_images 陷阱**：点帖子缩略图直接跳裸图页（缩略图 href 就是图片文件本身）。vision 无语义标签只能靠坐标猜，比 dom/som 更易踩中。→ 这条催生了 **P33 的 reddit 路径扩展**。

### 为什么这个 cell 是 2.93%

动作模态错误 + 语义级选错目标的复合体，且 reddit 站本身评论/发帖类任务占比高，使 vision 在缺少文本结构辅助定位时被放大打击 —— 比 dom (6.83%) 低一半以上。

## 4. 🔁 Self-evolving — 提议规则

- → 已落码为 **P46**（COMMENT_INTENT_NO_TYPE）与 **P33 reddit 路径扩展**

> 这些提议**尚未落码**。按 discover-then-freeze 纪律，reddit 六 mode × 三 model 的 discover 产物应合并成一批（R1–R8 + H2）后统一 bump `RULESET_VERSION` 到 `8-reddit-*` 并全量重扫，而不是逐条落码逐次重扫。

## 5. Actionable

- ⚠️ **本 cell 的 success 含 task 160（B-1889 benchmark-FP）**。若排除，SR 2.93% → 2.44%。排除与否属 prereg 级改动，**待 user / advisor 决策**，本 digest 不自行调整数字。
- 未发现需要开 B-number 的 scaffold-bug（本轮范围内）。

---

**Cross-link**: 笔记 §387.6 / §387.7 · master_bug_catalog B-1889 (task 160 passive-FP) / B-1890 (footprint 字段恒 0，勿用作判据) · `/tmp/diag_red/` Tier-1 原始扫描产物
