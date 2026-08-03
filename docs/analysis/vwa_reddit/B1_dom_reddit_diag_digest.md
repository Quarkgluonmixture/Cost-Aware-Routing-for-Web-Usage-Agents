# /diag digest — B1 × `dom` × reddit

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
| **Run** | `B1_dom_reddit_20260703` |
| **Condition** | `phase1_dom_router_0` |
| **Site / Mode / Model** | reddit / `dom` / B1 = Qwen3-VL-4B (local) |
| **Episodes** | 205 |
| **SR** | **6.83%** (14 success / 191 failed) |
| **ruleset_version** | `8-reddit-p41p46-b1890fix` |
| **Tier-1 三子集** | failed+hit 174 · **failed-NO-hit 17** · success+hit 4 |

## 2. Tier-1 规则分布（failed 侧）

| 规则 | 含义 | step-level hits | 命中 episode 数 |
|---|---|---|---|
| `P36` | WALK_FAIL_DEGENERATE | 631 | 111 |
| `P31` | budget 耗尽未完成 | 135 | 135 |
| `P5` | 感知缺失循环 | 96 | 67 |
| `P44` | HALLUCINATED_ELEMENT_REF | 82 | 23 |
| `P45` | IDENTICAL_FAILED_ACTION_STREAK | 62 | 47 |
| `P43` | PAGE_EMBEDDED_VISUAL_NO_SCREENSHOT(中性标签) | 60 | 60 |
| `P33` | 导航至裸图片 URL 幻觉 | 49 | 49 |
| `P14` | URL 自环 | 37 | 33 |
| `P25` | 跨站任务跳过其中一站 | 29 | 29 |
| `P12` | 从不翻页 | 16 | 16 |
| `P10` | 跨步数值记忆失败 | 11 | 5 |
| `P46` | COMMENT_INTENT_NO_TYPE | 9 | 9 |
| `P13` | 搜索代替浏览 | 2 | 2 |
| `P27` | 找不到即放弃 | 1 | 1 |

**success 侧 fire 的规则（presence-only 误报审计对象）**: `P33`×2, `P25`×1, `P42`×1, `P41`×1

**failed-NO-hit episode（deterministic 盲区）**: [2, 6, 14, 23, 38, 74, 118, 138, 139, 145, 172, 189, 191, 193, 199, 200, 205]

**success episode**: [17, 18, 36, 40, 42, 58, 92, 129, 130, 131, 160, 171, 179, 188]


## 3. Tier-2 深挖

**覆盖范围**：7 ep（no-hit 分层抽样 5 + success 审计 2）· 1 sonnet sub-agent

**三分类**：agent-limit 5 · benchmark-FP 2（两个 success 均判 FP）· scaffold-bug 0 · unclear 0

### P36 因果审计

B1 **也有** perseveration，但形态是「谱系」而非 B2 那种单一死循环：task 160 在 step 3/4/5/6 连续 4 次同一 `walk_fail:no_actionable_within_walk`（同 element_id=3949），step 15/16 复发 2 次；task 114 则是语义级循环（连续 17/23 步在三个 forum 名的字面搜索间打转），最终靠切换到直接 URL 导航跳出，代价是耗掉 74% 预算。即「locator 层刚性重复（未能自纠）」到「策略层松散重复但最终自纠」的连续谱。

### 具体发现

- ⚠️ **P36 计数被系统性低估**：task 160 真实发生了 6 次 walk_fail，但因该 episode `success=true`、而 P36 对 success episode 直接 `return []`，这些完全没进 Tier-1 统计。（v8 未改此行为——success-safe 是刻意设计，但读 P36 数字时要知道它只覆盖 failed 侧。）
- **task 91 / 95 / 102** — dom 模式 `input_image=0`，任务要读帖子配图的颜色/计数。task 102 诚实认输，**task 95 则在零视觉输入下自信幻觉**（thought 称「I can see the snow... appears white」，真值 purple/pink，confidence 0.95）。
- **task 138** — 正确从参考图提取姓名 Patrick、正确导航到 account 页、正确输入用户名，**但直接 finish 未点任何 Save/提交**，修改未持久化。这是「差最后一步」类失败。
- **task 58 / 160** — 两个 success 均判 benchmark-FP（→ B-1892 / B-1889）。

### 为什么这个 cell 是 6.83%

B1 在「放弃」与「固执」之间偏向**过早放弃**（多个 episode 1-2 步内 confidence=0.0 直接 finish），而 B2 偏向**过度固执**。量化对照：B1_som 188 failed 中 P36 命中 54.8% / P31 命中 67.0%；B2_som 202 failed 中 64.9% / 83.7% —— 两项 B2 都显著更高。

## 4. 🔁 Self-evolving — 提议规则

- `P-unsaved-form`（最后一个非 finish 动作是 type 表单字段，其后无提交类 click 即 finish，且 eval 要求字段持久化）—— 命中 task 138 这类「差最后一步」
- P27 `ABANDONMENT_RE` 扩充 'unable to determine' + 同时扫 `thought` 字段（现仅扫 answer/text）

> 这些提议**尚未落码**。按 discover-then-freeze 纪律，reddit 六 mode × 三 model 的 discover 产物应合并成一批（R1–R8 + H2）后统一 bump `RULESET_VERSION` 到 `8-reddit-*` 并全量重扫，而不是逐条落码逐次重扫。

## 5. Actionable

- ⚠️ **本 cell 的 success 含 task 160（B-1889 benchmark-FP）**。若排除，SR 6.83% → 6.34%。排除与否属 prereg 级改动，**待 user / advisor 决策**，本 digest 不自行调整数字。
- 未发现需要开 B-number 的 scaffold-bug（本轮范围内）。

---

**Cross-link**: 笔记 §387.6 / §387.7 · master_bug_catalog B-1889 (task 160 passive-FP) / B-1890 (footprint 字段恒 0，勿用作判据) · `/tmp/diag_red/` Tier-1 原始扫描产物

---

### v11 数字块（`11-intent-text-fallback`，2026-08-03 补）

> 本 digest 正文成稿于更早的 ruleset。v10 落了 **+P49 / P36 carve-out / P14 carve-out**，
> v11 给 **P34/P48 换用 `_finish_intent_text()`**（answer 为空时 fallback 读 `thought`——
> B0 惯于把结论写进 `answer`，B1 留在 `thought`，旧口径因此变成了模型行为检测器）。
> 全部 48 个 canonical condition 已在 v11 下重扫，**cross-mode / cross-model 聚合以本块为准**。

| 字段 | 值 |
|---|---|
| Run | `B1_dom_reddit_20260703` |
| Episodes | 205（success 14 · SR 6.83%） |
| 三子集 | failed+hit 173 · failed-NO-hit 18 · success+hit 4 |
| config_missing | 0 |

| 规则 | 含义 | step 级 | episode 级 |
|---|---|---:|---:|
| `P31` | budget耗尽未完成 | 135 | 135 |
| `P36` | WALK_FAIL_DEGENERATE | 504 | 95 |
| `P5` | 感知缺失循环 | 96 | 67 |
| `P43` | PAGE_EMBEDDED_VISUAL_NO_SCREENSHOT | 60 | 60 |
| `P33` | 导航至裸图片URL幻觉 | 49 | 49 |
| `P45` | IDENTICAL_FAILED_ACTION_STREAK | 62 | 47 |
| `P14` | URL 自环 | 36 | 32 |
| `P25` | 跨站任务跳过其中一站 | 29 | 29 |
| `P44` | HALLUCINATED_ELEMENT_REF | 82 | 23 |
| `P12` | 从不翻页 | 16 | 16 |
| `P46` | COMMENT_INTENT_NO_TYPE | 9 | 9 |
| `P10` | 跨步数值记忆失败 | 11 | 5 |
| `P13` | 搜索代替浏览 | 2 | 2 |
| `P27` | 找不到即放弃 | 1 | 1 |
| `P49` | SUBMIT_PAGE_ANCHOR_MISCLICK | 1 | 1 |

> ⚠️ **解读约束**（`docs/analysis/_data_quality_audit.md`）：
> ① 本表是**症状分布，不是死因分布** —— P36/P31 经 10 例跨 benchmark 因果验证均判为 risk-marker；
> ② `P2`/`P4` 依赖 `element_bbox`，在 **vision 上结构性为 0（假 0）**；
> ③ `P36` 在 vision 上只覆盖 `type` 步（click 无 `locator_route_meta`）→ **分母与 dom/som 不同**。
