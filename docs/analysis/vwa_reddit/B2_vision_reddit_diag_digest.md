# /diag digest — B2 × `vision` × reddit

*生成 2026-07-27（Tier-1 全量 + Tier-2 深挖）*

> **定位声明**：本 digest 是**单 condition** 的失败归因，不下 cross-mode / cross-model 结论。
> 跨 mode 定量比较须等 reddit 规则批（R1–R8 + H2）落地、`RULESET_VERSION` 升到 `8-reddit-*`
> 并全量重扫后再做（/diag skill「discover-then-freeze」硬纪律）。


## 1. Header

| 字段 | 值 |
|---|---|
| **Run** | `B2_vision_reddit_20260719` |
| **Condition** | `phase1_vision_router_0` |
| **Site / Mode / Model** | reddit / `vision` / B2 = Gemma3-VL `google/gemma-3-4b-it` (local) |
| **Episodes** | 205 |
| **SR** | **2.44%** (5 success / 200 failed) |
| **ruleset_version** | `7-p6p16clsgate-b1860coord` |
| **Tier-1 三子集** | failed+hit 192 · **failed-NO-hit 8** · success+hit 0 |

## 2. Tier-1 规则分布（failed 侧）

| 规则 | 含义 | step-level hits | 命中 episode 数 |
|---|---|---|---|
| `P36` | WALK_FAIL_DEGENERATE | 456 | 84 |
| `P5` | 感知缺失循环 | 281 | 147 |
| `P31` | budget 耗尽未完成 | 170 | 170 |
| `P14` | URL 自环 | 94 | 79 |
| `P1` | 元素中心越界 | 57 | 9 |
| `P25` | 跨站任务跳过其中一站 | 33 | 33 |
| `P12` | 从不翻页 | 15 | 15 |
| `P10` | 跨步数值记忆失败 | 3 | 3 |
| `P27` | 找不到即放弃 | 1 | 1 |

**success 侧 fire 的规则**: 无（success 侧 0 命中）

**failed-NO-hit episode（deterministic 盲区）**: [64, 69, 76, 89, 91, 103, 141, 148]

**success episode**: [77, 78, 98, 120, 160]


## 3. Tier-2 深挖

**覆盖范围**：8 ep（no-hit 7 + success 1）· 1 sonnet sub-agent

**三分类**：agent-limit 7 · benchmark-FP 1（task 160）· scaffold-bug 0 · unclear 0

### 具体发现

- **主导失败模式是「动作模态错误」，比 grounding/perception 更根本**：5 个要求「发一条真实评论」的任务（69/76/89/91/103）里，模型从未使用 `type`，一律用 `finish(answer=...)` 把答案当文字描述交上去。**task 103 的视觉判断完全正确**（'blue' 与 reference 字面精确一致），但答案没写进评论框，评测读站点内容时依然判失败。
- **坐标映射类 scaffold bug 已排除**：全部 8 个 episode `image_meta_recorded=True`、`input_image_tokens=768`（截图确实送进模型）；click 的 `coordinate_normalization` 全部 `recovered=true / true_oob=false / malformed=false`，无系统性错位。点击命中的是「错的」元素（帖子缩略图本身指向裸图文件），属语义级选错目标。
- **submission_images URL 陷阱**：点击帖内缩略图直接跳到 `/submission_images/*.jpg` 裸图页，丢失评论框上下文。vision mode 无语义标签只能靠坐标猜，比 dom/som 更容易踩中（站点结构 + 模型选择的复合问题，非 runner bug）。
- **task 160 = benchmark-FP** → B-1889。

### 为什么这个 cell 是 2.44%

vision mode（只有坐标、没有 element_id）下模型把「评论/发帖类操作任务」系统性误解成「纯 QA 问答任务」，跨 5/7 个 no-hit episode 可复现，比单个 perception 误判（数错 Jupiter 数量、少读一个零）更具规律性。

## 4. 🔁 Self-evolving — 提议规则

- `never-posted-comment`：eval locator 含 `reddit_get_latest_comment_content_by_username` 类函数 且全程 `type` 成功次数=0 → 把「答案生成了但从未提交」从「感知错误」里精确剥离
- `success-via-inaction`：success 且 agent_finished=false 且全程 obs_url 唯一值=1（=start_url）→ 强制复核

> 这些提议**尚未落码**。按 discover-then-freeze 纪律，reddit 六 mode × 三 model 的 discover 产物应合并成一批（R1–R8 + H2）后统一 bump `RULESET_VERSION` 到 `8-reddit-*` 并全量重扫，而不是逐条落码逐次重扫。

## 5. Actionable

- ⚠️ **本 cell 的 success 含 task 160（B-1889 benchmark-FP）**。若排除，SR 2.44% → 1.95%。排除与否属 prereg 级改动，**待 user / advisor 决策**，本 digest 不自行调整数字。
- 未发现需要开 B-number 的 scaffold-bug（本轮范围内）。

---

**Cross-link**: 笔记 §387.6 / §387.7 · master_bug_catalog B-1889 (task 160 passive-FP) / B-1890 (footprint 字段恒 0，勿用作判据) · `/tmp/diag_red/` Tier-1 原始扫描产物
