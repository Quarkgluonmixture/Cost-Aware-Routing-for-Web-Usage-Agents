# /diag digest — B0 × `phantom_prompt` × reddit

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
| **Run** | `B0_phantom_prompt_reddit_20260709` |
| **Condition** | `phase1_phantom_prompt_router_0` |
| **Site / Mode / Model** | reddit / `phantom_prompt` / B0 = Qwen3-VL-235B-A22B (proxy) |
| **Episodes** | 205 |
| **SR** | **12.68%** (26 success / 179 failed) |
| **ruleset_version** | `8-reddit-p41p46-b1890fix` |
| **Tier-1 三子集** | failed+hit 161 · **failed-NO-hit 18** · success+hit 12 |

## 2. Tier-1 规则分布（failed 侧）

| 规则 | 含义 | step-level hits | 命中 episode 数 |
|---|---|---|---|
| `P36` | WALK_FAIL_DEGENERATE | 367 | 69 |
| `P31` | budget 耗尽未完成 | 89 | 89 |
| `P5` | 感知缺失循环 | 86 | 50 |
| `P33` | 导航至裸图片 URL 幻觉 | 72 | 72 |
| `P43` | PAGE_EMBEDDED_VISUAL_NO_SCREENSHOT(中性标签) | 59 | 59 |
| `P45` | IDENTICAL_FAILED_ACTION_STREAK | 52 | 30 |
| `P14` | URL 自环 | 19 | 17 |
| `P12` | 从不翻页 | 15 | 15 |
| `P25` | 跨站任务跳过其中一站 | 15 | 15 |
| `P44` | HALLUCINATED_ELEMENT_REF | 9 | 8 |
| `P46` | COMMENT_INTENT_NO_TYPE | 7 | 7 |
| `P27` | 找不到即放弃 | 4 | 4 |
| `P10` | 跨步数值记忆失败 | 4 | 4 |
| `P4` | 根节点误操作 | 3 | 1 |

**success 侧 fire 的规则（presence-only 误报审计对象）**: `P33`×11, `P25`×1, `P42`×1

**failed-NO-hit episode（deterministic 盲区）**: [5, 13, 20, 22, 33, 38, 55, 60, 62, 76, 106, 129, 144, 146, 163, 177, 196, 197]

**success episode**: [0, 18, 19, 36, 40, 42, 49, 58, 66, 77, 93, 94, 105, 107, 130, 131, 138, 155, 157, 161, 162, 178, 181, 188, 189, 200]


## 3. Tier-2 深挖

**覆盖范围**：6 ep（no-hit 5 + success 1）· 1 sonnet sub-agent + 全 48 no-hit 的 task-config 级扫描

**三分类**：agent-limit 5 · unclear 1 · scaffold-bug 0 · benchmark-FP 0

### 具体发现

- ⭐ **本 condition 的 no-hit 是全 18 条里最多的（48/205 = 23.4%）**，扫描显示 **39/48（81%）命中「图像相关」信号**，本次抽样 5/5 全部落在该桶。→ 这是 P43 落码的最直接依据。
- ⭐ **规则库的结构性偏置**：这类失败「过程干净利落，只是给错了答案」（短 episode、无循环、无预算耗尽、无 URL 自环），**恰好精确避开所有现有 P-rule 的触发条件** —— 现有规则大多是「过程性」病理探测器，而这一整类是「结局性」的。
- 🔍 **一条重要的代码事实核实**（本 agent 主动查证，纠正了初始假设）：B0 proxy 的 `reference_images` **无视 observation_mode 一律真实发送**（task 109 实测 `image_payload_bytes_ref=172032`），只有**页面实时截图**才受 phantom 约束。→ 这条事实后来收窄了另外 4 个 agent 的「phantom = 完全无图」推断。
- **B0 未表现出 B2 的灾难性 perseveration**：5 个 episode 步数 2/13/8/5/23（上限 30），**全部主动 finish，无一跑满预算**。confidence 有起伏（0.7–1.0）而非 B2 的恒定 0.95，但在「盲猜终局」动作上依然普遍偏高。

### 为什么这个 cell 是 12.68%

信息在该 mode 的 substrate 里不存在 → B0 快速合理化猜测后止损。这是能力天花板与表征限制的叠加，但 §387.10 显示补图并不能兑现预期增益。

## 4. 🔁 Self-evolving — 提议规则

- → 已落码为 **P43**（中性标签版，命名刻意避开 sub-agent 提议的「guaranteed fail」）

> 这些提议**尚未落码**。按 discover-then-freeze 纪律，reddit 六 mode × 三 model 的 discover 产物应合并成一批（R1–R8 + H2）后统一 bump `RULESET_VERSION` 到 `8-reddit-*` 并全量重扫，而不是逐条落码逐次重扫。

## 5. Actionable

- 本 cell 的 success 不含 task 160（B-1889 不影响本 cell 的 SR）。
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
| Run | `B0_phantom_prompt_reddit_20260709` |
| Episodes | 205（success 26 · SR 12.68%） |
| 三子集 | failed+hit 159 · failed-NO-hit 20 · success+hit 12 |
| config_missing | 0 |

| 规则 | 含义 | step 级 | episode 级 |
|---|---|---:|---:|
| `P31` | budget耗尽未完成 | 89 | 89 |
| `P33` | 导航至裸图片URL幻觉 | 72 | 72 |
| `P43` | PAGE_EMBEDDED_VISUAL_NO_SCREENSHOT | 59 | 59 |
| `P5` | 感知缺失循环 | 86 | 50 |
| `P36` | WALK_FAIL_DEGENERATE | 334 | 49 |
| `P45` | IDENTICAL_FAILED_ACTION_STREAK | 52 | 30 |
| `P14` | URL 自环 | 18 | 16 |
| `P12` | 从不翻页 | 15 | 15 |
| `P25` | 跨站任务跳过其中一站 | 15 | 15 |
| `P44` | HALLUCINATED_ELEMENT_REF | 9 | 8 |
| `P46` | COMMENT_INTENT_NO_TYPE | 7 | 7 |
| `P27` | 找不到即放弃 | 4 | 4 |
| `P10` | 跨步数值记忆失败 | 4 | 4 |
| `P4` | 根节点误操作 | 3 | 1 |

> ⚠️ **解读约束**（`docs/analysis/_data_quality_audit.md`）：
> ① 本表是**症状分布，不是死因分布** —— P36/P31 经 10 例跨 benchmark 因果验证均判为 risk-marker；
> ② `P2`/`P4` 依赖 `element_bbox`，在 **vision 上结构性为 0（假 0）**；
> ③ `P36` 在 vision 上只覆盖 `type` 步（click 无 `locator_route_meta`）→ **分母与 dom/som 不同**。
