# /diag digest — B0 × `phantom_text` × reddit

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
| **Run** | `B0_phantom_text_reddit_20260629_140253_060787566_3384189_R32139` |
| **Condition** | `phase1_phantom_text_router_0` |
| **Site / Mode / Model** | reddit / `phantom_text` / B0 = Qwen3-VL-235B-A22B (proxy) |
| **Episodes** | 205 |
| **SR** | **13.66%** (28 success / 177 failed) |
| **ruleset_version** | `8-reddit-p41p46-b1890fix` |
| **Tier-1 三子集** | failed+hit 157 · **failed-NO-hit 20** · success+hit 10 |

## 2. Tier-1 规则分布（failed 侧）

| 规则 | 含义 | step-level hits | 命中 episode 数 |
|---|---|---|---|
| `P36` | WALK_FAIL_DEGENERATE | 314 | 61 |
| `P31` | budget 耗尽未完成 | 120 | 120 |
| `P5` | 感知缺失循环 | 115 | 69 |
| `P4` | 根节点误操作 | 78 | 19 |
| `P33` | 导航至裸图片 URL 幻觉 | 66 | 66 |
| `P43` | PAGE_EMBEDDED_VISUAL_NO_SCREENSHOT(中性标签) | 56 | 56 |
| `P14` | URL 自环 | 50 | 44 |
| `P45` | IDENTICAL_FAILED_ACTION_STREAK | 48 | 25 |
| `P25` | 跨站任务跳过其中一站 | 37 | 37 |
| `P12` | 从不翻页 | 34 | 34 |
| `P44` | HALLUCINATED_ELEMENT_REF | 6 | 6 |
| `P46` | COMMENT_INTENT_NO_TYPE | 3 | 3 |
| `P13` | 搜索代替浏览 | 1 | 1 |
| `P27` | 找不到即放弃 | 1 | 1 |
| `P10` | 跨步数值记忆失败 | 1 | 1 |

**success 侧 fire 的规则（presence-only 误报审计对象）**: `P33`×9, `P25`×1, `P42`×1

**failed-NO-hit episode（deterministic 盲区）**: [6, 20, 23, 38, 76, 128, 129, 132, 138, 140, 144, 146, 166, 172, 190, 191, 199, 201, 202, 203]

**success episode**: [0, 2, 15, 17, 19, 36, 40, 42, 58, 81, 93, 94, 98, 99, 101, 105, 107, 131, 133, 155, 157, 162, 178, 179, 181, 188, 189, 200]


## 3. Tier-2 深挖

**覆盖范围**：7 ep（no-hit 5 + success 审计 2）· 1 sonnet sub-agent

**三分类**：agent-limit 5 · scaffold-bug 1（P39 误报，见下）· benchmark-FP 1 · unclear 0

### 具体发现

- ⭐ **B0 的失败形态与 B2 本质不同**：对「零信号」任务，B0 **快速优雅放弃**（task 120 仅 1 步就诚实承认无法判断）；即使编造错误答案（task 147/149）也在 5–7 步内干净收场，不做无意义重试。但 B0 **确实会**在「UI 反馈模糊」时短程重复（task 41 连续 17 步、task 129 连续 10 步点同一 toggle），关键区别是**规模减半**（10–17 步 vs B2 的 20–30）**且会自我打断**（task 129 第 11 步出现元推理「the button says Unsubscribe... I will assume... finish」主动跳出）。
- ⭐ **4/5 no-hit 是「表征而非能力」的失败**（task 41/120/147/149）：所需信息在 phantom_text 的文本 substrate 里根本不存在。其中 2/5 命中同一个 `intent_template_id=60`（「数图中 X 数量」）→ **系统性任务族缺陷而非零散噪声**。只有 1/5（task 129）是即使有图也答错的语义粒度错误。⚠️ 但注意 §387.10 的受控对比显示，给这类任务补上截图的实测增益 ≈0 —— 所以「表征失败」不等于「换 mode 就能救」。
- 🐛 **task 19 的 P39 命中是假警报** → 直接催生 **B-1890 的规则层修复**：P39 判据 `effective_mutating_action_count` 恒为 0，而逐步核查显示 step 2 有一次真实生效的点赞，且 eval 用 isolated context 直接查服务端状态。**v8 已把 P35/P39 改为从 step record 派生突变计数**，本 condition 的 P39 命中在 v8 下已消失。

### 为什么这个 cell 是 13.66%

B0 遇到结构性缺图任务时「快速合理化猜测后主动止损」，而非「卡死重复直到预算耗尽」。

## 4. 🔁 Self-evolving — 提议规则

- → 已落码：**P39/P35 的 B-1890 修复**、**P43**（中性标签版）

> 这些提议**尚未落码**。按 discover-then-freeze 纪律，reddit 六 mode × 三 model 的 discover 产物应合并成一批（R1–R8 + H2）后统一 bump `RULESET_VERSION` 到 `8-reddit-*` 并全量重扫，而不是逐条落码逐次重扫。

## 5. Actionable

- 本 cell 的 success 不含 task 160（B-1889 不影响本 cell 的 SR）。
- 未发现需要开 B-number 的 scaffold-bug（本轮范围内）。

---

**Cross-link**: 笔记 §387.6 / §387.7 · master_bug_catalog B-1889 (task 160 passive-FP) / B-1890 (footprint 字段恒 0，勿用作判据) · `/tmp/diag_red/` Tier-1 原始扫描产物
