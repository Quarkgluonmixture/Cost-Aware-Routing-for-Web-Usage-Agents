# /diag digest — B1 × `som` × reddit

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
| **Run** | `B1_som_reddit_20260706` |
| **Condition** | `phase1_som_router_0` |
| **Site / Mode / Model** | reddit / `som` / B1 = Qwen3-VL-4B (local) |
| **Episodes** | 205 |
| **SR** | **8.29%** (17 success / 188 failed) |
| **ruleset_version** | `8-reddit-p41p46-b1890fix` |
| **Tier-1 三子集** | failed+hit 167 · **failed-NO-hit 21** · success+hit 6 |

## 2. Tier-1 规则分布（failed 侧）

| 规则 | 含义 | step-level hits | 命中 episode 数 |
|---|---|---|---|
| `P36` | WALK_FAIL_DEGENERATE | 850 | 103 |
| `P31` | budget 耗尽未完成 | 126 | 126 |
| `P5` | 感知缺失循环 | 113 | 77 |
| `P4` | 根节点误操作 | 106 | 22 |
| `P45` | IDENTICAL_FAILED_ACTION_STREAK | 85 | 58 |
| `P14` | URL 自环 | 60 | 49 |
| `P25` | 跨站任务跳过其中一站 | 38 | 38 |
| `P33` | 导航至裸图片 URL 幻觉 | 33 | 33 |
| `P12` | 从不翻页 | 26 | 26 |
| `P44` | HALLUCINATED_ELEMENT_REF | 16 | 5 |
| `P46` | COMMENT_INTENT_NO_TYPE | 6 | 6 |
| `P27` | 找不到即放弃 | 5 | 5 |
| `P10` | 跨步数值记忆失败 | 2 | 2 |

**success 侧 fire 的规则（presence-only 误报审计对象）**: `P33`×4, `P25`×1, `P42`×1, `P27`×1, `P41`×1

**failed-NO-hit episode（deterministic 盲区）**: [14, 22, 38, 60, 78, 87, 92, 93, 94, 95, 96, 97, 103, 125, 134, 156, 169, 175, 191, 192, 201]

**success episode**: [0, 18, 36, 40, 42, 58, 77, 120, 129, 130, 131, 139, 160, 171, 188, 189, 200]


## 3. Tier-2 深挖

**覆盖范围**：7 ep（no-hit 5 + success 审计 2）· 1 sonnet sub-agent

**三分类**：agent-limit 5 · **benchmark-FP 2（两个 success 全部可疑）** · scaffold-bug 0 · unclear 0

### 具体发现

- ✅ **标注图确实送达模型**：每步 `image_meta` 的 `input_image` token 数非零（576 或 1344），`som.enabled=true`，`mark_count` 在 2–136 间正常变化 —— 机制层面没坏。
- **但抽样的 5 个 no-hit 没有一个是「看错/点错标注框」** —— 全部败在上游的视觉推理 / 计数 / OCR / 指令理解。
- **task 132** — 用真实 reddit 的 `/r/<sub>` 路径规范（本站应为 `/f/<sub>`）反复 goto，造成 4 次同构 404 循环。这是**预训练先验污染站内导航**的清晰例子。
- **task 175 / 203** — 「过早放弃」型：1-2 步内 confidence=0.0 直接 finish。task 203 的放弃措辞只写在 `thought` 里、`answer` 为空字符串，因此 P27 完全看不到。
- **task 58** 触发 P25 且判成功 → 本 digest 首次提出该跨站捷径疑点，后经跨 18-cell 复核确立为 **B-1892**。

### 为什么这个 cell 是 8.29%

标注图对 B1 的净增益边际：扣除 task 160 后 som 7.80% vs dom 6.34%（17 vs 14 个成功，n=205 下大概率是噪声量级）。真正撑住 B1 表现的是 **DOM/AXTree 文本本身** —— B1_vision（无文本纯截图）只有 2.93%，远低于所有含文本的 mode。

## 4. 🔁 Self-evolving — 提议规则

- P27 `ABANDONMENT_RE` 加 'unable to determine' + 扫 `thought`（本 condition 5 个 no-hit 中 2 个因此漏检）
- `EMPTY_ANSWER_SURRENDER`（finish 且 answer=='' 且 confidence==0.0）
- `REAL_REDDIT_PATH_HALLUCINATION`（goto url 匹配 `/r/<name>` ≥2 次）

> 这些提议**尚未落码**。按 discover-then-freeze 纪律，reddit 六 mode × 三 model 的 discover 产物应合并成一批（R1–R8 + H2）后统一 bump `RULESET_VERSION` 到 `8-reddit-*` 并全量重扫，而不是逐条落码逐次重扫。

## 5. Actionable

- ⚠️ **本 cell 的 success 含 task 160（B-1889 benchmark-FP）**。若排除，SR 8.29% → 7.80%。排除与否属 prereg 级改动，**待 user / advisor 决策**，本 digest 不自行调整数字。
- 未发现需要开 B-number 的 scaffold-bug（本轮范围内）。

---

**Cross-link**: 笔记 §387.6 / §387.7 · master_bug_catalog B-1889 (task 160 passive-FP) / B-1890 (footprint 字段恒 0，勿用作判据) · `/tmp/diag_red/` Tier-1 原始扫描产物
