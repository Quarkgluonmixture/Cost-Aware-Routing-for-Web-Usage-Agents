# R9755 失败错因 digest — B0 dom classifieds

> **生成方式**: `/diagnose` skill 3-tier pipeline (2026-05-22 首次完整运行)。Tier-1 deterministic 全扫 (`diag_pattern_match.py`, 0 token) → Tier-2 Claude sub-agent 深挖 35 个 no-hit 盲区 → Tier-3 整合 (本文件)。
> **Run**: `B0_dom_classifieds_20260521_125142_282975264_567142_R9755` (Fire-6 first completed condition, manifest-bound authoritative)
> **Condition**: `phase1_dom_router_0` | site classifieds | mode **dom** | model **B0 = Qwen3-VL-235B (proxy)**

---

## Verdict

| 维度 | 结论 |
|---|---|
| Episodes | 224 (33 success / **191 failed**) — SR **14.7%** |
| 三分类 | **agent-limit 100% · scaffold-bug 0 · benchmark-FP 0** |
| Deterministic coverage | **93%** (178/191 failed 被 P-rule 命中) |
| 纯视觉真盲区 | 13 (应 route 到 som/vision，非补 DOM 规则) |

**R9755 的失败全部是真实模型能力局限，零框架 bug，零评测误判。** 这独立复现了 `fire_manifest.json` 声明的 `parse_error_rate=0 / benchmark_noise_rate=0`——Tier-2 对最难的 35 个 no-hit 盲区逐 episode 深挖，parse/tool_call 全 valid、finish-vs-reference 全真错，无一例外。

> ⚠️ **presence ≠ causation**: 178 deterministic-covered = "命中 agent-limit 类 P-rule"（特征存在），不等于逐个证因。**0 scaffold / 0 FP 的强结论来自 35 no-hit 子集的逐 episode 深挖**（+ scaffold 规则 P8 在全 run 零命中）。33 个 success episode 也全部命中 P-rule（P6 27 / P14 15），证明这些规则是风险标记而非死因判定。

---

## 失败模式 taxonomy

### Tier-1 规则分布 (failed-only, 17 规则)

```
P14 URL自环              109   ████████████████████  (检测到，但与 P6 大量重叠)
P6  视觉任务DOM必败       97   █████████████████
P16 视觉图像内容          47   █████████  ← self-evolving 新增
P17 click-back振荡        37   ███████    ← self-evolving 新增
P5  感知缺失循环          36   ███████
P10 跨步数值记忆失败      21   ████
P7  sCity=州名            19   ███
P18 cheapest漏排序        15   ███        ← self-evolving 新增
P2  容器节点误点          14   ███
P15 gallery行位置          6   █          ← self-evolving 新增
P4 根节点 / P12 不翻页 / P13 搜索代浏览  少量
```

### 四大死因类别 (cls + dom + B0)

**1. 视觉表征天花板 (压倒性主导)** — DOM 看不到任何像素，cls 大量任务需要看图：
- **图像内容任务** (P16, 47 failed): "in the image" / "on the cover" — intent 引用一张给定图片或要求读缩略图内容。例 task 81 "cheapest book with hurricane **on the cover**" → agent 文本搜 hurricane 选错书；task 84 "ring with selfie image" / task 221 "number of bowls" (数量仅在图中)。
- **gallery 空间行位置** (P15, 6 failed): "second/last row" — DOM 把视觉网格线性化，无法知道哪些 item 物理上在第 N 行。例 task 14/41/42。
- **颜色感知** (P6, 97 failed): 含颜色词/颜色形容词 ("dark color" task 21) 或 task 带 image 字段。
- **cross-site 视觉** (真盲区): task 207/230 需从 OneStopMarket 图读颜色/时钟时间。

**2. 导航行为缺陷 (cross-mode, router-relevant)**:
- **click-back 振荡** (P17, 37 failed): 同一 item 反复进入→退出 (例 task 40 item ×4 / task 111 item ×4)，detail↔list 横跳。agent 无记忆、无"已到正确答案"判断，振荡耗尽 step budget。**这类失败换表征 (som/vision) 救不了，需 retry/memory 模块** → paper router 设计直接证据。
- **感知缺失循环 / URL 自环** (P5/P14): 重复同一无效 action / 卡同一 URL。

**3. 搜索逻辑缺陷**:
- **cheapest 漏价格排序** (P18, 15 failed): intent 要 "cheapest" 但从不按 `i_price` 排序，在乱序结果取第一项 (例 task 216 取 $421 而非真最低)。
- **关键词歧义** (真盲区): 搜索词匹配错误语义类 (task 210 "lamb" → "Lambs Ear plants" 植物，agent 发现歧义但仍选错)。无通用 0-token signal (lamb-specific 规则会高 FP)，故未补规则。

**4. 跨步推理 / 记忆失败** (P10, 21 failed): thought 提到数字 X，action/answer 用了不同数字。

---

## 13 个纯视觉真盲区 → routing 论点

self-evolving 后仍 no-hit 的 13 个: `[84, 97, 106, 119, 124, 129, 131, 162, 207, 208, 210, 221, 230]`。

除 task 210 (关键词歧义) 外，全部是**图像内容决策任务**——agent 在 DOM 模式下原则上无法获得任何像素信息，任何需要"看图"区分 listing 的任务结构性不可解。sub-agent 一致判定 `deterministic_candidate=false`：**不该为 DOM 写更多兜底规则，应路由到 som/vision**。

这是 phantom routing 的核心实证：**DOM 表征有结构性天花板，只能靠换表征 (routing) 而非补规则突破。** cls 的视觉任务占比远超 P6 原统计 (96)——加上 P15/P16 揭示的 gallery row + 图像内容，真实"DOM 不可解视觉任务"是 cls 失败的最大来源。

---

## Self-evolving changelog (规则库 13 → 17)

Tier-2 深挖发现的可 deterministic 化模式，已落 `diag_pattern_match.py`:

| 规则 | 模式 | 来源 task | failed hit | FP 率 |
|---|---|---|---|---|
| **P6 ext** | 颜色形容词 (dark/light/pale) | 21 | (+1) | — |
| **P15** | gallery 行位置 (sShowAs=gallery + "N row") | 14/41/42 | 6 | 0% |
| **P16** | 图像内容 ("on the cover"/"in its image") | 80/81 | 47 | 11% |
| **P17** | click-back 振荡 (item revisit ≥3 + back ≥2) | 40/111 | 37 | 5% |
| **P18** | cheapest 漏排序 (cheapest intent + 无 i_price) | 216 | 15 | 6% |

效果: deterministic coverage **82% → 93%** (+11pp), no-hit failed **35 → 13** (-63%), 全程 success 误报 +0。新规则 FP 率全 ≤11% (与现有 P6 22%/P14 12% 同量级或更低)。

**未来同类任务 0-token 自动覆盖** —— Tier-2 的一次 quota (4 sub-agent / ~280K token) 换来 4 条新规则 (P15/P16/P17/P18) + P6 扩展。

---

## 代表 episode

| task | 类别 | intent (节选) | finish | reference | 死因 |
|---|---|---|---|---|---|
| 81 | 图像内容 | hurricane **on the cover** | id=21162 | id=4727 | 文本搜 hurricane，没看封面 |
| 14 | gallery 行 | second row painting 的 email | john.dubois394 | olga.jones341 | DOM 无法定位视觉行 |
| 221 | 图像唯一信息 | number of bowls in set | "not specified" | 6 | 数量仅在图中，DOM 无 |
| 40 | click-back 振荡 | most recent stainless steel dishwasher | GE | LG | item ×4 振荡，判不了材质 |
| 216 | cheapest 漏排序 | cheapest oval table $420-430 | $421 | id=66046 (真最低) | 按日期排序取第一项 |
| 210 | 关键词歧义 | cheapest lamb (Farm+garden) | "Lambs Ear plants" | id=81060 | 发现是植物仍选错 |

---

## 后续行动

- **无 scaffold-bug** → 无需出 B-number (R9755 框架层干净)
- **无 benchmark-FP** → 无需 task 排除
- **router 设计输入**: P17 click-back 振荡 (37) + P18 cheapest 漏排序 (15) 是 cross-mode agent 行为缺陷，换表征救不了 → 支持 paper router 的 retry/memory 模块论证
- **跨 condition 复用**: 17 条规则现可 0-token 应用到 Fire-6 其余 35 conditions (B1/B2 × 6 mode × 2 site) —— Tier-1 全扫即可对比各 condition 的失败模式分布
