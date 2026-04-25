# B1 Vision Baseline 分析报告（Classifieds，完整 234 tasks）

> **数据更新说明**：定量指标已更新至 `B1_3mode_classifieds_20260413` 运行。定性案例分析基于原始运行的逐 episode 审查，行为模式在结构上仍然有效。
> **注：visual_fp 层已在 §95 中废弃，adjusted SR 仅扣除 N/A FP + eval FP**

> 数据来源：`B1_3mode_classifieds_20260413`，classifieds 全部 234 tasks
> 分析方法：自动化 post analysis + 逐 episode 轨迹人工审阅
> 本报告**仅分析 Vision 模式**（纯截图，无 AXTree / SoM 标注）。
> 三模式共性缺陷与定量对比见 `B1_findings.md`。

---

## 总体概况

| 指标 | 数值 |
|------|------|
| 已完成 episode | 234 / 234 |
| Raw SR | 11.11%（26/234）|
| N/A FP | 10 |
| Eval FP | 0 |
| **Adjusted SR** | **7.14%**（16/224） |
| 平均步数 | 6.73 步 |
| 平均成本 | $0.0133 / episode |
| p95 步延迟 | 64.5s |
| 早停触发分布 | action_failed: 430, page_unchanged_streak: 196, no_progress_streak: 196 |

> §95 变更：Vision 无 visual_fp（Vision 本身有截图），§95 对 Vision adjusted SR 无影响，仍为 7.14%（16/224）。

### 与 DOM / SoM 全量对比

| 指标 | DOM | SoM | Vision |
|------|-----|-----|--------|
| Raw SR | 11.11% (26) | 17.52% (41) | **11.11% (26)** |
| Adjusted SR | 7.59% (17) | 13.84% (31) | **7.14% (16)** |
| 平均步数 | 13.83 | 9.90 | **6.73** |
| 平均成本 | $0.0399 | $0.0347 | **$0.0133** |
| p95 步延迟 | 43.2s | **30.2s** | 64.5s |

**Vision vs SoM**：McNemar p=0.006（**显著**），Vision SR 显著低于 SoM。
**Vision vs DOM**：McNemar p=0.728（**不显著**），Vision 与 DOM 无显著差异。

> §95 变更：DOM adjusted SR 从 4.91% 上升至 7.59%（visual_fp 废弃），DOM 与 Vision（7.14%）几乎持平。Vision vs DOM 仍不显著。

---

## 成功案例分析

### Vision-Only 成功（6 tasks，adjusted，cross_rep）

| Task | 任务类型 | 评测方式 | 描述 |
|------|---------|---------|------|
| 14 | grid_position | string_match | 找第二行绘画的卖家邮箱（需 grid 空间布局理解） |
| 40 | single_navigation | url_match | 找不锈钢洗碗机（需视觉确认材质） |
| 97 | single_navigation | url_match | 找动物形状的物品（需视觉形状识别） |
| 124 | page_reading | string_match | 找图片背景是草地的 item |
| 152 | page_reading | string_match | 找图片中包含人手的 item |
| 187 | single_navigation | url_match | 找图片中有 Lightning McQueen 的 item |

**共性**：全部需要从截图中识别视觉属性，DOM 纯文本无法获取这些信息。

### Task 14 — Grid 空间布局理解（三模式对比关键 case）

Vision 是唯一成功的模式。DOM 和 SoM 均因 AXTree 线性序列化导致行列混淆失败。纯视觉模式下没有 AXTree 文本干扰，模型直接从截图的视觉布局中感知 grid 结构。这是 Mirage Effect 的正面案例。

---

## 失败模式详解

### F1. 坐标定位失败（Coordinate Misclick）

Vision 模式 click 成功率约 49%（SoM element-ID 约 72%，差 23pp）。失败 click 按根因分为五类：

| 故障类型 | 占比 | 根因 |
|---------|------|------|
| viewport 内偏移 | ~43% | 模型空间定位精度不足 |
| 混合格式 | ~20% | x 归一化 + y 像素混用 |
| 越界坐标 | ~20% | 模型不感知 viewport 边界 |
| 顶部误点 | ~10% | y 坐标估计严重偏移 |
| 零纠错重试 | ~50%（叠加） | 4B 模型 click 失败后重试完全相同坐标 |

### F2. Scroll-Down 到底早停（不会翻页）

Agent 到达页面底部后连续 scroll down 无变化触发早停。Vision 模式更严重：没有 AXTree 暴露分页链接的文本/ID。

### F3. 过早结束（Premature Finish，Vision 特有高发）

| 失败原因 | 数量 | 比例 |
|---------|------|------|
| fail_no_progress | **136** | **58.1%** |
| fail_early_finish | 33 | 14.1% |
| fail_incomplete_or_stuck | 16 | 6.8% |
| fail_finish_wrong_url_not_found | 10 | 4.3% |
| fail_finish_eval_mismatch | 5 | 2.1% |

Vision 模式 fail_no_progress 支配（58.1%），因坐标 misclick 频繁导致无效动作累积。

### F4. 不可交互元素误点（UI 理解失败）

Vision 模式下 agent 无法区分可交互与不可交互元素——DOM 模式的 AXTree 会标注元素类型，但纯截图没有这些信息。

### F5. 信息充分幻觉——列表页直接 finish 不进详情页（Vision 特有）

Vision 截图给 agent "我已经看到了所有信息"的错觉。DOM/SoM 的 AXTree link 结构隐式传达了"点进去还有更多内容"的信号。

### F6. 幻觉与推理错误

Task 5 最严重：agent 点击坐标反复进入错误页面，但 thought 每次都说正确目标——视觉内容与语言推理完全脱节。

---

## Vision 模式特有的结构性劣势

| 问题 | DOM/SoM 有而 Vision 无 | 影响 |
|------|----------------------|------|
| 元素 ID 点击 | AXTree 提供 `[id]` 精确点击 | Vision 只能用坐标，misclick 率高 |
| 元素类型标注 | `link` / `button` / `textbox` | Vision 无法区分可交互 vs 只读 |
| 结构化导航 | 搜索框、分类链接、分页控件有文本标注 | Vision 需视觉识别 UI 控件 |
| 文本信息 | AXTree 提供精确数值/标签 | Vision 依赖 OCR |

---

## 与 DOM / SoM 的对比定位（全量）

| 维度 | DOM | SoM | Vision |
|------|-----|-----|--------|
| Raw SR | 11.11% | **17.52%** | 11.11% |
| Adjusted SR | 7.59% | **13.84%** | 7.14% |
| 平均步数 | 13.83 | 9.90 | **6.73** |
| 平均成本 | $0.0399 | $0.0347 | **$0.0133** |

**成本效率**：Vision 成本仅为 SoM 的 38%、DOM 的 33%。每成功 episode 的成本（adjusted）：Vision $0.0133/0.0714 ~ $0.19，SoM $0.0347/0.1384 ~ $0.25。Vision 在成本效率上有优势。

---

## Venn 图集合分析（Adjusted, /234 分母）

| 区域 | 数量 | 占比 |
|------|------|------|
| 三模式均失败 | 184 | 78.6% |
| 仅 SoM | 15 | 6.4% |
| **仅 DOM** | **13** | **5.6%** |
| SoM + Vision（非 DOM） | 9 | 3.9% |
| **仅 Vision** | **6** | **2.6%** |
| DOM + SoM（非 Vision） | 5 | 2.1% |
| 三模式均成功 | 2 | 0.9% |

> §95 变更：DOM 独占成功从 7 个升至 13 个（visual_fp 废弃后更多 DOM 成功保留）。DOM 路由价值进一步提升。

### Oracle Routing（Adjusted, /234 分母）

| 指标 | 数值 |
|------|------|
| Oracle ceiling (adjusted) | **21.37%** (50 tasks) |
| Best single (SoM adjusted) | 13.25% |
| Routing headroom | **8.12pp** |
| Oracle 选择分布 (adj) | DOM:16, Vision:17, SoM:17 |

Oracle 中三模式贡献均衡（DOM 32%, Vision 34%, SoM 34%）。

---

## Task 58 — 翻页 + City 过滤：Vision 独有的"聪明"行为

Task 58（Washington, D.C. Furniture 中找蓝色椅子）展示了 Vision 模式最高策略水平：翻页（step 7）+ City 过滤（step 8）。DOM/SoM thought 中多次提到 City 过滤但始终未能执行——典型"认知-执行鸿沟"，Vision 截图弥合了这一鸿沟。

---

## 路由信号质量（Vision 模式）

### 信号区分力（AUROC，adjusted labels）

| 信号类型 | 最佳指标 | AUROC | 95% CI |
|---------|---------|-------|--------|
| 行为信号 | url_revisit_max | **0.816** | [0.753, 0.872] |
| 行为信号 | action_diversity | **0.809** | [0.706, 0.891] |
| 行为信号 | url_revisit_count | 0.799 | [0.728, 0.860] |
| Verbalized | ep_mean_verbalized | **0.757** | [0.623, 0.879] |
| 行为信号 | max_repeat_streak | 0.744 | [0.648, 0.836] |
| Token-level | ep_max_entropy | 0.541 | [0.419, 0.656] |

行为信号（url_revisit_max AUROC=0.816）和 verbalized confidence（AUROC=0.757）具有区分力。Token-level 全部无用。

---

## 关键发现

1. **Vision 与 DOM 持平（§95 后更明显）**：Adjusted SR 7.14% vs DOM 7.59%，McNemar p=0.728（不显著）
2. **成本效率最优**：Vision 成本仅为其他模式的 33-38%
3. **6 个 Vision-only 成功全是纯视觉任务**：图片内容识别是 Vision 独有且不可替代的能力
4. **DOM 独占成功大幅增加**：13 个（§95 后），DOM 路由价值提升
5. **坐标精度是最大瓶颈**：misclick 后不自纠正
6. **认知-执行鸿沟弥合**（Task 58）：Vision 截图触发 DOM/SoM 无法执行的操作
7. **路由信号可用**：行为信号 AUROC=0.816，verbalized AUROC=0.757

---

*生成时间：2026-04-12*
*更新时间：2026-04-25（§95 FP 重构：废弃 visual_fp；DOM adjusted SR 上升至 7.59%，DOM 独占成功增至 13 个；更新全部定量数据和集合分析）*
*数据来源：B1_3mode_classifieds_20260413 完整三模式运行*
