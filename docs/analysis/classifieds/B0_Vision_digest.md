# B0 Vision Baseline 分析报告（Classifieds）

> B0 = Qwen3-VL-235B-A22B（proxy API），Vision 模式，classifieds 站点
> 对应 B1 分析见 `B1_Vision_digest.md`；B0 vs B1 跨模型对比见 `B0_B1_findings.md`
>
> 数据来源：`phase1_vision_router_0`，classifieds 全部 234 tasks
> 分析方法：自动化 post analysis（condition_summary_v2 + reason_diagnostics + cross_representation）

---

## 一、总体概况

| 指标 | 数值 |
|------|------|
| Vision condition 总 episode 数 | 234 |
| 成功（raw） | 33（14.10%） |
| N/A FP | 9（10 个 N/A reference task 中 9 个误判） |
| Visual FP | 0（Vision 模式本身有截图，无 visual lucky hits） |
| **Adjusted SR** | **10.71%**（24/224） |
| 平均步数 | 8.02 步 |
| 平均成本 | $0.0256 / episode |
| 平均无效步率（no-op） | 27.4% |
| 平均页面无变化率 | 41.8% |
| 早停触发分布 | action_failed: 412, page_unchanged_streak: 312, no_progress_streak: 214 |

### 与 DOM / SoM 对比（B0 三模式）

| 指标 | DOM | SoM | **Vision** |
|------|-----|-----|-----------|
| Raw SR | 15.02% | 15.81% | **14.10%** |
| Adjusted SR | 8.07% | 12.05% | **10.71%** |
| 平均步数 | 14.10 | 8.27 | **8.02** |
| 平均成本 | $0.0457 | $0.0411 | **$0.0256** |
| 主导失败原因 | incomplete(23.8%) | parse_error(20.1%) | **no_progress(32.1%)** |

B0 Vision 成本最低（$0.0256/ep），adjusted SR 居中（10.71%，介于 SoM 12.05% 与 DOM 8.07% 之间）。

### 与 B1 Vision 的关键对比

| 模型 | Raw SR | Adjusted SR | avg Steps | avg Cost |
|------|--------|-------------|-----------|----------|
| **B0 (235B)** | 14.10% | **10.71%** | 8.02 | $0.0256 |
| B1 (4B) | 12.39% | 8.12% | 8.0 | $0.029 |

**B0 Vision 优于 B1 Vision（+2.59pp adjusted）**——235B 模型在纯视觉模式下能力略有提升，且成本更低。

---

## 二、失败原因分布

| 失败原因 | 数量 | 比例 | 备注 |
|---------|------|------|------|
| **fail_no_progress** | **72** | **32.1%** | ★ 最大失败源，Vision 特有高发 |
| fail_incomplete_or_stuck | 50 | 22.3% | 页面卡住/信息不完整 |
| success | 33 | 14.7% | (raw) |
| fail_finish_wrong_url_not_found | 19 | 8.5% | 完成时 URL 不匹配 |
| fail_finish_eval_mismatch | 13 | 5.8% | 评测不一致 |
| fail_parse_error | 12 | 5.4% | JSON 解析错误（低于 SoM 的 20.1%） |
| fail_early_finish | 10 | 4.5% | 过早结束 |
| fail_max_steps_target_unreachable | 7 | 3.1% | 目标不可达 |
| fail_finish_claim_missing | 5 | 2.2% | finish 时声明缺失 |
| fail_max_steps_search_repeat | 2 | 0.9% | 搜索循环 |
| fail_finish_empty_answer | 1 | 0.4% | 空答案 |

---

## 三、核心异常：fail_no_progress 率 32.1%

B0 Vision 的 fail_no_progress 率（32.1%，72/234）是三模式中最高的：DOM 11.2%、SoM 9.8%。这是 Vision 模式特有的高发失败。

### 3.1 机制分析

`fail_no_progress` 触发条件：连续若干步动作执行了但页面没有向目标进展（no_progress_streak 阈值）。

Vision 模式高发的可能原因：

**原因 A：坐标 misclick 积累**
Vision 模式纯靠归一化坐标 `[x,y]` 点击，235B 模型同样存在坐标精度问题：
- 连续 misclick（页面有反应但点错了位置）不会触发 page_unchanged，但会累积 no_progress 计数
- B1 Vision 中已记录：misclick 后不自纠正，连续相同坐标重复 3-4 步（task 48 验证）
- B0 Vision 中预计存在同类行为，但 235B 的语义理解更强，可能偶尔自纠正

**原因 B：action_failed 极高（412 次）**
Vision 模式的 action_failed（412 次）是三模式最高（DOM 185 次，SoM 112 次），说明大量操作因坐标无效、元素不存在等原因失败。高 action_failed 率直接贡献 no_progress 计数。

**原因 C：scroll 到底后持续 scroll**
没有 AXTree 文本指引，agent 到达页面底部后可能继续 scroll down（page_unchanged 计数上升），或触发 no_progress。与 B1 Vision 相同的模式。

**原因 D：page_unchanged_rate 最高（41.8%）**
B0 Vision 的平均 page_unchanged_rate（41.8%）是三模式最高（DOM 26.8%，SoM 31.6%），反映了 Vision 模式的大量无效动作。

### 3.2 与 B1 Vision 对比

B1 Vision 的 page_unchanged_streak 触发 164 次（vs B0 的 312 次），但 B1 总 episodes=234（相同）。B0 的 page_unchanged_streak 更高，可能因为 235B 模型更"执着"地尝试相同操作（temperature=0.1 vs B1 greedy，轻度随机反而使模型重复同一错误动作）。

---

## 四、成本效率分析

### 4.1 Vision 成本优势

| 指标 | B0 Vision | B0 SoM | B0 DOM |
|------|-----------|--------|--------|
| 平均成本/ep | **$0.0256** | $0.0411 | $0.0457 |
| 每成功 ep 成本（adjusted） | $0.0256/0.1071 = **$0.239** | $0.0411/0.1205 = **$0.341** | $0.0457/0.0807 = **$0.566** |
| 成本 vs SoM | **62%** | — | 111% |

**Vision 成本效率最优**（$0.239/成功 ep vs SoM $0.341/成功 ep）。Vision 不发送 AXTree 文字，也不需要 SoM 标注图像，token 量最少。

### 4.2 Vision 步数优势

Vision 平均步数（8.02）略低于 SoM（8.27）——视觉信息使 agent 决策更直接，不需要在长 AXTree 文本中搜索。但高 no_progress 率表明大量步数被无效动作浪费。

---

## 五、坐标行为分析（B0 vs B1 Vision 对比）

### 5.1 坐标精度

B1 Vision 已记录多个系统性坐标偏移案例（task 100/101/102 三 seed 同一偏移，task 9/11 连续 misclick）。B0 235B 模型的坐标精度预计更高，但仍受以下限制：
- VWA viewport 尺寸（1280×720）的坐标空间精度要求
- API 代理调用的延迟可能影响页面状态采集时机

### 5.2 坐标格式稳定性

B1 Vision 中记录了混合坐标格式（归一化/像素混用）问题。B0 使用 temperature=0.1，轻度随机采样可能偶发格式不一致。`vwa_wrapper.py` 的防御性归一化（>1.0 自动除以 viewport）可处理大部分情况。

### 5.3 scroll dy 约定（B0 特有）

B0 DOM 中记录了 scroll dy 符号不稳定（见 B0_DOM_digest），Vision 模式中同样可能存在此问题。Vision 模式的高 no_progress 率（32.1%）部分可能来自 scroll 方向错误（往错误方向 scroll 但页面反应正常，不影响 page_changed 但不进展）。

---

## 六、B0 Vision 的独特优势（vs B1）

### 6.1 纯视觉任务表现改善

B0 Vision adjusted SR（10.71%）> B1 Vision（8.12%）。235B 模型的视觉理解能力更强：
- 图片内容识别（颜色、形状、人物、场景）更准确
- 更少的幻觉（对不存在的视觉内容编造描述）

### 6.2 Oracle 中 Vision 贡献比例更高

跨模式分析中，B0 oracle 选择 Vision 的次数（20/46）高于 B1（18/46）：

| 指标 | B0 | B1 |
|------|----|----|
| Oracle 选 Vision 次数 | **20** | 18 |
| Oracle 选 SoM 次数 | 14 | 26 |
| Oracle 选 DOM 次数 | **12** | 2 |

B0 Vision 在 oracle 中贡献更多，特别是 single_navigation+url_match 类任务（B0 Vision SR=12.2% 在此类型上最高）。

---

## 七、共性脚手架缺陷（与 B1 Vision 相同）

- **无结构化导航信息**：搜索框、分类链接、分页控件只能靠视觉识别，4-8B 和 235B 模型均受限
- **N/A 任务 False Positive（9/10）**：1 个 task（189）B0 Vision 未判为 success，为例外；其余 9 个全部误判
- **极少翻页**：Vision 模式分页控件需视觉识别，即使 235B 也很少翻页
- **confirm 弹窗不可交互**：Delete 操作在 Vision 模式下同样受 VWA 框架限制

### 7.1 string_match 格式假负例（Format False Negative）

3 个 Vision episode 语义正确但被 `string_match` 评测器拒绝，导致 Vision SR 被低估约 1.3pp：

| task_id | agent 回答 | must_include | 差异 |
|---------|-----------|-------------|------|
| 42 | `$5.00 - $120.00` | `["5", "120"]` | tokenizer 把 `5.00`/`120.00` 与 `5`/`120` 判为不匹配 |
| 209 | `208.00` | `["$208"]` | 正确值 208，缺少 `$` 符号 |
| 222 | "...is correct based on the measuring tape..." | `["yes"]` | 用 "correct" 替代 "yes"，语义等价 |

**对跨模式对比的影响**：这 3 个均为视觉任务，DOM 模式因无法看到图片而答错（真负例）。Format FN 仅单向压低 Vision SR，不影响 DOM/SoM。校正后 B0 Vision adjusted SR 约为 12.0%（+1.3pp），但为保持 pipeline 一致性，不在 adjusted_success 中修正，仅作为已知评测偏差记录。

---

## 八、与路由的关联

B0 Vision 在 oracle 中贡献 20 个任务（adjusted），是三模式中贡献第二多的。相比 B1 Vision（18 个），B0 Vision 的路由价值略高，特别是在 single_navigation+url_match 类任务上（B0 Vision SR=12.2%，是三模式最高）。

B0 Vision 的低成本（$0.0256/ep vs SoM $0.0411/ep）使其成为路由策略的自然"低成本替代"选项。结合 SoM 的 parse_error 问题，B0 的最优路由方向可能是 **Vision ↔ SoM**（与 B1 相同），但 B0 三模式调整后 SR 更接近，DOM 也有更多路由价值（12 个 oracle 选择）。

---

*生成时间：2026-04-15*
*数据来源：B0_3mode_classifieds_20260413 phase1_vision_router_0*
*B0 Vision 三模式定量对比见 `B0_findings.md`；B0 vs B1 跨模型对比见 `B0_B1_findings.md`*
