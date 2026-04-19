# B0 Vision Baseline 分析报告（Classifieds）

> B0 = Qwen3-VL-235B-A22B（proxy API），Vision 模式，classifieds 站点
> 对应 B1 分析见 `B1_Vision_digest.md`；B0 vs B1 跨模型对比见 `B0_B1_findings.md`
>
> 数据来源：`phase1_vision_router_0`，classifieds 全部 234 tasks
> 分析方法：自动化 post analysis（condition_summary_v2 + reason_diagnostics + cross_representation）
> **本版数据含 parse_error 修复后重跑结果（2026-04-18）**

---

## 一、总体概况

| 指标 | 数值 |
|------|------|
| Vision condition 总 episode 数 | 234 |
| 成功（raw） | 36（15.38%） |
| N/A FP | 10（10 个 N/A reference task 中 10 个误判） |
| Visual FP | 0（Vision 模式本身有截图，无 visual lucky hits） |
| **Adjusted SR** | **11.61%**（26/224） |
| 平均步数 | 7.67 步 |
| 平均成本 | $0.0241 / episode |
| 平均无效步率（no-op） | 31.7% |
| 平均页面无变化率 | 39.6% |
| 早停触发分布 | action_failed: 455, page_unchanged_streak: 250, no_progress_streak: 250 |
| Bootstrap 95% CI | [7.26%, 15.38%]（raw SR） |

### 与 DOM / SoM 对比（B0 三模式）

| 指标 | DOM | SoM | **Vision** |
|------|-----|-----|-----------|
| Raw SR | 14.96% | **24.36%** | 15.38% |
| Adjusted SR | 8.04% | **21.43%** | 11.61% |
| 平均步数 | 12.25 | 7.25 | 7.67 |
| 平均成本 | $0.0467 | $0.0355 | **$0.0241** |
| P95 延迟 | **10,537ms** | 74,042ms | 50,540ms |
| 主导失败原因 | no_progress(26.3%) | wrong_url(23.3%) | **no_progress(39.3%)** |

B0 Vision 成本最低（$0.0241/ep），adjusted SR 居中（11.61%，介于 SoM 21.43% 与 DOM 8.04% 之间）。

### 统计显著性

| 对比 | p 值 | 显著性 |
|------|------|--------|
| Vision vs DOM | 0.152 | — |
| SoM vs Vision | 1.1e-4 | ★★★ |

Vision 与 DOM 差异不显著（p=0.152）；SoM 显著优于 Vision（p=1.1e-4）。

### 与 B1 Vision 的关键对比

| 模型 | Raw SR | Adjusted SR | avg Steps | avg Cost |
|------|--------|-------------|-----------|----------|
| **B0 (235B)** | 15.38% | **11.61%** | 7.67 | $0.0241 |
| B1 (4B) | 12.39% | 8.12% | 8.0 | $0.029 |

**B0 Vision 优于 B1 Vision（+3.49pp adjusted）**——235B 模型在纯视觉模式下能力提升明显，且成本更低。

---

## 二、失败原因分布

| 失败原因 | 数量 | 比例 | 备注 |
|---------|------|------|------|
| **fail_no_progress** | **92** | **39.3%** | ★ 最大失败源，Vision 特有高发 |
| success | 36 | 15.4% | (raw) |
| fail_finish_wrong_url_not_found | 23 | 9.8% | 完成时 URL 不匹配 |
| fail_incomplete_or_stuck | 21 | 9.0% | 页面卡住/信息不完整 |
| fail_early_finish | 19 | 8.1% | 过早结束 |
| fail_finish_eval_mismatch | 17 | 7.3% | 评测不一致 |
| fail_finish_empty_answer | 9 | 3.8% | 空答案 |
| fail_finish_claim_missing | 7 | 3.0% | finish 时声明缺失 |
| fail_max_steps_target_unreachable | 4 | 1.7% | 目标不可达 |
| fail_max_steps | 3 | 1.3% | 达到最大步数 |
| fail_parse_error | 2 | 0.9% | JSON 解析错误（三模式最低） |
| fail_max_steps_search_repeat | 1 | 0.4% | 搜索循环 |

---

## 三、核心异常：fail_no_progress 率 39.3%

B0 Vision 的 fail_no_progress 率（39.3%，92/234）是三模式中最高的：DOM 26.3%、SoM 10.3%。这是 Vision 模式特有的高发失败。

### 3.1 机制分析

`fail_no_progress` 触发条件：连续若干步动作执行了但页面没有向目标进展（no_progress_streak 阈值）。

Vision 模式高发的多重原因：

**原因 A：坐标 misclick 积累**
Vision 模式纯靠归一化坐标 `[x,y]` 点击，连续 misclick 不会触发 page_unchanged（页面有反应但点错了位置），但累积 no_progress 计数。no-op rate 高达 31.7%（三模式最高）证实了大量无效操作。

**原因 B：action_failed 极高（455 次）**
Vision 模式的 action_failed（455 次）是三模式最高（DOM 360 次，SoM 113 次），说明大量操作因坐标无效、元素不存在等原因失败。

**原因 C：scroll 到底后持续 scroll**
没有 AXTree 文本指引，agent 到达页面底部后可能继续 scroll down（page_unchanged 计数上升），或触发 no_progress。

**原因 D：page_unchanged_rate 最高（39.6%）**
B0 Vision 的平均 page_unchanged_rate（39.6%）是三模式最高（DOM 25.2%，SoM 26.5%），反映了大量无效动作。

### 3.2 与 B1 Vision 对比

B1 Vision 同样以 no_progress 为主要失败原因。B0 的 page_unchanged_streak 更高（250 vs B1 164），可能因为 235B 模型更"执着"地尝试相同操作——temperature=0.1 的轻度随机反而使模型重复同一错误动作。

---

## 四、成本效率分析

### 4.1 Vision 成本优势

| 指标 | B0 Vision | B0 SoM | B0 DOM |
|------|-----------|--------|--------|
| 平均成本/ep | **$0.0241** | $0.0355 | $0.0467 |
| 每成功 ep 成本（adjusted） | $0.0241/0.1161 = **$0.208** | $0.0355/0.2143 = $0.166 | $0.0467/0.0804 = $0.581 |
| cost_efficiency_ratio (SR/cost) | 0.140 | **0.186** | 0.102 |

**Vision 每 episode 成本最低**（$0.0241），但由于 adjusted SR 低于 SoM，**每成功 episode 成本** SoM 反而更优（$0.166 vs $0.208）。SoM 的 cost_efficiency_ratio 最高。

### 4.2 Vision 步数分析

Vision 平均步数（7.67）略高于 SoM（7.25），远低于 DOM（12.25）。但高 no_progress 率（39.3%）和 no-op 率（31.7%）表明大量步数被无效动作浪费。

---

## 五、坐标行为分析

### 5.1 坐标精度

Vision 模式纯靠截图中的坐标点击，B0 235B 模型的坐标精度理论上优于 B1 4B，但仍受以下限制：
- VWA viewport 尺寸（1280×720）的坐标空间精度要求
- API 代理调用的延迟可能影响页面状态采集时机
- no-op rate 31.7% 暗示约 1/3 的操作无效

### 5.2 scroll dy 约定（B0 特有）

B0 DOM 中记录了 scroll dy 符号不稳定（见 B0_DOM_digest），Vision 模式中同样存在此问题。Vision 模式的高 no_progress 率（39.3%）部分可能来自 scroll 方向错误。

### 5.3 TinyMCE iframe 交互限制（三模式共性）

Classifieds 编辑页面的"Description"字段使用 TinyMCE 富文本编辑器，渲染在 `<iframe>` 内。**三种观测模式均无法正常编辑此字段**：

| 模式 | 交互方式 | 失败机制 |
|------|---------|---------|
| Vision | 坐标 click/type | 坐标落在 iframe 容器上，无法穿透到内部编辑区 |
| DOM | element_id click/type | AXTree 中 iframe 内容不可见，element_id 指向外层容器 |
| SoM | element_id click/type | 同 DOM，SoM 标记不覆盖 iframe 内部元素 |

**典型案例**：Task 4（编辑白色汽车 listing，修改价格+描述）
- Vision 步骤 3-4：点击 description 区域 (0.499, 0.749)，page_changed=False，操作无效
- DOM 步骤 23-25：点击 eid=66771（description 容器），同样无法编辑
- DOM 最终成功是因为修改了 Price `<input>` 字段（非 iframe），**并非穿透了 iframe**

**影响范围**：
- 18 个编辑类任务中，15 个三模式全败（iframe 是共同障碍之一）
- Vision 因 iframe 独立失败的 task 仅 2 个（task 4, 160），占总 234 tasks 的 0.85%
- **对 Vision SR 影响 < 1pp，不构成对 Vision 的不公平**

**结论**：这是 VWA benchmark 的已知限制，三模式同等受限，不需要修复。

---

## 六、共性脚手架缺陷（与 B1 Vision 相同）

- **无结构化导航信息**：搜索框、分类链接、分页控件只能靠视觉识别
- **N/A 任务 False Positive（10/10）**：全部 10 个 N/A reference task 均误判为 success
- **极少翻页**：Vision 模式分页控件需视觉识别，即使 235B 也很少翻页
- **confirm 弹窗不可交互**：Delete 操作在 Vision 模式下同样受 VWA 框架限制

### 6.1 string_match 格式假负例（Format False Negative）

3 个 Vision episode 语义正确但被 `string_match` 评测器拒绝，导致 Vision SR 被低估约 1.3pp：

| task_id | agent 回答 | must_include | 差异 |
|---------|-----------|-------------|------|
| 42 | `$5.00 - $120.00` | `["5", "120"]` | tokenizer 把 `5.00`/`120.00` 与 `5`/`120` 判为不匹配 |
| 209 | `208.00` | `["$208"]` | 正确值 208，缺少 `$` 符号 |
| 222 | "...is correct based on the measuring tape..." | `["yes"]` | 用 "correct" 替代 "yes"，语义等价 |

**对跨模式对比的影响**：这 3 个均为视觉任务，DOM 模式因无法看到图片而答错（真负例）。Format FN 仅单向压低 Vision SR，不影响 DOM/SoM。校正后 B0 Vision adjusted SR 约为 12.9%（+1.3pp），但为保持 pipeline 一致性，不在 adjusted_success 中修正，仅作为已知评测偏差记录。

---

## 七、Vision 在路由中的角色

三模式 cross_representation 分析显示，Vision 在 oracle 路由中贡献显著：

| 指标 | 数值 |
|------|------|
| Oracle 选择（adjusted） | **21/58**（36.2%） |
| page_reading 类型 oracle | **14/31**（45.2%，三模式最高） |
| single_navigation 类型 oracle | 13/42（31.0%） |
| Vision 独占成功 | 3 tasks |
| SoM+Vision 共享成功（not DOM） | 13 tasks |

**Vision 的路由价值**：
1. **page_reading 类型主导**：Vision 在 page_reading 任务中 oracle 选择最多（14/31），纯截图足以完成页面信息提取
2. **低成本替代 SoM**：som+vision 共享的 13 个 task 可用 Vision 替代 SoM，每 episode 节省 $0.0114（32%）
3. **独占贡献**：3 个 task 仅 Vision 成功，不可被其他模式替代
4. **成本最低**：$0.0241/ep，是路由策略中自然的"低成本优先尝试"选项

---

*更新时间：2026-04-18*
*数据来源：B0_3mode_classifieds_20260413 phase1_vision_router_0（parse_error 修复后重跑数据）*
*B0 三模式定量对比见 `B0_findings.md`；B0 vs B1 跨模型对比见 `B0_B1_findings.md`*
