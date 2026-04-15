# B0 SoM Baseline 分析报告（Classifieds）

> B0 = Qwen3-VL-235B-A22B（proxy API），SoM 模式，classifieds 站点
> 对应 B1 分析见 `B1_SOM_digest.md`；B0 vs B1 跨模型对比见 `B0_B1_findings.md`
>
> 数据来源：`phase1_som_router_0`，classifieds 全部 234 tasks
> 分析方法：自动化 post analysis（condition_summary_v2 + reason_diagnostics + cross_representation）

---

## 一、总体概况

| 指标 | 数值 |
|------|------|
| SoM condition 总 episode 数 | 234 |
| 成功（raw） | 37（15.81%） |
| N/A FP | 10（10 个 N/A reference task 全部误判） |
| Visual FP | 0（SoM 有截图，无 visual lucky hits） |
| **Adjusted SR** | **12.05%**（27/224） |
| 平均步数 | 8.27 步 |
| 平均成本 | $0.0411 / episode |
| 平均无效步率（no-op） | 7.6% |
| 平均页面无变化率 | 31.6% |
| 早停触发分布 | action_failed: 112, page_unchanged_streak: 98, no_progress_streak: 33 |

### 与 DOM / Vision 对比（B0 三模式）

| 指标 | DOM | **SoM** | Vision |
|------|-----|---------|--------|
| Raw SR | 15.02% | **15.81%** | 14.10% |
| Adjusted SR | 8.07% | **12.05%** | 10.71% |
| 平均步数 | 14.10 | **8.27** | 8.02 |
| 平均成本 | $0.0457 | $0.0411 | $0.0256 |
| 最高 parse_error 率 | 1.3% | **20.1%** | 5.4% |

B0 三模式中，SoM adjusted SR 最高（12.05%），但 parse_error 率远超其他两种模式（20.1%）。

### 与 B1 SoM 的关键对比

| 模型 | Raw SR | Adjusted SR |
|------|--------|-------------|
| **B0 (235B)** | 15.81% | **12.05%** |
| B1 (4B) | 20.51% | 16.24% |

**B0 SoM 显著低于 B1 SoM（-4.19pp adjusted）**。这是本报告最重要的异常：更大的模型在 SoM 模式下反而表现更差。详细分析见 §五。

---

## 二、失败原因分布

| 失败原因 | 数量 | 比例 | 备注 |
|---------|------|------|------|
| **fail_parse_error** | **45** | **20.1%** | ★ 最大失败源，SoM 特有高发 |
| fail_incomplete_or_stuck | 44 | 19.6% | 页面卡住/信息不完整 |
| success | 37 | 16.5% | (raw) |
| fail_finish_wrong_url_not_found | 29 | 12.9% | 完成时 URL 不匹配 |
| fail_no_progress | 22 | 9.8% | 连续无进展步骤 |
| fail_finish_eval_mismatch | 12 | 5.4% | 评测不一致 |
| fail_max_steps_target_unreachable | 12 | 5.4% | 目标不可达 |
| fail_early_finish | 12 | 5.4% | 过早结束 |
| fail_max_steps_click_back_loop | 5 | 2.2% | Click-back 循环 |
| fail_max_steps_search_repeat | 3 | 1.3% | 搜索循环 |
| 其他 | 3 | 1.3% | claim_missing / max_steps 等 |

---

## 三、核心异常：parse_error 率 20.1%

B0 SoM 的 parse_error 率（20.1%，45/224）是三模式中最高的：DOM 1.3%、Vision 5.4%。这是 SoM 模式特有的失败模式。

### 3.1 可能根因分析

**原因 A：多模态输入复杂度提升**

SoM 模式同时传递文字（`[SOM_MARKS]` 标注列表）和截图（base64 图像），组合输入比 DOM（纯文字）或 Vision（纯图）更复杂。235B 模型在处理这种混合格式时可能生成更长、更复杂的 thought，偶尔导致 JSON 输出格式出错。

**原因 B：temperature=0.1 的随机采样干扰**

B0 使用 temperature=0.1（轻度随机），B1 使用 do_sample=False（贪婪）。在 SoM 模式下，多模态输入触发更复杂的推理路径，加上温度采样，模型有一定概率生成不符合格式的输出（如 JSON 括号不匹配、字符串未闭合等）。DOM 和 Vision 的推理路径相对单一，parse_error 率更低。

**原因 C：proxy API 格式处理**

B0 通过 proxy API 调用 235B 模型，SoM 模式的 image+text 混合请求在 proxy 层的格式构建（`proxy_api_agent.py`）可能存在边缘情况。§46/§47 已修复 reference_images 和 obs_section 格式问题，但可能仍有遗留。

### 3.2 影响评估

45 个 parse_error episode 直接贡献了 20.1% 的失败率。若 parse_error 全部修复（假设均能正常执行），B0 SoM adjusted SR 可能达到 **~15-16%**——接近甚至超越 B1 SoM（16.24%）。这意味着 B0 SoM 的性能劣势可能部分来自基础设施问题而非模型能力不足。

### 3.3 与 B1 SoM 的对比

B1 SoM parse_error 极少（B1_findings.md 未单独列出；早停触发中 action_failed=49/234，其中 parse_error 为子集，估计 <5%）。B1 使用 do_sample=False 贪婪解码，JSON 输出格式极度稳定。B0 的温度采样在 SoM 多模态复杂输入下增加了格式失败风险。

---

## 四、其他失败模式

### 4.1 fail_incomplete_or_stuck（44 例，19.6%）

代表性表现：
- Agent 进入目标类别但无法找到满足条件的 listing（如颜色/视觉属性条件）
- SoM 截图可见但 agent 无法精确点击目标元素（SoM marks 坐标命中率受 API agent 限制）
- 反复 scroll 但未能触发有效导航

### 4.2 fail_finish_wrong_url_not_found（29 例，12.9%）

- Agent 自认为完成但 `finish` 时的 URL 不满足 url_match 评测条件
- 常见于需要精确进入某个 listing 详情页的任务
- 与 B1 SoM（15.1%）相比略低

### 4.3 fail_early_finish（12 例，5.4%）

SoM 模式下过早结束：
- Agent 看到截图中有相关内容即 finish，未验证答案完整性
- "信息充分幻觉"：SoM 截图+标注给 agent "已看到所有信息"的感觉

### 4.4 action_failed 早停（112 次）

action_failed 是 SoM 模式最常见的早停触发（112 次，vs DOM 的 185 次）。SoM 使用 element_id（来自 SoM marks）点击，id 不存在或过期时触发 action_failed。较 DOM 低是因为 SoM 标注的 element_id 更新鲜（每步重新生成标注）。

---

## 五、B0 vs B1 SoM 差异根因分析

### 5.1 结果对比（adjusted）

| 模型 | Adjusted SR | Parse Error | avg Steps | avg Cost |
|------|------------|-------------|-----------|----------|
| B0 (235B, API) | 12.05% | 20.1% | 8.27 | $0.041 |
| B1 (4B, local) | 16.24% | ~<5% | 11.8 | $0.077 |

### 5.2 "SoM 反转"假说

B0 在 DOM（8.07% vs 0.85%，9.5×）和 Vision（10.71% vs 8.12%）均优于或等于 B1，但在 SoM 模式下显著落后（-4.19pp）。可能解释：

**假说 A：parse_error drag（最可能）**
45 个 parse_error episode = 15.6% 的额外失败率。若扣除这些基础设施失败，B0 SoM 真实能力 SR ≈ 12.05% + 若干 = 可能 ~14-16%，接近 B1。

**假说 B：text-over-vision 随模型规模增强**
235B 模型的文字推理能力更强，在 SoM 图文混合输入下"文字盖过图像"（text-over-vision，Mirage Effect）的倾向可能更明显——模型更倾向于用 SoM marks 文字列表推理，忽略截图的视觉信息。4B 模型文字推理相对弱，反而更依赖截图，在视觉类任务上有意外优势。

**假说 C：capability-environment gap 放大**
B0 235B 更"自信"地执行操作，在 SoM 模式下可能更早 finish（错误），而 B1 4B 更保守，多探索步骤反而偶尔找到正确路径。数据支持：B0 SoM fail_early_finish=12（5.4%）vs B1 SoM fail_early_finish 估计更低。

### 5.3 结论

B0 SoM 的劣势无法单一归因，**parse_error 是主要可量化原因**（可能占 4.19pp 差距的大部分）；text-over-vision scale effect 和 early finish 是次要假说，需要通过 parse_error 修复后的重跑数据验证。

---

## 六、共性脚手架缺陷（与 B1 SoM 相同）

以下问题在 B1 SoM 中已记录，B0 再次确认：

- **`<select>` 下拉菜单三层不可达**（VWA 框架级缺陷，B0 尤其严重——capability-environment gap 使 B0 更执着于"正确"路径，cycle detection 触发更快，见 B0_DOM_digest §5）
- **confirm 弹窗不可交互**：Delete 操作被取消，所有删除任务均失败
- **N/A 任务 False Positive（10/10）**：B0 N/A 任务全部误判 success=1.0，机制与 B1 相同（Agent prompt 无 N/A 出口）
- **`<select>` 分类导航**：B0 SoM 与 DOM 类似，因 capability-environment gap 被困（见 DOM digest）
- **极少翻页**：SoM 模式下 agent 仍以 scroll 为主，分页导航依赖程度低于 B0 DOM

---

## 七、B0 SoM 效率分析

| 指标 | B0 SoM | B0 DOM | B0 Vision |
|------|--------|--------|-----------|
| 平均步数 | 8.27 | 14.10 | 8.02 |
| 平均成本/ep | $0.0411 | $0.0457 | **$0.0256** |
| cost_efficiency_ratio | 0.0854 | 0.1161 | — |
| avg_wasted_cost | $0.0376 | $0.0464 | — |

SoM 步数（8.27）远少于 DOM（14.10），接近 Vision（8.02）——SoM 截图使 agent 决策更快（无需在 DOM 文本中搜索信息）。成本略低于 DOM，但高于 Vision（图像 token 开销高）。

---

*生成时间：2026-04-15*
*数据来源：B0_3mode_classifieds_20260413 phase1_som_router_0*
*B0 SoM 三模式定量对比见 `B0_findings.md`；B0 vs B1 跨模型对比见 `B0_B1_findings.md`*
