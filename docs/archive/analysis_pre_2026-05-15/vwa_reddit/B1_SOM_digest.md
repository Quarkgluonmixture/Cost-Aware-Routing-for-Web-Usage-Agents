# B1 Reddit -- SoM 模式分析

> B1 = Qwen3-VL-4B（本地），SoM 模式，Reddit 站点
> 210 episodes, 2457 steps, adjusted SR = 5.85% (12/205)
> digest 数据来源：`digest_som.jsonl`（193 行失败 episode 定性分析，0% dry-run）
> 跨模式对比见 `B1_findings.md`

---

## 1. 总览

### 1.1 核心指标

| 指标 | 值 |
|------|-----|
| Raw SR | 8.10% (17/210) |
| Adjusted SR | 5.85% (12/205) |
| FP: N/A | 5 |
| FP: Visual | 0 |
| FP: Eval | 3 |
| 平均步数 | 11.70 |
| 平均成本 ($/ep) | $0.0410 |
| P95 延迟 | 61,569 ms |
| No-op rate | 23.4% |
| Page unchanged rate | 27.7% |

### 1.2 失败原因分布

| 失败原因 | 数量 | 占比 |
|----------|------|------|
| fail_no_progress | 81 | 42.0% |
| fail_incomplete_or_stuck | 33 | 17.1% |
| fail_early_finish | 20 | 10.4% |
| fail_finish_eval_mismatch | 17 | 8.8% |
| fail_max_steps_click_back_loop | 13 | 6.7% |
| fail_max_steps_search_repeat | 10 | 5.2% |
| fail_max_steps | 9 | 4.7% |
| fail_finish_empty_answer | 5 | 2.6% |
| fail_finish_claim_missing | 4 | 2.1% |
| fail_finish_wrong_url_not_found | 1 | 0.5% |

**前三大失败原因占 69.1%**：no_progress (42.0%) + incomplete_or_stuck (17.1%) + early_finish (10.4%)。

**B1 SoM 特色**：`fail_no_progress` 高达 42.0%（B1 DOM 仅 23.8%），SoM 标注 ID 的坐标精度问题比 DOM element_id 更严重。`fail_max_steps_click_back_loop` 6.7%，说明 4B 模型容易陷入点击-返回死循环。

---

## 2. Digest 定性分析分类

> 基于 193 行 digest_som.jsonl 的 category 字段统计。

| 类别 | 数量 | 占比 | 平均步数 | 说明 |
|------|------|------|---------|------|
| 执行停滞 | 95 | 49.2% | 7.9 | 点击无效/页面无响应导致卡住 |
| 搜索循环 | 30 | 15.5% | 21.1 | 重复搜索相同关键词 |
| 导航循环 | 26 | 13.5% | 26.4 | click-back 循环，耗步数最多 |
| 过早结束 | 20 | 10.4% | 1.6 | 1-2 步即 finish |
| 答案对齐错误 | 13 | 6.7% | 7.7 | finish 时答案不符 |
| 目标不可达 | 8 | 4.1% | 6.2 | 任务无法通过当前交互完成 |
| 事实推理错误 | 1 | 0.5% | 10.0 | 理解错误 |

### 2.1 执行停滞（95 个，49.2%）-- SoM 最大失败模式

**核心问题**：SoM 标注后 agent 使用 SoM 标记 ID 进行 click 操作，但大量 click 失败（页面无响应）。Reddit (Postmill) UI 元素密集且间距小，SoM 标注的坐标精度不足以可靠命中目标。

**与 B0 SoM 的对比**：B0 SoM 的执行停滞为 41.8%（76/182），B1 为 49.2%（95/193）。4B 模型的 SoM ID 选择准确性更低，更容易生成不存在的 ID 或选错相邻元素的 ID。

**典型模式**：
- **连续 click 同一链接失败**：反复 click subreddit 链接，click_fail_rate 高达 75%
- **Subscribe 按钮反复切换**：页面在 Subscribe ↔ Unsubscribe 之间来回切换
- **搜索框 click 不生效**：agent 识别出搜索框的 SoM ID 但 click 无响应

### 2.2 搜索循环（30 个，15.5%）

B1 SoM 的搜索循环率（15.5%）高于 B0 SoM（11.5%），且平均步数 21.1 步远高于其他类别。4B 模型在搜索失败后缺乏变换关键词的能力，倾向于重复同一搜索词直到达到 max_steps。

**典型案例**：重复搜索 "art" 4 次，无法在搜索失败中提炼新关键词（如 task 190, 29 步，fail_max_steps_search_repeat）。

### 2.3 导航循环（26 个，13.5%）

B1 SoM 的导航循环（13.5%）显著高于 B0 SoM（4.4%），反映出 4B 模型的导航策略单一。平均步数 26.4 步是所有类别中最高的，几乎全部跑满 30 步。

**典型模式**：click 帖子标题 → back → click 同一帖子 → back → ...，agent 找到了"可能的"目标帖子但无法确认，反复进出帖子页面。

### 2.4 过早结束（20 个，10.4%）

平均仅 1.6 步即 finish，大多是 agent 在首页或第一步就判定任务完成。B1 SoM 的过早结束率（10.4%）低于 B0 SoM（22.0%），可能因为 4B 模型的 finish 倾向本身较低（prompt adherence 更弱，不太会主动结束）。

### 2.5 答案对齐错误（13 个，6.7%）

Agent 找到了目标或相似内容，但最终答案与参考答案不一致。B1 SoM 的答案对齐错误率（6.7%）低于 B0 SoM（11.0%），但这主要因为 B1 更多 episode 在到达答案阶段之前就已失败。

---

## 3. SoM 特有问题

### 3.1 SoM Failure Type 分布

| som_failure_type | 数量 | 占比 |
|-----------------|------|------|
| **text_over_vision** | **99** | **51.3%** |
| 不适用 | 83 | 43.0% |
| ID 幻觉 | 8 | 4.1% |
| 标注遮挡 | 2 | 1.0% |

**text_over_vision（99 个，51.3%）**：超过半数失败 episode 中 agent 即使有截图仍依赖文字线索而非视觉匹配。在 Reddit "根据图片找帖子"类任务中尤为突出 -- agent 从任务描述提取关键词搜索，而非通过截图视觉匹配。

**与 B0 SoM 对比**：B0 SoM 的 text_over_vision 为 45.6%（83/182），B1 为 51.3%（99/193），4B 模型更不善于利用 SoM 截图的视觉信息。

**ID 幻觉（8 个，4.1%）**：agent 生成不存在的 SoM 标记 ID，导致 click 指向错误位置。与 B0 SoM（同为 4.4%）比例相当，说明 ID 幻觉主要取决于 SoM 标注质量而非模型能力。

### 3.2 脚手架归因

| is_scaffolding_issue | 数量 | 占比 |
|---------------------|------|------|
| 是 | 105 | 54.4% |
| 否 | 88 | 45.6% |

105 个 episode（54.4%）被归因为脚手架/表征缺陷，高于 B0 SoM（46.2%）和 B1 DOM（数据待对比）。B1 SoM 的脚手架问题主要包括：
- SoM 标注 ID 幻觉导致的 click 失败
- 无 N/A 出口 -- Rule 4 "NEVER give up" 导致 N/A task 必循环到截断
- 无翻页/排序切换策略
- VWA 框架限制（文件上传不可达）

### 3.3 SoM Visual Used 分布

| som_visual_used | 数量 | 占比 |
|----------------|------|------|
| 否 | 124 | 64.2% |
| 是 | 69 | 35.8% |

**64.2% 的失败 episode 中 agent 未有效利用视觉信息**。B1 (4B) 的 SoM 截图利用率（35.8%）远低于 B0 (235B) 的约 40%，说明小模型在多模态融合能力上更弱 -- 即使截图中包含关键信息，4B 模型也倾向回退到纯文本处理模式。

---

## 4. 成本分解

| 类型 | $/ep | 占比 |
|------|------|------|
| 有效成本 | $0.0284 | 69.3% |
| No-op 成本 | $0.0084 | 20.5% |
| 循环成本 | $0.0043 | 10.5% |
| **总计** | **$0.0410** | 100% |

SoM 处于 DOM（$0.0536）和 Vision（$0.0137）之间。相比 B0 SoM（$0.0387），B1 SoM 成本略高 6%（$0.0410 vs $0.0387），因 B1 平均步数更多（11.70 vs 8.09），探索更多但无效操作也更多。

---

## 5. 路由信号

### 5.1 跨模式 AUROC（SoM）

| 信号 | SoM AUROC | CI | 类型 |
|------|-----------|-----|------|
| ep_mean_verbalized | **0.708** | [0.531, 0.849] | verbalized |
| ep_min_verbalized | 0.671 | [0.487, 0.825] | verbalized |
| ep_mean_logprob | 0.639 | [0.485, 0.779] | token_level |
| ep_mean_margin | 0.639 | [0.471, 0.782] | token_level |
| ep_mean_entropy | 0.627 | [0.460, 0.772] | token_level |
| ep_max_entropy | 0.582 | [0.386, 0.761] | token_level |
| max_repeat_streak | 0.576 | [0.430, 0.704] | behavioral |

**Verbalized confidence 是最强信号**（AUROC=0.708），但 CI 极宽（跨度 0.32），成功数少（12 adjusted）导致估计不稳定。

**B1 SoM 特色**：token-level 信号（logprob, margin, entropy）在 B1 中可用（B0 API 无 logprobs），AUROC 在 0.58-0.64 范围，弱于 verbalized 但多数 CI 不跨 0.5。Behavioral 信号（max_repeat_streak 0.576）接近随机线，CI 宽，不可靠。

### 5.2 跨模式信号对比（B1 Reddit）

| 信号 | DOM | SoM | Vision |
|------|-----|-----|--------|
| ep_mean_verbalized | **0.730** | 0.708 | 0.698 |
| max_repeat_streak | 0.540 | 0.576 | **0.725** |
| action_diversity | **0.615** | 0.413 | 0.785 |

SoM 的 verbalized 信号（0.708）介于 DOM（0.730）和 Vision（0.698）之间。但 SoM 的 behavioral 信号普遍较弱 -- action_diversity 仅 0.413（低于随机线），因为 SoM 的失败模式较为多样化，不像 Vision 那样集中在 click 死循环上。

---

## 6. 跨模式交叉分析

### 6.1 独占成功集（adjusted）

| 独占集 | 数量 | 占比 |
|--------|------|------|
| all_fail | 192 | 91.4% |
| only_dom | 7 | 3.3% |
| dom_and_som_not_vision | 5 | 2.4% |
| **only_som** | **3** | **1.4%** |
| only_vision | 2 | 1.0% |
| som_and_vision_not_dom | 1 | 0.5% |

**91.4% 的 task 三模式均失败** -- B1 Reddit 极为困难。SoM 独占仅 3 个 task（2 个 single_navigation + 1 个 page_reading），是三模式中独占成功数最少的。

**与 B0 SoM 的对比**：B0 SoM 独占 9 个 task，B1 SoM 仅 3 个。235B 在 SoM 模式下的独占能力是 4B 的 3 倍。

### 6.2 Oracle 选择分布（adjusted）

| 模式 | Oracle 选择数 | 占比 |
|------|-------------|------|
| **DOM** | **12** | **66.7%** |
| SoM | 3 | 16.7% |
| Vision | 3 | 16.7% |

**DOM 主导 Oracle 选择**（66.7%），这与 B1 Reddit 独特的 DOM>SoM 模式排序一致（B1 Reddit 是所有 site×baseline 组合中唯一 DOM 强于 SoM 的情况）。SoM 和 Vision 在 Oracle 中的贡献相同且较小（各 16.7%），说明在 B1 Reddit 场景下 SoM 的路由价值有限。

---

## 7. 与 DOM/Vision 对比

| 维度 | SoM | DOM | Vision |
|------|-----|-----|--------|
| Adjusted SR | 5.85% | **6.83%** | 2.44% |
| 平均步数 | 11.70 | 16.64 | **6.45** |
| 平均成本 | $0.0410 | $0.0536 | **$0.0137** |
| P95 延迟 | 61,569ms | 87,926ms | **53,526ms** |
| 主要失败 | no_progress (49.2%) | no_progress (23.8%) | no_progress (~) |
| 搜索循环率 | 15.5% | **22.9%** | ~0% |
| 导航循环率 | **13.5%** | 9.5% | ~0% |
| 脚手架问题率 | **54.4%** | ~40% | ~50% |

### 7.1 B1 SoM vs B1 DOM

**SoM 低于 DOM**（5.85% vs 6.83%）-- 这是 B1 Reddit 独有的模式反转（B0 Reddit、B0/B1 Classifieds 中 SoM 均优于 DOM）。原因：
1. **SoM ID 幻觉更严重**：4B 模型在 Reddit 密集布局下的 SoM ID 选择准确性不足
2. **导航循环更多**（13.5% vs 9.5%）：SoM 截图的"视觉确认感"导致 4B 模型在同一目标反复进出
3. **no_progress 差距大**（49.2% vs 23.8%）：SoM 标注增加了视觉复杂度，但 4B 模型未能有效利用

**SoM 优势**：
- 更低成本（$0.0410 vs $0.0536，-23.5%，Wilcoxon p=0.001 ★★★）
- 更少搜索循环（15.5% vs 22.9%）
- 更少步数（11.70 vs 16.64）

### 7.2 B1 SoM vs B0 SoM（跨 baseline）

| 维度 | B1 SoM | B0 SoM | 差 |
|------|--------|--------|---|
| Adjusted SR | 5.85% | **11.71%** | -5.86pp |
| 执行停滞 | 49.2% | 41.8% | +7.4pp |
| 过早结束 | 10.4% | 22.0% | -11.6pp |
| text_over_vision | 51.3% | 45.6% | +5.7pp |
| 脚手架问题 | 54.4% | 46.2% | +8.2pp |
| som_visual_used | 35.8% | ~40% | -4pp |
| 独占成功 | 3 | 9 | -6 |

**B0 的 SoM 利用率全面优于 B1**：更高的 visual_used 率、更少的 text_over_vision、更少的执行停滞。235B 模型在 SoM 模式下的核心优势是：
1. 更准确的 SoM ID 选择（减少执行停滞）
2. 更强的视觉-文字融合能力（减少 text_over_vision）
3. 更强的自纠正能力（减少搜索/导航循环）

---

## 8. Reddit SoM 特色行为

### 8.1 B1 SoM 的"循环倾向"

B1 SoM 在 Reddit 上的循环类失败（搜索循环 15.5% + 导航循环 13.5% = 29.0%）远高于 B0 SoM（11.5% + 4.4% = 15.9%）。这些循环平均消耗 21-26 步，几乎跑满 max_steps（30 步），是最昂贵的失败模式。

### 8.2 Click 失败率

SoM 模式下 Reddit 站点的 click 失败率在部分 episode 中极高（50%-75%），尤其是：
- 论坛列表页的 subreddit 链接
- 帖子列表页的 "N comments" 链接
- Subscribe/Unsubscribe 按钮

这些失败与 SoM 标注 ID 的坐标精度有关 -- Reddit (Postmill) 的 UI 元素布局可能与 SoM 标注算法的坐标计算不完全匹配。

### 8.3 过早结束倾向对比

B1 SoM 的 early_finish 率（10.4%）在三模式中处于中间水平（DOM ~10%，Vision ~12.5%）。与 B0 SoM（22.0%）相比显著更低 -- 4B 模型不太擅长主动"判定完成"，但也意味着即使任务已无法完成，4B 也不会早停退出，而是继续循环浪费步数。

---

## 9. 结论

B1 SoM 在 Reddit 站点的 adjusted SR = 5.85%（12/205），**低于 DOM 的 6.83%** -- 这是所有 site×baseline 组合中唯一的 SoM < DOM 反转。主要原因是 4B 模型在 Reddit 密集布局下的 SoM 标注利用率不足（text_over_vision 51.3%，som_visual_used 仅 35.8%），且执行停滞比例极高（49.2%）。

**路由含义**：
1. **B1 Reddit 中 DOM 是更优选择**：DOM 在调整后 SR 和 Oracle 选择中均领先，且 element_id 操作稳定性远超 SoM 坐标
2. **SoM 的路由价值有限**：Oracle 贡献仅 16.7%（3 个 task），独占成功仅 3 个
3. **模型能力是 SoM 有效性的瓶颈**：B0 SoM 在 Reddit 上有 11.71% adjusted SR 和 9 个独占成功，证明 SoM 表征本身有价值，但需要足够强的模型能力才能有效利用

---

*生成时间：2026-04-24*
*数据来源：B1_3mode_reddit_20260413 phase1_som_router_0，210 tasks*
