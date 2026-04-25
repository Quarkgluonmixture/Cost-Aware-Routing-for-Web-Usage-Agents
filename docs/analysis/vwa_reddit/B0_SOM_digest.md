# B0 Reddit -- SoM 模式分析

> B0 = Qwen3-VL-235B-A22B（proxy API），SoM 模式，Reddit 站点
> 210 episodes, 1682 steps, adjusted SR = 10.48% (22/210)
> digest 数据来源：`digest_som.jsonl`（182 行失败 episode 定性分析）
> 跨模式对比见 `B0_findings.md`

---

## 1. 总览

### 1.1 核心指标

| 指标 | 值 |
|------|-----|
| Raw SR | 13.33% (28/210) |
| Adjusted SR | 10.48% (22/210) |
| FP: N/A | 3 |
| FP: Visual | 0 |
| FP: Eval | 2 |
| 平均步数 | 8.01 |
| 平均成本 ($/ep) | $0.0384 |
| P95 延迟 | 78,542 ms |
| No-op rate | 20.2% |
| Page unchanged rate | 30.2% |
| Cost efficiency ratio | 0.1357 |

### 1.2 失败原因分布

| 失败原因 | 数量 | 占比 |
|----------|------|------|
| fail_no_progress | 72 | 34.3% |
| fail_early_finish | 30 | 14.3% |
| fail_finish_eval_mismatch | 25 | 11.9% |
| fail_max_steps_search_repeat | 13 | 6.2% |
| fail_finish_wrong_url_not_found | 11 | 5.2% |
| fail_parse_error | 8 | 3.8% |
| fail_incomplete_or_stuck | 8 | 3.8% |
| fail_finish_claim_missing | 7 | 3.3% |
| fail_finish_empty_answer | 4 | 1.9% |
| fail_max_steps_click_back_loop | 2 | 1.0% |
| fail_max_steps | 1 | 0.5% |
| fail_finish_wrong_url_left_target | 1 | 0.5% |

**前三大失败原因占 60.5%**：no_progress (34.3%) + early_finish (14.3%) + eval_mismatch (11.9%)。

**SoM 特色**：`fail_no_progress` 高达 34.3%（DOM 仅 22.4%），是 SoM 最显著的问题。`fail_early_finish` 14.3%（DOM 仅 6.2%），说明 SoM agent 更容易过早判定任务完成。

---

## 2. Digest 定性分析分类

> 基于 182 行 digest_som.jsonl 的 category 字段统计。

| 类别 | 数量 | 占比 | 说明 |
|------|------|------|------|
| 执行停滞 | 76 | 41.8% | 点击失败/页面无响应导致卡住 |
| 过早结束 | 40 | 22.0% | 未完成关键步骤即 finish |
| 搜索循环 | 21 | 11.5% | 重复使用同一搜索词 |
| 答案对齐错误 | 20 | 11.0% | 找到内容但答案不匹配 |
| 目标不可达 | 10 | 5.5% | 任务本身无法完成 |
| 导航循环 | 8 | 4.4% | click-back 循环 |
| 导航失败 | 4 | 2.2% | 未进入目标页面 |
| 事实推理错误 | 2 | 1.1% | 误判页面内容 |
| 导航错误 | 1 | 0.5% | 进入错误 subreddit |

### 2.1 执行停滞（76 个，41.8%）-- SoM 最大失败模式

**核心问题**：SoM 标注后 agent 使用 SoM 标记 ID 进行 click 操作，但大量 click 失败（页面无响应）。这在 Reddit 站点尤为严重。

**典型模式**：
- **连续 click 同一链接失败**：如 task 1 连续 5 次 click "61 comments" 链接，click_fail_rate=80%
- **subreddit 链接点击无响应**：如 task 16 连续 5 次 click "pics" 链接，task 35 连续 3 次 click "washingtondc" 链接
- **Choose File 按钮不可交互**：如 task 27/29 连续 3 次 click "Choose File"，VWA Playwright 限制

**与 DOM 的差异**：DOM 的执行停滞仅 21.5%，因为 DOM 使用 element_id 直接操作（更稳定），而 SoM 的视觉标注 ID 可能指向错误坐标。

### 2.2 过早结束（40 个，22.0%）

SoM agent 看到截图后更容易"视觉确认"并过早 finish：
- **未切换排序即结束**：如 task 34，agent 进入 /f/boston 切换到 Top 排序后看到 "There's nothing here..." 直接 finish
- **误认已到达目标**：如 task 38，agent 点击 nyc 链接进入 /f/nyc 后立即 finish，但任务可能需要进一步操作
- **task 33**：agent 到达用户评论汇总页面后误认为已完成"导航到评论区"任务

**这是 SoM 的"过度自信"问题**：截图提供了视觉确认感，agent 更快地判定任务完成，但往往遗漏了关键步骤（如排序切换、进入评论区、验证帖子内容）。

### 2.3 搜索循环（21 个，11.5%）

SoM 的搜索循环率（11.5%）低于 DOM（29.6%），说明 SoM 的视觉信息帮助 agent 更快定位目标（或更快放弃搜索转而 early_finish）。但仍有部分任务陷入搜索循环：
- task 23: "pumpkin robot" x15（与 DOM 完全相同的搜索词）
- task 28: "baseball" 重复搜索

### 2.4 答案对齐错误（20 个，11.0%）

Agent 找到了目标或相似内容，但最终答案与参考答案不一致：
- task 5: 回答 "13" 但参考答案要求不同（找错帖子）
- task 22: 报告的 top comment 与参考答案不符
- task 25: 回答 "Love seeing success for the little guy" 但非目标帖子

---

## 3. SoM 特有问题

### 3.1 SoM Failure Type 分布

| som_failure_type | 数量 | 占比 |
|-----------------|------|------|
| 不适用 | 89 | 48.9% |
| **text_over_vision** | **83** | **45.6%** |
| ID 幻觉 | 8 | 4.4% |
| 标注遮挡 | 2 | 1.1% |

**text_over_vision（83 个，45.6%）**：agent 有截图但仍依赖文字线索而非视觉匹配。这在 Reddit "根据图片找帖子"类任务中尤为突出 -- agent 从任务描述中提取关键词搜索，而非通过截图视觉匹配图片内容。

**典型案例**：
- task 12: 用 "mariupol" 搜索而非视觉匹配目标图片
- task 13: 用 "prudential building" 搜索而非识别城市景观
- task 17: 在 memes 版块反复 click-back 循环，未利用视觉确认图片是否匹配
- task 24: 通过 "astronaut painting" 搜索找到帖子后在图片页和详情页间循环
- task 34: 切换到 Top 排序后未利用视觉信息判断帖子是否为城市照片

**ID 幻觉（8 个，4.4%）**：agent 生成了不存在的 SoM 标记 ID，导致 click 操作指向错误位置。典型表现为连续多次 click 同一目标但页面无响应（如 task 1, task 16）。

**标注遮挡（2 个，1.1%）**：SoM 标注框遮挡了页面内容（如 task 30），影响了 agent 的操作。

### 3.2 脚手架归因

| is_scaffolding_issue | 数量 | 占比 |
|---------------------|------|------|
| 否 | 98 | 53.8% |
| 是 | 84 | 46.2% |

84 个 episode（46.2%）被归因为脚手架/表征缺陷，高于 DOM 的 34.4%。SoM 的脚手架问题主要包括：
- SoM 标注 ID 幻觉导致的 click 失败
- click-back 循环（导航循环）
- VWA 框架限制（文件上传不可达）
- SoM 标注遮挡

### 3.3 SoM Visual Used 分布

> `som_visual_used` 字段标记 agent 是否实际利用了 SoM 截图中的视觉信息。

在 182 个失败 episode 中，`som_visual_used="是"` 的比例约为 40%，说明多数失败 episode 中 agent 并未有效利用 SoM 提供的视觉信息，回退到了文字搜索模式。

---

## 4. Reddit SoM 特色行为

### 4.1 早期终止倾向

SoM agent 在 Reddit 上更倾向于早期终止（fail_early_finish 14.3%，DOM 仅 6.2%）。这与 Classifieds 中 SoM early_finish 12.8% 的模式一致 -- SoM 截图提供的"视觉确认"使 agent 更果断但也更草率。

Reddit 的 early_finish 案例多为：
- 进入 subreddit 后立即 finish（未进入具体帖子）
- 进入帖子后未进入评论区即 finish
- 未完成排序切换即 finish

### 4.2 Click 失败率

SoM 模式下 Reddit 站点的 click 失败率在某些 episode 中极高（50%-80%），尤其是：
- 论坛列表页的 subreddit 链接
- 帖子列表页的 "N comments" 链接
- "Choose File" 上传按钮

这些失败与 SoM 标注 ID 的坐标精度有关 -- Reddit (Postmill) 的 UI 元素布局可能与 SoM 标注算法的坐标计算不完全匹配。

### 4.3 导航循环

SoM 模式下有 8 个导航循环案例（4.4%），典型模式：
- click 帖子标题 -> back -> click 同一帖子 -> back -> ...（如 task 3, task 18）
- click 图片 -> back -> click 帖子 -> back -> ...（如 task 24）

这种循环在 SoM 模式下更常见于 agent 通过视觉匹配找到了"可能的"目标帖子，但无法确认是否正确，反复进出帖子页面。

---

## 5. 与 DOM 对比

### 5.1 模式对比总结

| 维度 | SoM | DOM |
|------|-----|-----|
| Adjusted SR | **10.48%** | 7.62% |
| 平均步数 | **8.01** | 12.70 |
| 平均成本 | **$0.0384** | $0.0516 |
| 主要失败模式 | 执行停滞 (41.8%) | 搜索循环 (29.6%) |
| 搜索循环率 | **11.5%** | 29.6% |
| 过早结束率 | 22.0% | **9.7%** |
| 脚手架问题率 | 46.2% | **34.4%** |
| text_over_vision | 45.6% | N/A |

### 5.2 SoM 优势

- **更高的 adjusted SR**（10.48% vs 7.62%，+2.86pp）
- **更低的成本**（$0.0384 vs $0.0516，-25.6%，Wilcoxon p<0.001）
- **更少的搜索循环**（11.5% vs 29.6%） -- 视觉信息帮助 agent 更快定位目标
- **独占 12 个 task**（vs DOM 独占 6 个）

### 5.3 SoM 劣势

- **更高的 no-op rate**（20.2% vs 12.3%） -- SoM ID 幻觉导致的 click 失败
- **更多的 early_finish**（14.3% vs 6.2%） -- 视觉过度自信
- **更高的脚手架问题率**（46.2% vs 34.4%） -- SoM 标注引入的额外问题
- **text_over_vision 严重**（45.6%） -- agent 未充分利用视觉信息

---

*更新时间：2026-04-23*
*数据来源：B0_3mode_reddit_20260422 analysis/digest/digest_som.jsonl*
