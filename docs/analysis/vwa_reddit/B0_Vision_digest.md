# B0 Reddit -- Vision 模式分析

> B0 = Qwen3-VL-235B-A22B（proxy API），Vision 模式，Reddit 站点
> 210 episodes, adjusted SR = 6.34% (13/205)
> digest 数据来源：`digest_vision.jsonl`（191 行失败 episode 定性分析，0% dry-run）
> 跨模式对比见 `B0_findings.md`

---

## 1. 总览

### 1.1 核心指标

| 指标 | 值 |
|------|-----|
| Raw SR | 8.57% (18/210) |
| Adjusted SR | 6.34% (13/205) |
| FP: N/A | 4 |
| FP: Visual | 0 |
| FP: Eval | 1 |
| 平均步数 | 6.87 |
| 平均成本 ($/ep) | $0.0227 |
| P95 延迟 | 55,568 ms |
| No-op rate | 26.4% |
| Page unchanged rate | 33.7% |
| Cost efficiency ratio | 0.0977 |

**与旧版（104/210 incomplete）的差异**：完整数据下 Vision 的 adjusted SR 从 0% 升至 6.34%（13 个真阳性成功），此前全部 raw success 为 FP 的结论已不成立。Vision 在 Reddit 上有有限但真实的能力。

### 1.2 失败原因分布

| 失败原因 | 数量 | 占比 |
|----------|------|------|
| fail_no_progress | 84 | 40.0% |
| fail_finish_eval_mismatch | 31 | 14.8% |
| fail_incomplete_or_stuck | 27 | 12.9% |
| fail_early_finish | 21 | 10.0% |
| success | 18 | 8.6% |
| fail_finish_claim_missing | 14 | 6.7% |
| fail_finish_wrong_url_not_found | 11 | 5.2% |
| fail_finish_empty_answer | 3 | 1.4% |
| fail_max_steps_search_repeat | 1 | 0.5% |

**前三大失败原因占 67.7%**：no_progress (40.0%) + eval_mismatch (14.8%) + incomplete_or_stuck (12.9%)。

**Vision 特色**：`fail_no_progress` 高达 40.0%（DOM 22.4%，SoM 34.3%），坐标点击不稳定性的直接体现。`fail_max_steps_search_repeat` 几乎为零（0.5%），因 Vision 模式下搜索操作极少（type 成功率低）。

---

## 2. Digest 定性分析分类

> 基于 191 行 digest_vision.jsonl 的 category 字段统计。

| 类别 | 数量 | 占比 | 说明 |
|------|------|------|------|
| 执行停滞 | 85 | 44.5% | 坐标点击失败/页面无响应导致卡住 |
| 过早结束 | 29 | 15.2% | 未完成关键步骤即 finish |
| 答案对齐错误 | 25 | 13.1% | 找到内容但答案不匹配 |
| 搜索循环 | 21 | 11.0% | 重复使用同一搜索词 |
| 目标不可达 | 11 | 5.8% | 任务本身无法通过当前交互完成 |
| 导航循环 | 9 | 4.7% | click-back 循环 |
| 事实推理错误 | 6 | 3.1% | 误判页面内容 |
| 导航失败 | 4 | 2.1% | 未进入目标页面 |
| 导航错误 | 1 | 0.5% | 进入错误 subreddit |

### 2.1 执行停滞（85 个，44.5%）-- Vision 最大失败模式

**核心问题**：Vision 模式依赖 agent 输出像素坐标进行 click 操作。Reddit (Postmill) 的 UI 元素通常较小（论坛链接、评论链接等），坐标精度不足导致大量 click 失败。

**典型模式**：
- **搜索栏 click 不 type**：agent 反复 click 搜索栏区域但从不输出 type 指令，缺乏 AXTree 的 `<input>` 语义提示
- **连续点击同一坐标无响应**：3+ 次 click 相同坐标，page_changed=false
- **帖子链接 click 失败**：论坛列表页中帖子标题/评论链接精度不足

**与 DOM 的差异**：DOM 的 no_progress 仅 22.4%，因为 DOM 使用 element_id 直接操作（更稳定），不依赖坐标。

### 2.2 过早结束（29 个，15.2%）

Vision agent 看到截图后更容易"视觉确认"并过早 finish：
- **step 0 直接 finish**：大量 `page_image_query` 类型任务，agent 从首页截图直接回答视觉问题
- **误认已到达目标**：进入 subreddit 后立即 finish，未进入具体帖子
- **错误视觉计数**：从截图数对象，数错后立即 finish

**B0 vs B1 差异**：B0 (235B) 的 early_finish 比例（15.2%）低于 B0 SoM（22.0%）但高于 B0 DOM（9.7%）。235B 的视觉理解能力使其比 4B 更倾向于相信截图，但也能更准确地判断是否需要进一步操作。

### 2.3 答案对齐错误（25 个，13.1%）

Agent 找到了目标或相似内容，但最终答案与参考答案不一致。Vision 模式的答案对齐错误（13.1%）高于 SoM（11.0%），因为视觉理解产生的答案表述更多样化。

### 2.4 搜索循环（21 个，11.0%）

尽管 Vision 模式下搜索操作极少（type 成功率低），仍有 11.0% 的 episode 出现搜索循环。这些案例中 agent 成功进入了搜索流程，但搜索词不精确导致循环。搜索循环率远低于 DOM（29.6%），与 SoM（11.5%）接近。

---

## 3. 成本分解

| 类型 | $/ep | 占比 |
|------|------|------|
| 有效成本 | $0.0145 | 63.9% |
| No-op 成本 | $0.0050 | 22.0% |
| 循环成本 | $0.0032 | 14.1% |
| **总计** | **$0.0227** | 100% |

Vision 是三模式中成本最低的（DOM $0.0516，SoM $0.0387），但有效成本占比也最高（63.9%），因为 Vision episode 步数少（6.87 步），来不及积累大量无效操作就已被早停终止。

---

## 4. 路由信号

### 4.1 跨模式 AUROC

| 信号 | Vision AUROC | CI | 类型 |
|------|-------------|-----|------|
| ep_mean_verbalized | **0.778** | [0.660, 0.873] | verbalized |
| max_repeat_streak | **0.709** | [0.598, 0.809] | behavioral |
| action_diversity | 0.689 | [0.555, 0.807] | behavioral |
| url_revisit_max | 0.670 | [0.540, 0.796] | behavioral |
| url_unique_count | 0.622 | [0.453, 0.781] | behavioral |
| ep_min_verbalized | 0.617 | [0.454, 0.768] | verbalized |
| url_revisit_count | 0.610 | [0.468, 0.746] | behavioral |
| action_unique_types | 0.609 | [0.436, 0.762] | behavioral |

**ep_mean_verbalized 是最强信号**（AUROC=0.778），超过 SoM（0.714）但低于 DOM（0.830）。与旧版（104 ep, AUROC=0.410）截然不同 -- 完整数据下 Vision 的 verbalized confidence 有良好的区分力。

**Behavioral 信号**中 `max_repeat_streak`（0.709）和 `action_diversity`（0.689）可用，CI 均不跨 0.5。

### 4.2 跨模式信号对比

| 信号 | DOM | SoM | Vision |
|------|-----|-----|--------|
| ep_mean_verbalized | **0.830** | 0.714 | 0.778 |
| max_repeat_streak | 0.678 | 0.636 | **0.709** |
| action_diversity | 0.581 | 0.603 | **0.689** |

Vision 模式下 behavioral 信号区分力更强（max_repeat_streak 0.709 > DOM 0.678），因为 Vision 的失败模式更极端（click 死循环 vs DOM 的渐进搜索），信号对比度更高。

---

## 5. 跨模式交叉分析

### 5.1 独占成功集（adjusted）

| 独占集 | 数量 | 占比 |
|--------|------|------|
| all_fail | 176 | 83.8% |
| only_som | 9 | 4.3% |
| dom_and_som_not_vision | 8 | 3.8% |
| **only_vision** | **5** | **2.4%** |
| only_dom | 4 | 1.9% |
| som_and_vision_not_dom | 4 | 1.9% |
| all_success | 2 | 1.0% |
| dom_and_vision_not_som | 2 | 1.0% |

**Vision 独占 5 个 task**（全部为 single_navigation）。这些任务 DOM 和 SoM 均失败但 Vision 成功，说明纯视觉输入在特定场景有不可替代的价值。

### 5.2 Oracle 选择分布（adjusted）

| 模式 | Oracle 选择数 | 占比 |
|------|-------------|------|
| SoM | 13 | 38.2% |
| Vision | 11 | 32.4% |
| DOM | 10 | 29.4% |

Oracle 路由中 Vision 贡献 32.4%，说明**即使是三模式中 SR 最低的 Vision，在路由框架中仍有重要角色** -- 它在 SoM 和 DOM 都失败的 task 上提供了独特的成功路径。

---

## 6. 与 DOM/SoM 对比

| 维度 | Vision | DOM | SoM |
|------|--------|-----|-----|
| Adjusted SR | 6.34% | 8.78% | **11.71%** |
| 平均步数 | **6.87** | 12.70 | 8.09 |
| 平均成本 | **$0.0227** | $0.0516 | $0.0387 |
| P95 延迟 | **55,568ms** | 73,618ms | 74,101ms |
| 主要失败 | no_progress (40.0%) | eval_mismatch (23.8%) | no_progress (34.3%) |
| 搜索循环率 | **0.5%** | 29.6% | 11.5% |
| 过早结束率 | 15.2% | 9.7% | **22.0%** |
| 独占成功数 | 5 | 4 | 9 |

### 6.1 Vision 优势

- **最低成本**（$0.0227，DOM 的 44%，SoM 的 59%，Wilcoxon p<0.001 ★★★）
- **最低延迟**（P95 55.6s vs DOM/SoM ~74s，Wilcoxon p<0.001 ★★★）
- **独占 5 个 task** -- 在纯文本和 SoM 标注都失败的场景提供视觉直觉路径
- **搜索循环几乎为零**（0.5%），不会浪费 30 步在无效搜索上

### 6.2 Vision 劣势

- **最低 adjusted SR**（6.34%，SoM vs Vision McNemar p=0.036 ★）
- **最高 no-op rate**（26.4% vs DOM 12.3%）-- 大量 click 失败
- **最高 page unchanged rate**（33.7%）-- 坐标精度不足
- **no_progress 占比最高**（40.0%）-- 坐标点击的固有不稳定性

---

## 7. 结论

Vision 模式在 Reddit 站点 adjusted SR = 6.34%，显著低于 SoM（11.71%，p=0.036 ★）。主要瓶颈是坐标点击精度不足（no_progress 40.0%，no-op rate 26.4%）。

**路由价值**：尽管 SR 最低，Vision 独占 5 个 task 且 Oracle 贡献占比 32.4%，在路由框架中不可省略。其极低的成本（$0.0227/ep）使其成为低成本探测的理想选择：在置信度不高时先用 Vision 试探，失败则切换到更贵但更稳定的 SoM/DOM。

**信号质量**：ep_mean_verbalized AUROC=0.778，可作为路由决策的可靠信号。

---

*更新时间：2026-04-24*
*数据来源：B0_3mode_reddit_20260422 analysis/（210/210 完整数据）*
*替代旧版（104/210 不完整数据，adjusted SR=0%）*
