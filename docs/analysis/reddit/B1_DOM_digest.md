# B1 Reddit DOM Digest

> 占位 — Reddit B1 DOM 模式完成后填充定量分析。

## 当前状态

运行中，RUN_ID=`B1_3mode_reddit_20260412`，DOM condition 进行中。

## 早期观察（前 4 个 episode）

### 反复点击 Comment 链接（task 0/1/3）

**现象**：Agent 成功导航到帖子页面后，反复点击 "N comments" 链接（如 "45 comments" / "171 comments"），但 URL 始终不变。每步的 element_id 不同（DOM 重新渲染所致），但 bbox 完全一致。Agent thought 每步几乎相同（"clicking will navigate to the comment section"），confidence 保持 0.95。

**受影响 task**：

| Task | Intent | 目标链接 | 循环步数 | 结果 |
|------|--------|---------|---------|------|
| 0 | Navigate to comment section of [homemade] Pumpkin Loaf | "45 comments" | step 1-5 (5 次) | 30 步用完，score=0 |
| 1 | 同上（重复 task） | "45 comments" | step 1-5 (5 次) | 30 步用完，score=0 |
| 3 | Count comments mentioning 'spicy' in Beef Noods post | "171 comments" | step 1-5 (5 次) | 30 步用完，score=0 |

**action 序列示例（task 0）**：

```
step 0: click [401] → food 列表页 → 帖子页 ✓（正确导航）
step 1: click [3187] "45 comments" → URL 不变 ✗
step 2: click [9481] "45 comments" → URL 不变 ✗
step 3: click [15775] "45 comments" → URL 不变 ✗
step 4: click [22069] "45 comments" → URL 不变 ✗
step 5: click [28363] "45 comments" → URL 不变 ✗
... (重复至 step 29)
```

**根因分析**：

1. **Agent 已在目标页但不自知**：Reddit（Postmill）的帖子页面本身就是 comment section，URL `f/food/18838/homemade-...` 已包含评论。Agent 期望点击 "45 comments" 后跳转到不同 URL，但实际上该链接指向当前页面自身（锚点或自链接）。
2. **零自纠正**：连续 5 次点击同一位置（bbox `[152, 705, 81, 14]`）后 URL 不变，Agent 不调整策略，不尝试 scroll down 查看评论，不 finish。
3. **URL stuck 早停应能捕获**：此模式满足 §33 新增的 URL stuck 检测条件（连续 5 次 click 同 URL），但这些 episode 在修复前运行。

**与 Classifieds 的对比**：

- Classifieds 的 "stuck on same page" 主要是搜索结果页不翻页
- Reddit 的 "comment link" 是自链接死循环——更接近 §33 修复的 URL stuck 场景
- 两者共性：Agent 缺乏「已到达目标」的判断能力

### Task 2：正确导航但空 answer 提交

Task 2（navigate to comment section for 'Late 90's thrillers'）路径不同：

```
step 0: click [297] "Comments" → /f/movies/comments ✓
step 1: click [3283] permalink → /f/movies/128396/-/comment/2561509 ✓
step 2: finish (empty answer) → score=0
```

Agent 成功到达目标但提交空 answer。对于 `url_match` 评测类型，即使不 finish 也能得分——问题在于 Agent 选择了 finish 而非继续浏览。

---

## 持久性 Evaluator 错误

210/210 task 全部完成。其中 3 个 task 存在持久性 `evaluator_error:Page.goto net::ERR_ABORTED`，重跑后仍一致复现，属于 benchmark 环境缺陷（reddit Docker 特定页面 Playwright 全量加载失败，curl 200 但浏览器 load 事件中断）。

| Task | 评测 URL | 评测类型 | Agent steps | 说明 |
|------|---------|---------|-------------|------|
| 72 | `f/memes/127531` | program_html | 7 | 检查 comment 内容 |
| 146 | `f/wallstreetbets/50335` | program_html | 5 | 检查 comment 内容 |
| 172 | `f/jerseycity/62526` | program_html | 8 | 检查 comment 内容 |

**处理**：2026-04-17 清除旧 stub summary 后重跑全部 5 个 error task，其中 task 149 和 151 修复成功（评测器正常运行，score=0），上述 3 个仍失败。评测器导航至 reference URL 时 `ERR_ABORTED`，Agent 侧执行正常。最终 score=0 合理（均为 comment-posting 任务，Agent 均未成功发表评论）。

**与 Classifieds 先例对比**（见 `classifieds/B1_DOM_digest.md` §10.4）：Classifieds 的 3 个 evaluator_error 来自 OpenAI API key 缺失或 program_html 超时，可离线重评修复；Reddit 的 3 个来自 Docker 页面加载缺陷，无法通过重评或重跑修复，属于 benchmark noise 的一种。

---

## 待填充

- [ ] DOM 全量完成后：GLM 定量分析
- [ ] 失败模式分类统计
- [ ] 与 classifieds DOM 的跨站对比
