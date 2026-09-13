# 台本 —— Showcase 十分钟演讲 · 中文引导 · 英文台词

_一句约定：**中文 = 你做什么、点哪、注意什么；英文引用块 = 原话，照念**；其余任何东西不在台上念。
v0 骨架 2026-09-11（模型初稿）。按 SOP §11.10：**你自己出声念一遍再重写，念顺的那版才是台本。**
权威数字只在 `../SHOWCASE_PREP.md §5`，禁语在 §6，全部问答在 §4；本文只放要念的和要做的。_

**已答应出去的**（SOP §2.1）：给 Zekun 的 DM 说过「三件事按序讲 + 三道任务的电脑 demo」；slot 后来定为 10 分钟。
本台本 = 那三件事（有上界 → 学不到 → 为什么）+ 一道任务的 demo，没有多答应别的。

**时间分配**（英文台词 **735 词**，`check_talk.py` 实数：÷140 ≈ 5.2 分，÷120 ≈ 6.1 分，加三分之一的点击与停顿 ≈ 7–8 分，再加 demo 步进约 1 分 ⇒ 8–9 分；你重写后再跑一次 check 更新这里）

| 幕 | 秒 | 张 |
|---|---|---|
| I 开场定位 | 0:00–0:35 | 1 |
| II Demo | 0:35–2:45 | 2（demo iframe） |
| III 尺子 | 2:45–4:15 | 3 |
| IV 没那么快 | 4:15–5:45 | 4 |
| V 为什么 | 5:45–7:00 | 5 |
| VI 不主张 + ask | 7:00–8:15 | 6 |
| VII 收尾 | 8:15–8:40 | 7 |

**裁剪顺序**（主持人说只剩 7 分钟时）：① 砍 VI 的「三个不主张」只留三个 ask → ② 砍 III 最后那句面板 3/4 → ③ 砍 I 的自陈两句。
**永不砍：II 的 demo；IV → V 的转场。**

---

## 0. 开讲前 5 分钟的固定动作

→ 操作细节全在 `RUNBOOK.md`。这里只放三张表（SOP 附录 C 的形状）。

**标签页，就这三个：**

| 标签 | 内容 | 状态 |
| --- | --- | --- |
| **① 片子** | `talk/index.html#1` | 停在第 1 张，全屏 |
| **② demo 演讲版** | `demo_portable.html?task=130&autoplay=0` | 停在 task 130 · step 1，Play 按钮显示 *Play* |
| **③ 兜底** | `talk/fallback.html` | 出事才切 |

**幻灯片对照：**

| 张 | 内容 | 在哪一幕用 |
| --- | --- | --- |
| 1 | 标题（海报题目） | 第一幕 |
| 2 | demo iframe（task 130） | 第二幕 —— 只按 →，**不按 space** |
| 3 | 面板 1 矩阵 + 面板 2 韦恩 | 第三幕 |
| 4 | 面板 5 赢区图 | 第四幕 —— 锁死句后停 3 秒 |
| 5 | 面板 6 label supply | 第五幕 |
| 6 | 不主张三行 · ask 三行 | 第六幕 |
| 7 | 标题再出现 + QR | 第七幕 |
| 8 | 备用：39 s 录屏 | 只在 demo 挂了才翻（End 键） |

**检查清单**（详见 RUNBOOK）：缩放 100%，不要放大 · 书签栏收起 · ① 带 `#1` 刷新一次不跳页 · 免打扰 · 电源 · 计时器 9:30。

---

# 第一幕 · 开场定位（0:35）

【第 1 张：标题 + 姓名 + 一行出处（按主办方模板；收尾页也有署名）】三句话嘴里说、不上屏。你的目的只有一个：让房间知道**这是量出来的东西，不是提案**。

> When is expensive perception worth paying for?
> A web agent can *look* at a page — a screenshot. It can *read* it — the page as text. Or both. They cost different amounts.
> This is my MSc thesis; the paper was accepted at the REALM workshop at EMNLP this year.
> I started out expecting to build the thing that chooses. What I have instead is a measurement of why, today, it doesn't learn.

⛔ 不在这里解释六种看法、八个设置 —— 那是第三幕的事。说完最后一句直接按 → 到 demo。

# 第二幕 · Demo（2:10）

【第 2 张：demo iframe，停在 task 130 第 0 步。三栏栏头自带 screenshot only / text tree only / marked screenshot】
先说三栏是什么，**再**开始按 →。每按一下说一句。你的目的只有一个：让观众**亲眼看到**同一任务三种看法三种账单，
然后看到 learned choice 选了唯一失败的那栏。

> **Same task, three ways of seeing it, three different bills.**
> One recorded run per view — nothing on this screen is live.
> The task: find the listing whose photo was taken at sunset.
> LOOK gets the screenshot only. READ gets the page as text, no image. BOTH gets the screenshot with numbered marks on it.

【→ 一下】

> LOOK sees the sunset and clicks it. Two steps. Solved.

【→ 连按，READ 一路 scroll；边按边说，到 READ 第 9 步为止】

> READ is scrolling. The text never says "sunset" — that word isn't on the page, it's in the picture.
> Nine steps, and it gives up on the wrong boat.
> BOTH finds it in three steps — at twice LOOK's bill.

【指屏幕上 task 下面那一行红字】

> Now the line under the task. A learned choice, trained without this task, picked READ. The one view that failed.

**停 2 秒。**

> The poster shows the mirror case: a price edit, where READ finishes in twelve steps and LOOK loops for twenty-six.
> So the view matters, per task. Two questions: how much is there to gain — and can you learn to choose?

⛔ 不切到 76 或 17；⛔ 不进 live 页；⛔ 不解释 CO₂e 那一行（被问再说：估算、区间、非实测）。

# 第三幕 · 尺子（1:30）

【第 3 张：面板 1 矩阵 + 面板 2 韦恩；kicker 写规模】你的目的只有一个：上界是真的，但要拿重跑当尺子读。

> **Hindsight says choosing pays; a rerun says how much of that is noise.**
> Six views, four of them image-free. Eight website-and-model settings across two benchmarks — about eight thousand nine hundred task attempts.
> No view wins everywhere. The best single view solves between two and thirty-six percent of tasks, depending on the setting.
> And the views solve *different* tasks — the sets overlap, but they don't coincide.
> So pick the right view per task after the fact — perfect hindsight — and you solve three and a half to sixteen more tasks in a hundred than the best single view. Same direction in all eight settings.
> Then the ruler. Rerun the *same* view on the same tasks, and ten to fourteen percent of outcomes flip by themselves. A plain rerun buys roughly two to eight points.
> The ceiling is real — but smaller than it looks.
> They also behave and fail differently; that's panels three and four on the poster, and I'll skip them here.

⛔ 「16」永远带 *in a hundred* 或 *points*，不说 percent；⛔ 不说 13.7、不说 12–14。

# 第四幕 · 没那么快（1:30）

【第 4 张：面板 5 赢区图】你的目的只有一个：0 of 8 之后**马上**给 1 of 8 —— 问题不在学习器。

> **Learned routers buy success only by spending more.**
> Five ways of learning the choice. Nested cross-validation, ten thousand permutations.
> The bar is always-cheapest: the single view that costs least on average in that setting, used for every task.
> The shaded corner is a win — cheaper, and no worse. Zero of eight learned choices land there. They do gain success — by paying for it.
> And here is the part that matters: even perfect hindsight reaches that corner in only one of eight.

**停 3 秒。不翻页。**

> So this is not primarily a weak classifier. Something upstream is missing.

⛔ 不说「the problem isn't the learner」（说过头）；说 *not primarily*。

# 第五幕 · 为什么（1:15）

【第 5 张：面板 6 label supply】你的目的只有一个：例子只在成功时才存在。

> **More routing upside, less usable training signal.**
> A training example for "which view" exists only when a task gets solved. In these settings, that is two to thirty-six percent of the time.
> Fifteen to ninety-seven usable examples per setting. Enough to train a classifier in two of the six.
> Each dot is a setting. The lower the agent's success, the more there is to gain from choosing — and the fewer examples there are to learn from.
> We priced it: the failing settings would need at least two to four times more tasks.

⛔ y 轴是计数不是百分比，别说 percent of labels。

# 第六幕 · 不主张什么 + ask（1:15）

【第 6 张：左三行 we do not claim · 右三行 three asks】说「不主张」时语速慢，只说一次；说 ask 时看向房间里做 agent 的那几个人。

> **Improve the agent first, collect reliable examples, then learn when to look.**
> Three things I do *not* claim. Not that routing is unlearnable — only that in this success regime, the supervision isn't there.
> Not that always-cheapest is a floor per task — it is cheapest on average; a few points sit left of it.
> And nothing in the live demo at the board is scored.
> Three things I'd ask. Come to the board and type a task — it runs in all three views while you watch.
> If you run web agents, tell me which view you use, and why.
> And if you have an agent that solves clearly more of its tasks than these did — lend it to us. That is where this result should be tested.

**停 2 秒**，再翻到收尾。

# 第七幕 · 收尾（0:25）

【第 7 张：第 1 张再出现 + QR】

> When is expensive perception worth paying for?
> On these benchmarks: when hindsight says so — and nothing we can train knows that in advance.
> The board is by the window. Thank you.

---

# 被问到时

最尖的四个放这里，其余全部在 `../SHOWCASE_PREP.md §4`（含「为什么六种看法」「可部署吗」「更强的模型」「重跑噪声这么大结果还算数吗」）。

**「0 of 8 是不是训练数据太少？」**（最可能的第一问）
> We tested exactly that. The learning curves are still rising, so there is signal. But more data pushes the learner toward perfect hindsight — and perfect hindsight itself reaches the win region in one of eight. More data can't cross a line hindsight doesn't cross. We priced it anyway: the failing settings need two to four times more tasks. That's a specification, not an impossibility.

**「always-cheapest 是每题选还是固定？」**
> Fixed. The one view that costs least on average in that setting, used for every task. Not a per-task pick.

**「+16 是真系统做到的吗？」**
> No — that's perfect hindsight, an upper bound, and the rerun band sits right next to it. What a learned choice actually reached is the plot: none in the win region.

**「那生产系统今天该怎么办？」**
> On these benchmarks, always using the cheapest view is hard to beat on both counts, and BOTH is the dearest. Pick one view for your task mix. A learned per-task choice isn't there yet — improve the agent first, collect reliable examples, then learn when to look.

答不上来：直说 + 立刻给「怎么才能知道」（"I don't know — the step records are public, and that's a one-line query; I'll check after."）。

# 演讲纪律

- **只逐字背六句**（= 六张片子的标题）：I *When is expensive perception worth paying for?* · II *Same task, three ways of seeing it, three different bills.* · III *Hindsight says choosing pays; a rerun says how much of that is noise.* · IV *Learned routers buy success only by spending more.* · V *More routing upside, less usable training signal.* · VI *Improve the agent first, collect reliable examples, then learn when to look.*
- **沉默三处**：II「the one view that failed」后 2 秒 · IV「only one of eight」后 3 秒不翻页 · VI「then learn when to look」后 2 秒。
- **被打断**：一句话答完 + 固定返回句 *"— so, same task, three views, three bills."*；下游问题推迟但像掌控：*"That's exactly where I'm going next — can I show you one more panel first?"*
- **数字纪律**：每个数带 baseline 短语（*vs the best single view* / *vs always-cheapest* / *in a hundred*）。禁语表 `../SHOWCASE_PREP.md §6`。

# 出事了怎么办

- demo 页不响应 / 白屏 → **不调试**。按 ③ 切兜底页，*"I've got these captured."*，按截图顺序讲完第二幕，继续。再也不提。
- 投影只认主办方电脑 → U 盘里 `demo_portable.html?task=130&autoplay=0` 或兜底页；再不行 → 片子里那张 webm。
- **永远不说 "it worked this morning"。**

# 最后一次彩排怎么练：只练四条转场链

1. II 末 *"…and can you learn to choose?"* → III 首 *"Hindsight says choosing pays; a rerun says how much of that is noise."*
2. III 末 *"…and I'll skip them here."* → IV 首 *"Learned routers buy success only by spending more."*
3. IV 末 *"Something upstream is missing."* → V 首 *"More routing upside, less usable training signal."*
4. V 末 *"…at least two to four times more tasks."* → VI 首 *"Improve the agent first, collect reliable examples, then learn when to look."*
