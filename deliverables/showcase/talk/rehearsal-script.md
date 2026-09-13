# 台本 —— Showcase 十分钟演讲 · 中文引导 · 英文台词

_一句约定：**中文 = 你做什么、点哪、注意什么；英文引用块 = 原话，照念**；其余任何东西不在台上念。
v1 2026-09-13：按 user 改的故事线重排 —— 先给观众带走什么，再 行为不同 → 失败不同 → 那就按任务选？→ 学不会 → 将来要什么；标题页与 demo 页沿用 v0。
仍是模型初稿。按 SOP §11.10：**你自己出声念一遍再重写，念顺的那版才是台本。**
权威数字只在 `../SHOWCASE_PREP.md §5`，禁语在 §6，全部问答在 §4；本文只放要念的和要做的。_

**已答应出去的**（SOP §2.1）：给 Zekun 的 DM 说过「三件事按序讲 + 三道任务的电脑 demo」；slot 定为 10 分钟；Zekun 09-13 要提前看片子。
v1 仍覆盖那三件事（有上界 → 学不到 → 为什么），前面加了「行为 / 失败不同」，没有多答应别的。

**时间分配**（英文台词 **824 词**，`check_talk.py` 实数：÷140 ≈ 5.9 分，÷120 ≈ 6.9 分，加三分之一的点击与停顿 ≈ 7.8–9.2 分，再加 demo 步进约 1 分；你重写后再跑一次 check 更新这里）

| 幕 | 秒 | 张 |
|---|---|---|
| I 开场 | 0:00–0:25 | 1 |
| II Demo | 0:25–2:15 | 2（demo iframe） |
| III 你带走什么 | 2:15–2:40 | 3 |
| IV 行为不同 | 2:40–3:25 | 4 |
| V 失败不同 | 3:25–4:20 | 5 |
| VI 那就按任务选？ | 4:20–5:20 | 6 |
| VII 学不会 | 5:20–6:25 | 7 |
| VIII 为什么还不行 | 6:25–7:10 | 8 |
| IX 将来要什么 + ask | 7:10–8:05 | 9 |
| X 收尾 | 8:05–8:25 | 10 |

**裁剪顺序**（10 分钟若含问答、或主持人说只剩 7 分钟）：① 第八幕只念锁死句 + 「例子只在做对时才有」一句 → ② 第六幕删重跑那两句 → ③ 第三幕只念三行 → ④ 第一幕删自陈两句。
**永不砍：II 的 demo；VII 的 0 of 8 与 1 of 8 必须连着说。**

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
| 1 | 标题（海报题目）+ 姓名 | 第一幕 |
| 2 | demo iframe（task 130） | 第二幕 —— 只按 →，**不按 space** |
| 3 | 三条收获（深色框） | 第三幕 |
| 4 | 行为：翻页 / 打字占比（海报面板 3） | 第四幕 |
| 5 | 失败：两侧怎么输（海报面板 4） | 第五幕 |
| 6 | 矩阵 + 韦恩：事后选有收益，重跑当尺子 | 第六幕 |
| 7 | 赢区图 | 第七幕 —— 「only one of eight」后停 3 秒 |
| 8 | label supply | 第八幕 |
| 9 | 要满足什么 · 三个请求 | 第九幕 |
| 10 | 标题再出现 + QR | 第十幕 |
| 11 | 备用：39 s 录屏 | 只在 demo 挂了才翻（Mac 上 Fn+→） |

**检查清单**（详见 RUNBOOK）：缩放 100%，不要放大 · 书签栏收起 · ① 带 `#1` 刷新一次不跳页 · 免打扰 · 电源 · 计时器 9:30。

---

# 第一幕 · 开场（0:25）

【第 1 张：标题 + 姓名 + 一行出处（按主办方模板；收尾页也有署名）】三句话嘴里说。你的目的只有一个：让房间知道**这是量出来的东西，不是提案**。

> When is expensive perception worth paying for?
> A web agent can *look* at a page — a screenshot. It can *read* it — the page as text. Or both. They cost different amounts.
> This is my MSc thesis, accepted at the REALM workshop at EMNLP this year.
> I set out to build something that chooses between them. Let me show you why that's harder than it sounds.

⛔ 不在这里解释六种看法、八个设置 —— 那是第四幕的事。说完直接按 → 到 demo。

# 第二幕 · Demo（1:50）

【第 2 张：demo iframe，停在 task 130 第 0 步。三栏栏头自带 screenshot only / text tree only / marked screenshot】
先说三栏是什么，**再**开始按 →。每按一下说一句。目的：让观众**亲眼看到**同一任务三种看法三种账单，然后看到 learned choice 选了唯一失败的那栏。

> **Same task, three ways of seeing it, three different bills.**
> One recorded run per view — nothing on this screen is live.
> The task: find the listing whose photo was taken at sunset.
> LOOK gets the screenshot only. READ gets the page as text, no image. BOTH gets the screenshot with numbered marks.

【→ 一下】

> LOOK sees the sunset and clicks it. Two steps. Solved.

【→ 连按，READ 一路 scroll；边按边说，到 READ 第 9 步为止】

> READ is scrolling. The word "sunset" isn't on the page — it's in the picture.
> Nine steps, and it gives up on the wrong boat.
> BOTH finds it in three steps — at twice LOOK's bill.

【指屏幕上 task 下面那一行红字】

> A learned choice, trained without this task, picked READ. The one view that failed.

**停 2 秒。**

> The poster has the mirror case: a price edit that READ finishes and LOOK loops on.
> So here is what I'd like you to leave with.

⛔ 不切到 76 或 17；⛔ 不进 live 页；⛔ 不解释 CO₂e 那一行（被问再说：估算、区间、非实测）。demo 到底再按 → 自动翻到第 3 张。

# 第三幕 · 你带走什么（0:25）

【第 3 张：深色框，三条编号】语速放慢，一条一停。目的：先把收获交出去，后面每张都是在兑现其中一条。

> If you build web agents, three things.
> How your agent sees the page changes what it does — and how it fails.
> Choosing the view per task would pay — in hindsight.
> And no learned choice gets there yet. A stronger agent is the test.
> Let me take them in order.

# 第四幕 · 行为不同（0:45）

【第 4 张：翻页 / 打字占比，海报面板 3】目的：看法一换，agent 做的事就换了。

> **With only a screenshot, the agent scrolls far more and types far less.**
> Six views, four of them with no image at all, across eight website-and-model settings — about eight thousand nine hundred task attempts.
> Each dot is the median over the eight settings; the line is the range.
> With only the screenshot, it spends about thirty percent of its steps scrolling. Every other view: about six.
> And it types about half as often. In every one of the eight settings, it's the view that scrolls most.

⛔ 不说「每个设置都是四倍」（逐设置只有 1.25–7 倍）；⛔ 不解释为什么（没做机制，被问见「被问到时」）。

# 第五幕 · 失败不同（0:55）

【第 5 张：两侧怎么输，海报面板 4】目的：不光做的事不同，输的方式也不同。

> **Text-only fails in ways you can name; image-only just never gets there.**
> Take the tasks only one side solved, and ask how the other side failed — compared with how that same side fails everywhere else.
> When the screenshot solved it and the text-only views didn't, they gave up early, looped back and forth, or never saw the target — each more than twice as often as usual.
> Flip it. Text solved it, the image-only views didn't. Nothing stands out: they fail there the way they fail everywhere. They just don't arrive.
> This pattern comes from the six VisualWebArena settings.

⛔ 不拿两侧互相比（文本侧是四种看法、图像侧两种）；⛔ 只说「出现得多」，不说「因为」；⛔ 不主讲 2.3（只有 13 次）。

# 第六幕 · 那就按任务选？（1:00）

【第 6 张：面板 1 矩阵 + 面板 2 韦恩】目的：事后看，选对看法有收益 —— 但要拿重跑当尺子读。

> So — choose the view per task?
> **Hindsight says choosing pays; a rerun says how much of that is noise.**
> No view wins everywhere, and the views solve different tasks.
> Pick the right view per task after the fact — perfect hindsight — and you solve three and a half to sixteen more tasks in a hundred than the best single view.
> Then the ruler. Rerun the same view on the same tasks, and ten to fourteen percent of outcomes flip by themselves. A plain rerun buys two to eight points.
> The upside is real — but smaller than it looks.

⛔ 「16」永远带 *in a hundred* 或 *points*，不说 percent；⛔ 不说 13.7、不说 12–14。

# 第七幕 · 学不会（1:05）

【第 7 张：面板 5 赢区图】目的：0 of 8 之后**马上**给 1 of 8 —— 问题主要不在学习器。

> **Learned routers buy success only by spending more.**
> Can a model learn that choice in advance? We tried five ways, with nested cross-validation.
> The bar is always-cheapest: the one view that costs least on average in that setting, used for every task.
> The shaded corner is a win — cheaper, and no worse. Zero of eight learned choices land there. They gain success only by paying for it.
> And even perfect hindsight reaches that corner in only one of eight.

**停 3 秒。不翻页。**

> So this is not mainly a weak classifier. Something upstream is missing.

⛔ 不说「the problem isn't the learner」（说过头）；说 *not mainly*。

# 第八幕 · 为什么还不行（0:45）

【第 8 张：面板 6 label supply】目的：例子只在做对时才存在。

> **More routing upside, less usable training signal.**
> A training example for "which view" only exists when a task gets solved — here, two to thirty-six percent of the time.
> Fifteen to ninety-seven usable examples per setting. Enough to train a classifier in two of the six.
> The weaker the agent, the more there is to gain from choosing — and the fewer examples to learn from.

⛔ y 轴是计数不是百分比，别说 percent of labels。

# 第九幕 · 将来要什么 + ask（0:55）

【第 9 张：左「What it takes」三行 · 右「Three asks」三行】说 ask 时看向房间里做 agent 的那几个人。

> **Improve the agent first, collect reliable examples, then learn when to look.**
> So I'm not claiming routing is unlearnable. Just not yet, at these success rates.
> What it takes: a stronger agent. The failing settings would need at least two to four times more tasks. And examples that survive a rerun.
> Three asks. Come to the board and type a task — it runs in all three views while you watch; nothing there is scored.
> If you run web agents, tell me which view you use.
> And if you have an agent that solves clearly more of its tasks — lend it to us. That's where this should be tested.

**停 2 秒**，再翻到收尾。

⛔ 不说「a stronger agent will make it learnable」（没测过）；说 *that's where this should be tested*。

# 第十幕 · 收尾（0:20）

【第 10 张：标题再出现 + QR】

> When is expensive perception worth paying for?
> On these benchmarks: when hindsight says so — and nothing we trained knows it yet.
> Come and find me at the board. Thank you.

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

**「只看截图为什么翻页那么多？」**
> We measured what each view does, not why — that's out of scope here. What I can say is that it holds in all eight settings.

答不上来：直说 + 立刻给「怎么才能知道」（"I don't know — the step records are public, and that's a one-line query; I'll check after."）。

# 演讲纪律

- **只逐字背六句**：I *When is expensive perception worth paying for?* · II *Same task, three ways of seeing it, three different bills.* · IV *With only a screenshot, the agent scrolls far more and types far less.* · V *Text-only fails in ways you can name; image-only just never gets there.* · VII *Learned routers buy success only by spending more.* · IX *Improve the agent first, collect reliable examples, then learn when to look.* 其余几张的标题照意思说，不必逐字。
- **沉默三处**：II「The one view that failed」后 2 秒 · VII「only one of eight」后 3 秒不翻页 · IX「That's where this should be tested」后 2 秒。
- **被打断**：一句话答完 + 固定返回句 *"— so, same task, three views, three bills."*；下游问题推迟但像掌控：*"That's exactly where I'm going next — can I show you one more slide first?"*
- **数字纪律**：每个数带 baseline 短语（*vs the best single view* / *vs always-cheapest* / *in a hundred*）；失败那张只说「出现得多」，不说「因为」，不拿两侧互相比。禁语表 `../SHOWCASE_PREP.md §6`。

# 出事了怎么办

- demo 页不响应 / 白屏 → **不调试**。按 ③ 切兜底页，*"I've got these captured."*，按截图顺序讲完第二幕，继续。再也不提。
- 投影只认主办方电脑 → U 盘里 `demo_portable.html?task=130&autoplay=0` 或兜底页；再不行 → 片子里那张 webm。
- **永远不说 "it worked this morning"。**

# 最后一次彩排怎么练：只练五条转场链

1. II 末 *"So here is what I'd like you to leave with."* → III 首 *"If you build web agents, three things."*
2. III 末 *"Let me take them in order."* → IV 首 *"With only a screenshot, the agent scrolls far more and types far less."*
3. V 末 *"This pattern comes from the six VisualWebArena settings."* → VI 首 *"So — choose the view per task?"*
4. VII 末 *"Something upstream is missing."* → VIII 首 *"More routing upside, less usable training signal."*
5. VIII 末 *"…and the fewer examples to learn from."* → IX 首 *"Improve the agent first, collect reliable examples, then learn when to look."*
