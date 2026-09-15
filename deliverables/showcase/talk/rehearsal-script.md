# Showcase 16 Sep 台本（v2）

_按 `presentation-playbook 3.md` 附录 C 重排（2026-09-13）。目标、证据与范围：[`../ROADMAP.md`](../ROADMAP.md) §0b 场次简报 与 §4；数字与口径：[`../SHOWCASE_PREP.md` §5](../SHOWCASE_PREP.md)，禁语 §6；操作：[`RUNBOOK.md`](RUNBOOK.md)。_

口播：英文。提示：中文。总时限：10 分钟（**含不含问答未确认** —— 先按含问答准备短版）。
**加粗的句子 = 精确措辞**，只锁核心主张、边界和请求（v3 §4）；其余照意思说，用你顺口的表达。页面按 id 引用，不按页码。

## 保底版本与裁剪

必须表达：看法改变行为和失败方式（`behaviour` / `failure`）· 事后选对有收益，但要对着重跑读（`hindsight`）· 学到的选择 0 of 8、事后最优也只有 1 of 8（`learned`）· 例子只在做对时才有（`why`）· 边界和三个请求（`not-yet`）。
先删：`question` 只念问题不念四个路标；`why` 并成 `learned` 末尾一句；`hindsight` 只留「每行都往右」和「但 hindsight 会夸大」两句。
再删或替换：`demo` 只讲 LOOK 与 READ，BOTH 一句带过（v3：demo 也可以裁）。
短版（约 6 分钟）：`opening` → `demo`（LOOK / READ）→ `question` → `behaviour` → `failure` → `learned`（带一句「例子只在做对时才有」）→ `not-yet` → `close`。

## 开场 —— 页面 `opening`

目的：交代为什么值得听 —— agent 看网页有三种方式、价钱不同，而且这是量出来的。

> When is expensive perception worth paying for?
> A web agent can *look* at a page — a screenshot. It can *read* it — the page as text. Or both. They cost different amounts.
> This is my MSc thesis, accepted at the REALM workshop at EMNLP this year.
> Let me show you one task first.

提示：说完按 → 到 `demo`。
转场：*"Let me show you one task first."* → `demo` *"Same task, three ways of seeing it…"*

## 段落 demo —— 页面 `demo` / 操作 RUNBOOK「台上的动线」II

目的：让观众亲眼看到同一任务三种看法、三种账单，以及 learned choice 选错。状态要说清：一次录制的回放，不是现场运行（v3 §5）。

> **Same task, three ways of seeing it, three different bills.**
> One recorded run per view — nothing on this screen is live.
> The task: find the listing whose photo was taken at sunset.
> LOOK gets the screenshot only. READ gets the page as text, no image. BOTH gets the screenshot with numbered marks.
> LOOK sees the sunset and clicks it. Two steps. Solved.
> READ is scrolling. The word "sunset" isn't on the page — it's in the picture.
> Nine steps, and it gives up on the wrong boat.
> BOTH finds it in three steps — at twice LOOK's bill.
> A learned choice, trained without this task, picked READ. The one view that failed.
> One task proves nothing on its own. So here is the real question.

提示：先说三栏是什么，再按 →；LOOK 那句后按一下；READ 边按边说到第 9 步；指 task 下面那行红字；「The one view that failed」后停一下，让大家看红框。demo 到底再按 → 自动翻到 `question`。
边界：三道题是示例，不是成功率。⛔ 不切 76 / 17，不进 live，不解释 CO₂e（被问：按 token 估算，是区间，不是实测）。
转场：*"So here is the real question."* → `question`

## 段落 question —— 页面 `question`

目的：提一个问题，后面四段各答一块，`close` 回答它。不提前给结论。

> **If you build a web agent, that's the question: should it look, read, or both — and can it learn to choose?**
> I'll take it in four steps: how the views behave, how they fail, what picking right would buy you, and whether it can be learned.

提示：念四个路标时从左到右指一下。
转场：→ `behaviour` *"First, behaviour."*

## 段落 behaviour —— 页面 `behaviour`

目的：看法一换，agent 做的事就换了。

> First, behaviour. We ran six views — LOOK, BOTH, READ, and three text-only variants — across eight website-and-model settings, about eight thousand nine hundred task attempts.
> With only the screenshot, the agent scrolls far more and types far less.
> Each dot is the median over the eight settings; the line is the range.
> LOOK spends about thirty percent of its steps scrolling. Every other view: about six. And it types about half as often.
> In every one of the eight settings, it is the view that scrolls most.

边界：⛔ 不说「每个设置都是四倍」（逐设置只有 1.25–7 倍）；⛔ 不解释为什么（没做机制）。
转场：→ `failure` *"They also fail differently."*

## 段落 failure —— 页面 `failure`

目的：不光做的事不同，输的方式也不同。

> They also fail differently.
> Take the tasks only one side solved, and ask how the other side failed — compared with how that same side fails everywhere else.
> When a screenshot view solved it and the text-only views didn't, the text views gave up early, looped back and forth, or never saw the target — each more than twice as often as usual.
> Flip it. Text solved it; LOOK and BOTH didn't. Nothing stands out. They fail there the way they fail everywhere — they just never get there.
> **This pattern comes from the six VisualWebArena settings.**

边界：⛔ 不拿两侧互比（文本侧四种看法，截图侧两种）；⛔ 只说「出现得多」，不说「因为」；⛔ 截图侧不叫 image-only（BOTH 也带文字）。
转场：→ `hindsight` *"Now picture something you actually do."*

## 段落 hindsight —— 页面 `hindsight`

目的：用台下每天真会遇到的场景，说清楚「选对看法」值多少 —— 三个大数字，好记就行（09-15 学长意见：演讲要抓人，别太严谨）。

> Now picture something you actually do. Your coding agent — Claude Code, say — fixes a bug in your web app, then opens a browser to check its own work.
> Is the new chart there? That answer is in the picture. Did the form save the right value? That one is in the page text.
> So what if it always picked the right view for each check? On our large model, across three websites:
> **Eleven to sixteen more tasks solved in every hundred.**
> **Seven to twenty-nine percent less carbon — estimated from tokens.**
> **And it's no slower: about the same on two sites, a third faster on the third.**
> That's the prize — if you pick perfectly.

提示：三个数字一个一个指，每说一个停半拍。
边界：⛔ 不说「快 34%」而不带「另外两个网站持平」；碳排放一定带 *estimated*；「if you pick perfectly」这半句别省 —— 下一页正是说今天做不到。数字出处 `talk/hindsight_efficiency.json`（B0 三个网站）。
转场：→ `learned` *"So can a model learn to make that pick?"*

## 段落 learned —— 页面 `learned`

目的：一个数字说完：今天学不会。

> So can a model learn to make that pick? We tried five different ways.
> **Zero out of eight.** In none of our eight settings does a learned picker beat simply always using the cheapest view.
> **Even perfect hindsight only manages it in one of eight** — so this isn't just a weak model.

提示：「Zero out of eight」后停一下，让大家看大数字。
边界：⛔ 不说 *routing doesn't work*；说 *not today*。
转场：→ `why` *"Here's why — call it the scaling law of routing."*

## 段落 why —— 页面 `why`

目的：一句好记的话讲清原因：没有成功，就没有例子。

> Here's why — call it the scaling law of routing: no wins, no examples.
> To learn which view was right, you need tasks the agent actually solved. These agents solve two to thirty-six percent.
> So the settings with the most to gain have the least to learn from. The ones short of examples need at least two to four times more tasks.

边界：「scaling law」是个好记的叫法，指「成功越多、例子越多」这个趋势（6 个设置），不是拟合出来的幂律；被追问就这么说。
转场：→ `not-yet`

## 段落 not-yet —— 页面 `not-yet`

目的：说清边界，然后提请求。

> **So I'm not claiming routing is unlearnable — just not yet, at these success rates.**
> Improve the agent first, collect reliable examples, then learn when to look.
> What it takes: a stronger agent, at least two to four times more tasks, and examples that survive a rerun.
> **Three asks. Come to the board and type a task — it runs in all three views while you watch; nothing there is scored.**
> **If you run web agents, tell me which view you use.**
> **And if you have an agent that solves clearly more of its tasks — lend it to us. That's where this should be tested.**

提示：说请求时看向房间里做 agent 的人；最后一句后停一下再翻页。
边界：⛔ 不说 *a stronger agent will make it learnable*（没测过）。
转场：→ `close`

## 收束 —— 页面 `close`

> **Look, read, or both? It depends on the task — and nothing we trained knows it yet.**
> Come and find me at the board. Thank you.

## 被问到时（口头答法只在这里维护；证据与完整问答见 `../SHOWCASE_PREP.md §4 / §5`）

最尖的四个放这里，其余全部在 `../SHOWCASE_PREP.md §4`（含「为什么六种看法」「可部署吗」「更强的模型」「重跑噪声这么大结果还算数吗」）。

**「0 of 8 是不是训练数据太少？」**（最可能的第一问）
> We tested exactly that. The learning curves are still rising, so there is signal. But more data pushes the learner toward perfect hindsight — and perfect hindsight itself reaches the win region in one of eight. More data can't cross a line hindsight doesn't cross. We priced it anyway: the failing settings need two to four times more tasks. That's a specification, not an impossibility.

**「always-cheapest 是每题选还是固定？」**
> Fixed. The one view that costs least on average in that setting, used for every task. Not a per-task pick.

**「事后选对的收益会不会只是重跑的噪声？」**
> Partly, at the margin we could test. In the one setting where every view was rerun, adding a second view bought about seven tasks in a hundred, and rerunning the same view bought four and a half to seven and a half. We only have one rerun, not five, so we don't claim the whole gain is noise — but we don't sell it as a result either.

**「+16 是真系统做到的吗？」**
> No — that's perfect hindsight, an upper bound, and the rerun band sits right next to it. What a learned choice actually reached is the plot: none in the win region.

**「那生产系统今天该怎么办？」**
> On these benchmarks, always using the cheapest view is hard to beat on both counts, and BOTH is the dearest. Pick one view for your task mix. A learned per-task choice isn't there yet — improve the agent first, collect reliable examples, then learn when to look.

**「只看截图为什么翻页那么多？」**
> We measured what each view does, not why — that's out of scope here. What I can say is that it holds in all eight settings.

答不上来：直说 + 立刻给「怎么才能知道」（"I don't know — the step records are public, and that's a one-line query; I'll check after."）。

## 切兜底时

- demo 页不响应 / 白屏 → **不调试**。按 ③ 切兜底页，*"I've got these captured."*，按截图顺序讲完第二幕，继续。再也不提。
- 投影只认主办方电脑 → U 盘里 `demo_portable.html?task=130&autoplay=0` 或兜底页；再不行 → 片子里那张 webm。
- **永远不说 "it worked this morning"。**


## 彩排记录

（尚未彩排。第一遍出声计时后在这里记：实测时长、卡壳的段落、画面等待的位置；改完删掉已解决项。）
