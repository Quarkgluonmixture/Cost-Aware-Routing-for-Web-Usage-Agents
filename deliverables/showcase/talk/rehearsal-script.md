# Showcase 16 Sep 台本（v2）

_按 `presentation-playbook 3.md` 附录 C 重排（2026-09-13）。目标、证据与范围：[`../ROADMAP.md`](../ROADMAP.md) §0b 场次简报 与 §4；数字与口径：[`../SHOWCASE_PREP.md` §5](../SHOWCASE_PREP.md)，禁语 §6；操作：[`RUNBOOK.md`](RUNBOOK.md)。_

口播：英文。提示：中文。总时限：10 分钟（**含不含问答未确认** —— 先按含问答准备短版）。
**加粗的句子 = 精确措辞**，只锁核心主张、边界和请求（v3 §4）；其余照意思说，用你顺口的表达。页面按 id 引用，不按页码。

## 保底版本与裁剪

必须表达：看法改变行为和失败方式（`behaviour` / `failure`）· 选对有收益，但那是上限（`prize`）· 学到的选择 0 of 8、事后最优也只有 1 of 8（`learned`）· 例子只在做对时才有（`why`）· 边界和三个请求（`not-yet`）。
先删：`question` 只念问题不念三个路标；`why` 并成 `learned` 末尾一句；`prize` 只念粗体第一句和「an upper bound — not Claude or GPT numbers」；`agents` 三块各一句。
再删或替换：`demo` 只讲 LOOK 与 READ，BOTH 一句带过（v3：demo 也可以裁）。
短版（约 6 分钟）：`opening` → `agents`（三块各一句）→ `demo`（LOOK / READ）→ `prize`（只念粗体第一句）→ `question` → `behaviour` → `failure` → `learned`（带一句「例子只在做对时才有」）→ `not-yet` → `close`。

## 开场 —— 页面 `opening`

目的：一句话把赌注亮出来 —— agent 看网页的方式不止一种、价钱不同，选对了同一个 agent 就能多做对、少花钱。数字不在这页，留到 demo 之后的 `prize`。

> When is expensive perception worth paying for?
> A web agent can see a web page in more than one way, and the ways cost different amounts. Pick the right way for each task, and the same agent does more, for less — I'll show you how much.
> This is my MSc thesis, accepted at the REALM workshop at EMNLP this year.
> First: what does a web agent actually see?

提示：说完最后一句按 → 到 `agents`。
转场：*"First: what does a web agent actually see?"* → `agents`

## 段落 agents —— 页面 `agents`

目的：用台下自己用的工具认出「web agent」，再让三种看法各留一个印象 —— 这一页只要让人记住「有三种、颜色各一」，细节留给 demo。三块各一个大 logo（Codex · Claude Code · browser-use）+ 同一张真实页面裁到注册表单的抓取；三块依次弹出。

> You've all used these. When Claude Code or Codex opens a browser and works a page for you — fills a form, checks a result — that's a web agent.
> Here's one real page, Wikipedia's sign-up form, captured this week. Three ways to see it.
> LOOK: a screenshot. Codex's computer use can work from this.
> READ: the page as text. This is what Claude Code gets through Playwright — this one page is a hundred and sixty-one lines.
> BOTH: the screenshot with every clickable thing boxed and numbered. browser-use does this by default; so does our agent.
> Each tool can use the other views too. Keep the three colours in mind — you're about to see them run.

提示：三块弹出时从左到右各指一下；READ 指高亮的 textbox / button 行；BOTH 指表单上的编号框。
边界：LOOK / READ 两块是用 Playwright MCP 对真实网页抓的（`talk/real_capture.py`，同一会话），BOTH 是 browser-use 0.13 自己对同一页的高亮截图（`talk/browser_use_capture.py`，它默认 `highlight_elements=True`，虚线框和编号是它画的）；三块都裁到表单区。都不是某个工具做任务的记录；logo 只标出是哪个工具，不说「Claude 做了这个」。Codex 的 computer use 实际是截图加无障碍文本混用（09-09 调研），OpenAI 文档原话是「uses screenshots and other tool results」，所以台上说「can work from this」和「each tool can use the other views」，不说「Codex 只看截图」。161 行是这一次抓取的实测，页面会变。demo 里 BOTH 泳道的框是我们自己的青色框，和 browser-use 的样式不同，口径是「同一类做法」。
转场：*"you're about to see them run."* → `demo`

## 段落 demo —— 页面 `demo` / 操作 RUNBOOK「台上的动线」II

目的：让观众亲眼看到同一任务三种看法、三种账单，以及 learned choice 选错。状态要说清：一次录制的回放，不是现场运行（v3 §5）。演讲版的 demo 不显示自己的标题和 task 标签页，task 句放大，关键词 *sunset* 高亮。

> **Same task, three ways of seeing it, three different bills.**
> One recorded run per view — nothing on this screen is live.
> The task is at the top: find the listing whose photo was taken at *sunset*. That word is the whole task.
> Same three colours: LOOK gets the screenshot only. READ gets the page as text, no image. BOTH gets the screenshot with numbered marks.
> LOOK sees the sunset and clicks it. Two steps. Solved.
> READ is scrolling. The word "sunset" isn't on the page — it's in the picture.
> Nine steps, and it gives up on the wrong boat.
> BOTH finds it in three steps — at twice LOOK's bill.
> A learned choice, trained without this task, picked READ. The one view that failed.
> One task proves nothing on its own. So what is choosing right worth across all of them?

提示：先指顶上高亮的 *sunset*，再说三栏是什么，再按 →；LOOK 那句后按一下；READ 边按边说到第 9 步；指 task 下面那行红字；「The one view that failed」后停一下，让大家看红框。demo 到底再按 → 自动翻到 `prize`。
边界：三道题是示例，不是成功率。⛔ 不切 76 / 17，不进 live，不解释 CO₂e（被问：按 token 估算，是区间，不是实测）。
转场：*"So what is choosing right worth across all of them?"* → `prize`

## 段落 prize —— 页面 `prize`

目的：把 demo 那一道题放大到全部题：同一个网站、同一个大模型，224 道题每道都选对看法值多少。两根条进页自动长出来，右边一个 −20%。观众刚看完 demo，已经知道「看法」和「选择」是什么，数字这时才有意义。

> That was one task. Now all two hundred and twenty-four on that site, same large open model.
> The best single view solves twenty-seven in a hundred. **Choose the right view for every task, and the same agent solves forty-three — sixteen more in every hundred — for twenty percent less on the token bill.**
> **That's an upper bound: a choice that already knows the answer for every task. On the other two sites it's eleven and sixteen more, fourteen and twenty-seven percent off. Not Claude or GPT numbers.**
> So the prize is real. The question is whether anything can learn to claim it.

提示：两根条长完再念数字；念到 forty-three 指第二根条，念到 twenty percent 指右边的 −20%。
边界：上限 = 事先知道每道题哪种看法能做对（后面 `learned` 页叫它 perfect router，这里先不用这个词）。27 / 43 / −20% 是大模型在 classifieds 一个设置的数（`talk/hindsight_efficiency.json`），另两个网站在页脚。CO₂e 不上片子（被问：−7 到 −29%，按 token 估算）；用时被问：一个网站快三分之一、两个持平。
转场：*"…whether anything can learn to claim it."* → `question`

## 段落 question —— 页面 `question`

目的：提一个问题，后面四段各答一块，`close` 回答它。不提前给结论。

> **If you build a web agent, that's the question: should it look, read, or both — and can it learn to choose?**
> I'll take it in three steps: how the views behave, how they fail, and whether a router can learn to pick.

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
转场：→ `learned` *"So can a router learn to make that pick?"*

## 段落 learned —— 页面 `learned`

目的：让大家看到绿色区域里没有学到的选择；0 of 8 是读图后的总结。

> So can a router — like the learned choice in the demo — learn to make that pick? We tried five different ways.
> The star is always using the cheapest view. Left is cheaper; up solves more. We want the green corner. None of the orange dots gets there — every one of them costs more than the star.
> **Zero out of eight.** In none of our eight settings does a learned router beat simply always using the cheapest view.
> **Even a perfect router only manages it in one of eight** — so this isn't just a weak model.

提示：先指星号，再指左上绿色角，再指右上角那行字和橙点（全在星号右边）；「Zero out of eight」后停一下。紫色方块是 perfect router（事后上限），绿色角里有一个，不能说所有选择都没进去。图上只放「没见过的题上测」的结果；「在训练题上测」的乐观结果（旧图浅橙三角）只留到问答。
边界：⛔ 不说 *routing doesn't work*；说 *not today*。
转场：→ `why` *"Here's why — call it the scaling law of routing."*

## 段落 why —— 页面 `why`

目的：一句好记的话讲清原因：没有成功，就没有例子。

> Here's one bottleneck — call it the scaling law of routing: no wins, no examples.
> Each dot is one real setting. Across: how many tasks the agent solves with some view. Up: examples of the second most common right view — to learn a choice, you need at least two different right answers, each seen often enough.
> Above the line there are enough examples to train a choice. Today only two of the six get there.
> **Stronger agents solve more tasks, and every solved task is an example — so they move toward the line.**
> Crossing it doesn't mean the choice wins; it means we can finally test it. At today's success rates, the settings below the line would need at least two to four times more tasks.

提示：先指横轴（用某种看法能做对的题）、纵轴（第二常见的正确看法有几条例子）；再指分界线，线上两个实心点、线下四个空心点；最后顺着紫色箭头往右上指。reddit 上的大模型也在线下：网站难，同样缺例子。
边界：说 one bottleneck，不把标签稀缺说成所有失败的唯一原因。箭头只是方向，不是拟合趋势，不说「做对多少题就能越线」。越过线 = 能训练、能检验，不等于赢过「永远用最便宜的看法」（事后最优也只有 1 of 8）。「scaling law」是「成功越多、例子越多」的叫法，不是拟合幂律。分界线 = 5 折里两类各至少 10 条训练例子（12.5 条，`router_undersampling_control.md` §D）；2–4× 假设正确看法的比例不变，是下界。
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
> We tested exactly that. The learning curves are still rising, so there is signal. But more data pushes the learner toward a perfect router — and a perfect router itself reaches the win region in one of eight. More data can't cross a line a perfect router doesn't cross. We priced it anyway: the failing settings need two to four times more tasks. That's a specification, not an impossibility.

**「always-cheapest 是每题选还是固定？」**
> Fixed. The one view that costs least on average in that setting, used for every task. Not a per-task pick.

**「事后选对的收益会不会只是重跑的噪声？」**
> Partly, at the margin we could test. In the one setting where every view was rerun, adding a second view bought about seven tasks in a hundred, and rerunning the same view bought four and a half to seven and a half. We only have one rerun, not five, so we don't claim the whole gain is noise — but we don't sell it as a result either.

**「+16 是真系统做到的吗？」**
> No — that's a perfect router, an upper bound that knows every outcome, and the rerun band sits right next to it. What a learned router actually reached is the plot: none in the win region.

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
