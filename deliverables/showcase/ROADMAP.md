---
type: showcase-planning
status: live
created: 2026-09-11
updated: 2026-09-11
event: 2026-09-16
---

# Showcase 09-16 路线图 —— 海报、demo、十分钟演讲怎么闭环

> 这是**向前看的 live 文件**：每个 phase 做完就把判据那一栏勾掉。过去发生了什么写笔记 §507 起，
> 别写在这里。当天要念的英文原稿仍在 `SHOWCASE_PREP.md`（§2 走读 · §4 问答 · §5 数字表 · §6 禁语）。

## 0. 先说结论

今天是周五（09-11），周三（09-16）开展，中间五天。海报已经印好，不再动。要闭环的是两件事：
**demo 收尾**（三处已定的修改加一个演讲模式）和**十分钟演讲**（从零到能排练）。

另外查出一个必须修的错位：`SHOWCASE_PREP.md` 的板前走读（§2）、讲稿（§3）和数字表（§5）
都是照 **v8 海报**写的，而印出来的是 **v9.10**。v8 的大数字条、「THE CATCH」、Fig 2 / Fig 3
在 v9 上都不存在，取而代之的是**六个面板**（下表）。站在板前照 §2 念，观众在纸上找不到你说的东西；
§5 那句「这里每个数字都在海报上」也已经不成立（v9 把大数字条去掉了）。所以走读和讲稿要按六面板
顺序重写，这是 Phase 2 里最大的一项。

印出来的海报长这样（`v9_preview.png`）：

| 位置 | 内容 |
|---|---|
| 标题条 | *When Is Expensive Perception Worth Paying For?* — 副标题 *Testing when richer web-agent representations help — and whether their value can be predicted cheaply* |
| 顶部通栏 | 系统图三段：Agent ↔ Web Page · Grounding representations（DOM / SoM / Vision）· Routing and outcomes |
| 中部通栏 | THE SAME TASK, SEEN TWO WAYS —— task 76：READ 12 步 · 8 页 · $0.09 solved；LOOK 26 步 · 9 页 · $0.10 gave up |
| 面板 1 | NO VIEW WINS EVERYWHERE（6 view × 8 setting 成功率矩阵） |
| 面板 2 | THEY SOLVE DIFFERENT TASKS（韦恩图，*the sets overlap — but do not coincide*） |
| 面板 3 | THEY BEHAVE DIFFERENTLY（*Vision scrolls ~4× more*） |
| 面板 4 | AND THEY FAIL DIFFERENTLY（text-only 偏 early give-up 2.3×；image-only 偏 stalled progress） |
| 面板 5 | SO BUILD A ROUTER? NOT SO FAST.（赢区图，*learned routers buy success only by spending more; only the hindsight oracle reaches the win region*） |
| 面板 6 | AND THIS IS WHY（label supply，*more routing upside, less usable training signal*） |
| 页脚 | 邮箱 `.25` · repo · QR「Explore all 8 settings」 |

## 1. 查实的现状（2026-09-11）

| 东西 | 状态 | 还差什么 |
|---|---|---|
| 海报 | v9.10 已印，A1 竖版 | 无。**不改** |
| demo 录像三题 | v2 已提交（commit `83c857b`），`demo_portable.html` 11.7 MB | §506.10 定下的三处修改还没做（红绿框对照上色 / 碳排悬停去 "published" / 四条措辞）；没有演讲模式（一打开就自动播放并轮换三题） |
| live 页 | DGX 上站点容器已跑 20 h，server 在**裸前台进程**里（不是 tmux） | 周三要重启；需要一个不会随终端断掉的运行方式 |
| slide | 模板未到（走 Slack，Gmail 里没有 showcase 邮件） | 内容一个字都没有 |
| 演讲 slot | 10 分钟已确认（user 09-10）；节目单 14:45–15:30「Student presentations」共 45 分钟 | 不知道：含不含问答、自带电脑还是统一电脑、接口 |
| 网络 | quark 在 UCL 校园网上 `ssh spark`（cloudflared）已验证可用（2026-05-28；Tailscale 被黑洞，cloudflared 通）。会场 = UCL Centre for AI，大概率同一网络 | 当天 09:00 仍要实测一次 |
| 投票 | 10:00 开始，**14:35 截止**，演讲 14:45 才开始 | 演讲不决定奖；13:15–14:35 站在板前决定 |
| GPU 侧 | B1 shopping 三格 ~09-13 落地（A100） | 只发下一条 chain，不开新分析，人力都在演讲上 |

## 2. 定下来的细节

编号 D，之后改动请改这里并写日期。

- **D1 演讲里的 demo 只放录像回放 task 130，不跑 live。** 130 是「找日落照片的那条 listing」：LOOK 2 步解出（$0.007）、READ 9 步失败（$0.041，文本树里根本没有 sunset 这个词）、BOTH 3 步解出（$0.014）。三秒能懂，最长一栏 9 步，手动步进约 45 秒。live 页首步要 ~20 秒、一栏最多 12 步、时长不可控，只在展板上给深聊的访客。
  为什么不用 76：76 已经印在海报中部（READ 解出 / LOOK 转圈），演讲讲 130 正好和它成一对 —— 130 是「贵的看法值」、76 是「便宜的看法值」，合起来就是题目那句问句；讲完 130 一句话把 76 指回海报。
  130 还有一个顺手的点：learned choice 选的是 READ，恰好是唯一失败的那栏 —— 直接引出「那能不能学会选」。
- **D2 demo 进演讲走三级故障梯。** ① quark 浏览器开 `demo_portable.html?task=130&autoplay=0`（演讲模式，Phase 1 实现：停在 130 第 0 步、不自动播、不轮换）→ ② 投影只认主办方电脑：U 盘上的同一个文件，任何电脑双击即开 → ③ 浏览器出问题：slide 里嵌 40 秒 MP4 录屏，再兜底一张三帧静态图。
- **D3 slide 默认走 pptx**（模板大概率是 pptx，主办方可能收文件）。做法照海报：`talk_content.md` 单一来源 → `build_talk.py`（python-pptx）→ pptx + PDF；模板到了只换 layout 不改内容。HTML deck（demo 直接 iframe 嵌进去、零切换）只在 Zekun 确认「自带电脑」且 user 想要时才做。
- **D4 演讲词汇 = 海报词汇，三套名字一次对齐。** 海报系统图写 DOM / SoM / Vision，海报截图带和 demo 写 READ / LOOK / BOTH。第 2 页说一次：LOOK = screenshot only（海报的 Vision）· READ = page text only, no image（海报的 DOM，是 accessibility tree 不是 HTML）· BOTH = screenshot with numbered marks（海报的 SoM）。之后全程 LOOK / READ / BOTH。
- **D5 演讲 7 页，顺序照海报六面板走**（§4 有逐页骨架），约 8 分 40 秒 + 80 秒缓冲。面板 3、4（行为不同 / 失败不同）各一句带过，不单独成页。
- **D6 不引未发表结果。** §505 的预算路由不上台，不写进 slide；问答被问「生产系统今天该怎么办」时用 §4 现成答案（always-cheapest 难打、先把 agent 做好再学选择），最多加一句「ongoing work, unpublished」。
- **D7 数字纪律不变**：只用 §5 数字表里的数字，每个数字带它的 baseline 短语；`13.7` 和 `12–14%` 永远不说（都已作废）。但 §5「海报上有 / 没有」那一列要按 v9.10 重审（Phase 2）。
- **D8 live 页周三的运行方式**：DGX 上 tmux 会话 `showcase` 两个窗格（站点 `docker compose up -d` + `server.py`）；quark 上单独一个 PowerShell 窗口只跑 `ssh -N -L 8799:localhost:8799 spark`；笔记本电源计划改「从不睡眠」；手机热点当备用网络。
- **D9 §506.10 三处修改照 1A / 2A / 3B 做**（Phase 1），改完重建 portable、像素比对、提交、写笔记和台账。
- **D10 板前按 Run 之前先说一句**「first step in about twenty seconds」——冷启动 ~20 秒是实测值，说出来就不像卡。

## 3. 六个 phase

每个 phase 的判据都能用一句话验证真假，验证通过就勾掉。

### Phase 0 · 今天 09-11 · 定方向、问清楼下条件（≤ 1 小时）

- [x] D1–D10 入档（本文件）
- [ ] Slack DM Zekun 四个问题（§6 有现成文字）：模板何时到 · 自带电脑还是统一电脑 · 10 分钟含不含问答 · 接口（HDMI / USB-C）和演讲顺序
- [ ] user 确认 D1（演讲只放录像 130）和 D3（默认 pptx）

**判据**：消息已发；D1 / D3 有 user 的一句「可以」。

### Phase 1 · 09-11 晚 → 09-12 · demo 收尾

- [ ] 1A 红绿框对照上色：录像页 `pickLine()` 与 live 页 `renderPick()` 同一规则 —— 只有被选中栏**唯一**答对才绿（"right — the only view that was"）；它对别栏也对为灰并点名更便宜的那栏（"right — but so was X, for less: the choice didn't matter"）；它错别栏对为红（"wrong — X got it"）；全错为灰。76 题要写出「唯一解出」
- [ ] 2A 碳排悬停文字去掉 "published"：输出 token 能耗是实测、输入 token 是推算
- [ ] 3B 四条措辞：README「Check」→「接线核对（4/5 折见过）」· 删「4 of its 5 models agreed」· 「trained on the recorded tasks」→「trained on recorded runs of this site's tasks」· README 补「看图的能耗未单独建模」
- [ ] 演讲模式：URL 参数 `?task=<id>` 直接落到该题第 0 步、`?autoplay=0` 不自动播不轮换（键盘 ← → 仍可用；`4` 仍进 live 页）
- [ ] 重建 `demo_portable.html` + 逐像素比对 + 在 quark 上双击验证三题
- [ ] 录 MP4：Playwright 录屏 task 130 三栏步进 40 秒（Playwright 自带 ffmpeg 转 mp4），存 `deliverables/showcase/demo/talk_130.mp4`
- [ ] 提交；笔记 §507 + 台账

**判据**：quark 上双击 portable 三题正常；`?task=130&autoplay=0` 打开后停在 130 第 0 步不动；mp4 能在 quark 上播放；有 commit hash。

### Phase 2 · 09-12 → 09-13 · 讲稿与走读重写（不等模板）

- [ ] `talk_content.md`：7 页，每页 = 标题 · 画面（用哪张海报图）· 口播原文（英文）· 秒数 · 每个数字指向 §5 的哪一行
- [ ] `SHOWCASE_PREP.md §2` 板前走读按六面板顺序重写（标题条 → 中部 76 → 面板 1/2 → 5 → 6 → 桌上电脑；20 秒版 = 标题 + 面板 5 + 面板 6）
- [ ] `SHOWCASE_PREP.md §3` 12 分钟版改成 10 分钟版（内容以 `talk_content.md` 为准，§3 只留指针）
- [ ] `SHOWCASE_PREP.md §5` 数字表：「海报上有」那一列按 v9.10 重审；v9 印出来的新数字补进去（76 的 12 / 26 步、$0.09 / $0.10、8 / 9 页；面板 4 的 2.3× / 2.2× / 1.6× / 1.2× / 0.9×；面板 3 的 ~4×）
- [ ] /stress spot-check（Mode A + Mode C；codex 无额度不补）只审讲稿的说法

**判据**：口播原文总字数 1,150–1,300 英文词（约 140 词/分钟 → 8–9 分钟）；每个数字在 §5 有一行；stress 没有 P0。

### Phase 3 · 模板到达当天，最迟 09-14 · 出 slide

- [ ] `build_talk.py` 把 `talk_content.md` 铺进模板 layout → pptx + PDF；断言：每页字数上限、图不越框、无未配对 `*`
- [ ] 模板 **09-14 18:00 还没到** → 用海报配色的中性 16:9 出稿，不再等
- [ ] 主办方若收 slide：09-15 12:00 前发 Zekun

**判据**：pptx 和 PDF 在 quark 上打开无字体替换、无溢出；每页在投影 3 米外能读（字号 ≥ 20pt 正文）。

### Phase 4 · 09-14 → 09-15 · 排练

- [ ] 计时排练 3 遍，其中 2 遍含「slide → 浏览器 → slide」切换；目标 ≤ 9:30
- [ ] 板前 90 秒走读（新 §2）练到不看稿
- [ ] §4 问答、§6 禁语各读两遍

**判据**：连续两遍 ≤ 9:30。

### Phase 5 · 09-15 晚 · 打包

- [ ] quark 桌面一个文件夹：`demo_portable.html` · slide pptx + PDF · `talk_130.mp4`；U 盘同一套
- [ ] 电源适配器 · HDMI 与 USB-C 转接 · 海报筒 · 手机热点已试
- [ ] quark 电源计划「从不睡眠」；浏览器书签 `http://localhost:8799/` 与本地 portable 两条
- [ ] DGX：tmux `showcase` 里 `docker compose up -d` + `server.py`；`curl -s localhost:8799/health` 返回 ok
- [ ] 从 quark 走隧道跑一次 live 全流程（一个 suggestion 任务，三栏出结果，✓/✗ 判定后红绿框正确）

**判据**：清单全勾 + 一张端到端 live 成功的截图存 `deliverables/showcase/demo/live/idle/` 旁。

### Phase 6 · 09-16 当天

| 时间 | 做什么 | 出问题怎么办 |
|---|---|---|
| 09:00 | 挂海报 | — |
| 09:10 | 笔记本：**先**双击离线 portable 让桌面有东西在动；再开隧道、`curl health`、跑一个测试任务 | 隧道不通 → 只用离线 portable，把「try your own」一句「not available on this network today」说清 |
| 09:45 | AV check：quark 接投影，试 slide → 浏览器 → slide 一遍 | 投影只认主办方电脑 → U 盘 portable；再不行 → mp4 |
| 10:00–13:15 | 桌上自动播放；有人问就走读 | — |
| **13:15–14:35** | 板前循环 90 秒走读；深聊的访客给 live 页（按 Run 前说 D10 那句） | live 卡住 → `Stop`，回录像页，别当场排错 |
| 14:40 | 演讲前 5 分钟：浏览器开到 `?task=130&autoplay=0`，slide 开到第 1 页 | — |
| 14:45–15:30 | 演讲（自己的 10 分钟） | 浏览器崩 → 切 mp4 那页 |
| 17:15 | 撤展；DGX 上 `fuser -k 8799/tcp`、`docker compose down` | — |

## 4. 十分钟演讲骨架（7 页）

口播原文写在 `talk_content.md`（Phase 2）；这里定每页**画面、要说的几句、秒数**。数字全部对着 §5 数字表。

| 页 | 秒 | 画面 | 要说的（关键句，英文原文在 talk_content.md） | 对应海报 |
|---|---|---|---|---|
| 1 | 0:00–0:40 | 海报标题 + 副标题 + 一行姓名/导师；角标 *REALM workshop @ EMNLP 2026, accepted* | A web agent can **look** at a page, **read** it, or do **both** — and they cost different amounts. The question: is the expensive context worth paying for, and can you tell in advance? | 标题条 |
| 2 | 0:40–1:50 | 系统图第 ② 段（三种 grounding，嵌真实截图）+ 一行映射 LOOK = Vision · READ = DOM · BOTH = SoM | Six views in total, four of them image-free. 8 website × model settings across VisualWebArena and WebArena, ~8,900 task attempts. Only the page view changes; everything else is held fixed. | 顶部通栏 ② |
| 3 | 1:50–3:30 | **切浏览器**：demo task 130（手动 → 步进） | Same task, three views, one recorded run each. LOOK: two steps, solved. READ: nine steps and never finds it — the text tree never says *sunset*. BOTH: three steps, solved, at twice LOOK's bill. And a learned choice, trained without this task, picked READ — the one view that failed. The poster's centre strip is the mirror case: a price edit READ finishes in 12 steps and LOOK loops for 26. So the view matters per task. Two questions: how much is there to gain, and can you learn to choose? | 中部通栏 76 |
| 4 | 3:30–4:40 | 面板 1 + 面板 2 并排 | No view wins everywhere: the best single view solves only 2–36% of tasks, depending on the setting. The sets overlap but do not coincide. Perfect hindsight — pick the right view per task after the fact — would solve 3.45 to 16.35 more tasks in 100 than the best single view, same direction in all 8. **The ruler**: rerunning one view flips 10–14% of outcomes by itself, and a rerun alone buys 2.0–7.6 points. The ceiling is real but smaller than it looks. (One sentence: they also behave and fail differently — Vision scrolls ~4× more; text-only gives up early, image-only stalls — panels 3 and 4.) | 面板 1 · 2（3 · 4 一句） |
| 5 | 4:40–6:10 | 面板 5 赢区图 | So build a router? Not so fast. Five routing policies, nested cross-validation, 10,000 permutations. **0 of 8** beat always-cheapest on both success and cost. Learned routers buy success only by spending more. Always-cheapest = the single view that costs least on average in that setting, used for every task. And **even perfect hindsight reaches the win region in only 1 of 8** — so the barrier is not primarily the learner. | 面板 5 |
| 6 | 6:10–7:40 | 面板 6 label supply 图 | And this is why. A "which view" training example exists only when a task is solved. Base success 2–36% → 15–97 usable examples per setting; enough to train a classifier in only 2 of 6. More routing upside, less usable training signal. We priced it: the failing settings need at least 2.1–4.2× more tasks. | 面板 6 |
| 7 | 7:40–8:40 | 一句结论 + QR + 「come to the board — type your own task, it runs live in all three views」 | On these benchmarks, always using the cheapest view is hard to beat on both counts, and BOTH is the dearest. A learned per-task choice isn't there yet — and the reason is the supervision, not the classifier: improve the agent first, collect reliable examples, then learn when to look. | 页脚 QR |
| — | 8:40–10:00 | 缓冲 / 问答 | 问答答案在 `SHOWCASE_PREP.md §4`，禁语在 §6 | — |

## 5. demo 进演讲的操作细节

- 演讲前把浏览器开到 `demo_portable.html?task=130&autoplay=0`，全屏（F11），放在 slide 后面一个窗口；切换用 Alt+Tab（排练 2 遍）。
- 步进节奏：按 → 一下一步，READ 栏 9 步最长；边按边说。不按 space（会开自动播放）。
- 讲完 130 **不要**切到 76 或 17；一句话指回海报中部就够。
- 三级故障梯见 D2。slide 里 mp4 那页放在第 3 页之后、默认跳过（PowerPoint 里设为隐藏页）。

## 6. 给 Zekun 的四个问题（Slack DM，今天发）

> Hi Zekun — four quick logistics questions for the 16th so I can finish the slides this weekend:
> 1. Is there a slide template we should use, and roughly when will it be shared?
> 2. Do speakers present from their own laptop, or from one shared machine? (My 10 minutes include a short browser demo — I can bring it on a USB stick as a single HTML file if it has to be the shared machine.)
> 3. Is the 10-minute slot inclusive of Q&A?
> 4. Projector input — HDMI or USB-C? And do you know the speaking order yet?
> Thanks!

## 7. 不做的事

- 演讲里不跑 live 页；live 只在展板上（D1）
- 不加新数字、不引 §505 未发表结果（D6）
- 不重录 demo、不改海报、不加第四道题
- 09-13 B1 shopping 落地后只发下一条 chain，不开新分析
- 不补 codex 那一份审查（无额度，以 Claude + agy 为准）
