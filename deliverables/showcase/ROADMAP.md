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
| 演讲 slot | 10 分钟已确认（user 09-10）；节目单 14:45–15:30「Student presentations」共 45 分钟；**从 quark 投大屏，slide 与 demo 同一台电脑**（user 09-11） | 不知道：含不含问答、接口、主办方收不收 slide 文件 |
| 网络 | quark 在 UCL 校园网上 `ssh spark`（cloudflared）已验证可用（2026-05-28；Tailscale 被黑洞，cloudflared 通）。会场 = UCL Centre for AI，大概率同一网络 | 当天 09:00 仍要实测一次 |
| 投票 | 10:00 开始，**14:35 截止**，演讲 14:45 才开始 | 演讲不决定奖；13:15–14:35 站在板前决定 |
| GPU 侧 | B1 shopping 三格 ~09-13 落地（A100） | 只发下一条 chain，不开新分析，人力都在演讲上 |

## 2. 定下来的细节

编号 D，之后改动请改这里并写日期。

- **D1 演讲里的 demo 只放录像回放 task 130，不跑 live。** 130 是「找日落照片的那条 listing」：LOOK 2 步解出（$0.007）、READ 9 步失败（$0.041，文本树里根本没有 sunset 这个词）、BOTH 3 步解出（$0.014）。三秒能懂，最长一栏 9 步，手动步进约 45 秒。live 页首步要 ~20 秒、一栏最多 12 步、时长不可控，只在展板上给深聊的访客。
  为什么不用 76：76 已经印在海报中部（READ 解出 / LOOK 转圈），演讲讲 130 正好和它成一对 —— 130 是「贵的看法值」、76 是「便宜的看法值」，合起来就是题目那句问句；讲完 130 一句话把 76 指回海报。
  130 还有一个顺手的点：learned choice 选的是 READ，恰好是唯一失败的那栏 —— 直接引出「那能不能学会选」。
- **D2 两处用法、一个页面（user 09-11 定）。** 展出时：海报旁边放 quark，跑现在这个 demo（三题自动播放 + `4` 进 live 页）。演讲时：从 quark 投大屏，demo 是**单独的一份演讲版**，同一个 `index.html` 加参数 `?task=130&autoplay=0`（停在 130 第 0 步、不自动播、不轮换，只认 ← →）。先展出再演讲，两处不打架。
- **D3 slide 用 HTML deck，demo 直接嵌在第 3 页里（推荐）；pptx 只作备用。** 既然从自己电脑投屏，HTML deck 能把演讲版 demo 用 iframe 嵌进去，一个浏览器窗口全屏，← → 既翻页也步进，不用 Alt+Tab；组会 deck 已经是这个形态。做法仍是 `talk_content.md` 单一来源 → `build_talk.py` 生成 `talk/index.html`，模板到了把它的标题条 / 页脚 / 配色搬进 CSS。同时导出一份 PDF（Playwright print）给主办方收 slide 用。**只有 Zekun 明确说「必须交 pptx 在统一电脑上放」**才走 python-pptx 铺模板那条路，那时 demo 回到 Alt+Tab 切浏览器。
  故障梯照旧三级：deck 里的 iframe 出问题 → 同一台电脑另开标签页 `demo_portable.html?task=130&autoplay=0` → deck 里第 3 页后面藏一页 39 秒 webm（`talk/talk_130.webm`）。
- **D4 演讲词汇 = 海报词汇，三套名字一次对齐。** 海报系统图写 DOM / SoM / Vision，海报截图带和 demo 写 READ / LOOK / BOTH。第 2 页说一次：LOOK = screenshot only（海报的 Vision）· READ = page text only, no image（海报的 DOM，是 accessibility tree 不是 HTML）· BOTH = screenshot with numbered marks（海报的 SoM）。之后全程 LOOK / READ / BOTH。
- **D5 演讲 7 页，顺序照海报六面板走**（§4 有逐页骨架），约 8 分 40 秒 + 80 秒缓冲。面板 3、4（行为不同 / 失败不同）各一句带过，不单独成页。
- **D6 不引未发表结果。** §505 的预算路由不上台，不写进 slide；问答被问「生产系统今天该怎么办」时用 §4 现成答案（always-cheapest 难打、先把 agent 做好再学选择），最多加一句「ongoing work, unpublished」。
- **D7 数字纪律不变**：只用 §5 数字表里的数字，每个数字带它的 baseline 短语；`13.7` 和 `12–14%` 永远不说（都已作废）。但 §5「海报上有 / 没有」那一列要按 v9.10 重审（Phase 2）。
- **D8 live 页周三的运行方式**：DGX 上 tmux 会话 `showcase` 两个窗格（站点 `docker compose up -d` + `server.py`）；quark 上单独一个 PowerShell 窗口只跑 `ssh -N -L 8799:localhost:8799 spark`；笔记本电源计划改「从不睡眠」；手机热点当备用网络。
- **D9 §506.10 三处修改照 1A / 2A / 3B 做**（Phase 1），改完重建 portable、像素比对、提交、写笔记和台账。
- **D10 板前按 Run 之前先说一句**「first step in about twenty seconds」——冷启动 ~20 秒是实测值，说出来就不像卡。
- **D11 演讲材料按 `presentation-playbook.md`（user 的演讲 SOP，09-11 拿进来）的五件产物做，互不复述。** 计划/权威文档 = 本文件（说什么、顺序）+ `SHOWCASE_PREP.md` §4 问答 / §5 数字 / §6 刻意不主张；片子 = `talk/index.html`；台本 = `talk/rehearsal-script.md`；runbook = `talk/RUNBOOK.md`（演讲）+ `demo/README.md` → *Live*（展板）；兜底 = `talk/fallback.html`（黑底满屏截图，每张一行该说什么）+ `talk/talk_130.webm`。数字只在 §5 维护，其余指向。
- **D12 字数按 SOP 实测规律定，不按感觉。** 上限 **1,000 英文词**（÷140 ≈ 7.1 分，÷120 ≈ 8.3 分，再加三分之一的点击与停顿就顶到 10 分钟）。台本 v0 用 `check_talk.py` 实数 **735 词**：5.2–6.1 分 + 三分之一 ≈ 7–8 分，再加 demo 步进约 1 分 ⇒ 8–9 分，留出问答。此前写的 1,150–1,300 词作废：SOP 08-28 实测 1,240 词讲成了 11–12 分钟。**真正的判据是 Phase 4 掐表 ≤ 9:30，字数只是事前估。**片子每张 **≤ 50 词**，开场那张 ≤ 15 词，收尾让它再出现一次。
- **D13 demo 在前，解释在后。** 第 2 页就是 demo（iframe），三栏的说明由 demo 页自己的栏头承担（screenshot only / text tree only / marked screenshot），LOOK=Vision、READ=DOM、BOTH=SoM 的映射口头说；「六种看法、八个设置」的方法页挪到 demo 之后当「我们怎么量的」。片子里不放 demo 截图（demo 活着时放它的照片等于自己跟自己抢），截图只进兜底页。
- **D14 演讲要有 ask，放在收尾句前，三件当场能给的小事**：① 会后到展板输一道自己的任务；② 在跑 web agent 的人告诉我你们用哪种看法、为什么；③ 谁有成功率更高的 agent，借我们测一次 label supply 的结论是否翻转。没有 ask 的汇报结局是礼貌点头散会。

## 3. 六个 phase

每个 phase 的判据都能用一句话验证真假，验证通过就勾掉。

### Phase 0 · 今天 09-11 · 定方向、问清楼下条件（≤ 1 小时）

- [x] D1–D10 入档（本文件）
- [ ] Slack DM Zekun 四个问题（§6 有现成文字）：模板何时到 · 自带电脑还是统一电脑 · 10 分钟含不含问答 · 接口（HDMI / USB-C）和演讲顺序
- [x] user 09-11：先展出再演讲；演讲从自己电脑投屏，demo 单独一份或嵌进 slide → D2/D3 按此改写（HTML deck 嵌 demo）

**判据**：消息已发。

### Phase 1 · 09-11 晚 → 09-12 · demo 收尾

- [x] 1A 红绿框对照上色：录像页 `pickLine()` 与 live 页 `renderPick()` 同一规则 —— 只有被选中栏**唯一**答对才绿（"right — the only view that was"）；它对别栏也对为灰并点名更便宜的那栏（"right — but so was X, for less: the choice didn't matter"）；它错别栏对为红（"wrong — X got it"）；全错为灰。76 题要写出「唯一解出」
- [x] 2A 碳排悬停文字去掉 "published"：输出 token 能耗是实测、输入 token 是推算
- [x] 3B 四条措辞：README「Check」→「接线核对（4/5 折见过）」· 删「4 of its 5 models agreed」· 「trained on the recorded tasks」→「trained on recorded runs of this site's tasks」· README 补「看图的能耗未单独建模」
- [x] 演讲模式：URL 参数 `?task=<id>` 直接落到该题第 0 步、`?autoplay=0` 不自动播不轮换（键盘 ← → 仍可用；`4` 仍进 live 页）
- [x] 重建 `demo_portable.html` + 逐像素比对（headless 两版同参数截图：内容帧相同，唯一差异是点击光环的淡出相位）· [ ] 在 quark 上双击验证三题（user）
- [x] 录屏：Playwright 录 task 130 三栏步进 39 秒，存 `deliverables/showcase/talk/talk_130.webm`（Playwright 自带的 ffmpeg 只有 VP8，出不了 mp4；HTML deck 里 `<video>` 放 webm 没问题，若以后要 mp4 装 `imageio-ffmpeg` 转一次）
- [x] 提交；笔记 §507.4 + 台账

**判据**：quark 上双击 portable 三题正常；`?task=130&autoplay=0` 打开后停在 130 第 0 步不动；webm 能在 quark 上播放；有 commit hash。**DGX 侧已验（headless Chromium 断言 + 截图）；quark 侧两条等 user 双击。**

### Phase 2 · 09-12 → 09-13 · 台本 + 走读重写（SOP §11，不等模板）

顺序按 SOP §0b：先台本骨架，再补英文台词，片子最后做。

- [x] 09-11 `talk/rehearsal-script.md` **v0 骨架**：一句话主张 · 七幕各一个目的句 · 六句锁死句（= 片子标题）· 三处沉默 · 转场链 · 裁剪顺序 · 被问到时（指 §4）· 出事了
- [ ] 每幕填英文台词（短句、一句一行引用块），出声念一遍改成自己的话；数词 ÷140 与 ÷120
- [ ] `SHOWCASE_PREP.md §2` 板前走读按六面板顺序重写（标题条 → 中部 76 → 面板 1/2 → 5 → 6 → 桌上电脑；20 秒版 = 标题 + 面板 5 + 面板 6）
- [ ] `SHOWCASE_PREP.md §3` 12 分钟版删掉，只留指针到台本
- [ ] `SHOWCASE_PREP.md §5` 数字表：「海报上有」那一列按 v9.10 重审；补 v9 印出来的新数字（76 的 12 / 26 步、$0.09 / $0.10、8 / 9 页；面板 4 的 2.3× / 2.2× / 1.6× / 1.2× / 0.9×；面板 3 的 ~4×）；**每个数先问「它在数什么」**
- [ ] 去黑话 grep（词表见 §4 末）跑一遍台本英文行
- [ ] /stress spot-check（Mode A + Mode C；codex 无额度不补）只审台本的说法
- [ ] **user 自己重写一遍台本**（SOP §11.10：模型写的是初稿，念顺的那版才是台本）

**判据**：英文台词 ≤ 1,000 词（v0 实数 735）；六句锁死句能不看稿背出；每个数字在 §5 有一行；黑话 grep 零命中（`check_talk.py` 已含）；stress 没有 P0。

### Phase 3 · 09-13 → 最迟 09-14 · 片子 + 兜底 + runbook（SOP §3 / §9）

- [x] 09-11 `talk/index.html`：附录 A 的样式与翻页脚本原样，7 张 + 1 张备用；每张 = kicker + 锁死句标题 + 一个图形；← → / 空格 / 点击；URL hash 记页；`@media print` 每张一页
- [x] 第 2 张 = `<iframe src="../demo_portable.html?task=130&autoplay=0">`（单文件，quark 上不需要 frames/）；翻到这张焦点交给 iframe；demo 到底后再按 → 它 `postMessage(demo-next)` 给 deck 翻页，第 0 步再按 ← 同理回上一张 —— headless 已测
- [x] 备用张（第 8 张，`class="backup"`，End 键直达）= `<video src="talk_130.webm">`
- [x] `talk/check_talk.py`：附录 E 第 3 步那条正则数词（≤ 50，开场 ≤ 15）· 片子里每个数字都在 `SHOWCASE_PREP.md` · 黑话词表 grep（片子 + 台本英文行）· 听众名/元层 grep → **CHECK PASS**（首版开场 22 词因署名行超线，署名移到收尾页）
- [x] `talk/fallback.html`（附录 B 原样）：6 张截图 —— 130 第 1 步 · LOOK 2/2 · READ 第 5 步 · 三栏结束 · 76 第 1 步 · 76 READ 12/12，每张一行「哪个 run · 该说什么」
- [x] 09-11 `talk/RUNBOOK.md`：开讲前 5 分钟固定动作（标签页从左到右 ① 片子 ② demo 演讲版 ③ 兜底；缩放 125–150%；收书签栏；DND；全屏；带 `#1` 刷新一次确认不跳页；电源；转接头）· 别碰清单（不切编辑器、不 sign in、不进 live 页、不按 space）· 出事了怎么办（不调试、切兜底、一句带过、⛔ 永不说 "it worked this morning"）
- [ ] 模板到了：搬它的标题条 / 页脚 / 配色 / 字体进 CSS；**09-14 18:00 还没到** → 海报配色出稿，不再等
- [x] Playwright 导 `talk/talk.pdf`（print 媒体，每张一页，iframe/视频不进 PDF）· [ ] 主办方若收 slide，09-15 12:00 前发 Zekun
- [ ] 只有 Zekun 说「必须交 pptx 在统一电脑放」才加 pptx 出口（python-pptx 铺模板），demo 改 Alt+Tab

顺手修了 demo 的一个真 bug：iframe 里视口只有 ~790px 时，三栏比行高还高，居中对齐让它们上下同时溢出，盖住了 pick 那一行（16:10 笔记本带浏览器栏也是 ~800px，展板上一样会遇到）。现在矮视口先缩画面（`.stage` 先收缩、`.frame` 按 max-height + aspect-ratio 变窄居中），900 高的板前版不变。

**判据**：`check_talk.py` 全绿；quark 浏览器全屏打开，← → 翻到第 2 张时 demo 停在 130 第 0 步，再按 → 是 demo 步进；`#4` 刷新不回第一张；print 预览每张一页；16:9 投影无溢出。

### Phase 4 · 09-14 → 09-15 · 彩排（SOP §6，照做不省）

- [ ] 第一遍：出声、站着、掐表、不停、**录音**，记卡壳位置
- [ ] 第二遍：只修卡壳，⛔ 不重写内容
- [ ] 第三遍：只练转场链（每幕最后一句 + 下幕第一句）
- [ ] 单独练一遍点击（不说话，走三遍动线：第 1 张 → 第 2 张 iframe 步进 10 次 → 第 3 张 …）
- [ ] 手机录一次，看 60 秒回放
- [ ] 彩排反馈全部采纳，**user 自己改台本**；模型只做片子与编号对齐
- [ ] 板前 90 秒走读（新 §2）练到不看稿；§4 问答、§6 禁语各读两遍

**判据**：连续两遍含 demo 步进 ≤ 9:30；转场链四条不断。演讲前 30 分钟不再练。

### Phase 5 · 09-15 晚 · 打包

- [ ] quark 桌面一个文件夹：`demo_portable.html` · `talk/`（index + fallback + webm，相对路径保持）· PDF；U 盘同一套
- [ ] 电源适配器 · HDMI 与 USB-C 转接 · 海报筒 · 手机热点已试
- [ ] quark 电源计划「从不睡眠」；浏览器书签三条：`http://localhost:8799/` · 本地 portable · 本地 `talk/index.html`
- [ ] DGX：tmux `showcase` 里 `docker compose up -d` + `server.py`；`curl -s localhost:8799/health` 返回 ok
- [ ] 从 quark 走隧道跑一次 live 全流程（一个 suggestion 任务，三栏出结果，三栏都判完后红绿灰框正确）
- [ ] 会前一句管理预期发 Zekun（「10 分钟：一段现场 demo 加六张片子，不是一份 deck」）

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
| 14:45–15:30 | 演讲（自己的 10 分钟） | 浏览器崩 → 切 webm 那页 |
| 17:15 | 撤展；DGX 上 `fuser -k 8799/tcp`、`docker compose down` | — |

## 4. 十分钟演讲骨架 —— 一句话主张、七幕、六句锁死句

**一句话主张**：Choosing how a web agent sees a page would pay; nothing we trained learns the choice, because the examples to learn from only exist when the agent already succeeds.

台本在 `talk/rehearsal-script.md`（中文引导 · 英文台词）。这里只定每幕**目的、画面、锁死句、秒数**。锁死句 = 片子那张的标题，
一字不改；其余允许即兴。数字全部对着 `SHOWCASE_PREP.md §5`。demo 在第 2 幕，解释在它之后（D13）。

| 幕 | 秒 | 画面（一张一个图形） | 目的句 | 锁死句（= 片子标题） | 对应海报 |
|---|---|---|---|---|---|
| I 开场定位 | 0:00–0:35 | 第 1 张：标题（≤ 15 词）+ 姓名/导师 + 角标 *REALM workshop, EMNLP 2026* | 让房间知道这是量出来的东西，不是提案 | *When is expensive perception worth paying for?* | 标题条 |
| II Demo | 0:35–2:45 | 第 2 张：demo iframe（`?task=130&autoplay=0`），三栏栏头自带说明，手动 → 步进 | 让观众亲眼看到同一任务三种看法三种账单，以及 learned choice 选错 | *Same task, three ways of seeing it, three different bills.* | 中部 76（一句话指回） |
| III 尺子 | 2:45–4:15 | 第 3 张：面板 1 + 面板 2（矩阵 + 韦恩）；kicker 写规模「6 views · 8 settings · ~8,900 attempts」 | 上界是真的但要拿重跑当尺子读 | *Hindsight says choosing pays; a rerun says how much of that is noise.* | 面板 1 · 2（3 · 4 各一句） |
| IV 没那么快 | 4:15–5:45 | 第 4 张：面板 5 赢区图 | 0 of 8，且 hindsight 也只有 1 of 8 → 问题不在学习器 | *Learned routers buy success only by spending more.*（海报原句）**停 3 秒** | 面板 5 |
| V 为什么 | 5:45–7:00 | 第 5 张：面板 6 label supply | 例子只在成功时才存在 | *More routing upside, less usable training signal.*（海报原句） | 面板 6 |
| VI 不主张什么 + ask | 7:00–8:15 | 第 6 张：左「we do not claim」三行 · 右「three asks」三行（≤ 50 词） | 可信度动作 + 把「听汇报」变成「一起做」 | *Improve the agent first, collect reliable examples, then learn when to look.* | — |
| VII 收尾 | 8:15–8:40 | 第 7 张：第 1 张再出现 + QR + 「the board is by the window」 | 隔 8 分钟回到题目那句问句 | （同 I） | 页脚 QR |
| — | 8:40–10:00 | 缓冲 / 问答 | 答案在 `SHOWCASE_PREP.md §4`，禁语在 §6 | | |

**裁剪顺序**（时间不够时）：先砍 VI 的「不主张」三行只留 ask → 再砍 III 里面板 3/4 那一句 → 再砍 I 的自陈。**II（demo）与 IV→V 的转场永不砍。**

**沉默三处**：IV 锁死句后停 3 秒不翻页；II 最后「the one view that failed」后停 2 秒；VI 「then learn when to look」后停 2 秒再翻到收尾。

**去黑话词表**（片子与台本英文行 grep，零命中才算过；海报自己用过的词除外）：`oracle`（只许 *hindsight oracle* 这一个海报原词）· `router`（说 *a learned choice*；海报标题 *SO BUILD A ROUTER?* 除外）· `mode`（说 *view*）· `pp`（说 *more tasks in 100* / *points*）· `AXTree` · `DOM`（说 *page text / text tree*；海报系统图除外）· `cell` · `condition` · `replicate`（说 *rerun*）· `episode`（说 *task attempt*）· `SR` · `P-text` / `P-SoM` / `P-prompt`（说 *text-only views*）· `canonical` · `phantom` · `baseline`（说 *always-cheapest* / *best single view*）。

## 5. 现场操作 → `talk/RUNBOOK.md`

演讲开讲前 5 分钟的固定动作、标签页顺序、别碰清单、出事怎么办，全在 runbook 里，这里不复述。两条硬规则先记住：
demo 这张只用 → 步进、不按 space；浏览器出问题不调试，切兜底页，一句带过，⛔ 永不说 "it worked this morning"。

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
