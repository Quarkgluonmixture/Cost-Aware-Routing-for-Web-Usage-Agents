---
type: showcase-planning
status: live
created: 2026-09-11
updated: 2026-09-15
event: 2026-09-16
---

# Showcase 09-16 路线图 —— 海报、demo、十分钟演讲怎么闭环

> 这是**向前看的 live 文件**：每个 phase 做完就把判据那一栏勾掉。过去发生了什么写笔记 §507 起，
> 别写在这里。当天要念的英文原稿仍在 `SHOWCASE_PREP.md`（§2 走读 · §4 问答 · §5 数字表 · §6 禁语）。

## 0. 先说结论

> **09-15 现状（先读这段，下面 09-11 的叙述保留作背景）**：演讲电脑 = **MacBook + Chrome**，quark 全天在展板；Zekun 回复：演讲**接自己的电脑**、片子**要提前发他看**（10 分钟含不含问答、顺序、桌子电源 Wi-Fi 仍未知）。片子 v2（10 页 + 参考页）已按 playbook v3 对齐（D16，commit `13e01ee`）；台本 v2 待 user 出声计时；板前走读已按 v9.10 重写（`SHOWCASE_PREP.md §2`，含「海报说过头处怎么说」表）。展板 live 服务在 DGX 上仍是裸进程（D8 的 tmux 未做），且三栏用 **B0 代理密钥** —— 代理余额 09-15 上午实测 $30.18。当天从头到尾的完整清单 → `day-of.html`。

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
| slide | v0 已做（09-11）。**09-13 核到：活动页 For presenters 已公布 *Speaker slide template (.pptx)*、A1 / A0 海报模板** —— 不是「模板未到」，是还没下载进仓库 | 把 pptx 下下来放进 `deliverables/showcase/`，按 Phase 3 搬配色/标题条进 deck CSS |
| 演讲 slot | 10 分钟已确认（user 09-10）；节目单 14:45–15:30「Student presentations」共 45 分钟；~~从 quark 投大屏~~ → **09-13 改为 MacBook 投屏**；**09-13 Zekun：接自己的电脑、片子提前发他**；**09-15 user：下午学生演讲第二个讲**（约 14:55，按每人 10 分钟估，D19） | 不知道：含不含问答、接口、主办方收不收 slide 文件 |
| 网络 | quark 在 UCL 校园网上 `ssh spark`（cloudflared）已验证可用（2026-05-28；Tailscale 被黑洞，cloudflared 通）。会场 = UCL Centre for AI，大概率同一网络 | 当天 09:50 到场后仍要实测一次 |
| 投票 | 10:00 开始，**14:35 截止**，学生演讲 14:45 才开始（user 第二个讲，约 14:55）。**09-13 核到**：观众在活动网站**填全名**投票，每人 3 票可集中可分散；奖金 £300 / £200 / £100，一等奖进 proceedings 封面 + 写 workshop paper 的 mentorship | 演讲不决定奖；板前决定。投票页按**海报标题**列出（活动页：「final poster titles and abstracts will be published once confirmed」）⇒ 登记的标题必须和印出来的一致，见 Phase 0 |
| GPU 侧 | B1 shopping 三格 ~09-13 落地（A100） | 只发下一条 chain，不开新分析，人力都在演讲上 |

## 2. 定下来的细节

编号 D，之后改动请改这里并写日期。

- **D1 演讲里的 demo 只放录像回放 task 130，不跑 live。** 130 是「找日落照片的那条 listing」：LOOK 2 步解出（$0.007）、READ 9 步失败（$0.041，文本树里根本没有 sunset 这个词）、BOTH 3 步解出（$0.014）。三秒能懂，最长一栏 9 步，手动步进约 45 秒。live 页首步要 ~20 秒、一栏最多 12 步、时长不可控，只在展板上给深聊的访客。
  为什么不用 76：76 已经印在海报中部（READ 解出 / LOOK 转圈），演讲讲 130 正好和它成一对 —— 130 是「贵的看法值」、76 是「便宜的看法值」，合起来就是题目那句问句；讲完 130 一句话把 76 指回海报。
  130 还有一个顺手的点：learned choice 选的是 READ，恰好是唯一失败的那栏 —— 直接引出「那能不能学会选」。
- **D2 两处用法、一个页面（user 09-11 定；投屏电脑 09-13 改为 MacBook，见 Phase 0）。** 展出时：海报旁边放 quark，跑现在这个 demo（三题自动播放 + `4` 进 live 页）。演讲时：~~从 quark 投大屏~~ MacBook 投屏，demo 是**单独的一份演讲版**，同一个 `index.html` 加参数 `?task=130&autoplay=0`（停在 130 第 0 步、不自动播、不轮换，只认 ← →）。先展出再演讲，两处不打架。
- **D3 slide 用 HTML deck，demo 直接嵌在第 3 页里（推荐）；pptx 只作备用。** 既然从自己电脑投屏，HTML deck 能把演讲版 demo 用 iframe 嵌进去，一个浏览器窗口全屏，← → 既翻页也步进，不用 Alt+Tab；组会 deck 已经是这个形态。做法仍是 `talk_content.md` 单一来源 → `build_talk.py` 生成 `talk/index.html`，模板到了把它的标题条 / 页脚 / 配色搬进 CSS。同时导出一份 PDF（Playwright print）给主办方收 slide 用。**只有 Zekun 明确说「必须交 pptx 在统一电脑上放」**才走 python-pptx 铺模板那条路，那时 demo 回到 Alt+Tab 切浏览器。
  故障梯照旧三级：deck 里的 iframe 出问题 → 同一台电脑另开标签页 `demo_portable.html?task=130&autoplay=0` → deck 里第 3 页后面藏一页 39 秒 webm（`talk/talk_130.webm`）。
- **D4 演讲词汇 = 海报词汇，三套名字一次对齐。** 海报系统图写 DOM / SoM / Vision，海报截图带和 demo 写 READ / LOOK / BOTH。第 2 页说一次：LOOK = screenshot only（海报的 Vision）· READ = page text only, no image（海报的 DOM，是 accessibility tree 不是 HTML）· BOTH = screenshot with numbered marks（海报的 SoM）。之后全程 LOOK / READ / BOTH。
- ~~**D5 演讲 7 页，顺序照海报六面板走**~~（09-13 作废，见 D15）（§4 有逐页骨架），约 8 分 40 秒 + 80 秒缓冲。面板 3、4（行为不同 / 失败不同）各一句带过，不单独成页。
- **D6 不引未发表结果。** §505 的预算路由不上台，不写进 slide；问答被问「生产系统今天该怎么办」时用 §4 现成答案（always-cheapest 难打、先把 agent 做好再学选择），最多加一句「ongoing work, unpublished」。
- **D7 数字纪律不变**：只用 §5 数字表里的数字，每个数字带它的 baseline 短语；`13.7` 和 `12–14%` 永远不说（都已作废）。但 §5「海报上有 / 没有」那一列要按 v9.10 重审（Phase 2）。
- **D8 live 页周三的运行方式**：DGX 上 tmux 会话 `showcase` 两个窗格（站点 `docker compose up -d` + `server.py`）；quark 上单独一个 PowerShell 窗口只跑 `ssh -N -L 8799:localhost:8799 spark`；笔记本电源计划改「从不睡眠」；手机热点当备用网络。
- **D9 §506.10 三处修改照 1A / 2A / 3B 做**（Phase 1），改完重建 portable、像素比对、提交、写笔记和台账。
- **D10 板前按 Run 之前先说一句**「first step in about twenty seconds」——冷启动 ~20 秒是实测值，说出来就不像卡。
- **D11 演讲材料按 `presentation-playbook.md`（user 的演讲 SOP，09-11 拿进来）的五件产物做，互不复述。** 计划/权威文档 = 本文件（说什么、顺序）+ `SHOWCASE_PREP.md` §4 问答 / §5 数字 / §6 刻意不主张；片子 = `talk/index.html`；台本 = `talk/rehearsal-script.md`；runbook = `talk/RUNBOOK.md`（演讲）+ `demo/README.md` → *Live*（展板）；兜底 = `talk/fallback.html`（黑底满屏截图，每张一行该说什么）+ `talk/talk_130.webm`。数字只在 §5 维护，其余指向。
- **D12 字数按 SOP 实测规律定，不按感觉。** 上限 **1,000 英文词**（÷140 ≈ 7.1 分，÷120 ≈ 8.3 分，再加三分之一的点击与停顿就顶到 10 分钟）。台本 v0 用 `check_talk.py` 实数 **735 词**：5.2–6.1 分 + 三分之一 ≈ 7–8 分，再加 demo 步进约 1 分 ⇒ 8–9 分，留出问答。此前写的 1,150–1,300 词作废：SOP 08-28 实测 1,240 词讲成了 11–12 分钟。**真正的判据是 Phase 4 掐表 ≤ 9:30，字数只是事前估。**片子每张 **≤ 50 词**，开场那张 ≤ 15 词，收尾让它再出现一次。
- **D13 demo 在前，解释在后。** 第 2 页就是 demo（iframe），三栏的说明由 demo 页自己的栏头承担（screenshot only / text tree only / marked screenshot），LOOK=Vision、READ=DOM、BOTH=SoM 的映射口头说；「六种看法、八个设置」的方法页挪到 demo 之后当「我们怎么量的」。片子里不放 demo 截图（demo 活着时放它的照片等于自己跟自己抢），截图只进兜底页。
- **D14 演讲要有 ask，放在收尾句前，三件当场能给的小事**：① 会后到展板输一道自己的任务；② 在跑 web agent 的人告诉我你们用哪种看法、为什么；③ 谁有成功率更高的 agent，借我们测一次 label supply 的结论是否翻转。没有 ask 的汇报结局是礼貌点头散会。
- **D15 故事线按 user 09-13 改：先给观众带走什么，再 行为不同 → 失败不同 → 那就按任务选？→ 学不会 → 将来要什么。** 标题页、demo 页不动；第 3 张用模板的 THE CLAIM 深色框放三条收获；新增行为（海报面板 3）、失败（面板 4 = REALM Table 41）两张；原「不主张」换成「要满足什么条件」。共 10 张 + 备用。台本 v1 按十幕重排（仍待 user 出声重写）。失败那张按源 JSON 重算后加了限定：规律来自 VWA 六格（WA 两格只占 10 / 38 题），脚注不拿只有 13 次命中的 2.3× 当主数（笔记 §512）。
- **D16 按 `talk/presentation-playbook 3.md`（user 09-13 给的 v3）对齐。** ① 第 3 页的三条结论作废（user：「一甩上去一头雾水」），改成**一个问题 + 四个路标**，`close` 回答它。② 图全部为演讲重画（`talk/talk_figures.py`，同一批源数据）：叫法统一成 demo 的 LOOK / READ / BOTH，三种纯文本变体按内容命名，去掉 oracle / pp / cell 代号；失败图的截图侧改叫 LOOK and BOTH（海报 `IMAGE-ONLY` 不对，BOTH 带文字）。③ `hindsight` 页删掉热力矩阵和三臂韦恩（韦恩那三臂的独有解题数下界可归零，§470.3），换成「单一最佳 → 事后最优」逐设置箭头图；脚注只放同臂数比较（加一种看法 +7.14 vs 重跑一次 +4.46–7.59，仅 cls·B0），不再把五臂收益和一次重跑并排（`noise_floor_inventory.md` §2 不许）。④ 片子每页有稳定 id，hash 用 id，参考页移出主讲流程（R 进入）。⑤ 台本 v2 按附录 C：段落按页面 id，精确措辞只锁核心主张 / 边界 / 请求，删掉「背六句」和「字数 ÷140 ÷120 加三分之一」（v3 §4：以出声计时为准）。D12 的字数公式随之作废。
- **D17（09-15，学长意见）演讲要抓人，别太严谨。** ① `hindsight` 改成台下自己的场景：Claude Code 改完网页、开浏览器检查自己的改动；三个大数字 = 大模型三个网站上「每题选对看法」vs 最佳单一看法：成功 +11 到 +16 / 百题 · CO₂e 估算 −7 到 −29% · 用时两个网站持平、一个快 34%（新脚本 `talk/hindsight_efficiency.py`，与 `oracle_sr_cost` 同一套选择，成本自检一致；延迟不是处处更快，所以片子写「same on 2 of 3 sites」）。原来的同臂数重跑比较移出片子，只留在问答。② `learned` 改成大数字「0 of 8」+ 白话图。③ `why` 改名 scaling law：*no wins, no examples* + 「2–4×」。④ 开场页加导师 *Supervisors: Prof. María Pérez-Ortiz · Zekun Wu*（照海报页眉；`check_talk.py` 不把署名行计入字数和听众名检查）。

- **D18（09-15，用户纠正，覆盖 D17 的三页呈现）** `hindsight` 用日常购物请求引入整组潜力，避免重复开场找船 demo；B0·classifieds 最佳固定 → 完美事后选择，约 27 → 43 /100，成本 100% → 80%，底部总结 +16 solved / −20% cost。书桌场景是类比，不是 Claude 实测。`learned` 以全宽结果图为主、0/8 与 1/8 放图下；`why` 保留 scaling law，用成功/全失败 → 赢家标签示意连到「更少成功 → 每个训练例子需要更多任务」，2–4× 降为页脚。

- **D19（09-15，user 告知）当天两处变化。** ① user **09:50 到场**（不是 09:00），先签到、帮忙布置展板，顺带挂自己的海报、把 quark 开起来；官方 AV check 09:45–10:15 只能在 10:15 前挤时间做，赶不上就 14:35 茶歇补接一次投影。② user 是**下午学生演讲第二个**：14:35 前都能留在板前；14:35–14:45 在讲台旁做不需要投影的准备（RUNBOOK 第 2–7 步），第一位讲完换场时再接投影，约 14:55 开讲（按每人 10 分钟估，以主持人为准）。§6 第 ③ 问因此已答。仍未知：10 分钟含不含问答、展板旁桌子 / 插座 / Wi-Fi。

- **D20（09-15，user 纠正 §520 的纯示意图）** `why` 页改回真实数据：6 个 VWA 设置的点，横轴「用某种看法能做对的题 %」，纵轴「第二常见的正确看法有几条例子」（训练判据所在，不用标签总数），加训练分界线 12.5 条；紫色箭头表示 agent 变强 → 做对的题变多 → 往线上走。口径：越过线 = 能训练、能检验，**不等于赢**（事后最优也只 1 of 8）；箭头是方向，不给越线点；2–4× 留在页脚，是下界。新图 `talk/fig/talk_capability.png`（`talk_figures.py::capability_supply`）；§520 的 `talk_scaling.png` 不再上片子。

- **D21（09-15，user 决定）演讲里不说 hindsight，改说 perfect router。** 听众是做 AI 的，router 一词熟悉，也和 `why` 页的 routing 一致；learned router / perfect router 成对出现。取代 §4 词表里「router → a learned choice」那条，`check_talk.py` 的 JARGON 删掉 router / routers。demo 录像界面上的 *learned choice* 不改，台本在 `learned` 页用 *a router — like the learned choice in the demo* 接上。第一次说 perfect router 时交代「事先知道哪种看法能做对」：它是上限，不是真系统。`learned` 页 kicker 改 *Can a router learn which view to use?*，标题改 *No learned router reaches the win corner — even a perfect one rarely does.*（0 of 8 / 1 of 8）；`hindsight` 页表头 Perfect picker → Perfect router。板前走读与板前问答仍用海报原词 *hindsight*，未改。

- **D22（09-15，user 与学长）开场改为真实抓取，删书桌页。** 标题页后直接放 demo 太突兀、信息太多 ⇒ 先放 `agents`：同一张真实 Wikipedia 注册页的两种真实抓取（左：Playwright MCP `browser_snapshot` 原文逐字节选，这一页 161 行；右：同一次会话的截图），标 *READ · Claude Code + Playwright MCP* / *LOOK · GPT-6 Astra computer use*；再放 `prize` 四张卡片（perfect router vs best single view，B0 三设置）；然后进 demo。中途做过一版写死的动画（学长：hard code，且与 demo 重复），未提交。`hindsight` 书桌页与 `prize` 重复，删除；`question` 路标 4 → 3，`learned` kicker ④ → ③；主流程 11 页。素材与脚本 `talk/real_capture.py`。两张图都不是 Claude 或 Astra 的运行记录；Astra 一侧只引 OpenAI computer-use 文档原话 *uses screenshots and other tool results*，Astra 自己的接口我们没有记录，不说「只看截图」。

## 3. 六个 phase

每个 phase 的判据都能用一句话验证真假，验证通过就勾掉。

### Phase 0 · 今天 09-11 · 定方向、问清楼下条件（≤ 1 小时）

- [x] D1–D10 入档（本文件）
- [x] Slack DM Zekun（09-13 已问）。**回复**：演讲接自己的电脑；片子提前发他看（他问「找到 template 了吗」—— 已套用）。**仍未知**：10 分钟含不含问答 · 顺序 · 板旁桌子 / 电源 / Wi-Fi。原问题：自带电脑还是统一电脑 · 10 分钟含不含问答 · 接口和演讲顺序 · 要不要提前交片子 · 板旁桌子 / 电源 / Wi-Fi（「模板何时到」已删，活动页有）
- [x] user 09-11：先展出再演讲；演讲从自己电脑投屏，demo 单独一份或嵌进 slide → D2/D3 按此改写（HTML deck 嵌 demo）
- [x] **09-13**：登记的海报标题 = 印出来的 v9.10（user 确认提交的是 v9），投票页对得上
- [x] **09-13 定：两台。** quark 全天放展板（live 页的 ssh 隧道只在 quark 上）；**MacBook + Chrome 只放片子**（离线，不需要网络 / 隧道 / DGX）。Zekun 若答「统一电脑」，片子走 U 盘，MacBook 退为备份。原判断：先等 Zekun 第 1 问。统一电脑 → 一台 quark 放展板 + U 盘；自带电脑且有第二台 → 两台（quark 放展板跑 live，第二台只放片子，不需要网络 / 隧道 / DGX）；只有 quark → 一台，彩排时把 14:35 换场练一遍。两台的好处不是展板不停（投票已截止），而是省掉 10 分钟换场 + 互为备份

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
- [x] 09-15 `SHOWCASE_PREP.md §2` 板前走读按 v9.10 重写（标题条 → 中部 76 → 面板 1–4 一口气 → 5 → 6 → 电脑；20 秒版 = ① ④ ⑤），另加「海报说过头的地方该怎么说」对照表（IMAGE-ONLY / stalled progress / ~4× / 横轴 / oracle / 韦恩 / +16）
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
- [x] 09-11 `talk/RUNBOOK.md`：开讲前 5 分钟固定动作（标签页从左到右 ① 片子 ② demo 演讲版 ③ 兜底；缩放 ~~125–150%~~（09-13 作废，改 100%）；收书签栏；DND；全屏；带 `#1` 刷新一次确认不跳页；电源；转接头）· 别碰清单（不切编辑器、不 sign in、不进 live 页、不按 space）· 出事了怎么办（不调试、切兜底、一句带过、⛔ 永不说 "it worked this morning"）
- [x] 09-13 模板 `talk/Showcase-Speaker-Deck.pptx` 套进 deck：顶部渐变条 · 页脚一行 · 深色标题页 / 收尾页（立方体背景 + 两个 logo）· 浅色内容页（紫方块 + 双色横线 + 白底图框，配图是白底）· Georgia / Arial / Consolas。页脚与 logo 全用 CSS 画，`check_talk.py` 不会把页脚的 9 个词和「16」算进去。第 1 张按模板加姓名，15 / 15 词
- [x] 第 7 张改成不带位置的「Come to the board」（RUNBOOK 规定演讲前一天之后不改片子，所以不留到当天改）· [ ] 台本收尾句「The board is by the window.」还是旧的 —— user 重写台本时一起改
- [x] 09-13 查出并修掉：demo 里只有截图会伸缩，iframe 一矮截图就被挤没。改之前 1080p 下浏览器 125% 截图 149px、150% **0px**，16:10 1280×800 56px；套模板后标题区更高，又少了 ~90px。改为 demo 固定按 1880×960 排版、整体缩放进片子区域 → 1080p 100% 299px · 150% 200px（物理像素与 100% 相同）· 1280×800 215px；点缩放后留出的边不翻页。RUNBOOK「缩放 125–150%」**作废**，改 100% + F11
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
- [ ] quark 上看展板 demo 的截图够不够大：Windows 显示缩放 ≥ 125% 或浏览器放大时截图被挤小，150% 时消失 → 浏览器 100% + F11，仍小就 Ctrl+− 缩到 80–90%（`demo/README.md` → Live 第 5 步）
- [ ] **MacBook**：装 Chrome；把 `tmp/showcase_talk_bundle.zip`（DGX 上，VS Code 里右键 Download）解压到桌面，保持 `showcase/demo_portable.html` 与 `showcase/talk/` 同级；Chrome 打开 `talk/index.html`：→ 翻完 8 张、第 2 张 demo 步进到底会自动翻页、第 8 张录屏能播、`fallback.html` 截图都在。同一份拷进 U 盘
- [ ] MacBook 转接头（机身没有 HDMI 口就带 USB-C→HDMI）+ 充电器；在家接一次电视或显示器试镜像
- [ ] DGX：tmux `showcase` 里 `docker compose up -d` + `server.py`；`curl -s localhost:8799/health` 返回 ok
- [ ] 从 quark 走隧道跑一次 live 全流程（一个 suggestion 任务，三栏出结果，三栏都判完后红绿灰框正确）
- [ ] 会前一句管理预期发 Zekun（「10 分钟：一段现场 demo 加六张片子，不是一份 deck」）

**判据**：清单全勾 + 一张端到端 live 成功的截图存 `deliverables/showcase/demo/live/idle/` 旁。

### Phase 6 · 09-16 当天

官方节目单（活动页，09-13 核到）× 你在做什么。台上有节目时观众多半在听，板前人少；**真正的投票窗口是 11:20 茶歇和 12:35 起的两小时午饭**。

| 时间（官方） | 节目单 | 你做什么 | 出问题怎么办 |
|---|---|---|---|
| **09:50**（官方布展 09:00–09:30） | Poster set-up | **user 09:50 才到（D19）**：签到，帮忙布置展板；挂海报；笔记本**先**双击离线 portable 让桌面有东西在动；再开隧道、`curl health`、跑一个测试任务；截图太小 → Ctrl+− 缩到 80–90% | 隧道不通 → 只用离线 portable，把「try your own」一句「not available on this network today」说清 |
| 09:45–10:15 | **AV check & speaker briefing** | **MacBook** 接投影，片子 → 第 2 张 iframe 步进几下 → 兜底页各一遍；当面问清含不含问答、自带电脑还是统一电脑（顺序已知：第二个）；**user 09:50 才到 → 10:15 前挤时间做，赶不上就 14:35 茶歇补测** | 投影只认主办方电脑 → U 盘 `talk/` + portable；再不行 → webm |
| 10:00 | Exhibition + voting opens | 回板前，demo 自动播放 | — |
| 10:30–11:20 | 两段 opening + collaboration talk | 板前自动播放即可 | — |
| 11:20–11:35 | Break | 站板前，20 秒版走读（标题 + 面板 5 + 面板 6） | — |
| 11:35–12:35 | Alumni spotlight + PhD route | 板前自动播放即可 | — |
| **12:35–14:35** | Lunch · 13:15–14:35 authors at boards | **12:35 起就站板前**（午饭那 40 分钟人已经在逛板）；循环 90 秒走读；深聊的访客给 live 页（按 Run 前说 D10 那句） | live 卡住 → `Stop`，回录像页，别当场排错 |
| **14:35** | 投票截止 · 10 分钟茶歇 | 带 MacBook 走去讲台，quark 留在展板；到了先做 RUNBOOK「开讲前准备」里不用投影的几步；早上没做 AV check 就趁此接一次投影 | MacBook 接不上 → U 盘插主办方电脑，Chrome 开 `talk/index.html` |
| 14:45–15:30 | Student presentations | **第二个讲**：第一位讲时候场，换场接投影，约 14:55 开讲，自己的 10 分钟 | 浏览器崩 → 切 webm 那页 |
| 15:30–16:30 | Keynote | — | — |
| 16:30–17:00 | Closing & networking | 回板前，live 页给感兴趣的人 | — |
| 17:00–17:30 | Poster take-down | 撤展；DGX 上 `fuser -k 8799/tcp`、`docker compose down` | — |

## 4. 十分钟演讲骨架 —— 一句话主张、页面与精确措辞（09-13 按 v3 改）

**一句话主张**（09-13 改）：How an agent sees the page changes what it does and how it fails; choosing per task would pay in hindsight, but nothing learns that choice yet — the examples only appear when the agent succeeds.

台本在 `talk/rehearsal-script.md`（v1 十幕，中文引导 · 英文台词）。这里只定每幕**目的、画面、锁死句、秒数**。数字全部对着 `SHOWCASE_PREP.md §5`。demo 仍在第 2 幕（D13）。

| 页面 id | 这页的作用（一页一事） | 画面 | 精确措辞 |
|---|---|---|---|
| `opening` | 为什么值得听 | 标题 + 姓名 + 出处 | — |
| `demo` | 亲眼看到三种看法三种账单，learned choice 选错 | demo iframe（`?task=130&autoplay=0`） | *Same task, three ways of seeing it, three different bills.* |
| `question` | 提出本场问题，预告四步 | 深色：一个问题 + 四个路标 | *Should it look, read, or both — and can it learn to choose?* |
| `behaviour` | ① 看法改变行为 | `talk_behaviour.png` | — |
| `failure` | ② 看法改变失败方式（VWA 六格） | `talk_failure.png` | 边界：*from the six VisualWebArena settings* |
| `hindsight` | ③ 事后选对有收益，对着重跑读 | `talk_hindsight.png` | 边界：同臂数比较那句 |
| `learned` | ④ 学不会：0 of 8，事后最优也只有 1 of 8 | `talk_routing.png` | *Zero of eight… only one of eight.* |
| `why` | 例子只在做对时才有 | `talk_label_supply.png` | — |
| `not-yet` | 边界 + 三个请求 | 两张卡 | 边界句 + 三个请求 |
| `close` | 回答 `question` | 标题再现 + QR | *It depends on the task — and nothing we trained knows it yet.* |
| `reference` | 备用录屏（不在主讲流程） | webm | — |
**裁剪顺序**：先把 VIII 压成一句 → 再删 VI 的重跑两句 → 再让 III 只念三行 → 再删 I 的自陈。**II（demo）永不砍；VII 的 0 of 8 与 1 of 8 必须连着说。**

**沉默三处**：II「the one view that failed」后 2 秒；VII「only one of eight」后 3 秒不翻页；IX「that's where this should be tested」后 2 秒。

**去黑话词表**（片子与台本英文行 grep，零命中才算过；海报自己用过的词除外）：`oracle`（只许 *hindsight oracle* 这一个海报原词）· ~~`router`（说 *a learned choice*）~~ **09-15 D21 起允许**，说 *learned router* / *perfect router*· `mode`（说 *view*）· `pp`（说 *more tasks in 100* / *points*）· `AXTree` · `DOM`（说 *page text / text tree*；海报系统图除外）· `cell` · `condition` · `replicate`（说 *rerun*）· `episode`（说 *task attempt*）· `SR` · `P-text` / `P-SoM` / `P-prompt`（说 *text-only views*）· `canonical` · `phantom` · `baseline`（说 *always-cheapest* / *best single view*）。

## 5. 现场操作 → `talk/RUNBOOK.md`

演讲开讲前 5 分钟的固定动作、标签页顺序、别碰清单、出事怎么办，全在 runbook 里，这里不复述。两条硬规则先记住：
demo 这张只用 → 步进、不按 space；浏览器出问题不调试，切兜底页，一句带过，⛔ 永不说 "it worked this morning"。

## 6. 给 Zekun 的五个问题（Slack DM，09-13 按 MacBook 改）

> Hi Zekun, a few quick logistics questions for Wednesday:
> 1. For the student presentations, can I present from my own laptop (a MacBook; I'll bring a USB-C to HDMI adapter, so is HDMI right for the projector?), or do all speakers use one shared machine? My slides are a single HTML page with a short recorded demo inside. It runs offline in Chrome or Edge, so I can also bring it on a USB stick.
> 2. Does the 10-minute slot include Q&A?
> 3. Do you know the speaking order yet?
> 4. Do you need the slides in advance? If so, by when, and is a PDF fine?
> 5. For the laptop demo next to my poster: will there be a small table and a power socket by the board, and Wi-Fi for presenters (eduroam is fine)?
>
> Thanks!

每问的答案会改什么：① 统一电脑 → U 盘，且那台要有 Chrome / Edge，只能放 PowerPoint 就放不了 demo · ② 含问答 → 台本压到 7–8 分钟 · ③ ~~第一个讲 → 14:35 立刻走~~ 已答（09-15）：第二个讲，见 D19 · ④ 要交 → `talk.pdf`，提醒第 2 张 demo 不在 PDF 里 · ⑤ 无 Wi-Fi → 手机热点，不行就只放离线录像；无电源 → 自带插线板。

## 7. 不做的事

- 演讲里不跑 live 页；live 只在展板上（D1）
- 不加新数字、不引 §505 未发表结果（D6）
- 不重录 demo、不改海报、不加第四道题
- 09-13 B1 shopping 落地后只发下一条 chain，不开新分析
- 不补 codex 那一份审查（无额度，以 Claude + agy 为准）
