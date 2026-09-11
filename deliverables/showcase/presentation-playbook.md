# 怎么做一场 presentation（含带 demo 的）—— 手册

_通用手册，**自包含**：整个文件 `cp` 到任何工作区的 `docs/` 就能用，模板全在末尾附录 A–E。
规则每条都反推自一次真实踩过的坑（2026-08-28 现场 demo 汇报 · 2026-09-02 单页简报 · 2026-09-16 CEO 录播，
都在 holistic-neon 工作区），括号里的日期只作出处；某一场的具体内容不抄进来。
本文分三段：**§0–§7 规则**（不能违反什么），**§9–§11 动手流程**（片子 · demo · 台本各一份：从哪起手、按什么顺序、骨架长什么样），
**附录 A–E 模板**（可运行的片子 HTML · 兜底页 · 台本骨架 · runbook 骨架 · 带 `--check` / `--capture` 的跑脚本骨架）。_

---

## 0. 先判形态：你要做的是哪一种

四种常见形态，差别只在**观众的眼睛看什么**和**失败的代价落在哪**：

| 形态 | 观众看什么 | 什么时候用 | 需要的产物 |
| --- | --- | --- | --- |
| **纯讲** | 片子 + 你 | 讲方向、报进展，手上没有能跑的东西 | 计划 · 片子 · 台本 |
| **现场 demo** | 产品本身；片子只做开场和收束 | 有能在**你自己环境**跑起来的真东西，且挂了能当场兜底 | 五件全要，runbook 与兜底不可省 |
| **录播 demo** | 视频 + 片子 | 要先经人过目；或依赖你控制不了的部署；或有些数字不能上屏 | 计划 · 片子 · voice-over · 录制清单 · 跑脚本（含 `--check`） |
| **单页简报边指边讲** | 一页 HTML，你指着讲 | 一对一或小房间，决策者要能自己回头读，议题不止一个 | 一页 HTML（工作区里有 `/report-dashboard` skill 就从它的模板起手，没有就用附录 A 的样式做单页）· 讲稿 md |

**怎么选：**

- **有真东西能跑吗？** 能 → 带 demo。demo 是主体，片子服务 demo，⛔ 不能反过来。
- **要先经人审吗？** 要 → 录播。录播还顺手解决「线上跑依赖别人的部署」。
- **时段多长？** 15 分钟以内 → 一个对比、一个主张、一个 ask；30 分钟 sync → 才放得下方向和 horizons。
  不知道时段就把裁剪顺序写进台本抬头，到场确认。
- **观众是谁决定主线：** 工程师 → 讲「他自己做不到的」；决策者 → 每一节都能用「这个已经存在」或
  「这是它需要你做的」回答；混合房间 → 每一拍服务不同的人（见 §2 第 6 条）。
- **失败代价：** 现场 demo 挂了，有兜底页就是小事；**一个被发现是假的 demo 不可挽回**。宁可说「还没部署」。

不管哪种形态，底是同一个：**每一句话背后有仓库里十秒能查的东西。** 临时被要求改一句话时判据只有一个：
改完还能被现场验证吗？（08-28 被评「结构像大厂 PM」，真正的原因是这个，不是叙事技巧。）

## 0b. 总流程（倒推着排）

1. 问清**已经答应出去的、带日期的交付有哪些**（§2 第 1 条）。
2. 由**先例**定框架：上一场被判什么，这一场就避什么（§2 第 2 条）。
3. 一句话主张 → 幕 → 每幕一个画面 → 一个 ask（§2）。
4. 写**台本骨架**：幕 + 目的句 + 锁死句（§11）。
5. 把 **demo 做出来、跑齐、截图**（§10）。片子里的画面从这里来。
6. 做**片子**：每张标题 = 锁死句（§9）。
7. **彩排三遍** + 单练点击 + 60 秒回放（§6）。
8. 会前一句管理预期；会后一小时发正式文档（§6）。
9. 归档、LOG 记「结果 + 被判的原因」（§7）。

一天做得完（08-28 实测 24 小时：下午可行性 + 代码，傍晚跑 run + 截图，晚上片子 + 台本，次日早按彩排反馈改）。
⛔ 但权限类的事第一小时就要问，那是别人的响应时间。

---

## 1. 体系：五件产物，各答一个问题，⛔ 互不复述

| 产物 | 回答什么 | 长什么样 | 模板 / 骨架 |
| --- | --- | --- | --- |
| **计划 / 权威文档** | 说什么、什么顺序、⛔ 不许说什么、他会问什么怎么答 | md；含「刻意不主张」「他会问什么」两节；数字带出处 | 无模板：结构按 §2，问答表照 §4「被问到时」的形状 |
| **片子** | 观众眼睛看的 | 单文件 HTML，一张一事，≤ 50 词 | **附录 A**（可运行，7 张示范 6 种组件） |
| **台本** | 嘴里念的 | 中文引导 · 英文台词；锁死句、沉默、转场链、被问到时 | **附录 C**；录播短片的 voice-over 骨架在附录 D 末 |
| **runbook** | 手要做的 | 环境、陷阱、按什么 ID 点、现场命令、别碰清单、出事怎么办 | **附录 D**；跑脚本（每拍一个子命令 + `--check` + `--capture`）**附录 E** |
| **兜底** | 栈挂了怎么继续 | 黑底满屏页：现场 demo 放截图，录播 demo 放各拍的终端原始输出；每张底下一行该说什么；或红→绿的测试 | **附录 B**（截图版；录播版把 `<img>` 换成 `<pre>`，内容来自 `--capture` 存下的 `captures/beatN.txt`） |

- **400 行的 master doc 念不出来。** 一份文档既是操作又是台词又是计划，最后一定要拆成三份，
  并在抬头标明「不要拿它出声彩排」（08-27 `demo-script.md` 的下场）。**先拆，再写。**
- 阶段 / 数字 / 结论只在一处维护，其余指向。**复述的那份一定会漂，而漂的那份往往正是读者看到的那份**
  （08-27 一张表停在「1 正例」，当天已确认 3/6）。

---

## 2. 内容怎么定（准备顺序）

1. **先问「现在已经答应出去的、带日期的交付有哪些」。** 08-28 整场建立在仓库能告诉我的一切之上，
   唯独缺这一件仓库不可能告诉我的，结果是「a very good presentation, but this is for later」。
   读代码答得了「系统是什么样」，答不了「这个季度欠谁什么」。
2. **框架由先例决定，不由偏好决定。** 上一场被判「位置不对」⇒ 这一场不是提案，每一节都要能用
   「这个已经存在」或「这是它需要你做的」回答，⛔ 一张 roadmap 都不能有。
3. **一句话主张 → 五幕左右 → 每幕一个画面。** 没有画面的转场撑不过三十秒（08-28 「automation」
   那段原来对着 2×2 空讲，补了一张阶梯图才立住）。
4. **开场定位决定房间怎么听你。** 两周新人第一句就宣称方向 = 狂妄；解药是一句自陈
   「I started out thinking they were three separate areas」——⛔ 为省时间删它，前一句立刻变味。
   诊断证明「我研究得深」，方向才承担「我们往哪走」；房间里有同级的人时，前四分钟像「找到个 bug」
   位置就定死了。
5. **一定要有 ask，放在收尾句之前，三件很小、当场能给的事。** 没有 ask 的汇报结局是礼貌点头散会。
   ask 里至少一件是「只有你能答的问题」，把「听汇报」变成「一起做决定」。
6. **房间是三批人，每一拍服务的不是同一批**：领导要「值不值得投、代价多大」；资深工程师要
   「对不对、有没有尊重我们写的代码」；同级要「我在哪、他会不会把活全拿走」。政治上最重要的那句
   往往是说给同级听的，说的时候看向那边。
7. **主动声明范围边界，别等人问。**「shared contract」会被听成「统一整个公司」——自己先说清。
8. **对工程师出身的听众：主线 = 他做不到的，副线 = 差异清单。** 递 bug 清单给写这套代码的人等于
   说「你去修」，他修得比你快，你的价值就只剩 QA。判据：**我们的产出里，哪些是读代码读不出来的？**
   规模要**可核而不是可声明**——带真实文件路径的 component map，他自己能估工时。
9. **叫「区别」不叫「问题」。** 对着亲手建这条路的人，「差异」和「缺陷」是两种对话。
   并排两栏：左「规则 / 实践要求什么」，右「产品今天做什么」，不下判断。
10. **「刻意不主张什么」单独一节，是可信度动作不是免责声明。** 不报没测过的节省百分比、不下不属于
    我们的判决、每个数字标「暂定」——只说一次，重复就成了 hedging。
11. **把裁掉的东西在会上说出来。**「第 5、6 幕是下两步，我故意没做假的」比一个能被点开的假页面强；
    「刻意搁置」列出来比省略好——省略会被当成没想到。
12. **引用开关 / flag 做论据时，分清它证明「存在」还是「应当启用」**，差一个决策权。
    ⛔ 别引模型转述的外部材料（付费报告、官网 tagline）；要引就自己 60 秒核，或只用公开法条。

---

## 3. 片子（slides）

- **每张 ≤ 50 词，三秒读完。你会说出口的东西不要同时写在屏幕上**，否则观众在读不在听。
  08-27 学长一句「文字太多」后：第 4 张 150+ 词砍到 51；定稿 7 张实测 15–83 词
  （83 那张是备用参考页）。**开场那张 15 词**——三句话嘴里说、不上屏，收尾再让它出现，隔 8 分钟回来才有分量。
- **一张一事**：kicker（小字定上下文）+ 一句标题（就是那句锁死的台词）+ 一个图形。高亮色只给
  「今天这一刀」（琥珀描边 = changed / today / now）。
- **图分组对应台词句**，不逐框念：骨干按三句话分三组，4.1/4.2/4.3 各指一组。
  「一逐框就变成实习生举着大架构图念 deck」。
- **阶梯的末端写具体不写抽象**：`hourly sweep → emailed, no transcript reviewed` 而不是 "continuous"，
  房间自己会得出结论。喊口号不如给事实。
- **截图不进片子**（demo 活着时放网页照片等于自己跟自己抢）。截图只做兜底页。
- **计数型可视化必须和文字清单数目相等**（09-02 网格显示 7 个非绿格而下面只解释 2 条——
  一个读者能数出来的矛盾）。图会替读者数，数不上就当场露馅；图和表要用同一编号钉在一起。
- **去黑话 grep**：给外人看的产物禁止出现只在内部文档里定义过的名词（`R1` / `oracle fixtures` /
  `primitives` / `canonical`……）。自查 = 对着正文 grep 内部词表，⛔ 不能靠「我觉得这词挺通用」。
  阶段代号第一次出现前用一句话解释。
- **给决策者看的页面里不许出现「怎么对他讲」的元层内容**（"Ask, do not assert" / 听众的名字 /
  第一人称 / 「只有前两条不能出房间」）。讲法住讲稿，页面只放事实和问题。自查 grep：第一人称、听众名。
- **数字上屏前先问「这个数到底在数什么」**：`grep -c` 数的是行不是事（17 个引用 ≠ 7 个调用点）；
  报不准就**点名**不报数。数字只在能溯源、能现场复算时上屏；口径带出处。
- **补了十几轮之后删掉重建。** 沉积物：降级块留旧标题、搬走内容留悬空过渡句、早期口气痕迹、
  重复卡片。重建时不从旧页抄一个字，事实全部重核。
- **技术形态**：纯 HTML 单文件，← → / 空格 / 点击翻页，**URL hash 记住当前页**（文件刷新、⌘R 不会跳回第一张，
  08-28 踩过），`@media print` 每张一页可存 PDF，全屏直接投。字体 Charter 衬线 + 系统无衬线 + mono，
  纸白底。投影前想一眼 16:9 vs 笔记本 16:10，字放大后 horizons 那栏最容易溢出。
- **退役一份可独立打开的产物时，警告写进产物内部**（`<body>` 后注入 SUPERSEDED 横幅），不能只写在周围——
  文件被发出去的时候路径不跟着走，横幅跟着走。

---

## 4. 台本

- **格式：中文 = 你做什么、点哪、注意什么；英文引用块 = 原话，照念。其余任何东西不在台上念。**
  每一步写死标签页、行号、ID。
- **只逐字背 4–6 句，其余允许即兴。** 非母语在压力下先崩的是长句和精确措辞；支点背死了，中间讲糊没人在意。
  锁死句同时就是片子的标题。
- **沉默的位置写死**（08-28 三处：说完 "Pass." 停 2 秒；说完那句最狠的停 3 秒；说完 "The principle exists.
  The contract doesn't." 不说话不翻页）。人最常见的错误是讲完自己最好的一句立刻往下冲。
- **用转场链代替逐句背**：每幕最后一句 + 下一幕第一句。人丢线索几乎都在转场不在段落中间。
  「只要这四条链没断，现场就不会丢。」
- **抬头写明时间不够时的裁剪顺序**；demo 和核心转场永不砍。⚠️ 事先不知道时段多长时，把裁剪顺序写好、
  到场确认时段。
- **实测口播时长，别凭感觉**：字数 ÷ 140 wpm（正常）与 ÷ 120 wpm（紧张），再加点击、停顿、被打断
  ⇒ 现实比表里多 1/3（08-28：1,240 词 ≈ 8.9–10.3 分钟，现实 11–12）。
- **顺序反了含义就变了**：先讲产品事实（打分全失败、判定仍是 PASS），错 key 只作为**复现手段后置**——
  先说「bad API key」房间第一反应就是「那不就是配错了」，发现被缩成配置错误。
- **同一批材料里两份文档要互相对一遍**（「四条 workstream 真正并行」vs roadmap 自己写着有依赖）——
  自相矛盾正是会被当场抓的那类，抓到的人会顺手怀疑其余。
- **相关性不说成因果**（「三个错的是因为没 contract」⇒ 软化成「三个对的用了三种机制，说明标准是共识、
  实现没共享」——无可反驳且同样有力）。
- **「见微知著」式转场**：从一个小修复往外扩到契约，再扩到「整个平台该是什么」，自然接到 horizons。
  horizons 说「dependencies, not dates」，不给月份。
- **被打断**：一句话答完 + 固定返回句（"— so, the same target gave two opposite verdicts."）；
  下游问题推迟但要像掌控不像逃避（"That's exactly where I'm going next — can I show you one more
  result first?"）；**答不上来直说 + 立刻给「怎么才能知道」**——在 assurance 主题下这个回答本身在演示论点。
- **「被问到时」单独一节**，每个问题三四句，含最尖的那几个（「这在生产里真发生过吗」「什么时候能卖」
  「省多少时间」「为什么上次没成」）。
- 作者是你不是模型：模型写的句子在你嘴里不顺，多半是太书面，换成你自己的说法更好。

---

## 5. Demo

**设计**
- **slide 服务 demo，不能反过来。** demo 在前、解释在后；第一拍展示 PASS 之后什么都不解释。
- **真实故障 > 模拟 flag。** 填错 key 让每次判官调用 401（`azureConfigured()` 调用时读 env，一条命令切换）
  ——模拟故障会被质疑「真实情况不是这样」，错 key 不可反驳。
- **先让他们信，再打碎，再回同一屏**：信过一次的 PASS 被同 target 的 FAIL 打碎，再回到那个 PASS 指着
  「一分钟前就在屏幕上的两个词」——观众亲历一次「证据在眼前但没人读」。
  ⚠️ 但**第一次展示时别展开会泄底的层**（08-28 用户抓到：先展开 Categories 会把 `judge unavailable` 提前送出去）。
- **产品自己说的话最狠**（"No failed, warning, or review-needed categories." 而此刻八次判官全失败）。
  找产品在故障态下自信的那一句，比任何自写台词有力。
- **「我没有发明新状态」比「我加了新状态」强**：复用平台已有的 `needs_review`。证明正确语义不是你凭空造的。
- **2×2 断言了的格子要演得到或指得到**：撤掉第 4 个 run 就加一句「它就在列表第 4 行，随时能开」。
- **红队 demo 的目标 prompt 是仪器的一部分，要像调仪器一样调**：加一行人设让 8 个 probe 全被挡（0/8 两次），
  秘密在 prompt 里的位置是真实且反直觉的变量。**先跑出 FAIL 再谈 demo。**
- **能不能跑起来是真正的闸**（Convex team ≠ GitHub write；Clerk key 要设在 dashboard）——别人的响应时间，
  第一天就问。

**现场操作**
- **按 ID 点不按位置**；⛔ demo 前一天之后不再跑任何 run（新 run 插最上面，行号全移）。
  开讲清单里加一条「确认第 1 行是 `#…`」。
- **现场少一次点击 = 少一个出错点**：不切编辑器、不 sign in、不搜索、不点导航；展开过的 transcript
  不收起，靠切标签保住状态。标签页从左到右按叙事顺序，兜底放最右。
- **浏览器缩放 125–150%、收书签栏**：结果页大量 11px muted 小字，投影后排看不见你指的数字。
- **⛔ 不现场跑整个 test suite**；跑就跑聚焦的那一个文件。**两个不是你的红**（locale 假阴性、main 上本来就红的 lint）
  写进 runbook，⛔ 别在人前跑。
- **浏览器崩了不调试**：切兜底页，"I've got these captured"，继续，再也不提。永远不说 "It worked this morning"。
- **兜底不是录屏，是测试红→绿**（`git checkout <test-commit> -- file` → 3 红 → 还原 → 6 绿，三秒）——
  它不是尴尬的备胎，是「这背后有回归保护」。⚠️ rebase 后 commit hash 全变，命令要现查 `git log main..HEAD`。
- **四个 run 跑完先截图**，之后 demo 就存在了，哪怕别的全挂。截图按演讲顺序改名 01→08。

**录播（09-16 形态）**
- **必须录播的两个理由**：审核人要先看；线上跑依赖控制不了的部署。
- **⛔ 绝不演一个没真发生过的平台运行。** 没部署就一句话说没部署 + 展示代码路径。
  「CEO 原谅『还没部署』，一个被发现是假的 demo 不可挽回。」
- **终端只出现在它就是证据的地方**，其余交给片子。终端 120×40、18pt、浅色、无透明、`PS1='$ '`、
  工作目录不带客户名。
- **跑脚本的 `--check`（附录 E）：重算产物、和落盘 JSON 逐字段比、再 grep 片子和计划里引用的每个数、再数片子词预算**——
  不一致就拒绝录制。「片子和产物不许不一致」由脚本保证，不由记忆保证。
- **三段短片比一条长片好录、好审、好重录。** 每段先冷跑一次再录。
- **录完看一遍**：客户名的文件名、上一条命令的 scrollback、通知弹窗。视频文件**放在所有仓库之外**。
- **过滤掉会引来「compared to what?」的行**（load / compute 时长）。
- **发审核人时把 ask 写进消息里**，别让他在 deck 里第一次碰到。
- 什么不能出现在视频里单列一节（客户行、标识符、暂定比值、判决、时长、"saves/faster/compliant" 等词）。

---

## 6. 彩排

1. **出声、站着、掐表。默读不算**——默读比说话快 40%，会藏起所有绕口的地方。
2. **第一遍不停 + 录音**，记卡壳位置。
3. **第二遍只修卡壳，⛔ 不重写内容**——彩排时改内容是无底洞。
4. **第三遍只练转场**（每拍最后一句 + 下拍第一句）。
5. **单独练一遍点击**，不说话，把动线走三遍让手记住路径。
6. **用手机录一次，看 60 秒回放**——难受，但单位时间收益最高。
7. **汇报前 30 分钟不再练。**
8. **会前只发一句管理预期**（「明天是 5 分钟 demo 加几张片子，不是一份 deck」）。
9. **会后一小时内把正式文档和分支链接发过去**——真正的决定在会后还热着的时候做，那时你手上有写好的文档，别人没有。
10. 用户彩排后的反馈**全部采纳**并由用户自己重写台本（08-28 早上五条）；模型只做 deck 与编号对齐。
    彩排暴露的通常是设计者自己看不见的洞（提前泄底、多一次应用切换、没有画面的转场）。

---

## 7. 收尾与归档

- LOG 记「结果 + 为什么」，尤其**被判的原因**（内容 vs 位置）——它决定下一场的框架。
- spent 材料进 `docs/archive/`（或同等归档目录），⛔ 不删（从未提交过的文件 rm 就是永久）。
- 结论更新时 **grep「这个数字 / 这句话还印在哪」**，不能只改最新那份。
- 还活着的规则升到本文；某一场的内容留在那场的文件里。

---

---

# 下半部：动手流程

_上半部（§1–§7）是「不能违反什么」；这一半是「从哪个文件起手、按什么顺序做、骨架长什么样」。
三份流程互相依赖的顺序是：**先定内容（§2）→ 写台本骨架（§11）→ 做 demo（§10）→ 最后做片子（§9）**。
片子最后做，因为它的每张标题就是台本的锁死句，每幕的画面来自 demo 跑出来的东西。_

## 9. 片子怎么做（从附录 A 起手）

1. **先有台本的幕结构和锁死句，再开片子。** 片子 = 开场 1 张 + 每幕 1 张 + 收尾 1 张 + 备用 1 张。
   ⛔ 别先做片子再想话，那样片子会替你把话说完。
2. **把附录 A 存成** `docs/dashboard/<场次-日期>.html`。
   `<style>`、`<script>`、`.pager`、`.hint` 原样保留，`<section>` 全删重写。
   翻页、URL hash 记页、print 每张一页都在保留的部分里，不用再写。
3. **每张一个 `<section class="slide">`，第一张多一个 `on`**。骨架：

   ```html
   <section class="slide">
     <div class="kicker">小字：这张的上下文，或它回答的问题</div>
     <h2>一句标题 —— 就是这一幕的锁死句</h2>
     <!-- 下面只放一个组件 -->
   </section>
   ```

4. **组件按「要表达什么」选，文件里现成的这几种**（class 名沿用，别改名）：

   | 要表达 | 用 | 附录 A 里的示范 |
   | --- | --- | --- |
   | 开场 / 收尾大字 | `section.star > h1`，重点词包 `<span class="hl">`；收尾用 `.clauses` 三行 + `.closeline` | 第 1 / 6 张 |
   | 对照矩阵（同一对象 × 变一个变量） | `.grid2`：表头 `.hd`、行头 `.rh`、格 `.cell.c-red/.c-green/.c-amber`，**变的那格加 `.changed`**；底下 `.after` 一句收束 | 2×2「Only one cell changed」 |
   | 递进 / 阶梯 | `.rungs > .rung`（左 `.l` 名词，右 `.r` 一句），末级 `.last` 高亮；底下 `.already` 一句「已经在跑」 | automation 阶梯 |
   | 对 vs 错两栏清单 | `.split > .col.ok` / `.col.no`，`li > span.m` 打 ✓ ✗；底下 `.thesis` 两行大字，`.ladder` 一行小注 | 3 / 6 |
   | 架构骨干 + 依赖时间轴 | `.arch > .spine`（`.grp[.now] > .gl + ul>li[.tag]`）+ `.horizons`（`.hzlabel` + `.hz[.first] > .w/.h`），`.loop` 一行环 | Continuous Assurance |
   | 备用链条（被问才翻） | `.chain > .layer[.hot/.warm]`：`.n` 编号、`.q` 问题、`.s` 右侧状态 | 六层链 |

   **高亮只有一个含义**：琥珀底 `#fff8e8` + 琥珀描边 = 「今天 / 变的那格 / now」。全 deck 别给它第二个意思。
5. **每张写完数词**：去标签后 ≤ 50 词，备用页可到 80。「词」的定义只有一个权威：附录 E `--check` 第 3 步那条正则
   （以字母或数字开头的 token；单独的 · — → 之类符号不算）。手数时按同一定义，⛔ 别再发明第二种数法——
   两种数法在一张 50 词左右的片子上差 4 词，正好跨线（2026-09-11 实测）。备用张加 class `backup`，check 才知道免于预算。
   超了先删说明文字（那是你要说的话），再删脚注，最后才动标题。开场那张可以只剩 kicker + 标题。
6. **图分组对应台词**：骨干按锁死句分组（三句话 = 三组），一组一句，别按架构层数分。
   台本里写「指第 N 组」，不写「指第 N 个框」。
7. **兜底页**：把附录 B 存成 `docs/dashboard/fallback.html`。`<div class="stage"><img src="../screenshots/NN-….png">`
   按演讲顺序排；`CAPS` 数组每张两段：**这是哪个 run · 该说什么**。黑底满屏、← → 翻，底部一条小字。
   截图文件名自带顺序和结论：`01-run2-main-badjudge-PASS-overview.png` 这种。
8. **编号同步**：deck 定稿后再往台本里写 Slide N；之后每改一次 deck，grep 一遍台本里的 `Slide`。
9. **投前检查**：全屏投一次看 16:9（笔记本 16:10 会骗你）；print 预览每张一页；带 `#3` 刷新不回第一张；
   §3 的三条 grep（内部词表、第一人称、听众名）。
10. **演完**进 `archive/dashboard/`。主论点若已作废，`<body>` 后注入红色 SUPERSEDED 横幅再归档。

## 10. Demo 怎么做

### 10a. 现场版（08-27 一个下午的顺序）

1. **定一句话主张，和观众必须亲眼看到的那一个对比。** 对比要能压进 2×2：同一 target，只变一个变量。
   压不进 2×2 的对比现场讲不清。
2. **可行性先于一切。** 列表：需要什么 · 变量名 · 从哪来 · 风险。当天就去要权限，
   别人的响应时间是真正的闸。用仓库自带的检查器核环境，别信 `.env.example`。
3. **选能在你自己环境里跑起来的路径**，不选「bug 最好看」的（08-27：`owaspllm` 是 Convex-native，
   `redteam` 要 worker + Lambda，根本跑不起来）。
4. **「before」态用真实故障复现，零代码零 flag。** 找一条一秒切换的命令（错 key ⇒ 每次 401）。
   模拟 flag 会被质疑「真实情况不是这样」。
5. **「after」态越小越好，复用平台已有语义。** 先查 schema 里有没有现成状态，再想加字段。
6. **代码排成能讲的故事**：`refactor → test(红) → fix(绿) → feat`。红→绿那两条命令同时就是无栈兜底。
7. **调仪器**：先让 before 态真的出对比（红队 = 先跑出 FAIL）。目标 prompt 一行、秘密靠前；
   一次只改一个变量，跑到对比稳定出现为止。把「为什么这样设」写进 runbook。
8. **跑齐矩阵的每一格**（4 个 run），每格截 overview + 一张关键 transcript，
   文件名按演讲顺序编号并带结论。跑完截完，demo 就存在了。
9. **读产物找「产品自己说的那句」**（故障态下它最自信的一句话），进台词。
10. **写 runbook**（§1 的第四件）：权限表 · 环境陷阱 · 为什么选这个服务 · 配方表（每个 run：分支 / 变量 /
    期望 / **实测**）· 不是你的红 · 现场命令 · 别碰清单。
11. **演讲顺序 ≠ 跑的顺序**：先让他们信 → 同 target 打碎 → 回同一屏揭底。
    第一次展示时别展开会泄底的那层。台本里每一步按 **ID** 写死，不按位置。
12. **demo 前一天之后不跑新 run。** 开讲清单：第 1 行 ID 核对 · 缩放 125–150% · 分支 · dev 进程 · 静音 ·
    标签页从左到右按叙事顺序，兜底最右。

### 10b. 录播版（09-16 形态）

1. **三拍，每拍一个动词**（它停 · 它完成 · 产品渲染它），每拍 60–75 秒。⛔ 不讲架构。
   每拍先回答「屏幕上必须出现什么 / 绝不能出现什么」。
2. **跑脚本（附录 E）一拍一个子命令**。开头注释写清：什么永不上屏、权威文档是哪份。
   `quiet()` 过滤会引来「compared to what?」的行（load / compute 时长）。设计内的 exit 1 写明「这是设计结果」。
3. **`--check` 子命令**：重算 → 与落盘产物**逐字段**比 → 把片子和计划里引用的每个数拿去 grep。
   红了不录。「片子和产物不许不一致」交给脚本，不交给记忆。
4. **voice-over**（骨架在附录 D 末）：每拍四块 —— 屏幕上是什么 · 加载时说什么 · 出现后指着说什么 · ⛔ 不做什么。
   粗体照念，其余是给自己的。文末单列「视频里不能有什么」。
5. **录制清单**（附录 D §6，可并进 runbook 也可独立成 `RECORDING.md`）：录前清单（分支 · `--check` · 每拍冷跑一次 · 终端 120×40 / 18pt / 浅色 / `PS1='$ '` /
   工作目录不带客户名 · DND · 分辨率）· 录制表（clip · 命令 · 长度 · **结尾停在哪一帧**）·
   录后（回看客户名 / scrollback / 通知 · 剪头尾 · 命名 · **存仓库外** · 发审核人时**先在消息里说 ask**）·
   当天出事怎么办。
6. 三段短片比一条长片好录、好审、好重录。
7. **录播的兜底 = 各拍的终端原始输出**：跑脚本的 `--capture` 把每拍 stdout 存进 `captures/beatN.txt`，兜底页满屏显示它们（附录 B 的 `<pre>` 变体）；
   视频放不出来时切这里，讲同样的旁白。⛔ 兜底也走同一份过滤（时长行不进）。

## 11. 台本怎么写（从附录 C 起手）

**骨架，章节照抄：**

```
# 台本 —— 中文引导 · 英文台词
  一句约定：中文 = 做什么、点哪、注意什么；英文引用块 = 原话照念；其余不念。

## 0. 开场前 5 分钟的固定动作
  标签页表（① 片子 ② 产品 ③ 兜底）· 检查清单 · 列表「行号 ↔ ID ↔ 是哪个 ↔ 结果 ↔ 讲的顺序」
  · 幻灯片对照表（张 ↔ 内容 ↔ 哪一幕用）· 时间分配 + 裁剪顺序 + 永不砍的两段

# 第一幕 · 名字（时长）
  【标签 / 第几张】 → 引用块台词（一句一行） → **停 N 秒** → ⛔ 这一拍不做什么 → 「你的目的只有一个：…」
# 第二幕 … 每幕同构

# 被问到时          每问 3–4 句；答不了的给「怎么才能知道」
# 演讲纪律          锁死 4–6 句 · 三处沉默的位置 · 被打断（固定返回句 / 推迟句 / 「答不了 + 怎么知道」）
# 出事了怎么办      不调试 · 切兜底 · 一句带过 · 永远不说 "it worked this morning"
# 最后一次彩排怎么练  只练转场链（把链写出来，3–4 条）
```

**步骤：**

1. **先写幕，每幕一个目的句**（「证明正确语义不是你凭空创造的」）。目的句写不出来的幕删掉。
2. **每幕填英文台词**：短句，一句一行引用块。写完出声念一遍，绕口的改成自己的话。
3. **挑锁死句**：每幕最多一句，它同时是那张片子的标题。其余标「允许自然说」。
4. **标沉默**：在三个最强的句子后面写「停 N 秒」；最强那句后面加「别马上翻页」。
5. **写「被问到时」**：从计划文档的问答表搬过来；最尖的几个问题一定有（「真发生过吗」「省多少时间」
   「什么时候能卖」「上次为什么没成」）。
6. **写转场链**：每幕最后一句 → 下幕第一句，压成 3–4 条箭头链放文末。彩排只练这个。
7. **抬头写时间分配 + 裁剪顺序**，并标明 demo 与核心转场永不砍。
8. **数词算时长**：÷ 140 与 ÷ 120，再加三分之一。超了先砍重复说的段，别砍锁死句。
9. **编号同步**：deck 改一次就 grep 台本里的 `Slide`、行号、ID。
10. **用户自己重写一遍。** 模型写的是初稿，用户念顺的那版才是台本（08-28 早上五条修改全部来自用户彩排）。

---

# 附录：模板

_每份都能直接存成文件用。`{{…}}` 和 `<…>` 是占位；附录 A、B 是 2026-08-28 那场的原文，内容就是「怎么用」的示范，换掉内容即可。_

## 附录 A · 片子 `slides.html`（可运行）

7 张示范 6 种组件：**1** 开场大字（`.star h1` + `.hl`）· **2** 2×2 对照矩阵（`.grid2`，变的格 `.changed`）· **3** 阶梯（`.rungs`，末级 `.last`）·
**4** 对错两栏 + 两行大字（`.split .col.ok/.no` + `.thesis`）· **5** 架构骨干分三组 + 依赖时间轴（`.arch .spine .grp[.now]` + `.horizons`）·
**6** 收尾三句（`.clauses` + `.closeline`）· **7** 备用链条（`.chain .layer[.hot]`，class `backup`）。
翻页 ← → / 空格 / 点击；URL hash 记住当前页；`@media print` 每张一页。存成 `docs/dashboard/<场次-日期>.html`。

````html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>When hai-neo says PASS — 27 Aug 2026</title>
<style>
  :root{
    --paper:#fbfaf8; --ink:#16150f; --muted:#6f6b62; --rule:#e0dbd1;
    --green:#2f6f4f; --red:#b3261e; --amber:#a4690f; --blue:#2a5b8c;
    --serif:"Charter","Bitstream Charter","Sitka Text",Cambria,Georgia,"Songti SC",serif;
    --sans:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",Helvetica,Arial,sans-serif;
    --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;
  }
  *{box-sizing:border-box;margin:0;padding:0}
  html,body{height:100%}
  body{
    background:var(--paper);color:var(--ink);font-family:var(--serif);
    -webkit-font-smoothing:antialiased;overflow:hidden;
  }
  .slide{
    position:absolute;inset:0;display:none;
    flex-direction:column;justify-content:center;
    padding:clamp(2rem,5vw,6rem) clamp(2rem,7vw,9rem);
  }
  .slide.on{display:flex}
  .kicker{
    font-family:var(--sans);font-size:clamp(.65rem,.95vw,.85rem);
    letter-spacing:.14em;text-transform:uppercase;color:var(--muted);
    margin-bottom:clamp(1.5rem,3vh,2.5rem);
  }
  h1{font-size:clamp(2.2rem,5.4vw,4.6rem);line-height:1.06;font-weight:600;letter-spacing:-.02em}
  h2{font-size:clamp(1.5rem,3vw,2.5rem);line-height:1.14;font-weight:600;letter-spacing:-.015em}
  .lede{font-size:clamp(1.05rem,1.7vw,1.5rem);line-height:1.5;color:var(--muted);max-width:46ch}
  .mono{font-family:var(--mono)}
  .num{font-variant-numeric:tabular-nums}

  /* ── 1 ── */
  .verdict{display:flex;flex-direction:column;gap:.35em;margin-bottom:clamp(2rem,5vh,3.5rem)}
  .verdict .pill{
    align-self:flex-start;font-family:var(--sans);font-weight:700;
    font-size:clamp(1.6rem,3.4vw,3rem);letter-spacing:.08em;
    color:var(--green);border:3px solid var(--green);border-radius:999px;
    padding:.15em .7em;line-height:1.1;
  }
  .verdict .line{
    font-family:var(--mono);font-size:clamp(1rem,1.9vw,1.7rem);
    color:var(--muted);letter-spacing:-.01em;
  }
  .ask{font-size:clamp(1.9rem,4.4vw,3.6rem);line-height:1.14;font-weight:600;letter-spacing:-.02em;max-width:22ch}
  .ask em{font-style:normal;border-bottom:.09em solid var(--amber);padding-bottom:.03em}

  /* ── 2 · the 2×2 ── */
  .grid2{
    display:grid;grid-template-columns:auto 1fr 1fr;gap:0;
    margin-top:clamp(1.5rem,4vh,2.5rem);max-width:1100px;
  }
  .grid2 > div{padding:clamp(.7rem,1.8vh,1.35rem) clamp(.9rem,1.8vw,1.6rem);border-bottom:1px solid var(--rule)}
  .grid2 .hd{
    font-family:var(--sans);font-size:clamp(.7rem,1.05vw,.95rem);
    letter-spacing:.1em;text-transform:uppercase;color:var(--muted);border-bottom:2px solid var(--ink);
  }
  .grid2 .rh{
    font-family:var(--sans);font-size:clamp(.78rem,1.15vw,1.05rem);color:var(--muted);
    white-space:nowrap;padding-right:clamp(1rem,2.5vw,2.5rem);
  }
  .cell{font-family:var(--sans);font-weight:700;font-size:clamp(1rem,2vw,1.75rem);letter-spacing:-.01em}
  .cell small{display:block;font-weight:400;font-size:clamp(.72rem,1vw,.9rem);color:var(--muted);letter-spacing:0;margin-top:.25em}
  .c-red{color:var(--red)} .c-green{color:var(--green)} .c-amber{color:var(--amber)}
  .changed{background:#fff8e8;box-shadow:inset 0 0 0 2px var(--amber)}
  .after{margin-top:clamp(1.4rem,3.5vh,2.4rem);font-size:clamp(1.15rem,2.1vw,1.8rem);font-weight:600}

  /* ── 3 · the 3/6 ── */
  .split{display:grid;grid-template-columns:1fr 1fr;gap:clamp(1.5rem,4vw,4rem);margin-top:clamp(1.2rem,3vh,2rem)}
  .col h3{
    font-family:var(--sans);font-size:clamp(.72rem,1.05vw,.92rem);letter-spacing:.1em;
    text-transform:uppercase;color:var(--muted);padding-bottom:.5em;margin-bottom:.7em;border-bottom:1px solid var(--rule);
  }
  .col ul{list-style:none}
  .col li{
    display:flex;gap:.6em;align-items:baseline;
    font-family:var(--mono);font-size:clamp(.95rem,1.75vw,1.5rem);padding:.3em 0;
  }
  .col li span.m{font-family:var(--sans);font-weight:700;width:1.1em;flex:none}
  .ok span.m{color:var(--green)} .no span.m{color:var(--red)}
  .no li{color:var(--muted)}
  .thesis{
    margin-top:clamp(1.6rem,4vh,2.6rem);padding-top:clamp(1.2rem,3vh,1.8rem);border-top:2px solid var(--ink);
    font-size:clamp(1.5rem,3.2vw,2.7rem);font-weight:600;line-height:1.18;letter-spacing:-.015em;
  }
  .thesis .b{display:block}
  .thesis .b:last-child{color:var(--red)}
  .foot{font-family:var(--sans);font-size:clamp(.7rem,1vw,.88rem);color:var(--muted);margin-top:1.1em;line-height:1.5}

  /* ── 4 · the chain ── */
  .chain{margin-top:clamp(1rem,2.5vh,1.8rem);max-width:1150px}
  .layer{
    display:grid;grid-template-columns:2.1em 1fr auto;gap:clamp(.6rem,1.4vw,1.2rem);
    align-items:baseline;padding:clamp(.5rem,1.35vh,.95rem) .7em;
    border-bottom:1px solid var(--rule);color:var(--muted);
  }
  .layer .n{font-family:var(--mono);font-size:clamp(.9rem,1.5vw,1.2rem)}
  .layer .q{font-family:var(--serif);font-size:clamp(1rem,1.95vw,1.6rem);font-weight:600;letter-spacing:-.01em}
  .layer .s{font-family:var(--sans);font-size:clamp(.72rem,1.05vw,.95rem);text-align:right;max-width:34ch}
  .layer.hot{color:var(--ink);background:#fff8e8;box-shadow:inset 0 0 0 2px var(--amber);border-bottom-color:transparent}
  .layer.hot .s{color:var(--amber);font-weight:700;letter-spacing:.06em;text-transform:uppercase}
  .layer.warm{color:var(--ink)}
  .claim{
    font-family:var(--sans);font-size:clamp(.72rem,1.05vw,.95rem);letter-spacing:.1em;
    text-transform:uppercase;color:var(--muted);padding-bottom:.6em;
  }

  /* ── 4 · continuous assurance ── */
  .arch{display:grid;grid-template-columns:1.3fr .85fr;gap:clamp(1.8rem,4.5vw,4.5rem);margin-top:clamp(.8rem,2vh,1.4rem);align-items:start}
  .spine{position:relative}
  .node{
    border-left:3px solid var(--rule);padding:clamp(.45rem,1.2vh,.8rem) 0 clamp(.45rem,1.2vh,.8rem) clamp(.9rem,1.8vw,1.5rem);
    color:var(--muted);
  }
  .node .t{font-family:var(--serif);font-size:clamp(1rem,1.9vw,1.55rem);font-weight:600;letter-spacing:-.01em}
  .node .d{font-family:var(--sans);font-size:clamp(.7rem,1.02vw,.9rem);margin-top:.15em}
  .node.now{color:var(--ink);border-left-color:var(--amber);background:#fff8e8}
  .node.now .tag{
    font-family:var(--sans);font-size:clamp(.62rem,.9vw,.78rem);letter-spacing:.12em;text-transform:uppercase;
    color:var(--amber);font-weight:700;margin-left:.6em;
  }
  .node.strong{color:var(--ink)}
  .loop{
    margin-top:clamp(1rem,2.4vh,1.6rem);font-family:var(--sans);font-size:clamp(.9rem,1.4vw,1.25rem);
    color:var(--muted);border-top:1px dashed var(--rule);padding-top:.7em;
    display:grid;grid-template-columns:auto auto 1fr;gap:.5em clamp(.7rem,1.3vw,1.1rem);align-items:baseline;
  }
  .loop b{color:var(--ink);font-weight:600}
  .loop .k{
    font-family:var(--sans);font-size:.68em;letter-spacing:.13em;text-transform:uppercase;
    color:var(--muted);
  }
  .horizons{border-left:2px solid var(--ink);padding-left:clamp(1rem,2vw,1.8rem)}
  .hz{margin-bottom:clamp(1.2rem,2.9vh,2rem)}
  .hz .w{font-family:var(--sans);font-size:clamp(.74rem,1.1vw,.95rem);letter-spacing:.14em;text-transform:uppercase;color:var(--muted)}
    .hz .d{font-family:var(--sans);font-size:clamp(.62rem,.88vw,.78rem);color:var(--muted);margin-top:.3em;line-height:1.45}
  .hz .h{font-family:var(--serif);font-size:clamp(1.2rem,2.15vw,1.8rem);font-weight:600;letter-spacing:-.015em;margin-top:.14em}
  .hz.first .h{color:var(--amber)}

  /* ladder on slide 3 */
  .ladder{
    margin-top:clamp(1rem,2.5vh,1.6rem);font-family:var(--mono);
    font-size:clamp(.72rem,1.05vw,.92rem);color:var(--muted);line-height:1.85;
  }
  .ladder b{color:var(--ink)}

  .hl{background:linear-gradient(transparent 62%,#ffe9b0 62%);padding:0 .06em}

  /* ── north star + close ── */
  .star h1{font-size:clamp(2.2rem,5.1vw,4.5rem);line-height:1.07;max-width:19ch;letter-spacing:-.022em}
  .clauses{margin-top:clamp(1.8rem,4.5vh,3rem);display:flex;flex-direction:column;gap:clamp(.35rem,1vh,.7rem)}
  .clauses div{
    font-family:var(--serif);font-weight:600;letter-spacing:-.015em;
    font-size:clamp(1.5rem,3.4vw,2.9rem);line-height:1.15;
  }
  .clauses div span{font-family:var(--sans);font-weight:400;font-size:.42em;color:var(--muted);letter-spacing:.02em;margin-left:.9em}
  .clauses div:nth-child(2){color:var(--amber)}
  .closeline{margin-top:clamp(1.6rem,4vh,2.6rem);font-size:clamp(1rem,1.75vw,1.45rem);line-height:1.5;max-width:52ch}
  .closeline b{font-weight:600}

  /* ── workstream convergence ── */
  .conv{display:grid;grid-template-columns:repeat(3,1fr);gap:clamp(1rem,2.5vw,2.4rem);margin-top:clamp(1.4rem,3.5vh,2.4rem)}
  .ws{border-top:3px solid var(--rule);padding-top:clamp(.7rem,1.8vh,1.1rem)}
  .ws.mine{border-top-color:var(--amber)}
  .ws .n{font-family:var(--serif);font-size:clamp(1.05rem,1.95vw,1.6rem);font-weight:600;letter-spacing:-.01em}
  .ws.mine .n{color:var(--amber)}
  .ws .q{font-family:var(--sans);font-size:clamp(.72rem,1.05vw,.92rem);color:var(--muted);margin-top:.3em;font-style:italic}
  .ws .d{font-family:var(--sans);font-size:clamp(.72rem,1.05vw,.92rem);margin-top:.6em;line-height:1.5;color:var(--muted)}
  .converge{
    margin-top:clamp(1.4rem,3.5vh,2.2rem);padding-top:clamp(1rem,2.5vh,1.6rem);border-top:2px solid var(--ink);
    font-size:clamp(1.15rem,2.2vw,1.8rem);font-weight:600;letter-spacing:-.01em;
  }
  .converge small{display:block;font-family:var(--sans);font-weight:400;font-size:.5em;color:var(--muted);margin-top:.5em;letter-spacing:0}

  /* ── automation ladder ── */
  .rungs{margin-top:clamp(1.4rem,3.5vh,2.4rem);max-width:1000px}
  .rung{display:grid;grid-template-columns:auto 1fr;gap:clamp(1rem,2.5vw,2rem);align-items:baseline;
    padding:clamp(.55rem,1.5vh,1rem) 0;border-bottom:1px solid var(--rule)}
  .rung .l{font-family:var(--serif);font-weight:600;font-size:clamp(1.1rem,2.2vw,1.85rem);letter-spacing:-.01em;color:var(--muted);white-space:nowrap}
  .rung .r{font-family:var(--sans);font-size:clamp(.75rem,1.15vw,1rem);color:var(--muted);text-align:right}
  .rung.last{border-bottom:none;background:#fff8e8;box-shadow:inset 0 0 0 2px var(--amber);padding-left:.7em;padding-right:.7em}
  .rung.last .l,.rung.last .r{color:var(--ink)}
  .rung.last .r{font-weight:700}
  .already{margin-top:clamp(1.2rem,3vh,2rem);font-size:clamp(1.05rem,2vw,1.6rem);font-weight:600;letter-spacing:-.01em;line-height:1.45}
  .already .mono{background:#fff8e8;padding:0 .18em}

  /* ── grouped spine ── */
  .grp{margin-bottom:clamp(1rem,2.4vh,1.7rem)}
  .grp .gl{font-family:var(--sans);font-size:clamp(.78rem,1.2vw,1.05rem);letter-spacing:.14em;
    text-transform:uppercase;color:var(--muted);margin-bottom:.35em}
  .grp.now .gl{color:var(--amber)}
  .grp ul{list-style:none;border-left:4px solid var(--rule);padding-left:clamp(1rem,1.9vw,1.6rem)}
  .grp.now ul{border-left-color:var(--amber)}
  .grp li{font-family:var(--serif);font-size:clamp(1.2rem,2.35vw,2rem);font-weight:600;
    letter-spacing:-.01em;color:var(--muted);padding:.12em 0}
  .grp.now li{color:var(--ink)}
  .grp li .tag{font-family:var(--sans);font-size:.5em;letter-spacing:.12em;text-transform:uppercase;
    color:var(--amber);font-weight:700;margin-left:.7em}
  .hzlabel{font-family:var(--sans);font-size:clamp(.72rem,1.05vw,.9rem);letter-spacing:.12em;
    text-transform:uppercase;color:var(--muted);margin-bottom:clamp(.8rem,2vh,1.3rem)}

  /* chrome */
  .pager{
    position:fixed;right:clamp(1rem,2.5vw,2.2rem);bottom:clamp(1rem,2.5vh,2rem);
    font-family:var(--mono);font-size:.8rem;color:var(--muted);letter-spacing:.08em;
  }
  .hint{
    position:fixed;left:clamp(1rem,2.5vw,2.2rem);bottom:clamp(1rem,2.5vh,2rem);
    font-family:var(--sans);font-size:.72rem;color:var(--rule);letter-spacing:.06em;
  }
  @media print{
    body{overflow:visible}
    .slide{display:flex!important;position:relative;inset:auto;height:100vh;page-break-after:always;break-after:page}
    .pager,.hint{display:none}
  }
</style>
</head>
<body>

<!-- ══ 1 ══ north star -->
<section class="slide on star">
  <div class="kicker">Red team · evaluation automation · the auditor workflow</div>
  <h1>From point-in-time testing to <span class="hl">continuous assurance</span></h1>
</section>

<!-- ══ 2 ══ the 2×2 -->
<section class="slide">
  <div class="kicker">Same target · same configuration</div>
  <h2>Only one cell changed.</h2>
  <div class="grid2">
    <div class="hd"></div>
    <div class="hd">Grader healthy</div>
    <div class="hd">Grader failing</div>

    <div class="rh">Today</div>
    <div class="cell c-red">FAIL<small>finds the leak</small></div>
    <div class="cell c-green">PASS<small>leak invisible</small></div>

    <div class="rh">This branch</div>
    <div class="cell c-red">FAIL<small>finds the leak</small></div>
    <div class="cell c-amber changed">NEEDS REVIEW<small>0 of 8 graded</small></div>
  </div>
  <p class="after">The fix didn't make the evaluator timid. It made the result honest.</p>
</section>

<!-- ══ 3 ══ the automation bridge -->
<section class="slide">
  <div class="kicker">Why this isn't just a bug</div>
  <h2>On an unattended run, the result <em style="font-style:normal" class="hl">is</em> what leaves the system.</h2>
  <div class="rungs">
    <div class="rung">
      <span class="l">Manual run</span>
      <span class="r">someone is there — they might open a transcript</span>
    </div>
    <div class="rung">
      <span class="l">Unattended run</span>
      <span class="r">the report is emailed; no transcript is read</span>
    </div>
    <div class="rung last">
      <span class="l">Silent grader failure</span>
      <span class="r">→ a confident, customer-facing claim</span>
    </div>
  </div>
  <p class="already">
    Already running — daily autopilot, hourly re-evaluation, reports emailed.<br>
    <span class="mono">owaspllm</span> is one of the six kinds in that loop.
  </p>
</section>

<!-- ══ 4 ══ 3 of 6 -->
<section class="slide">
  <div class="kicker">The six hai-neo services that decide a verdict with an LLM judge</div>
  <div class="split">
    <div class="col ok">
      <h3>Preserve the uncertainty</h3>
      <ul>
        <li><span class="m">✓</span>hallucination</li>
        <li><span class="m">✓</span>control-audit</li>
        <li><span class="m">✓</span>control-monitor</li>
      </ul>
    </div>
    <div class="col no">
      <h3>Lose it</h3>
      <ul>
        <li><span class="m">✗</span>bias</li>
        <li><span class="m">✗</span>owaspllm</li>
        <li><span class="m">✗</span>web-redteam</li>
      </ul>
    </div>
  </div>
  <p class="thesis"><span class="b">The principle exists.</span><span class="b">The contract doesn't.</span></p>
  <p class="ladder">Three correct implementations — three different mechanisms.</p>
</section>

<!-- ══ 5 ══ continuous assurance -->
<section class="slide">
  <div class="kicker">Pull back one level from the contract</div>
  <h2>Continuous Assurance</h2>
  <div class="arch">
    <div class="spine">
      <div class="grp">
        <div class="gl">Test continuously</div>
        <ul>
          <li>Risk &amp; obligations</li>
          <li>Assurance policy</li>
          <li>Red team &amp; evaluations</li>
        </ul>
      </div>
      <div class="grp now">
        <div class="gl">Know what happened</div>
        <ul><li>Evaluation contract<span class="tag">today</span></li></ul>
      </div>
      <div class="grp">
        <div class="gl">Prove the claim</div>
        <ul>
          <li>Evidence</li>
          <li>Audit</li>
          <li>Report · Trust Center · Certificate</li>
        </ul>
      </div>
      <p class="loop">
        <span>↺</span><span class="k">automation</span>
        <b>detect → remediate → re-test → re-prove</b>
        <span>⚖</span><span class="k">assurance</span>
        <b>finding → candidate evidence → a person decides</b>
      </p>
    </div>
    <div class="horizons">
      <div class="hzlabel">Dependencies, not dates</div>
      <div class="hz first"><div class="w">Now</div><div class="h">Trust the measurement</div></div>
      <div class="hz"><div class="w">Next</div><div class="h">Test continuously</div></div>
      <div class="hz"><div class="w">Then</div><div class="h">Prove the evidence</div></div>
      <div class="hz"><div class="w">Later</div><div class="h">Expand assurance</div></div>
    </div>
  </div>
</section>

<!-- ══ 6 ══ close -->
<section class="slide star">
  <div class="clauses">
    <div>Test continuously.</div>
    <div>Know what happened.</div>
    <div>Prove the claim.</div>
  </div>
  <p class="closeline"><b>hai-neo is already becoming this product.</b><br>
    The missing piece is the connective tissue.</p>
</section>

<!-- ══ 7 · backup, only if asked — class "backup" exempts it from the word budget in --check ══ -->
<section class="slide backup">
  <div class="claim">Backup — hai-neo makes a claim: safe · compliant · trustworthy</div>
  <div class="chain">
    <div class="layer hot">
      <span class="n">1</span>
      <span class="q">Can we trust the measurement?</span>
      <span class="s">today's cut</span>
    </div>
    <div class="layer">
      <span class="n">2</span>
      <span class="q">Do we know what we tested?</span>
      <span class="s">risk, strategy and channel are represented inconsistently</span>
    </div>
    <div class="layer warm">
      <span class="n">3</span>
      <span class="q">Can the result become evidence?</span>
      <span class="s">no shared provenance or qualification contract</span>
    </div>
    <div class="layer">
      <span class="n">4</span>
      <span class="q">Which requirement does it support?</span>
      <span class="s">no control ↔ evidence mapping</span>
    </div>
    <div class="layer">
      <span class="n">5</span>
      <span class="q">Can an auditor rely on it?</span>
      <span class="s">no accept / insufficient / reject</span>
    </div>
    <div class="layer warm">
      <span class="n">6</span>
      <span class="q">Can we defend the conclusion?</span>
      <span class="s">report ↔ finding ↔ evidence lineage is absent</span>
    </div>
  </div>
</section>

<div class="pager"><span id="cur">1</span> / <span id="tot">5</span></div>
<div class="hint">← →  ·  space  ·  click</div>

<script>
  const slides=[...document.querySelectorAll('.slide')];let i=0;
  const cur=document.getElementById('cur');document.getElementById('tot').textContent=slides.length;

  // Remember the current slide across reloads (a file-watcher refresh, an
  // accidental ⌘R, or reopening the tab) — otherwise you snap back to slide 1
  // mid-talk. Stored in the URL hash so it also survives a fresh window, with
  // sessionStorage as a fallback when the hash is stripped.
  function save(n){
    try{ history.replaceState(null,'','#'+n); }catch(e){ location.hash=String(n); }
    try{ sessionStorage.setItem('haineo-slide',String(n)); }catch(e){}
  }
  function restore(){
    const inRange=v=>Number.isInteger(v)&&v>=1&&v<=slides.length;
    const h=parseInt(location.hash.slice(1),10);
    if(inRange(h)) return h-1;
    let v=NaN; try{ v=parseInt(sessionStorage.getItem('haineo-slide'),10); }catch(e){}
    return inRange(v)?v-1:0;
  }

  function go(n){i=Math.max(0,Math.min(slides.length-1,n));
    slides.forEach((s,k)=>s.classList.toggle('on',k===i));cur.textContent=i+1;save(i+1);}
  addEventListener('keydown',e=>{
    if(['ArrowRight','ArrowDown',' ','PageDown','n'].includes(e.key)){e.preventDefault();go(i+1)}
    if(['ArrowLeft','ArrowUp','PageUp','p'].includes(e.key)){e.preventDefault();go(i-1)}
    if(e.key==='Home')go(0); if(e.key==='End')go(slides.length-1);
  });
  addEventListener('click',e=>go(e.clientX<innerWidth*0.2?i-1:i+1));
  go(restore());
</script>
</body>
</html>
````

## 附录 B · 兜底页 `fallback.html`

黑底满屏，← → 翻，底部一行「这是哪个 run · 该说什么」。占位 = `<img src>` 八个路径 + `CAPS` 数组。
**录播变体**：把每个 `<img …>` 换成 `<pre>`（内容来自 `captures/beatN.txt`），并在 `<style>` 里加
`pre{color:#eee;font:500 clamp(14px,1.45vw,22px)/1.45 ui-monospace,Menlo,monospace;white-space:pre-wrap;max-width:100%;max-height:100%;overflow:hidden;padding:2rem}`。

````html
<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fallback — captured runs</title>
<style>
  :root{--paper:#111;--ink:#eee;--muted:#888;--amber:#e0a33a}
  *{box-sizing:border-box;margin:0;padding:0}
  html,body{height:100%;background:var(--paper);overflow:hidden;
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,sans-serif}
  .stage{position:absolute;inset:0;display:none;align-items:center;justify-content:center;padding:0}
  .stage.on{display:flex}
  img{max-width:100%;max-height:100%;object-fit:contain;display:block}
  .bar{position:fixed;left:0;right:0;bottom:0;display:flex;gap:1.2rem;align-items:baseline;
    padding:.5rem 1rem;background:rgba(0,0,0,.72);color:var(--ink);font-size:.8rem;letter-spacing:.02em}
  .bar .n{color:var(--muted);font-variant-numeric:tabular-nums}
  .bar .hot{color:var(--amber);font-weight:600}
  .bar .hint{margin-left:auto;color:#555;font-size:.72rem}
</style></head><body>

<div class="stage on"><img src="../screenshots/01-run2-main-badjudge-PASS-overview.png"></div>
<div class="stage"><img src="../screenshots/02-run2-transcript-judge-unavailable.png"></div>
<div class="stage"><img src="../screenshots/03-run1-main-goodjudge-FAIL-overview.png"></div>
<div class="stage"><img src="../screenshots/04-run1-token-game-LEAK.png"></div>
<div class="stage"><img src="../screenshots/05-run3-branch-badjudge-NEEDSREVIEW-overview.png"></div>
<div class="stage"><img src="../screenshots/06-run3-not-graded-transcripts.png"></div>
<div class="stage"><img src="../screenshots/07-run4-branch-goodjudge-FAIL-overview.png"></div>
<div class="stage"><img src="../screenshots/08-run4-transcript.png"></div>

<div class="bar">
  <span class="n"><span id="cur">1</span>/8</span>
  <span id="cap"></span>
  <span class="hint">← →</span>
</div>

<script>
  const CAPS = [
    ['Run 2 · today · grader down', 'PASS — “No failed, warning, or review-needed categories.”'],
    ['Run 2 · transcripts', 'green “blocked”, and judge: judge unavailable'],
    ['Run 1 · today · grader healthy', 'FAIL — ASR 38%, same target, minutes earlier'],
    ['Run 1 · token_game', '…8. code  9. ACME-7741 — the model counts out its own secret'],
    ['Run 3 · branch · grader down', 'NEEDS REVIEW — 8 attempted, 0 graded, 8 ungraded'],
    ['Run 3 · transcripts', 'amber “not graded” — the third state'],
    ['Run 4 · branch · grader healthy', 'FAIL — still finds the leak. Only one cell changed.'],
    ['Run 4 · transcripts', 'graded, and judged'],
  ];
  const st=[...document.querySelectorAll('.stage')];let i=0;
  const cur=document.getElementById('cur'), cap=document.getElementById('cap');
  function go(n){i=Math.max(0,Math.min(st.length-1,n));
    st.forEach((s,k)=>s.classList.toggle('on',k===i));
    cur.textContent=i+1;
    cap.innerHTML='<span class="hot">'+CAPS[i][0]+'</span> &nbsp;·&nbsp; '+CAPS[i][1];}
  addEventListener('keydown',e=>{
    if(['ArrowRight','ArrowDown',' ','PageDown'].includes(e.key)){e.preventDefault();go(i+1)}
    if(['ArrowLeft','ArrowUp','PageUp'].includes(e.key)){e.preventDefault();go(i-1)}});
  addEventListener('click',e=>go(e.clientX<innerWidth*0.2?i-1:i+1));
  go(0);
</script></body></html>
````

## 附录 C · 台本骨架 `rehearsal-script.md`

````markdown
# <场次> 台本 —— 中文引导 · 英文台词

**中文 = 你做什么、点哪、注意什么。英文引用块 = 原话，照念。** 其余任何东西都不要在台上念。

_权威：说什么 / 不说什么 / 他会问什么 = `<计划文档.md>`。本文只是嘴里念的那一份。手要做的 = `<runbook.md>`。_

---

## 时间不够时的裁剪顺序

1. 先砍 <哪一张 / 哪一段>（保留锁死句，删展开）。
2. 再砍 <…>。
3. ⛔ **永不砍**：<demo 段> · <核心转场> · ask。到场先确认时段。

---

## 0. 开场前 5 分钟的固定动作

**标签页，就这几个：**

| 标签 | 内容 | 状态 |
| --- | --- | --- |
| **① 片子** | `docs/dashboard/<slides>.html` | 停在第 1 张，全屏 |
| **② 产品 / 短片** | <URL 或播放器> → <停在哪个列表 / 哪一拍开头> | <…> |
| **③ 兜底** | `docs/dashboard/fallback.html` | 出事才切 |

**检查清单：**

- [ ] 浏览器缩放 125–150%，书签栏收起
- [ ] <dev 进程 / 分支 / 播放器音量>
- [ ] 手机静音，通知关掉
- [ ] **列表第 1 行是 `<ID>`**（不是的话说明多跑了，照 ID 点别数行号）

**列表对照，从上往下：**

| 第几行 | ID | 是哪个 | 结果 | 讲的顺序 |
| --- | --- | --- | --- | --- |
| 第 1 行 | `<id>` | <分支 · 变量> | <结果> | ② |
| 第 2 行 | `<id>` | <…> | <…> | ①③ |

讲的顺序：**第 2 行 → 第 1 行 → 回第 2 行 → 第 3 行 → Slide 2 → …**
⛔ 今天之后不要再跑新的 run。

**幻灯片对照：**

| 张 | 内容 | 在哪一幕用 |
| --- | --- | --- |
| 1 | <…> | 第一幕 |
| N | <…>（备用） | 被问才翻 |

**时间分配：**

```text
0:30  slide 1   <…>
3:00  产品      <…>
…
```

---

# 第一幕 · <名字>（<时长>）

**标签 ①，第 1 张。不要念屏幕上的字。**

> <英文台词，一句一行>

**停半秒。**

> <…>

⚠️ 「<这一句>」绝对不能省。

---

# 第二幕 · <名字>（<时长>）

## 2.1 <这一拍的名字> —— `<ID>`

**切标签 ② → 点 <…> → 停在 <…>。**

> <…>

**停两秒。**

⛔ 不要点 <…>。不要让他们提前看到 <会泄底的东西>。

这一拍的目的只有一个：**<一句话>**

## 2.2 … （每拍同构）

---

# 被问到时

## 「<问题>」

> <3–4 句。答不了的：I can't answer that from the code today. + 怎么才能知道>

---

# 演讲纪律

## 现在锁死这 N 句（其余允许自然说）

1. > **<…>**
2. > **<…>**

## 沉默的位置

- 说完「<…>」停 2 秒。
- 说完「<最狠那句>」停 3 秒。
- 说完「<锁死句>」不说话、不翻页，让它在屏幕上待着。

## 被打断时

一句话答完，然后回主线。固定返回句：

> **— so, <一句把线拉回来的话>.**

提前问到后面的内容：

> That's exactly where I'm going next — can I show you one more thing first?

问到不知道的业务事实：

> I can't answer that from the code today. <谁能答 / 怎么验证>

---

# 出事了怎么办

浏览器崩了：**不调试。** 切兜底：

> I've got these captured.

继续。永远不要说 "It worked this morning"。不现场跑整个 test suite；真要跑只跑 `<聚焦的那一条>`。

---

# 最后一次彩排怎么练

只练转场链：

```text
<拍 1 的最后一句>
↓
<拍 2 的第一句>
↓
…
```

只要这几条链没断，现场就不会丢。
````

## 附录 D · runbook 骨架 `runbook.md`

````markdown
# <场次> runbook —— 操作那一半

_只写手要做的，不写说什么（那在 `rehearsal-script.md`）。_

## 1. 环境 / 权限 —— 第一小时就要问的

| 需要 | 变量 / 来源 | 风险 |
| --- | --- | --- |
| <自己的 dev 部署> | `<VAR>` | **Blocker.** <GitHub write ≠ 平台 team access 之类> |
| <登录> | `<VAR>`（还要在 <哪个 dashboard> 里设 <…>） | **Blocker.** |
| <判官 / 模型> | `<VAR>` | low |

核查命令：`<仓库自带的检查器>`。
> ⚠ 陷阱：<示例文件写的是 X，代码实际读的是 Y>。

## 2. 让对比存在的设置

| 设置 | 值 | 为什么 |
| --- | --- | --- |
| <类别 / 范围> | <…> | <房间看得懂 · 现场跑得完> |
| <目标 / 模型> | <…> | <…> |
| <目标 prompt / 秘密的位置> | <一行，秘密靠前> | <加人设会让它全挡住 —— 实测> |

**配方表：**

| # | 分支 | 变量 | 期望 | **实测** | 用来说明什么 |
| --- | --- | --- | --- | --- | --- |
| 1 | main | 好 | <…> | <…> | 这个目标的真相 |
| 2 | main | **坏** | <…> | <…> | 危害 |
| 3 | 分支 | 坏 | <…> | <…> | 修复 |
| 4 | 分支 | 好 | <…> | <…> | 「那以后是不是老这样」 |

「before」的复现（零代码零 flag）：`<一条命令>`；复原：`<一条命令>`。
产品在故障态下自己说的那句：「<…>」—— 进台词。
⚠ <表单不回填哪些字段 / 采样模型数字会动但判定不动 / UI 上的词是什么不是什么>。

## 3. 不是你的红

| 现象 | 原因 | 替代 |
| --- | --- | --- |
| `<命令>` 1 条失败 | <locale / main 上本来就红> | 只跑 `<聚焦那一条>` |

⛔ 别在人前跑 <…>。

## 4. 现场命令（也是无栈兜底）

```bash
git log --oneline main..HEAD                 # hash 会因 rebase 变，现查
git checkout <test-commit> -- <file>         # 红
<focused test>
git checkout HEAD -- <file>                  # 绿
<focused test>
```

## 5. 状态 · 别碰清单

**Done.** <分支 · commit 数 · 测试状态 · 是否已推>。
**Left.** <…>。
⛔ **会前别碰**：<…>。

## 6. 录播版清单（录播形态才有）

**录前**
- [ ] 分支对、工作树干净
- [ ] `run_demo.sh --check` 打印 `CHECK PASS`；红了先修产物或文档，⛔ 不录
- [ ] 每拍冷跑一次（第一次是冷的）
- [ ] 终端 120×40、18pt、浅色、无透明、无 tab、`PS1='$ '`，工作目录不带客户名
- [ ] 通知关、Finder 关、分辨率 1920×1080

**录制**（macOS ⇧⌘5 选区 + 麦克风；照 voice-over 粗体念）

| Clip | 命令 | 长度 | 结尾停在 |
| --- | --- | --- | --- |
| 1 | `./run_demo.sh 1` | ≈75 s | <哪一行>，停两秒 |
| 2 | `./run_demo.sh 2` | ≈75 s | <…> |
| 3 | `./run_demo.sh 3` | ≈60 s | <…> |

**录后**
- [ ] 每段全屏看一遍：客户名的文件名 · 上一条命令的 scrollback · 通知弹窗
- [ ] 剪头尾；不加转场不加字幕
- [ ] 命名 `<场次>-beat{1,2,3}.mov`，**存在所有仓库之外**
- [ ] 发审核人：短片 + 片子 + 计划文档（§「变了什么」打开）；**ask 先写在消息里**，别让他在 deck 里第一次碰到

**当天出事**：某拍在镜头前失败 → 停，不解说，下机重跑再录；设计内的 exit 1 不算失败。
`--check` 早上红了 → 最可能是产物被重生成或文档被改，读 ✗ 行。

## 附 · voice-over 骨架（`VOICEOVER.md`）

每拍四块，粗体照念，其余给自己：

### Beat N —— <动词>（≈<秒>）
**Screen:** `run_demo.sh N`。<屏幕上会出现什么>。
**Say, while it loads:** > **<…>**
**When <…> appears, point at <…>:** > **<…>**
⛔ Do not <scroll / open the file / say how long it took>.

文末单列 **What must not be in the video**：<客户行 · 标识符 · 暂定比值 · 判决 · 时长 · "saves / faster / compliant" 这类词>。
````

## 附录 E · 跑脚本骨架 `run_demo.sh`

三个子命令族：`N` 一拍一个 · `--check` 三件事（重算比对 / 引用数都在产物里 / 片子词预算）· `--capture` 喂兜底页。
「词」的唯一定义就是第 3 步那条正则。

````bash
#!/usr/bin/env bash
# <场次> demo — 每拍一个子命令，加一个 --check 保证「屏幕上印的 = 落盘产物 = 片子/计划里引用的数」。
#
#   run_demo.sh --check     重算 → 与落盘产物逐字段比 → 片子/计划里引用的每个数都在产物里 → 片子词预算
#   run_demo.sh --capture   把每拍 stdout 存进 captures/beatN.txt（喂 fallback.html）
#   run_demo.sh 1|2|3       第 N 拍
#
# 什么永不上屏（按 <计划文档> §…）：<…>
# 权威：说什么 = <计划文档>；念什么 = VOICEOVER.md；手做什么 = runbook.md
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLIDES="$HERE/../dashboard/<slides>.html"
PLAN="$HERE/../<plan>.md"
OUT="<落盘产物目录>"                       # 片子和计划引用的数从这里来
TMP="${TMPDIR:-/tmp}/<slug>"; mkdir -p "$TMP" "$HERE/captures"

# 会引来 "compared to what?" 的行，一律不上屏
quiet() { grep -vE '^(load|compute|peak RSS|written)\s'; }

beat1() { echo "▶ <一句话：这一拍是什么>"; echo; <命令 …> 2>&1 | quiet || true; }   # exit 1 是设计结果时在这里写明
beat2() { echo "▶ <…>"; echo; <命令 …> --out "$TMP/beat2.json" 2>&1 | quiet; }
beat3() { echo "▶ <…>"; echo; <命令 …> 2>&1 | quiet; }

check() {
  local fail=0
  echo "── 1/3 重算，并与落盘产物逐字段比"
  <重算命令 …> --out "$TMP/fresh.json" >/dev/null 2>&1 || { echo "⛔ recompute did not complete"; exit 1; }
  python3 - "$OUT/<kept>.json" "$TMP/fresh.json" <<'EOF' || fail=1
import json, sys
kept, fresh = (json.load(open(p)) for p in sys.argv[1:3])
bad = 0
for k in ("<field1>", "<field2>", "<version_field>"):
    ok = kept.get(k) == fresh.get(k); bad += (not ok)
    print(f"  {'✓' if ok else '✗'} {k}: {'identical' if ok else f'{kept.get(k)!r} vs {fresh.get(k)!r}'}")
sys.exit(1 if bad else 0)
EOF

  echo "── 2/3 片子和计划里引用的每个数，都在产物里"
  python3 - "$OUT/<kept>.json" "$SLIDES" "$PLAN" <<'EOF' || fail=1
import json, sys, pathlib
d = json.load(open(sys.argv[1])); docs = [pathlib.Path(p).read_text() for p in sys.argv[2:]]
cited = {                                  # 每个上屏的数：从产物格式化，⛔ 不手抄
    "<label1>": f"{d['<field1>']:,}",
    "<label2>": f"{d['<field2>']:,}",
    "<engine>": d["<version_field>"],
}
bad = 0
for label, val in cited.items():
    hit = any(val in t for t in docs); bad += (not hit)
    print(f"  {'✓' if hit else '✗'} {label} = {val}")
sys.exit(1 if bad else 0)
EOF

  echo "── 3/3 词预算：非备用张 ≤ 50 词（playbook §3）"
  python3 - "$SLIDES" <<'EOF' || fail=1
import re, sys, html
s = open(sys.argv[1]).read(); body = s.split("<body", 1)[1]; body = re.sub(r"<script.*?</script>", "", body, flags=re.S)
bad = 0
for n, (tag, inner) in enumerate(re.findall(r"(<section[^>]*>)(.*?)</section>", body, flags=re.S), 1):
    backup = "backup" in tag
    text = html.unescape(re.sub(r"<[^>]+>", " ", inner))
    words = len(re.findall(r"[A-Za-z0-9][\w',.%/§-]*", text))     # ← 「词」的唯一定义
    ok = words <= 50 or backup; bad += (not ok)
    print(f"  {'✓' if ok else '✗'} slide {n}: {words} words{' (backup, exempt)' if backup else ''}")
sys.exit(1 if bad else 0)
EOF

  if [ "$fail" = 0 ]; then echo "CHECK PASS"; else echo "CHECK FAIL"; exit 1; fi
}

capture() {
  for n in 1 2 3; do "beat$n" > "$HERE/captures/beat$n.txt" 2>&1 || true; done
  echo "captured → $HERE/captures/"
}

case "${1:-}" in
  1) beat1 ;; 2) beat2 ;; 3) beat3 ;;
  --check) check ;;
  --capture) capture ;;
  *) sed -n '2,8p' "$0"; exit 2 ;;
esac
````
