---
type: framing-input
status: proposal (not adopted — user decides)
created: 2026-10-08
purpose: a COLING 2027 frame chosen against the evidence layer as of §547; numbers are pointers to their owners, the live status of each attack surface stays in naacl_evidence_delta.md §0
---

# COLING 2027 frame — proposal (2026-10-08)

## 0. 一句话

**保留 08-01 user 定的问题与四步骨架，把第 4 步从「标签太少」换成一个三段式的、可迁移的诊断；把 rerun control 抬成方法贡献放在中心；撤掉 VLM4RWD 里「更强的 agent 可能推翻结论」那句。**

候选题目（按推荐顺序）：
1. *One Run Is Not a Label: What Per-Task Observation Routing for Web Agents Can and Cannot Learn*
2. *Text or Pixels, Task by Task? Reruns, Reliability, and the Limits of Observation Routing for Web Agents*

## 1. 问题陈述（沿用 08-01，user 原话的工业版本）

Web agent 看网页有三类方式：只看文本（DOM / 无截图的 phantom 族）、只看截图（Vision）、融合（SoM）。企业里真正的问题是 **vision 还是 text**，以及**每道题该不该花钱买截图**。问题陈述：「每道题有限的调用预算，怎么花在表征上」，答案按题给（task-level：开跑前决定一次，整题不换）。

## 2. 论证（四步，第 4 步重写）

**① 表面上的上限有一半是假币。** 各表征互补、每种都解出别人解不出的题，oracle 看着很大；但同条件重跑会改写 12–14% 的结局，再跑一次已有的表征 ≈ 加一个新表征（C2，三格）。⇒ 任何「多臂上限」都必须对照重跑读。
证据：`noise_floor_inventory`（24 对）· delta §1 C2 · VLM4RWD §3。

**② 默认答案最贵，没挣回来。** SoM（唯一的融合模式）多数格最好也最贵；学到的路由贴着「固定 mode + 随机混合」的前沿走，不比它好。
证据：§539 / §540（前沿 + 合并）· §545（换 GPU 时间 / 墙钟口径结论不变）。

**③ 该买哪一类随任务类型与站点翻转 —— 而且这个翻转在「题类」层面是可预知的。** 一条 0 token 的意图正则在 classifieds 上事先找出截图值钱的题（cls_B0 +22.54pp vs 其余 +0.65pp，B1 同向，样本外），reddit 上不成立。
证据：`visual_intent_routing`。⚠️ 这条是本稿目前最强的「正面、可部署」信号之一，VLM4RWD 没放在主线；见 §4 风险 1。

**④ 压不到按题（重写，三段式诊断 —— 本稿新贡献的核心）。**
- **④a 目标是真的，但单次标签测不准。** 无交互零模型（条件于题难度与臂易度，curveball 精确抽样）下，三个六臂全复测格的 task×mode 交互都能跨重跑复现（Holm 全过）；但单次 run 的标签信度只有 0.14–0.46，每题要 2–6 次重跑信度才过 0.5（D-study）。—— §543
- **④b 同题偏好会保持，开跑前特征带不到新题。** 一次 run 的同题偏好搬到重跑上比前沿多 +2.4~+10.5pp（red_B0、WA_B1）；可部署路由器换模板或换 run 后平均都在前沿以下。—— §546
- **④c 不是这批模型太弱。** 最强 backbone（GPT-5.6，5 个有效臂）上可部署路由器 −2.45 / −1.63pp，前沿检验 p 0.13–0.77。—— §547
- **量化收口**：只在训练数据上定工作点的路由器，8 格合并六头显著低于混合前沿（95% 上界 −0.14pp），triage ≤ +0.81pp；shopping（剔除 harness 污染题）与 B5 同向。—— §544 / §547

**收束（方法层，可迁移）**：报告多臂 agent 结果时附重跑带；要训练按题路由，先用 D-study 定每题要跑几次。预算（题难度）这一半是可学的，但属于已有的 early-abort / 预算感知家族，**不作新颖性主张**（§535），前瞻检验只在 P-prompt 上可识别地通过（§542）。

## 3. 相对 VLM4RWD 必须改的句子

| VLM4RWD 原句 | 新证据 | 改成 |
|---|---|---|
| 障碍是「监督按 agent 成功率产出，越弱越少」 | §543：目标存在但单次信度低；§546：特征推广不了 | 障碍是**单次标签信度** + **特征到新题的推广**，不只是数量 |
| 「a stronger agent can overturn the result」 | §547：B5 上同样不成立（单格） | 删；改为「在本研究最强的 backbone 上也未出现」，并限定 B5 只有 classifieds 一格 |
| 供给与机会相关 0.95 | Codex：「routable」定义为 >1 个 mode 解出，唯一可解题被排除；8 格不独立 | 降为描述，或重定义后重算再用 |
| 竞争题不稳定「超出难度 floor」 | §541：同口径下 0.83–0.94×，已撤回 | 不再说「有额外结构的不稳定」；结构证据改用 §543 |
| （C1）重跑 floor 按 serving path 分组 | §529.4 已撤回 | 不出现 |

## 4. 证据覆盖核查：用上了吗

**进了主线的**：8 格（VWA classifieds / reddit × B0/B1/B2 + WA reddit × B0/B1）· B5 classifieds（5 臂）· shopping B0/B1（剔污染题与全量两版）· 三个六臂全复测格 + 两个部分复测格 · 计费 / GPU 时间 / 墙钟三种成本口径 · 预算路由前瞻检验。

**有但还没进主线、建议进的**：
- `visual_intent_routing`（第 ③ 步的正面信号）—— 需要在 B5 与 WA 上补同一张表再用（0-compute，约 1–2 h）。
- 失败结构差异（`cross_mode_failure_signatures`、`failure_modes_per_cell`）—— 支撑「互补」，VLM4RWD 已用，保留。
- 「能力 ≠ 可复现性」（B5 的 DOM 重跑 floor 12.95% 落在 B0 范围内，delta §1 C3）—— 一句话，单臂。
- harness 卫生（B-2002 在 shopping 上 43% 的搜索提交被拼接、reddit 状态泄漏 6 个成功、产出脚本里硬编码的结论）—— 作为「为什么多臂比较要先审 harness」的方法附录，审稿人通常加分。

**拿不到的**：A100 上 09-22 之后的复测（B1 shopping SoM 余下 340 集、B2 × 3 臂）—— 进来后无带格 5 → 3，不改变 §543 / §546 的三格结论。需要 user 连 UCL VPN 重签证书。

**刻意不用的**：mechanism 线（05-14 起 shelved；linear probe 是错工具，§111.2）· serving-path floor（撤回）· B3 / B4（空目录 / 冒烟）。

## 5. 风险

1. **第 ③ 步与第 ④ 步的张力**：既然文本规则能找出「截图值钱的题类」，为什么按题路由学不到？答案要写清：规则给的是**题类平均**的增益，而 SoM 已经包含截图，所以「按这条规则在 DOM 与 Vision 之间切」打不过「一直用 SoM」；按题的、超出题类的偏好才是 ④ 说学不到的东西。没写清会被当成自相矛盾。
2. **三格外推**：④a/④b 只有三个六臂全复测格（其中两个 B0、一个 B1，WA 只有一格）；B5 只有一格。必须在正文限定。
3. **单格显著不可单引**：triage 哪格过 Holm 随成本口径变（§545）；跨重跑单格点估计对折划分敏感（§546）。正文只报合并数与区间。
4. **页数 / 模板**：COLING 2027 首届走 ARR，CFP 页数上限未核（任务卡已标）；正文能放下 ①–④ + 方法收束，第 ③ 步的扩展表与 harness 卫生进附录。

## 6. COLING 的切入点

COLING 是计算语言学会议：主线落在**「文本表征什么时候够用、什么时候必须看像素」**（DOM / 无障碍树文本 vs 截图）和**语言 agent 评测的可复现性**（单次 run 的结局不是标签）。路由是用来提出这两个问题的实验框架，不是卖点。

## 7. 要 user 定的

1. 采用这个骨架，还是回到 09-09 的「Route the budget, not the representation」（预算线已知非新颖、前瞻检验只半边可识别 ⇒ 我不推荐作标题主张）。
2. 第 ③ 步（文本规则找截图值钱的题）是否进主线 —— 进的话先补 B5 / WA 两格。
3. delta §0 第 4 / 5 行（Pareto 计数、噪声带口径）—— 推荐见 2026-10-07 对话：第 4 行用交叉拟合 1/8 作辅助、主数换 §544；第 5 行拆成「重跑变化」与「显著门槛」两个量，等 A100 数据进来再钉。
